"""Lightweight background shell manager for BashTool.

Allows starting shell commands that keep running while the caller continues.
Output can be polled via the BashOutput tool and commands can be terminated
via the KillBash tool.
"""

import asyncio
import concurrent.futures
import contextlib
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import atexit

from ripperdoc.utils.shell_utils import build_shell_command, find_suitable_shell
from ripperdoc.utils.log import get_logger


logger = get_logger()


@dataclass
class BackgroundTask:
    """In-memory record of a background shell command."""

    id: str
    command: str
    process: asyncio.subprocess.Process
    start_time: float
    timeout: Optional[float] = None
    stdout_chunks: List[str] = field(default_factory=list)
    stderr_chunks: List[str] = field(default_factory=list)
    exit_code: Optional[int] = None
    killed: bool = False
    timed_out: bool = False
    reader_tasks: List[asyncio.Task] = field(default_factory=list)
    done_event: asyncio.Event = field(default_factory=asyncio.Event)


_tasks: Dict[str, BackgroundTask] = {}
_tasks_lock = threading.Lock()
_background_loop: Optional[asyncio.AbstractEventLoop] = None
_background_thread: Optional[threading.Thread] = None
_loop_lock = threading.Lock()
_shutdown_registered = False


def _safe_log_exception(message: str, **extra: Any) -> None:
    """Log an exception but never let logging failures bubble up."""
    try:
        logger.exception(message, extra=extra)
    except (OSError, RuntimeError, ValueError):
        pass


def _ensure_background_loop() -> asyncio.AbstractEventLoop:
    """Create (or return) a dedicated loop for background processes."""
    global _background_loop, _background_thread

    if _background_loop and _background_loop.is_running():
        return _background_loop

    with _loop_lock:
        if _background_loop and _background_loop.is_running():
            return _background_loop

        loop = asyncio.new_event_loop()
        ready = threading.Event()

        def _run_loop() -> None:
            asyncio.set_event_loop(loop)
            ready.set()
            loop.run_forever()

        thread = threading.Thread(
            target=_run_loop,
            name="ripperdoc-bg-loop",
            daemon=True,
        )
        thread.start()
        ready.wait()

        _background_loop = loop
        _background_thread = thread
        _register_shutdown_hook()
        return loop


def _register_shutdown_hook() -> None:
    global _shutdown_registered
    if _shutdown_registered:
        return
    atexit.register(shutdown_background_shell)
    _shutdown_registered = True


def _submit_to_background_loop(coro: Any) -> concurrent.futures.Future:
    """Run a coroutine on the background loop and return a thread-safe future."""
    loop = _ensure_background_loop()
    return asyncio.run_coroutine_threadsafe(coro, loop)


async def _pump_stream(stream: asyncio.StreamReader, sink: List[str]) -> None:
    """Continuously read from a stream into a buffer."""
    try:
        while True:
            chunk = await stream.read(4096)
            if not chunk:
                break
            text = chunk.decode("utf-8", errors="replace")
            with _tasks_lock:
                sink.append(text)
    except (OSError, RuntimeError, asyncio.CancelledError) as exc:
        if isinstance(exc, asyncio.CancelledError):
            return  # Normal cancellation
        # Best effort; ignore stream read errors to avoid leaking tasks.
        logger.debug(
            f"Stream pump error for background task: {exc}",
            exc_info=True,
        )


async def _finalize_reader_tasks(reader_tasks: List[asyncio.Task], timeout: float = 1.0) -> None:
    """Wait for stream reader tasks to finish, cancelling if they hang."""
    if not reader_tasks:
        return

    try:
        await asyncio.wait_for(
            asyncio.gather(*reader_tasks, return_exceptions=True), timeout=timeout
        )
    except asyncio.TimeoutError:
        for task in reader_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*reader_tasks, return_exceptions=True)


async def _monitor_task(task: BackgroundTask) -> None:
    """Wait for a background process to finish or timeout, then mark status."""
    try:
        if task.timeout:
            await asyncio.wait_for(task.process.wait(), timeout=task.timeout)
        else:
            await task.process.wait()
        with _tasks_lock:
            task.exit_code = task.process.returncode
    except asyncio.TimeoutError:
        logger.warning(f"Background task {task.id} timed out after {task.timeout}s: {task.command}")
        with _tasks_lock:
            task.timed_out = True
        task.process.kill()
        await task.process.wait()
        with _tasks_lock:
            task.exit_code = -1
    except asyncio.CancelledError:
        return
    except (OSError, RuntimeError, ProcessLookupError) as exc:
        logger.warning(
            "Error monitoring background task: %s: %s",
            type(exc).__name__,
            exc,
            extra={"task_id": task.id, "command": task.command},
        )
        with _tasks_lock:
            task.exit_code = -1
    finally:
        # Ensure readers are finished before marking done.
        await _finalize_reader_tasks(task.reader_tasks)
        task.done_event.set()


async def _start_background_command(
    command: str, timeout: Optional[float] = None, shell_executable: Optional[str] = None
) -> str:
    """Launch a background shell command on the dedicated loop."""
    selected_shell = shell_executable or find_suitable_shell()
    argv = build_shell_command(selected_shell, command)
    process = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        stdin=asyncio.subprocess.DEVNULL,
        start_new_session=False,
    )

    task_id = f"bash_{uuid.uuid4().hex[:8]}"
    record = BackgroundTask(
        id=task_id,
        command=command,
        process=process,
        start_time=_loop_time(),
        timeout=timeout,
    )
    with _tasks_lock:
        _tasks[task_id] = record

    # Start stream pumps and monitor task.
    if process.stdout:
        record.reader_tasks.append(
            asyncio.create_task(_pump_stream(process.stdout, record.stdout_chunks))
        )
    if process.stderr:
        record.reader_tasks.append(
            asyncio.create_task(_pump_stream(process.stderr, record.stderr_chunks))
        )
    asyncio.create_task(_monitor_task(record))

    return task_id


async def start_background_command(
    command: str, timeout: Optional[float] = None, shell_executable: Optional[str] = None
) -> str:
    """Launch a background shell command and return its task id."""
    future = _submit_to_background_loop(
        _start_background_command(command, timeout, shell_executable)
    )
    return await asyncio.wrap_future(future)


def _compute_status(task: BackgroundTask) -> str:
    """Return a human-friendly status string."""
    if task.killed:
        return "killed"
    if task.timed_out:
        return "failed"
    if task.exit_code is None:
        return "running"
    return "completed" if task.exit_code == 0 else "failed"


def _loop_time() -> float:
    """Return a monotonic timestamp without requiring a running event loop."""
    try:
        return asyncio.get_running_loop().time()
    except RuntimeError:
        return time.monotonic()


def get_background_status(task_id: str, consume: bool = True) -> dict:
    """Fetch the current status and buffered output of a background command.

    If consume is True, buffered stdout/stderr are cleared after reading.
    """
    with _tasks_lock:
        if task_id not in _tasks:
            raise KeyError(f"No background task found with id '{task_id}'")

        task = _tasks[task_id]
        stdout = "".join(task.stdout_chunks)
        stderr = "".join(task.stderr_chunks)

        if consume:
            task.stdout_chunks.clear()
            task.stderr_chunks.clear()

        return {
            "id": task.id,
            "command": task.command,
            "status": _compute_status(task),
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": task.exit_code,
            "timed_out": task.timed_out,
            "killed": task.killed,
            "duration_ms": (_loop_time() - task.start_time) * 1000.0,
        }


async def kill_background_task(task_id: str) -> bool:
    """Attempt to kill a running background task."""
    KILL_WAIT_SECONDS = 2.0

    async def _kill(task_id: str) -> bool:
        with _tasks_lock:
            task = _tasks.get(task_id)
            if not task:
                return False

            if task.exit_code is not None:
                return False

        try:
            task.killed = True
            task.process.kill()
            try:
                await asyncio.wait_for(task.process.wait(), timeout=KILL_WAIT_SECONDS)
            except asyncio.TimeoutError:
                # Best effort: force kill and don't block.
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    task.process.kill()
                await asyncio.wait_for(task.process.wait(), timeout=1.0)

            with _tasks_lock:
                task.exit_code = task.process.returncode or -1
            return True
        finally:
            await _finalize_reader_tasks(task.reader_tasks)
            task.done_event.set()

    future = _submit_to_background_loop(_kill(task_id))
    return await asyncio.wrap_future(future)


def list_background_tasks() -> List[str]:
    """Return known background task ids."""
    with _tasks_lock:
        return list(_tasks.keys())


async def _shutdown_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Drain running background processes before stopping the loop."""
    with _tasks_lock:
        tasks = list(_tasks.values())
        _tasks.clear()

    for task in tasks:
        try:
            task.killed = True
            with contextlib.suppress(ProcessLookupError):
                task.process.kill()
            try:
                with contextlib.suppress(ProcessLookupError):
                    await asyncio.wait_for(task.process.wait(), timeout=1.5)
            except asyncio.TimeoutError:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    task.process.kill()
                with contextlib.suppress(asyncio.TimeoutError, ProcessLookupError):
                    await asyncio.wait_for(task.process.wait(), timeout=0.5)
            task.exit_code = task.process.returncode or -1
        except (OSError, RuntimeError, asyncio.CancelledError) as exc:
            if not isinstance(exc, asyncio.CancelledError):
                _safe_log_exception(
                    "Error shutting down background task",
                    task_id=task.id,
                    command=task.command,
                )
        finally:
            await _finalize_reader_tasks(task.reader_tasks)
            task.done_event.set()

    current = asyncio.current_task()
    pending = [t for t in asyncio.all_tasks(loop) if t is not current]
    for pending_task in pending:
        pending_task.cancel()
    if pending:
        with contextlib.suppress(Exception):
            await asyncio.gather(*pending, return_exceptions=True)

    with contextlib.suppress(Exception):
        await loop.shutdown_asyncgens()


def shutdown_background_shell() -> None:
    """Stop background tasks/loop to avoid asyncio 'Event loop is closed' warnings."""
    global _background_loop, _background_thread

    loop = _background_loop
    thread = _background_thread

    if not loop or loop.is_closed():
        _background_loop = None
        _background_thread = None
        return

    try:
        if loop.is_running():
            try:
                fut = asyncio.run_coroutine_threadsafe(_shutdown_loop(loop), loop)
                fut.result(timeout=3)
            except (RuntimeError, TimeoutError, concurrent.futures.TimeoutError):
                logger.debug("Failed to cleanly shutdown background loop", exc_info=True)
            try:
                loop.call_soon_threadsafe(loop.stop)
            except (RuntimeError, OSError):
                logger.debug("Failed to stop background loop", exc_info=True)
        else:
            loop.run_until_complete(_shutdown_loop(loop))
    finally:
        if thread and thread.is_alive():
            thread.join(timeout=2)
        with contextlib.suppress(Exception):
            loop.close()
        _background_loop = None
        _background_thread = None
