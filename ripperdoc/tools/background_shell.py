"""Lightweight background shell manager for BashTool.

Allows starting shell commands that keep running while the caller continues.
Output can be polled via the BashOutput tool and commands can be terminated
via the KillBash tool.
"""

import asyncio
import concurrent.futures
import contextlib
import os
import threading
import time
import uuid
import weakref
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


DEFAULT_TASK_TTL_SEC = float(os.getenv("RIPPERDOC_BASH_TASK_TTL_SEC", "3600"))


class BackgroundShellManager:
    """Manager for background shell tasks with proper lifecycle control.

    This class encapsulates all global state for background shell management,
    providing better testability and proper resource cleanup.
    """

    _instance: Optional["BackgroundShellManager"] = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        """Initialize the manager. Use get_instance() for singleton access."""
        self._tasks: Dict[str, BackgroundTask] = {}
        self._tasks_lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._loop_lock = threading.Lock()
        self._shutdown_event = threading.Event()
        self._shutdown_registered = False
        self._is_shutting_down = False

    @classmethod
    def get_instance(cls) -> "BackgroundShellManager":
        """Get or create the singleton instance."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance. Useful for testing.

        This method shuts down the current instance and clears it,
        allowing a fresh instance to be created on next access.
        """
        with cls._instance_lock:
            if cls._instance is not None:
                cls._instance.shutdown()
                cls._instance = None

    @classmethod
    def _set_instance_for_testing(cls, instance: Optional["BackgroundShellManager"]) -> None:
        """Set a custom instance for testing purposes."""
        with cls._instance_lock:
            cls._instance = instance

    @property
    def tasks(self) -> Dict[str, BackgroundTask]:
        """Access to tasks dict (for internal use)."""
        return self._tasks

    @property
    def tasks_lock(self) -> threading.Lock:
        """Access to tasks lock (for internal use)."""
        return self._tasks_lock

    def ensure_loop(self) -> asyncio.AbstractEventLoop:
        """Create (or return) a dedicated loop for background processes."""
        if self._loop and self._loop.is_running():
            return self._loop

        with self._loop_lock:
            if self._loop and self._loop.is_running():
                return self._loop

            loop = asyncio.new_event_loop()
            ready = threading.Event()
            shutdown_event = self._shutdown_event

            def _run_loop() -> None:
                asyncio.set_event_loop(loop)
                ready.set()
                try:
                    loop.run_forever()
                finally:
                    # Ensure cleanup happens even if loop is stopped abruptly
                    shutdown_event.set()

            # Use non-daemon thread to ensure atexit handlers can complete
            thread = threading.Thread(
                target=_run_loop,
                name="ripperdoc-bg-loop",
                daemon=False,  # Non-daemon for proper shutdown
            )
            thread.start()
            ready.wait()

            self._loop = loop
            self._thread = thread
            self._register_shutdown_hook()
            return loop

    def _register_shutdown_hook(self) -> None:
        """Register atexit handler for cleanup."""
        if self._shutdown_registered:
            return

        # Use weakref to avoid preventing garbage collection
        manager_ref = weakref.ref(self)

        def _shutdown_callback() -> None:
            manager = manager_ref()
            if manager is not None:
                manager.shutdown()

        atexit.register(_shutdown_callback)
        self._shutdown_registered = True

    def submit_to_loop(self, coro: Any) -> concurrent.futures.Future:
        """Run a coroutine on the background loop and return a thread-safe future."""
        loop = self.ensure_loop()
        return asyncio.run_coroutine_threadsafe(coro, loop)

    def shutdown(self, force: bool = False) -> None:
        """Stop background tasks/loop to avoid resource leaks.

        Args:
            force: If True, use minimal timeouts for faster exit.
        """
        if self._is_shutting_down:
            # If already shutting down and force is requested, just mark done
            if force:
                self._shutdown_event.set()
            return
        self._is_shutting_down = True

        loop = self._loop
        thread = self._thread

        if not loop or loop.is_closed():
            self._loop = None
            self._thread = None
            self._is_shutting_down = False
            return

        # Use shorter timeouts for faster exit
        async_timeout = 0.5 if force else 2.0
        join_timeout = 0.5 if force else 1.0

        try:
            if loop.is_running():
                try:
                    fut = asyncio.run_coroutine_threadsafe(
                        self._shutdown_loop_async(loop, force=force), loop
                    )
                    fut.result(timeout=async_timeout)
                except (RuntimeError, TimeoutError, concurrent.futures.TimeoutError):
                    logger.debug("Failed to cleanly shutdown background loop", exc_info=True)
                try:
                    loop.call_soon_threadsafe(loop.stop)
                except (RuntimeError, OSError):
                    logger.debug("Failed to stop background loop", exc_info=True)
            else:
                # If loop isn't running, try to run cleanup synchronously
                try:
                    loop.run_until_complete(self._shutdown_loop_async(loop, force=force))
                except RuntimeError:
                    pass  # Loop may already be closed
        finally:
            if thread and thread.is_alive():
                thread.join(timeout=join_timeout)
                # If thread is still alive after timeout, don't block further
                if thread.is_alive():
                    logger.debug("Background thread did not stop in time, continuing shutdown")
            with contextlib.suppress(Exception):
                if not loop.is_closed():
                    loop.close()
            self._loop = None
            self._thread = None
            self._shutdown_event.set()
            self._is_shutting_down = False

    async def _shutdown_loop_async(
        self, loop: asyncio.AbstractEventLoop, force: bool = False
    ) -> None:
        """Drain running background processes before stopping the loop.

        Args:
            loop: The event loop to shutdown.
            force: If True, use minimal timeouts for faster exit.
        """
        with self._tasks_lock:
            tasks = list(self._tasks.values())
            self._tasks.clear()

        # Use shorter timeouts when force is True
        wait_timeout = 0.3 if force else 1.5
        kill_timeout = 0.2 if force else 0.5

        for task in tasks:
            try:
                task.killed = True
                with contextlib.suppress(ProcessLookupError):
                    task.process.kill()
                try:
                    with contextlib.suppress(ProcessLookupError):
                        await asyncio.wait_for(task.process.wait(), timeout=wait_timeout)
                except asyncio.TimeoutError:
                    with contextlib.suppress(ProcessLookupError, PermissionError):
                        task.process.kill()
                    with contextlib.suppress(asyncio.TimeoutError, ProcessLookupError):
                        await asyncio.wait_for(task.process.wait(), timeout=kill_timeout)
                task.exit_code = task.process.returncode or -1
            except (OSError, RuntimeError, asyncio.CancelledError) as exc:
                if not isinstance(exc, asyncio.CancelledError):
                    _safe_log_exception(
                        "Error shutting down background task",
                        task_id=task.id,
                        command=task.command,
                    )
            finally:
                await _finalize_reader_tasks(task.reader_tasks, timeout=0.3 if force else 1.0)
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


# Module-level functions that delegate to the singleton manager
# These maintain backward compatibility with existing code

def _get_manager() -> BackgroundShellManager:
    """Get the singleton manager instance."""
    return BackgroundShellManager.get_instance()


def _get_tasks_lock() -> threading.Lock:
    """Get the tasks lock from the manager."""
    return _get_manager().tasks_lock


def _get_tasks() -> Dict[str, BackgroundTask]:
    """Get the tasks dict from the manager."""
    return _get_manager().tasks


def _safe_log_exception(message: str, **extra: Any) -> None:
    """Log an exception but never let logging failures bubble up."""
    try:
        logger.exception(message, extra=extra)
    except (OSError, RuntimeError, ValueError):
        pass


def _ensure_background_loop() -> asyncio.AbstractEventLoop:
    """Create (or return) a dedicated loop for background processes."""
    return _get_manager().ensure_loop()


def _submit_to_background_loop(coro: Any) -> concurrent.futures.Future:
    """Run a coroutine on the background loop and return a thread-safe future."""
    return _get_manager().submit_to_loop(coro)


async def _pump_stream(stream: asyncio.StreamReader, sink: List[str]) -> None:
    """Continuously read from a stream into a buffer."""
    try:
        while True:
            chunk = await stream.read(4096)
            if not chunk:
                break
            text = chunk.decode("utf-8", errors="replace")
            with _get_tasks_lock():
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
        with _get_tasks_lock():
            task.exit_code = task.process.returncode
    except asyncio.TimeoutError:
        logger.warning(f"Background task {task.id} timed out after {task.timeout}s: {task.command}")
        with _get_tasks_lock():
            task.timed_out = True
        task.process.kill()
        await task.process.wait()
        with _get_tasks_lock():
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
        with _get_tasks_lock():
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
    with _get_tasks_lock():
        _get_tasks()[task_id] = record

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
    tasks = _get_tasks()
    with _get_tasks_lock():
        if task_id not in tasks:
            raise KeyError(f"No background task found with id '{task_id}'")

        task = tasks[task_id]
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
        tasks = _get_tasks()
        with _get_tasks_lock():
            task = tasks.get(task_id)
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

            with _get_tasks_lock():
                task.exit_code = task.process.returncode or -1
            return True
        finally:
            await _finalize_reader_tasks(task.reader_tasks)
            task.done_event.set()

    future = _submit_to_background_loop(_kill(task_id))
    return await asyncio.wrap_future(future)


def list_background_tasks() -> List[str]:
    """Return known background task ids."""
    _prune_background_tasks()
    with _get_tasks_lock():
        return list(_get_tasks().keys())


def _prune_background_tasks(max_age_seconds: Optional[float] = None) -> int:
    """Remove finished background tasks older than the TTL."""
    ttl = DEFAULT_TASK_TTL_SEC if max_age_seconds is None else max_age_seconds
    if ttl is None or ttl <= 0:
        return 0
    now = _loop_time()
    removed = 0
    tasks = _get_tasks()
    with _get_tasks_lock():
        for task_id, task in list(tasks.items()):
            if task.exit_code is None:
                continue
            age = (now - task.start_time) if task.start_time else 0.0
            if age > ttl:
                tasks.pop(task_id, None)
                removed += 1
    return removed


def shutdown_background_shell(force: bool = False) -> None:
    """Stop background tasks/loop to avoid asyncio 'Event loop is closed' warnings.

    This function maintains backward compatibility by delegating to the manager.

    Args:
        force: If True, use minimal timeouts for faster exit.
    """
    _get_manager().shutdown(force=force)


def reset_background_shell_for_testing() -> None:
    """Reset all background shell state. Useful for testing.

    This function shuts down the current manager instance and clears it,
    allowing a fresh instance to be created on next access.
    """
    BackgroundShellManager.reset_instance()
