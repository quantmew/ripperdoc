"""Internal utility functions for background shell management."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from ripperdoc.utils.log import get_logger
from ripperdoc.utils.shell.shell_utils import build_shell_command, find_suitable_shell
from ripperdoc.tools.background_shell._models import BackgroundTask
from ripperdoc.tools.background_shell._manager import BackgroundShellManager

logger = get_logger()


def _get_manager() -> BackgroundShellManager:
    """Get the singleton manager instance."""
    return BackgroundShellManager.get_instance()


def _get_tasks_lock() -> threading.Lock:
    """Get the tasks lock from the manager."""
    return _get_manager().tasks_lock


def _get_tasks() -> Dict[str, BackgroundTask]:
    """Get the tasks dict from the manager."""
    return _get_manager().tasks


def _ensure_background_loop() -> asyncio.AbstractEventLoop:
    """Create (or return) a dedicated loop for background processes."""
    return _get_manager().ensure_loop()


def _submit_to_background_loop(coro: Any) -> concurrent.futures.Future:
    """Run a coroutine on the background loop and return a thread-safe future."""
    return _get_manager().submit_to_loop(coro)


def _loop_time() -> float:
    """Return a monotonic timestamp without requiring a running event loop."""
    try:
        return asyncio.get_running_loop().time()
    except RuntimeError:
        return time.monotonic()


def _run_completion_callbacks(task: BackgroundTask) -> None:
    """Invoke completion callbacks safely for a finished task."""
    if not task.completion_callbacks:
        return
    for callback in list(task.completion_callbacks):
        try:
            callback(task)
        except Exception:
            logger.debug(
                "Background task completion callback failed",
                exc_info=True,
                extra={"task_id": task.id, "command": task.command},
            )


def _compute_status(task: BackgroundTask) -> str:
    """Return a human-friendly status string."""
    if task.killed:
        return "killed"
    if task.timed_out:
        return "failed"
    if task.exit_code is None:
        return "running"
    return "completed" if task.exit_code == 0 else "failed"


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
            return
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
            task.end_time = task.end_time or _loop_time()
    except asyncio.TimeoutError:
        logger.warning(f"Background task {task.id} timed out after {task.timeout}s: {task.command}")
        with _get_tasks_lock():
            task.timed_out = True
        task.process.kill()
        await task.process.wait()
        with _get_tasks_lock():
            task.exit_code = -1
            task.end_time = task.end_time or _loop_time()
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
            task.end_time = task.end_time or _loop_time()
    finally:
        await _finalize_reader_tasks(task.reader_tasks)
        task.done_event.set()
        _run_completion_callbacks(task)


async def _start_background_command(
    command: str,
    timeout: Optional[float] = None,
    shell_executable: Optional[str] = None,
    cwd: Optional[str] = None,
    completion_callbacks: Optional[List[Callable[["BackgroundTask"], None]]] = None,
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
        cwd=cwd,
    )

    task_id = f"bash_{uuid.uuid4().hex[:8]}"
    record = BackgroundTask(
        id=task_id,
        command=command,
        process=process,
        start_time=_loop_time(),
        timeout=timeout,
        completion_callbacks=list(completion_callbacks or []),
    )
    with _get_tasks_lock():
        _get_tasks()[task_id] = record

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
