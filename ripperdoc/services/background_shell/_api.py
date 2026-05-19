"""Public API for background shell management.

Provides module-level async functions for starting, monitoring, and
controlling background shell commands.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import threading
import uuid
from typing import Any, Callable, Dict, List, Optional

from ripperdoc.utils.log import get_logger
from ripperdoc.services.background_shell._models import BackgroundTask
from ripperdoc.services.background_shell._manager import (
    BackgroundShellManager,
    DEFAULT_TASK_TTL_SEC,
)
from ripperdoc.services.background_shell._utils import (
    _compute_status,
    _finalize_reader_tasks,
    _get_manager,
    _get_tasks,
    _get_tasks_lock,
    _loop_time,
    _monitor_task,
    _pump_stream,
    _run_completion_callbacks,
    _start_background_command,
    _submit_to_background_loop,
)

logger = get_logger()


async def start_background_command(
    command: str,
    timeout: Optional[float] = None,
    shell_executable: Optional[str] = None,
    cwd: Optional[str] = None,
    completion_callbacks: Optional[List[Callable[["BackgroundTask"], None]]] = None,
) -> str:
    """Launch a background shell command and return its task id."""
    future = _submit_to_background_loop(
        _start_background_command(
            command,
            timeout,
            shell_executable,
            cwd,
            completion_callbacks=completion_callbacks,
        )
    )
    return await asyncio.wrap_future(future)


async def register_existing_process(
    command: str,
    process: asyncio.subprocess.Process,
    *,
    timeout: Optional[float] = None,
    task_id: Optional[str] = None,
    stdout_chunks: Optional[List[str]] = None,
    stderr_chunks: Optional[List[str]] = None,
    reader_tasks: Optional[List[asyncio.Task]] = None,
    completion_callbacks: Optional[List[Callable[["BackgroundTask"], None]]] = None,
) -> str:
    """Register an already-running process as a background task.

    This is used to auto-background a foreground command after it exceeds its
    foreground timeout budget, without restarting the process.
    """
    normalized_task_id = task_id or f"bash_{uuid.uuid4().hex[:8]}"

    record = BackgroundTask(
        id=normalized_task_id,
        command=command,
        process=process,
        start_time=_loop_time(),
        timeout=timeout,
        stdout_chunks=stdout_chunks if stdout_chunks is not None else [],
        stderr_chunks=stderr_chunks if stderr_chunks is not None else [],
        reader_tasks=list(reader_tasks) if reader_tasks is not None else [],
        completion_callbacks=list(completion_callbacks or []),
    )

    with _get_tasks_lock():
        _get_tasks()[normalized_task_id] = record

    if not record.reader_tasks:
        if process.stdout:
            record.reader_tasks.append(
                asyncio.create_task(_pump_stream(process.stdout, record.stdout_chunks))
            )
        if process.stderr:
            record.reader_tasks.append(
                asyncio.create_task(_pump_stream(process.stderr, record.stderr_chunks))
            )

    asyncio.create_task(_monitor_task(record))
    return normalized_task_id


def get_background_status(task_id: str, consume: bool = True) -> dict:
    """Fetch the current status and buffered output of a background command.

    If consume is True, buffered stdout/stderr are cleared after reading.
    """
    now = _loop_time()
    tasks = _get_tasks()
    should_dispatch_callbacks = False

    status = {}
    with _get_tasks_lock():
        if task_id not in tasks:
            raise KeyError(f"No background task found with id '{task_id}'")

        task = tasks[task_id]
        stdout = "".join(task.stdout_chunks)
        stderr = "".join(task.stderr_chunks)

        if task.exit_code is None and task.process.returncode is not None:
            task.exit_code = task.process.returncode
            task.end_time = task.end_time or now
            task.done_event.set()
            should_dispatch_callbacks = True

        finished = task.exit_code is not None or task.killed or task.timed_out
        if finished and task.end_time is None:
            task.end_time = now
        duration_ms = (
            ((task.end_time or now) - task.start_time) * 1000.0 if task.start_time else None
        )
        age_ms = (now - task.start_time) * 1000.0 if task.start_time else None

        if consume:
            task.stdout_chunks.clear()
            task.stderr_chunks.clear()

        status = {
            "id": task.id,
            "command": task.command,
            "status": _compute_status(task),
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": task.exit_code,
            "timed_out": task.timed_out,
            "killed": task.killed,
            "duration_ms": duration_ms,
            "age_ms": age_ms,
        }

    if should_dispatch_callbacks:
        _run_completion_callbacks(task)
        return status

    return status


async def kill_background_task(task_id: str) -> bool:
    """Attempt to kill a running background task."""
    KILL_WAIT_SECONDS = 2.0

    def _resolve_task_loop(task: BackgroundTask) -> Optional[asyncio.AbstractEventLoop]:
        for reader_task in task.reader_tasks:
            try:
                return reader_task.get_loop()
            except RuntimeError:
                continue
        return None

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
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    task.process.kill()
                await asyncio.wait_for(task.process.wait(), timeout=1.0)
            except RuntimeError:
                pass

            with _get_tasks_lock():
                task.exit_code = task.process.returncode or -1
                task.end_time = task.end_time or _loop_time()
            return True
        finally:
            try:
                await _finalize_reader_tasks(task.reader_tasks)
            except RuntimeError:
                pass
            with contextlib.suppress(RuntimeError):
                task.done_event.set()

    with _get_tasks_lock():
        task = _get_tasks().get(task_id)

    if task is not None:
        target_loop = _resolve_task_loop(task)
        if target_loop is not None and target_loop.is_running():
            try:
                current_loop = asyncio.get_running_loop()
            except RuntimeError:
                current_loop = None
            if current_loop is target_loop:
                return await _kill(task_id)
            future = asyncio.run_coroutine_threadsafe(_kill(task_id), target_loop)
            return await asyncio.wrap_future(future)

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

    Args:
        force: If True, use minimal timeouts for faster exit.
    """
    _get_manager().shutdown(force=force)


def reset_background_shell_for_testing() -> None:
    """Reset all background shell state. Useful for testing."""
    BackgroundShellManager.reset_instance()
