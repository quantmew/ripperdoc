"""Manager for background shell tasks with proper lifecycle control.

This class encapsulates all global state for background shell management,
providing better testability and proper resource cleanup.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import os
import threading
import time
import weakref
from typing import Any, Dict, List, Optional

import atexit

from ripperdoc.utils.log import get_logger
from ripperdoc.services.background_shell._models import BackgroundTask

logger = get_logger()

DEFAULT_TASK_TTL_SEC = float(os.getenv("RIPPERDOC_BASH_TASK_TTL_SEC", "3600"))


class BackgroundShellManager:
    """Manager for background shell tasks with proper lifecycle control."""

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
        """Reset the singleton instance. Useful for testing."""
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
                    shutdown_event.set()

            thread = threading.Thread(
                target=_run_loop,
                name="ripperdoc-bg-loop",
                daemon=False,
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
                try:
                    loop.run_until_complete(self._shutdown_loop_async(loop, force=force))
                except RuntimeError:
                    pass
        finally:
            if thread and thread.is_alive():
                thread.join(timeout=join_timeout)
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
        """Drain running background processes before stopping the loop."""
        from ripperdoc.services.background_shell._utils import (
            _finalize_reader_tasks,
            _loop_time,
            _run_completion_callbacks,
        )

        with self._tasks_lock:
            tasks = list(self._tasks.values())
            self._tasks.clear()

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
                task.end_time = task.end_time or _loop_time()
            except (OSError, RuntimeError, asyncio.CancelledError) as exc:
                if not isinstance(exc, asyncio.CancelledError):
                    logger.exception(
                        "Error shutting down background task",
                        extra={"task_id": task.id, "command": task.command},
                    )
            finally:
                await _finalize_reader_tasks(task.reader_tasks, timeout=0.3 if force else 1.0)
                task.done_event.set()
                _run_completion_callbacks(task)

        current = asyncio.current_task()
        pending = [t for t in asyncio.all_tasks(loop) if t is not current]
        for pending_task in pending:
            pending_task.cancel()
        if pending:
            with contextlib.suppress(Exception):
                await asyncio.gather(*pending, return_exceptions=True)

        with contextlib.suppress(Exception):
            await loop.shutdown_asyncgens()
