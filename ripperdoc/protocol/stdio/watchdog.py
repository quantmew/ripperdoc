"""Watchdog helpers for stdio protocol operations."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class OperationWatchdog:
    """Watchdog that monitors long-running operations and triggers timeout if stuck."""

    def __init__(self, timeout_sec: float, check_interval: float = 30.0):
        """Initialize watchdog.

        Args:
            timeout_sec: Maximum seconds allowed before watchdog triggers
            check_interval: Seconds between activity checks
        """
        self.timeout_sec = timeout_sec
        self.check_interval = check_interval
        self._last_activity: float = time.time()
        self._stopped = False
        self._task: asyncio.Task[None] | None = None
        self._monitoring_task: asyncio.Task[None] | None = None

    def _update_activity(self) -> None:
        """Update the last activity timestamp."""
        self._last_activity = time.time()

    def ping(self) -> None:
        """Update activity timestamp to prevent watchdog timeout."""
        self._update_activity()
        logger.debug(
            f"[watchdog] Activity ping recorded, time since last: {time.time() - self._last_activity:.1f}s"
        )

    async def _watchdog_loop(self) -> None:
        """Background task that monitors activity and triggers timeout if stuck."""
        while not self._stopped:
            try:
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break

            time_since_activity = time.time() - self._last_activity
            if time_since_activity > self.timeout_sec:
                logger.error(
                    f"[watchdog] No activity for {time_since_activity:.1f}s "
                    f"(timeout={self.timeout_sec:.1f}s) - triggering cancellation"
                )
                # Cancel the task being monitored
                if self._monitoring_task and not self._monitoring_task.done():
                    self._monitoring_task.cancel()
                break

    async def __aenter__(self) -> "OperationWatchdog":
        """Start the watchdog."""
        self._stopped = False
        self._monitoring_task = asyncio.current_task()
        self._task = asyncio.create_task(self._watchdog_loop())
        logger.debug(
            f"[watchdog] Started with timeout={self.timeout_sec}s, check_interval={self.check_interval}s"
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Stop the watchdog."""
        self._stopped = True
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.debug("[watchdog] Stopped")


__all__ = ["OperationWatchdog"]
