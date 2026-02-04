"""Asyncio compatibility helpers (Python 3.10+)."""

from __future__ import annotations

import asyncio
from typing import Optional


class _TimeoutCompat:
    """Backport of asyncio.timeout for Python 3.10.

    This mirrors the basic behavior: cancel the current task after the
    deadline and translate that cancellation into asyncio.TimeoutError.
    """

    def __init__(self, delay: float) -> None:
        self._delay = delay
        self._task: Optional[asyncio.Task[object]] = None
        self._handle: Optional[asyncio.Handle] = None
        self._timed_out = False

    def _trigger_timeout(self) -> None:
        self._timed_out = True
        if self._task is not None and not self._task.cancelled():
            self._task.cancel()

    async def __aenter__(self) -> None:
        if self._delay is None:
            return None
        self._task = asyncio.current_task()
        if self._task is None:
            raise RuntimeError("asyncio_timeout must be used within a running task")
        loop = asyncio.get_running_loop()
        if self._delay <= 0:
            self._handle = loop.call_soon(self._trigger_timeout)
        else:
            self._handle = loop.call_later(self._delay, self._trigger_timeout)
        return None

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        if self._handle is not None:
            self._handle.cancel()
        if exc_type is asyncio.CancelledError and self._timed_out:
            raise asyncio.TimeoutError() from None
        return False


def asyncio_timeout(delay: float) -> object:
    """Return asyncio.timeout when available, otherwise a compat context manager."""
    timeout_factory = getattr(asyncio, "timeout", None)
    if timeout_factory is not None:
        return timeout_factory(delay)
    return _TimeoutCompat(delay)


__all__ = ["asyncio_timeout"]
