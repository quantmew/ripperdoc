"""Timeouts and timeout utilities for stdio protocol handling."""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Callable

logger = logging.getLogger(__name__)

# Timeout constants for stdio operations
STDIO_READ_TIMEOUT_SEC = float(os.getenv("RIPPERDOC_STDIO_READ_TIMEOUT", "300"))  # 5 minutes default
STDIO_QUERY_TIMEOUT_SEC = float(os.getenv("RIPPERDOC_STDIO_QUERY_TIMEOUT", "600"))  # 10 minutes default
STDIO_WATCHDOG_INTERVAL_SEC = float(os.getenv("RIPPERDOC_STDIO_WATCHDOG_INTERVAL", "30"))  # 30 seconds
STDIO_TOOL_TIMEOUT_SEC = float(os.getenv("RIPPERDOC_STDIO_TOOL_TIMEOUT", "300"))  # 5 minutes per tool
STDIO_HOOK_TIMEOUT_SEC = float(os.getenv("RIPPERDOC_STDIO_HOOK_TIMEOUT", "30"))  # 30 seconds for hooks


@asynccontextmanager
async def timeout_wrapper(
    timeout_sec: float,
    operation_name: str,
    on_timeout: Callable[[str], Any] | None = None,
) -> AsyncGenerator[None, None]:
    """Context manager that wraps an async operation with timeout and comprehensive error handling.

    Args:
        timeout_sec: Maximum seconds to wait for the operation
        operation_name: Human-readable name for logging
        on_timeout: Optional callback called on timeout

    Yields:
        None

    Raises:
        asyncio.TimeoutError: If operation exceeds timeout
    """
    try:
        async with asyncio.timeout(timeout_sec):
            yield
    except asyncio.TimeoutError:
        error_msg = f"{operation_name} timed out after {timeout_sec:.1f}s"
        logger.error(f"[timeout] {error_msg}", exc_info=True)
        if on_timeout:
            result = on_timeout(error_msg)
            if inspect.isawaitable(result):
                await result
        raise
    except Exception as e:
        logger.error(f"[timeout] {operation_name} failed: {type(e).__name__}: {e}", exc_info=True)
        raise


__all__ = [
    "STDIO_READ_TIMEOUT_SEC",
    "STDIO_QUERY_TIMEOUT_SEC",
    "STDIO_WATCHDOG_INTERVAL_SEC",
    "STDIO_TOOL_TIMEOUT_SEC",
    "STDIO_HOOK_TIMEOUT_SEC",
    "timeout_wrapper",
]
