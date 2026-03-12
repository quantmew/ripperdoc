"""Timeouts for stdio protocol handling."""

from __future__ import annotations

import os

# Timeout constants for stdio operations
STDIO_QUERY_TIMEOUT_SEC = float(os.getenv("RIPPERDOC_STDIO_QUERY_TIMEOUT", "600"))  # 10 minutes default
STDIO_WATCHDOG_INTERVAL_SEC = float(os.getenv("RIPPERDOC_STDIO_WATCHDOG_INTERVAL", "30"))  # 30 seconds
STDIO_HOOK_TIMEOUT_SEC = float(os.getenv("RIPPERDOC_STDIO_HOOK_TIMEOUT", "30"))  # 30 seconds for hooks

__all__ = [
    "STDIO_QUERY_TIMEOUT_SEC",
    "STDIO_WATCHDOG_INTERVAL_SEC",
    "STDIO_HOOK_TIMEOUT_SEC",
]
