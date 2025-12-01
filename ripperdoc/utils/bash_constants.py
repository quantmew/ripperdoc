"""Bash-related constants and helpers."""

from __future__ import annotations

import os
from typing import Optional

# Baseline defaults (kept in sync with the reference implementation)
_BASH_DEFAULT_TIMEOUT_MS = 120_000
_BASH_MAX_TIMEOUT_MS = 600_000
_BASH_MAX_OUTPUT_LENGTH = 30_000


def _parse_positive_int(value: Optional[str]) -> Optional[int]:
    """Best-effort conversion of an environment variable to a positive int."""
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def get_bash_max_output_length() -> int:
    """Return the maximum output length, honoring an env override when valid."""
    override = _parse_positive_int(os.getenv("BASH_MAX_OUTPUT_LENGTH"))
    return override or _BASH_MAX_OUTPUT_LENGTH


def get_bash_default_timeout_ms() -> int:
    """Return the default timeout, honoring an env override when valid."""
    override = _parse_positive_int(os.getenv("BASH_DEFAULT_TIMEOUT_MS"))
    return override or _BASH_DEFAULT_TIMEOUT_MS


def get_bash_max_timeout_ms() -> int:
    """Return the maximum timeout, never lower than the default timeout."""
    override = _parse_positive_int(os.getenv("BASH_MAX_TIMEOUT_MS"))
    baseline = _parse_positive_int(os.getenv("BASH_DEFAULT_TIMEOUT_MS"))
    default_timeout = baseline or _BASH_DEFAULT_TIMEOUT_MS
    if override:
        return max(override, default_timeout)
    return max(_BASH_MAX_TIMEOUT_MS, default_timeout)


__all__ = [
    "get_bash_max_output_length",
    "get_bash_default_timeout_ms",
    "get_bash_max_timeout_ms",
]
