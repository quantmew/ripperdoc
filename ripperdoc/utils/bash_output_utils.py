"""Output helpers for BashTool."""

from __future__ import annotations

import os

from ripperdoc.utils.output_utils import trim_blank_lines, truncate_output
from ripperdoc.utils.safe_get_cwd import get_original_cwd, safe_get_cwd


def append_cwd_reset_message(message: str) -> str:
    """Append a notice when the working directory gets reset."""
    cleaned = message.rstrip()
    suffix = f"Shell cwd was reset to {get_original_cwd()}"
    if cleaned:
        return f"{cleaned}\n{suffix}"
    return suffix


def reset_cwd_if_needed(allowed_directories: set[str] | None = None) -> bool:
    """Placeholder that mirrors the reference contract.

    In this environment we simply report whether the current cwd is outside the
    provided allowed set and reset to the original cwd if so.
    """
    allowed_directories = allowed_directories or set()
    current = safe_get_cwd()
    if not allowed_directories:
        return False
    if current in allowed_directories:
        return False
    os.chdir(get_original_cwd())
    return True


__all__ = [
    "append_cwd_reset_message",
    "reset_cwd_if_needed",
    "trim_blank_lines",
    "truncate_output",
    "safe_get_cwd",
    "get_original_cwd",
]
