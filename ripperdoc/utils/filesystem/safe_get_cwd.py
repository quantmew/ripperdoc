"""Safe helpers for tracking and restoring the working directory."""

from __future__ import annotations

import os
from pathlib import Path
from ripperdoc.utils.log import get_logger

logger = get_logger()

_ORIGINAL_CWD = Path(os.getcwd()).resolve()


def get_original_cwd() -> str:
    """Return the process's initial working directory."""
    return str(_ORIGINAL_CWD)


def safe_get_cwd() -> str:
    """Return the current working directory, falling back to the original on error."""
    try:
        return str(Path(os.getcwd()).resolve())
    except (OSError, RuntimeError, ValueError) as exc:
        logger.warning(
            "[safe_get_cwd] Failed to resolve cwd: %s: %s",
            type(exc).__name__,
            exc,
        )
        return get_original_cwd()


__all__ = ["get_original_cwd", "safe_get_cwd"]
