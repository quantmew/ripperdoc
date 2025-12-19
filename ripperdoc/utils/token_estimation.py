"""Shared token estimation utilities."""

from __future__ import annotations

import math

from ripperdoc.utils.log import get_logger

logger = get_logger()

# Optional: use tiktoken for accurate counts when available.
_TIKTOKEN_ENCODING: tiktoken.Encoding | None = None
try:  # pragma: no cover - optional dependency
    import tiktoken  # type: ignore

    _TIKTOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")
except (
    ImportError,
    ModuleNotFoundError,
    OSError,
    RuntimeError,
):  # pragma: no cover - runtime fallback
    pass


def estimate_tokens(text: str) -> int:
    """Estimate token count, preferring tiktoken when available."""
    if not text:
        return 0
    if _TIKTOKEN_ENCODING:
        try:
            return len(_TIKTOKEN_ENCODING.encode(text))
        except (UnicodeDecodeError, ValueError, RuntimeError):
            logger.debug("[token_estimation] tiktoken encode failed; falling back to heuristic")
    # Heuristic: ~4 characters per token
    return max(1, math.ceil(len(text) / 4))


__all__ = ["estimate_tokens"]
