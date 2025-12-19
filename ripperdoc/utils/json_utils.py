"""JSON helper utilities for Ripperdoc."""

from __future__ import annotations

import json
from typing import Any, Optional

from ripperdoc.utils.log import get_logger


logger = get_logger()


def safe_parse_json(json_text: Optional[str], log_error: bool = True) -> Optional[Any]:
    """Best-effort JSON.parse wrapper that returns None on failure."""
    if not json_text:
        return None
    try:
        return json.loads(json_text)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        if log_error:
            logger.debug(
                "[json_utils] Failed to parse JSON: %s: %s",
                type(exc).__name__,
                exc,
                extra={"length": len(json_text)},
            )
        return None
