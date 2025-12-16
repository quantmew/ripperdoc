"""Lightweight parsing helpers for permissive type coercion."""

from __future__ import annotations

from typing import Optional


def parse_boolish(value: object, default: bool = False) -> bool:
    """Parse a truthy/falsey value from common representations."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def parse_optional_int(value: object) -> Optional[int]:
    """Best-effort int parsing; returns None on failure."""
    try:
        if value is None:
            return None
        if isinstance(value, bool):
            return int(value)
        return int(str(value).strip())
    except (ValueError, TypeError):
        return None
