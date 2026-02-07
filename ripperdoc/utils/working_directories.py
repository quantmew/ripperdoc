"""Helpers for parsing and normalizing additional working directories."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


def coerce_directory_list(value: Any) -> list[str]:
    """Coerce a settings value into a list of directory strings."""
    if value is None:
        return []
    if isinstance(value, str):
        trimmed = value.strip()
        return [trimmed] if trimmed else []
    if isinstance(value, (list, tuple, set)):
        result: list[str] = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                result.append(text)
        return result
    return []


def extract_additional_directories(settings_data: Any) -> list[str]:
    """Extract additional directory entries from parsed settings payload."""
    if not isinstance(settings_data, dict):
        return []

    if "working_directories" in settings_data:
        return coerce_directory_list(settings_data.get("working_directories"))
    return []


def resolve_directory_path(raw_path: str | Path, *, base_dir: Path) -> Path:
    """Resolve a potentially relative directory path against a base directory."""
    candidate = Path(str(raw_path).strip()).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate.resolve()


def normalize_directory_inputs(
    raw_paths: Iterable[str | Path],
    *,
    base_dir: Path,
    require_exists: bool,
) -> tuple[list[str], list[str]]:
    """Normalize directory inputs into resolved absolute paths.

    Returns:
        A tuple ``(normalized_paths, errors)`` where errors contains user-facing
        messages for invalid items.
    """
    normalized: list[str] = []
    errors: list[str] = []
    seen: set[str] = set()

    for raw in raw_paths:
        text = str(raw).strip()
        if not text:
            continue
        try:
            resolved = resolve_directory_path(text, base_dir=base_dir)
        except (OSError, RuntimeError, ValueError) as exc:
            errors.append(f"{text}: failed to resolve path ({type(exc).__name__})")
            continue

        if require_exists:
            if not resolved.exists():
                errors.append(f"{text}: directory does not exist")
                continue
            if not resolved.is_dir():
                errors.append(f"{text}: path is not a directory")
                continue

        resolved_str = str(resolved)
        if resolved_str in seen:
            continue
        seen.add(resolved_str)
        normalized.append(resolved_str)

    return normalized, errors


__all__ = [
    "coerce_directory_list",
    "extract_additional_directories",
    "normalize_directory_inputs",
    "resolve_directory_path",
]
