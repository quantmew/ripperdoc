"""Filesystem path helpers."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


def _legacy_sanitize_project_path(project_path: Path) -> str:
    """Legacy sanitizer that strips non-alphanumeric characters."""
    normalized = str(project_path.resolve())
    return re.sub(r"[^a-zA-Z0-9]+", "-", normalized).strip("-") or "project"


def sanitize_project_path(project_path: Path) -> str:
    """Make a project path safe for directory names and avoid collisions.

    Non-alphanumeric characters (including non-ASCII) are replaced with "-".
    A short hash of the full resolved path is appended to prevent collisions
    between different paths that would otherwise sanitize to the same string.
    """
    normalized = str(project_path.resolve())
    safe = _legacy_sanitize_project_path(project_path)
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:8]
    return f"{safe}-{digest}"


def project_storage_dir(base_dir: Path, project_path: Path, ensure: bool = False) -> Path:
    """Return a storage directory path for a project, with legacy fallback.

    Prefers a hashed, collision-safe name but will reuse an existing legacy
    directory (pre-hash) to avoid stranding older data.
    """
    hashed_name = sanitize_project_path(project_path)
    legacy_name = _legacy_sanitize_project_path(project_path)

    hashed_dir = base_dir / hashed_name
    legacy_dir = base_dir / legacy_name

    chosen = hashed_dir if hashed_dir.exists() or not legacy_dir.exists() else legacy_dir

    if ensure:
        chosen.mkdir(parents=True, exist_ok=True)

    return chosen
