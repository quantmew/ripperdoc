"""Memory tool persistent file operations."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from ripperdoc.utils.log import get_logger
from ripperdoc.utils.memory import AUTO_MEMORY_FILE_NAME, auto_memory_directory_path

logger = get_logger()

MAX_VIEW_CHARS = 100_000


def memory_root(memory_dir_override: Optional[Path] = None, project_path: Optional[Path] = None) -> Path:
    """Get the memory root directory."""
    root = memory_dir_override or auto_memory_directory_path(project_path)
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def resolve_path(raw_path: Optional[str], *, root: Path, allow_root: bool) -> Path:
    """Resolve a path relative to the memory root, with safety checks."""
    token = "." if raw_path is None else str(raw_path).strip()
    if not token:
        token = "."

    if token in {".", "/"}:
        resolved = root
    else:
        candidate = Path(token).expanduser()
        resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()

    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Path '{token}' is outside memory directory '{root}'."
        ) from exc

    if resolved == root and not allow_root:
        raise ValueError("This command requires a specific path inside the memory directory.")

    return resolved


def display_path(path: Path, root: Path) -> str:
    """Display a path relative to the memory root."""
    try:
        rel = path.resolve().relative_to(root)
    except ValueError:
        return str(path)
    if str(rel) == ".":
        return "."
    return rel.as_posix()


def read_text(path: Path) -> str:
    """Read text file content."""
    return path.read_text(encoding="utf-8", errors="ignore")


def write_text(path: Path, content: str) -> None:
    """Write text file content, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def delete_path(target: Path) -> str:
    """Delete a file or directory."""
    if target.is_dir():
        shutil.rmtree(target)
        return f"Deleted memory directory: {display_path(target, target.parent)}"
    else:
        target.unlink()
        return f"Deleted memory file: {display_path(target, target.parent)}"
