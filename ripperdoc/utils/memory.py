"""Helpers for loading AGENTS.md memory files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set
from ripperdoc.utils.log import get_logger

logger = get_logger()

MEMORY_FILE_NAME = "AGENTS.md"
LOCAL_MEMORY_FILE_NAME = "AGENTS.local.md"

MEMORY_INSTRUCTIONS = (
    "Codebase and user instructions are shown below. Be sure to adhere to these "
    "instructions. IMPORTANT: These instructions OVERRIDE any default behavior "
    "and you MUST follow them exactly as written."
)

MAX_CONTENT_LENGTH = 40_000
MAX_INCLUDE_DEPTH = 5

_CODE_FENCE_RE = re.compile(r"```.*?```", flags=re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`]*`")
_MENTION_RE = re.compile(r"(?:^|\s)@((?:[^\s\\]|\\ )+)")
_PUNCT_START_RE = re.compile(r"^[#%^&*()]+")
_VALID_START_RE = re.compile(r"^[A-Za-z0-9._-]")


@dataclass
class MemoryFile:
    """Representation of a loaded memory file."""

    path: str
    type: str
    content: str
    parent: Optional[str] = None
    is_nested: bool = False


def _is_path_under_directory(path: Path, directory: Path) -> bool:
    """Return True if path is inside directory (after resolving)."""
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except (ValueError, OSError) as exc:
        logger.warning(
            "[memory] Failed to compare path containment: %s: %s",
            type(exc).__name__,
            exc,
            extra={"path": str(path), "directory": str(directory)},
        )
        return False


def _resolve_relative_path(raw_path: str, base_path: Path) -> Path:
    """Resolve a mention (./foo, ~/bar, /abs, or relative) against a base file."""
    normalized = raw_path.replace("\\ ", " ")
    if normalized.startswith("~/"):
        return (Path.home() / normalized[2:]).resolve()
    candidate = Path(normalized)
    if not candidate.is_absolute():
        return (base_path.parent / candidate).resolve()
    return candidate.resolve()


def _read_file_with_type(file_path: Path, file_type: str) -> Optional[MemoryFile]:
    """Read a file if it exists, returning a MemoryFile entry."""
    try:
        if not file_path.exists() or not file_path.is_file():
            return None
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        return MemoryFile(path=str(file_path), type=file_type, content=content)
    except PermissionError:
        logger.exception("[memory] Permission error reading file", extra={"path": str(file_path)})
        return None
    except OSError:
        logger.exception("[memory] OS error reading file", extra={"path": str(file_path)})
        return None


def _extract_relative_paths_from_markdown(markdown_content: str, base_path: Path) -> List[Path]:
    """Extract @-mentions that look like file paths from markdown content."""
    if not markdown_content:
        return []

    cleaned = _CODE_FENCE_RE.sub("", markdown_content)
    cleaned = _INLINE_CODE_RE.sub("", cleaned)

    relative_paths: Set[Path] = set()
    for match in _MENTION_RE.finditer(cleaned):
        mention = (match.group(1) or "").replace("\\ ", " ").strip()
        if not mention or mention.startswith("@"):
            continue

        if not (
            mention.startswith("./")
            or mention.startswith("~/")
            or (mention.startswith("/") and mention != "/")
            or (not _PUNCT_START_RE.match(mention) and _VALID_START_RE.match(mention))
        ):
            continue

        resolved = _resolve_relative_path(mention, base_path)
        relative_paths.add(resolved)

    return list(relative_paths)


def _collect_files(
    file_path: Path,
    file_type: str,
    visited: Set[str],
    allow_outside_cwd: bool,
    depth: int = 0,
    parent_path: Optional[Path] = None,
) -> List[MemoryFile]:
    """Collect a memory file and any nested references."""
    if depth >= MAX_INCLUDE_DEPTH:
        return []

    resolved_path = file_path.expanduser()
    try:
        resolved_path = resolved_path.resolve()
    except (OSError, ValueError) as exc:
        logger.warning(
            "[memory] Failed to resolve memory file path: %s: %s",
            type(exc).__name__,
            exc,
            extra={"path": str(resolved_path)},
        )

    resolved_key = str(resolved_path)
    if resolved_key in visited:
        return []

    current_file = _read_file_with_type(resolved_path, file_type)
    if not current_file or not current_file.content.strip():
        return []

    if parent_path is not None:
        current_file.parent = str(parent_path)
        current_file.is_nested = depth > 0

    visited.add(resolved_key)

    collected: List[MemoryFile] = [current_file]
    relative_paths = _extract_relative_paths_from_markdown(current_file.content, resolved_path)
    for nested_path in relative_paths:
        if not allow_outside_cwd and not _is_path_under_directory(nested_path, Path.cwd()):
            continue
        collected.extend(
            _collect_files(
                nested_path,
                file_type,
                visited,
                allow_outside_cwd,
                depth + 1,
                resolved_path,
            )
        )

    return collected


def collect_all_memory_files(force_include_external: bool = False) -> List[MemoryFile]:
    """Collect all AGENTS memory files reachable from the working directory."""
    visited: Set[str] = set()
    files: List[MemoryFile] = []

    # Global/user-level memories live in home and ~/.ripperdoc.
    user_memory_paths = [
        Path.home() / ".ripperdoc" / MEMORY_FILE_NAME,
        Path.home() / MEMORY_FILE_NAME,
    ]
    for user_memory_path in user_memory_paths:
        files.extend(
            _collect_files(
                user_memory_path,
                "User",
                visited,
                allow_outside_cwd=True,
            )
        )

    # Project memories from the current working directory up to the filesystem root.
    ancestor_dirs: List[Path] = []
    current_dir = Path.cwd()
    while True:
        ancestor_dirs.append(current_dir)
        if current_dir.parent == current_dir:
            break
        current_dir = current_dir.parent

    for directory in reversed(ancestor_dirs):
        files.extend(
            _collect_files(
                directory / MEMORY_FILE_NAME,
                "Project",
                visited,
                allow_outside_cwd=force_include_external,
            )
        )
        files.extend(
            _collect_files(
                directory / LOCAL_MEMORY_FILE_NAME,
                "Local",
                visited,
                allow_outside_cwd=force_include_external,
            )
        )

    return files


def get_oversized_memory_files() -> List[MemoryFile]:
    """Return memory files that exceed the recommended length."""
    return [file for file in collect_all_memory_files() if len(file.content) > MAX_CONTENT_LENGTH]


def build_memory_instructions() -> str:
    """Build the instruction block to append to the system prompt."""
    memory_files = collect_all_memory_files()
    snippets: List[str] = []
    for memory_file in memory_files:
        if not memory_file.content:
            continue
        type_description = (
            " (project instructions, checked into the codebase)"
            if memory_file.type == "Project"
            else (
                " (user's private project instructions, not checked in)"
                if memory_file.type == "Local"
                else " (user's private global instructions)"
            )
        )
        snippets.append(
            f"Contents of {memory_file.path}{type_description}:\n\n{memory_file.content}"
        )

    if not snippets:
        return ""

    return f"{MEMORY_INSTRUCTIONS}\n\n" + "\n\n".join(snippets)


__all__ = [
    "MemoryFile",
    "collect_all_memory_files",
    "build_memory_instructions",
    "get_oversized_memory_files",
    "MAX_CONTENT_LENGTH",
]
