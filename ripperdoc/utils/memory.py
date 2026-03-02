"""Helpers for loading AGENTS.md and auto-memory instruction files."""

from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set

from ripperdoc.core.config import get_effective_config, get_global_config_path
from ripperdoc.utils.git_utils import get_git_root
from ripperdoc.utils.log import get_logger
from ripperdoc.utils.path_utils import sanitize_project_path

logger = get_logger()

MEMORY_FILE_NAME = "AGENTS.md"
LOCAL_MEMORY_FILE_NAME = "AGENTS.local.md"
AUTO_MEMORY_DIR_NAME = "memory"
AUTO_MEMORY_FILE_NAME = "MEMORY.md"
AUTO_MEMORY_NAME = "auto memory"
AUTO_MEMORY_LINE_LIMIT = 200

MEMORY_INSTRUCTIONS = (
    "Codebase and user instructions are shown below. Be sure to adhere to these "
    "instructions. IMPORTANT: These instructions OVERRIDE any default behavior "
    "and you MUST follow them exactly as written."
)

MAX_CONTENT_LENGTH = 40_000
MAX_INCLUDE_DEPTH = 5

_AUTO_MEMORY_DISABLE_ENV_KEYS = (
    "RIPPERDOC_DISABLE_AUTO_MEMORY",
)
_REMOTE_ENV_KEYS = ("RIPPERDOC_REMOTE",)
_REMOTE_MEMORY_DIR_ENV_KEYS = (
    "RIPPERDOC_REMOTE_MEMORY_DIR",
)
_ENABLE_SESSION_SEARCH_GUIDANCE_ENV = "RIPPERDOC_MEMORY_INCLUDE_SESSION_SEARCH"

_CODE_FENCE_RE = re.compile(r"```.*?```", flags=re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`]*`")
_MENTION_RE = re.compile(r"(?:^|\s)@((?:[^\s\\]|\\ )+)")
_PUNCT_START_RE = re.compile(r"^[#%^&*()]+")
_VALID_START_RE = re.compile(r"^[A-Za-z0-9._-]")


def _is_truthy(value: object) -> bool:
    """Return True for common truthy environment values."""
    if not value:
        return False
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    return normalized in {"1", "true", "yes", "on"}


def _is_false_string(value: object) -> bool:
    """Return True for common false-y string values (explicit toggle)."""
    if value is None:
        return False
    if isinstance(value, bool):
        return not value
    normalized = str(value).strip().lower()
    return normalized in {"0", "false", "no", "off"}


def _first_env_value(keys: tuple[str, ...]) -> Optional[str]:
    """Return the first non-empty environment value from candidate keys."""
    for key in keys:
        value = os.getenv(key)
        if value is not None and value != "":
            return value
    return None


def _normalize_nfc_path(path: Path) -> Path:
    """Normalize path representation to NFC to match upstream behavior."""
    return Path(unicodedata.normalize("NFC", str(path)))


def is_auto_memory_enabled(project_path: Optional[Path] = None) -> bool:
    """Check whether auto-memory is enabled.

    1) explicit env disable/enable (via false-string),
    2) remote mode restriction,
    3) settings field,
    4) default fallback (off in ripperdoc).
    """
    auto_memory_disabled = _first_env_value(_AUTO_MEMORY_DISABLE_ENV_KEYS)
    if _is_truthy(auto_memory_disabled):
        return False
    if _is_false_string(auto_memory_disabled):
        return True

    remote_mode = _first_env_value(_REMOTE_ENV_KEYS)
    remote_memory_dir = _first_env_value(_REMOTE_MEMORY_DIR_ENV_KEYS)
    if _is_truthy(remote_mode) and not remote_memory_dir:
        return False

    try:
        config = get_effective_config(project_path=project_path or Path.cwd())
        if config.auto_memory_enabled is not None:
            return bool(config.auto_memory_enabled)
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning("[memory] Failed to read effective config for auto-memory: %s", exc)

    return False


def memory_base_path() -> Path:
    """Get the base directory for memory storage."""
    remote_memory_dir = _first_env_value(_REMOTE_MEMORY_DIR_ENV_KEYS)
    if remote_memory_dir:
        return _normalize_nfc_path(Path(remote_memory_dir).expanduser())
    return _normalize_nfc_path(get_global_config_path().parent)


def _project_memory_root(project_path: Optional[Path] = None) -> Path:
    """Use git root when available; fallback to the provided/current project path."""
    root = (project_path or Path.cwd()).resolve()
    git_root = get_git_root(root)
    return (git_root or root).resolve()


def auto_memory_directory_path(project_path: Optional[Path] = None) -> Path:
    """Get the auto-memory directory path for the current project."""
    identifier = sanitize_project_path(_project_memory_root(project_path))
    path = memory_base_path() / "projects" / identifier / AUTO_MEMORY_DIR_NAME
    return _normalize_nfc_path(path)


def auto_memory_file_path(project_path: Optional[Path] = None) -> Path:
    """Get the auto-memory MEMORY.md path for the current project."""
    return auto_memory_directory_path(project_path) / AUTO_MEMORY_FILE_NAME


def is_auto_memory_path(file_path: str | Path, project_path: Optional[Path] = None) -> bool:
    """Return True when file_path is under the auto-memory directory."""
    base = os.path.normpath(str(auto_memory_directory_path(project_path)))
    candidate_path = Path(file_path).expanduser()
    if not candidate_path.is_absolute():
        candidate_path = (project_path or Path.cwd()) / candidate_path
    try:
        candidate = os.path.normpath(str(candidate_path.resolve()))
    except (OSError, RuntimeError, ValueError):
        candidate = os.path.normpath(str(candidate_path))
    return candidate == base or candidate.startswith(base + os.sep)


def _generate_search_guidelines(markdown_search_directory: Path) -> list[str]:
    """Generate optional memory/session-search guidance."""
    if not _is_truthy(os.getenv(_ENABLE_SESSION_SEARCH_GUIDANCE_ENV)):
        return []

    sessions_directory = get_global_config_path().parent / "projects"
    return [
        "## Searching past context",
        "",
        "When looking for past context:",
        "1. Search topic files in your memory directory:",
        "```",
        f'Grep with pattern="<search term>" path="{markdown_search_directory}" glob="*.md"',
        "```",
        "2. Session transcript logs (last resort — large files, slow):",
        "```",
        f'Grep with pattern="<search term>" path="{sessions_directory}/" glob="*.jsonl"',
        "```",
        "Use narrow search terms (error messages, file paths, function names) rather than broad keywords.",
        "",
    ]


def generate_auto_memory_guidelines(project_path: Optional[Path] = None) -> str:
    """Generate the auto-memory instruction block."""
    auto_memory_dir = auto_memory_directory_path(project_path)
    lines: list[str] = [
        f"# {AUTO_MEMORY_NAME}",
        "",
        f"You have a persistent auto memory directory at `{auto_memory_dir}`. Its contents persist across conversations.",
        "",
        "As you work, consult your memory files to build on previous experience.",
        "",
        "## How to save memories:",
        "- Organize memory semantically by topic, not chronologically",
        "- Prefer the Memory tool for memory file operations; use Write/Edit only when needed",
        f"- `{AUTO_MEMORY_FILE_NAME}` is always loaded into your conversation context — lines after {AUTO_MEMORY_LINE_LIMIT} will be truncated, so keep it concise",
        "- Create separate topic files (e.g., `debugging.md`, `patterns.md`) for detailed notes and link to them from MEMORY.md",
        "- Update or remove memories that turn out to be wrong or outdated",
        "- Do not write duplicate memories. First check if there is an existing memory you can update before writing a new one.",
        "",
        "## What to save:",
        "- Stable patterns and conventions confirmed across multiple interactions",
        "- Key architectural decisions, important file paths, and project structure",
        "- User preferences for workflow, tools, and communication style",
        "- Solutions to recurring problems and debugging insights",
        "",
        "## What NOT to save:",
        "- Session-specific context (current task details, in-progress work, temporary state)",
        "- Information that might be incomplete — verify against project docs before writing",
        "- Anything that duplicates or contradicts existing AGENTS.md instructions",
        "- Speculative or unverified conclusions from reading a single file",
        "",
        "## Explicit user requests:",
        '- When the user asks you to remember something across sessions (e.g., "always use bun", "never auto-commit"), save it — no need to wait for multiple interactions',
        "- When the user asks to forget or stop remembering something, find and remove the relevant entries from your memory files",
        "",
    ]
    lines.extend(_generate_search_guidelines(auto_memory_dir))
    return "\n".join(lines)


def _load_auto_memory_file_section(project_path: Optional[Path] = None) -> str:
    """Load MEMORY.md into prompt context with truncation rules."""
    auto_memory_dir = auto_memory_directory_path(project_path)
    auto_memory_file = auto_memory_dir / AUTO_MEMORY_FILE_NAME

    try:
        auto_memory_dir.mkdir(parents=True, exist_ok=True)
    except (OSError, RuntimeError, ValueError):
        pass

    file_content = ""
    try:
        file_content = auto_memory_file.read_text(encoding="utf-8", errors="ignore")
    except (OSError, RuntimeError, ValueError):
        file_content = ""

    if file_content.strip():
        lines = file_content.strip().split("\n")
        truncated_content = file_content.strip()
        if len(lines) > AUTO_MEMORY_LINE_LIMIT:
            truncated_content = (
                "\n".join(lines[:AUTO_MEMORY_LINE_LIMIT])
                + "\n\n"
                + (
                    f"> WARNING: {AUTO_MEMORY_FILE_NAME} is {len(lines)} lines "
                    f"(limit: {AUTO_MEMORY_LINE_LIMIT}). Only the first {AUTO_MEMORY_LINE_LIMIT} "
                    "lines were loaded. Move detailed content into separate topic files and keep "
                    f"{AUTO_MEMORY_FILE_NAME} as a concise index."
                )
            )
        return f"## {AUTO_MEMORY_FILE_NAME}\n\n{truncated_content}"

    return (
        f"## {AUTO_MEMORY_FILE_NAME}\n\n"
        f"Your {AUTO_MEMORY_FILE_NAME} is currently empty. When you notice a pattern worth "
        f"preserving across sessions, save it here. Anything in {AUTO_MEMORY_FILE_NAME} "
        "will be included in your system prompt next time."
    )


def generate_auto_memory_config(project_path: Optional[Path] = None) -> str:
    """Generate full auto-memory prompt block (guidelines + MEMORY.md contents)."""
    return "\n\n".join(
        [
            generate_auto_memory_guidelines(project_path=project_path),
            _load_auto_memory_file_section(project_path=project_path),
        ]
    )


def get_auto_memory_config(project_path: Optional[Path] = None) -> Optional[str]:
    """Return auto-memory instructions for the system prompt, or None when disabled."""
    if not is_auto_memory_enabled(project_path):
        auto_memory_disabled = _first_env_value(_AUTO_MEMORY_DISABLE_ENV_KEYS)
        disabled_by_env = _is_truthy(auto_memory_disabled)
        disabled_by_setting = False
        if not disabled_by_env:
            try:
                disabled_by_setting = (
                    get_effective_config(project_path=project_path or Path.cwd()).auto_memory_enabled
                    is False
                )
            except Exception:  # pragma: no cover - defensive logging only
                disabled_by_setting = False
        logger.debug(
            "[memory] Auto-memory disabled",
            extra={
                "disabled_by_env_var": disabled_by_env,
                "disabled_by_setting": disabled_by_setting,
            },
        )
        return None

    return generate_auto_memory_config(project_path=project_path)


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
    sections: list[str] = []

    auto_memory_config = get_auto_memory_config()
    if auto_memory_config:
        sections.append(auto_memory_config)

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

    if snippets:
        sections.append(f"{MEMORY_INSTRUCTIONS}\n\n" + "\n\n".join(snippets))

    if not sections:
        return ""
    return "\n\n".join(sections)


__all__ = [
    "MemoryFile",
    "AUTO_MEMORY_DIR_NAME",
    "AUTO_MEMORY_FILE_NAME",
    "AUTO_MEMORY_NAME",
    "collect_all_memory_files",
    "build_memory_instructions",
    "is_auto_memory_enabled",
    "memory_base_path",
    "auto_memory_directory_path",
    "auto_memory_file_path",
    "is_auto_memory_path",
    "generate_auto_memory_guidelines",
    "generate_auto_memory_config",
    "get_auto_memory_config",
    "get_oversized_memory_files",
    "MAX_CONTENT_LENGTH",
]
