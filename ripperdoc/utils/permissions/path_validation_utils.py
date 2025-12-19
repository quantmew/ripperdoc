"""Path validation utilities for shell commands (cd/ls/find)."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Set

from ripperdoc.utils.safe_get_cwd import safe_get_cwd
from ripperdoc.utils.shell_token_utils import parse_and_clean_shell_tokens
from ripperdoc.utils.log import get_logger

logger = get_logger()

_GLOB_PATTERN = re.compile(r"[*?\[\]{}]")
_MAX_VISIBLE_ITEMS = 5


@dataclass
class ValidationResponse:
    behavior: str  # 'passthrough' | 'ask' | 'deny'
    message: str
    rule_suggestions: None | list[str] = None


def _format_allowed_dirs_preview(allowed_dirs: Iterable[str]) -> str:
    dirs = list(allowed_dirs)
    if len(dirs) <= _MAX_VISIBLE_ITEMS:
        return ", ".join(f"'{item}'" for item in dirs)
    return (
        ", ".join(f"'{item}'" for item in dirs[:_MAX_VISIBLE_ITEMS])
        + f", and {len(dirs) - _MAX_VISIBLE_ITEMS} more"
    )


def _expand_tilde(path_str: str) -> str:
    if path_str == "~" or path_str.startswith("~/"):
        return os.path.expanduser(path_str)
    return path_str


def _resolve_path(raw_path: str, cwd: str) -> Path:
    expanded = _expand_tilde(raw_path.strip("'\""))
    candidate = Path(expanded)
    if not candidate.is_absolute():
        candidate = Path(cwd) / candidate
    try:
        return candidate.resolve()
    except (OSError, ValueError) as exc:
        logger.warning(
            "[path_validation] Failed to resolve path: %s: %s",
            type(exc).__name__,
            exc,
            extra={"raw_path": raw_path, "cwd": cwd},
        )
        return candidate


def _extract_directory_from_glob(glob_pattern: str) -> str:
    match = _GLOB_PATTERN.search(glob_pattern)
    if not match or match.start() == 0:
        return glob_pattern
    prefix = glob_pattern[: match.start()]
    if "/" not in prefix:
        return "."
    return prefix.rsplit("/", 1)[0] or "/"


def _is_path_allowed(resolved_path: Path, allowed_dirs: Set[str]) -> bool:
    resolved_str = str(resolved_path)
    for allowed in allowed_dirs:
        normalized_allowed = os.path.abspath(allowed)
        normalized = os.path.abspath(resolved_str)
        if normalized == normalized_allowed:
            return True
        if normalized.startswith(normalized_allowed.rstrip(os.sep) + os.sep):
            return True
    return False


def _validate_path(raw_path: str, cwd: str, allowed_dirs: Set[str]) -> tuple[bool, str]:
    expanded = _expand_tilde(raw_path.strip() or ".")
    if _GLOB_PATTERN.search(expanded):
        directory = _extract_directory_from_glob(expanded)
        resolved_dir = _resolve_path(directory, cwd)
        return _is_path_allowed(resolved_dir, allowed_dirs), str(resolved_dir)

    resolved = _resolve_path(expanded, cwd)
    return _is_path_allowed(resolved, allowed_dirs), str(resolved)


def _check_command_paths(
    command: str, args: List[str], cwd: str, allowed_dirs: Set[str]
) -> ValidationResponse:
    if command == "cd":
        target = args[0] if args else os.path.expanduser("~")
        allowed, resolved = _validate_path(target, cwd, allowed_dirs)
    elif command == "ls":
        # ls is a read-only command, allow it to run on any path
        # This enables viewing system directories like /usr, /etc, etc.
        return ValidationResponse(
            behavior="passthrough",
            message="ls is a read-only command, no path restrictions applied",
            rule_suggestions=None,
        )
    elif command == "find":
        paths: list[str] = []
        for arg in args:
            if arg.startswith("-"):
                continue
            paths.append(arg)
        if not paths:
            paths = ["."]
        allowed = True
        resolved = ""
        for candidate in paths:
            allowed, resolved = _validate_path(candidate, cwd, allowed_dirs)
            if not allowed:
                break
    else:
        return ValidationResponse(
            behavior="passthrough",
            message=f"Command '{command}' is not path-restricted",
            rule_suggestions=None,
        )

    if allowed:
        return ValidationResponse(
            behavior="passthrough",
            message="Path validation passed",
            rule_suggestions=None,
        )

    preview = _format_allowed_dirs_preview(sorted(allowed_dirs))
    action = {
        "cd": "change directory to",
        "find": "search files in",
    }.get(command, "access")
    return ValidationResponse(
        behavior="ask",
        message=f"Requesting permission to {action} '{resolved}' (outside allowed directories: {preview})",
        rule_suggestions=None,
    )


def validate_shell_command_paths(
    shell_command: str | object, cwd: str | None = None, allowed_dirs: Set[str] | None = None
) -> ValidationResponse:
    """Validate path-oriented shell commands against allowed working directories."""
    command_str = shell_command.command if hasattr(shell_command, "command") else str(shell_command)
    cwd = cwd or safe_get_cwd()
    allowed_dirs = allowed_dirs or {cwd}

    tokens = parse_and_clean_shell_tokens(command_str)
    if not tokens:
        return ValidationResponse(
            behavior="passthrough",
            message="Empty command - no paths to validate",
            rule_suggestions=None,
        )

    first, *rest = tokens
    if first not in {"cd", "ls", "find"}:
        return ValidationResponse(
            behavior="passthrough",
            message=f"Command '{first}' is not a path-restricted command",
            rule_suggestions=None,
        )

    return _check_command_paths(first, rest, cwd, allowed_dirs)


__all__ = ["ValidationResponse", "validate_shell_command_paths"]
