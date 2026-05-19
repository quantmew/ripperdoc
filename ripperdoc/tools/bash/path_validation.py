"""Path validation for bash commands.


Provides path extraction from shell commands and validation against
allowed working directories. Handles tilde expansion, POSIX -- handling,
dangerous removal detection, and git internal path protection.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

from ripperdoc.utils.bash.shell_quote import try_parse_shell_command
from ripperdoc.utils.bash.commands import (
    extract_output_redirections,
)
from ripperdoc.security import PermissionResult


# ============================================================================
# Types
# ============================================================================

# All commands that have path extractors defined
PathCommand = str

# Operation type for each command: 'read', 'write', 'create'
COMMAND_OPERATION_TYPE: Dict[str, str] = {
    "cd": "read",
    "ls": "read", "find": "read",
    "mkdir": "create", "touch": "create",
    "rm": "write", "rmdir": "write",
    "mv": "write", "cp": "create",
    "cat": "read", "head": "read", "tail": "read",
    "sort": "read", "uniq": "read", "wc": "read",
    "cut": "read", "paste": "read", "column": "read",
    "tr": "read", "file": "read", "stat": "read",
    "diff": "read",
    "strings": "read", "hexdump": "read", "od": "read",
    "base64": "read", "nl": "read",
    "grep": "read", "rg": "read",
    "sed": "write",
    "jq": "read",
    "sha256sum": "read", "sha1sum": "read", "md5sum": "read",
}


def _is_dangerous_removal_path(path_str: str) -> bool:
    """Check if a path is dangerously destructive to remove.

    Args:
        path_str: Absolute path to check.

    Returns:
        True if the path is dangerously destructive.
    """
    dangerous = {"/", "/etc", "/bin", "/sbin", "/usr", "/boot", "/dev", "/proc", "/sys"}
    return path_str in dangerous or path_str.rstrip("/") in dangerous


# ============================================================================
# Path extraction helpers
# ============================================================================


def filter_out_flags(args: List[str]) -> List[str]:
    """Extract positional arguments, handling POSIX `--` correctly.

    Args:
        args: List of argument tokens.

    Returns:
        List of positional arguments only.
    """
    result: List[str] = []
    after_dd = False
    for arg in args:
        if after_dd:
            result.append(arg)
        elif arg == "--":
            after_dd = True
        elif not arg.startswith("-"):
            result.append(arg)
    return result


def parse_pattern_command(
    args: List[str],
    flags_with_args: Set[str],
    defaults: Optional[List[str]] = None,
) -> List[str]:
    """Parse grep/rg style commands (pattern then paths).

    Args:
        args: Tokenized arguments after command name.
        flags_with_args: Set of flags that consume an argument.
        defaults: Default paths if none found.

    Returns:
        List of extracted paths.
    """
    paths: List[str] = []
    pattern_found = False
    after_dd = False

    for i, arg in enumerate(args):
        if after_dd:
            paths.append(arg)
            continue

        if arg == "--":
            after_dd = True
            continue

        if arg.startswith("-"):
            flag = arg.split("=")[0]
            if flag in ("-e", "--regexp", "-f", "--file"):
                pattern_found = True
            if flag in flags_with_args and "=" not in arg:
                i += 1  # Skip next (the argument)
            continue

        if not pattern_found:
            pattern_found = True
            continue

        paths.append(arg)

    return paths if paths else (defaults or [])


# ============================================================================
# PATH_EXTRACTORS — path extraction function per command
# ============================================================================

PATH_EXTRACTORS: Dict[str, Callable[[List[str]], List[str]]] = {
    "cd": lambda args: [os.path.expanduser("~")] if len(args) == 0 or args[0] == "~" else [args[0]] if not args[0].startswith("-") else [],
    "ls": lambda args: filter_out_flags(args) or ["."],
    "find": lambda args: _extract_find_paths(args),
    "mkdir": lambda args: filter_out_flags(args),
    "touch": lambda args: filter_out_flags(args),
    "rm": lambda args: filter_out_flags(args),
    "rmdir": lambda args: filter_out_flags(args),
    "mv": lambda args: filter_out_flags(args),
    "cp": lambda args: filter_out_flags(args),
    "cat": lambda args: filter_out_flags(args),
    "head": lambda args: filter_out_flags(args),
    "tail": lambda args: filter_out_flags(args),
    "sort": lambda args: filter_out_flags(args),
    "uniq": lambda args: filter_out_flags(args),
    "wc": lambda args: filter_out_flags(args),
    "cut": lambda args: filter_out_flags(args),
    "paste": lambda args: filter_out_flags(args),
    "column": lambda args: filter_out_flags(args),
    "tr": lambda args: filter_out_flags(args),
    "file": lambda args: filter_out_flags(args),
    "stat": lambda args: filter_out_flags(args),
    "diff": lambda args: filter_out_flags(args),
    "strings": lambda args: filter_out_flags(args),
    "hexdump": lambda args: filter_out_flags(args),
    "od": lambda args: filter_out_flags(args),
    "base64": lambda args: filter_out_flags(args),
    "nl": lambda args: filter_out_flags(args),
    "grep": lambda args: parse_pattern_command(args, {"-e", "--regexp", "-f", "--file", "-D", "-d"}, ["."]),
    "rg": lambda args: parse_pattern_command(args, {"-e", "--regexp", "-f", "--file", "-g", "--glob", "-t", "--type", "-T", "--type-not"}, ["."]),
    "sed": lambda args: _extract_sed_paths(args),
    "jq": lambda args: filter_out_flags(args),
    "sha256sum": lambda args: filter_out_flags(args),
    "sha1sum": lambda args: filter_out_flags(args),
    "md5sum": lambda args: filter_out_flags(args),
    "awk": lambda args: _extract_awk_paths(args),
}


def _extract_find_paths(args: List[str]) -> List[str]:
    """Extract leading path arguments from find command."""
    paths: List[str] = []
    for arg in args:
        if arg in {"(", ")", "!", ","} or arg.startswith("-"):
            break
        paths.append(arg)
    if not paths:
        paths.append(".")
    return paths


def _extract_sed_paths(args: List[str]) -> List[str]:
    """Extract file arguments from sed command."""
    paths: List[str] = []
    found_e = False
    found_expr = False
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("-e", "--expression"):
            found_e = True
            i += 2  # Skip expression
            continue
        if arg.startswith("--expression=") or arg.startswith("-e="):
            found_e = True
            i += 1
            continue
        if arg == "-i" or arg == "--in-place":
            # -i optionally takes a backup suffix (next arg)
            if i + 1 < len(args) and not args[i + 1].startswith("-"):
                i += 2
            else:
                i += 1
            continue
        if arg.startswith("-"):
            i += 1
            continue
        if not found_e and not found_expr:
            found_expr = True  # First non-flag is expression
            i += 1
            continue
        # Remaining non-flag args are file paths
        paths.append(arg)
        i += 1
    return paths


def _extract_awk_paths(args: List[str]) -> List[str]:
    """Extract file arguments from awk command."""
    paths: List[str] = []
    found_program = False
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("-f", "--file"):
            if i + 1 < len(args):
                paths.append(args[i + 1])
                i += 2
            else:
                i += 1
            continue
        if arg == "-v" or arg == "--assign":
            i += 2
            continue
        if not found_program and not arg.startswith("-"):
            found_program = True  # This is the awk program
            i += 1
            continue
        if not arg.startswith("-"):
            paths.append(arg)
        i += 1
    return paths


# ============================================================================
# Path resolution and validation
# ============================================================================


def _expand_tilde(path_str: str) -> str:
    """Expand ~ to the user's home directory.

    Args:
        path_str: Path potentially starting with ~.

    Returns:
        Expanded path.
    """
    if path_str == "~" or path_str.startswith("~/"):
        return os.path.expanduser(path_str)
    return path_str


def resolve_path(raw_path: str, cwd: str) -> str:
    """Resolve a path, expanding tilde and making absolute.

    Args:
        raw_path: The raw path string.
        cwd: Current working directory.

    Returns:
        Resolved absolute path.
    """
    expanded = _expand_tilde(raw_path.strip("\"'"))
    candidate = Path(expanded)
    if not candidate.is_absolute():
        candidate = Path(cwd) / candidate
    try:
        return str(candidate.resolve())
    except (OSError, ValueError):
        return str(candidate.absolute())


def _is_path_allowed(resolved_path: str, allowed_dirs: Set[str]) -> bool:
    """Check if a resolved path is within allowed directories.

    Args:
        resolved_path: The resolved absolute path.
        allowed_dirs: Set of allowed directory paths.

    Returns:
        True if the path is allowed.
    """
    for allowed in allowed_dirs:
        normalized_allowed = os.path.abspath(allowed)
        normalized = os.path.abspath(resolved_path)
        if normalized == normalized_allowed:
            return True
        if normalized.startswith(normalized_allowed.rstrip(os.sep) + os.sep):
            return True
    return False


# ============================================================================
# Git path extractors (for git subcommands)
# ============================================================================


def _extract_git_read_paths(args: List[str]) -> List[str]:
    """Extract file paths from git read commands."""
    return filter_out_flags(args)


# ============================================================================
# Main entry point
# ============================================================================


def check_path_constraints(
    command: str,
    cwd: str,
    allowed_dirs: Optional[Set[str]] = None,
) -> PermissionResult:
    """Check path constraints for a bash command.

    Validates that all file paths referenced in the command are within
    allowed working directories.

    Args:
        command: The command string.
        cwd: Current working directory.
        allowed_dirs: Set of allowed directory paths.

    Returns:
        PermissionResult indicating whether the paths are valid.
    """
    if allowed_dirs is None:
        allowed_dirs = {cwd}

    # Handle cases where command is an object with .command attribute (backward compat)
    if hasattr(command, "command"):
        command = command.command  # type: ignore[union-attr]

    # Shell-quote parse the command, then extract base command and args
    parse_result = try_parse_shell_command(str(command))
    if not parse_result.success:
        return PermissionResult.passthrough("Cannot parse command for path validation")

    tokens = [str(t) for t in parse_result.tokens]
    if not tokens:
        return PermissionResult.passthrough("No tokens to validate")

    base_cmd = tokens[0]

    for match in re.finditer(r"(?:^|\s)(?:[012])?(?:>>?|>&)\s*(\S+)", str(command)):
        target = match.group(1)
        if target.startswith("&") or target in ("/dev/null", "/dev/stdout", "/dev/stderr"):
            continue
        if os.path.isabs(target):
            resolved = resolve_path(target, cwd)
            if not _is_path_allowed(resolved, allowed_dirs):
                return PermissionResult.ask(
                    f"Requesting permission to write to '{resolved}' (outside allowed directories)",
                    reason={"type": "sensitive_directory_access"},
                )

    # Extract paths using PATH_EXTRACTORS
    if base_cmd not in PATH_EXTRACTORS:
        # Try git subcommands
        if base_cmd == "git" and len(tokens) > 1:
            return _check_git_paths(tokens[1:], cwd, allowed_dirs)
        return PermissionResult.passthrough(f"Command '{base_cmd}' is not path-restricted")

    extractor = PATH_EXTRACTORS[base_cmd]
    paths = extractor(tokens[1:])

    # For write commands, check dangerous removal paths
    op_type = COMMAND_OPERATION_TYPE.get(base_cmd)
    if base_cmd in ("rm", "rmdir") and op_type == "write":
        for p in paths:
            clean_path = _expand_tilde(p.strip("\"'"))
            abs_path = os.path.abspath(clean_path) if not os.path.isabs(clean_path) else clean_path
            if _is_dangerous_removal_path(abs_path):
                return PermissionResult.ask(
                    f"Dangerous {base_cmd} operation detected: '{abs_path}'",
                    reason={"type": "other", "reason": f"Dangerous {base_cmd} on critical path"},
                )

    # Validate each path
    for p in paths:
        resolved = resolve_path(p, cwd)
        if not _is_path_allowed(resolved, allowed_dirs):
            dirs_preview = ", ".join(f"'{d}'" for d in sorted(allowed_dirs)[:5])
            action = "change directory to" if base_cmd == "cd" else "access"
            return PermissionResult.ask(
                f"Requesting permission to {action} '{resolved}' (outside allowed directories: {dirs_preview})",
                reason={"type": "sensitive_directory_access"},
            )

    # Check output redirections
    result = extract_output_redirections(command)
    for r in result.redirections:
        target = r.target
        if not target.startswith("/") or target in ("/dev/null", "/dev/stdout", "/dev/stderr"):
            continue
        resolved = resolve_path(target, cwd)
        if not _is_path_allowed(resolved, allowed_dirs):
            return PermissionResult.ask(
                f"Requesting permission to write to '{resolved}' (outside allowed directories)",
                reason={"type": "sensitive_directory_access"},
            )

    return PermissionResult.passthrough("Path validation passed")


def _check_git_paths(
    args: List[str],
    cwd: str,
    allowed_dirs: Set[str],
) -> PermissionResult:
    """Validate paths for git commands.

    Args:
        args: Arguments after 'git'.
        cwd: Current working directory.
        allowed_dirs: Set of allowed directories.

    Returns:
        PermissionResult.
    """
    if not args:
        return PermissionResult.passthrough("No git subcommand")

    subcmd = args[0]
    # Most read-only git operations on allowed paths are fine
    read_only_git_commands = {
        "status", "log", "show", "diff", "cat-file", "ls-files",
        "branch", "remote", "rev-parse", "tag", "blame", "reflog",
        "ls-remote", "config", "stash",
    }

    if subcmd in read_only_git_commands:
        return PermissionResult.passthrough("Git read-only command on allowed paths")

    # For write commands, validate file paths
    git_paths = _extract_git_read_paths(args)
    for p in git_paths:
        resolved = resolve_path(p, cwd)
        if not _is_path_allowed(resolved, allowed_dirs):
            return PermissionResult.ask(
                f"Requesting permission to access '{resolved}' (outside allowed directories)",
                reason={"type": "sensitive_directory_access"},
            )

    return PermissionResult.passthrough("Git path validation passed")


__all__ = [
    "check_path_constraints",
    "PATH_EXTRACTORS",
    "COMMAND_OPERATION_TYPE",
    "filter_out_flags",
    "resolve_path",
]
