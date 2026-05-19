"""Shared read-only command validation maps.


Provides complete command configuration maps with per-flag safety validation
for Git, GitHub CLI, Docker, ripgrep, pyright, and cross-shell commands.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

FLAG_ARG_NONE = "none"
FLAG_ARG_NUMBER = "number"
FLAG_ARG_STRING = "string"
FLAG_ARG_CHAR = "char"
FLAG_ARG_BRACES = "{}"
FLAG_ARG_EOF = "EOF"

FlagArgType = str


@dataclass
class CommandConfig:
    """Configuration for a command's flag allowlist."""
    safe_flags: Dict[str, str] = field(default_factory=dict)
    additional_command_is_dangerous_callback: Optional[Callable[[str, List[str]], bool]] = None
    respects_double_dash: bool = True
    regex: Optional[re.Pattern] = None


def contains_vulnerable_unc_path(command: str) -> bool:
    """Check for Windows UNC paths that could be vulnerable to WebDAV attacks."""
    return bool(re.search(r"\\\\[a-zA-Z0-9._-]+\\", command))


# ============================================================================
# Git read-only commands — comprehensive flag allowlists
# ============================================================================

GIT_REF_SELECTION_FLAGS: Dict[str, str] = {
    "--all": FLAG_ARG_NONE, "--branches": FLAG_ARG_NONE,
    "--tags": FLAG_ARG_NONE, "--remotes": FLAG_ARG_NONE,
}

GIT_DATE_FILTER_FLAGS: Dict[str, str] = {
    "--since": FLAG_ARG_STRING, "--after": FLAG_ARG_STRING,
    "--until": FLAG_ARG_STRING, "--before": FLAG_ARG_STRING,
}

GIT_LOG_DISPLAY_FLAGS: Dict[str, str] = {
    "--oneline": FLAG_ARG_NONE, "--graph": FLAG_ARG_NONE,
    "--decorate": FLAG_ARG_NONE, "--no-decorate": FLAG_ARG_NONE,
    "--date": FLAG_ARG_STRING, "--relative-date": FLAG_ARG_NONE,
}

GIT_COUNT_FLAGS: Dict[str, str] = {
    "--max-count": FLAG_ARG_NUMBER, "-n": FLAG_ARG_NUMBER,
}

GIT_STAT_FLAGS: Dict[str, str] = {
    "--stat": FLAG_ARG_NONE, "--numstat": FLAG_ARG_NONE,
    "--shortstat": FLAG_ARG_NONE,
    "--name-only": FLAG_ARG_NONE, "--name-status": FLAG_ARG_NONE,
}

GIT_COLOR_FLAGS: Dict[str, str] = {
    "--color": FLAG_ARG_NONE, "--no-color": FLAG_ARG_NONE,
}

GIT_PATCH_FLAGS: Dict[str, str] = {
    "--patch": FLAG_ARG_NONE, "-p": FLAG_ARG_NONE,
    "--no-patch": FLAG_ARG_NONE, "--no-ext-diff": FLAG_ARG_NONE, "-s": FLAG_ARG_NONE,
}

GIT_AUTHOR_FILTER_FLAGS: Dict[str, str] = {
    "--author": FLAG_ARG_STRING, "--committer": FLAG_ARG_STRING, "--grep": FLAG_ARG_STRING,
}

GIT_DIFF_SELECTION_FLAGS: Dict[str, str] = {
    "--cached": FLAG_ARG_NONE, "--staged": FLAG_ARG_NONE, "--merge-base": FLAG_ARG_NONE, "-R": FLAG_ARG_NONE,
}

GIT_DIFF_ALGORITHM_FLAGS: Dict[str, str] = {
    "--diff-algorithm": FLAG_ARG_STRING, "--minimal": FLAG_ARG_NONE, "--histogram": FLAG_ARG_NONE,
    "--patience": FLAG_ARG_NONE, "--anchored": FLAG_ARG_STRING,
    "--no-renames": FLAG_ARG_NONE, "--rename-threshold": FLAG_ARG_NUMBER,
    "--find-renames": FLAG_ARG_NUMBER, "--break-rewrites": FLAG_ARG_NUMBER,
    "--find-copies": FLAG_ARG_NUMBER,
    "--ignore-space-change": FLAG_ARG_NONE, "--ignore-all-space": FLAG_ARG_NONE,
    "--ignore-blank-lines": FLAG_ARG_NONE, "--function-context": FLAG_ARG_NONE,
    "--binary": FLAG_ARG_NONE, "--full-index": FLAG_ARG_NONE, "--abbrev": FLAG_ARG_NUMBER,
    "--src-prefix": FLAG_ARG_STRING, "--dst-prefix": FLAG_ARG_STRING, "--no-prefix": FLAG_ARG_NONE,
    "--line-prefix": FLAG_ARG_STRING, "--inter-hunk-context": FLAG_ARG_NUMBER,
    "--output-indicator-new": FLAG_ARG_CHAR, "--output-indicator-old": FLAG_ARG_CHAR,
    "--output-indicator-context": FLAG_ARG_CHAR,
    "--ws-error-highlight": FLAG_ARG_STRING, "--submodule": FLAG_ARG_STRING,
    "-z": FLAG_ARG_NONE, "-U": FLAG_ARG_NUMBER,
}

GIT_LOG_FORMAT_FLAGS: Dict[str, str] = {
    "--format": FLAG_ARG_STRING, "--pretty": FLAG_ARG_STRING,
    "--abbrev-commit": FLAG_ARG_NONE, "--no-abbrev-commit": FLAG_ARG_NONE,
    "--no-notes": FLAG_ARG_NONE, "--show-notes": FLAG_ARG_STRING,
    "--raw": FLAG_ARG_NONE, "--relative": FLAG_ARG_NONE,
    "--left-right": FLAG_ARG_NONE, "--first-parent": FLAG_ARG_NONE,
    "--no-walk": FLAG_ARG_NONE, "--parents": FLAG_ARG_NONE,
    "--children": FLAG_ARG_NONE, "--sparse": FLAG_ARG_NONE,
    "--simplify-merges": FLAG_ARG_NONE, "--ancestry-path": FLAG_ARG_NONE,
    "--simplify-by-decoration": FLAG_ARG_NONE,
    "--reflog": FLAG_ARG_NONE,
    "--regexp-ignore-case": FLAG_ARG_NONE, "--extended-regexp": FLAG_ARG_NONE,
    "--fixed-strings": FLAG_ARG_NONE, "--perl-regexp": FLAG_ARG_NONE,
    "--remove-empty": FLAG_ARG_NONE,
    "--all-match": FLAG_ARG_NONE, "--invert-grep": FLAG_ARG_NONE,
    "-i": FLAG_ARG_NONE, "-L": FLAG_ARG_STRING,
}

GIT_LOG_FLAGS: Dict[str, str] = {}
for d in [GIT_REF_SELECTION_FLAGS, GIT_DATE_FILTER_FLAGS, GIT_LOG_DISPLAY_FLAGS,
          GIT_COUNT_FLAGS, GIT_STAT_FLAGS, GIT_COLOR_FLAGS, GIT_PATCH_FLAGS,
          GIT_AUTHOR_FILTER_FLAGS, GIT_LOG_FORMAT_FLAGS]:
    GIT_LOG_FLAGS.update(d)

GIT_DIFF_FLAGS: Dict[str, str] = {}
for d in [GIT_DIFF_SELECTION_FLAGS, GIT_DIFF_ALGORITHM_FLAGS, GIT_STAT_FLAGS, GIT_COLOR_FLAGS]:
    GIT_DIFF_FLAGS.update(d)

GIT_SHOW_FLAGS: Dict[str, str] = {}
for d in [GIT_DIFF_FLAGS, GIT_LOG_FORMAT_FLAGS, GIT_PATCH_FLAGS]:
    GIT_SHOW_FLAGS.update(d)

GIT_CAT_FILE_FLAGS: Dict[str, str] = {
    "-p": FLAG_ARG_NONE, "-t": FLAG_ARG_NONE, "-s": FLAG_ARG_NONE,
    "--batch": FLAG_ARG_NONE, "--batch-check": FLAG_ARG_NONE,
}

GIT_BRANCH_FLAGS: Dict[str, str] = {
    "-a": FLAG_ARG_NONE, "--all": FLAG_ARG_NONE, "-r": FLAG_ARG_NONE, "--remotes": FLAG_ARG_NONE,
    "--list": FLAG_ARG_NONE, "--no-color": FLAG_ARG_NONE, "--sort": FLAG_ARG_STRING,
    "--merged": FLAG_ARG_STRING, "--no-merged": FLAG_ARG_STRING,
    "--contains": FLAG_ARG_STRING, "--no-contains": FLAG_ARG_STRING,
    "--format": FLAG_ARG_STRING, "-v": FLAG_ARG_NONE, "--verbose": FLAG_ARG_NONE,
    "-q": FLAG_ARG_NONE, "--quiet": FLAG_ARG_NONE,
}

GIT_REMOTE_FLAGS: Dict[str, str] = {"-v": FLAG_ARG_NONE, "--verbose": FLAG_ARG_NONE}

GIT_REV_PARSE_FLAGS: Dict[str, str] = {
    "--abbrev-ref": FLAG_ARG_NONE, "--symbolic-full-name": FLAG_ARG_NONE,
    "--verify": FLAG_ARG_NONE, "-q": FLAG_ARG_NONE, "--quiet": FLAG_ARG_NONE,
    "--short": FLAG_ARG_NONE, "--show-toplevel": FLAG_ARG_NONE,
    "--is-inside-work-tree": FLAG_ARG_NONE, "--is-inside-git-dir": FLAG_ARG_NONE,
    "--is-bare-repository": FLAG_ARG_NONE,
    "--all": FLAG_ARG_NONE, "--heads": FLAG_ARG_NONE, "--tags": FLAG_ARG_NONE,
    "--glob": FLAG_ARG_STRING, "--symbolic": FLAG_ARG_NONE,
}

GIT_LS_REMOTE_FLAGS: Dict[str, str] = {
    "-h": FLAG_ARG_NONE, "--heads": FLAG_ARG_NONE,
    "-t": FLAG_ARG_NONE, "--tags": FLAG_ARG_NONE,
    "--refs": FLAG_ARG_NONE, "--get-url": FLAG_ARG_NONE,
    "--sort": FLAG_ARG_STRING, "-q": FLAG_ARG_NONE, "--quiet": FLAG_ARG_NONE,
}

GIT_BLAME_FLAGS: Dict[str, str] = {
    "-s": FLAG_ARG_NONE, "-L": FLAG_ARG_STRING,
    "-e": FLAG_ARG_NONE, "--show-email": FLAG_ARG_NONE,
    "-w": FLAG_ARG_NONE, "--ignore-whitespace": FLAG_ARG_NONE,
    "--root": FLAG_ARG_NONE, "--show-stats": FLAG_ARG_NONE,
    "--reverse": FLAG_ARG_NONE, "--date": FLAG_ARG_STRING,
    "--porcelain": FLAG_ARG_NONE,
}

GIT_REFLOG_FLAGS: Dict[str, str] = {
    "--all": FLAG_ARG_NONE,
}
GIT_REFLOG_FLAGS.update(GIT_COUNT_FLAGS)
GIT_REFLOG_FLAGS.update(GIT_DATE_FILTER_FLAGS)
GIT_REFLOG_FLAGS.update(GIT_AUTHOR_FILTER_FLAGS)

GIT_READ_ONLY_COMMANDS: Dict[str, CommandConfig] = {
    "git status": CommandConfig(safe_flags={
        "-s": FLAG_ARG_NONE, "--short": FLAG_ARG_NONE,
        "-b": FLAG_ARG_NONE, "--branch": FLAG_ARG_NONE,
        "--porcelain": FLAG_ARG_NONE, "--long": FLAG_ARG_NONE,
        "-v": FLAG_ARG_NONE, "--verbose": FLAG_ARG_NONE,
        "-u": FLAG_ARG_NONE, "--untracked-files": FLAG_ARG_STRING,
        "--ignore-submodules": FLAG_ARG_STRING,
        "-z": FLAG_ARG_NONE,
    }),
    "git log": CommandConfig(safe_flags=GIT_LOG_FLAGS),
    "git show": CommandConfig(safe_flags=GIT_SHOW_FLAGS),
    "git diff": CommandConfig(safe_flags=GIT_DIFF_FLAGS),
    "git cat-file": CommandConfig(safe_flags=GIT_CAT_FILE_FLAGS),
    "git branch": CommandConfig(safe_flags=GIT_BRANCH_FLAGS),
    "git remote": CommandConfig(safe_flags=GIT_REMOTE_FLAGS),
    "git rev-parse": CommandConfig(safe_flags=GIT_REV_PARSE_FLAGS),
    "git ls-remote": CommandConfig(safe_flags=GIT_LS_REMOTE_FLAGS),
    "git blame": CommandConfig(safe_flags=GIT_BLAME_FLAGS),
    "git reflog": CommandConfig(safe_flags=GIT_REFLOG_FLAGS),
}

# ============================================================================
# ripgrep
# ============================================================================

RIPGREP_SAFE_FLAGS: Dict[str, str] = {
    "-e": FLAG_ARG_STRING, "--regexp": FLAG_ARG_STRING,
    "-f": FLAG_ARG_STRING, "--file": FLAG_ARG_STRING,
    "-F": FLAG_ARG_NONE, "--fixed-strings": FLAG_ARG_NONE,
    "-g": FLAG_ARG_STRING, "--glob": FLAG_ARG_STRING,
    "-i": FLAG_ARG_NONE, "--ignore-case": FLAG_ARG_NONE,
    "-S": FLAG_ARG_NONE, "--smart-case": FLAG_ARG_NONE,
    "-s": FLAG_ARG_NONE, "--case-sensitive": FLAG_ARG_NONE,
    "-c": FLAG_ARG_NONE, "--count": FLAG_ARG_NONE,
    "-C": FLAG_ARG_NUMBER, "--context": FLAG_ARG_NUMBER,
    "-B": FLAG_ARG_NUMBER, "--before-context": FLAG_ARG_NUMBER,
    "-A": FLAG_ARG_NUMBER, "--after-context": FLAG_ARG_NUMBER,
    "--color": FLAG_ARG_STRING,
    "-E": FLAG_ARG_STRING, "--encoding": FLAG_ARG_STRING,
    "-l": FLAG_ARG_NONE, "--files-with-matches": FLAG_ARG_NONE,
    "-L": FLAG_ARG_NONE, "--follow": FLAG_ARG_NONE,
    "-H": FLAG_ARG_NONE, "--heading": FLAG_ARG_NONE,
    "--hidden": FLAG_ARG_NONE,
    "-n": FLAG_ARG_NONE, "--line-number": FLAG_ARG_NONE,
    "-N": FLAG_ARG_NONE, "--no-line-number": FLAG_ARG_NONE,
    "-m": FLAG_ARG_NUMBER, "--max-count": FLAG_ARG_NUMBER,
    "--max-depth": FLAG_ARG_NUMBER,
    "--max-filesize": FLAG_ARG_STRING,
    "-o": FLAG_ARG_NONE, "--only-matching": FLAG_ARG_NONE,
    "-q": FLAG_ARG_NONE, "--quiet": FLAG_ARG_NONE,
    "-r": FLAG_ARG_STRING, "--replace": FLAG_ARG_STRING,
    "-t": FLAG_ARG_STRING, "--type": FLAG_ARG_STRING,
    "-T": FLAG_ARG_STRING, "--type-not": FLAG_ARG_STRING,
    "-u": FLAG_ARG_NONE, "--unrestricted": FLAG_ARG_NONE,
    "-v": FLAG_ARG_NONE, "--invert-match": FLAG_ARG_NONE,
    "-V": FLAG_ARG_NONE, "--version": FLAG_ARG_NONE,
    "-w": FLAG_ARG_NONE, "--word-regexp": FLAG_ARG_NONE,
    "-x": FLAG_ARG_NONE, "--line-regexp": FLAG_ARG_NONE,
    "-z": FLAG_ARG_NONE, "--search-zip": FLAG_ARG_NONE,
    "--multiline": FLAG_ARG_NONE, "--multiline-dotall": FLAG_ARG_NONE,
    "--passthru": FLAG_ARG_NONE,
    "-P": FLAG_ARG_NONE, "--pcre2": FLAG_ARG_NONE,
    "--no-ignore": FLAG_ARG_NONE,
    "--no-ignore-vcs": FLAG_ARG_NONE,
    "--no-ignore-parent": FLAG_ARG_NONE,
    "--no-ignore-global": FLAG_ARG_NONE,
    "--sort": FLAG_ARG_STRING, "--sortr": FLAG_ARG_STRING,
    "--stats": FLAG_ARG_NONE,
    "-j": FLAG_ARG_NUMBER, "--threads": FLAG_ARG_NUMBER,
    "-a": FLAG_ARG_NONE, "--text": FLAG_ARG_NONE,
    "--files": FLAG_ARG_NONE,
    "--null": FLAG_ARG_NONE,
}

RIPGREP_READ_ONLY_COMMANDS: Dict[str, CommandConfig] = {
    "rg": CommandConfig(safe_flags=RIPGREP_SAFE_FLAGS),
    "ripgrep": CommandConfig(safe_flags=RIPGREP_SAFE_FLAGS),
}

# ============================================================================
# pyright
# ============================================================================

PYRIGHT_SAFE_FLAGS: Dict[str, str] = {
    "--project": FLAG_ARG_STRING, "--pythonversion": FLAG_ARG_STRING,
    "--typeshed-path": FLAG_ARG_STRING, "--venv-path": FLAG_ARG_STRING,
    "--verbose": FLAG_ARG_NONE, "--warnings": FLAG_ARG_NONE,
    "--level": FLAG_ARG_STRING, "--outputjson": FLAG_ARG_NONE,
    "--skipunmodified": FLAG_ARG_NONE, "--ignoreexternal": FLAG_ARG_NONE,
    "-p": FLAG_ARG_STRING, "--lib": FLAG_ARG_NONE,
}

PYRIGHT_READ_ONLY_COMMANDS: Dict[str, CommandConfig] = {
    "pyright": CommandConfig(safe_flags=PYRIGHT_SAFE_FLAGS),
}

# ============================================================================
# Docker read-only commands
# ============================================================================

DOCKER_READ_ONLY_COMMANDS: Dict[str, CommandConfig] = {
    "docker ps": CommandConfig(safe_flags={
        "-a": FLAG_ARG_NONE, "--all": FLAG_ARG_NONE,
        "-f": FLAG_ARG_STRING, "--filter": FLAG_ARG_STRING,
        "--format": FLAG_ARG_STRING, "-q": FLAG_ARG_NONE, "--quiet": FLAG_ARG_NONE,
        "-s": FLAG_ARG_NONE, "--size": FLAG_ARG_NONE,
        "-l": FLAG_ARG_NONE, "--latest": FLAG_ARG_NONE,
        "-n": FLAG_ARG_NUMBER, "--last": FLAG_ARG_NUMBER,
    }),
    "docker images": CommandConfig(safe_flags={
        "-a": FLAG_ARG_NONE, "--all": FLAG_ARG_NONE,
        "-f": FLAG_ARG_STRING, "--filter": FLAG_ARG_STRING,
        "--format": FLAG_ARG_STRING, "-q": FLAG_ARG_NONE, "--quiet": FLAG_ARG_NONE,
    }),
    "docker inspect": CommandConfig(safe_flags={
        "-f": FLAG_ARG_STRING, "--format": FLAG_ARG_STRING,
        "-s": FLAG_ARG_NONE, "--size": FLAG_ARG_NONE,
    }),
    "docker info": CommandConfig(safe_flags={
        "-f": FLAG_ARG_STRING, "--format": FLAG_ARG_STRING,
    }),
    "docker version": CommandConfig(safe_flags={
        "-f": FLAG_ARG_STRING, "--format": FLAG_ARG_STRING,
    }),
    "docker network ls": CommandConfig(safe_flags={
        "-f": FLAG_ARG_STRING, "--filter": FLAG_ARG_STRING,
        "--format": FLAG_ARG_STRING, "-q": FLAG_ARG_NONE, "--quiet": FLAG_ARG_NONE,
    }),
    "docker volume ls": CommandConfig(safe_flags={
        "-f": FLAG_ARG_STRING, "--filter": FLAG_ARG_STRING,
        "--format": FLAG_ARG_STRING, "-q": FLAG_ARG_NONE, "--quiet": FLAG_ARG_NONE,
    }),
}

# ============================================================================
# Cross-shell read-only command names
# ============================================================================

EXTERNAL_READONLY_COMMANDS: List[str] = [
    "cat", "head", "tail", "wc",
    "basename", "dirname", "realpath",
    "uname", "whoami", "id",
    "echo", "printf", "sleep", "which",
    "diff", "cmp", "comm",
    "grep", "sort", "uniq", "cut", "tr",
]


# ============================================================================
# Flag validation
# ============================================================================


def validate_flags(
    tokens: List[str],
    command_start_idx: int,
    config: CommandConfig,
    *,
    command_name: str = "",
    raw_command: str = "",
    xargs_target_commands: Optional[List[str]] = None,
) -> bool:
    """Validate flags against a command's allowlist configuration.

    Handles combined flags, argument consumption, POSIX -- handling.

    Args:
        tokens: Full tokenized command.
        command_start_idx: Index where tokens after command name start.
        config: The command's allowlist configuration.
        command_name: Name of the command (for diagnostics).
        raw_command: The original command string.
        xargs_target_commands: Safe target commands for xargs.

    Returns:
        True if all flags are valid.
    """
    respects_dd = config.respects_double_dash
    safe = config.safe_flags

    i = command_start_idx
    after_double_dash = False
    target_found = False

    while i < len(tokens):
        token = tokens[i]

        if target_found:
            i += 1
            continue

        if not after_double_dash and token == "--":
            after_double_dash = True
            if not respects_dd:
                after_double_dash = False
            i += 1
            continue

        if after_double_dash or not isinstance(token, str) or not token.startswith("-"):
            if xargs_target_commands and token in xargs_target_commands:
                target_found = True
            i += 1
            continue

        # Handle negative numbers like -5 (shorthand for --max-count=5 in git log -5)
        # These look like flags but are positional numeric arguments
        if re.match(r"^-\d+$", token):
            # Treat as positional arg, skip flag validation
            if xargs_target_commands and token in xargs_target_commands:
                target_found = True
            i += 1
            continue

        # Handle --flag=value
        if token.startswith("--") and "=" in token:
            flag = token.split("=", 1)[0]
            expected = safe.get(flag)
            if expected is None:
                return False
            if expected == FLAG_ARG_NONE:
                return False
            value = token[len(flag) + 1:]
            if expected == FLAG_ARG_NUMBER and not re.match(r"^-?\d+$", value):
                return False
            if expected == FLAG_ARG_BRACES and value != "{}":
                return False
            if expected == FLAG_ARG_EOF and value != "EOF":
                return False
            i += 1
            continue

        # Handle combined short flags: -nE
        if token.startswith("-") and not token.startswith("--") and len(token) > 2:
            all_valid = True
            for ch in token[1:]:
                sf = f"-{ch}"
                if sf not in safe or safe[sf] != FLAG_ARG_NONE:
                    all_valid = False
                    break
            if all_valid:
                i += 1
                continue

        # Single flag
        expected = safe.get(token)
        if expected is None:
            return False

        if expected != FLAG_ARG_NONE and i + 1 < len(tokens):
            value = tokens[i + 1]
            if not isinstance(value, str):
                return False
            if expected == FLAG_ARG_NUMBER and not re.match(r"^-?\d+$", value):
                return False
            if expected == FLAG_ARG_BRACES and value != "{}":
                return False
            if expected == FLAG_ARG_EOF and value != "EOF":
                return False
            i += 2
        else:
            i += 1

    return True


__all__ = [
    "CommandConfig",
    "FLAG_ARG_NONE", "FLAG_ARG_NUMBER", "FLAG_ARG_STRING",
    "FLAG_ARG_CHAR", "FLAG_ARG_BRACES", "FLAG_ARG_EOF",
    "contains_vulnerable_unc_path",
    "validate_flags",
    "GIT_READ_ONLY_COMMANDS",
    "DOCKER_READ_ONLY_COMMANDS",
    "RIPGREP_READ_ONLY_COMMANDS",
    "PYRIGHT_READ_ONLY_COMMANDS",
    "EXTERNAL_READONLY_COMMANDS",
]
