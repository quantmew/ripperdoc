"""Destructive command detection helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ripperdoc.utils.permissions.interpreter import extract_code_string, is_interpreter_command
from ripperdoc.utils.permissions.parse import strip_quoted_content_for_destructive_check

# =============================================================================
# Destructive Command Detection (Cross-platform)
# =============================================================================
# These patterns detect commands that can cause irreversible data loss.
# Inspired by the Gemini incident where `cmd /c "rmdir /s /q ..."` deleted C:\

# Windows destructive commands
_WINDOWS_DESTRUCTIVE_PATTERNS: List[Tuple[re.Pattern[str], str]] = [
    # rmdir /s - Recursive directory deletion (Windows)
    (
        re.compile(r"\brmdir\s+.*(/s|/S)", re.IGNORECASE),
        "Command contains 'rmdir /s' which recursively deletes directories",
    ),
    # del /s or del /q - Recursive or quiet file deletion (Windows)
    (
        re.compile(r"\bdel\s+.*(/s|/S|/q|/Q)", re.IGNORECASE),
        "Command contains 'del' with dangerous flags (/s or /q)",
    ),
    # rd /s - Alias for rmdir /s (Windows)
    (
        re.compile(r"\brd\s+.*(/s|/S)", re.IGNORECASE),
        "Command contains 'rd /s' which recursively deletes directories",
    ),
    # format command (Windows)
    (
        re.compile(r"\bformat\s+[a-zA-Z]:", re.IGNORECASE),
        "Command contains 'format' which erases entire drives",
    ),
    # cmd /c with destructive subcommand
    (
        re.compile(r"\bcmd\s+/[cC]\s+.*\b(rmdir|rd|del|format)\b", re.IGNORECASE),
        "Command uses 'cmd /c' to execute a destructive subcommand",
    ),
    # PowerShell Remove-Item -Recurse
    (
        re.compile(r"\b(Remove-Item|rm|ri|del)\s+.*-Recurse", re.IGNORECASE),
        "Command contains 'Remove-Item -Recurse' which recursively deletes items",
    ),
    # PowerShell with -Force flag on destructive commands
    (
        re.compile(r"\b(Remove-Item|rm|ri|del)\s+.*-Force", re.IGNORECASE),
        "Command contains destructive command with -Force flag",
    ),
]

# Unix/Linux destructive commands
_UNIX_DESTRUCTIVE_PATTERNS: List[Tuple[re.Pattern[str], str]] = [
    # rm -rf or rm -r (recursive deletion) - must be at word boundary and followed by space/path
    (
        re.compile(r'(?<!["\'])\brm\s+(-[a-zA-Z]*r[a-zA-Z]*\s+|\s*-[a-zA-Z]*r[a-zA-Z]*$)'),
        "Command contains 'rm -r' which recursively deletes files and directories",
    ),
    # rm with force flag on system paths
    (
        re.compile(
            r'(?<!["\'])\brm\s+-[a-zA-Z]*f[a-zA-Z]*\s+(/|~|/home|/usr|/var|/etc|/root|\$HOME)'
        ),
        "Command contains 'rm -f' targeting a critical system path",
    ),
    # dd command (can overwrite disks)
    (re.compile(r"\bdd\s+.*of=/dev/"), "Command contains 'dd' writing to a device file"),
    # mkfs (creates filesystem, destroys data)
    (re.compile(r"\bmkfs\b"), "Command contains 'mkfs' which formats storage devices"),
    # shred (secure deletion)
    (re.compile(r"\bshred\s+"), "Command contains 'shred' which irreversibly destroys file data"),
    # chmod 777 on sensitive paths
    (
        re.compile(r"\bchmod\s+777\s+(/|/etc|/usr|/var|/home)"),
        "Command contains 'chmod 777' on a sensitive system path",
    ),
    # chown on system paths
    (
        re.compile(r"\bchown\s+.*\s+(/etc|/usr|/var|/bin|/sbin)"),
        "Command contains 'chown' on a critical system path",
    ),
]

# Path patterns that should trigger extra scrutiny
_CRITICAL_PATH_PATTERNS: List[re.Pattern[str]] = [
    # Windows critical paths
    re.compile(r'["\']?[A-Za-z]:\\(?:Windows|Program Files|Users)\\?["\']?', re.IGNORECASE),
    re.compile(r'["\']?[A-Za-z]:\\["\']?(?:\s|$)', re.IGNORECASE),  # Root of any drive
    # Unix critical paths
    re.compile(r'["\']?/(?:etc|usr|var|bin|sbin|lib|boot|root|home)["\']?(?:\s|$)'),
    re.compile(r'["\']?/["\']?(?:\s|$)'),  # Root directory
    re.compile(r'["\']?~["\']?(?:\s|$)'),  # Home directory
]

# Commands with nested/escaped quotes that might cause parsing issues
_NESTED_QUOTE_PATTERNS: List[Tuple[re.Pattern[str], str]] = [
    # Windows cmd with escaped quotes inside
    (
        re.compile(r'\bcmd\s+/[cC]\s+"[^"]*\\"[^"]*"'),
        "Command contains 'cmd /c' with nested escaped quotes which may cause unexpected parsing",
    ),
    # PowerShell with complex quoting
    (
        re.compile(r'\bpowershell\s+.*-[Cc]ommand\s+["\'][^"\']*["\'][^"\']*["\']'),
        "Command contains PowerShell with complex nested quotes",
    ),
]


@dataclass
class DestructiveCheckResult:
    behavior: str
    message: str
    rule_suggestions: Optional[list[str]] = None


def _check_destructive_commands_in_code_string(
    code_string: str, interpreter: str
) -> Optional[DestructiveCheckResult]:
    """Check for destructive commands in interpreter code strings."""
    # For shell interpreters (bash, sh, zsh), the code string is shell code
    if interpreter in ("bash", "sh", "zsh"):
        stripped_code = strip_quoted_content_for_destructive_check(code_string)

        for pattern, message in _UNIX_DESTRUCTIVE_PATTERNS:
            if pattern.search(stripped_code):
                return DestructiveCheckResult(
                    behavior="ask",
                    message=f"Code string contains {message}",
                    rule_suggestions=None,
                )

    # For Python, check for os.system, subprocess, etc.
    elif interpreter in ("python", "python3"):
        system_patterns = [
            (
                r'\bos\.system\s*\(\s*["\'][^"\']*rm\s+-[a-zA-Z]*r',
                "Python code executes destructive shell command",
            ),
            (
                r"\bsubprocess\.(run|call|Popen)\s*\(\s*[^)]*rm\s+-[a-zA-Z]*r",
                "Python code executes destructive shell command",
            ),
        ]

        for pattern_str, message in system_patterns:
            if re.search(pattern_str, code_string):
                return DestructiveCheckResult(
                    behavior="ask",
                    message=message,
                    rule_suggestions=None,
                )

    return None


def check_destructive_commands(command: str) -> Optional[DestructiveCheckResult]:
    """Check for destructive commands that could cause irreversible data loss."""
    # Check for nested quote issues first (like the Gemini incident)
    for pattern, message in _NESTED_QUOTE_PATTERNS:
        if pattern.search(command):
            has_critical_path = any(p.search(command) for p in _CRITICAL_PATH_PATTERNS)
            if has_critical_path:
                return DestructiveCheckResult(
                    behavior="deny",
                    message=f"BLOCKED: {message} targeting a critical system path",
                    rule_suggestions=None,
                )
            return DestructiveCheckResult(
                behavior="ask",
                message=message,
                rule_suggestions=None,
            )

    # Check if command targets critical paths with any destructive operation
    has_critical_path = any(p.search(command) for p in _CRITICAL_PATH_PATTERNS)

    trimmed = command.strip()
    first_token = trimmed.split()[0] if trimmed.split() else ""

    if is_interpreter_command(command, first_token):
        code_string = extract_code_string(command, first_token)
        if code_string:
            code_check_result = _check_destructive_commands_in_code_string(code_string, first_token)
            if code_check_result:
                if has_critical_path:
                    return DestructiveCheckResult(
                        behavior="deny",
                        message=f"BLOCKED: {code_check_result.message} targeting a critical system path",
                        rule_suggestions=None,
                    )
                return code_check_result

    # Strip quoted content to avoid false positives like 'echo "rmdir /s /q folder"'
    command_without_quotes = strip_quoted_content_for_destructive_check(command)

    for pattern, message in _WINDOWS_DESTRUCTIVE_PATTERNS:
        if pattern.search(command_without_quotes):
            if has_critical_path:
                return DestructiveCheckResult(
                    behavior="deny",
                    message=f"BLOCKED: {message} targeting a critical system path",
                    rule_suggestions=None,
                )
            return DestructiveCheckResult(
                behavior="ask",
                message=message,
                rule_suggestions=None,
            )

    for pattern, message in _UNIX_DESTRUCTIVE_PATTERNS:
        if pattern.search(command_without_quotes):
            if has_critical_path:
                return DestructiveCheckResult(
                    behavior="deny",
                    message=f"BLOCKED: {message} targeting a critical system path",
                    rule_suggestions=None,
                )
            return DestructiveCheckResult(
                behavior="ask",
                message=message,
                rule_suggestions=None,
            )

    # Special check for the exact Gemini incident pattern
    if re.search(r'\bcmd\s+/[cC]\s+"[^"]*\\[""][^"]*"', command):
        if has_critical_path:
            return DestructiveCheckResult(
                behavior="deny",
                message=(
                    "BLOCKED: Command contains 'cmd /c' with escaped quotes - "
                    "this pattern has caused data loss incidents"
                ),
                rule_suggestions=None,
            )
        return DestructiveCheckResult(
            behavior="ask",
            message=(
                "Command contains 'cmd /c' with escaped quotes inside double quotes - "
                "this pattern has caused data loss incidents due to quote parsing issues"
            ),
            rule_suggestions=None,
        )

    return None


__all__ = ["DestructiveCheckResult", "check_destructive_commands"]
