"""Rules and regex patterns for permission checks."""

from __future__ import annotations

import re
from typing import List, Tuple

# =============================================================================
# Safe command patterns
# =============================================================================

_SAFE_COMMAND_PATTERNS: List[re.Pattern[str]] = [
    # Common version checks
    re.compile(r"^\s*(python|python3|node|npm|git|bash|sh)\s+--version\s*$", re.IGNORECASE),
    re.compile(r"^\s*(python|python3|node|npm|git|bash|sh)\s+-v\s*$", re.IGNORECASE),
    re.compile(r"^\s*(python|python3|node|npm|git|bash|sh)\s+-V\s*$", re.IGNORECASE),
    # Common help commands
    re.compile(r"^\s*\w+\s+--help\s*$", re.IGNORECASE),
    re.compile(r"^\s*\w+\s+-h\s*$", re.IGNORECASE),
    re.compile(r"^\s*\w+\s+help\s*$", re.IGNORECASE),
    # Simple echo/print commands
    re.compile(r'^\s*echo\s+["\'].*["\']\s*$', re.IGNORECASE),
    re.compile(r"^\s*print(env|f)?\s+.*$", re.IGNORECASE),
    # Directory listing with common options
    re.compile(r'^\s*ls\s+(-[a-zA-Z]*[lhtr]*\s*)*["\']?[^;&|<>]*["\']?\s*$', re.IGNORECASE),
    re.compile(r"^\s*dir\s+.*$", re.IGNORECASE),
    # Current directory
    re.compile(r"^\s*pwd\s*$", re.IGNORECASE),
    # Environment variable checks
    re.compile(r"^\s*env\s*$", re.IGNORECASE),
    re.compile(r"^\s*printenv\s*$", re.IGNORECASE),
    # Which/whereis commands
    re.compile(r"^\s*which\s+\w+\s*$", re.IGNORECASE),
    re.compile(r"^\s*whereis\s+\w+\s*$", re.IGNORECASE),
    # Type/command commands
    re.compile(r"^\s*type\s+\w+\s*$", re.IGNORECASE),
    re.compile(r"^\s*command\s+-v\s+\w+\s*$", re.IGNORECASE),
]


def is_safe_command_pattern(command: str) -> bool:
    """Check if command matches known safe patterns."""
    for pattern in _SAFE_COMMAND_PATTERNS:
        if pattern.match(command):
            return True
    return False


# =============================================================================
# Dangerous patterns to check in sanitized commands
# =============================================================================

DANGEROUS_PATTERNS: List[Tuple[re.Pattern[str], str]] = [
    # Newlines can break out of quotes
    (re.compile(r"[\n\r]"), "Command contains newlines which could break out of quotes"),
    # Process substitution
    (re.compile(r"<\("), "Command contains process substitution <()"),
    (re.compile(r">\("), "Command contains process substitution >()"),
    # Command substitution
    (re.compile(r"[`‵]"), "Command contains backticks (`) for command substitution"),
    (re.compile(r"\$\("), "Command contains $() command substitution"),
    # Parameter substitution
    (re.compile(r"\$\{"), "Command contains ${} parameter substitution"),
    # Input/output redirection
    (
        re.compile(r"<(?!\()"),
        "Command contains input redirection (<) which could read sensitive files",
    ),
    (
        re.compile(r">(?!\()"),
        "Command contains output redirection (>) which could write to arbitrary files",
    ),
    # Zsh-specific patterns
    (re.compile(r"~\["), "Command contains Zsh-style parameter expansion"),
    (re.compile(r"\(e:"), "Command contains Zsh-style glob qualifiers"),
]

# Patterns that indicate dangerous metacharacters in find/grep arguments
DANGEROUS_METACHARACTER_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r'-name\s+["\']?[^"\']*[;|&][^"\']*["\']?'),
    re.compile(r'-path\s+["\']?[^"\']*[;|&][^"\']*["\']?'),
    re.compile(r'-iname\s+["\']?[^"\']*[;|&][^"\']*["\']?'),
    re.compile(r'-regex\s+["\']?[^"\']*[;|&][^"\']*["\']?'),
]


__all__ = ["DANGEROUS_PATTERNS", "DANGEROUS_METACHARACTER_PATTERNS", "is_safe_command_pattern"]
