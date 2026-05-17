"""Detect potentially destructive bash commands and return human-readable warnings.

Purely informational — does not affect permission logic or auto-approval.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

_PATTERN_DEFS: List[Tuple[str, str]] = [
    # Git — data loss / hard to reverse
    (r"\bgit\s+reset\s+--hard\b", "Note: may discard uncommitted changes"),
    (r"\bgit\s+push\b[^;&|\n]*[ \t](--force|--force-with-lease|-f)\b", "Note: may overwrite remote history"),
    (r"\bgit\s+clean\b(?![^;&|\n]*(?:-[a-zA-Z]*n|--dry-run))[^;&|\n]*-[a-zA-Z]*f", "Note: may permanently delete untracked files"),
    (r"\bgit\s+checkout\s+(--\s+)?\.[ \t]*($|[;&|\n])", "Note: may discard all working tree changes"),
    (r"\bgit\s+restore\s+(--\s+)?\.[ \t]*($|[;&|\n])", "Note: may discard all working tree changes"),
    (r"\bgit\s+stash[ \t]+(drop|clear)\b", "Note: may permanently remove stashed changes"),
    (r"\bgit\s+branch\s+(-D[ \t]|--delete\s+--force|--force\s+--delete)\b", "Note: may force-delete a branch"),
    # Git — safety bypass
    (r"\bgit\s+(commit|push|merge)\b[^;&|\n]*--no-verify\b", "Note: may skip safety hooks"),
    (r"\bgit\s+commit\b[^;&|\n]*--amend\b", "Note: may rewrite the last commit"),
    # File deletion
    (r"(^|[;&|\n]\s*)rm\s+-[a-zA-Z]*[rR][a-zA-Z]*f|(^|[;&|\n]\s*)rm\s+-[a-zA-Z]*f[a-zA-Z]*[rR]", "Note: may recursively force-remove files"),
    (r"(^|[;&|\n]\s*)rm\s+-[a-zA-Z]*[rR]", "Note: may recursively remove files"),
    (r"(^|[;&|\n]\s*)rm\s+-[a-zA-Z]*f", "Note: may force-remove files"),
    # Database
    (r"\b(DROP|TRUNCATE)\s+(TABLE|DATABASE|SCHEMA)\b", "Note: may drop or truncate database objects"),
    (r"\bDELETE\s+FROM\s+\w+[ \t]*(;|\"|'|\n|$)", "Note: may delete all rows from a database table"),
    # Infrastructure
    (r"\bkubectl\s+delete\b", "Note: may delete Kubernetes resources"),
    (r"\bterraform\s+destroy\b", "Note: may destroy Terraform infrastructure"),
]

_COMPILED: List[Tuple[re.Pattern[str], str]] = [
    (re.compile(p, re.IGNORECASE), w) for p, w in _PATTERN_DEFS
]


def get_destructive_command_warning(command: str) -> Optional[str]:
    """Return a human-readable warning if *command* matches a known destructive pattern."""
    for pattern, warning in _COMPILED:
        if pattern.search(command):
            return warning
    return None
