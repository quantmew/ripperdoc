"""Shell permission rule matching utilities.


Provides rule parsing, wildcard pattern matching, and suggestion generation
for bash permission rules.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import List, Optional



# ============================================================================
# Permission Rule types
# ============================================================================


@dataclass
class ShellPermissionRule:
    """A parsed shell permission rule."""
    type: str  # 'exact' | 'prefix' | 'wildcard'
    command: str = ""
    prefix: str = ""
    pattern: str = ""


@dataclass
class PermissionUpdate:
    """A permission update suggestion."""
    tool_name: str
    rule_content: str
    rule_type: str = "allow"


def parse_permission_rule(rule: str) -> ShellPermissionRule:
    """Parse a permission rule string into a structured object.

    Supports:
    - Exact match: 'git status'
    - Prefix match: 'npm:*' or 'npm:*' 
    - Wildcard: 'git *', 'npm install *'

    Args:
        rule: The rule string.

    Returns:
        ShellPermissionRule with type and value.
    """
    rule = rule.strip()
    if not rule:
        return ShellPermissionRule(type="exact", command="")

    # Legacy prefix format: "npm:*" means "npm *"
    if rule.endswith(":*"):
        return ShellPermissionRule(
            type="prefix",
            prefix=rule[:-2],
        )

    # Wildcard patterns
    if "*" in rule or "?" in rule:
        return ShellPermissionRule(
            type="wildcard",
            pattern=rule,
        )

    # Default: exact match
    return ShellPermissionRule(type="exact", command=rule)


def match_wildcard_pattern(pattern: str, command: str) -> bool:
    """Match a command against a glob-style wildcard pattern.

    Args:
        pattern: The glob pattern (e.g., 'git *', 'npm install *').
        command: The command string to check.

    Returns:
        True if the command matches the pattern.
    """
    return fnmatch.fnmatch(command.strip(), pattern.strip())


def permission_rule_extract_prefix(rule: str) -> Optional[str]:
    """Extract the prefix from a legacy :* syntax rule.

    Args:
        rule: The rule string (e.g., "npm:*").

    Returns:
        The prefix (e.g., "npm"), or None if not a prefix rule.
    """
    if rule.endswith(":*"):
        return rule[:-2]
    return None


def suggestion_for_exact_command(tool_name: str, command: str) -> List[PermissionUpdate]:
    """Generate rule suggestions for an exact command match.

    Args:
        tool_name: The tool name (e.g., 'Bash').
        command: The command string.

    Returns:
        List of PermissionUpdate suggestions.
    """
    return [
        PermissionUpdate(tool_name=tool_name, rule_content=command, rule_type="allow"),
    ]


def suggestion_for_prefix(tool_name: str, prefix: str) -> List[PermissionUpdate]:
    """Generate rule suggestions for a prefix pattern.

    Args:
        tool_name: The tool name.
        prefix: The command prefix (e.g., "npm run").

    Returns:
        List of PermissionUpdate suggestions.
    """
    return [
        PermissionUpdate(
            tool_name=tool_name,
            rule_content=f"{prefix}:*",
            rule_type="allow",
        ),
    ]


__all__ = [
    "ShellPermissionRule",
    "PermissionUpdate",
    "parse_permission_rule",
    "match_wildcard_pattern",
    "permission_rule_extract_prefix",
    "suggestion_for_exact_command",
    "suggestion_for_prefix",
]
