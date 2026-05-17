"""Permission types and rule matching utilities.

This module provides the core PermissionDecision and ToolRule types
used by the permission engine, along with legacy rule matching logic.

Kept for backward compatibility with the general permission system
(permission_engine.py, handler_control.py, rule_syntax.py).
New bash-specific logic lives in ripperdoc/tools/bash/.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from typing import List, Optional, Union

from ripperdoc.utils.shell.shell_token_utils import parse_shell_tokens


@dataclass
class ToolRule:
    tool_name: str
    rule_content: str
    behavior: str = "allow"


@dataclass
class PermissionDecision:
    behavior: str  # 'allow' | 'deny' | 'ask' | 'passthrough'
    message: Optional[str] = None
    updated_input: Optional[object] = None
    decision_reason: Optional[dict] = None
    rule_suggestions: Optional[Union[List[ToolRule], List[str]]] = None


def create_wildcard_rule(rule_name: str) -> str:
    """Create a glob wildcard rule string.

    Args:
        rule_name: The command prefix (e.g., "git", "npm")

    Returns:
        Wildcard rule string in glob format.
    """
    return f"{rule_name} *"


def create_tool_rule(rule_content: str) -> List[ToolRule]:
    return [ToolRule(tool_name="Bash", rule_content=rule_content)]


def create_wildcard_tool_rule(rule_name: str) -> List[ToolRule]:
    """Create a wildcard tool rule.

    Args:
        rule_name: The command prefix

    Returns:
        List containing a single ToolRule with wildcard pattern
    """
    return [ToolRule(tool_name="Bash", rule_content=create_wildcard_rule(rule_name))]


def _has_unquoted_shell_operators(command: str) -> bool:
    """Check if command contains shell operators outside of quotes.

    This prevents wildcard rules from matching commands with shell operators
    like &&, ||, ;, | which could be used to chain dangerous commands.

    Args:
        command: The command to check

    Returns:
        True if command contains unquoted shell operators, False otherwise
    """
    tokens = parse_shell_tokens(command)

    for token in tokens:
        if token in {"&&", "||", "|"}:
            return True
        if token.endswith(";") or token.startswith(";"):
            return True

    return False


def match_rule(command: str, rule: str) -> bool:
    """Return True if a command matches a rule (exact or glob pattern).

    Supports two formats:
    - Exact match: "git status" matches "git status" only
    - Glob patterns: "git * main" matches "git push main", "git pull main", etc.

    Security: Wildcard rules will NOT match commands
    containing shell operators (&&, ||, ;, |) outside of quotes.

    Args:
        command: The command to check
        rule: The rule pattern to match against

    Returns:
        True if command matches rule, False otherwise
    """
    command = command.strip()
    if not command:
        return False
    rule = rule.strip()
    if not rule:
        return False

    # Backward-compatibility: legacy "cmd:*" means "cmd *".
    rule = re.sub(r"(?<!/):\*", " *", rule)

    # Glob-style patterns with wildcards
    if "*" in rule or "?" in rule or "[" in rule:
        if _has_unquoted_shell_operators(command):
            return False
        return fnmatch.fnmatch(command, rule)

    # Exact match
    return command == rule


__all__ = [
    "PermissionDecision",
    "ToolRule",
    "create_wildcard_rule",
    "create_tool_rule",
    "create_wildcard_tool_rule",
    "match_rule",
    "_has_unquoted_shell_operators",
]
