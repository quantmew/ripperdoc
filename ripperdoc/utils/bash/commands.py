"""Command splitting and analysis utilities.

Provides safe command splitting with operator preservation,
output redirection extraction, and prefix extraction.
"""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ripperdoc.utils.bash.heredoc import extract_heredocs, restore_heredocs
from ripperdoc.utils.bash.shell_quote import try_parse_shell_command


def _generate_placeholders() -> Dict[str, str]:
    """Generate placeholder strings with random salt to prevent injection.

    The salt prevents malicious commands from containing literal placeholder
    strings that would be replaced during parsing.
    """
    salt = secrets.token_hex(8)
    return {
        "SINGLE_QUOTE": f"__SQ_{salt}__",
        "DOUBLE_QUOTE": f"__DQ_{salt}__",
        "NEW_LINE": f"__NL_{salt}__",
        "ESCAPED_OPEN_PAREN": f"__EOP_{salt}__",
        "ESCAPED_CLOSE_PAREN": f"__ECP_{salt}__",
    }


# File descriptors for standard streams
ALLOWED_FILE_DESCRIPTORS = frozenset({"0", "1", "2"})


def is_static_redirect_target(target: str) -> bool:
    """Check if a redirection target is a simple static file path.

    Returns False for targets containing dynamic content (variables, command
    substitutions, globs, shell expansions) which should remain visible in
    permission prompts for security.

    Args:
        target: The redirection target to check.

    Returns:
        True if the target is a static file path.
    """
    if re.search(r"[\s'\"]", target):
        return False
    if len(target) == 0:
        return False
    if target.startswith("#"):
        return False
    if (
        target.startswith("!")
        or target.startswith("=")
        or "$" in target
        or "`" in target
        or "*" in target
        or "?" in target
        or "[" in target
        or "{" in target
        or "~" in target
        or "(" in target
        or "<" in target
        or target.startswith("&")
    ):
        return False
    return True


@dataclass
class RedirectInfo:
    """Information about an output redirection."""
    operator: str  # '>' | '>>' | '&>' | '2>' etc.
    target: str
    fd: Optional[int] = None


@dataclass
class OutputRedirectionsResult:
    """Result of extracting output redirections."""
    command_without_redirections: str
    redirections: List[RedirectInfo] = field(default_factory=list)


def extract_output_redirections(command: str) -> OutputRedirectionsResult:
    """Extract output redirections from a command string.

    Handles: >file, >>file, &>file, 2>file, 2>&1, >/dev/null, etc.

    Args:
        command: The command string.

    Returns:
        OutputRedirectionsResult with the cleaned command and extracted redirections.
    """
    redirections: List[RedirectInfo] = []
    remaining = command.strip()

    # Pattern for output redirections: optional FD (0,1,2), then > or >>, then target
    # SECURITY: Match greedily from the end — bash resolves the LAST redirect.
    pattern = re.compile(
        r"(?P<fd>[012])?\s*(?P<op>>|>>|>&)\s*(?P<target>\S+)"
    )

    def _extract_one(cmd: str) -> tuple[str, Optional[RedirectInfo]]:
        """Try to extract one redirection from the end of the command."""
        # Match redirections at the end of the command
        m = re.search(r"(?P<fd>[012])?\s*(?P<op>>|>>|>&)\s*(?P<target>\S+)\s*$", cmd)
        if not m:
            return cmd, None

        fd_str = m.group("fd")
        fd = int(fd_str) if fd_str else None
        op = m.group("op")
        target = m.group("target")

        info = RedirectInfo(operator=op, target=target, fd=fd)
        # Remove the matched portion from the end
        cleaned = cmd[: m.start()].rstrip()
        return cleaned, info

    # Extract redirections one at a time from the end
    while True:
        remaining, info = _extract_one(remaining)
        if info is None:
            break
        redirections.append(info)

    redirections.reverse()

    return OutputRedirectionsResult(
        command_without_redirections=remaining,
        redirections=redirections,
    )


def split_command_with_operators(command: str) -> List[str]:
    """Split a command into parts on shell operators (|, &&, ||, ;).

    Preserves the operators as separate entries in the result list,
    so callers can distinguish between commands and operators.

    Uses placeholder-based security to prevent injection via literal
    placeholder strings in the command.

    Args:
        command: The command string to split.

    Returns:
        List of command parts and operators, in order.
    """
    if not command:
        return []

    placeholders = _generate_placeholders()

    # Extract heredocs before processing
    processed, heredocs = extract_heredocs(command)

    # Replace continuation lines (backslash + newline) with placeholder
    # This must be done BEFORE newline tokenization
    processed = re.sub(r"\\\n", placeholders["NEW_LINE"], processed)

    # Protect quoted strings by replacing their content
    # Replace single-quoted strings
    def _protect_sq(m: re.Match) -> str:
        return placeholders["SINGLE_QUOTE"]
    processed = re.sub(r"'[^']*'", _protect_sq, processed)

    # Replace double-quoted strings
    def _protect_dq(m: re.Match) -> str:
        return placeholders["DOUBLE_QUOTE"]
    processed = re.sub(r'"[^"]*"', _protect_dq, processed)

    # Protect escaped parentheses
    processed = processed.replace(r"\(", placeholders["ESCAPED_OPEN_PAREN"])
    processed = processed.replace(r"\)", placeholders["ESCAPED_CLOSE_PAREN"])

    # Now split on operators (outside quotes — quotes are placeholder-protected)
    # Split on &&, ||, |, ;, & preserving the operators
    parts: List[str] = []
    current: List[str] = []
    tokens = processed.split()

    for token in tokens:
        if token in ("&&", "||", "|", ";", "&"):
            if current:
                parts.append(" ".join(current))
                current = []
            parts.append(token)
        else:
            current.append(token)

    if current:
        parts.append(" ".join(current))

    return parts


def split_command(command: str) -> list[str]:
    """Split a compound command into individual subcommands.

    DEPRECATED: Prefer split_command_with_operators() for new code.
    This version discards operators and only returns command segments,
    using DEPRECATED splitCommand logic.

    Handles &&, ||, ;, | as separators.

    Args:
        command: The compound command string.

    Returns:
        List of individual command strings (operators removed).
    """
    parts = split_command_with_operators(command)
    # Filter out operators, keep only command parts
    operators = {"&&", "||", "|", ";", "&"}
    return [p for p in parts if p.strip() and p not in operators]


@dataclass
class CommandPrefixResult:
    """Result of command prefix extraction."""
    prefix: str
    subcommand: Optional[str] = None


def get_command_subcommand_prefix(command: str) -> Optional[CommandPrefixResult]:
    """Extract a command+subcommand prefix from a command.

    Examples:
        'git commit -m "msg"' → { prefix: 'git', subcommand: 'commit' }
        'npm run build' → { prefix: 'npm', subcommand: 'run' }
        'ls -la' → None (no subcommand)

    Args:
        command: The command string.

    Returns:
        CommandPrefixResult or None if no subcommand is detected.
    """
    tokens = command.strip().split()
    if len(tokens) < 2:
        return None

    # Second token must look like a subcommand (not a flag, path, or number)
    subcmd = tokens[1]
    if not re.match(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$", subcmd):
        return None

    return CommandPrefixResult(prefix=tokens[0], subcommand=subcmd)


__all__ = [
    "RedirectInfo",
    "OutputRedirectionsResult",
    "CommandPrefixResult",
    "extract_output_redirections",
    "split_command_with_operators",
    "split_command",
    "get_command_subcommand_prefix",
    "is_static_redirect_target",
    "ALLOWED_FILE_DESCRIPTORS",
]
