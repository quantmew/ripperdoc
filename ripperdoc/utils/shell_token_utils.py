"""Shell token parsing utilities."""

from __future__ import annotations

import re
import shlex
from typing import Iterable, List

# Operators and redirections that should not be treated as executable tokens.
SHELL_OPERATORS_WITH_REDIRECTION: set[str] = {
    "|",
    "||",
    "&&",
    ";",
    ">",
    ">>",
    "<",
    "<<",
    "2>",
    "&>",
    "2>&1",
    "|&",
}

_REDIRECTION_PATTERNS = (
    re.compile(r"^\d?>?&\d+$"),  # 2>&1, >&2, etc.
    re.compile(r"^\d?>/dev/null$"),  # 2>/dev/null, >/dev/null
    re.compile(r"^/dev/null$"),
)


def parse_shell_tokens(shell_command: str) -> List[str]:
    """Parse a shell command into tokens, preserving operators for inspection."""
    if not shell_command:
        return []

    lexer = shlex.shlex(shell_command, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = ""

    try:
        return list(lexer)
    except ValueError:
        # Fall back to a coarse split to avoid hard failures.
        return shell_command.split()


def filter_valid_tokens(tokens: Iterable[str]) -> list[str]:
    """Remove shell control operators and redirection tokens."""
    return [token for token in tokens if token not in SHELL_OPERATORS_WITH_REDIRECTION]


def _is_redirection_token(token: str) -> bool:
    return any(pattern.match(token) for pattern in _REDIRECTION_PATTERNS)


def parse_and_clean_shell_tokens(raw_shell_string: str) -> List[str]:
    """Parse tokens and strip benign redirections to mirror reference cleaning."""
    tokens = parse_shell_tokens(raw_shell_string)
    if not tokens:
        return []

    cleaned: list[str] = []
    skip_next = False

    for idx, token in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue

        # Handle explicit redirection operators that are followed by a target.
        if token in {">&", ">", "1>", "2>", ">>"}:
            if idx + 1 < len(tokens):
                next_token = tokens[idx + 1]
                if _is_redirection_token(next_token):
                    skip_next = True
                    continue
            cleaned.append(token)
            continue

        # Skip inlined redirection tokens to /dev/null or file descriptors.
        if _is_redirection_token(token):
            continue

        cleaned.append(token)

    return filter_valid_tokens(cleaned)


__all__ = [
    "parse_shell_tokens",
    "parse_and_clean_shell_tokens",
    "filter_valid_tokens",
    "SHELL_OPERATORS_WITH_REDIRECTION",
]
