"""Shell quoting utilities.

Provides safe wrappers around shlex for parsing shell commands,
detecting malformed tokens, and shell-quote single-quote bugs.
"""

from __future__ import annotations

import re
import shlex
from typing import Any, Callable, Optional


class ParseResult:
    """Result of a shell command parse attempt."""

    def __init__(self, success: bool, tokens: Optional[list[Any]] = None, error: Optional[str] = None):
        self.success = success
        self.tokens = tokens or []
        self.error = error


def try_parse_shell_command(
    command: str,
    env_callback: Optional[Callable[..., Any]] = None,
) -> ParseResult:
    """Parse a shell command into tokens, handling errors gracefully.

    This is a Python implementation of shell-quote's parsing via shlex.

    Args:
        command: The shell command string to parse.
        env_callback: Optional callback for variable expansion references.
                      Receives variable name, returns replacement string.

    Returns:
        ParseResult with success flag, tokens list, and optional error message.
    """
    if not command:
        return ParseResult(success=True, tokens=[])

    try:
        lexer = shlex.shlex(command, posix=True)
        lexer.whitespace_split = True
        lexer.commenters = ""

        tokens: list[Any] = []
        for token in lexer:
            tokens.append(token)

        return ParseResult(success=True, tokens=tokens)
    except ValueError as exc:
        return ParseResult(success=False, error=str(exc))


def has_malformed_tokens(command: str) -> bool:
    """Check if a command has malformed tokens that could indicate a parser differential.

    Detects issues like:
    - Unclosed quotes
    - Invalid escape sequences
    - Other shlex parsing failures

    Args:
        command: The shell command to check.

    Returns:
        True if malformed tokens are detected.
    """
    result = try_parse_shell_command(command)
    return not result.success


def has_shell_quote_single_quote_bug(command: str) -> bool:
    """Check for shell-quote's known single-quote parsing bug.

    shell-quote has a known issue where patterns like `'\'` (backslash inside
    single quotes) cause quote state desynchronization, allowing injection.

    Args:
        command: The shell command to check.

    Returns:
        True if the potentially buggy pattern is detected.
    """
    # Look for patterns like: '\'  or  '\''  inside a larger command
    # These are cases where a backslash appears inside single quotes,
    # which shell-quote handles incorrectly but bash handles correctly.
    if not command:
        return False

    # Check for unescaped backslash inside single quotes
    in_single_quote = False
    escaped = False

    for char in command:
        if escaped:
            escaped = False
            continue

        if char == "\\" and not in_single_quote:
            escaped = True
            continue

        if char == "'":
            in_single_quote = not in_single_quote
            continue

        # If we see a backslash inside single quotes, that's the bug pattern
        if char == "\\" and in_single_quote:
            return True

    return False


def quote(s: str) -> str:
    """Shell-quote a string, wrapping in single quotes with proper escaping.

    Implementation of shell-quote's quote() function.

    Args:
        s: The string to quote.

    Returns:
        Properly shell-quoted string.
    """
    if not s:
        return "''"

    # If the string only contains safe characters, no quoting needed
    if re.match(r'^[A-Za-z0-9_./:-]+$', s):
        return s

    # Use single quotes, with single quotes inside escaped via: '\'' 
    return "'" + s.replace("'", "'\\''") + "'"


__all__ = [
    "ParseResult",
    "try_parse_shell_command",
    "has_malformed_tokens",
    "has_shell_quote_single_quote_bug",
    "quote",
]
