"""Parsing and cleaning helpers for permission checks."""

from __future__ import annotations

import re
import shlex


def strip_single_quotes(shell_command: str) -> str:
    """Strip content inside single quotes, handling escapes properly.

    Single-quoted content in shell is literal and cannot contain command
    substitution, so we can safely ignore it for security analysis.

    Double quotes are kept for analysis since they can contain variable
    expansions and command substitutions.
    """
    in_single_quote = False
    escaped = False
    result: list[str] = []

    i = 0
    while i < len(shell_command):
        char = shell_command[i]

        if escaped:
            escaped = False
            result.append(char)
            i += 1
            continue

        if char == "\\":
            escaped = True
            result.append(char)
            i += 1
            continue

        if char == "'":
            in_single_quote = not in_single_quote
            i += 1
            continue

        if not in_single_quote:
            result.append(char)

        i += 1

    return "".join(result)


def strip_quotes_for_analysis(command: str) -> str:
    """Strip content inside both single and double quotes for security analysis.

    This is used for checking shell metacharacters in arguments.
    Double quotes are stripped because they can contain variable expansions
    and command substitutions that need to be analyzed.
    """
    result: list[str] = []
    in_single_quote = False
    in_double_quote = False
    escaped = False

    i = 0
    while i < len(command):
        char = command[i]

        if escaped:
            escaped = False
            i += 1
            continue

        if char == "\\":
            escaped = True
            i += 1
            continue

        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            i += 1
            continue

        if char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            i += 1
            continue

        if not in_single_quote and not in_double_quote:
            result.append(char)

        i += 1

    return "".join(result)


def sanitize_safe_redirections(command: str) -> str:
    """Remove safe redirection patterns that don't need security checks."""
    # Remove stderr to stdout redirection (2>&1)
    sanitized = re.sub(r"\s+2\s*>&\s*1(?=\s|$)", "", command)
    # Remove redirections to /dev/null
    sanitized = re.sub(r"[012]?\s*>\s*/dev/null", "", sanitized)
    # Remove input from /dev/null
    sanitized = re.sub(r"\s*<\s*/dev/null", "", sanitized)
    return sanitized


def strip_quoted_content_for_destructive_check(command: str) -> str:
    """Strip content inside quotes for destructive command checking.

    This prevents false positives like 'find . -name "rm -rf /"' from
    triggering the rm -rf detection.
    """
    result: list[str] = []
    in_single_quote = False
    in_double_quote = False
    escaped = False

    for char in command:
        if escaped:
            escaped = False
            if not in_single_quote and not in_double_quote:
                result.append(char)
            continue

        if char == "\\":
            escaped = True
            if not in_single_quote and not in_double_quote:
                result.append(char)
            continue

        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            continue

        if char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            continue

        if not in_single_quote and not in_double_quote:
            result.append(char)

    return "".join(result)


def has_metachars_outside_quotes(command: str) -> bool:
    """Return True if shell metacharacters appear outside of quotes.

    Uses shlex for tokenization and preserves find -exec escaping rules.
    """
    lex = shlex.shlex(command, posix=True)
    lex.whitespace_split = True
    lex.commenters = ""

    tokens: list[str] = []
    try:
        while True:
            token = lex.get_token()
            if token == lex.eof:
                break
            tokens.append(token)
    except ValueError:
        # If shlex fails (e.g., unmatched quotes), be cautious.
        return True

    i = 0
    in_exec = False
    while i < len(tokens):
        token = tokens[i]
        if token in ("-exec", "-execdir"):
            in_exec = True
            i += 1
            continue
        if token == ";":
            if in_exec:
                in_exec = False
                i += 1
                continue
            return True
        if token == "+" and in_exec:
            in_exec = False
            i += 1
            continue
        if token in ("&", "|"):
            if i + 1 < len(tokens) and tokens[i + 1] == token:
                i += 2
                continue
            return True
        i += 1

    return False


__all__ = [
    "strip_single_quotes",
    "strip_quotes_for_analysis",
    "sanitize_safe_redirections",
    "strip_quoted_content_for_destructive_check",
    "has_metachars_outside_quotes",
]
