"""Interpreter command helpers for permission checks."""

from __future__ import annotations

import re


_INTERPRETER_TOKENS: set[str] = {"python", "python3", "node", "bash", "sh", "zsh", "perl", "ruby"}


def is_interpreter_command(command: str, first_token: str | None = None) -> bool:
    """Check if the command is an interpreter command that executes code strings.

    Interpreter commands like `python -c "code"`, `node -e "code"`, `bash -c "code"`
    should have different validation rules for their code strings.
    """
    token = first_token
    if token is None:
        trimmed = command.strip()
        token = trimmed.split()[0] if trimmed.split() else ""

    if token not in _INTERPRETER_TOKENS:
        return False

    pattern = rf"\b{re.escape(token)}\s+-(c|e)\s+[\"']"
    return bool(re.search(pattern, command))


def extract_code_string(command: str, first_token: str | None = None) -> str:
    """Extract the code string from an interpreter command."""
    token = first_token
    if token is None:
        trimmed = command.strip()
        token = trimmed.split()[0] if trimmed.split() else ""

    if not is_interpreter_command(command, token):
        return ""

    pattern = rf"{re.escape(token)}\s+-(c|e)\s+([\"'])(.*?)(?<!\\)\2"
    match = re.search(pattern, command, re.DOTALL)

    if match:
        code_string = match.group(3)
        return code_string.replace('\\"', '"').replace("\\'", "'")

    return ""


def strip_interpreter_code_strings(command: str, first_token: str | None = None) -> str:
    """Strip code strings from interpreter commands for validation."""
    token = first_token
    if token is None:
        trimmed = command.strip()
        token = trimmed.split()[0] if trimmed.split() else ""

    if not is_interpreter_command(command, token):
        return command

    pattern = rf"({re.escape(token)}\s+-(c|e)\s+)([\"'])(.*?)(?<!\\)\3"

    def replace_code_string(match: re.Match[str]) -> str:
        prefix = match.group(1)
        quote = match.group(3)
        return f"{prefix}{quote}__CODE_STRING__{quote}"

    return re.sub(pattern, replace_code_string, command, flags=re.DOTALL)


__all__ = ["is_interpreter_command", "extract_code_string", "strip_interpreter_code_strings"]
