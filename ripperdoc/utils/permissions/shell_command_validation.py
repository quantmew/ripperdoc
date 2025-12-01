"""Lightweight shell command validation heuristics."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ValidationResult:
    behavior: str  # 'passthrough' | 'ask' | 'allow' | 'deny'
    message: str
    rule_suggestions: Optional[List[str]] = None


_DANGEROUS_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"[`‵]"), "Command contains backticks for command substitution"),
    (re.compile(r"\$\("), "Command contains $() command substitution"),
    (re.compile(r"\$\{"), "Command contains ${} parameter substitution"),
    (re.compile(r"<\("), "Command contains process substitution <()"),
    (re.compile(r">\("), "Command contains process substitution >()"),
    (re.compile(r"<<<?"), "Command contains heredoc redirection"),
    (re.compile(r"(^|\s)source\s+"), "Command sources another script"),
]


def validate_shell_command(shell_command: str) -> ValidationResult:
    """Validate a shell command for risky constructs."""
    if not shell_command or not shell_command.strip():
        return ValidationResult(
            behavior="passthrough",
            message="Empty command is safe",
            rule_suggestions=None,
        )

    trimmed = shell_command.strip()

    if re.search(r"\bjq\b.*\bsystem\s*\(", trimmed):
        return ValidationResult(
            behavior="ask",
            message="jq command contains system() which executes arbitrary commands",
            rule_suggestions=None,
        )

    if re.search(r"\b(cat|tee)\s+<<\s*['\"]?EOF", trimmed):
        return ValidationResult(
            behavior="ask",
            message="Command contains heredoc which may run arbitrary content",
            rule_suggestions=None,
        )

    if re.search(r"[<>]\s*/dev/null", trimmed):
        # Explicit /dev/null redirection is benign for our purposes.
        sanitized = re.sub(r"\s*[<>]\s*/dev/null", "", trimmed)
    else:
        sanitized = trimmed

    for pattern, message in _DANGEROUS_PATTERNS:
        if pattern.search(sanitized):
            return ValidationResult(
                behavior="ask",
                message=message,
                rule_suggestions=None,
            )

    return ValidationResult(
        behavior="passthrough",
        message="Command passed validation",
        rule_suggestions=None,
    )


__all__ = ["validate_shell_command", "ValidationResult"]
