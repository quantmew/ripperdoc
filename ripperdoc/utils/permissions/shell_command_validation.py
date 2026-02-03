"""Shell command validation with comprehensive security checks.

This module implements security checks for shell commands to detect
potentially dangerous constructs before execution.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from ripperdoc.utils.permissions.destructive import check_destructive_commands
from ripperdoc.utils.permissions.interpreter import (
    extract_code_string,
    is_interpreter_command,
    strip_interpreter_code_strings,
)
from ripperdoc.utils.permissions.parse import (
    has_metachars_outside_quotes,
    sanitize_safe_redirections,
    strip_quotes_for_analysis,
    strip_single_quotes,
)
from ripperdoc.utils.permissions.rules import (
    DANGEROUS_METACHARACTER_PATTERNS,
    DANGEROUS_PATTERNS,
    is_safe_command_pattern,
)


@dataclass
class ValidationResult:
    """Result of shell command validation."""

    behavior: str  # 'passthrough' | 'ask' | 'allow' | 'deny'
    message: str
    rule_suggestions: Optional[List[str]] = None


def validate_shell_command(shell_command: str) -> ValidationResult:
    """Validate a shell command for security risks.

    This function checks for potentially dangerous shell constructs that could
    be used for command injection or other security issues.

    Args:
        shell_command: The shell command to validate.

    Returns:
        ValidationResult with behavior indicating whether the command is safe.
    """
    if not shell_command or not shell_command.strip():
        return ValidationResult(
            behavior="passthrough",
            message="Empty command is safe",
            rule_suggestions=None,
        )

    trimmed = shell_command.strip()
    first_token = trimmed.split()[0] if trimmed.split() else ""

    # Check for safe command patterns first
    if is_safe_command_pattern(trimmed):
        return ValidationResult(
            behavior="passthrough",
            message="Command matches safe pattern",
            rule_suggestions=None,
        )

    # FIRST: Check for destructive commands (highest priority)
    destructive_result = check_destructive_commands(trimmed)
    if destructive_result:
        return ValidationResult(
            behavior=destructive_result.behavior,
            message=destructive_result.message,
            rule_suggestions=destructive_result.rule_suggestions,
        )

    # Special handling for jq commands
    if first_token == "jq":
        # Check for system() function which can execute arbitrary commands
        if re.search(r"\bsystem\s*\(", trimmed):
            return ValidationResult(
                behavior="ask",
                message="jq command contains system() function which executes arbitrary commands",
                rule_suggestions=None,
            )

        # Check if jq is reading from files (should only read from stdin)
        jq_args = trimmed[3:].strip() if len(trimmed) > 3 else ""
        if re.search(r'(?:^|\s)(?:[^\'"\s-][^\s]*\s+)?(?:/|~|\w+\.\w+)', jq_args):
            if not re.match(r"^\.[^\s]+$", jq_args):
                return ValidationResult(
                    behavior="ask",
                    message=(
                        "jq command contains file arguments - jq should only read from stdin in read-only mode"
                    ),
                    rule_suggestions=None,
                )

    # Allow git commit with single-quoted heredoc (common pattern for commit messages)
    if re.search(r"git\s+commit\s+.*-m\s+\"?\$\(cat\s*<<'[^']+'[\s\S]*?\)\"?", trimmed):
        return ValidationResult(
            behavior="passthrough",
            message="Git commit with single-quoted heredoc is allowed",
            rule_suggestions=None,
        )

    # Check for heredoc with command substitution (dangerous)
    if re.search(r"['\"]?\$\(cat\s*<<(?!')", trimmed):
        return ValidationResult(
            behavior="ask",
            message="Command contains heredoc with command substitution",
            rule_suggestions=None,
        )

    # Check for heredoc patterns that could run arbitrary content
    if re.search(r"\b(cat|tee)\s+<<\s*['\"]?EOF", trimmed):
        return ValidationResult(
            behavior="ask",
            message="Command contains heredoc which may run arbitrary content",
            rule_suggestions=None,
        )

    # Strip single-quoted content for further analysis
    sanitized = strip_single_quotes(trimmed)

    # For interpreter commands, strip code strings before checking shell metacharacters
    sanitized_for_metachar_check = sanitized
    if is_interpreter_command(trimmed, first_token):
        sanitized_for_metachar_check = strip_interpreter_code_strings(sanitized, first_token)

    # Remove safe redirections
    sanitized = sanitize_safe_redirections(sanitized)
    sanitized_for_metachar_check = sanitize_safe_redirections(sanitized_for_metachar_check)

    # Check for shell metacharacters outside of quotes
    if has_metachars_outside_quotes(sanitized_for_metachar_check):
        return ValidationResult(
            behavior="ask",
            message="Command contains shell metacharacters (;, |, or &) outside of quoted arguments",
            rule_suggestions=None,
        )

    # Check for dangerous metacharacters in find/grep arguments
    stripped_for_pattern_check = strip_quotes_for_analysis(sanitized)
    for pattern in DANGEROUS_METACHARACTER_PATTERNS:
        if pattern.search(stripped_for_pattern_check):
            return ValidationResult(
                behavior="ask",
                message="Command contains shell metacharacters (;, |, or &) in find/grep arguments",
                rule_suggestions=None,
            )

    # Check for variables in dangerous contexts (redirections or pipes)
    if re.search(r"[<>|]\s*\$[A-Za-z_]", sanitized) or re.search(
        r"\$[A-Za-z_][A-Za-z0-9_]*\s*[|<>]", sanitized
    ):
        return ValidationResult(
            behavior="ask",
            message="Command contains variables in dangerous contexts (redirections or pipes)",
            rule_suggestions=None,
        )

    # Check for sourcing scripts
    if re.search(r"(^|\s)source\s+", sanitized):
        return ValidationResult(
            behavior="ask",
            message="Command sources another script which may execute arbitrary code",
            rule_suggestions=None,
        )
    if re.search(r"(^|\s)\.\s+[/~\w]", sanitized):
        if not re.search(r"\.\s+-[a-z]", sanitized):
            return ValidationResult(
                behavior="ask",
                message="Command sources another script which may execute arbitrary code",
                rule_suggestions=None,
            )

    # Check for eval
    if re.search(r"(^|\s)eval\s+", sanitized):
        return ValidationResult(
            behavior="ask",
            message="Command uses eval which executes arbitrary code",
            rule_suggestions=None,
        )

    # Check all dangerous patterns
    for pattern, message in DANGEROUS_PATTERNS:
        if pattern.search(sanitized):
            # Special handling for newlines
            if "newlines" in message:
                in_quote = False
                quote_char = None
                escaped = False
                newline_outside_quotes = False

                i = 0
                while i < len(trimmed):
                    char = trimmed[i]

                    if escaped:
                        escaped = False
                        i += 1
                        continue

                    if char == "\\":
                        escaped = True
                        i += 1
                        continue

                    if char in ("'", '"') and not escaped:
                        if not in_quote:
                            in_quote = True
                            quote_char = char
                        elif char == quote_char:
                            in_quote = False
                            quote_char = None

                    if char in ("\n", "\r") and not in_quote:
                        newline_outside_quotes = True
                        break

                    i += 1

                if not newline_outside_quotes:
                    if is_interpreter_command(trimmed, first_token):
                        code_string = extract_code_string(trimmed, first_token)
                        if code_string and any(c in code_string for c in ("\n", "\r")):
                            continue
                    else:
                        continue

            return ValidationResult(
                behavior="ask",
                message=message,
                rule_suggestions=None,
            )

    return ValidationResult(
        behavior="passthrough",
        message="Command passed all security checks",
        rule_suggestions=None,
    )


def is_complex_unsafe_shell_command(command: str) -> bool:
    """Check if a command contains complex shell operators that need special handling.

    This detects commands with control operators like &&, ||, ;, etc. that
    combine multiple commands.
    """
    if not command:
        return False

    in_single_quote = False
    in_double_quote = False
    escaped = False

    for i, char in enumerate(command):
        if escaped:
            escaped = False
            continue

        if char == "\\":
            escaped = True
            continue

        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            continue

        if char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            continue

        if not in_single_quote and not in_double_quote:
            if char in ("&", "|") and i + 1 < len(command):
                next_char = command[i + 1]
                if (char == "&" and next_char == "&") or (char == "|" and next_char == "|"):
                    return True

            if char == ";":
                if i + 1 >= len(command) or command[i + 1] != ";":
                    return True

    return False


__all__ = ["validate_shell_command", "ValidationResult", "is_complex_unsafe_shell_command"]
