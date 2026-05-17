"""Sed command security validation.


Provides allowlist-based validation for sed commands to detect and block
dangerous operations (write commands, execute commands, etc.).
"""

from __future__ import annotations

import re

from typing import List

from ripperdoc.utils.bash.shell_quote import try_parse_shell_command
from ripperdoc.utils.bash.commands import split_command
from ripperdoc.security import PermissionResult


# ---------------------------------------------------------------------------
# Helper: validate flags against allowlist
# ---------------------------------------------------------------------------


def _validate_flags_against_allowlist(flags: List[str], allowed_flags: List[str]) -> bool:
    """Validate flags against an allowlist, handling combined flags.

    Args:
        flags: List of flag strings (e.g., ['-n', '-E', '-r']).
        allowed_flags: List of allowed flag strings.

    Returns:
        True if all flags are valid.
    """
    for flag in flags:
        if flag.startswith("-") and not flag.startswith("--") and len(flag) > 2:
            # Combined flags like -nE
            for ch in flag[1:]:
                if f"-{ch}" not in allowed_flags:
                    return False
        else:
            if flag not in allowed_flags:
                return False
    return True


# ---------------------------------------------------------------------------
# Pattern 1: Line printing command (sed -n 'Np')
# ---------------------------------------------------------------------------


def is_line_printing_command(command: str, expressions: List[str]) -> bool:
    """Check if this is a line printing command with -n flag.

    Allows: sed -n 'N' | sed -n 'N,M' with optional -E, -r, -z flags.
    Allows semicolon-separated print commands like: sed -n '1p;2p;3p'
    File arguments are ALLOWED for this pattern.

    Args:
        command: The full sed command.
        expressions: Extracted sed expressions.

    Returns:
        True if the command matches the line printing pattern.
    """
    sed_match = re.match(r"^\s*sed\s+", command)
    if not sed_match:
        return False

    without_sed = command[sed_match.end():]
    parse_result = try_parse_shell_command(without_sed)
    if not parse_result.success:
        return False
    parsed = parse_result.tokens
    parsed_strs = [str(t) for t in parsed if isinstance(t, str)]

    # Extract flags
    flags = [t for t in parsed_strs if t.startswith("-") and t != "--"]

    # Validate flags
    allowed_flags = [
        "-n", "--quiet", "--silent",
        "-E", "--regexp-extended", "-r",
        "-z", "--zero-terminated", "--posix",
    ]
    if not _validate_flags_against_allowlist(flags, allowed_flags):
        return False

    # Check for -n flag
    has_n = False
    for flag in flags:
        if flag in ("-n", "--quiet", "--silent"):
            has_n = True
            break
        if flag.startswith("-") and not flag.startswith("--") and "n" in flag:
            has_n = True
            break
    if not has_n:
        return False

    # Must have at least one expression
    if not expressions:
        return False

    # All expressions must be print commands
    for expr in expressions:
        sub_commands = expr.split(";")
        for sub in sub_commands:
            if not _is_print_command(sub.strip()):
                return False

    return True


def _is_print_command(cmd: str) -> bool:
    """Check if a single sed command is a valid print command.

    STRICT ALLOWLIST: only p, Np, N,Mp patterns.
    """
    if not cmd:
        return False
    return bool(re.match(r"^(?:\d+|\d+,\d+)?p$", cmd))


# ---------------------------------------------------------------------------
# Pattern 2: Substitution command (sed 's/pattern/replacement/flags')
# ---------------------------------------------------------------------------


def _is_substitution_command(
    command: str,
    expressions: List[str],
    has_file_arguments: bool,
    allow_file_writes: bool = False,
) -> bool:
    """Check if this is a safe substitution command.

    Args:
        command: The full sed command.
        expressions: Extracted sed expressions.
        has_file_arguments: Whether the command has file arguments.
        allow_file_writes: Whether to allow -i flag and file arguments.

    Returns:
        True if the command matches the substitution pattern.
    """
    if not allow_file_writes and has_file_arguments:
        return False

    sed_match = re.match(r"^\s*sed\s+", command)
    if not sed_match:
        return False

    without_sed = command[sed_match.end():]
    parse_result = try_parse_shell_command(without_sed)
    if not parse_result.success:
        return False

    parsed = parse_result.tokens
    parsed_strs = [str(t) for t in parsed if isinstance(t, str)]

    flags = [t for t in parsed_strs if t.startswith("-") and t != "--"]

    allowed_flags = ["-E", "--regexp-extended", "-r", "--posix"]
    if allow_file_writes:
        allowed_flags.extend(["-i", "--in-place"])

    if not _validate_flags_against_allowlist(flags, allowed_flags):
        return False

    if len(expressions) != 1:
        return False

    expr = expressions[0].strip()
    if not expr.startswith("s"):
        return False

    # Parse substitution: s/pattern/replacement/flags
    subst_match = re.match(r"^s/(.*?)$", expr)
    if not subst_match:
        return False

    rest = subst_match.group(1)
    delimiter_count = rest.count("/")

    if delimiter_count != 2:
        return False

    # Extract flags after last delimiter
    last_delim = rest.rindex("/")
    expr_flags = rest[last_delim + 1:]

    # Only allow g, p, i, I, m, M and optionally one digit 1-9
    if not re.match(r"^[gpimIM]*[1-9]?[gpimIM]*$", expr_flags):
        return False

    return True


# ---------------------------------------------------------------------------
# Sed expression extraction
# ---------------------------------------------------------------------------


def _has_file_args(command: str) -> bool:
    """Check if a sed command has file arguments (not just stdin).

    Args:
        command: The sed command.

    Returns:
        True if the command has file arguments.
    """
    sed_match = re.match(r"^\s*sed\s+", command)
    if not sed_match:
        return False

    without_sed = command[sed_match.end():]
    parse_result = try_parse_shell_command(without_sed)
    if not parse_result.success:
        return True  # Assume dangerous if parsing fails

    parsed = parse_result.tokens
    try:
        arg_count = 0
        has_e_flag = False
        i = 0
        while i < len(parsed):
            arg = parsed[i]
            if not isinstance(arg, str):
                if isinstance(arg, dict) and arg.get("op") == "glob":
                    return True
                i += 1
                continue

            if arg in ("-e", "--expression") and i + 1 < len(parsed):
                has_e_flag = True
                i += 2
                continue
            if arg.startswith("--expression=") or arg.startswith("-e="):
                has_e_flag = True
                i += 1
                continue
            if arg.startswith("-"):
                i += 1
                continue

            arg_count += 1
            if has_e_flag:
                return True
            if arg_count > 1:
                return True
            i += 1

        return False
    except Exception:
        return True


def _extract_sed_expressions(command: str) -> List[str]:
    """Extract sed expressions from a command.

    Args:
        command: Full sed command.

    Returns:
        List of sed expressions.

    Raises:
        ValueError: If parsing fails or dangerous flag combos detected.
    """
    expressions: List[str] = []

    sed_match = re.match(r"^\s*sed\s+", command)
    if not sed_match:
        return expressions

    without_sed = command[sed_match.end():]

    # Reject dangerous flag combinations
    if re.search(r"-e[wWe]", without_sed) or re.search(r"-w[eE]", without_sed):
        raise ValueError("Dangerous flag combination detected")

    parse_result = try_parse_shell_command(without_sed)
    if not parse_result.success:
        raise ValueError(f"Malformed shell syntax: {parse_result.error}")

    parsed = parse_result.tokens
    try:
        found_e_flag = False
        found_expression = False

        i = 0
        while i < len(parsed):
            arg = parsed[i]
            if not isinstance(arg, str):
                i += 1
                continue

            if arg in ("-e", "--expression") and i + 1 < len(parsed):
                found_e_flag = True
                next_arg = parsed[i + 1]
                if isinstance(next_arg, str):
                    expressions.append(next_arg)
                    i += 1
                i += 1
                continue

            if arg.startswith("--expression="):
                found_e_flag = True
                expressions.append(arg[len("--expression="):])
                i += 1
                continue

            if arg.startswith("-e="):
                found_e_flag = True
                expressions.append(arg[3:])
                i += 1
                continue

            if arg.startswith("-"):
                i += 1
                continue

            if not found_e_flag and not found_expression:
                expressions.append(arg)
                found_expression = True
                i += 1
                continue

            break

    except Exception as exc:
        raise ValueError(f"Failed to parse sed command: {exc}")

    return expressions


# ---------------------------------------------------------------------------
# Dangerous operations denylist
# ---------------------------------------------------------------------------


def _contains_dangerous_operations(expression: str) -> bool:
    """Check if a sed expression contains dangerous operations.

    DENYLIST: Rejects w/W (write), e/E (execute), and other dangerous patterns.

    Args:
        expression: A single sed expression.

    Returns:
        True if dangerous operations are detected.
    """
    cmd = expression.strip()
    if not cmd:
        return False

    # Reject non-ASCII characters
    if re.search(r"[^\x01-\x7F]", cmd):
        return True

    # Reject curly braces (blocks)
    if "{" in cmd or "}" in cmd:
        return True

    # Reject newlines
    if "\n" in cmd:
        return True

    # Reject comments (# not after s)
    hash_idx = cmd.find("#")
    if hash_idx != -1 and not (hash_idx > 0 and cmd[hash_idx - 1] == "s"):
        return True

    # Reject negation operator
    if re.match(r"^!", cmd) or re.search(r"[/\d$]!", cmd):
        return True

    # Reject tilde step address
    if re.search(r"\d\s*~\s*\d|,\s*~\s*\d|\$\s*~\s*\d", cmd):
        return True

    # Reject comma at start
    if re.match(r"^,", cmd):
        return True

    # Reject comma +/-
    if re.search(r",\s*[+-]", cmd):
        return True

    # Reject backslash tricks
    if re.search(r"s\\\\", cmd) or re.search(r"\\[|#%@]", cmd):
        return True

    # Reject escaped slashes with w/W
    if re.search(r"\\/.*[wW]", cmd):
        return True

    # Reject /pattern/ w/e patterns
    if re.search(r"/[^/]*\s+[wWeE]", cmd):
        return True

    # Reject malformed substitution
    if re.match(r"^s/", cmd) and not re.match(r"^s/[^/]*/[^/]*/[^/]*$", cmd):
        return True

    # PARANOID: Reject s-commands ending in w/W/e/E
    if re.match(r"^s.", cmd) and re.search(r"[wWeE]$", cmd):
        proper = re.match(r"^s([^\\\n]).*?\1.*?\1[^wWeE]*$", cmd)
        if not proper:
            return True

    # Check for dangerous write commands
    if (
        re.match(r"^[wW]\s*\S+", cmd)
        or re.match(r"^\d+\s*[wW]\s*\S+", cmd)
        or re.match(r"^\$\s*[wW]\s*\S+", cmd)
        or re.match(r"^/[^/]*/[IMim]*\s*[wW]\s*\S+", cmd)
        or re.match(r"^\d+,\d+\s*[wW]\s*\S+", cmd)
        or re.match(r"^\d+,\$\s*[wW]\s*\S+", cmd)
        or re.match(r"^/[^/]*/[IMim]*,\/[^/]*\/[IMim]*\s*[wW]\s*\S+", cmd)
    ):
        return True

    # Check for dangerous execute commands
    if (
        re.match(r"^e", cmd)
        or re.match(r"^\d+\s*e", cmd)
        or re.match(r"^\$\s*e", cmd)
        or re.match(r"^/[^/]*/[IMim]*\s*e", cmd)
        or re.match(r"^\d+,\d+\s*e", cmd)
        or re.match(r"^\d+,\$\s*e", cmd)
        or re.match(r"^/[^/]*/[IMim]*,\/[^/]*\/[IMim]*\s*e", cmd)
    ):
        return True

    # Check substitution flags for w/e
    subst_match = re.match(r"s([^\\\n]).*?\1.*?\1(.*?)$", cmd)
    if subst_match:
        flags = subst_match.group(2) or ""
        if "w" in flags or "W" in flags or "e" in flags or "E" in flags:
            return True

    # Check y (transliterate) command
    y_match = re.match(r"y([^\\\n])", cmd)
    if y_match and re.search(r"[wWeE]", cmd):
        return True

    return False


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------


def sed_command_is_allowed_by_allowlist(
    command: str,
    allow_file_writes: bool = False,
) -> bool:
    """Check if a sed command is allowed by the allowlist.

    Args:
        command: The sed command to check.
        allow_file_writes: When True, allows -i flag and file arguments
                           for substitution commands.

    Returns:
        True if the command is allowed.
    """
    try:
        expressions = _extract_sed_expressions(command)
    except (ValueError, Exception):
        return False

    has_file_arguments = _has_file_args(command)

    is_pattern1 = False
    is_pattern2 = False

    if allow_file_writes:
        is_pattern2 = _is_substitution_command(
            command, expressions, has_file_arguments, allow_file_writes=True
        )
    else:
        is_pattern1 = is_line_printing_command(command, expressions)
        is_pattern2 = _is_substitution_command(
            command, expressions, has_file_arguments
        )

    if not is_pattern1 and not is_pattern2:
        return False

    # Pattern 2 does not allow semicolons
    for expr in expressions:
        if is_pattern2 and ";" in expr:
            return False

    # Defense-in-depth: check denylist
    for expr in expressions:
        if _contains_dangerous_operations(expr):
            return False

    return True


def check_sed_constraints(
    command: str,
    mode: str = "default",
) -> PermissionResult:
    """Cross-cutting validation step for sed commands.

    Args:
        command: The command string.
        mode: The permission mode ('default', 'acceptEdits', etc.).

    Returns:
        PermissionResult indicating whether the sed command is safe.
    """
    commands = split_command(command)

    for cmd in commands:
        trimmed = cmd.strip()
        base_cmd = trimmed.split()[0] if trimmed.split() else ""
        if base_cmd != "sed":
            continue

        allow_file_writes = mode == "acceptEdits"
        is_allowed = sed_command_is_allowed_by_allowlist(trimmed, allow_file_writes=allow_file_writes)

        if not is_allowed:
            return PermissionResult.ask(
                "sed command requires approval (contains potentially dangerous operations)",
                reason={
                    "type": "other",
                    "reason": "sed command contains operations that require explicit approval",
                },
            )

    return PermissionResult.passthrough("No dangerous sed operations detected")


__all__ = [
    "is_line_printing_command",
    "sed_command_is_allowed_by_allowlist",
    "check_sed_constraints",
]
