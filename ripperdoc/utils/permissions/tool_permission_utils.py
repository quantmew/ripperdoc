"""Permission evaluation helpers."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Set

from ripperdoc.utils.permissions.path_validation_utils import validate_shell_command_paths
from ripperdoc.utils.permissions.shell_command_validation import validate_shell_command
from ripperdoc.utils.safe_get_cwd import safe_get_cwd
from ripperdoc.utils.shell_token_utils import parse_and_clean_shell_tokens, parse_shell_tokens


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
    rule_suggestions: Optional[List[ToolRule] | List[str]] = None


def create_wildcard_rule(rule_name: str, use_glob_style: bool = False) -> str:
    """Create a wildcard/prefix rule string.

    Args:
        rule_name: The command prefix (e.g., "git", "npm")
        use_glob_style: If True, creates "rule_name *" format;
                       if False, creates "rule_name:*" format (default for compatibility)

    Returns:
        Wildcard rule string in requested format

    Examples:
        >>> create_wildcard_rule("git")
        "git:*"
        >>> create_wildcard_rule("git", use_glob_style=True)
        "git *"
    """
    if use_glob_style:
        return f"{rule_name} *"
    return f"{rule_name}:*"


def create_tool_rule(rule_content: str) -> List[ToolRule]:
    return [ToolRule(tool_name="Bash", rule_content=rule_content)]


def create_wildcard_tool_rule(rule_name: str, use_glob_style: bool = False) -> List[ToolRule]:
    """Create a wildcard tool rule.

    Args:
        rule_name: The command prefix
        use_glob_style: Whether to use glob-style format

    Returns:
        List containing a single ToolRule with wildcard pattern
    """
    return [ToolRule(tool_name="Bash", rule_content=create_wildcard_rule(rule_name, use_glob_style))]


def extract_rule_prefix(rule_string: str) -> Optional[str]:
    return rule_string[:-2] if rule_string.endswith(":*") else None


def _has_unquoted_shell_operators(command: str) -> bool:
    """Check if command contains shell operators outside of quotes.

    This prevents wildcard rules from matching commands with shell operators
    like &&, ||, ;, | which could be used to chain dangerous commands.

    Args:
        command: The command to check

    Returns:
        True if command contains unquoted shell operators, False otherwise

    Examples:
        >>> _has_unquoted_shell_operators("git status && rm -rf /")
        True
        >>> _has_unquoted_shell_operators("echo 'foo && bar'")
        False
        >>> _has_unquoted_shell_operators("ls | grep foo")
        True
        >>> _has_unquoted_shell_operators("echo hi; rm -rf /")
        True
    """
    # Parse the command into tokens, which handles quotes correctly
    tokens = parse_shell_tokens(command)

    # Check for shell operators in the tokens
    # Note: parse_shell_tokens may attach semicolon to adjacent tokens
    # (e.g., "echo hi; rm" -> ["echo", "hi;", "rm"] or ";echo" -> [";echo"])
    for token in tokens:
        # Check for separate operator tokens
        if token in {"&&", "||", "|"}:
            return True
        # Check for semicolon (may be attached to previous or next token)
        if token.endswith(";") or token.startswith(";"):
            return True

    return False


def match_rule(command: str, rule: str) -> bool:
    """Return True if a command matches a rule (exact, prefix wildcard, or glob pattern).

    Supports three formats:
    - Exact match: "git status" matches "git status" only
    - Legacy prefix: "git:*" matches any command starting with "git"
    - Glob patterns: "git * main" matches "git push main", "git pull main", etc.

    Glob patterns support:
    - * (matches any characters)
    - ? (matches single character)
    - [seq] (matches any character in seq)
    - [!seq] (matches any character NOT in seq)

    Security: Wildcard rules (both legacy and glob) will NOT match commands
    containing shell operators (&&, ||, ;, |) outside of quotes. This prevents
    a rule like "git:*" from matching "git status && rm -rf /". Exact match
    rules still work with operators (user explicitly approved the full command).

    Args:
        command: The command to check
        rule: The rule pattern to match against

    Returns:
        True if command matches rule, False otherwise

    Examples:
        >>> match_rule("git status", "git:*")
        True
        >>> match_rule("npm install", "npm *")
        True
        >>> match_rule("pip install", "* install")
        True
        >>> match_rule("git push main", "git * main")
        True
        >>> match_rule("git status && rm -rf /", "git:*")
        False
        >>> match_rule("git status && git commit", "git status && git commit")
        True
    """
    command = command.strip()
    if not command:
        return False

    # Legacy format: prefix:* (highest priority for backward compatibility)
    prefix = extract_rule_prefix(rule)
    if prefix is not None:
        # For wildcard rules, reject commands with shell operators for security
        if _has_unquoted_shell_operators(command):
            return False
        return command.startswith(prefix)

    # New format: glob-style patterns with wildcards
    if "*" in rule or "?" in rule or "[" in rule:
        # For wildcard rules, reject commands with shell operators for security
        if _has_unquoted_shell_operators(command):
            return False
        return fnmatch.fnmatch(command, rule)

    # Exact match - allow even with operators (user explicitly approved full command)
    return command == rule


def _merge_rules(*rules: Iterable[str]) -> Set[str]:
    merged: Set[str] = set()
    for collection in rules:
        merged.update(collection)
    return merged


def _is_command_read_only(
    command: str,
    injection_check: Callable[[str], bool],
) -> bool:
    """Heuristic read-only detector mirroring the reference intent."""
    validation = validate_shell_command(command)
    if validation.behavior != "passthrough":
        return False

    cleaned_tokens = parse_and_clean_shell_tokens(command)
    if not cleaned_tokens:
        return True

    # Treat pipelines/compound commands as read-only only if every segment is safe.
    tokens = parse_shell_tokens(command)
    if "|" in tokens:
        parts: list[str] = []
        current: list[str] = []
        for token in tokens:
            if token == "|":
                if current:
                    parts.append(" ".join(current))
                current = []
            else:
                current.append(token)
        if current:
            parts.append(" ".join(current))
        return all(_is_command_read_only(part, injection_check) for part in parts)

    dangerous_prefixes = {
        "rm",
        "mv",
        "chmod",
        "chown",
        "sudo",
        "dd",
        "tee",
        "truncate",
        "kill",
        "pkill",
        "systemctl",
        "service",
    }
    first = cleaned_tokens[0]
    if first in dangerous_prefixes:
        return False
    if first == "git":
        if len(cleaned_tokens) < 2:
            return False
        allowed_git = {
            "status",
            "diff",
            "show",
            "log",
            "rev-parse",
            "ls-files",
            "remote",
            "branch",
            "tag",
            "blame",
            "reflog",
        }
        return cleaned_tokens[1] in allowed_git

    # If no injection was detected and the command is free of mutations, treat as read-only.
    return not injection_check(command)


def _collect_rule_suggestions(command: str) -> List[ToolRule]:
    """Collect rule suggestions for a command.

    Suggests three options:
    1. Exact command match
    2. Legacy prefix wildcard (git:*)
    3. Glob-style wildcard (git *)

    This gives users choice between formats while maintaining backward compatibility.

    Args:
        command: The command to generate suggestions for

    Returns:
        List of suggested ToolRule objects
    """
    suggestions: list[ToolRule] = [
        # Exact match
        ToolRule(tool_name="Bash", rule_content=command)
    ]

    tokens = parse_and_clean_shell_tokens(command)
    if tokens:
        # Legacy prefix format
        suggestions.append(
            ToolRule(tool_name="Bash", rule_content=create_wildcard_rule(tokens[0], use_glob_style=False))
        )
        # New glob-style format
        suggestions.append(
            ToolRule(tool_name="Bash", rule_content=create_wildcard_rule(tokens[0], use_glob_style=True))
        )

    return suggestions


def evaluate_shell_command_permissions(
    tool_request: object,
    allowed_rules: Iterable[str],
    denied_rules: Iterable[str],
    allowed_working_dirs: Set[str] | None = None,
    *,
    danger_detector: Callable[[str], bool] | None = None,
    read_only_detector: Callable[[str, Callable[[str], bool]], bool] | None = None,
) -> PermissionDecision:
    """Evaluate whether a bash command should be allowed.

    Args:
        tool_request: The tool request containing the command.
        allowed_rules: Rules that allow commands.
        denied_rules: Rules that deny commands.
        allowed_working_dirs: Allowed working directories.
        danger_detector: Callable that returns True if command contains dangerous
            shell patterns (injection, destructive commands, etc.).
        read_only_detector: Callable that returns True if command is read-only.

    Returns:
        PermissionDecision indicating whether the command should be allowed.
    """
    command = tool_request.command if hasattr(tool_request, "command") else str(tool_request)
    trimmed_command = command.strip()
    allowed_working_dirs = allowed_working_dirs or {safe_get_cwd()}
    danger_detector = danger_detector or (
        lambda cmd: validate_shell_command(cmd).behavior != "passthrough"
    )
    read_only_detector = read_only_detector or _is_command_read_only

    # Detect dangerous patterns once, reuse the result
    has_dangerous_patterns = danger_detector(trimmed_command)

    merged_denied = _merge_rules(denied_rules)
    merged_allowed = _merge_rules(allowed_rules)

    if any(match_rule(trimmed_command, rule) for rule in merged_denied):
        return PermissionDecision(
            behavior="deny",
            message=f"Permission to run '{trimmed_command}' has been denied.",
            decision_reason={"type": "rule"},
            rule_suggestions=None,
        )

    if any(match_rule(trimmed_command, rule) for rule in merged_allowed):
        return PermissionDecision(
            behavior="allow",
            updated_input=tool_request,
            decision_reason={"type": "rule"},
            message="Command approved by configured rule.",
        )

    path_result = validate_shell_command_paths(
        trimmed_command, safe_get_cwd(), allowed_working_dirs
    )
    if path_result.behavior != "passthrough":
        return PermissionDecision(
            behavior="ask",
            message=path_result.message,
            decision_reason={"type": "path_validation"},
            rule_suggestions=None,
        )

    validation_result = validate_shell_command(trimmed_command)
    if validation_result.behavior != "passthrough":
        return PermissionDecision(
            behavior="ask",
            message=validation_result.message,
            decision_reason={"type": "validation"},
            rule_suggestions=None,
        )

    tokens = parse_shell_tokens(trimmed_command)
    if "|" in tokens:
        left_tokens = []
        right_tokens = []
        pipe_seen = False
        for token in tokens:
            if token == "|":
                pipe_seen = True
                continue
            if pipe_seen:
                right_tokens.append(token)
            else:
                left_tokens.append(token)
        left_command = " ".join(left_tokens).strip()
        right_command = " ".join(right_tokens).strip()

        left_result = evaluate_shell_command_permissions(
            type("Cmd", (), {"command": left_command}),
            merged_allowed,
            merged_denied,
            allowed_working_dirs,
            danger_detector=danger_detector,
            read_only_detector=read_only_detector,
        )
        right_read_only = read_only_detector(right_command, danger_detector)

        if left_result.behavior == "deny":
            return left_result
        if not right_read_only:
            return PermissionDecision(
                behavior="ask",
                message="Pipe right-hand command is not read-only.",
                decision_reason={"type": "subcommand"},
                rule_suggestions=_collect_rule_suggestions(right_command),
            )
        if left_result.behavior == "allow":
            return PermissionDecision(
                behavior="allow",
                updated_input=tool_request,
                decision_reason={"type": "subcommand"},
            )
        return PermissionDecision(
            behavior="ask",
            message="Permission required for piped command.",
            decision_reason={"type": "subcommand"},
            rule_suggestions=_collect_rule_suggestions(trimmed_command),
        )

    if read_only_detector(trimmed_command, danger_detector) and not has_dangerous_patterns:
        return PermissionDecision(
            behavior="allow",
            updated_input=tool_request,
            decision_reason={"type": "other", "reason": "Read-only command"},
        )

    return PermissionDecision(
        behavior="passthrough",
        message="Command requires permission",
        decision_reason={"type": "default"},
        rule_suggestions=_collect_rule_suggestions(trimmed_command),
    )


def is_command_read_only(command: str) -> bool:
    """Public wrapper to test if a command is read-only using reference heuristics."""
    return _is_command_read_only(
        command, lambda cmd: validate_shell_command(cmd).behavior != "passthrough"
    )


__all__ = [
    "PermissionDecision",
    "ToolRule",
    "create_tool_rule",
    "create_wildcard_tool_rule",
    "evaluate_shell_command_permissions",
    "extract_rule_prefix",
    "match_rule",
    "is_command_read_only",
]
