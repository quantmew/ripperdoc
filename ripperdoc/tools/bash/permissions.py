"""Bash tool permission checking pipeline.


Provides the full bash_tool_has_permission() pipeline:
0. AST-based security parse (tree-sitter too-complex/semantics detection)
1. Sandbox auto-allow
2. Bypass mode
3. Mode validation (acceptEdits)
4. Compound command AST check
5. Single command check (rules, paths, sed, read-only)
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Set

from ripperdoc.utils.bash.commands import split_command, get_command_subcommand_prefix
from ripperdoc.utils.bash.ast import parse_for_security_from_ast, check_semantics
from ripperdoc.utils.bash.parsed_command import ParsedCommand
from ripperdoc.security import PermissionResult, bash_command_is_safe
from ripperdoc.utils.permissions.shell_rule_matching import (
    parse_permission_rule,
    match_wildcard_pattern,
    permission_rule_extract_prefix,
    suggestion_for_exact_command,
    suggestion_for_prefix,
)
from ripperdoc.tools.bash.command_helpers import (
    check_command_operator_permissions,
    is_normalized_cd_command,
    is_normalized_git_command,
)
from ripperdoc.tools.bash.read_only_validation import (
    check_read_only_constraints,
    is_command_read_only,
    command_has_any_git,
)
from ripperdoc.tools.bash.sed_validation import check_sed_constraints
from ripperdoc.tools.bash.mode_validation import check_permission_mode
from ripperdoc.tools.bash.path_validation import check_path_constraints
from ripperdoc.tools.bash.sandbox_decision import should_use_sandbox
from ripperdoc.utils.filesystem.safe_get_cwd import safe_get_cwd
from ripperdoc.utils.shell.sandbox_utils import is_sandbox_available


# ============================================================================
# Constants
# ============================================================================

# Env var assignment pattern
ENV_VAR_ASSIGN_RE = re.compile(r"^[A-Za-z_]\w*=")

# Max subcommands for security check (ReDoS cap)
MAX_SUBCOMMANDS_FOR_SECURITY_CHECK = 50

# Max suggested rules for compound commands
MAX_SUGGESTED_RULES_FOR_COMPOUND = 5

# Safe env vars that are safe to strip from commands.
# These CANNOT execute code or load libraries.
SAFE_ENV_VARS: Set[str] = {
    "GOEXPERIMENT", "GOOS", "GOARCH", "CGO_ENABLED", "GO111MODULE",
    "RUST_BACKTRACE", "RUST_LOG",
    "NODE_ENV",
    "PYTHONUNBUFFERED", "PYTHONDONTWRITEBYTECODE",
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD", "PYTEST_DEBUG",
    "LANG", "LANGUAGE", "LC_ALL", "LC_CTYPE", "LC_TIME", "CHARSET",
    "TERM", "COLORTERM", "NO_COLOR", "FORCE_COLOR", "TZ",
    "LS_COLORS", "LSCOLORS", "GREP_COLOR", "GREP_COLORS", "GCC_COLORS",
    "TIME_STYLE", "BLOCK_SIZE", "BLOCKSIZE",
}

# Shell wrappers that exec their arguments (safe to strip for permission matching)
BARE_SHELL_PREFIXES: Set[str] = {
    "sh", "bash", "zsh", "fish", "csh", "tcsh", "ksh", "dash",
    "cmd", "powershell", "pwsh",
    "env", "xargs",
    "nice", "stdbuf", "nohup", "timeout", "time",
    "sudo", "doas", "pkexec",
}

# Binary hijack env vars
BINARY_HIJACK_VARS: Set[str] = {
    "LD_PRELOAD", "LD_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES",
    "PYTHONPATH", "NODE_PATH", "RUBYLIB", "PERL5LIB",
    "CLASSPATH",
}


# ============================================================================
# Helper functions
# ============================================================================


def get_simple_command_prefix(command: str) -> Optional[str]:
    """Extract a stable command prefix (command + subcommand) from a raw command.

    Skips leading safe env var assignments. Returns None if a non-safe env var
    is encountered, or if the second token doesn't look like a subcommand.

    Examples:
        'git commit -m "msg"' → 'git commit'
        'NODE_ENV=prod npm run build' → 'npm run'
        'ls -la' → None

    Args:
        command: The command string.

    Returns:
        Two-word prefix or None.
    """
    tokens = command.strip().split()
    if len(tokens) < 2:
        return None

    # Skip safe env var assignments
    i = 0
    while i < len(tokens) and ENV_VAR_ASSIGN_RE.match(tokens[i]):
        var_name = tokens[i].split("=")[0]
        if var_name not in SAFE_ENV_VARS:
            return None
        i += 1

    remaining = tokens[i:]
    if len(remaining) < 2:
        return None

    subcmd = remaining[1]
    # Second token must look like a subcommand
    if not re.match(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$", subcmd):
        return None

    return f"{remaining[0]} {remaining[1]}"


def get_first_word_prefix(command: str) -> Optional[str]:
    """Extract the first word alone when getSimpleCommandPrefix declines.

    Args:
        command: The command string.

    Returns:
        First word or None.
    """
    tokens = command.strip().split()
    i = 0
    while i < len(tokens) and ENV_VAR_ASSIGN_RE.match(tokens[i]):
        var_name = tokens[i].split("=")[0]
        if var_name not in SAFE_ENV_VARS:
            return None
        i += 1

    if i >= len(tokens):
        return None

    cmd = tokens[i]
    if not re.match(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$", cmd):
        return None
    if cmd in BARE_SHELL_PREFIXES:
        return None

    return cmd


def strip_comment_lines(command: str) -> str:
    """Remove full-line comments from a command.

    Args:
        command: The command string.

    Returns:
        Command with comment lines removed.
    """
    lines = command.split("\n")
    non_comment = [l for l in lines if l.strip() and not l.strip().startswith("#")]
    if not non_comment:
        return command
    return "\n".join(non_comment)


def strip_safe_wrappers(command: str) -> str:
    """Strip safe wrapper commands (timeout, nice, nohup, stdbuf).


    Args:
        command: The command string.

    Returns:
        Command with wrappers stripped.
    """
    # Security: Use [ \t]+ not \s+ to avoid matching across newlines
    SAFE_WRAPPER_PATTERNS = [
        re.compile(
            r"^timeout[ \t]+"
            r"(?:(?:--(?:foreground|preserve-status|verbose)"
            r"|--(?:kill-after|signal)=[A-Za-z0-9_.+-]+"
            r"|--(?:kill-after|signal)[ \t]+[A-Za-z0-9_.+-]+"
            r"|-v|-[ks][ \t]+[A-Za-z0-9_.+-]+|-[ks][A-Za-z0-9_.+-]+)[ \t]+)*"
            r"(?:--[ \t]+)?\d+(?:\.\d+)?[smhd]?[ \t]+"
        ),
        re.compile(r"^time[ \t]+(?:--[ \t]+)?"),
        re.compile(r"^nice(?:[ \t]+-n[ \t]+-?\d+|[ \t]+-\d+)?[ \t]+(?:--[ \t]+)?"),
        re.compile(r"^stdbuf(?:[ \t]+-[ioe][LN0-9]+)+[ \t]+(?:--[ \t]+)?"),
        re.compile(r"^nohup[ \t]+(?:--[ \t]+)?"),
    ]

    # Env var pattern: VAR=value followed by horizontal whitespace
    ENV_VAR_PATTERN = re.compile(
        r"^([A-Za-z_][A-Za-z0-9_]*)=([A-Za-z0-9_./:-]+)[ \t]+"
    )

    stripped = command

    # Phase 1: Strip leading env vars and comments
    changed = True
    while changed:
        changed = False
        stripped = strip_comment_lines(stripped)

        m = ENV_VAR_PATTERN.match(stripped)
        if m and m.group(1) in SAFE_ENV_VARS:
            stripped = stripped[m.end():]
            changed = True

    # Phase 2: Strip wrapper commands
    for pattern in SAFE_WRAPPER_PATTERNS:
        m = pattern.match(stripped)
        if m:
            stripped = stripped[m.end():]
            break

    return stripped


def strip_all_leading_env_vars(command: str, hijack_vars: Set[str]) -> str:
    """Strip all leading env var assignments from a command.

    Unlike strip_safe_wrappers, this strips ALL env vars including
    dangerous ones, for the purpose of matching against deny rules.

    Args:
        command: The command string.
        hijack_vars: Set of variable names to consider as hijack risks.

    Returns:
        Command with leading env vars stripped.
    """
    # Strip VAR=value patterns at the start
    result = command
    while True:
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=([^ \t]+)[ \t]+", result)
        if m:
            result = result[m.end():]
        else:
            break
    return result


# ============================================================================
# Rule matching helpers
# ============================================================================


def _match_rules(command: str, rules: Set[str]) -> bool:
    """Check if a command matches any of the given rules.

    Args:
        command: The command to check.
        rules: Set of rule strings.

    Returns:
        True if the command matches any rule.
    """
    for rule in rules:
        parsed = parse_permission_rule(rule)
        if parsed.type == "exact":
            if command == parsed.command:
                return True
        elif parsed.type == "prefix":
            if command == parsed.prefix or command.startswith(parsed.prefix + " "):
                return True
        elif parsed.type == "wildcard":
            if match_wildcard_pattern(parsed.pattern, command):
                return True
    return False


def _collect_suggestions(command: str) -> List[Dict[str, str]]:
    """Collect permission rule suggestions for a command.

    Args:
        command: The command string.

    Returns:
        List of suggestion dicts.
    """
    suggestions: List[Dict[str, str]] = [
        {"tool_name": "Bash", "rule_content": command, "rule_type": "allow"},
    ]

    # Multi-word prefix
    prefix = get_simple_command_prefix(command)
    if prefix:
        suggestions.append({
            "tool_name": "Bash",
            "rule_content": f"{prefix}:*",
            "rule_type": "allow",
        })

    # Single-word prefix
    first = get_first_word_prefix(command)
    if first:
        suggestions.append({
            "tool_name": "Bash",
            "rule_content": f"{first}:*",
            "rule_type": "allow",
        })

    return suggestions


# ============================================================================
# Main permission pipeline
# ============================================================================


def _normalized_for_rule_matching(command: str) -> str:
    return strip_all_leading_env_vars(
        strip_safe_wrappers(strip_comment_lines(command)),
        BINARY_HIJACK_VARS,
    )


def _apply_dont_ask(result: PermissionResult, mode: str, command: str) -> PermissionResult:
    if mode == "dontAsk" and result.behavior == "ask":
        return PermissionResult.deny(
            f"Permission denied for Bash command because permission mode is dontAsk: {command}",
            reason={"type": "mode", "mode": "dontAsk"},
        )
    return result


def _check_early_exit_deny(
    command: str,
    denied_rules: Set[str],
) -> Optional[PermissionResult]:
    """Enforce deny rules before early-ask on too-complex/unsafe commands.

    Checks exact match first, then prefix/wildcard. Returns None if no
    deny rule matched (caller should proceed with ask).
    """
    if not denied_rules:
        return None

    if command in denied_rules:
        return PermissionResult.deny(
            f"Permission to run '{command}' has been denied.",
            reason={"type": "rule"},
        )

    if _match_rules(_normalized_for_rule_matching(command), denied_rules):
        return PermissionResult.deny(
            f"Permission to run '{command}' has been denied.",
            reason={"type": "rule"},
        )

    return None


def _check_sandbox_auto_allow(
    command: str,
    *,
    input_data: Any,
    mode: str,
    denied_rules: Set[str],
    ask_rules: Set[str],
    allowed_dirs: Set[str],
    cwd: str,
) -> PermissionResult:
    sandbox_requested = getattr(input_data, "sandbox", False) and not getattr(
        input_data, "dangerously_disable_sandbox", False
    )
    if not sandbox_requested:
        return PermissionResult.passthrough("Sandbox not requested")

    normalized = _normalized_for_rule_matching(command)
    if command in denied_rules or _match_rules(normalized, denied_rules):
        return PermissionResult.deny(
            f"Permission to run '{command}' has been denied.",
            reason={"type": "rule"},
        )
    if command in ask_rules or _match_rules(normalized, ask_rules):
        return _apply_dont_ask(
            PermissionResult.ask(
                "Command requires confirmation by rule.",
                reason={"type": "rule"},
            ),
            mode,
            command,
        )

    path_result = check_path_constraints(command, cwd, allowed_dirs)
    if path_result.behavior == "ask":
        return _apply_dont_ask(path_result, mode, command)

    return PermissionResult.allow(
        updated_input=input_data,
        reason={"type": "sandbox"},
    )


def _check_single_command(
    command: str,
    *,
    mode: str,
    allowed_rules: Set[str],
    denied_rules: Set[str],
    ask_rules: Set[str],
    allowed_dirs: Set[str],
    cwd: str,
) -> PermissionResult:
    """Check permissions for a single (non-compound) command.

    Runs rules matching, path/sed/read-only constraints, and returns a decision.
    This function does NOT recurse into compound command checking.

    Args:
        command: The single command string to check.
        mode: Permission mode string.
        allowed_rules: Set of allow rule strings.
        denied_rules: Set of deny rule strings.
        ask_rules: Set of ask rule strings.
        allowed_dirs: Set of allowed directory paths.
        cwd: Current working directory.

    Returns:
        PermissionResult.
    """
    # Exact match check (deny > ask > allow)
    if denied_rules and command in denied_rules:
        return PermissionResult.deny(
            f"Permission to run '{command}' has been denied.",
            reason={"type": "rule"},
        )
    if ask_rules and command in ask_rules:
        return PermissionResult.ask(
            "Command requires confirmation by rule.",
            reason={"type": "rule"},
        )
    exact_allowed = command in allowed_rules if allowed_rules else False

    fully_stripped = _normalized_for_rule_matching(command)

    # Prefix/wildcard deny rules
    if denied_rules and _match_rules(fully_stripped, denied_rules):
        return PermissionResult.deny(
            f"Permission to run '{command}' has been denied.",
            reason={"type": "rule"},
        )

    # Prefix/wildcard ask rules
    if ask_rules and _match_rules(fully_stripped, ask_rules):
        return PermissionResult.ask(
            "Command requires confirmation by rule.",
            reason={"type": "rule"},
        )

    # Path constraints
    path_result = check_path_constraints(command, cwd, allowed_dirs)
    if path_result.behavior == "ask":
        return path_result

    # Exact match allow
    if exact_allowed:
        return PermissionResult.allow(reason={"type": "rule"})

    # Prefix/wildcard allow rules
    if allowed_rules and _match_rules(fully_stripped, allowed_rules):
        return PermissionResult.allow(reason={"type": "rule"})

    # Sed constraints
    sed_result = check_sed_constraints(command, mode)
    if sed_result.behavior == "ask":
        return sed_result

    # Read-only constraints
    compound_has_cd = any(
        is_normalized_cd_command(sub.strip())
        for sub in split_command(command)
    )
    read_only_result = check_read_only_constraints(command, compound_has_cd)
    if read_only_result.behavior == "allow":
        return read_only_result

    # Default: ask with suggestions
    return PermissionResult.ask(
        "Command requires permission",
        reason={"type": "default"},
    )


async def bash_tool_has_permission(
    input_data: Any,
    permission_context: Dict[str, Any],
    command_override: Optional[str] = None,
) -> PermissionResult:
    """Main permission checking pipeline for bash commands.


    Args:
        input_data: The tool input (must have .command attribute).
        permission_context: Context dict with:
            - 'mode': Permission mode string.
            - 'allowed_rules': Set of allow rule strings.
            - 'denied_rules': Set of deny rule strings.
            - 'ask_rules': Set of ask rule strings.
            - 'allowed_working_directories': Set of allowed directory paths.
        command_override: If set, check this command instead of input_data.command.

    Returns:
        PermissionResult.
    """
    command = command_override if command_override else getattr(input_data, "command", "")
    if not command:
        return PermissionResult.allow(
            updated_input=input_data,
            reason={"type": "other", "reason": "Empty command"},
        )

    mode = permission_context.get("mode", "default")
    allowed_rules: Set[str] = set(permission_context.get("allowed_rules") or [])
    denied_rules: Set[str] = set(permission_context.get("denied_rules") or [])
    ask_rules: Set[str] = set(permission_context.get("ask_rules") or [])
    allowed_dirs: Set[str] = set(permission_context.get(
        "allowed_working_directories", [safe_get_cwd()]
    ))
    cwd = safe_get_cwd()

    # ======================================================================
    # Step 0: AST-based security parse (tree-sitter)
    #
    # tree-sitter produces either a clean SimpleCommand[] (quotes resolved,
    # no hidden substitutions) or 'too-complex' — which tells us whether
    # splitCommand's output can be trusted.
    # ======================================================================
    ast_root: Any = None
    ast_parse_succeeded: bool = False

    from ripperdoc.utils.bash.shell_quote import try_parse_shell_command

    ast_result = parse_for_security_from_ast(command)

    if ast_result["kind"] == "too-complex":
        # Parse succeeded but found structure we can't statically analyze
        # (command substitution, expansion, control flow).
        # Respect exact-match deny/ask, then prefix/wildcard deny.
        # Only fall through to ask if no deny matched.
        early_exit = _check_early_exit_deny(command, denied_rules)
        if early_exit is not None:
            return early_exit

        return _apply_dont_ask(
            PermissionResult.ask(
                ast_result["reason"],
                reason={"type": "too_complex", "reason": ast_result["reason"]},
            ),
            mode,
            command,
        )

    if ast_result["kind"] == "simple":
        ast_result = check_semantics(command, ast_result)
        ast_parse_succeeded = True
        ast_commands = ast_result.get("commands", [])

    if ast_result["kind"] == "parse-unavailable":
        # Legacy shell-quote pre-check (tree-sitter unavailable)
        parse_result = try_parse_shell_command(command)
        if not parse_result.success:
            return _apply_dont_ask(
                PermissionResult.ask(
                    f"Command contains malformed syntax that cannot be parsed: {parse_result.error or 'unknown error'}",
                    reason={"type": "parse_error"},
                ),
                mode,
                command,
            )

    # ======================================================================
    # Step 1: Sandbox auto-allow
    # ======================================================================
    sandbox_result = _check_sandbox_auto_allow(
        command,
        input_data=input_data,
        mode=mode,
        denied_rules=denied_rules,
        ask_rules=ask_rules,
        allowed_dirs=allowed_dirs,
        cwd=cwd,
    )
    if sandbox_result.behavior != "passthrough":
        return sandbox_result

    # ======================================================================
    # Step 2: Bypass mode (handled upstream)
    # ======================================================================
    if mode == "bypassPermissions":
        return PermissionResult.allow(
            updated_input=input_data,
            reason={"type": "mode", "mode": mode},
        )

    # ======================================================================
    # Step 3: Mode validation (acceptEdits auto-allow for fs commands)
    # ======================================================================
    mode_result = check_permission_mode(command, mode)
    if mode_result.behavior != "passthrough":
        return _apply_dont_ask(mode_result, mode, command)

    # ======================================================================
    # Step 4: Compound command permissions (pipes, subshells)
    # Each segment is checked via _check_single_command — no recursion.
    # ======================================================================
    checkers = (is_normalized_cd_command, is_normalized_git_command)
    compound_result = await check_command_operator_permissions(
        input_data,
        checkers,
        single_command_checker=lambda cmd: _check_single_command(
            cmd,
            mode=mode,
            allowed_rules=allowed_rules,
            denied_rules=denied_rules,
            ask_rules=ask_rules,
            allowed_dirs=allowed_dirs,
            cwd=cwd,
        ),
    )
    if compound_result.behavior != "passthrough":
        if compound_result.behavior == "allow":
            path_result = check_path_constraints(command, cwd, allowed_dirs)
            if path_result.behavior == "ask":
                return _apply_dont_ask(path_result, mode, command)
        return _apply_dont_ask(compound_result, mode, command)

    # ======================================================================
    # Step 5: Single command check (rules, paths, sed, read-only)
    # ======================================================================
    single_result = _check_single_command(
        command,
        mode=mode,
        allowed_rules=allowed_rules,
        denied_rules=denied_rules,
        ask_rules=ask_rules,
        allowed_dirs=allowed_dirs,
        cwd=cwd,
    )
    return _apply_dont_ask(single_result, mode, command)


__all__ = [
    "bash_tool_has_permission",
    "get_simple_command_prefix",
    "get_first_word_prefix",
    "strip_safe_wrappers",
    "strip_all_leading_env_vars",
    "strip_comment_lines",
    "SAFE_ENV_VARS",
    "BARE_SHELL_PREFIXES",
    "BINARY_HIJACK_VARS",
]
