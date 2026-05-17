"""Permission utilities."""

from .tool_permission_utils import (
    PermissionDecision,
    ToolRule,
    create_wildcard_rule,
    create_tool_rule,
    create_wildcard_tool_rule,
    match_rule,
)
from .shell_rule_matching import (
    ShellPermissionRule,
    PermissionUpdate,
    parse_permission_rule,
    match_wildcard_pattern,
    permission_rule_extract_prefix,
    suggestion_for_exact_command,
    suggestion_for_prefix,
)
from .read_only_command_validation import (
    CommandConfig,
    FLAG_ARG_NONE,
    FLAG_ARG_NUMBER,
    FLAG_ARG_STRING,
    FLAG_ARG_CHAR,
    FLAG_ARG_BRACES,
    FLAG_ARG_EOF,
    contains_vulnerable_unc_path,
    validate_flags,
    GIT_READ_ONLY_COMMANDS,
    DOCKER_READ_ONLY_COMMANDS,
    RIPGREP_READ_ONLY_COMMANDS,
    PYRIGHT_READ_ONLY_COMMANDS,
    EXTERNAL_READONLY_COMMANDS,
)

__all__ = [
    "PermissionDecision",
    "ToolRule",
    "create_wildcard_rule",
    "create_tool_rule",
    "create_wildcard_tool_rule",
    "match_rule",
    # Shell rule matching
    "ShellPermissionRule",
    "PermissionUpdate",
    "parse_permission_rule",
    "match_wildcard_pattern",
    "permission_rule_extract_prefix",
    "suggestion_for_exact_command",
    "suggestion_for_prefix",
    # Read-only command validation
    "CommandConfig",
    "FLAG_ARG_NONE",
    "FLAG_ARG_NUMBER",
    "FLAG_ARG_STRING",
    "FLAG_ARG_CHAR",
    "FLAG_ARG_BRACES",
    "FLAG_ARG_EOF",
    "contains_vulnerable_unc_path",
    "validate_flags",
    "GIT_READ_ONLY_COMMANDS",
    "DOCKER_READ_ONLY_COMMANDS",
    "RIPGREP_READ_ONLY_COMMANDS",
    "PYRIGHT_READ_ONLY_COMMANDS",
    "EXTERNAL_READONLY_COMMANDS",
]
