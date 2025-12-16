"""Permission utilities."""

from .path_validation_utils import validate_shell_command_paths
from .shell_command_validation import (
    validate_shell_command,
    is_complex_unsafe_shell_command,
    ValidationResult,
)
from .tool_permission_utils import (
    PermissionDecision,
    ToolRule,
    evaluate_shell_command_permissions,
    extract_rule_prefix,
    match_rule,
)

__all__ = [
    "PermissionDecision",
    "ToolRule",
    "ValidationResult",
    "evaluate_shell_command_permissions",
    "extract_rule_prefix",
    "is_complex_unsafe_shell_command",
    "match_rule",
    "validate_shell_command_paths",
    "validate_shell_command",
]
