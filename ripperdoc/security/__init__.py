"""Security detection package.

"""

from .bash_security import (
    PermissionResult,
    bash_command_is_safe,
    bash_command_is_safe_async,
    strip_safe_heredoc_substitutions,
    has_safe_heredoc_substitution,
    extract_quoted_content,
    strip_safe_redirections,
    COMMAND_SUBSTITUTION_PATTERNS,
    ZSH_DANGEROUS_COMMANDS,
)

__all__ = [
    "PermissionResult",
    "bash_command_is_safe",
    "bash_command_is_safe_async",
    "strip_safe_heredoc_substitutions",
    "has_safe_heredoc_substitution",
    "extract_quoted_content",
    "strip_safe_redirections",
    "COMMAND_SUBSTITUTION_PATTERNS",
    "ZSH_DANGEROUS_COMMANDS",
]
