"""Permissions bridge for Bash tool.

Thin wrapper that connects the existing BashTool to the new
permission pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from ripperdoc.tools.bash._models import BashToolInput
from ripperdoc.tools.bash.permissions import bash_tool_has_permission
from ripperdoc.tools.bash.read_only_validation import is_command_read_only
from ripperdoc.security import PermissionResult


def is_background_allowed(command: str) -> bool:
    """Skip backgrounding trivial ignored commands unless combined with other operators."""
    from ripperdoc.utils.shell.exit_code_handlers import IGNORED_COMMANDS

    normalized = command.strip()
    if not normalized:
        return True

    if any(op in normalized for op in ("&&", "||", "|", ";")):
        return True

    parts = normalized.split(maxsplit=1)
    if normalized in IGNORED_COMMANDS:
        return False
    if len(parts) == 1 and parts[0] in IGNORED_COMMANDS:
        return False
    return True


def detect_auto_background(command: str) -> Tuple[str, bool]:
    """Detect trailing '&' requests and strip them for execution."""
    stripped = command.rstrip()
    if not stripped:
        return command, False

    if stripped.endswith("&") and not stripped.endswith("&&"):
        cleaned = stripped.rstrip("&").rstrip()
        return cleaned, True

    return command, False


async def check_permissions(
    input_data: BashToolInput,
    permission_context: Dict[str, Any],
) -> PermissionResult:
    """Evaluate permissions using the permission pipeline.

    Args:
        input_data: The bash tool input.
        permission_context: Context with rules, mode, etc.

    Returns:
        PermissionResult from the full bash_tool_has_permission pipeline.
    """
    return await bash_tool_has_permission(input_data, permission_context)


__all__ = [
    "check_permissions",
    "is_background_allowed",
    "detect_auto_background",
]
