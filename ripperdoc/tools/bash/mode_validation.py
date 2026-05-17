"""Permission mode validation for bash commands.


Handles mode-based auto-approval, e.g., Accept Edits mode for
filesystem operations.
"""

from __future__ import annotations

from typing import List

from ripperdoc.utils.bash.commands import split_command
from ripperdoc.security import PermissionResult


# Commands that are auto-allowed in acceptEdits mode
ACCEPT_EDITS_ALLOWED_COMMANDS = [
    "mkdir",
    "touch",
    "rm",
    "rmdir",
    "mv",
    "cp",
    "sed",
]


def check_permission_mode(
    command: str,
    mode: str,
) -> PermissionResult:
    """Check if commands should be handled differently based on permission mode.

    Args:
        command: The bash command to check.
        mode: The current permission mode ('bypassPermissions', 'acceptEdits', etc.).

    Returns:
        - 'allow' if the current mode permits auto-approval
        - 'ask' if the command needs approval in current mode
        - 'passthrough' if no mode-specific handling applies
    """
    # Skip if in bypass mode (handled elsewhere)
    if mode == "bypassPermissions":
        return PermissionResult.passthrough("Bypass mode is handled in main permission flow")

    if mode == "dontAsk":
        return PermissionResult.passthrough("DontAsk mode is handled in main permission flow")

    # In acceptEdits mode, auto-allow filesystem operations
    if mode == "acceptEdits":
        subcommands = split_command(command)
        for cmd in subcommands:
            trimmed = cmd.strip()
            base_cmd = trimmed.split()[0] if trimmed.split() else ""
            if base_cmd in ACCEPT_EDITS_ALLOWED_COMMANDS:
                return PermissionResult.allow(
                    updated_input={"command": cmd},
                    reason={"type": "mode", "mode": "acceptEdits"},
                )

    return PermissionResult.passthrough("No mode-specific validation required")


def get_auto_allowed_commands(mode: str) -> List[str]:
    """Get the list of commands that are auto-allowed in the given mode.

    Args:
        mode: The permission mode.

    Returns:
        List of command names that are auto-allowed.
    """
    if mode == "acceptEdits":
        return list(ACCEPT_EDITS_ALLOWED_COMMANDS)
    return []


__all__ = [
    "check_permission_mode",
    "get_auto_allowed_commands",
    "ACCEPT_EDITS_ALLOWED_COMMANDS",
]
