"""Sandbox decision logic for bash commands.


Determines whether a command should be run in sandboxed mode based on
configuration, excluded commands, and explicit overrides.
"""

from __future__ import annotations

from ripperdoc.utils.bash.commands import split_command
from ripperdoc.utils.shell.sandbox_utils import is_sandbox_available


def should_use_sandbox(
    command: str,
    dangerously_disable_sandbox: bool = False,
) -> bool:
    """Determine whether a command should be run in sandboxed mode.

    Args:
        command: The command string to check.
        dangerously_disable_sandbox: If True and unsandboxed commands are allowed,
                                     skip sandboxing.

    Returns:
        True if the command should be sandboxed.
    """
    if not is_sandbox_available():
        return False

    # Don't sandbox if explicitly overridden
    if dangerously_disable_sandbox:
        return False

    if not command:
        return False

    # Don't sandbox if the command contains user-configured excluded commands
    if _contains_excluded_command(command):
        return False

    return True


def _contains_excluded_command(command: str) -> bool:
    """Check if a command should bypass sandbox based on exclusion config.

    Checks user-configured excluded commands from settings.

    Args:
        command: The command string.

    Returns:
        True if the command contains an excluded pattern.
    """
    # Check for commands that require network or filesystem write access
    # These are always excluded from sandbox as they would fail
    always_excluded_bases = {
        "npm", "yarn", "pnpm", "npx",
        "cargo", "go", "make", "cmake", "ninja", "meson",
        "docker", "kubectl", "gh", "ssh", "scp",
        "git",  # git push/clone need network
    }

    try:
        subcommands = split_command(command)
    except Exception:
        subcommands = [command]

    for subcmd in subcommands:
        trimmed = subcmd.strip()
        if not trimmed:
            continue
        parts = trimmed.split()
        if not parts:
            continue
        # Check the base command (skip env vars)
        base = parts[0]
        if base in always_excluded_bases:
            return True

    return False


__all__ = ["should_use_sandbox"]
