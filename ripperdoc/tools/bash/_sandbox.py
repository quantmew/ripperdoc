"""Sandbox management for Bash tool."""

from __future__ import annotations

from typing import Any, Optional, Tuple

from ripperdoc.utils.log import get_logger
from ripperdoc.utils.shell.sandbox_utils import create_sandbox_wrapper, is_sandbox_available
from ripperdoc.tools.bash._models import BashToolOutput

logger = get_logger()


def _create_error_output(command: str, stderr: str, sandbox: bool) -> BashToolOutput:
    """Create a standardized error output."""
    return BashToolOutput(
        stdout="",
        stderr=stderr,
        exit_code=-1,
        command=command,
        sandbox=sandbox,
        is_error=True,
    )


def setup_sandbox(
    command: str, sandbox_requested: bool
) -> Tuple[Optional[str], Optional[BashToolOutput], Optional[Any]]:
    """Setup sandbox environment if requested.

    Returns:
        Tuple of (final_command, error_output, cleanup_fn).
        If error_output is not None, sandbox setup failed.
    """
    if not sandbox_requested:
        return command, None, None

    if not is_sandbox_available():
        return (
            None,
            _create_error_output(
                command, "Sandbox mode requested but not available on this system", True
            ),
            None,
        )

    try:
        wrapper = create_sandbox_wrapper(command)
        return wrapper.final_command, None, wrapper.cleanup
    except (OSError, RuntimeError, ValueError) as exc:
        logger.warning(
            "[bash_tool] Failed to enable sandbox: %s: %s",
            type(exc).__name__,
            exc,
            extra={"command": command},
        )
        return (
            None,
            _create_error_output(command, f"Failed to enable sandbox: {exc}", True),
            None,
        )
