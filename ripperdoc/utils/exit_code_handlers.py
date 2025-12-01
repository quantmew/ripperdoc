"""Smart exit code handlers for common shell commands and related helpers.

Provides intelligent interpretation of exit codes for commands like grep, diff, test, etc.
where non-zero exit codes don't necessarily indicate errors. Also includes small utilities
shared by bash tooling such as command classification, preview sizing, and lightweight
command/result schemas.
"""

import shlex
from typing import Callable, Optional
from pydantic import BaseModel, Field
from dataclasses import dataclass


@dataclass
class ExitCodeResult:
    """Result of exit code interpretation."""

    is_error: bool
    message: Optional[str] = None
    semantic_meaning: Optional[str] = None


ExitCodeHandler = Callable[[int, str, str], ExitCodeResult]


# Default/max timeouts exposed for bash tooling (keep aligned with BashTool).
DEFAULT_BASH_TIMEOUT_MS = 120000
MAX_BASH_TIMEOUT_MS = 600000

# Commands we intentionally ignore in certain contexts (e.g., background-safety checks).
IGNORED_COMMANDS: tuple[str, ...] = ("sleep",)

# Preview limits for rendering long commands compactly.
MAX_PREVIEW_LINES = 2
MAX_PREVIEW_CHARS = 160

# Heuristic command classification list (mirrors the reference set).
COMMON_COMMANDS: tuple[str, ...] = (
    "npm",
    "yarn",
    "pnpm",
    "node",
    "python",
    "python3",
    "go",
    "cargo",
    "make",
    "docker",
    "terraform",
    "webpack",
    "vite",
    "jest",
    "pytest",
    "curl",
    "wget",
    "build",
    "test",
    "serve",
    "watch",
    "dev",
)


class BashCommandSchema(BaseModel):
    """Schema describing a bash command request."""

    command: str = Field(description="The command to execute")
    timeout: Optional[int] = Field(
        default=None, description=f"Optional timeout in milliseconds (max {MAX_BASH_TIMEOUT_MS})"
    )
    description: Optional[str] = Field(
        default=None,
        description="Clear, concise description of what this command does in 5-10 words.",
    )
    run_in_background: bool = Field(
        default=False, description="Set to true to run this command in the background."
    )


class ExtendedBashCommandSchema(BashCommandSchema):
    """Schema describing an extended bash command request."""

    sandbox: Optional[bool] = Field(
        default=None, description="Whether to request sandboxed execution (read-only)."
    )
    shell_executable: Optional[str] = Field(
        default=None, description="Optional shell path to use instead of the default shell."
    )


class CommandResultSchema(BaseModel):
    """Schema describing the shape of a command result."""

    stdout: str = Field(description="The standard output of the command")
    stderr: str = Field(description="The standard error output of the command")
    summary: Optional[str] = Field(default=None, description="Summarized output when available")
    interrupted: bool = Field(default=False, description="Whether the command was interrupted")
    is_image: Optional[bool] = Field(
        default=None, description="Flag to indicate if stdout contains image data"
    )
    background_task_id: Optional[str] = Field(
        default=None, description="ID of the background task if command is running in background"
    )
    sandbox: Optional[bool] = Field(
        default=None, description="Flag to indicate if the command was run in sandbox mode"
    )
    return_code_interpretation: Optional[str] = Field(
        default=None,
        description="Semantic interpretation for non-error exit codes with special meaning",
    )


def default_handler(exit_code: int, stdout: str, stderr: str) -> ExitCodeResult:
    """Default exit code handler - non-zero is error."""
    return ExitCodeResult(
        is_error=exit_code != 0,
        message=f"Command failed with exit code {exit_code}" if exit_code != 0 else None,
    )


def grep_handler(exit_code: int, stdout: str, stderr: str) -> ExitCodeResult:
    """Handle grep/rg exit codes: 0=found, 1=not found, 2+=error."""
    if exit_code == 0:
        return ExitCodeResult(is_error=False)
    elif exit_code == 1:
        return ExitCodeResult(is_error=False, semantic_meaning="No matches found")
    else:
        return ExitCodeResult(is_error=True, message=f"grep failed with exit code {exit_code}")


def diff_handler(exit_code: int, stdout: str, stderr: str) -> ExitCodeResult:
    """Handle diff exit codes: 0=same, 1=different, 2+=error."""
    if exit_code == 0:
        return ExitCodeResult(is_error=False, semantic_meaning="Files are identical")
    elif exit_code == 1:
        return ExitCodeResult(is_error=False, semantic_meaning="Files differ")
    else:
        return ExitCodeResult(is_error=True, message=f"diff failed with exit code {exit_code}")


def test_handler(exit_code: int, stdout: str, stderr: str) -> ExitCodeResult:
    """Handle test/[ exit codes: 0=true, 1=false, 2+=error."""
    if exit_code == 0:
        return ExitCodeResult(is_error=False, semantic_meaning="Condition is true")
    elif exit_code == 1:
        return ExitCodeResult(is_error=False, semantic_meaning="Condition is false")
    else:
        return ExitCodeResult(
            is_error=True, message=f"test command failed with exit code {exit_code}"
        )


def find_handler(exit_code: int, stdout: str, stderr: str) -> ExitCodeResult:
    """Handle find exit codes: 0=ok, 1=partial, 2+=error."""
    if exit_code == 0:
        return ExitCodeResult(is_error=False)
    elif exit_code == 1:
        return ExitCodeResult(is_error=False, semantic_meaning="Some directories were inaccessible")
    else:
        return ExitCodeResult(is_error=True, message=f"find failed with exit code {exit_code}")


# Command-specific handlers
COMMAND_HANDLERS: dict[str, ExitCodeHandler] = {
    "grep": grep_handler,
    "rg": grep_handler,
    "ripgrep": grep_handler,
    "diff": diff_handler,
    "test": test_handler,
    "[": test_handler,
    "find": find_handler,
}


def normalize_command(command: str) -> str:
    """Extract the base command from a command string.

    Handles pipes, command chains, and extracts the final command.
    Examples:
        'git status' -> 'git'
        'cat file | grep pattern' -> 'grep'
        'ls -la' -> 'ls'
    """
    # Get the last command in a pipe chain
    if "|" in command:
        command = command.split("|")[-1].strip()

    # Get the first word (the actual command)
    command = command.strip().split()[0] if command.strip() else ""

    return command


def classify_command(command: str) -> str:
    """Classify a shell command into a known category or 'other'."""
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()

    if not tokens:
        return "other"

    for token in tokens:
        cleaned = token.strip()
        if not cleaned or cleaned in {"&&", "||", ";", "|"}:
            continue

        first_word = cleaned.split()[0].lower()
        if first_word in COMMON_COMMANDS:
            return first_word

    return "other"


def get_exit_code_handler(command: str) -> ExitCodeHandler:
    """Get the appropriate exit code handler for a command."""
    normalized = normalize_command(command)
    return COMMAND_HANDLERS.get(normalized, default_handler)


def interpret_exit_code(command: str, exit_code: int, stdout: str, stderr: str) -> ExitCodeResult:
    """Interpret an exit code in the context of the command.

    Args:
        command: The shell command that was executed
        exit_code: The exit code returned
        stdout: Standard output from the command
        stderr: Standard error from the command

    Returns:
        ExitCodeResult with interpretation
    """
    handler = get_exit_code_handler(command)
    return handler(exit_code, stdout, stderr)


def create_exit_result(command: str, exit_code: int, stdout: str, stderr: str) -> ExitCodeResult:
    """Convenience wrapper to mirror reference API naming."""
    return interpret_exit_code(command, exit_code, stdout, stderr)
