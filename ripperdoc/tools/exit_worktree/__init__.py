"""ExitWorktree tool for leaving a worktree session."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import AsyncGenerator, Optional

from pydantic import BaseModel, Field

from ripperdoc.core.tool import Tool, ToolOutput, ToolResult, ToolUseContext, ValidationResult
from ripperdoc.utils.collaboration.worktree import (
    WorktreeSession,
    cleanup_worktree_session,
    list_session_worktrees,
    unregister_session_worktree,
)
from ripperdoc.utils.log import get_logger

logger = get_logger()

TOOL_NAME = "ExitWorktree"


class ExitWorktreeToolInput(BaseModel):
    """Input for ExitWorktree."""

    action: str = Field(
        default="keep",
        description="Whether to 'keep' or 'remove' the worktree.",
    )
    discard_changes: bool = Field(
        default=False,
        description="Only meaningful when action='remove'. If true, discard uncommitted changes.",
    )


class ExitWorktreeToolOutput(BaseModel):
    """Output for ExitWorktree."""

    original_path: str
    worktree_removed: bool = False
    message: str


def _find_current_worktree() -> Optional[WorktreeSession]:
    cwd = Path.cwd().resolve()
    for session in list_session_worktrees():
        session_path = session.worktree_path.resolve()
        if cwd == session_path or session_path in cwd.parents:
            return session
    return None


async def _count_uncommitted_files(worktree_path: Path) -> int:
    """Count uncommitted files (modified + untracked) in the worktree."""
    try:
        process = await asyncio.create_subprocess_exec(
            "git", "status", "--porcelain",
            cwd=str(worktree_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await process.communicate()
        if process.returncode == 0:
            lines = [l for l in stdout.decode("utf-8", errors="replace").split("\n") if l.strip()]
            return len(lines)
    except (OSError, FileNotFoundError):
        pass
    return 0


async def _kill_tmux_session(worktree_path: Path) -> bool:
    """Kill any tmux session associated with this worktree."""
    try:
        # List tmux sessions and find one matching the worktree path
        process = await asyncio.create_subprocess_exec(
            "tmux", "list-sessions", "-F", "#{session_name}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await process.communicate()
        if process.returncode != 0:
            return False

        session_name = str(worktree_path).replace("/", "_").strip("_")
        for line in stdout.decode("utf-8", errors="replace").split("\n"):
            name = line.strip()
            if name and (session_name in name or name in session_name):
                kill_process = await asyncio.create_subprocess_exec(
                    "tmux", "kill-session", "-t", name,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await kill_process.communicate()
                return True
    except (OSError, FileNotFoundError):
        pass
    return False


class ExitWorktreeTool(Tool[ExitWorktreeToolInput, ExitWorktreeToolOutput]):
    """Leave a worktree session and optionally clean up the worktree."""

    @property
    def name(self) -> str:
        return TOOL_NAME

    async def description(self) -> str:
        return "Exits the current worktree session, returning to the original directory. Optionally removes the worktree."

    @property
    def input_schema(self) -> type[ExitWorktreeToolInput]:
        return ExitWorktreeToolInput

    async def prompt(self, yolo_mode: bool = False) -> str:  # noqa: ARG002
        return (
            "Use this tool ONLY when the user explicitly asks to exit a worktree, "
            "or when you want to leave a worktree session. "
            "Supports two actions: 'keep' (leave worktree in place) and 'remove' (delete worktree). "
            "The 'remove' action will fail if there are uncommitted changes unless discard_changes=true. "
            "After exit, the session returns to the original working directory."
        )

    def needs_permissions(self, _input_data: Optional[ExitWorktreeToolInput] = None) -> bool:
        return False

    async def validate_input(
        self,
        input_data: ExitWorktreeToolInput,
        _context: Optional[ToolUseContext] = None,
    ) -> ValidationResult:
        if input_data.action not in ("keep", "remove"):
            return ValidationResult(
                result=False,
                message="action must be 'keep' or 'remove'",
            )
        return ValidationResult(result=True)

    def render_result_for_assistant(self, output: ExitWorktreeToolOutput) -> str:
        return output.message

    def render_tool_use_message(
        self, input_data: ExitWorktreeToolInput, _verbose: bool = False
    ) -> str:
        return f"Exiting worktree (action={input_data.action})"

    async def call(
        self,
        input_data: ExitWorktreeToolInput,
        context: ToolUseContext,
    ) -> AsyncGenerator[ToolOutput, None]:
        session = _find_current_worktree()
        if session is None:
            raise ValueError(
                "Not currently in a worktree session. "
                "Use EnterWorktree to create one first."
            )

        worktree_removed = False
        if input_data.action == "remove":
            force = input_data.discard_changes

            # Count uncommitted files before removal
            if not force:
                uncommitted = await _count_uncommitted_files(session.worktree_path)
                if uncommitted > 0:
                    raise ValueError(
                        f"Worktree has {uncommitted} uncommitted file(s). "
                        "Set discard_changes=true to force removal, or use action='keep'."
                    )

            # Kill associated tmux session
            await _kill_tmux_session(session.worktree_path)

            cleanup = cleanup_worktree_session(session, force=force)
            if cleanup.error and not force:
                raise ValueError(
                    f"Cannot remove worktree: {cleanup.error}. "
                    "Set discard_changes=true to force removal."
                )
            worktree_removed = cleanup.removed
            if cleanup.removed:
                unregister_session_worktree(session.worktree_path)

            message = (
                f"Exited and removed worktree at {session.worktree_path}"
                if worktree_removed
                else f"Exited worktree but removal failed: {cleanup.error or 'unknown'}"
            )
        else:
            message = f"Exited worktree session (kept at {session.worktree_path})"

        original_path = str(
            session.repo_root.resolve() if session.repo_root else Path.cwd()
        )
        output = ExitWorktreeToolOutput(
            original_path=original_path,
            worktree_removed=worktree_removed,
            message=message,
        )
        yield ToolResult(
            data=output,
            result_for_assistant=self.render_result_for_assistant(output),
        )
