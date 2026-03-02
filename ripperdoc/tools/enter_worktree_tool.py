"""EnterWorktree tool for mid-session worktree isolation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import AsyncGenerator, Optional

from pydantic import BaseModel, Field

from ripperdoc.core.tool import Tool, ToolOutput, ToolResult, ToolUseContext, ValidationResult
from ripperdoc.utils.worktree import (
    create_task_worktree,
    generate_session_worktree_name,
    list_session_worktrees,
)

TOOL_NAME = "EnterWorktree"


class EnterWorktreeToolInput(BaseModel):
    """Input for EnterWorktree."""

    name: Optional[str] = Field(
        default=None,
        description="Optional name for the worktree. A random name is generated if not provided.",
    )


class EnterWorktreeToolOutput(BaseModel):
    """Output for EnterWorktree."""

    worktreePath: str
    worktreeBranch: Optional[str] = None
    message: str


def _looks_like_worktree_path(path: Path) -> bool:
    resolved = path.resolve()
    parts = resolved.parts
    for marker in (".ripperdoc",):
        if marker in parts:
            idx = parts.index(marker)
            if idx + 1 < len(parts) and parts[idx + 1] == "worktrees":
                return True
    return False


def _is_in_active_worktree(path: Path) -> bool:
    resolved = path.resolve()
    for session in list_session_worktrees():
        session_path = session.worktree_path.resolve()
        if resolved == session_path:
            return True
        if session_path in resolved.parents:
            return True
    return _looks_like_worktree_path(resolved)


class EnterWorktreeTool(Tool[EnterWorktreeToolInput, EnterWorktreeToolOutput]):
    """Create an isolated worktree and switch this session into it."""

    @property
    def name(self) -> str:
        return TOOL_NAME

    async def description(self) -> str:
        return "Creates an isolated worktree (via git or configured hooks) and switches the session into it"

    @property
    def input_schema(self) -> type[EnterWorktreeToolInput]:
        return EnterWorktreeToolInput

    async def prompt(self, yolo_mode: bool = False) -> str:  # noqa: ARG002
        return (
            "Use this tool ONLY when the user explicitly asks to work in a worktree. "
            "Never use it for normal branch switching or routine git workflows. "
            "Requirements: must not already be in a worktree session; must be in a git repository "
            "or have WorktreeCreate/WorktreeRemove hooks configured. "
            "Behavior: creates an isolated worktree and switches this session into it; "
            "on exit, user can keep or remove it."
        )

    def defer_loading(self) -> bool:
        return True

    def user_facing_name(self) -> str:
        return "Creating worktree"

    def is_concurrency_safe(self) -> bool:
        return False

    def is_read_only(self) -> bool:
        return False

    async def validate_input(
        self,
        input_data: EnterWorktreeToolInput,  # noqa: ARG002
        context: Optional[ToolUseContext] = None,
    ) -> ValidationResult:
        if context and context.agent_id:
            return ValidationResult(
                result=False,
                message="EnterWorktree tool cannot be used in agent contexts",
            )

        current_path = (
            Path(context.working_directory).resolve()
            if context and context.working_directory
            else Path.cwd().resolve()
        )
        if _is_in_active_worktree(current_path):
            return ValidationResult(
                result=False,
                message="Already in a worktree session",
            )

        return ValidationResult(result=True)

    def render_result_for_assistant(self, output: EnterWorktreeToolOutput) -> str:
        return output.message

    def render_tool_use_message(
        self,
        input_data: EnterWorktreeToolInput,  # noqa: ARG002
        verbose: bool = False,  # noqa: ARG002
    ) -> str:
        return "Creating worktree"

    async def call(
        self,
        input_data: EnterWorktreeToolInput,
        context: ToolUseContext,
    ) -> AsyncGenerator[ToolOutput, None]:
        start_path = (
            Path(context.working_directory).resolve()
            if context.working_directory
            else Path.cwd().resolve()
        )

        requested_name = input_data.name or generate_session_worktree_name(
            session_id=os.getenv("RIPPERDOC_SESSION_ID"),
        )
        session = create_task_worktree(
            task_id=context.message_id or "session",
            base_path=start_path,
            requested_name=requested_name,
        )

        os.chdir(session.worktree_path)
        if context.set_working_directory:
            context.set_working_directory(str(session.worktree_path))

        branch_suffix = f" on branch {session.branch}" if session.branch else ""
        output = EnterWorktreeToolOutput(
            worktreePath=str(session.worktree_path),
            worktreeBranch=session.branch or None,
            message=(
                f"Created worktree at {session.worktree_path}{branch_suffix}. "
                "The session is now working in the worktree. "
                "On exit, you will be prompted to keep or remove it."
            ),
        )
        yield ToolResult(
            data=output,
            result_for_assistant=self.render_result_for_assistant(output),
        )
