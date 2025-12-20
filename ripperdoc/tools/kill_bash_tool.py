"""Tool to terminate a running background bash task."""

from __future__ import annotations

from typing import AsyncGenerator, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from ripperdoc.core.tool import Tool, ToolResult, ToolUseContext, ValidationResult
from ripperdoc.tools.background_shell import (
    get_background_status,
    kill_background_task,
)
from ripperdoc.utils.permissions import PermissionDecision


KILL_BASH_PROMPT = """
- Kills a running background bash shell by its ID
- Takes a shell_id parameter identifying the shell to kill
- Returns a success or failure status
- Use this tool when you need to terminate a long-running shell
- Shell IDs can be found using the Bash tool (run_in_background) and BashOutput
""".strip()


class KillBashInput(BaseModel):
    """Input schema for KillBash."""

    task_id: str = Field(
        description="Background task id to kill",
        validation_alias=AliasChoices("task_id", "shell_id"),
        serialization_alias="task_id",
    )
    model_config = ConfigDict(validate_by_alias=True, validate_by_name=True, extra="ignore")


class KillBashOutput(BaseModel):
    """Result of attempting to kill a background task."""

    task_id: str
    success: bool
    message: str
    status: Optional[str] = None


class KillBashTool(Tool[KillBashInput, KillBashOutput]):
    """Terminate a background bash command."""

    @property
    def name(self) -> str:
        return "KillBash"

    async def description(self) -> str:
        return "Kill a background bash shell by ID"

    async def prompt(self, yolo_mode: bool = False) -> str:
        return KILL_BASH_PROMPT

    @property
    def input_schema(self) -> type[KillBashInput]:
        return KillBashInput

    def is_read_only(self) -> bool:
        return False

    def is_concurrency_safe(self) -> bool:
        return True

    def needs_permissions(self, input_data: Optional[KillBashInput] = None) -> bool:
        return True

    async def check_permissions(
        self, input_data: KillBashInput, permission_context: object
    ) -> PermissionDecision:
        # Killing is destructive; require explicit confirmation upstream.
        return PermissionDecision(behavior="allow", updated_input=input_data)

    async def validate_input(
        self, input_data: KillBashInput, context: Optional[ToolUseContext] = None
    ) -> ValidationResult:
        try:
            get_background_status(input_data.task_id, consume=False)
        except KeyError:
            return ValidationResult(
                result=False, message=f"No background task found with id '{input_data.task_id}'"
            )
        return ValidationResult(result=True)

    def render_result_for_assistant(self, output: KillBashOutput) -> str:
        return output.message

    def render_tool_use_message(self, input_data: KillBashInput, verbose: bool = False) -> str:
        return f"$ kill-background {input_data.task_id}"

    async def call(
        self, input_data: KillBashInput, context: ToolUseContext
    ) -> AsyncGenerator[ToolResult, None]:
        try:
            status = get_background_status(input_data.task_id, consume=False)
        except KeyError:
            output = KillBashOutput(
                task_id=input_data.task_id,
                success=False,
                message=f"No shell found with ID: {input_data.task_id}",
                status=None,
            )
            yield ToolResult(
                data=output, result_for_assistant=self.render_result_for_assistant(output)
            )
            return

        if status["status"] != "running":
            output = KillBashOutput(
                task_id=input_data.task_id,
                success=False,
                message=f"Shell {input_data.task_id} is not running (status: {status['status']}).",
                status=status["status"],
            )
            yield ToolResult(
                data=output, result_for_assistant=self.render_result_for_assistant(output)
            )
            return

        killed = await kill_background_task(input_data.task_id)
        message = (
            f"Successfully killed shell: {input_data.task_id} ({status['command']})"
            if killed
            else f"Failed to kill shell: {input_data.task_id}"
        )
        output = KillBashOutput(
            task_id=input_data.task_id,
            success=killed,
            message=message,
            status="killed" if killed else status["status"],
        )
        yield ToolResult(data=output, result_for_assistant=self.render_result_for_assistant(output))
