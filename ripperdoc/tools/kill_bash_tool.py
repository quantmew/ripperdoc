"""Tool to terminate a running background bash task."""

from typing import AsyncGenerator
from pydantic import BaseModel, Field

from ripperdoc.core.tool import Tool, ToolUseContext, ToolResult, ValidationResult
from ripperdoc.tools.background_shell import kill_background_task, get_background_status


KILL_BASH_PROMPT = (
    "- Kills a running background bash shell by its ID\n"
    "- Takes a task_id parameter identifying the shell to kill\n"
    "- Returns a success or failure status\n"
    "- Use this tool when you need to terminate a long-running shell\n"
    "- Shell IDs can be found using the /bashes command"
)


class KillBashInput(BaseModel):
    """Input schema for KillBash."""
    task_id: str = Field(description="Background task id to kill")


class KillBashOutput(BaseModel):
    """Result of attempting to kill a background task."""
    task_id: str
    success: bool
    message: str


class KillBashTool(Tool[KillBashInput, KillBashOutput]):
    """Terminate a background bash command."""

    @property
    def name(self) -> str:
        return "KillBash"

    async def description(self) -> str:
        return "Kill a background bash shell by ID"

    async def prompt(self, safe_mode: bool = False) -> str:
        return KILL_BASH_PROMPT

    @property
    def input_schema(self) -> type[KillBashInput]:
        return KillBashInput

    def is_read_only(self) -> bool:
        return False

    def is_concurrency_safe(self) -> bool:
        return True

    def needs_permissions(self, input_data=None) -> bool:
        return True

    async def validate_input(self, input_data: KillBashInput, context: ToolUseContext) -> ValidationResult:
        try:
            get_background_status(input_data.task_id, consume=False)
        except KeyError:
            return ValidationResult(result=False, message=f"No background task found with id '{input_data.task_id}'")
        return ValidationResult(result=True)

    def render_result_for_assistant(self, output: KillBashOutput) -> str:
        return output.message

    def render_tool_use_message(self, input_data: KillBashInput, verbose: bool = False) -> str:
        return f"$ kill-background {input_data.task_id}"

    async def call(
        self,
        input_data: KillBashInput,
        context: ToolUseContext
    ) -> AsyncGenerator[ToolResult, None]:
        alive = await kill_background_task(input_data.task_id)
        if alive:
            message = f"Killed background task '{input_data.task_id}'."
        else:
            message = f"Background task '{input_data.task_id}' is not running or not found."
        output = KillBashOutput(
            task_id=input_data.task_id,
            success=alive,
            message=message
        )
        yield ToolResult(
            data=output,
            result_for_assistant=self.render_result_for_assistant(output)
        )
