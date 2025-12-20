"""Tool to retrieve output from background bash tasks."""

from typing import Any, AsyncGenerator, Optional
from pydantic import BaseModel, Field

from ripperdoc.core.tool import Tool, ToolUseContext, ToolResult, ValidationResult
from ripperdoc.tools.background_shell import get_background_status


class BashOutputInput(BaseModel):
    """Input schema for BashOutput."""

    task_id: str = Field(
        description="Background task id returned by BashTool when run_in_background is true"
    )
    consume: bool = Field(
        default=True, description="Whether to clear buffered output after reading (default: True)"
    )


class BashOutputData(BaseModel):
    """Snapshot of a background task."""

    task_id: str
    command: str
    status: str
    stdout: str
    stderr: str
    exit_code: Optional[int]
    duration_ms: float


class BashOutputTool(Tool[BashOutputInput, BashOutputData]):
    """Read buffered output from a background bash task."""

    @property
    def name(self) -> str:
        return "BashOutput"

    async def description(self) -> str:
        return "Read output and status from a background bash command started with BashTool(run_in_background=True)."

    async def prompt(self, yolo_mode: bool = False) -> str:
        return "Fetch buffered output and status for a background bash task by id."

    @property
    def input_schema(self) -> type[BashOutputInput]:
        return BashOutputInput

    def is_read_only(self) -> bool:
        return True

    def is_concurrency_safe(self) -> bool:
        return True

    def needs_permissions(self, input_data: Any = None) -> bool:
        return False

    async def validate_input(
        self, input_data: BashOutputInput, context: Optional[ToolUseContext] = None
    ) -> ValidationResult:
        try:
            get_background_status(input_data.task_id, consume=False)
        except KeyError:
            return ValidationResult(
                result=False, message=f"No background task found with id '{input_data.task_id}'"
            )
        return ValidationResult(result=True)

    def render_result_for_assistant(self, output: BashOutputData) -> str:
        parts = [
            f"status: {output.status}",
            f"exit code: {output.exit_code if output.exit_code is not None else 'running'}",
        ]
        if output.stdout:
            parts.append(f"stdout:\n{output.stdout}")
        if output.stderr:
            parts.append(f"stderr:\n{output.stderr}")
        return "\n\n".join(parts)

    def render_tool_use_message(self, input_data: BashOutputInput, verbose: bool = False) -> str:
        suffix = " (consume=0)" if not input_data.consume else ""
        return f"$ bash-output {input_data.task_id}{suffix}"

    async def call(
        self, input_data: BashOutputInput, context: ToolUseContext
    ) -> AsyncGenerator[ToolResult, None]:
        status = get_background_status(input_data.task_id, consume=input_data.consume)
        output = BashOutputData(
            task_id=status["id"],
            command=status["command"],
            status=status["status"],
            stdout=status["stdout"],
            stderr=status["stderr"],
            exit_code=status["exit_code"],
            duration_ms=status["duration_ms"],
        )
        yield ToolResult(data=output, result_for_assistant=self.render_result_for_assistant(output))
