"""CronDelete tool — cancel a scheduled cron job by ID."""

from __future__ import annotations

from typing import AsyncGenerator, Optional

from pydantic import BaseModel, Field

from ripperdoc.core.tool import Tool, ToolOutput, ToolResult, ToolUseContext, ValidationResult
from ripperdoc.utils.log import get_logger

logger = get_logger()

TOOL_NAME = "CronDelete"


class CronDeleteInput(BaseModel):
    id: str = Field(description="Job ID returned by CronCreate")


class CronDeleteOutput(BaseModel):
    job_id: str
    message: str


class CronDeleteTool(Tool[CronDeleteInput, CronDeleteOutput]):
    """Cancel a scheduled cron job by ID."""

    @property
    def name(self) -> str:
        return TOOL_NAME

    async def description(self) -> str:
        return "Cancel a scheduled cron job by ID."

    @property
    def input_schema(self) -> type[CronDeleteInput]:
        return CronDeleteInput

    async def prompt(self, yolo_mode: bool = False) -> str:
        from ripperdoc.tools.cron_delete._prompt import CRON_DELETE_PROMPT
        return CRON_DELETE_PROMPT


    def is_read_only(self) -> bool:
        return False

    def is_concurrency_safe(self) -> bool:
        return True

    def needs_permissions(self, _input_data: Optional[CronDeleteInput] = None) -> bool:
        return False

    async def validate_input(
        self,
        input_data: CronDeleteInput,
        _context: Optional[ToolUseContext] = None,
    ) -> ValidationResult:
        from ripperdoc.tools.schedule_cron._cron_store import get_all_jobs

        all_jobs = get_all_jobs()
        if input_data.id not in all_jobs:
            return ValidationResult(
                result=False, message=f"No scheduled job with id '{input_data.id}'",
            )
        return ValidationResult(result=True)

    def render_result_for_assistant(self, output: CronDeleteOutput) -> str:
        return output.message

    def render_tool_use_message(self, input_data: CronDeleteInput, _verbose: bool = False) -> str:
        return f"CronDelete: {input_data.id}"

    async def call(
        self,
        input_data: CronDeleteInput,
        _context: ToolUseContext,
    ) -> AsyncGenerator[ToolOutput, None]:
        from ripperdoc.tools.schedule_cron._cron_store import remove_job

        remove_job(input_data.id)
        output = CronDeleteOutput(
            job_id=input_data.id,
            message=f"Cancelled job {input_data.id}.",
        )
        yield ToolResult(data=output, result_for_assistant=self.render_result_for_assistant(output))
