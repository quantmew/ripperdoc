"""CronCreate tool — schedule a recurring or one-shot cron task."""

from __future__ import annotations

from typing import AsyncGenerator, Optional

from pydantic import BaseModel, Field

from ripperdoc.core.tool import Tool, ToolOutput, ToolResult, ToolUseContext, ValidationResult
from ripperdoc.utils.log import get_logger

logger = get_logger()

TOOL_NAME = "CronCreate"


class CronCreateInput(BaseModel):
    cron: str = Field(
        description='Standard 5-field cron expression in local time: "M H DoM Mon DoW"',
    )
    prompt: str = Field(
        description="The prompt to enqueue at each fire time",
    )
    recurring: bool = Field(
        default=True,
        description="true = fire on every cron match until deleted. false = fire once then auto-delete.",
    )
    durable: bool = Field(
        default=False,
        description="true = persist to .ripperdoc/scheduled_tasks.json. false = session-only.",
    )


class CronCreateOutput(BaseModel):
    job_id: str
    message: str


class CronCreateTool(Tool[CronCreateInput, CronCreateOutput]):
    """Schedule a prompt to run at a future time using cron expressions."""

    @property
    def name(self) -> str:
        return TOOL_NAME

    async def description(self) -> str:
        return "Schedule a prompt to be enqueued at a future time."

    @property
    def input_schema(self) -> type[CronCreateInput]:
        return CronCreateInput

    async def prompt(self, yolo_mode: bool = False) -> str:  # noqa: ARG002
        return (
            "Schedule a prompt to be enqueued at a future time. "
            'Uses standard 5-field cron in local time: "M H DoM Mon DoW". '
            "Recurring tasks auto-expire after 7 days. "
            "Returns a job ID you can pass to CronDelete."
        )

    def is_read_only(self) -> bool:
        return False

    def is_concurrency_safe(self) -> bool:
        return True

    def needs_permissions(self, _input_data: Optional[CronCreateInput] = None) -> bool:
        return False

    async def validate_input(
        self,
        input_data: CronCreateInput,
        _context: Optional[ToolUseContext] = None,
    ) -> ValidationResult:
        try:
            from croniter import croniter
            croniter(input_data.cron)
        except ImportError:
            return ValidationResult(
                result=False, message="croniter package required. Install with: pip install croniter",
            )
        except (ValueError, KeyError) as exc:
            return ValidationResult(result=False, message=f"Invalid cron expression: {exc}")
        from ripperdoc.tools.schedule_cron._cron_store import job_count, MAX_JOBS

        if job_count() >= MAX_JOBS:
            return ValidationResult(
                result=False, message=f"Too many scheduled jobs (max {MAX_JOBS}). Cancel one first.",
            )
        return ValidationResult(result=True)

    def render_result_for_assistant(self, output: CronCreateOutput) -> str:
        return output.message

    def render_tool_use_message(self, input_data: CronCreateInput, _verbose: bool = False) -> str:
        return f"CronCreate: {input_data.cron}"

    async def call(
        self,
        input_data: CronCreateInput,
        _context: ToolUseContext,
    ) -> AsyncGenerator[ToolOutput, None]:
        from ripperdoc.tools.schedule_cron._cron_store import add_job

        job_id = add_job(
            input_data.cron, input_data.prompt,
            input_data.recurring, input_data.durable,
        )
        schedule_type = "recurring" if input_data.recurring else "one-shot"
        storage = "durable" if input_data.durable else "session-only"

        output = CronCreateOutput(
            job_id=job_id,
            message=f"Scheduled {schedule_type} job {job_id} ({storage}): cron='{input_data.cron}'",
        )
        yield ToolResult(data=output, result_for_assistant=self.render_result_for_assistant(output))
