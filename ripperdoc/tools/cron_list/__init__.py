"""CronList tool — list all scheduled cron jobs."""

from __future__ import annotations

from typing import AsyncGenerator, List, Optional

from pydantic import BaseModel

from ripperdoc.core.tool import Tool, ToolOutput, ToolResult, ToolUseContext, ValidationResult
from ripperdoc.utils.log import get_logger

logger = get_logger()

TOOL_NAME = "CronList"


class CronListInput(BaseModel):
    pass


class CronJobInfo(BaseModel):
    id: str
    cron: str
    prompt: str
    recurring: bool
    durable: bool


class CronListOutput(BaseModel):
    jobs: List[CronJobInfo] = []


class CronListTool(Tool[CronListInput, CronListOutput]):
    """List all scheduled cron jobs."""

    @property
    def name(self) -> str:
        return TOOL_NAME

    async def description(self) -> str:
        return "List all scheduled cron jobs."

    @property
    def input_schema(self) -> type[CronListInput]:
        return CronListInput

    async def prompt(self, yolo_mode: bool = False) -> str:
        from ripperdoc.tools.cron_list._prompt import CRON_LIST_PROMPT
        return CRON_LIST_PROMPT


    def is_read_only(self) -> bool:
        return True

    def is_concurrency_safe(self) -> bool:
        return True

    def needs_permissions(self, _input_data: Optional[CronListInput] = None) -> bool:
        return False

    async def validate_input(
        self,
        input_data: CronListInput,
        _context: Optional[ToolUseContext] = None,
    ) -> ValidationResult:
        return ValidationResult(result=True)

    def render_result_for_assistant(self, output: CronListOutput) -> str:
        if not output.jobs:
            return "No scheduled jobs."
        lines = []
        for j in output.jobs:
            tag = "recurring" if j.recurring else "one-shot"
            storage = "durable" if j.durable else "session-only"
            prompt_preview = j.prompt[:80] + "..." if len(j.prompt) > 80 else j.prompt
            lines.append(f"{j.id} — {j.cron} ({tag}, {storage}): {prompt_preview}")
        return "\n".join(lines)

    def render_tool_use_message(self, input_data: CronListInput, _verbose: bool = False) -> str:
        return "CronList"

    async def call(
        self,
        input_data: CronListInput,
        _context: ToolUseContext,
    ) -> AsyncGenerator[ToolOutput, None]:
        from ripperdoc.tools.schedule_cron._cron_store import list_jobs

        jobs = [
            CronJobInfo(
                id=j["id"],
                cron=j["cron"],
                prompt=j["prompt"],
                recurring=j.get("recurring", True),
                durable=j.get("durable", False),
            )
            for j in list_jobs()
        ]
        output = CronListOutput(jobs=jobs)
        yield ToolResult(data=output, result_for_assistant=self.render_result_for_assistant(output))
