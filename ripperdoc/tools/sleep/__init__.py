"""Sleep tool for non-blocking concurrent sleep."""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator, Optional

from pydantic import BaseModel, Field

from ripperdoc.core.tool import Tool, ToolOutput, ToolResult, ToolUseContext, ValidationResult

TOOL_NAME = "Sleep"


class SleepToolInput(BaseModel):
    """Input for SleepTool."""

    duration: float = Field(
        description="Duration to sleep in seconds",
        gt=0,
        le=3600,
    )


class SleepToolOutput(BaseModel):
    """Output for SleepTool."""

    duration: float
    message: str


class SleepTool(Tool[SleepToolInput, SleepToolOutput]):
    """Non-blocking concurrent sleep tool."""

    @property
    def name(self) -> str:
        return TOOL_NAME

    async def description(self) -> str:
        return "Sleep for a specified duration without blocking other operations."

    @property
    def input_schema(self) -> type[SleepToolInput]:
        return SleepToolInput

    async def prompt(self, yolo_mode: bool = False) -> str:  # noqa: ARG002
        return (
            "Use this tool to sleep for a specified duration. "
            "This is non-blocking and concurrency-safe — other tools can run concurrently. "
            "Maximum duration is 3600 seconds (1 hour)."
        )

    def is_read_only(self) -> bool:
        return True

    def is_concurrency_safe(self) -> bool:
        return True

    def needs_permissions(self, _input_data: Optional[SleepToolInput] = None) -> bool:
        return False

    async def validate_input(
        self,
        input_data: SleepToolInput,
        _context: Optional[ToolUseContext] = None,
    ) -> ValidationResult:
        if input_data.duration <= 0:
            return ValidationResult(result=False, message="duration must be positive")
        if input_data.duration > 3600:
            return ValidationResult(result=False, message="duration must not exceed 3600 seconds")
        return ValidationResult(result=True)

    def render_result_for_assistant(self, output: SleepToolOutput) -> str:
        return output.message

    def render_tool_use_message(
        self, input_data: SleepToolInput, _verbose: bool = False
    ) -> str:
        return f"Sleeping for {input_data.duration}s"

    async def call(
        self,
        input_data: SleepToolInput,
        _context: ToolUseContext,
    ) -> AsyncGenerator[ToolOutput, None]:
        await asyncio.sleep(input_data.duration)

        output = SleepToolOutput(
            duration=input_data.duration,
            message=f"Slept for {input_data.duration}s",
        )
        yield ToolResult(
            data=output,
            result_for_assistant=self.render_result_for_assistant(output),
        )
