"""Tests for plan mode callbacks and tool context wiring."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from ripperdoc.core.query import QueryContext
from ripperdoc.core.query import loop as loop_module
from ripperdoc.core.tool import Tool, ToolResult, ToolUseContext
from ripperdoc.tools.enter_plan_mode_tool import EnterPlanModeTool, EnterPlanModeToolInput
from ripperdoc.tools.exit_plan_mode_tool import ExitPlanModeTool, ExitPlanModeToolInput


class DummyInput(BaseModel):
    value: str


class DummyTool(Tool[DummyInput, str]):
    @property
    def name(self) -> str:
        return "Dummy"

    async def description(self) -> str:
        return "Dummy"

    @property
    def input_schema(self) -> type[DummyInput]:
        return DummyInput

    async def prompt(self, yolo_mode: bool = False) -> str:  # noqa: ARG002
        return ""

    def render_result_for_assistant(self, output: str) -> str:
        return output

    def render_tool_use_message(self, input_data: DummyInput, verbose: bool = False) -> str:
        del verbose
        return input_data.value

    async def call(self, input_data: DummyInput, context: ToolUseContext):
        del input_data, context
        yield ToolResult(data="ok")


@pytest.mark.asyncio
async def test_enter_and_exit_plan_mode_tools_invoke_callbacks() -> None:
    callbacks: list[str] = []
    context = ToolUseContext(
        on_enter_plan_mode=lambda: callbacks.append("enter"),
        on_exit_plan_mode=lambda: callbacks.append("exit"),
    )

    enter_tool = EnterPlanModeTool()
    enter_outputs = [item async for item in enter_tool.call(EnterPlanModeToolInput(), context)]
    assert callbacks == ["enter"]
    assert enter_outputs
    assert enter_outputs[-1].data.entered is True

    exit_tool = ExitPlanModeTool()
    exit_outputs = [
        item async for item in exit_tool.call(ExitPlanModeToolInput(plan="step1"), context)
    ]
    assert callbacks == ["enter", "exit"]
    assert exit_outputs
    assert exit_outputs[-1].data.plan == "step1"


@pytest.mark.asyncio
async def test_parse_and_validate_tool_context_includes_plan_callbacks() -> None:
    def on_enter() -> None:
        return None

    def on_exit() -> None:
        return None

    query_context = QueryContext(
        tools=[DummyTool()],
        yolo_mode=False,
        verbose=False,
        on_enter_plan_mode=on_enter,
        on_exit_plan_mode=on_exit,
    )

    parsed_input, tool_context, tool_input_dict, error = (
        await loop_module._parse_and_validate_tool_input_for_call(
            tool=DummyTool(),
            tool_name="Dummy",
            tool_use_id="toolu_1",
            tool_input={"value": "ok"},
            query_context=query_context,
            can_use_tool_fn=None,
            messages=[],
        )
    )

    assert error is None
    assert parsed_input.value == "ok"
    assert tool_input_dict == {"value": "ok"}
    assert tool_context.on_enter_plan_mode is on_enter
    assert tool_context.on_exit_plan_mode is on_exit
