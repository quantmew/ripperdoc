"""Tests for plan mode callbacks and tool context wiring."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from ripperdoc.core.config import ModelProfile, ProtocolType
from ripperdoc.core.query import QueryContext
from ripperdoc.core.query import loop as loop_module
from ripperdoc.core.tool import Tool, ToolResult, ToolUseContext
from ripperdoc.tools.enter_plan_mode_tool import EnterPlanModeTool, EnterPlanModeToolInput
from ripperdoc.tools.exit_plan_mode_tool import ExitPlanModeTool, ExitPlanModeToolInput
from ripperdoc.utils.messaging.pending_messages import PendingMessageQueue


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
async def test_enter_and_exit_plan_mode_tools_invoke_callbacks(tmp_path) -> None:
    callbacks: list[str] = []
    plan_file = tmp_path / "plan.md"
    pending = PendingMessageQueue()
    context = ToolUseContext(
        on_enter_plan_mode=lambda: callbacks.append("enter"),
        on_exit_plan_mode=lambda: callbacks.append("exit"),
        plan_file_path=str(plan_file),
        pending_message_queue=pending,
    )

    enter_tool = EnterPlanModeTool()
    enter_outputs = [item async for item in enter_tool.call(EnterPlanModeToolInput(), context)]
    assert callbacks == ["enter"]
    assert enter_outputs
    assert enter_outputs[-1].data.entered is True
    assert str(plan_file) in enter_outputs[-1].data.message

    exit_tool = ExitPlanModeTool()
    exit_outputs = [
        item async for item in exit_tool.call(ExitPlanModeToolInput(plan="step1"), context)
    ]
    assert callbacks == ["enter", "exit"]
    assert exit_outputs
    assert exit_outputs[-1].data.plan == "step1"
    pending_messages = pending.drain()
    assert len(pending_messages) == 1
    assert pending_messages[0].type == "attachment"
    assert pending_messages[0].attachment_type == "plan_mode_exit"


@pytest.mark.asyncio
async def test_exit_plan_mode_tool_respects_callback_decision_payload(tmp_path) -> None:
    plan_file = tmp_path / "plan.md"
    plan_file.write_text("step1", encoding="utf-8")

    async def _decision_callback(**kwargs):  # noqa: ANN003
        assert kwargs["plan"] == "step1"
        return {"approved": False, "permission_mode": "plan", "clear_context": False}

    pending = PendingMessageQueue()
    context = ToolUseContext(
        on_exit_plan_mode=_decision_callback,
        plan_file_path=str(plan_file),
        pending_message_queue=pending,
    )
    exit_tool = ExitPlanModeTool()
    outputs = [item async for item in exit_tool.call(ExitPlanModeToolInput(), context)]

    assert outputs
    payload = outputs[-1].data
    assert payload.approved is False
    assert payload.permission_mode == "plan"
    assert payload.clear_context is False
    pending_messages = pending.drain()
    assert len(pending_messages) == 1
    assert "Rejected plan:" in str(pending_messages[0].message.content)


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
    assert tool_context.plan_file_path == query_context.plan_file_path


def test_build_iteration_plan_includes_reentry_attachment_after_exit(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    plan_file = tmp_path / ".ripperdoc" / "plans" / "main.md"
    plan_file.parent.mkdir(parents=True, exist_ok=True)
    plan_file.write_text("# Existing plan", encoding="utf-8")

    query_context = QueryContext(
        tools=[DummyTool()],
        permission_mode="plan",
        working_directory=str(tmp_path),
    )
    query_context.has_exited_plan_mode = True

    monkeypatch.setattr(
        loop_module,
        "resolve_model_profile",
        lambda _model: ModelProfile(
            protocol=ProtocolType.ANTHROPIC,
            model="claude-sonnet-4-20250514",
            max_input_tokens=1000,
        ),
    )
    monkeypatch.setattr(loop_module, "determine_tool_mode", lambda _profile: "native")

    plan = loop_module._build_iteration_plan(
        messages=[],
        system_prompt="system",
        context={},
        query_context=query_context,
    )

    assert len(plan.plan_mode_messages) == 2
    assert plan.plan_mode_messages[0].attachment_type == "plan_mode_reentry"
    assert plan.plan_mode_messages[1].attachment_type == "plan_mode"
    assert query_context.has_exited_plan_mode is False
