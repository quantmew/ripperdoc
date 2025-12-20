"""Regression tests for query abort behavior."""

from typing import Any, List

import pytest
from pydantic import BaseModel

from ripperdoc.core.permissions import PermissionResult
from ripperdoc.core.query import QueryContext, query
from ripperdoc.core.tool import Tool, ToolResult, ToolUseContext
from ripperdoc.utils.messages import create_assistant_message, create_user_message


class DummyInput(BaseModel):
    command: str


class DummyTool(Tool[DummyInput, None]):
    @property
    def name(self) -> str:
        return "Dummy"

    async def description(self) -> str:
        return "Dummy tool"

    @property
    def input_schema(self) -> type[DummyInput]:
        return DummyInput

    async def prompt(self, yolo_mode: bool = False) -> str:  # noqa: ARG002
        return "dummy"

    def render_result_for_assistant(self, output: None) -> str:  # noqa: ARG002
        return "done"

    def render_tool_use_message(self, input_data: DummyInput, verbose: bool = False) -> str:  # noqa: ARG002
        return input_data.command

    async def call(self, input_data: DummyInput, context: ToolUseContext):  # noqa: ARG002
        yield ToolResult(data=None)


class RecordingTool(DummyTool):
    """Dummy tool that records whether it was actually executed."""

    def __init__(self) -> None:
        super().__init__()
        self.called = False

    async def call(self, input_data: DummyInput, context: ToolUseContext):  # noqa: ARG002
        self.called = True
        yield ToolResult(data=None)


@pytest.mark.asyncio
async def test_permission_denial_does_not_set_abort(monkeypatch: Any):
    """Permission denial should not flip abort flag for subsequent queries."""
    tool = DummyTool()
    query_context = QueryContext(tools=[tool], yolo_mode=True, verbose=False)

    async def fake_query_llm(
        messages: List[Any],
        system_prompt: str,
        tools: List[Any],
        max_thinking_tokens: int = 0,
        model: str = "main",
        abort_signal: Any = None,
        **_: Any,
    ):
        # If we've already returned a tool_result, end the loop with plain text.
        for msg in messages:
            content = getattr(msg, "message", None)
            if content and isinstance(content.content, list):
                if any(getattr(block, "type", None) == "tool_result" for block in content.content):
                    return create_assistant_message("done")

        return create_assistant_message(
            [
                {
                    "type": "tool_use",
                    "id": "t1",
                    "tool_use_id": "t1",
                    "name": tool.name,
                    "input": {"command": "echo hi"},
                }
            ]
        )

    monkeypatch.setattr("ripperdoc.core.query.query_llm", fake_query_llm)

    async def deny_permissions(_: Tool[Any, Any], __: Any) -> PermissionResult:
        return PermissionResult(result=False, message="denied")

    messages = [create_user_message("hi")]
    results: list[Any] = []

    async for message in query(messages, "", {}, query_context, deny_permissions):
        results.append(message)

    assert query_context.abort_controller.is_set() is False
    # We should have produced at least one tool_result message describing the denial.
    assert any(
        getattr(block, "type", None) == "tool_result"
        for msg in results
        if hasattr(msg, "message") and isinstance(getattr(msg.message, "content", None), list)
        for block in msg.message.content  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
async def test_permission_denial_prevents_tool_execution(monkeypatch: Any):
    """When permission is denied, the tool should not be executed."""
    tool = RecordingTool()
    query_context = QueryContext(tools=[tool], yolo_mode=True, verbose=False)

    async def fake_query_llm(
        messages: List[Any],
        system_prompt: str,
        tools: List[Any],
        max_thinking_tokens: int = 0,
        model: str = "main",
        abort_signal: Any = None,
        **_: Any,
    ):
        # Always ask to run the tool once.
        return create_assistant_message(
            [
                {
                    "type": "tool_use",
                    "id": "t1",
                    "tool_use_id": "t1",
                    "name": tool.name,
                    "input": {"command": "echo hi"},
                }
            ]
        )

    monkeypatch.setattr("ripperdoc.core.query.query_llm", fake_query_llm)

    async def deny_permissions(_: Tool[Any, Any], __: Any) -> PermissionResult:
        return PermissionResult(result=False, message="denied")

    messages = [create_user_message("hi")]

    results: list[Any] = []
    async for message in query(messages, "", {}, query_context, deny_permissions):
        results.append(message)

    assert tool.called is False
    assert any(
        getattr(block, "type", None) == "tool_result" and (block.text or "").strip()
        for msg in results
        if hasattr(msg, "message") and isinstance(getattr(msg.message, "content", None), list)
        for block in msg.message.content  # type: ignore[arg-type]
    )
