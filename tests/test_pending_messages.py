"""Tests for pending conversation message injection."""

from typing import Any, List

import pytest
from pydantic import BaseModel

from ripperdoc.core.query import QueryContext, query
from ripperdoc.core.tool import Tool, ToolResult, ToolUseContext
from ripperdoc.utils.messages import create_assistant_message, create_user_message


class NoOpInput(BaseModel):
    value: str = "x"


class NoOpTool(Tool[NoOpInput, None]):
    @property
    def name(self) -> str:  # pragma: no cover - trivial
        return "NoOp"

    async def description(self) -> str:  # pragma: no cover - trivial
        return "no-op"

    @property
    def input_schema(self) -> type[NoOpInput]:  # pragma: no cover - trivial
        return NoOpInput

    async def prompt(self, yolo_mode: bool = False) -> str:  # pragma: no cover - trivial
        return "noop"

    def render_result_for_assistant(self, output: None) -> str:  # pragma: no cover - trivial
        return "done"

    def render_tool_use_message(
        self, input_data: NoOpInput, verbose: bool = False
    ) -> str:  # pragma: no cover - trivial
        return "noop"

    async def call(self, input_data: NoOpInput, context: ToolUseContext):  # noqa: ARG002
        yield ToolResult(data=None)


@pytest.mark.asyncio
async def test_pending_messages_injected_between_iterations(monkeypatch: Any):
    """Queued messages should appear before the next tool cycle starts."""

    tool = NoOpTool()
    query_context = QueryContext(tools=[tool], yolo_mode=True, verbose=False)

    async def fake_query_llm(messages: List[Any], *_args: Any, **_kwargs: Any):
        # If no tool_result yet, request the tool.
        has_tool_result = False
        for msg in messages:
            if getattr(msg, "type", None) != "user":
                continue
            content = getattr(getattr(msg, "message", None), "content", None)
            if isinstance(content, list):
                for block in content:
                    if getattr(block, "type", None) == "tool_result":
                        has_tool_result = True
                        break
            if has_tool_result:
                break

        if not has_tool_result:
            return create_assistant_message(
                [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "tool_use_id": "t1",
                        "name": tool.name,
                        "input": {"value": "x"},
                    }
                ]
            )

        # Otherwise respond with the latest plain-text user content (the queued interjection).
        latest_text = ""
        for msg in reversed(messages):
            if getattr(msg, "type", None) != "user":
                continue
            content = getattr(getattr(msg, "message", None), "content", None)
            if isinstance(content, str):
                latest_text = content
                break

        return create_assistant_message(f"ack:{latest_text or 'none'}")

    monkeypatch.setattr("ripperdoc.core.query.query_llm", fake_query_llm)

    # Kick off the query with an initial user message.
    messages = [create_user_message("start")]
    collected: List[Any] = []

    async for message in query(messages, "", {}, query_context, None):
        collected.append(message)
        # After tool_result arrives, enqueue an interjection to be injected next.
        if getattr(message, "type", None) == "user":
            content = getattr(getattr(message, "message", None), "content", None)
            if isinstance(content, list) and any(
                getattr(block, "type", None) == "tool_result" for block in content
            ):
                query_context.enqueue_user_message("interjection")

    message_types = [getattr(msg, "type", None) for msg in collected]
    assert message_types == ["assistant", "user", "user", "assistant"]
    final_assistant = [msg for msg in collected if getattr(msg, "type", None) == "assistant"][-1]
    assert getattr(getattr(final_assistant, "message", None), "content", None) == "ack:interjection"
