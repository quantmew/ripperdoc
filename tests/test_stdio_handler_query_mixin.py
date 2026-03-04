"""Focused tests for stdio query mixin stream/result helpers."""

from __future__ import annotations

from typing import Any

import pytest

from ripperdoc.protocol.stdio.handler_query import _QueryRuntimeState, StdioQueryMixin
from ripperdoc.protocol.models import UserMessageData, UserStreamMessage
from ripperdoc.utils.messaging.messages import create_assistant_message, create_progress_message


class _DummyHandler(StdioQueryMixin):
    def __init__(self) -> None:
        self.streamed_payloads: list[dict[str, Any]] = []
        self._conversation_messages: list[Any] = []
        self._query_context = None

    def _build_hook_notice_stream_message(
        self,
        text: str,
        hook_event: str,
        tool_name: Any = None,
        level: str = "info",
    ) -> UserStreamMessage:
        return UserStreamMessage(
            session_id="s1",
            message=UserMessageData(
                content=text,
                metadata={
                    "hook_notice": True,
                    "hook_event": hook_event,
                    "tool_name": str(tool_name) if tool_name else None,
                    "level": level,
                },
            ),
        )

    async def _write_message_stream(self, payload: dict[str, Any]) -> None:
        self.streamed_payloads.append(payload)

    async def _write_control_response(
        self,
        request_id: str,
        response: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        self.streamed_payloads.append(
            {"type": "control_response", "request_id": request_id, "response": response, "error": error}
        )

    def _convert_message_to_sdk(self, message: Any) -> dict[str, Any] | None:
        return {
            "type": getattr(message, "type", "unknown"),
            "content": getattr(getattr(message, "message", None), "content", None),
        }


def test_update_final_result_text_resets_on_tool_use() -> None:
    handler = _DummyHandler()
    state = _QueryRuntimeState(start_time=0.0)

    assistant_message = create_assistant_message(
        [
            {"type": "text", "text": "before"},
            {"type": "tool_use", "id": "t1", "name": "Read", "input": {}},
            {"type": "text", "text": "after"},
        ]
    )

    handler._update_final_result_text(assistant_message, state)

    assert state.final_result_text == "after"


@pytest.mark.asyncio
async def test_handle_progress_stream_message_for_hook_notice() -> None:
    handler = _DummyHandler()
    state = _QueryRuntimeState(start_time=0.0)

    notice = create_progress_message(
        tool_use_id="hook",
        sibling_tool_use_ids=set(),
        content={
            "type": "hook_notice",
            "text": "blocked",
            "hook_event": "PreToolUse",
            "level": "warning",
        },
    )

    await handler._handle_progress_stream_message(notice, state)

    assert handler.streamed_payloads
    assert handler.streamed_payloads[0]["type"] == "user"
    metadata = handler.streamed_payloads[0]["message"]["metadata"]
    assert metadata["hook_notice"] is True
    assert metadata["hook_event"] == "PreToolUse"


@pytest.mark.asyncio
async def test_handle_progress_stream_message_for_subagent_assistant() -> None:
    handler = _DummyHandler()
    state = _QueryRuntimeState(start_time=0.0)

    subagent_assistant = create_assistant_message(
        "subagent response",
        input_tokens=3,
        output_tokens=5,
        cache_read_tokens=2,
        cache_creation_tokens=1,
    )
    progress = create_progress_message(
        tool_use_id="subagent",
        sibling_tool_use_ids=set(),
        content=subagent_assistant,
        is_subagent_message=True,
    )

    await handler._handle_progress_stream_message(progress, state)

    assert state.num_turns == 1
    assert state.total_input_tokens == 3
    assert state.total_output_tokens == 5
    assert state.total_cache_read_tokens == 2
    assert state.total_cache_creation_tokens == 1
    assert handler._conversation_messages and handler._conversation_messages[0] is subagent_assistant
    assert handler.streamed_payloads and handler.streamed_payloads[0]["type"] == "assistant"


@pytest.mark.asyncio
async def test_summarize_query_stage_emits_sdk_result_fields() -> None:
    handler = _DummyHandler()
    handler._session_id = "session-1"
    state = _QueryRuntimeState(start_time=0.0, num_turns=2)

    state.final_result_text = "done"

    await handler._summarize_query_stage("q-1", state)

    assert handler.streamed_payloads
    result = next(
        (item for item in handler.streamed_payloads if item.get("type") == "result"),
        None,
    )
    assert result is not None
    assert result["type"] == "result"
    assert result["subtype"] == "success"
    assert result["result"] == "done"
    assert result["num_turns"] == 2
    assert result["session_id"] == "session-1"
    assert result["is_error"] is False
    assert isinstance(result["duration_ms"], int)
    assert result["duration_ms"] >= 0
    assert result["duration_api_ms"] == result["duration_ms"]


@pytest.mark.asyncio
async def test_summarize_query_stage_marks_error_during_execution() -> None:
    handler = _DummyHandler()
    handler._session_id = "session-2"
    state = _QueryRuntimeState(start_time=0.0, is_error=True)

    state.final_result_text = "error"

    await handler._summarize_query_stage("q-2", state)

    assert handler.streamed_payloads
    result = next(
        (item for item in handler.streamed_payloads if item.get("type") == "result"),
        None,
    )
    assert result is not None
    assert result["type"] == "result"
    assert result["subtype"] == "error_during_execution"
    assert result["is_error"] is True
