"""Focused regression tests for OpenAI provider refactor helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ripperdoc.core.providers.openai import (
    _StreamAccumulator,
    _build_non_stream_empty_or_error_response,
    _build_stream_content_blocks,
    _consume_stream_chunk,
)


@pytest.mark.asyncio
async def test_consume_stream_chunk_collects_tool_stream_state() -> None:
    progress_updates: list[str] = []

    async def capture(chunk: str) -> None:
        progress_updates.append(chunk)

    tool_delta = SimpleNamespace(
        index=0,
        id="tool_1",
        function=SimpleNamespace(name="Read", arguments='{"path":"README.md"}'),
    )
    delta = SimpleNamespace(
        content="partial text",
        reasoning_content="thinking",
        reasoning_details={"kind": "trace"},
        tool_calls=[tool_delta],
    )
    chunk = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3),
        choices=[SimpleNamespace(delta=delta)],
    )

    state = _StreamAccumulator()
    await _consume_stream_chunk(
        chunk,
        stream_state=state,
        can_stream_tools=True,
        progress_callback=capture,
    )

    assert state.usage_tokens["input_tokens"] == 2
    assert state.usage_tokens["output_tokens"] == 3
    assert state.streamed_tool_text == ["partial text"]
    assert state.reasoning_text == ["thinking"]
    assert state.reasoning_details == [{"kind": "trace"}]
    assert state.streamed_tool_calls[0]["name"] == "Read"
    assert state.streamed_tool_calls[0]["arguments"] == '{"path":"README.md"}'
    assert progress_updates == ["partial text", '{"path":"README.md"}', "[tool:Read]"]


def test_build_stream_content_blocks_emits_text_and_tool_use() -> None:
    state = _StreamAccumulator(
        streamed_tool_text=["intro"],
        streamed_tool_calls={
            0: {"id": "tool_1", "name": "Read", "arguments": '{"path":"README.md"}'}
        },
    )
    blocks, finish_reason = _build_stream_content_blocks(
        state,
        can_stream_text=False,
        can_stream_tools=True,
    )

    assert finish_reason == "stream"
    assert blocks[0] == {"type": "text", "text": "intro"}
    assert blocks[1]["type"] == "tool_use"
    assert blocks[1]["tool_use_id"] == "tool_1"
    assert blocks[1]["name"] == "Read"
    assert blocks[1]["input"] == {"path": "README.md"}


def test_non_stream_empty_or_error_response_handles_error_payload() -> None:
    response = SimpleNamespace(status=429, message="rate limited")
    result = _build_non_stream_empty_or_error_response(
        openai_response=response,
        model="gpt-test",
        duration_ms=12.0,
        usage_tokens={"input_tokens": 0, "output_tokens": 0},
        cost_usd=0.0,
        response_metadata={},
    )

    assert result is not None
    assert result.is_error is True
    assert result.error_code == "api_error"
    assert "429" in (result.error_message or "")
