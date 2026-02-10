"""Focused regression tests for Anthropic provider stream helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ripperdoc.core.providers.anthropic import (
    AnthropicClient,
    _classify_anthropic_error,
    _content_blocks_from_stream_state,
)
from ripperdoc.core.providers.errors import ProviderMappedError, ProviderTimeoutError


def test_content_blocks_from_stream_state_includes_signature_when_available() -> None:
    blocks = _content_blocks_from_stream_state(
        collected_text=["answer"],
        collected_thinking=["step-1"],
        collected_tool_calls={},
        thinking_signature="sig_123",
    )

    assert blocks[0] == {"type": "thinking", "thinking": "step-1", "signature": "sig_123"}
    assert blocks[1] == {"type": "text", "text": "answer"}


def test_content_blocks_from_stream_state_omits_signature_when_missing() -> None:
    blocks = _content_blocks_from_stream_state(
        collected_text=[],
        collected_thinking=["step-1"],
        collected_tool_calls={},
    )

    assert blocks[0] == {"type": "thinking", "thinking": "step-1"}
    assert "signature" not in blocks[0]


@pytest.mark.asyncio
async def test_handle_stream_event_captures_signature_deltas() -> None:
    client = AnthropicClient()
    thinking_signature_ref: list[str | None] = [None]

    await client._handle_stream_event(
        event=SimpleNamespace(
            type="content_block_delta",
            index=0,
            delta=SimpleNamespace(type="signature_delta", signature="sig_delta"),
        ),
        collected_text=[],
        collected_thinking=[],
        collected_tool_calls={},
        usage_tokens={},
        progress_callback=None,
        thinking_signature_ref=thinking_signature_ref,
        current_block_index_ref=[-1],
        current_block_type_ref=[None],
    )
    assert thinking_signature_ref[0] == "sig_delta"

    await client._handle_stream_event(
        event=SimpleNamespace(type="signature", signature="sig_event"),
        collected_text=[],
        collected_thinking=[],
        collected_tool_calls={},
        usage_tokens={},
        progress_callback=None,
        thinking_signature_ref=thinking_signature_ref,
        current_block_index_ref=[-1],
        current_block_type_ref=[None],
    )
    assert thinking_signature_ref[0] == "sig_event"


def test_classify_anthropic_error_accepts_provider_mapped_errors() -> None:
    code, message = _classify_anthropic_error(ProviderTimeoutError("Request timed out: x"))
    assert code == "timeout"
    assert "timed out" in message

    code, message = _classify_anthropic_error(
        ProviderMappedError("rate_limit", "Rate limit exceeded: x")
    )
    assert code == "rate_limit"
    assert "Rate limit exceeded" in message
