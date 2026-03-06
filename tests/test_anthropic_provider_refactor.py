"""Focused regression tests for Anthropic provider stream helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import AsyncGenerator

import pytest
from pydantic import BaseModel

from ripperdoc.core.message_utils import build_anthropic_tool_schemas
from ripperdoc.core.tool import Tool, ToolOutput, ToolUseContext
from ripperdoc.core.providers.anthropic import (
    AnthropicClient,
    _classify_anthropic_error,
    _content_blocks_from_response,
    _content_blocks_from_stream_state,
)
from ripperdoc.core.message_utils import (
    ANTHROPIC_SYSTEM_PROMPT_DYNAMIC_BOUNDARY,
    anthropic_cache_control,
    apply_anthropic_prompt_cache_control_to_messages,
    apply_anthropic_prompt_cache_control_to_tool_schemas,
    build_anthropic_system_blocks,
)
from ripperdoc.core.providers.errors import ProviderMappedError, ProviderTimeoutError
from ripperdoc.tools.memory_tool import MemoryTool


class _DummyInput(BaseModel):
    value: str = "ok"


class _DummyTool(Tool[_DummyInput, str]):
    @property
    def name(self) -> str:
        return "DummyTool"

    async def description(self) -> str:
        return "dummy description"

    @property
    def input_schema(self) -> type[_DummyInput]:
        return _DummyInput

    async def prompt(self, yolo_mode: bool = False) -> str:  # noqa: ARG002
        return ""

    def render_result_for_assistant(self, output: str) -> str:
        return output

    def render_tool_use_message(self, input_data: _DummyInput, verbose: bool = False) -> str:  # noqa: ARG002
        return f"dummy:{input_data.value}"

    async def call(
        self, input_data: _DummyInput, context: ToolUseContext  # noqa: ARG002
    ) -> AsyncGenerator[ToolOutput, None]:
        del input_data
        if False:
            yield


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


def test_content_blocks_from_stream_state_supports_server_tool_and_search_result() -> None:
    blocks = _content_blocks_from_stream_state(
        collected_text=[],
        collected_thinking=[],
        collected_tool_calls={
            0: {
                "type": "server_tool_use",
                "id": "srvtoolu_01ABC123",
                "name": "tool_search_tool_regex",
                "input": {"query": "weather"},
            },
            1: {
                "type": "tool_search_tool_result",
                "tool_use_id": "srvtoolu_01ABC123",
                "content": {
                    "type": "tool_search_tool_search_result",
                    "tool_references": [
                        {"type": "tool_reference", "tool_name": "mcp__weather__get_weather"}
                    ],
                },
            },
        },
    )

    assert blocks[0]["type"] == "server_tool_use"
    assert blocks[0]["id"] == "srvtoolu_01ABC123"
    assert blocks[1]["type"] == "tool_search_tool_result"
    assert blocks[1]["content"]["tool_references"][0]["tool_name"] == "mcp__weather__get_weather"


def test_content_blocks_from_response_supports_tool_reference_family_blocks() -> None:
    response = SimpleNamespace(
        content=[
            SimpleNamespace(
                type="server_tool_use",
                id="srvtoolu_01ABC123",
                name="tool_search_tool_regex",
                input={"query": "weather"},
            ),
            SimpleNamespace(
                type="tool_search_tool_result",
                tool_use_id="srvtoolu_01ABC123",
                content={
                    "type": "tool_search_tool_search_result",
                    "tool_references": [
                        {"type": "tool_reference", "tool_name": "mcp__weather__get_weather"}
                    ],
                },
            ),
            SimpleNamespace(type="tool_reference", tool_name="mcp__weather__get_weather"),
        ]
    )

    blocks = _content_blocks_from_response(response)
    assert blocks[0]["type"] == "server_tool_use"
    assert blocks[1]["type"] == "tool_search_tool_result"
    assert blocks[1]["content"]["tool_references"][0]["tool_name"] == "mcp__weather__get_weather"
    assert blocks[2] == {"type": "tool_reference", "tool_name": "mcp__weather__get_weather"}


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


@pytest.mark.asyncio
async def test_build_anthropic_tool_schemas_uses_standard_memory_tool_shape(tmp_path) -> None:
    memory_tool = MemoryTool(memory_dir=tmp_path / "memory")
    schemas = await build_anthropic_tool_schemas([memory_tool])
    assert len(schemas) == 1
    schema = schemas[0]
    assert schema["name"] == "Memory"
    assert "input_schema" in schema


@pytest.mark.asyncio
async def test_build_anthropic_tool_schemas_keeps_function_tool_shape() -> None:
    schemas = await build_anthropic_tool_schemas([_DummyTool()])
    assert len(schemas) == 1
    schema = schemas[0]
    assert schema["name"] == "DummyTool"
    assert "input_schema" in schema
    assert schema["input_schema"]["type"] == "object"


def test_build_anthropic_system_blocks_marks_whole_prompt_cacheable() -> None:
    blocks = build_anthropic_system_blocks(
        "You are a coding assistant.",
        enable_prompt_caching=True,
    )

    assert isinstance(blocks, list)
    assert blocks == [
        {
            "type": "text",
            "text": "You are a coding assistant.",
            "cache_control": anthropic_cache_control(),
        }
    ]


def test_build_anthropic_system_blocks_respects_dynamic_boundary() -> None:
    system_prompt = (
        "Static prelude\n"
        f"{ANTHROPIC_SYSTEM_PROMPT_DYNAMIC_BOUNDARY}\n"
        "Dynamic tail"
    )

    blocks = build_anthropic_system_blocks(system_prompt, enable_prompt_caching=True)

    assert isinstance(blocks, list)
    assert blocks[0]["text"] == "Static prelude"
    assert blocks[0]["cache_control"] == anthropic_cache_control()
    assert blocks[1] == {"type": "text", "text": "Dynamic tail"}


def test_apply_anthropic_prompt_cache_control_to_messages_marks_last_two_messages() -> None:
    messages = [
        {"role": "user", "content": "old user"},
        {"role": "assistant", "content": [{"type": "text", "text": "old assistant"}]},
        {"role": "user", "content": "recent user"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "step", "signature": "sig"},
                {"type": "text", "text": "recent assistant"},
            ],
        },
    ]

    shaped = apply_anthropic_prompt_cache_control_to_messages(
        messages,
        enable_prompt_caching=True,
    )

    assert shaped[0]["content"] == "old user"
    assert shaped[1]["content"] == [{"type": "text", "text": "old assistant"}]
    assert shaped[2]["content"][0]["cache_control"] == anthropic_cache_control()
    assert shaped[3]["content"][0]["type"] == "thinking"
    assert "cache_control" not in shaped[3]["content"][0]
    assert shaped[3]["content"][1]["cache_control"] == anthropic_cache_control()


def test_apply_anthropic_prompt_cache_control_to_tool_schemas_marks_all_tools() -> None:
    schemas = [{"name": "Read", "input_schema": {"type": "object"}}]

    shaped = apply_anthropic_prompt_cache_control_to_tool_schemas(
        schemas,
        enable_prompt_caching=True,
    )

    assert shaped[0]["cache_control"] == anthropic_cache_control()
    assert "cache_control" not in schemas[0]
