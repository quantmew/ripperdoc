"""Focused regression tests for OpenAI provider refactor helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import httpx
import openai
import pytest

from ripperdoc.core.providers.base import call_with_timeout_and_retries
from ripperdoc.core.config import ModelProfile, ProtocolType
from ripperdoc.core.oauth import OAuthToken, OAuthTokenType
from ripperdoc.core.providers.errors import ProviderMappedError
from ripperdoc.core.providers.openai import (
    OpenAIClient,
    _StreamAccumulator,
    _build_codex_oauth_tools,
    _build_codex_responses_input,
    _extract_content_blocks_from_responses_payload,
    _extract_from_codex_sse_events,
    _build_non_stream_empty_or_error_response,
    _parse_sse_json_events,
    _build_stream_content_blocks,
    _consume_stream_chunk,
    _run_with_provider_error_mapping,
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


@pytest.mark.asyncio
async def test_timeout_connection_error_reuses_shared_retry_logic(monkeypatch) -> None:
    attempts = {"count": 0}

    async def _no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr("ripperdoc.core.providers.base.asyncio.sleep", _no_sleep)

    async def flaky_request() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise openai.APIConnectionError(
                message="Request timed out.",
                request=httpx.Request("POST", "https://example.com/v1/chat/completions"),
            )
        return "ok"

    result = await call_with_timeout_and_retries(
        lambda: _run_with_provider_error_mapping(flaky_request),
        request_timeout=None,
        max_retries=5,
    )

    assert result == "ok"
    assert attempts["count"] == 3


@pytest.mark.asyncio
async def test_non_timeout_connection_error_does_not_retry(monkeypatch) -> None:
    attempts = {"count": 0}

    async def _no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr("ripperdoc.core.providers.base.asyncio.sleep", _no_sleep)

    async def bad_request() -> str:
        attempts["count"] += 1
        raise openai.APIConnectionError(
            message="Connection error.",
            request=httpx.Request("POST", "https://example.com/v1/chat/completions"),
        )

    with pytest.raises(ProviderMappedError) as exc_info:
        await call_with_timeout_and_retries(
            lambda: _run_with_provider_error_mapping(bad_request),
            request_timeout=None,
            max_retries=5,
        )

    assert attempts["count"] == 1
    assert exc_info.value.error_code == "connection_error"


@pytest.mark.asyncio
async def test_rate_limit_is_mapped_to_provider_error() -> None:
    request = httpx.Request("POST", "https://example.com/v1/chat/completions")
    response = httpx.Response(status_code=429, request=request)
    err = openai.RateLimitError("Too many requests", response=response, body={})

    async def bad_request() -> str:
        raise err

    with pytest.raises(ProviderMappedError) as exc_info:
        await _run_with_provider_error_mapping(bad_request)

    assert exc_info.value.error_code == "rate_limit"


@pytest.mark.asyncio
async def test_timeout_exhaustion_surfaces_as_provider_timeout(monkeypatch) -> None:
    attempts = {"count": 0}

    async def _no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr("ripperdoc.core.providers.base.asyncio.sleep", _no_sleep)

    async def always_timeout() -> str:
        attempts["count"] += 1
        raise openai.APIConnectionError(
            message="Request timed out.",
            request=httpx.Request("POST", "https://example.com/v1/chat/completions"),
        )

    with pytest.raises(ProviderMappedError) as exc_info:
        await call_with_timeout_and_retries(
            lambda: _run_with_provider_error_mapping(always_timeout),
            request_timeout=None,
            max_retries=2,
        )

    assert exc_info.value.error_code == "timeout"
    assert attempts["count"] == 3


@pytest.mark.asyncio
async def test_remote_protocol_error_is_mapped_and_retried(monkeypatch) -> None:
    attempts = {"count": 0}

    async def _no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr("ripperdoc.core.providers.base.asyncio.sleep", _no_sleep)

    async def flaky_request() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise httpx.RemoteProtocolError(
                "peer closed connection without sending complete message body (incomplete chunked read)"
            )
        return "ok"

    result = await call_with_timeout_and_retries(
        lambda: _run_with_provider_error_mapping(flaky_request),
        request_timeout=None,
        max_retries=5,
    )

    assert result == "ok"
    assert attempts["count"] == 3


@pytest.mark.asyncio
async def test_oauth_profile_missing_token_returns_auth_error(monkeypatch) -> None:
    monkeypatch.setattr("ripperdoc.core.providers.openai.get_oauth_token", lambda _name: None)

    client = OpenAIClient()
    profile = ModelProfile(
        protocol=ProtocolType.OAUTH,
        model="gpt-5.3-codex",
        oauth_token_name="missing-token",
    )
    result = await client.call(
        model_profile=profile,
        system_prompt="You are a test assistant.",
        normalized_messages=[],
        tools=[],
        tool_mode="native",
        stream=False,
        progress_callback=None,
        request_timeout=1.0,
        max_retries=0,
        max_thinking_tokens=0,
    )

    assert result.is_error is True
    assert result.error_code == "authentication_error"


def test_build_codex_responses_input_flattens_messages() -> None:
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]},
        {"role": "tool", "content": [{"type": "tool_result", "content": "done"}]},
    ]

    payload = _build_codex_responses_input(messages)

    assert payload[0]["role"] == "user"
    assert payload[0]["content"][0]["type"] == "input_text"
    assert payload[1]["role"] == "assistant"
    assert payload[1]["content"][0]["type"] == "output_text"
    assert payload[2]["role"] == "assistant"
    assert payload[2]["content"][0]["text"] == "done"


def test_parse_codex_sse_events_and_extract_content() -> None:
    sse = (
        'event: response.output_text.delta\n'
        'data: {"type":"response.output_text.delta","delta":"Hel"}\n'
        "\n"
        'event: response.output_text.delta\n'
        'data: {"type":"response.output_text.delta","delta":"lo"}\n'
        "\n"
        'event: response.completed\n'
        'data: {"type":"response.completed","response":{"usage":{"prompt_tokens":2,"completion_tokens":3}}}\n'
        "\n"
        "data: [DONE]\n"
        "\n"
    )
    events = _parse_sse_json_events(sse)
    text, usage, final_response = _extract_from_codex_sse_events(events)

    assert len(events) == 3
    assert text == "Hello"
    assert usage["input_tokens"] == 2
    assert usage["output_tokens"] == 3
    assert isinstance(final_response, dict)


def test_build_codex_oauth_tools_converts_function_shape() -> None:
    openai_tools = [
        {
            "type": "function",
            "function": {
                "name": "Read",
                "description": "Read file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
        }
    ]
    converted = _build_codex_oauth_tools(openai_tools)
    assert converted[0]["type"] == "function"
    assert converted[0]["name"] == "Read"
    assert "function" not in converted[0]


def test_extract_content_blocks_from_responses_payload_includes_tool_use() -> None:
    payload = {
        "output": [
            {
                "type": "function_call",
                "name": "Read",
                "call_id": "call_1",
                "arguments": '{"path":"README.md"}',
            },
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "done"}],
            },
        ]
    }
    blocks = _extract_content_blocks_from_responses_payload(payload)
    assert blocks[0]["type"] == "tool_use"
    assert blocks[0]["name"] == "Read"
    assert blocks[0]["input"] == {"path": "README.md"}
    assert blocks[1] == {"type": "text", "text": "done"}


@pytest.mark.asyncio
async def test_oauth_codex_payload_sets_store_false(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeAsyncClient:
        def __init__(self, timeout: float = 0.0) -> None:  # noqa: ARG002
            return

        async def __aenter__(self) -> "_FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
            return False

        async def post(self, url: str, *, json: dict, headers: dict) -> httpx.Response:  # noqa: A002
            captured["payload"] = json
            captured["headers"] = headers
            return httpx.Response(
                status_code=200,
                request=httpx.Request("POST", url),
                json={
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": "ok"}],
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            )

    monkeypatch.setattr(
        "ripperdoc.core.providers.openai.httpx.AsyncClient",
        _FakeAsyncClient,
    )
    monkeypatch.setattr(
        "ripperdoc.core.providers.openai.get_oauth_token",
        lambda _name: OAuthToken(
            type=OAuthTokenType.CODEX,
            access_token="abcd1234efgh",
            refresh_token=None,
            expires_at=None,
            account_id="acct_1",
        ),
    )

    client = OpenAIClient()
    profile = ModelProfile(
        protocol=ProtocolType.OAUTH,
        model="gpt-5.3-codex",
        oauth_token_name="codex",
    )
    result = await client.call(
        model_profile=profile,
        system_prompt="You are a test assistant.",
        normalized_messages=[{"role": "user", "content": "hello"}],
        tools=[],
        tool_mode="native",
        stream=False,
        progress_callback=None,
        request_timeout=1.0,
        max_retries=0,
        max_thinking_tokens=0,
    )

    assert result.is_error is False
    assert isinstance(captured.get("payload"), dict)
    assert cast(dict[str, object], captured["payload"]).get("store") is False
    assert cast(dict[str, object], captured["payload"]).get("stream") is True
    assert "max_output_tokens" not in cast(dict[str, object], captured["payload"])


@pytest.mark.asyncio
async def test_oauth_codex_unsupported_parameter_is_removed_and_retried(monkeypatch) -> None:
    payloads: list[dict] = []
    calls = {"count": 0}

    class _FakeAsyncClient:
        def __init__(self, timeout: float = 0.0) -> None:  # noqa: ARG002
            return

        async def __aenter__(self) -> "_FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
            return False

        async def post(self, url: str, *, json: dict, headers: dict) -> httpx.Response:  # noqa: A002, ARG002
            calls["count"] += 1
            payloads.append(dict(json))
            if calls["count"] == 1:
                return httpx.Response(
                    status_code=400,
                    request=httpx.Request("POST", url),
                    json={"detail": "Unsupported parameter: temperature"},
                )
            return httpx.Response(
                status_code=200,
                request=httpx.Request("POST", url),
                json={
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": "ok"}],
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            )

    monkeypatch.setattr(
        "ripperdoc.core.providers.openai.httpx.AsyncClient",
        _FakeAsyncClient,
    )
    monkeypatch.setattr(
        "ripperdoc.core.providers.openai.get_oauth_token",
        lambda _name: OAuthToken(
            type=OAuthTokenType.CODEX,
            access_token="abcd1234efgh",
            refresh_token=None,
            expires_at=None,
            account_id="acct_1",
        ),
    )

    client = OpenAIClient()
    profile = ModelProfile(
        protocol=ProtocolType.OAUTH,
        model="gpt-5.3-codex",
        oauth_token_name="codex",
        temperature=0.7,
    )
    result = await client.call(
        model_profile=profile,
        system_prompt="You are a test assistant.",
        normalized_messages=[{"role": "user", "content": "hello"}],
        tools=[],
        tool_mode="native",
        stream=False,
        progress_callback=None,
        request_timeout=1.0,
        max_retries=0,
        max_thinking_tokens=0,
    )

    assert result.is_error is False
    assert calls["count"] == 2
    assert "temperature" in payloads[0]
    assert "temperature" not in payloads[1]
