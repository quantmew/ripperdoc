"""Focused regression tests for OpenAI provider refactor helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import httpx
import openai
import pytest

from ripperdoc.core.providers.base import ProviderResponse, call_with_timeout_and_retries
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
from ripperdoc.core.providers.openai_responses import (
    build_input_from_normalized_messages,
    extract_content_blocks_from_output,
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
    monkeypatch.setattr(
        "ripperdoc.core.providers.openai_oauth_codex.get_oauth_token",
        lambda _name: None,
    )

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


@pytest.mark.asyncio
async def test_non_oauth_mode_responses_uses_responses_api(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeResponseObject:
        usage = {"input_tokens": 2, "output_tokens": 3}
        output_text = "hello from responses"
        status = "completed"

        @staticmethod
        def model_dump(*, exclude_none: bool = True) -> dict[str, object]:  # noqa: ARG004
            return {
                "status": "completed",
                "usage": {"input_tokens": 2, "output_tokens": 3},
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "hello from responses"}],
                    }
                ],
            }

    class _FakeAsyncOpenAI:
        def __init__(self, **kwargs: object) -> None:  # noqa: ARG002
            self.responses = SimpleNamespace(create=self._responses_create)
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._chat_create),
            )

        async def __aenter__(self) -> "_FakeAsyncOpenAI":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
            return False

        async def _responses_create(self, **kwargs: object) -> _FakeResponseObject:
            captured["responses_kwargs"] = kwargs
            return _FakeResponseObject()

        async def _chat_create(self, **kwargs: object) -> object:  # noqa: ARG002
            raise AssertionError("chat.completions.create should not be used for mode=responses")

    monkeypatch.setattr(
        "ripperdoc.core.providers.openai_non_oauth_strategies.AsyncOpenAI",
        _FakeAsyncOpenAI,
    )

    client = OpenAIClient()
    profile = ModelProfile(
        protocol=ProtocolType.OPENAI_COMPATIBLE,
        model="gpt-5",
        mode="responses",
        api_key="sk-test",
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
    assert result.content_blocks == [{"type": "text", "text": "hello from responses"}]
    assert result.metadata.get("status") == "completed"
    kwargs = cast(dict[str, object], captured["responses_kwargs"])
    assert kwargs["model"] == "gpt-5"
    assert kwargs["instructions"] == "You are a test assistant."
    assert isinstance(kwargs.get("input"), list)
    assert "messages" not in kwargs


@pytest.mark.asyncio
async def test_non_oauth_mode_responses_assistant_history_uses_output_text_and_phase(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeResponseObject:
        usage = {"input_tokens": 1, "output_tokens": 1}
        output_text = "ok"
        status = "completed"

        @staticmethod
        def model_dump(*, exclude_none: bool = True) -> dict[str, object]:  # noqa: ARG004
            return {
                "status": "completed",
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "ok"}],
                    }
                ],
            }

    class _FakeAsyncOpenAI:
        def __init__(self, **kwargs: object) -> None:  # noqa: ARG002
            self.responses = SimpleNamespace(create=self._responses_create)
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat_create))

        async def __aenter__(self) -> "_FakeAsyncOpenAI":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
            return False

        async def _responses_create(self, **kwargs: object) -> _FakeResponseObject:
            captured["responses_kwargs"] = kwargs
            return _FakeResponseObject()

        async def _chat_create(self, **kwargs: object) -> object:  # noqa: ARG002
            raise AssertionError("chat.completions.create should not be used for mode=responses")

    monkeypatch.setattr(
        "ripperdoc.core.providers.openai_non_oauth_strategies.AsyncOpenAI",
        _FakeAsyncOpenAI,
    )

    client = OpenAIClient()
    profile = ModelProfile(
        protocol=ProtocolType.OPENAI_COMPATIBLE,
        model="gpt-5",
        mode="responses",
        api_key="sk-test",
    )
    result = await client.call(
        model_profile=profile,
        system_prompt="You are a test assistant.",
        normalized_messages=[{"role": "assistant", "content": "hello", "phase": "commentary"}],
        tools=[],
        tool_mode="native",
        stream=False,
        progress_callback=None,
        request_timeout=1.0,
        max_retries=0,
        max_thinking_tokens=0,
    )

    assert result.is_error is False
    kwargs = cast(dict[str, object], captured["responses_kwargs"])
    payload_input = cast(list[dict[str, object]], kwargs.get("input"))
    assert payload_input[0].get("phase") == "commentary"
    content_blocks = cast(list[dict[str, object]], payload_input[0].get("content"))
    assert content_blocks[0].get("type") == "output_text"


@pytest.mark.asyncio
async def test_non_oauth_mode_responses_error_payload_returns_error(monkeypatch) -> None:
    class _FakeResponseObject:
        usage = {"input_tokens": 0, "output_tokens": 0}
        output_text = ""
        status = "failed"

        @staticmethod
        def model_dump(*, exclude_none: bool = True) -> dict[str, object]:  # noqa: ARG004
            return {
                "status": "failed",
                "error": {"message": "Invalid prompt"},
            }

    class _FakeAsyncOpenAI:
        def __init__(self, **kwargs: object) -> None:  # noqa: ARG002
            self.responses = SimpleNamespace(create=self._responses_create)
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._chat_create),
            )

        async def __aenter__(self) -> "_FakeAsyncOpenAI":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
            return False

        async def _responses_create(self, **kwargs: object) -> _FakeResponseObject:  # noqa: ARG002
            return _FakeResponseObject()

        async def _chat_create(self, **kwargs: object) -> object:  # noqa: ARG002
            raise AssertionError("chat.completions.create should not be used for mode=responses")

    monkeypatch.setattr(
        "ripperdoc.core.providers.openai_non_oauth_strategies.AsyncOpenAI",
        _FakeAsyncOpenAI,
    )

    client = OpenAIClient()
    profile = ModelProfile(
        protocol=ProtocolType.OPENAI_COMPATIBLE,
        model="gpt-5",
        mode="responses",
        api_key="sk-test",
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

    assert result.is_error is True
    assert result.error_code == "api_error"
    assert result.error_message == "Invalid prompt"


@pytest.mark.asyncio
async def test_non_oauth_mode_chat_uses_chat_completions_api(monkeypatch) -> None:
    calls = {"chat": 0, "responses": 0}

    class _FakeAsyncOpenAI:
        def __init__(self, **kwargs: object) -> None:  # noqa: ARG002
            self.responses = SimpleNamespace(create=self._responses_create)
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._chat_create),
            )

        async def __aenter__(self) -> "_FakeAsyncOpenAI":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
            return False

        async def _responses_create(self, **kwargs: object) -> object:  # noqa: ARG002
            calls["responses"] += 1
            raise AssertionError("responses.create should not be used for mode=chat")

        async def _chat_create(self, **kwargs: object) -> object:
            calls["chat"] += 1
            return SimpleNamespace(
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(
                            content="hello from chat",
                            tool_calls=[],
                        ),
                    )
                ],
            )

    monkeypatch.setattr(
        "ripperdoc.core.providers.openai_non_oauth_strategies.AsyncOpenAI",
        _FakeAsyncOpenAI,
    )

    client = OpenAIClient()
    profile = ModelProfile(
        protocol=ProtocolType.OPENAI_COMPATIBLE,
        model="gpt-4o-mini",
        mode="chat",
        api_key="sk-test",
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
    assert calls["chat"] == 1
    assert calls["responses"] == 0


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


def test_build_responses_input_supports_assistant_phase_and_input_text() -> None:
    payload = build_input_from_normalized_messages(
        [
            {
                "role": "assistant",
                "content": "Interim thought",
                "phase": "commentary",
            }
        ],
        assistant_text_type="input_text",
        include_phase=True,
    )

    assert payload[0]["role"] == "assistant"
    assert payload[0]["content"][0]["type"] == "input_text"
    assert payload[0]["phase"] == "commentary"


def test_build_responses_input_converts_tool_history_to_structured_items() -> None:
    payload = build_input_from_normalized_messages(
        [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I will read the file."},
                    {
                        "type": "tool_use",
                        "tool_use_id": "call_1",
                        "name": "Read",
                        "input": {"path": "README.md"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_1",
                        "content": [{"type": "text", "text": "README content"}],
                    },
                    {"type": "text", "text": "continue"},
                ],
            },
        ],
        assistant_text_type="input_text",
        include_phase=False,
    )

    assert payload[0] == {
        "role": "assistant",
        "content": [{"type": "input_text", "text": "I will read the file."}],
    }
    assert payload[1] == {
        "type": "function_call",
        "call_id": "call_1",
        "name": "Read",
        "arguments": '{"path":"README.md"}',
    }
    assert payload[2] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "README content",
    }
    assert payload[3] == {
        "role": "user",
        "content": [{"type": "input_text", "text": "continue"}],
    }


def test_extract_content_blocks_from_output_handles_refusal_items() -> None:
    payload = {
        "output": [
            {
                "type": "message",
                "content": [{"type": "refusal", "refusal": "I can't help with that."}],
            }
        ]
    }
    blocks = extract_content_blocks_from_output(payload)

    assert blocks == [{"type": "text", "text": "I can't help with that."}]


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
        "ripperdoc.core.providers.openai_oauth_codex.httpx.AsyncClient",
        _FakeAsyncClient,
    )
    monkeypatch.setattr(
        "ripperdoc.core.providers.openai_oauth_codex.get_oauth_token",
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
        normalized_messages=[
            {"role": "assistant", "content": "hello", "phase": "commentary"},
        ],
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
    payload = cast(dict[str, object], captured["payload"])
    assert payload.get("store") is False
    assert payload.get("stream") is True
    assert "max_output_tokens" not in payload
    payload_input = cast(list[dict[str, object]], payload.get("input"))
    assert payload_input[0].get("phase") == "commentary"
    content_blocks = cast(list[dict[str, object]], payload_input[0].get("content"))
    assert content_blocks[0].get("type") == "output_text"


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
        "ripperdoc.core.providers.openai_oauth_codex.httpx.AsyncClient",
        _FakeAsyncClient,
    )
    monkeypatch.setattr(
        "ripperdoc.core.providers.openai_oauth_codex.get_oauth_token",
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


@pytest.mark.asyncio
async def test_oauth_copilot_dispatch_uses_openai_strategy(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeStrategy:
        name = "chat"

        async def call(self, **kwargs: object) -> ProviderResponse:
            captured.update(kwargs)
            return ProviderResponse(
                content_blocks=[{"type": "text", "text": "ok"}],
                usage_tokens={"input_tokens": 1, "output_tokens": 1},
                cost_usd=0.0,
                duration_ms=1.0,
                metadata={"strategy": "chat"},
            )

    monkeypatch.setattr(
        "ripperdoc.core.providers.openai.get_oauth_token",
        lambda _name: OAuthToken(
            type=OAuthTokenType.COPILOT,
            access_token="copilot_token",
            refresh_token="copilot_token",
            expires_at=None,
            account_id=None,
        ),
    )
    monkeypatch.setattr(
        "ripperdoc.core.providers.openai.build_non_oauth_openai_strategy",
        lambda **kwargs: _FakeStrategy(),  # noqa: ARG005
    )

    client = OpenAIClient()
    profile = ModelProfile(
        protocol=ProtocolType.OAUTH,
        model="gpt-4.1",
        oauth_token_name="copilot-main",
        oauth_token_type=OAuthTokenType.COPILOT,
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
    called_profile = cast(ModelProfile, captured["model_profile"])
    assert called_profile.protocol == ProtocolType.OPENAI_COMPATIBLE
    assert called_profile.api_key == "copilot_token"
    assert called_profile.api_base == "https://api.githubcopilot.com"
    called_headers = cast(dict[str, str], captured["default_headers"])
    assert called_headers["Openai-Intent"] == "conversation-edits"
    assert result.metadata["oauth_token_type"] == "copilot"


@pytest.mark.asyncio
async def test_oauth_gitlab_requires_gateway_base_url(monkeypatch) -> None:
    monkeypatch.delenv("GITLAB_AI_GATEWAY_URL", raising=False)
    monkeypatch.delenv("RIPPERDOC_GITLAB_API_BASE", raising=False)
    monkeypatch.setattr(
        "ripperdoc.core.providers.openai.get_oauth_token",
        lambda _name: OAuthToken(
            type=OAuthTokenType.GITLAB,
            access_token="gitlab_token",
            refresh_token="refresh_token",
            expires_at=None,
            account_id="https://gitlab.com",
        ),
    )

    client = OpenAIClient()
    profile = ModelProfile(
        protocol=ProtocolType.OAUTH,
        model="duo-chat-haiku-4-5",
        oauth_token_name="gitlab-main",
        oauth_token_type=OAuthTokenType.GITLAB,
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

    assert result.is_error is True
    assert result.error_code == "configuration_error"
