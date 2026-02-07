"""Focused regression tests for query loop helpers and stop-path behavior."""

from __future__ import annotations

from typing import Any, List

import pytest

from ripperdoc.core.config import ModelProfile, ProtocolType
from ripperdoc.core.providers.base import ProviderResponse
from ripperdoc.core.query import QueryContext, query
from ripperdoc.core.query import loop as loop_module
from ripperdoc.utils.messages import create_assistant_message, create_user_message


def test_infer_thinking_mode_respects_explicit_disable() -> None:
    profile = ModelProfile(
         protocol=ProtocolType.OPENAI_COMPATIBLE,
        model="deepseek-reasoner",
        thinking_mode="off",
    )
    assert loop_module.infer_thinking_mode(profile) is None


@pytest.mark.parametrize(
    ("api_base", "model_name", "expected"),
    [
        ("https://api.deepseek.com", "deepseek-chat", "deepseek"),
        ("https://dashscope.aliyuncs.com", "qwen-max", "qwen"),
        ("https://openrouter.ai/api/v1", "openai/gpt-4o", "openrouter"),
        ("https://generativelanguage.googleapis.com", "gemini-2.5-pro", "gemini_openai"),
    ],
)
def test_infer_thinking_mode_auto_detection(api_base: str, model_name: str, expected: str) -> None:
    profile = ModelProfile(
         protocol=ProtocolType.OPENAI_COMPATIBLE,
        model=model_name,
        api_base=api_base,
    )
    assert loop_module.infer_thinking_mode(profile) == expected


@pytest.mark.asyncio
async def test_query_llm_uses_reasoning_mode_for_reasoner_models(monkeypatch: Any) -> None:
    profile = ModelProfile(
         protocol=ProtocolType.OPENAI_COMPATIBLE,
        model="deepseek-reasoner",
        api_base="https://api.deepseek.com",
    )

    captured: dict[str, Any] = {}

    def fake_resolve_model_profile(_model: str) -> ModelProfile:
        return profile

    def fake_provider_protocol(_provider: ProtocolType) -> str:
        return "openai"

    def fake_determine_tool_mode(_profile: ModelProfile) -> str:
        return "native"

    def fake_normalize_messages_for_api(
        messages: list[Any],
        protocol: str,
        tool_mode: str,
        thinking_mode: str | None,
    ) -> list[dict[str, Any]]:
        captured["thinking_mode"] = thinking_mode
        captured["protocol"] = protocol
        captured["tool_mode"] = tool_mode
        captured["messages_count"] = len(messages)
        return [{"role": "user", "content": "hello"}]

    class FakeClient:
        async def call(self, **kwargs: Any) -> ProviderResponse:
            captured["normalized_messages"] = kwargs["normalized_messages"]
            return ProviderResponse(
                content_blocks=[{"type": "text", "text": "ok"}],
                usage_tokens={"input_tokens": 1, "output_tokens": 1},
                cost_usd=0.001,
                duration_ms=10.0,
                metadata={"reasoning_content": "x"},
            )

    monkeypatch.setattr(loop_module, "resolve_model_profile", fake_resolve_model_profile)
    monkeypatch.setattr(loop_module, "provider_protocol", fake_provider_protocol)
    monkeypatch.setattr(loop_module, "determine_tool_mode", fake_determine_tool_mode)
    monkeypatch.setattr(loop_module, "normalize_messages_for_api", fake_normalize_messages_for_api)
    monkeypatch.setattr(loop_module, "get_provider_client", lambda _provider: FakeClient())

    result = await loop_module.query_llm(
        [create_user_message("hi")],
        "system",
        tools=[],
        max_thinking_tokens=0,
    )

    assert captured["thinking_mode"] == "deepseek"
    assert captured["protocol"] == "openai"
    assert captured["tool_mode"] == "native"
    assert captured["normalized_messages"] == [{"role": "user", "content": "hello"}]
    assert isinstance(result.message.content, list)
    assert result.message.content[0].type == "text"
    assert result.message.content[0].text == "ok"


@pytest.mark.asyncio
async def test_query_continues_when_trailing_pending_messages_arrive(monkeypatch: Any) -> None:
    query_context = QueryContext(tools=[], yolo_mode=True, verbose=False)

    class DummyProfile:
        model = "dummy"

    monkeypatch.setattr(loop_module, "resolve_model_profile", lambda _model: DummyProfile())

    async def fake_run_query_iteration(
        messages: List[Any],
        system_prompt: str,
        context: dict[str, str],
        query_context_obj: QueryContext,
        can_use_tool_fn: Any,
        iteration: int,
        result: Any,
    ):
        del messages, system_prompt, context, can_use_tool_fn
        if iteration == 1:
            first = create_assistant_message("first")
            result.assistant_message = first
            result.tool_results = []
            result.should_stop = True
            query_context_obj.enqueue_user_message("late-interjection")
            yield first
        else:
            second = create_assistant_message("second")
            result.assistant_message = second
            result.tool_results = []
            result.should_stop = True
            yield second

    monkeypatch.setattr(loop_module, "_run_query_iteration", fake_run_query_iteration)

    collected: list[Any] = []
    async for item in query(
        messages=[create_user_message("start")],
        system_prompt="system",
        context={},
        query_context=query_context,
        can_use_tool_fn=None,
    ):
        collected.append(item)

    kinds = [getattr(msg, "type", None) for msg in collected]
    assert kinds == ["assistant", "user", "assistant"]
    assert getattr(collected[1].message, "content", None) == "late-interjection"
