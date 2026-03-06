"""Focused regression tests for query loop helpers and stop-path behavior."""

from __future__ import annotations

from typing import Any, AsyncGenerator, List

import pytest
from pydantic import BaseModel

from ripperdoc.core.config import ModelProfile, ProtocolType
from ripperdoc.core.providers.base import ProviderResponse
from ripperdoc.core.tool import Tool, ToolOutput, ToolResult, ToolUseContext
from ripperdoc.core.query import QueryContext, query
from ripperdoc.core.query import loop as loop_module
from ripperdoc.tools.tool_search_tool import ToolSearchTool
from ripperdoc.utils.messaging.messages import (
    create_assistant_message,
    create_hook_additional_context_message,
    create_plan_mode_attachment_message,
    create_user_message,
)


class _ToolInput(BaseModel):
    payload: str = ""


class _ActiveTool(Tool[_ToolInput, str]):
    @property
    def name(self) -> str:
        return "ActiveTool"

    async def description(self) -> str:
        return "active"

    @property
    def input_schema(self) -> type[_ToolInput]:
        return _ToolInput

    async def prompt(self, yolo_mode: bool = False) -> str:  # noqa: ARG002
        return ""

    def render_result_for_assistant(self, output: str) -> str:
        return output

    def render_tool_use_message(self, input_data: _ToolInput, verbose: bool = False) -> str:  # noqa: ARG002
        return "active"

    async def call(  # type: ignore[override]
        self,
        input_data: _ToolInput,  # noqa: ARG002
        context: ToolUseContext,  # noqa: ARG002
    ) -> AsyncGenerator[ToolOutput, None]:
        yield ToolResult(data="ok", result_for_assistant="ok")


class _DeferredMcpTool(Tool[_ToolInput, str]):
    is_mcp = True

    @property
    def name(self) -> str:
        return "mcp__demo__heavy_tool"

    async def description(self) -> str:
        return "deferred mcp"

    @property
    def input_schema(self) -> type[_ToolInput]:
        return _ToolInput

    async def prompt(self, yolo_mode: bool = False) -> str:  # noqa: ARG002
        return ""

    def defer_loading(self) -> bool:
        return True

    def render_result_for_assistant(self, output: str) -> str:
        return output

    def render_tool_use_message(self, input_data: _ToolInput, verbose: bool = False) -> str:  # noqa: ARG002
        return "deferred"

    async def call(  # type: ignore[override]
        self,
        input_data: _ToolInput,  # noqa: ARG002
        context: ToolUseContext,  # noqa: ARG002
    ) -> AsyncGenerator[ToolOutput, None]:
        yield ToolResult(data="ok", result_for_assistant="ok")


class _SecondActiveTool(_ActiveTool):
    @property
    def name(self) -> str:
        return "SecondActiveTool"


class _SecondDeferredMcpTool(_DeferredMcpTool):
    @property
    def name(self) -> str:
        return "mcp__demo__second_heavy_tool"


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


def test_build_iteration_plan_enable_tool_search_true_hides_deferred_mcp(monkeypatch: Any) -> None:
    query_context = QueryContext(tools=[_ActiveTool(), ToolSearchTool(), _DeferredMcpTool()])
    monkeypatch.setenv("ENABLE_TOOL_SEARCH", "true")
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
        system_prompt="system",
        context={},
        query_context=query_context,
    )

    names = {tool.name for tool in plan.tools_for_model}
    assert "ActiveTool" in names
    assert "ToolSearch" in names
    assert "mcp__demo__heavy_tool" not in names


def test_build_iteration_plan_enable_tool_search_false_keeps_deferred_mcp(monkeypatch: Any) -> None:
    query_context = QueryContext(tools=[_ActiveTool(), ToolSearchTool(), _DeferredMcpTool()])
    monkeypatch.setenv("ENABLE_TOOL_SEARCH", "false")
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
        system_prompt="system",
        context={},
        query_context=query_context,
    )

    names = {tool.name for tool in plan.tools_for_model}
    assert "ToolSearch" not in names
    assert "mcp__demo__heavy_tool" in names


def test_build_iteration_plan_injects_full_plan_mode_attachment(monkeypatch: Any, tmp_path) -> None:
    query_context = QueryContext(
        tools=[_ActiveTool()],
        permission_mode="plan",
        working_directory=str(tmp_path),
    )
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
        messages=[create_user_message("start planning")],
        system_prompt="system",
        context={},
        query_context=query_context,
    )

    assert len(plan.plan_mode_messages) == 1
    attachment = plan.plan_mode_messages[0]
    assert attachment.type == "attachment"
    assert attachment.metadata["plan_mode_attachment"] is True
    assert attachment.metadata["plan_mode_reminder_type"] == "full"
    assert "Plan mode is active." in str(attachment.content)


def test_build_iteration_plan_skips_recent_plan_mode_attachment(monkeypatch: Any, tmp_path) -> None:
    query_context = QueryContext(
        tools=[_ActiveTool()],
        permission_mode="plan",
        working_directory=str(tmp_path),
    )
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

    prior_attachment = create_plan_mode_attachment_message(
        "full reminder",
        plan_file_path=query_context.plan_file_path or str(tmp_path / "main.md"),
        reminder_type="full",
    )
    messages = [
        create_user_message("start planning"),
        prior_attachment,
        create_assistant_message("first reply"),
    ]

    plan = loop_module._build_iteration_plan(
        messages=messages,
        system_prompt="system",
        context={},
        query_context=query_context,
    )

    assert plan.plan_mode_messages == []


def test_build_iteration_plan_uses_sparse_attachment_after_five_turns(monkeypatch: Any, tmp_path) -> None:
    query_context = QueryContext(
        tools=[_ActiveTool()],
        permission_mode="plan",
        working_directory=str(tmp_path),
    )
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

    prior_attachment = create_plan_mode_attachment_message(
        "full reminder",
        plan_file_path=query_context.plan_file_path or str(tmp_path / "main.md"),
        reminder_type="full",
    )
    messages = [create_user_message("start planning"), prior_attachment]
    messages.extend(create_assistant_message(f"reply {idx}") for idx in range(5))

    plan = loop_module._build_iteration_plan(
        messages=messages,
        system_prompt="system",
        context={},
        query_context=query_context,
    )

    assert len(plan.plan_mode_messages) == 1
    attachment = plan.plan_mode_messages[0]
    assert attachment.metadata["plan_mode_reminder_type"] == "sparse"
    assert "Plan mode still active" in str(attachment.content)


def test_build_iteration_plan_enable_tool_search_auto_uses_threshold(monkeypatch: Any) -> None:
    monkeypatch.setenv("ENABLE_TOOL_SEARCH", "auto:50")
    monkeypatch.setattr(
        loop_module,
        "resolve_model_profile",
        lambda _model: ModelProfile(
            protocol=ProtocolType.ANTHROPIC,
            model="claude-sonnet-4-20250514",
            max_input_tokens=4000,
        ),
    )
    monkeypatch.setattr(loop_module, "determine_tool_mode", lambda _profile: "native")

    query_context = QueryContext(tools=[_ActiveTool(), ToolSearchTool(), _DeferredMcpTool()])
    plan = loop_module._build_iteration_plan(
        system_prompt="system",
        context={},
        query_context=query_context,
    )
    names = {tool.name for tool in plan.tools_for_model}
    assert "ToolSearch" not in names
    assert "mcp__demo__heavy_tool" in names

    monkeypatch.setenv("ENABLE_TOOL_SEARCH", "auto:0")
    # New query context should re-evaluate against new threshold.
    query_context_new = QueryContext(tools=[_ActiveTool(), ToolSearchTool(), _DeferredMcpTool()])
    plan_on = loop_module._build_iteration_plan(
        system_prompt="system",
        context={},
        query_context=query_context_new,
    )
    names_on = {tool.name for tool in plan_on.tools_for_model}
    assert "ToolSearch" in names_on
    assert "mcp__demo__heavy_tool" not in names_on


def test_build_iteration_plan_enable_tool_search_invalid_value_falls_back_auto(monkeypatch: Any) -> None:
    query_context = QueryContext(tools=[_ActiveTool(), ToolSearchTool(), _DeferredMcpTool()])
    monkeypatch.setenv("ENABLE_TOOL_SEARCH", "auto:not-a-number")
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
        system_prompt="system",
        context={},
        query_context=query_context,
    )

    # auto:10 fallback, and this small tool payload stays below threshold.
    names = {tool.name for tool in plan.tools_for_model}
    assert "ToolSearch" not in names
    assert "mcp__demo__heavy_tool" in names


def test_build_iteration_plan_tool_search_decision_is_sticky_within_query_context(
    monkeypatch: Any,
) -> None:
    query_context = QueryContext(tools=[_ActiveTool(), ToolSearchTool(), _DeferredMcpTool()])
    monkeypatch.setenv("ENABLE_TOOL_SEARCH", "auto:50")
    monkeypatch.setattr(
        loop_module,
        "resolve_model_profile",
        lambda _model: ModelProfile(
            protocol=ProtocolType.ANTHROPIC,
            model="claude-sonnet-4-20250514",
            max_input_tokens=100,
        ),
    )
    monkeypatch.setattr(loop_module, "determine_tool_mode", lambda _profile: "native")
    monkeypatch.setattr(loop_module, "get_model_context_limit", lambda _profile: 100)

    calls = {"count": 0}

    def _fake_estimate(_tools: list[Any]) -> int:
        calls["count"] += 1
        # First pass enables auto mode (60 >= 50 threshold); second pass would disable if re-evaluated.
        return 60 if calls["count"] == 1 else 10

    monkeypatch.setattr(loop_module, "_estimate_deferred_mcp_schema_tokens", _fake_estimate)

    first_plan = loop_module._build_iteration_plan(
        system_prompt="system",
        context={},
        query_context=query_context,
    )
    first_names = {tool.name for tool in first_plan.tools_for_model}
    assert "ToolSearch" in first_names
    assert "mcp__demo__heavy_tool" not in first_names

    second_plan = loop_module._build_iteration_plan(
        system_prompt="system",
        context={},
        query_context=query_context,
    )
    second_names = {tool.name for tool in second_plan.tools_for_model}
    assert "ToolSearch" in second_names
    assert "mcp__demo__heavy_tool" not in second_names
    assert calls["count"] == 1


def test_query_context_tools_setter_noop_keeps_cached_tool_search_decision() -> None:
    query_context = QueryContext(tools=[_ActiveTool(), ToolSearchTool(), _DeferredMcpTool()])
    query_context.mcp_tool_search_enabled = True
    query_context.mcp_tool_search_reason = "cached"

    # New instances but identical inventory should be treated as no-op.
    query_context.tools = [_ActiveTool(), ToolSearchTool(), _DeferredMcpTool()]

    assert query_context.mcp_tool_search_enabled is True
    assert query_context.mcp_tool_search_reason == "cached"


def test_query_context_tools_setter_keeps_cached_decision_when_only_non_deferred_changes() -> None:
    query_context = QueryContext(tools=[_ActiveTool(), ToolSearchTool(), _DeferredMcpTool()])
    query_context.mcp_tool_search_enabled = True
    query_context.mcp_tool_search_reason = "cached"

    query_context.tools = [
        _ActiveTool(),
        _SecondActiveTool(),
        ToolSearchTool(),
        _DeferredMcpTool(),
    ]

    assert query_context.mcp_tool_search_enabled is True
    assert query_context.mcp_tool_search_reason == "cached"


def test_query_context_tools_setter_resets_cached_decision_when_deferred_changes() -> None:
    query_context = QueryContext(tools=[_ActiveTool(), ToolSearchTool(), _DeferredMcpTool()])
    query_context.mcp_tool_search_enabled = True
    query_context.mcp_tool_search_reason = "cached"

    query_context.tools = [
        _ActiveTool(),
        ToolSearchTool(),
        _DeferredMcpTool(),
        _SecondDeferredMcpTool(),
    ]

    assert query_context.mcp_tool_search_enabled is None
    assert query_context.mcp_tool_search_reason is None


def test_query_context_tools_setter_resets_cached_decision_when_tool_search_removed() -> None:
    query_context = QueryContext(tools=[_ActiveTool(), ToolSearchTool(), _DeferredMcpTool()])
    query_context.mcp_tool_search_enabled = True
    query_context.mcp_tool_search_reason = "cached"

    query_context.tools = [_ActiveTool(), _DeferredMcpTool()]

    assert query_context.mcp_tool_search_enabled is None
    assert query_context.mcp_tool_search_reason is None


def test_build_iteration_plan_enable_tool_search_true_without_tool_search_tool_falls_back_standard(
    monkeypatch: Any,
) -> None:
    query_context = QueryContext(tools=[_ActiveTool(), _DeferredMcpTool()])
    monkeypatch.setenv("ENABLE_TOOL_SEARCH", "true")
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
        system_prompt="system",
        context={},
        query_context=query_context,
    )

    names = {tool.name for tool in plan.tools_for_model}
    assert "mcp__demo__heavy_tool" in names


def test_build_iteration_plan_enable_tool_search_uses_discovered_tools_from_history(
    monkeypatch: Any,
) -> None:
    query_context = QueryContext(tools=[_ActiveTool(), ToolSearchTool(), _DeferredMcpTool()])
    monkeypatch.setenv("ENABLE_TOOL_SEARCH", "true")
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

    messages = [
        create_user_message(
            [{"type": "tool_result", "tool_use_id": "tool-search-1", "text": "toolsearch-result"}],
            tool_use_result={
                "matches": [
                    {
                        "name": "mcp__demo__heavy_tool",
                        "deferred": True,
                    }
                ],
                "activated": [],
                "deferred_remaining": 0,
            },
        )
    ]
    plan = loop_module._build_iteration_plan(
        messages=messages,
        system_prompt="system",
        context={},
        query_context=query_context,
    )

    names = {tool.name for tool in plan.tools_for_model}
    assert "ToolSearch" in names
    assert "mcp__demo__heavy_tool" in names


def test_build_iteration_plan_enable_tool_search_uses_tool_reference_blocks(
    monkeypatch: Any,
) -> None:
    query_context = QueryContext(tools=[_ActiveTool(), ToolSearchTool(), _DeferredMcpTool()])
    monkeypatch.setenv("ENABLE_TOOL_SEARCH", "true")
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

    messages = [
        create_assistant_message(
            [
                {
                    "type": "tool_search_tool_result",
                    "tool_use_id": "srvtoolu_01ABC123",
                    "content": {
                        "type": "tool_search_tool_search_result",
                        "tool_references": [
                            {"type": "tool_reference", "tool_name": "mcp__demo__heavy_tool"}
                        ],
                    },
                }
            ]
        )
    ]
    plan = loop_module._build_iteration_plan(
        messages=messages,
        system_prompt="system",
        context={},
        query_context=query_context,
    )

    names = {tool.name for tool in plan.tools_for_model}
    assert "ToolSearch" in names
    assert "mcp__demo__heavy_tool" in names


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


def test_compose_next_iteration_messages_reorders_tool_hook_lifecycle() -> None:
    base_messages = [
        create_user_message("start"),
        create_assistant_message(
            [{"type": "tool_use", "id": "call_1", "name": "Read", "input": {"file": "a.txt"}}]
        ),
    ]
    pre_msg = create_hook_additional_context_message(
        "pre",
        hook_name="PreToolUse:Read",
        hook_event="PreToolUse",
        parent_tool_use_id="call_1",
    )
    post_msg = create_hook_additional_context_message(
        "post",
        hook_name="PostToolUse:Read",
        hook_event="PostToolUse",
        parent_tool_use_id="call_1",
    )
    assert pre_msg is not None
    assert post_msg is not None
    tool_result_msg = create_user_message(
        [{"type": "tool_result", "tool_use_id": "call_1", "text": "ok"}],
        parent_tool_use_id="call_1",
    )

    composed = loop_module._compose_next_iteration_messages(
        base_messages,
        assistant_message=None,
        tool_results=[post_msg, tool_result_msg, pre_msg],
    )

    assert len(composed) == 5
    assert getattr(composed[0], "type", None) == "user"
    assert getattr(composed[1], "type", None) == "assistant"
    assert (
        isinstance(composed[2], type(pre_msg))
        and composed[2].metadata.get("hook_event") == "PreToolUse"
    )
    assert isinstance(composed[3], type(tool_result_msg))
    assert isinstance(composed[3].message.content, list)
    assert composed[3].message.content[0].type == "tool_result"
    assert (
        isinstance(composed[4], type(post_msg))
        and composed[4].metadata.get("hook_event") == "PostToolUse"
    )


def test_compose_next_iteration_messages_drops_unmatched_tool_use_only_assistant() -> None:
    composed = loop_module._compose_next_iteration_messages(
        [create_user_message("start")],
        assistant_message=create_assistant_message(
            [{"type": "tool_use", "id": "orphan_call", "name": "Read", "input": {}}]
        ),
        tool_results=[],
    )

    assert len(composed) == 1
    assert getattr(composed[0], "type", None) == "user"
