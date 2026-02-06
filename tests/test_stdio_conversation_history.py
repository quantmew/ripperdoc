"""Regression tests for stdio conversation history updates."""

from __future__ import annotations

import asyncio
from typing import Any, List

import pytest

from ripperdoc.core.config import ModelProfile, ProviderType
from ripperdoc.core.hooks.manager import hook_manager
from ripperdoc.protocol.stdio import handler as handler_module
from ripperdoc.protocol.stdio import handler_config, handler_query, handler_session
from ripperdoc.utils.messages import create_assistant_message, create_progress_message
from ripperdoc.core.query_utils import tool_result_message


def _patch_stdio_dependencies(monkeypatch, tools: List[Any]) -> None:
    monkeypatch.setattr(handler_session, "get_project_config", lambda _path: None)
    monkeypatch.setattr(handler_session, "get_effective_model_profile", lambda _model: object())
    monkeypatch.setattr(handler_session, "get_default_tools", lambda **_: tools)
    monkeypatch.setattr(handler_session, "format_mcp_instructions", lambda _servers: "")
    monkeypatch.setattr(handler_query, "format_mcp_instructions", lambda _servers: "")
    monkeypatch.setattr(handler_config, "build_system_prompt", lambda *_args, **_kwargs: "system")
    monkeypatch.setattr(handler_config, "build_memory_instructions", lambda: "")
    monkeypatch.setattr(handler_session, "list_custom_commands", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(handler_session, "list_slash_commands", lambda: [])

    async def fake_load_mcp_servers_async(_path):
        return []

    async def fake_load_dynamic_mcp_tools_async(_path):
        return []

    monkeypatch.setattr(handler_session, "load_mcp_servers_async", fake_load_mcp_servers_async)
    monkeypatch.setattr(handler_session, "load_dynamic_mcp_tools_async", fake_load_dynamic_mcp_tools_async)
    monkeypatch.setattr(handler_query, "load_mcp_servers_async", fake_load_mcp_servers_async)

    from ripperdoc.core import skills as skills_module

    class DummySkillResult:
        skills: list = []

    monkeypatch.setattr(skills_module, "load_all_skills", lambda _path: DummySkillResult())
    monkeypatch.setattr(skills_module, "build_skill_summary", lambda _skills: "")

    async def fake_session_start_async(_hook_event):
        class Dummy:
            outputs = []
            system_message = None
            additional_context = None

        return Dummy()

    async def fake_prompt_submit_async(_prompt):
        class Dummy:
            should_block = False
            block_reason = None
            system_message = None
            additional_context = None

        return Dummy()

    monkeypatch.setattr(hook_manager, "run_session_start_async", fake_session_start_async)
    monkeypatch.setattr(hook_manager, "run_user_prompt_submit_async", fake_prompt_submit_async)


def _messages_have_tool_result(messages: List[Any]) -> bool:
    for message in messages:
        if getattr(message, "type", None) != "user":
            continue
        msg = getattr(message, "message", None)
        content = getattr(msg, "content", None)
        if isinstance(content, list):
            for block in content:
                if getattr(block, "type", None) == "tool_result":
                    return True
    return False


@pytest.mark.asyncio
async def test_stdio_history_includes_tool_results(monkeypatch, tmp_path):
    _patch_stdio_dependencies(monkeypatch, tools=[])

    call_state = {"count": 0, "tool_result_seen": False}

    async def fake_query(messages, system_prompt, context, query_context, can_use_tool):  # noqa: ARG001
        call_state["count"] += 1
        if call_state["count"] == 2:
            call_state["tool_result_seen"] = _messages_have_tool_result(list(messages))
        if call_state["count"] == 1:
            yield create_assistant_message("ok")
            yield tool_result_message("tool1", "done")
        else:
            yield create_assistant_message("ok2")

    monkeypatch.setattr(handler_query, "query", fake_query)

    handler = handler_module.StdioProtocolHandler()
    handler._project_path = tmp_path

    async def noop_write_control_response(*_args, **_kwargs):
        return None

    async def noop_write_message_stream(*_args, **_kwargs):
        return None

    monkeypatch.setattr(handler, "_write_control_response", noop_write_control_response)
    monkeypatch.setattr(handler, "_write_message_stream", noop_write_message_stream)

    await handler._handle_initialize({"options": {"permission_mode": "bypassPermissions"}}, "init")

    await handler._handle_query({"prompt": "first"}, "q1")
    await handler._handle_query({"prompt": "second"}, "q2")

    assert call_state["count"] == 2
    assert call_state["tool_result_seen"] is True


@pytest.mark.asyncio
async def test_stdio_query_timeout_reports_timeout_not_name_error(monkeypatch, tmp_path):
    _patch_stdio_dependencies(monkeypatch, tools=[])

    async def fake_query(messages, system_prompt, context, query_context, can_use_tool):  # noqa: ARG001
        await asyncio.sleep(0.05)
        if False:
            yield create_assistant_message("never")

    monkeypatch.setattr(handler_query, "query", fake_query)
    monkeypatch.setattr(handler_query, "STDIO_QUERY_TIMEOUT_SEC", 0.01)

    handler = handler_module.StdioProtocolHandler()
    handler._project_path = tmp_path

    control_responses: list[dict[str, Any]] = []
    stream_messages: list[dict[str, Any]] = []

    async def capture_control(request_id: str, response: dict[str, Any] | None = None, error: str | None = None):
        control_responses.append({"request_id": request_id, "response": response, "error": error})

    async def capture_stream(payload: dict[str, Any]):
        stream_messages.append(payload)

    monkeypatch.setattr(handler, "_write_control_response", capture_control)
    monkeypatch.setattr(handler, "_write_message_stream", capture_stream)

    await handler._handle_initialize({"options": {"permission_mode": "bypassPermissions"}}, "init")
    await handler._handle_query({"prompt": "timeout test"}, "q_timeout")

    assert not any(
        response.get("error") and "name 'asyncio' is not defined" in response.get("error", "")
        for response in control_responses
    )

    result_messages = [msg for msg in stream_messages if msg.get("type") == "result"]
    assert result_messages
    assert "Query timed out" in (result_messages[-1].get("result") or "")


@pytest.mark.asyncio
async def test_stdio_result_cost_uses_model_profile_pricing(monkeypatch, tmp_path):
    _patch_stdio_dependencies(monkeypatch, tools=[])

    async def fake_query(messages, system_prompt, context, query_context, can_use_tool):  # noqa: ARG001
        yield create_assistant_message(
            "priced reply",
            input_tokens=1000,
            output_tokens=500,
            cache_read_tokens=100,
            cache_creation_tokens=50,
        )

    monkeypatch.setattr(handler_query, "query", fake_query)
    monkeypatch.setattr(
        handler_query,
        "resolve_model_profile",
        lambda _model: ModelProfile(
            provider=ProviderType.OPENAI_COMPATIBLE,
            model="priced-model",
            input_cost_per_million_tokens=10.0,
            output_cost_per_million_tokens=20.0,
        ),
    )

    handler = handler_module.StdioProtocolHandler()
    handler._project_path = tmp_path

    stream_messages: list[dict[str, Any]] = []

    async def noop_write_control_response(*_args, **_kwargs):
        return None

    async def capture_stream(payload: dict[str, Any]):
        stream_messages.append(payload)

    monkeypatch.setattr(handler, "_write_control_response", noop_write_control_response)
    monkeypatch.setattr(handler, "_write_message_stream", capture_stream)

    await handler._handle_initialize({"options": {"permission_mode": "bypassPermissions"}}, "init")
    await handler._handle_query({"prompt": "cost test"}, "q_cost")

    result_messages = [msg for msg in stream_messages if msg.get("type") == "result"]
    assert result_messages
    assert result_messages[-1].get("total_cost_usd") == pytest.approx(0.0215, rel=0, abs=1e-8)


@pytest.mark.asyncio
async def test_stdio_num_turns_excludes_progress_messages(monkeypatch, tmp_path):
    _patch_stdio_dependencies(monkeypatch, tools=[])

    async def fake_query(messages, system_prompt, context, query_context, can_use_tool):  # noqa: ARG001
        yield create_progress_message("tool-1", set(), "progress before assistant")
        yield create_assistant_message("assistant turn")
        yield create_progress_message("tool-1", set(), "progress after assistant")

    monkeypatch.setattr(handler_query, "query", fake_query)

    handler = handler_module.StdioProtocolHandler()
    handler._project_path = tmp_path

    stream_messages: list[dict[str, Any]] = []

    async def noop_write_control_response(*_args, **_kwargs):
        return None

    async def capture_stream(payload: dict[str, Any]):
        stream_messages.append(payload)

    monkeypatch.setattr(handler, "_write_control_response", noop_write_control_response)
    monkeypatch.setattr(handler, "_write_message_stream", capture_stream)

    await handler._handle_initialize({"options": {"permission_mode": "bypassPermissions"}}, "init")
    await handler._handle_query({"prompt": "turn count test"}, "q_turns")

    result_messages = [msg for msg in stream_messages if msg.get("type") == "result"]
    assert result_messages
    assert result_messages[-1].get("num_turns") == 1


@pytest.mark.asyncio
async def test_stdio_num_turns_excludes_tool_result_user_messages(monkeypatch, tmp_path):
    _patch_stdio_dependencies(monkeypatch, tools=[])

    async def fake_query(messages, system_prompt, context, query_context, can_use_tool):  # noqa: ARG001
        yield create_assistant_message("assistant turn")
        yield tool_result_message("tool-1", "tool finished")

    monkeypatch.setattr(handler_query, "query", fake_query)

    handler = handler_module.StdioProtocolHandler()
    handler._project_path = tmp_path

    stream_messages: list[dict[str, Any]] = []

    async def noop_write_control_response(*_args, **_kwargs):
        return None

    async def capture_stream(payload: dict[str, Any]):
        stream_messages.append(payload)

    monkeypatch.setattr(handler, "_write_control_response", noop_write_control_response)
    monkeypatch.setattr(handler, "_write_message_stream", capture_stream)

    await handler._handle_initialize({"options": {"permission_mode": "bypassPermissions"}}, "init")
    await handler._handle_query({"prompt": "turn count tool-result test"}, "q_turns_tool_result")

    result_messages = [msg for msg in stream_messages if msg.get("type") == "result"]
    assert result_messages
    assert result_messages[-1].get("num_turns") == 1
