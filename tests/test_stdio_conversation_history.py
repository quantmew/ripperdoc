"""Regression tests for stdio conversation history updates."""

from __future__ import annotations

from typing import Any, List

import pytest

from ripperdoc.protocol.stdio import handler as handler_module
from ripperdoc.utils.messages import create_assistant_message
from ripperdoc.core.query_utils import tool_result_message


def _patch_stdio_dependencies(monkeypatch, tools: List[Any]) -> None:
    monkeypatch.setattr(handler_module, "get_project_config", lambda _path: None)
    monkeypatch.setattr(handler_module, "get_effective_model_profile", lambda _model: object())
    monkeypatch.setattr(handler_module, "get_default_tools", lambda **_: tools)
    monkeypatch.setattr(handler_module, "format_mcp_instructions", lambda _servers: "")
    monkeypatch.setattr(handler_module, "build_system_prompt", lambda *_args, **_kwargs: "system")
    monkeypatch.setattr(handler_module, "build_memory_instructions", lambda: "")
    monkeypatch.setattr(handler_module, "list_custom_commands", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(handler_module, "list_slash_commands", lambda: [])

    async def fake_load_mcp_servers_async(_path):
        return []

    async def fake_load_dynamic_mcp_tools_async(_path):
        return []

    monkeypatch.setattr(handler_module, "load_mcp_servers_async", fake_load_mcp_servers_async)
    monkeypatch.setattr(handler_module, "load_dynamic_mcp_tools_async", fake_load_dynamic_mcp_tools_async)

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

    monkeypatch.setattr(handler_module.hook_manager, "run_session_start_async", fake_session_start_async)
    monkeypatch.setattr(handler_module.hook_manager, "run_user_prompt_submit_async", fake_prompt_submit_async)


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

    monkeypatch.setattr(handler_module, "query", fake_query)

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
