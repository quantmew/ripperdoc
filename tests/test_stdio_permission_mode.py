"""Regression tests for stdio permission mode switching."""

from __future__ import annotations

import pytest

from ripperdoc.core.hooks.manager import hook_manager
from ripperdoc.protocol.stdio import handler as handler_module
from ripperdoc.protocol.stdio import handler_config, handler_session


@pytest.mark.asyncio
async def test_stdio_permission_mode_switch(monkeypatch, tmp_path):
    handler = handler_module.StdioProtocolHandler()
    handler._project_path = tmp_path

    # Avoid touching real config or external dependencies during init.
    monkeypatch.setattr(handler_session, "get_project_config", lambda _path: None)
    monkeypatch.setattr(handler_session, "get_effective_model_profile", lambda _model: object())
    monkeypatch.setattr(handler_session, "get_default_tools", lambda **_: [])
    monkeypatch.setattr(handler_session, "format_mcp_instructions", lambda _servers: "")
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

    # Patch skill helpers to avoid filesystem access.
    from ripperdoc.core import skills as skills_module

    class DummySkillResult:
        skills = []

    monkeypatch.setattr(skills_module, "load_all_skills", lambda _path: DummySkillResult())
    monkeypatch.setattr(skills_module, "build_skill_summary", lambda _skills: "")

    # Capture permission checker instantiation.
    checker_calls = []

    def fake_make_permission_checker(project_path, yolo_mode=False):
        checker = object()
        checker_calls.append((project_path, yolo_mode, checker))
        return checker

    monkeypatch.setattr(handler_config, "make_permission_checker", fake_make_permission_checker)

    # Silence stdout writes from control responses.
    async def noop_write_control_response(*_args, **_kwargs):
        return None

    monkeypatch.setattr(handler, "_write_control_response", noop_write_control_response)

    previous_mode = hook_manager.permission_mode
    try:
        await handler._handle_initialize(
            {"options": {"permission_mode": "plan", "sdk_can_use_tool": False}},
            "init",
        )

        assert handler._query_context is not None
        assert handler._query_context.permission_mode == "plan"
        assert handler._query_context.yolo_mode is False
        assert hook_manager.permission_mode == "plan"
        assert handler._can_use_tool is checker_calls[-1][2]
        assert len(checker_calls) == 1

        await handler._handle_set_permission_mode({"mode": "bypassPermissions"}, "set1")
        assert handler._query_context.permission_mode == "bypassPermissions"
        assert handler._query_context.yolo_mode is True
        assert hook_manager.permission_mode == "bypassPermissions"
        assert handler._can_use_tool is None
        assert len(checker_calls) == 1

        await handler._handle_set_permission_mode({"mode": "default"}, "set2")
        assert handler._query_context.permission_mode == "default"
        assert handler._query_context.yolo_mode is False
        assert hook_manager.permission_mode == "default"
        assert handler._can_use_tool is checker_calls[-1][2]
        assert len(checker_calls) == 2
    finally:
        hook_manager.set_permission_mode(previous_mode)
