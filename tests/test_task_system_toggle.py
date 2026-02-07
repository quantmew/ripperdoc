"""Tests for RIPPERDOC_ENABLE_TASKS tool-surface switching."""

from __future__ import annotations

from ripperdoc.core import tool_defaults


def test_default_tools_enable_task_graph_by_default(monkeypatch):
    monkeypatch.delenv("RIPPERDOC_ENABLE_TASKS", raising=False)
    monkeypatch.setattr(tool_defaults, "load_dynamic_mcp_tools_sync", lambda *_args, **_kwargs: [])

    tools = tool_defaults.get_default_tools()
    names = {tool.name for tool in tools}

    assert {"TaskCreate", "TaskGet", "TaskUpdate", "TaskList"}.issubset(names)
    assert {"TeamCreate", "TeamDelete", "SendMessage"}.issubset(names)
    assert "TeamGet" not in names
    assert "TeamList" not in names
    assert "TeamMemberUpsert" not in names
    assert "TodoRead" not in names
    assert "TodoWrite" not in names


def test_default_tools_fallback_to_todos_when_tasks_disabled(monkeypatch):
    monkeypatch.setenv("RIPPERDOC_ENABLE_TASKS", "false")
    monkeypatch.setattr(tool_defaults, "load_dynamic_mcp_tools_sync", lambda *_args, **_kwargs: [])

    tools = tool_defaults.get_default_tools()
    names = {tool.name for tool in tools}

    assert {"TodoRead", "TodoWrite"}.issubset(names)
    assert "TaskCreate" not in names
    assert "TaskList" not in names
    assert "TeamCreate" not in names
    assert "SendMessage" not in names
