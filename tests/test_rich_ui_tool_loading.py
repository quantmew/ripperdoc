from types import SimpleNamespace

import pytest

from ripperdoc.cli.ui.rich_ui.session import RichUI


@pytest.mark.asyncio
async def test_get_default_tools_async_loads_dynamic_tools(monkeypatch, tmp_path):
    ui = RichUI.__new__(RichUI)
    ui.query_context = None
    ui.allowed_tools = ["Base", "Dyn", "Task"]
    ui.disable_slash_commands = False
    ui.project_path = tmp_path

    merged_tools = [SimpleNamespace(name="Base"), SimpleNamespace(name="Dyn"), SimpleNamespace(name="Task")]

    async def _fake_get_default_tools_async(*, project_path, allowed_tools=None):
        assert project_path == tmp_path
        assert allowed_tools == ["Base", "Dyn", "Task"]
        return merged_tools

    monkeypatch.setattr("ripperdoc.cli.ui.rich_ui.session.get_default_tools_async", _fake_get_default_tools_async)

    assert await ui._get_default_tools_async() == merged_tools


@pytest.mark.asyncio
async def test_get_default_tools_async_prefers_query_context_tools(monkeypatch, tmp_path):
    ui = RichUI.__new__(RichUI)
    existing_tools = [SimpleNamespace(name="Existing"), SimpleNamespace(name="Task")]
    ui.query_context = SimpleNamespace(
        tool_registry=SimpleNamespace(all_tools=existing_tools),
    )
    ui.allowed_tools = None
    ui.disable_slash_commands = False
    ui.project_path = tmp_path

    async def _unexpected_get_default_tools_async(**_kwargs):
        raise AssertionError("tool factory should be skipped when query context exists")

    monkeypatch.setattr("ripperdoc.cli.ui.rich_ui.session.get_default_tools_async", _unexpected_get_default_tools_async)

    assert await ui._get_default_tools_async() == existing_tools


@pytest.mark.asyncio
async def test_get_default_tools_async_filters_skill_when_slash_commands_disabled(monkeypatch, tmp_path):
    ui = RichUI.__new__(RichUI)
    ui.query_context = None
    ui.allowed_tools = None
    ui.disable_slash_commands = True
    ui.project_path = tmp_path

    async def _fake_get_default_tools_async(**_kwargs):
        return [
            SimpleNamespace(name="Base"),
            SimpleNamespace(name="Skill"),
            SimpleNamespace(name="Task"),
        ]

    monkeypatch.setattr("ripperdoc.cli.ui.rich_ui.session.get_default_tools_async", _fake_get_default_tools_async)

    tools = await ui._get_default_tools_async()
    assert [tool.name for tool in tools] == ["Base", "Task"]
