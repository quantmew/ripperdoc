from types import SimpleNamespace

import pytest

from ripperdoc.core import tool_defaults


@pytest.mark.asyncio
async def test_get_default_tools_async_merges_dynamic_tools(monkeypatch, tmp_path):
    monkeypatch.setattr(tool_defaults, "Tool", object)
    monkeypatch.setattr(
        tool_defaults,
        "_build_base_tools",
        lambda: [SimpleNamespace(name="Base")],
    )

    async def _fake_dynamic(project_path):
        assert project_path == tmp_path
        return [SimpleNamespace(name="Dyn")]

    monkeypatch.setattr(tool_defaults, "load_dynamic_mcp_tools_async", _fake_dynamic)

    def _fake_merge(base_tools, dynamic_tools):
        assert [tool.name for tool in base_tools] == ["Base"]
        assert [tool.name for tool in dynamic_tools] == ["Dyn"]
        return [
            SimpleNamespace(name="Base"),
            SimpleNamespace(name="Dyn"),
            SimpleNamespace(name="Task"),
        ]

    monkeypatch.setattr(tool_defaults, "merge_tools_with_dynamic", _fake_merge)

    result = await tool_defaults.get_default_tools_async(project_path=tmp_path)
    assert [tool.name for tool in result] == ["Base", "Dyn", "Task"]


@pytest.mark.asyncio
async def test_get_default_tools_async_ignores_allowed_tools_for_tool_set(monkeypatch, tmp_path):
    """allowed_tools controls permissions, not the tool set."""
    monkeypatch.setattr(tool_defaults, "Tool", object)
    monkeypatch.setattr(
        tool_defaults,
        "_build_base_tools",
        lambda: [SimpleNamespace(name="Base"), SimpleNamespace(name="Extra")],
    )

    async def _fake_dynamic(project_path):
        assert project_path == tmp_path
        return []

    monkeypatch.setattr(tool_defaults, "load_dynamic_mcp_tools_async", _fake_dynamic)

    result = await tool_defaults.get_default_tools_async(
        project_path=tmp_path,
        allowed_tools=["Base", "Task"],
    )
    # All tools are returned; allowed_tools does not filter the tool set.
    assert [tool.name for tool in result] == ["Base", "Extra", "Task"]
