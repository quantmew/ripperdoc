"""Tests for the ToolSearch tool and deferred activation."""

import asyncio
from pydantic import BaseModel

from ripperdoc.core.query import ToolRegistry
from ripperdoc.core.tool import Tool, ToolResult, ToolUseContext
from ripperdoc.tools.tool_search_tool import ToolSearchInput, ToolSearchTool


class DummyInput(BaseModel):
    """Minimal input model for dummy tools."""

    pass


class DummyTool(Tool[DummyInput, str]):
    """Lightweight tool implementation for search tests."""

    def __init__(self, name: str, description: str = "", deferred: bool = False) -> None:
        super().__init__()
        self._name = name
        self._description = description or name
        self._deferred = deferred

    @property
    def name(self) -> str:
        return self._name

    async def description(self) -> str:
        return self._description

    @property
    def input_schema(self) -> type[DummyInput]:
        return DummyInput

    async def prompt(self, yolo_mode: bool = False) -> str:  # noqa: ARG002
        return ""

    def defer_loading(self) -> bool:
        return self._deferred

    def render_result_for_assistant(self, output: str) -> str:
        return output

    def render_tool_use_message(self, input_data: DummyInput, verbose: bool = False) -> str:  # noqa: ARG002
        return self._name

    async def call(  # type: ignore[override]
        self,
        input_data: DummyInput,
        context: ToolUseContext,  # noqa: ARG002
    ):
        yield ToolResult(data="ok", result_for_assistant="ok")


def _collect_async(gen):
    async def _runner():
        items = []
        async for item in gen:
            items.append(item)
        return items

    return asyncio.run(_runner())


def test_tool_search_regex_matches_deferred():
    """Regex queries should return matching deferred tools."""
    deferred_tool = DummyTool("mcp__alpha__doThing", "alpha tool", deferred=True)
    active_tool = DummyTool("BetaTool", "beta")
    registry = ToolRegistry([ToolSearchTool(), active_tool, deferred_tool])

    search_tool = ToolSearchTool()
    ctx = ToolUseContext(tool_registry=registry)
    inp = ToolSearchInput(query="/alpha/", max_results=5, include_active=True)

    outputs = _collect_async(search_tool.call(inp, ctx))
    result = outputs[-1] if outputs else None

    assert result is not None
    match_names = [m.name for m in result.data.matches]
    assert "mcp__alpha__doThing" in match_names


def test_tool_search_auto_activates_deferred():
    """ToolSearch should activate deferred tools it finds."""
    deferred_tool = DummyTool("DeferredTool", "defer me", deferred=True)
    registry = ToolRegistry([ToolSearchTool(), deferred_tool])
    assert not registry.is_active("DeferredTool")

    search_tool = ToolSearchTool()
    ctx = ToolUseContext(tool_registry=registry)
    inp = ToolSearchInput(query="deferred", max_results=3, include_active=True)

    outputs = _collect_async(search_tool.call(inp, ctx))
    result = outputs[-1] if outputs else None

    assert result is not None
    assert registry.is_active("DeferredTool")
    assert any(m.name == "DeferredTool" for m in result.data.matches)
