"""Tests for the dedicated Memory tool."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from ripperdoc.core.query import ToolRegistry
from ripperdoc.core.tool import ToolResult, ToolUseContext
from ripperdoc.tools.memory_tool import MemoryTool, MemoryToolInput, MemoryToolOutput


async def _run_memory_tool(tool: MemoryTool, payload: dict) -> MemoryToolOutput:
    input_data = MemoryToolInput(**payload)
    result = None
    async for output in tool.call(input_data, ToolUseContext()):
        result = output
    assert isinstance(result, ToolResult)
    assert isinstance(result.data, MemoryToolOutput)
    return result.data


@pytest.mark.asyncio
async def test_memory_tool_create_and_view_file(tmp_path: Path) -> None:
    tool = MemoryTool(memory_dir=tmp_path / "memories")

    create_validation = await tool.validate_input(
        MemoryToolInput(command="create", path="facts.md", file_text="alpha\nbeta\n")
    )
    assert create_validation.result is True

    created = await _run_memory_tool(
        tool,
        {"command": "create", "path": "facts.md", "file_text": "alpha\nbeta\n"},
    )
    assert created.success is True

    viewed = await _run_memory_tool(tool, {"command": "view", "path": "facts.md"})
    assert viewed.success is True
    assert viewed.content == "alpha\nbeta\n"
    assert viewed.line_count == 2


@pytest.mark.asyncio
async def test_memory_tool_view_directory_lists_entries(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memories"
    (memory_dir / "topic").mkdir(parents=True)
    (memory_dir / "topic" / "debug.md").write_text("notes", encoding="utf-8")
    (memory_dir / "MEMORY.md").write_text("index", encoding="utf-8")
    tool = MemoryTool(memory_dir=memory_dir)

    viewed = await _run_memory_tool(tool, {"command": "view"})
    assert viewed.success is True
    assert "MEMORY.md" in viewed.entries
    assert "topic/" in viewed.entries
    assert "topic/debug.md" in viewed.entries


@pytest.mark.asyncio
async def test_memory_tool_str_replace_replace_all_semantics(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memories"
    memory_dir.mkdir(parents=True)
    (memory_dir / "facts.md").write_text("foo\nfoo\n", encoding="utf-8")
    tool = MemoryTool(memory_dir=memory_dir)

    not_unique = await _run_memory_tool(
        tool,
        {
            "command": "str_replace",
            "path": "facts.md",
            "old_str": "foo",
            "new_str": "bar",
        },
    )
    assert not_unique.success is False
    assert "replace_all=true" in not_unique.message

    replaced = await _run_memory_tool(
        tool,
        {
            "command": "str_replace",
            "path": "facts.md",
            "old_str": "foo",
            "new_str": "bar",
            "replace_all": True,
        },
    )
    assert replaced.success is True
    assert replaced.replacements_made == 2
    assert (memory_dir / "facts.md").read_text(encoding="utf-8") == "bar\nbar\n"


@pytest.mark.asyncio
async def test_memory_tool_insert_rename_delete(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memories"
    memory_dir.mkdir(parents=True)
    (memory_dir / "facts.md").write_text("line1\nline3\n", encoding="utf-8")
    tool = MemoryTool(memory_dir=memory_dir)

    inserted = await _run_memory_tool(
        tool,
        {"command": "insert", "path": "facts.md", "insert_line": 2, "insert_text": "line2"},
    )
    assert inserted.success is True
    assert (memory_dir / "facts.md").read_text(encoding="utf-8") == "line1\nline2\nline3\n"

    renamed = await _run_memory_tool(
        tool,
        {"command": "rename", "old_path": "facts.md", "new_path": "topic/facts-renamed.md"},
    )
    assert renamed.success is True
    assert (memory_dir / "topic" / "facts-renamed.md").exists()

    deleted = await _run_memory_tool(
        tool,
        {"command": "delete", "path": "topic/facts-renamed.md"},
    )
    assert deleted.success is True
    assert not (memory_dir / "topic" / "facts-renamed.md").exists()


@pytest.mark.asyncio
async def test_memory_tool_blocks_path_escape_and_view_is_read_only(tmp_path: Path) -> None:
    tool = MemoryTool(memory_dir=tmp_path / "memories")

    invalid = await tool.validate_input(MemoryToolInput(command="create", path="../escape.md", content="x"))
    assert invalid.result is False
    assert "outside memory directory" in (invalid.message or "")

    assert tool.needs_permissions(MemoryToolInput(command="view")) is False
    assert tool.needs_permissions(MemoryToolInput(command="delete", path="a.md")) is True


def test_tool_registry_resolves_memory_tool_case_insensitively(tmp_path: Path) -> None:
    tool = MemoryTool(memory_dir=tmp_path / "memories")
    registry = ToolRegistry([tool])
    assert registry.get("Memory") is tool
    assert registry.get("memory") is tool


@pytest.mark.asyncio
async def test_memory_tool_view_range_and_canonical_fields(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memories"
    memory_dir.mkdir(parents=True)
    (memory_dir / "facts.md").write_text("a\nb\nc\nd\n", encoding="utf-8")
    tool = MemoryTool(memory_dir=memory_dir)

    validation = await tool.validate_input(
        MemoryToolInput(command="view", path="facts.md", view_range=[2, 3])
    )
    assert validation.result is True

    viewed = await _run_memory_tool(
        tool,
        {"command": "view", "path": "facts.md", "view_range": [2, 3]},
    )
    assert viewed.success is True
    assert viewed.content == "b\nc\n"


def test_memory_tool_rejects_operation_without_command() -> None:
    with pytest.raises(ValidationError):
        MemoryToolInput(
            operation="edit",
            path="MEMORY.md",
            old_string="name: old",
            new_string="name: Lethon",
        )
