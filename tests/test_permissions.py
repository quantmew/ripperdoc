"""Tests for the permission system."""

import asyncio
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from ripperdoc.core.permissions import (
    PermissionResult,
    make_permission_checker,
)
from ripperdoc.core.config import config_manager
from ripperdoc.core.tool import Tool, ToolResult, ToolUseContext


class DummyInput(BaseModel):
    command: str


class DummyTool(Tool[DummyInput, None]):
    @property
    def name(self) -> str:
        return "DummyTool"

    async def description(self) -> str:
        return "Dummy tool for testing permissions."

    @property
    def input_schema(self) -> type[DummyInput]:
        return DummyInput

    async def prompt(self, yolo_mode: bool = False) -> str:  # noqa: ARG002
        return "dummy prompt"

    def render_result_for_assistant(self, output: None) -> str:
        return "done"

    def render_tool_use_message(self, input_data: DummyInput, verbose: bool = False) -> str:
        return f"run {input_data.command}"

    async def call(self, input_data: DummyInput, context: ToolUseContext):
        yield ToolResult(data=None)


class DictPermissionTool(DummyTool):
    """Dummy tool whose check_permissions returns a dict (legacy behavior)."""

    async def check_permissions(self, input_data: DummyInput, permission_context: Any):
        return {"behavior": "allow", "updated_input": input_data}


def test_yolo_mode_off_does_not_persist_permissions(tmp_path: Path):
    """Approvals should not be written to project config."""
    tool = DummyTool()
    parsed_input = DummyInput(command="echo hello")

    prompt_calls = 0

    def prompt_fn(_: str) -> str:
        nonlocal prompt_calls
        prompt_calls += 1
        return "a"  # approve for the session only

    checker = make_permission_checker(tmp_path, yolo_mode=False, prompt_fn=prompt_fn)
    result = asyncio.run(checker(tool, parsed_input))
    assert isinstance(result, PermissionResult)
    assert result.result is True

    config = config_manager.get_project_config(tmp_path)
    assert tool.name not in config.allowed_tools

    # New checker should prompt again because approvals are session-only
    second_prompt_calls = 0

    def prompt_fn_again(_: str) -> str:
        nonlocal second_prompt_calls
        second_prompt_calls += 1
        return "y"

    checker_again = make_permission_checker(tmp_path, yolo_mode=False, prompt_fn=prompt_fn_again)
    second = asyncio.run(checker_again(tool, DummyInput(command="echo goodbye")))
    assert second.result is True
    assert prompt_calls == 1
    assert second_prompt_calls == 1


def test_session_always_allows_similar_commands(tmp_path: Path):
    """Session-level approvals should skip prompts for the same tool."""
    tool = DummyTool()
    prompts = iter(["a"])
    prompt_calls = 0

    def prompt_fn(_: str) -> str:
        nonlocal prompt_calls
        prompt_calls += 1
        return next(prompts)

    checker = make_permission_checker(tmp_path, yolo_mode=False, prompt_fn=prompt_fn)

    first = asyncio.run(checker(tool, DummyInput(command="echo first")))
    second = asyncio.run(checker(tool, DummyInput(command="echo second")))

    assert first.result is True
    assert second.result is True
    assert prompt_calls == 1

    # Session approvals should not be persisted to disk
    config = config_manager.get_project_config(tmp_path)
    assert tool.name not in config.allowed_tools


def test_yolo_mode_off_respects_read_only_tools(tmp_path: Path):
    """Read-only tools should bypass permission prompts even when prompts are on."""
    from ripperdoc.tools.file_read_tool import FileReadTool, FileReadToolInput

    temp_file = tmp_path / "file.txt"
    temp_file.write_text("hello")

    checker = make_permission_checker(
        tmp_path, yolo_mode=False, prompt_fn=lambda _: pytest.fail("prompted unexpectedly")
    )
    tool = FileReadTool()
    parsed_input = FileReadToolInput(file_path=str(temp_file))

    result = asyncio.run(checker(tool, parsed_input))
    assert result.result is True


def test_yolo_mode_allows_without_prompt(tmp_path: Path):
    """When yolo mode is enabled, tools should run without permission checks."""
    tool = DummyTool()
    parsed_input = DummyInput(command="rm -rf /tmp/test")

    checker = make_permission_checker(
        tmp_path, yolo_mode=True, prompt_fn=lambda _: pytest.fail("prompted unexpectedly")
    )
    result = asyncio.run(checker(tool, parsed_input))
    assert result.result is True


def test_dict_permission_result_is_handled(tmp_path: Path):
    """Tools returning dict-based permission decisions should still be accepted."""
    tool = DictPermissionTool()
    parsed_input = DummyInput(command="echo ok")

    checker = make_permission_checker(tmp_path, yolo_mode=False, prompt_fn=lambda _: "y")
    result = asyncio.run(checker(tool, parsed_input))
    assert result.result is True
