"""Tests for system prompt composition."""

from __future__ import annotations

from typing import AsyncGenerator

from pydantic import BaseModel

from ripperdoc.core.system_prompt import build_system_prompt
from ripperdoc.core.tool import Tool, ToolOutput, ToolUseContext


class DummyInput(BaseModel):
    value: str = "ok"


class DummyTool(Tool[DummyInput, str]):
    def __init__(self, name: str) -> None:
        super().__init__()
        self._tool_name = name

    @property
    def name(self) -> str:
        return self._tool_name

    async def description(self) -> str:
        return f"{self._tool_name} tool"

    @property
    def input_schema(self) -> type[DummyInput]:
        return DummyInput

    async def prompt(self, yolo_mode: bool = False) -> str:  # noqa: ARG002
        return ""

    def render_result_for_assistant(self, output: str) -> str:
        return output

    def render_tool_use_message(self, input_data: DummyInput, verbose: bool = False) -> str:  # noqa: ARG002
        return f"{self._tool_name}({input_data.value})"

    async def call(
        self, input_data: DummyInput, context: ToolUseContext
    ) -> AsyncGenerator[ToolOutput, None]:  # noqa: ARG002
        if False:
            yield


def test_system_prompt_without_shell_omits_shell_instructions() -> None:
    prompt = build_system_prompt([DummyTool("Read"), DummyTool("Edit")])

    assert "Bash" not in prompt
    assert "multiple bash tool calls" not in prompt
    assert "# Committing changes with git" not in prompt
    assert "No shell command tool is available in this session." in prompt
    assert "A shell command tool is not available in this session." in prompt


def test_system_prompt_with_shell_includes_shell_instructions() -> None:
    prompt = build_system_prompt([DummyTool("Read"), DummyTool("Bash")])

    assert "# Explain Your Code: Shell Command Transparency" in prompt
    assert "# Committing changes with git" in prompt
    assert "via the Bash tool" in prompt
    assert "No shell command tool is available in this session." not in prompt
