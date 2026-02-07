"""Tests for system prompt composition."""

from __future__ import annotations

from pathlib import Path
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


def test_explanatory_style_removes_concise_constraints() -> None:
    prompt = build_system_prompt([DummyTool("Read"), DummyTool("Bash")], output_style="explanatory")

    assert "Output Style: Explanatory" in prompt
    assert "fewer than 4 lines" not in prompt
    assert "responses should be short and concise" not in prompt
    assert "Verify the solution if possible with tests." in prompt


def test_custom_style_excludes_coding_instructions_by_default(tmp_path: Path) -> None:
    styles_dir = tmp_path / ".ripperdoc" / "output-styles"
    styles_dir.mkdir(parents=True)
    (styles_dir / "product.md").write_text(
        """---
name: Product
description: product planner
---
Focus on product planning and tradeoffs.
""",
        encoding="utf-8",
    )

    prompt = build_system_prompt(
        [DummyTool("Read"), DummyTool("Bash")],
        output_style="product",
        project_path=tmp_path,
    )

    assert "Focus on product planning and tradeoffs." in prompt
    assert "Verify the solution if possible with tests." not in prompt
    assert "responses should be short and concise" not in prompt
    assert "# Committing changes with git" not in prompt


def test_custom_style_keep_coding_instructions_true(tmp_path: Path) -> None:
    styles_dir = tmp_path / ".ripperdoc" / "output-styles"
    styles_dir.mkdir(parents=True)
    (styles_dir / "mentor.md").write_text(
        """---
name: Mentor
description: coding mentor
keep-coding-instructions: true
---
Teach with coding examples.
""",
        encoding="utf-8",
    )

    prompt = build_system_prompt(
        [DummyTool("Read"), DummyTool("Bash")],
        output_style="mentor",
        project_path=tmp_path,
    )

    assert "Teach with coding examples." in prompt
    assert "Verify the solution if possible with tests." in prompt
    assert "# Committing changes with git" in prompt


def test_system_prompt_includes_output_language_instruction_when_configured() -> None:
    prompt = build_system_prompt(
        [DummyTool("Read"), DummyTool("Bash")],
        output_language="Chinese",
    )

    assert "# Output language" in prompt
    assert "respond in Chinese" in prompt


def test_system_prompt_omits_output_language_instruction_for_auto() -> None:
    prompt = build_system_prompt(
        [DummyTool("Read"), DummyTool("Bash")],
        output_language="auto",
    )

    assert "# Output language" not in prompt


def test_system_prompt_prefers_task_graph_when_available() -> None:
    prompt = build_system_prompt(
        [
            DummyTool("Read"),
            DummyTool("TaskCreate"),
            DummyTool("TaskGet"),
            DummyTool("TaskUpdate"),
            DummyTool("TaskList"),
        ]
    )

    assert "persistent task graph" in prompt
    assert "TaskCreate" in prompt
    assert "TaskUpdate" in prompt
    assert "`subject`, `description`, and optional `activeForm` / `metadata`" in prompt
    assert "optional `owner`, and optional `blocks` / `blockedBy`" not in prompt
