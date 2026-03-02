"""Tests for tool-level input alias application."""

from __future__ import annotations

from typing import AsyncGenerator

from pydantic import BaseModel

from ripperdoc.core.query.loop import _apply_tool_input_aliases
from ripperdoc.core.tool import Tool, ToolOutput, ToolUseContext


class _AliasInput(BaseModel):
    command: str


class _AliasTool(Tool[_AliasInput, str]):
    @property
    def name(self) -> str:
        return "AliasTool"

    async def description(self) -> str:
        return "alias tool"

    @property
    def input_schema(self) -> type[_AliasInput]:
        return _AliasInput

    async def prompt(self, yolo_mode: bool = False) -> str:  # noqa: ARG002
        return ""

    def input_param_aliases(self) -> dict[str, str]:
        return {"operation": "command"}

    def render_result_for_assistant(self, output: str) -> str:
        return output

    def render_tool_use_message(self, input_data: _AliasInput, verbose: bool = False) -> str:  # noqa: ARG002
        return input_data.command

    async def call(
        self, input_data: _AliasInput, context: ToolUseContext  # noqa: ARG002
    ) -> AsyncGenerator[ToolOutput, None]:
        del input_data
        if False:
            yield


def test_apply_tool_input_aliases_maps_when_canonical_missing() -> None:
    tool = _AliasTool()
    normalized = _apply_tool_input_aliases(tool, {"operation": "view"})
    assert normalized["command"] == "view"
    assert normalized["operation"] == "view"


def test_apply_tool_input_aliases_does_not_override_canonical() -> None:
    tool = _AliasTool()
    normalized = _apply_tool_input_aliases(
        tool,
        {"operation": "view", "command": "create"},
    )
    assert normalized["command"] == "create"
