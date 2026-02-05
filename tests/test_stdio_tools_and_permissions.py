"""Tests for stdio tool filtering and can_use_tool permission handling."""

from __future__ import annotations

from typing import Any, List

import pytest
from pydantic import BaseModel

from ripperdoc.core.permissions import PermissionResult
from ripperdoc.core.tool import Tool, ToolResult, ToolUseContext
from ripperdoc.protocol.stdio import handler as handler_module
from ripperdoc.protocol.stdio import handler_config, handler_session
from ripperdoc.tools.task_tool import TaskTool


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

    def render_tool_use_message(self, input_data: DummyInput, verbose: bool = False) -> str:
        return f"{self._tool_name}({input_data.value})"

    async def call(self, input_data: DummyInput, context: ToolUseContext):
        yield ToolResult(data="ok")


def _patch_stdio_dependencies(monkeypatch, tools: List[Any]) -> None:
    monkeypatch.setattr(handler_session, "get_project_config", lambda _path: None)
    monkeypatch.setattr(handler_session, "get_effective_model_profile", lambda _model: object())
    monkeypatch.setattr(handler_session, "get_default_tools", lambda **_: tools)
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

    from ripperdoc.core import skills as skills_module

    class DummySkillResult:
        skills: list = []

    monkeypatch.setattr(skills_module, "load_all_skills", lambda _path: DummySkillResult())
    monkeypatch.setattr(skills_module, "build_skill_summary", lambda _skills: "")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options,expected",
    [
        (
            {
                "tools": ["Read", "Bash", "Task"],
                "allowed_tools": ["Read", "Edit", "Task"],
                "disallowed_tools": ["Bash"],
            },
            ["Read", "Task"],
        ),
        (
            {
                "allowed_tools": ["Read", "Bash", "Task"],
                "disallowed_tools": ["Bash"],
            },
            ["Read", "Task"],
        ),
        (
            {
                "disallowed_tools": ["Edit"],
            },
            ["Read", "Bash", "Task"],
        ),
    ],
)
async def test_stdio_initialize_tool_filters(monkeypatch, tmp_path, options, expected):
    base_tools = [DummyTool("Read"), DummyTool("Edit"), DummyTool("Bash")]
    task_tool = TaskTool(lambda: base_tools)
    tools = base_tools + [task_tool]

    _patch_stdio_dependencies(monkeypatch, tools)

    handler = handler_module.StdioProtocolHandler()
    handler._project_path = tmp_path

    responses: dict[str, dict[str, Any]] = {}

    async def capture(request_id: str, response: dict[str, Any] | None = None, error: str | None = None):
        responses[request_id] = {"response": response, "error": error}

    monkeypatch.setattr(handler, "_write_control_response", capture)

    await handler._handle_initialize({"options": options}, "init")

    assert responses["init"]["error"] is None
    assert responses["init"]["response"]["tools"] == expected


@pytest.mark.asyncio
async def test_stdio_can_use_tool_permission_result(monkeypatch, tmp_path):
    tools = [DummyTool("Read")]
    _patch_stdio_dependencies(monkeypatch, tools)

    async def allow_checker(tool, parsed_input):
        assert tool.name == "Read"
        assert parsed_input.value == "hello"
        return PermissionResult(result=True, updated_input={"value": "updated"})

    monkeypatch.setattr(handler_config, "make_permission_checker", lambda *_args, **_kwargs: allow_checker)

    handler = handler_module.StdioProtocolHandler()
    handler._project_path = tmp_path

    responses: dict[str, dict[str, Any]] = {}

    async def capture(request_id: str, response: dict[str, Any] | None = None, error: str | None = None):
        responses[request_id] = {"response": response, "error": error}

    monkeypatch.setattr(handler, "_write_control_response", capture)

    await handler._handle_initialize({"options": {}}, "init")

    await handler._handle_can_use_tool(
        {"tool_name": "Read", "input": {"value": "hello"}},
        "req_allow",
    )
    allow_response = responses["req_allow"]["response"]
    assert allow_response["decision"] == "allow"
    assert allow_response["updatedInput"] == {"value": "updated"}

    async def deny_checker(tool, parsed_input):
        assert tool.name == "Read"
        assert parsed_input.value == "hello"
        return PermissionResult(result=False, message="nope")

    handler._can_use_tool = deny_checker

    await handler._handle_can_use_tool(
        {"tool_name": "Read", "input": {"value": "hello"}},
        "req_deny",
    )
    deny_response = responses["req_deny"]["response"]
    assert deny_response["decision"] == "deny"
    assert deny_response["message"] == "nope"
