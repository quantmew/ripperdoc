"""Tests for stdio tool filtering and can_use_tool permission handling."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, List

import pytest
from pydantic import BaseModel

from ripperdoc.core.permission_engine import PermissionPreview, PermissionResult
from ripperdoc.core.tool import Tool, ToolResult, ToolUseContext
from ripperdoc.protocol.stdio import handler as handler_module
from ripperdoc.protocol.stdio import handler_control
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


class DummyReadOnlyTool(DummyTool):
    def is_read_only(self) -> bool:
        return True


class DummyAskTool(DummyTool):
    @property
    def name(self) -> str:
        return "AskUserQuestion"

    def needs_permissions(self, input_data: DummyInput | None = None) -> bool:  # noqa: ARG002
        return False


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
async def test_stdio_initialize_applies_fallback_model_and_thinking_tokens(monkeypatch, tmp_path):
    tools = [DummyTool("Read")]
    _patch_stdio_dependencies(monkeypatch, tools)

    def fake_profile_lookup(model_name: str):
        if model_name == "backup-model":
            return object()
        return None

    monkeypatch.setattr(handler_session, "get_effective_model_profile", fake_profile_lookup)

    handler = handler_module.StdioProtocolHandler()
    handler._project_path = tmp_path

    responses: dict[str, dict[str, Any]] = {}

    async def capture(request_id: str, response: dict[str, Any] | None = None, error: str | None = None):
        responses[request_id] = {"response": response, "error": error}

    monkeypatch.setattr(handler, "_write_control_response", capture)

    await handler._handle_initialize(
        {
            "options": {
                "model": "missing-model",
                "fallback_model": "backup-model",
                "max_thinking_tokens": "256",
            }
        },
        "init",
    )

    assert responses["init"]["error"] is None
    assert handler._query_context is not None
    assert handler._query_context.model == "backup-model"
    assert handler._query_context.max_thinking_tokens == 256


@pytest.mark.asyncio
async def test_stdio_initialize_applies_session_agent_prompt(monkeypatch, tmp_path):
    tools = [DummyTool("Read")]
    _patch_stdio_dependencies(monkeypatch, tools)

    captured: dict[str, Any] = {}

    def fake_build_system_prompt(*_args, **kwargs):
        captured["additional_instructions"] = kwargs.get("additional_instructions")
        return "system"

    monkeypatch.setattr(handler_config, "build_system_prompt", fake_build_system_prompt)

    handler = handler_module.StdioProtocolHandler()
    handler._project_path = tmp_path

    responses: dict[str, dict[str, Any]] = {}

    async def capture(request_id: str, response: dict[str, Any] | None = None, error: str | None = None):
        responses[request_id] = {"response": response, "error": error}

    monkeypatch.setattr(handler, "_write_control_response", capture)

    await handler._handle_initialize(
        {
            "options": {
                "agent": "reviewer",
                "agents": {"reviewer": {"prompt": "Focus on correctness and risk."}},
            }
        },
        "init",
    )

    assert responses["init"]["error"] is None
    assert handler._session_agent_name == "reviewer"
    assert handler._session_agent_prompt == "Focus on correctness and risk."
    assert "Focus on correctness and risk." in (captured.get("additional_instructions") or [])


@pytest.mark.asyncio
async def test_stdio_initialize_rejects_unknown_session_agent(monkeypatch, tmp_path):
    tools = [DummyTool("Read")]
    _patch_stdio_dependencies(monkeypatch, tools)

    handler = handler_module.StdioProtocolHandler()
    handler._project_path = tmp_path

    responses: dict[str, dict[str, Any]] = {}

    async def capture(request_id: str, response: dict[str, Any] | None = None, error: str | None = None):
        responses[request_id] = {"response": response, "error": error}

    monkeypatch.setattr(handler, "_write_control_response", capture)

    await handler._handle_initialize(
        {
            "options": {
                "agent": "unknown",
                "agents": {"reviewer": {"prompt": "Focus on correctness and risk."}},
            }
        },
        "init",
    )

    assert responses["init"]["error"] == "Unknown agent 'unknown'. Available agents: reviewer."


@pytest.mark.asyncio
async def test_stdio_initialize_disable_slash_commands_removes_skill_tool(monkeypatch, tmp_path):
    tools = [DummyTool("Read"), DummyTool("Skill")]
    _patch_stdio_dependencies(monkeypatch, tools)
    monkeypatch.setattr(
        handler_session,
        "list_slash_commands",
        lambda: [SimpleNamespace(name="help"), SimpleNamespace(name="skills")],
    )
    monkeypatch.setattr(handler_session, "list_custom_commands", lambda *_args, **_kwargs: [])

    handler = handler_module.StdioProtocolHandler()
    handler._project_path = tmp_path

    responses: dict[str, dict[str, Any]] = {}

    async def capture(request_id: str, response: dict[str, Any] | None = None, error: str | None = None):
        responses[request_id] = {"response": response, "error": error}

    monkeypatch.setattr(handler, "_write_control_response", capture)

    await handler._handle_initialize(
        {"options": {"disable_slash_commands": True}},
        "init",
    )

    assert responses["init"]["error"] is None
    response = responses["init"]["response"]
    assert "Skill" not in response["tools"]
    assert response["slash_commands"] == []


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

    handler._local_can_use_tool = deny_checker

    await handler._handle_can_use_tool(
        {"tool_name": "Read", "input": {"value": "hello"}},
        "req_deny",
    )
    deny_response = responses["req_deny"]["response"]
    assert deny_response["decision"] == "deny"
    assert deny_response["message"] == "nope"


@pytest.mark.asyncio
async def test_stdio_sdk_can_use_tool_bridges_ask_user_question(monkeypatch, tmp_path):
    tools = [DummyAskTool("ignored")]
    _patch_stdio_dependencies(monkeypatch, tools)

    # Local checker should not be used when SDK bridge is active and responds.
    async def local_checker(_tool, _parsed_input):
        return PermissionResult(result=False, message="local checker should not run")

    monkeypatch.setattr(handler_config, "make_permission_checker", lambda *_args, **_kwargs: local_checker)

    handler = handler_module.StdioProtocolHandler()
    handler._project_path = tmp_path

    async def capture_control_response(*_args, **_kwargs):
        return None

    requests: list[dict[str, Any]] = []

    async def fake_send_control_request(request: dict[str, Any], *, timeout: float | None = None):
        requests.append({"request": request, "timeout": timeout})
        assert request["subtype"] == "can_use_tool"
        assert request["tool_name"] == "AskUserQuestion"
        return {
            "decision": "allow",
            "updatedInput": {
                "value": "sdk-answer",
            },
        }

    monkeypatch.setattr(handler, "_write_control_response", capture_control_response)
    monkeypatch.setattr(handler, "_send_control_request", fake_send_control_request)

    await handler._handle_initialize({"options": {}}, "init")

    tool = handler._query_context.tool_registry.get("AskUserQuestion")
    assert tool is not None
    result = await handler._can_use_tool(tool, tool.input_schema(value="question"))  # type: ignore[misc]

    assert isinstance(result, PermissionResult)
    assert result.result is True
    assert result.updated_input == {"value": "sdk-answer"}
    assert len(requests) == 1
    assert requests[0]["timeout"] == 0.0


@pytest.mark.asyncio
async def test_stdio_sdk_can_use_tool_skips_read_only_tools(monkeypatch, tmp_path):
    tools = [DummyReadOnlyTool("Read")]
    _patch_stdio_dependencies(monkeypatch, tools)

    monkeypatch.setattr(
        handler_config,
        "make_permission_checker",
        lambda *_args, **_kwargs: (lambda *_a, **_k: PermissionResult(result=True)),
    )

    handler = handler_module.StdioProtocolHandler()
    handler._project_path = tmp_path

    async def capture_control_response(*_args, **_kwargs):
        return None

    called = {"sdk": 0}

    async def fake_send_control_request(*_args, **_kwargs):
        called["sdk"] += 1
        raise AssertionError("SDK can_use_tool should not run for read-only tool")

    monkeypatch.setattr(handler, "_write_control_response", capture_control_response)
    monkeypatch.setattr(handler, "_send_control_request", fake_send_control_request)

    await handler._handle_initialize({"options": {}}, "init")

    tool = handler._query_context.tool_registry.get("Read")
    assert tool is not None
    result = await handler._can_use_tool(tool, tool.input_schema(value="ok"))  # type: ignore[misc]

    assert isinstance(result, PermissionResult)
    assert result.result is True
    assert called["sdk"] == 0


@pytest.mark.asyncio
async def test_stdio_sdk_can_use_tool_uses_preview_for_rule_level_ask(monkeypatch, tmp_path):
    tools = [DummyReadOnlyTool("Read")]
    _patch_stdio_dependencies(monkeypatch, tools)

    async def local_checker(_tool, _parsed_input):
        return PermissionResult(result=True)

    async def local_preview(_tool, _parsed_input):
        return PermissionPreview(requires_user_input=True)

    async def local_preview_force(_tool, _parsed_input):
        return PermissionPreview(requires_user_input=True)

    setattr(local_checker, "preview", local_preview)
    setattr(local_checker, "preview_force_prompt", local_preview_force)

    monkeypatch.setattr(handler_config, "make_permission_checker", lambda *_args, **_kwargs: local_checker)

    handler = handler_module.StdioProtocolHandler()
    handler._project_path = tmp_path

    async def capture_control_response(*_args, **_kwargs):
        return None

    calls = {"sdk": 0}

    async def fake_send_control_request(request: dict[str, Any], *, timeout: float | None = None):
        calls["sdk"] += 1
        assert request["subtype"] == "can_use_tool"
        assert request["tool_name"] == "Read"
        return {"decision": "allow", "updatedInput": {"value": "approved"}}

    monkeypatch.setattr(handler, "_write_control_response", capture_control_response)
    monkeypatch.setattr(handler, "_send_control_request", fake_send_control_request)

    await handler._handle_initialize({"options": {}}, "init")

    tool = handler._query_context.tool_registry.get("Read")
    assert tool is not None
    result = await handler._can_use_tool(tool, tool.input_schema(value="ok"))  # type: ignore[misc]

    assert isinstance(result, PermissionResult)
    assert result.result is True
    assert result.updated_input == {"value": "approved"}
    assert calls["sdk"] == 1


@pytest.mark.asyncio
async def test_stdio_sdk_can_use_tool_falls_back_to_local_checker(monkeypatch, tmp_path):
    tools = [DummyTool("Write")]
    _patch_stdio_dependencies(monkeypatch, tools)

    local_calls = {"count": 0}

    async def local_checker(_tool, parsed_input):
        local_calls["count"] += 1
        return PermissionResult(result=True, updated_input={"value": parsed_input.value + "-local"})

    monkeypatch.setattr(handler_config, "make_permission_checker", lambda *_args, **_kwargs: local_checker)

    handler = handler_module.StdioProtocolHandler()
    handler._project_path = tmp_path

    async def capture_control_response(*_args, **_kwargs):
        return None

    sdk_calls = {"count": 0}

    async def fake_send_control_request(*_args, **_kwargs):
        sdk_calls["count"] += 1
        raise RuntimeError("Unknown request subtype: can_use_tool")

    monkeypatch.setattr(handler, "_write_control_response", capture_control_response)
    monkeypatch.setattr(handler, "_send_control_request", fake_send_control_request)

    await handler._handle_initialize({"options": {}}, "init")

    tool = handler._query_context.tool_registry.get("Write")
    assert tool is not None

    first = await handler._can_use_tool(tool, tool.input_schema(value="a"))  # type: ignore[misc]
    second = await handler._can_use_tool(tool, tool.input_schema(value="b"))  # type: ignore[misc]

    assert isinstance(first, PermissionResult)
    assert isinstance(second, PermissionResult)
    assert first.result is True
    assert second.result is True
    assert first.updated_input == {"value": "a-local"}
    assert second.updated_input == {"value": "b-local"}
    # First call probes SDK and then degrades; second call should remain local-only.
    assert sdk_calls["count"] == 1
    assert local_calls["count"] == 2


@pytest.mark.asyncio
async def test_stdio_initialize_passes_additional_dirs_to_permission_checker(monkeypatch, tmp_path):
    tools = [DummyTool("Read")]
    _patch_stdio_dependencies(monkeypatch, tools)

    outside = tmp_path / "outside"
    outside.mkdir()
    captured: dict[str, Any] = {}

    async def allow_checker(_tool, _parsed_input):
        return PermissionResult(result=True)

    def fake_make_permission_checker(project_path, yolo_mode=False, **kwargs):  # noqa: ARG001
        captured["dirs"] = kwargs.get("session_additional_working_dirs")
        return allow_checker

    monkeypatch.setattr(handler_config, "make_permission_checker", fake_make_permission_checker)

    handler = handler_module.StdioProtocolHandler()
    handler._project_path = tmp_path

    async def capture_control_response(*_args, **_kwargs):
        return None

    monkeypatch.setattr(handler, "_write_control_response", capture_control_response)

    await handler._handle_initialize(
        {"options": {"additional_directories": [str(outside)]}},
        "init",
    )

    assert "dirs" in captured
    assert str(outside.resolve()) in set(captured["dirs"] or [])


@pytest.mark.asyncio
async def test_stdio_initialize_ignores_legacy_add_dirs_alias(monkeypatch, tmp_path):
    tools = [DummyTool("Read")]
    _patch_stdio_dependencies(monkeypatch, tools)

    outside = tmp_path / "outside"
    outside.mkdir()
    captured: dict[str, Any] = {}

    async def allow_checker(_tool, _parsed_input):
        return PermissionResult(result=True)

    def fake_make_permission_checker(project_path, yolo_mode=False, **kwargs):  # noqa: ARG001
        captured["dirs"] = kwargs.get("session_additional_working_dirs")
        return allow_checker

    monkeypatch.setattr(handler_config, "make_permission_checker", fake_make_permission_checker)

    handler = handler_module.StdioProtocolHandler()
    handler._project_path = tmp_path

    async def capture_control_response(*_args, **_kwargs):
        return None

    monkeypatch.setattr(handler, "_write_control_response", capture_control_response)

    await handler._handle_initialize(
        {"options": {"add_dirs": [str(outside)]}},
        "init",
    )

    assert "dirs" in captured
    assert str(outside.resolve()) not in set(captured["dirs"] or [])


@pytest.mark.asyncio
async def test_stdio_can_use_tool_requires_snake_case_tool_name(monkeypatch, tmp_path):
    tools = [DummyTool("Read")]
    _patch_stdio_dependencies(monkeypatch, tools)

    async def allow_checker(_tool, _parsed_input):
        return PermissionResult(result=True)

    monkeypatch.setattr(handler_config, "make_permission_checker", lambda *_args, **_kwargs: allow_checker)

    handler = handler_module.StdioProtocolHandler()
    handler._project_path = tmp_path

    responses: dict[str, dict[str, Any]] = {}

    async def capture(request_id: str, response: dict[str, Any] | None = None, error: str | None = None):
        responses[request_id] = {"response": response, "error": error}

    monkeypatch.setattr(handler, "_write_control_response", capture)

    await handler._handle_initialize({"options": {}}, "init")

    await handler._handle_can_use_tool(
        {"toolName": "Read", "input": {"value": "hello"}},
        "legacy_tool_name",
    )

    assert responses["legacy_tool_name"]["error"] == "Missing tool_name"


@pytest.mark.asyncio
async def test_stdio_set_output_style_ignores_legacy_output_style_alias(monkeypatch, tmp_path):
    tools = [DummyTool("Read")]
    _patch_stdio_dependencies(monkeypatch, tools)

    class _Style:
        def __init__(self, key: str) -> None:
            self.key = key

    captured_style: dict[str, str] = {}

    def fake_resolve_output_style(name: str, project_path=None):  # noqa: ARG001
        captured_style["name"] = name
        return _Style(str(name)), None

    monkeypatch.setattr(handler_control, "resolve_output_style", fake_resolve_output_style)

    handler = handler_module.StdioProtocolHandler()
    handler._project_path = tmp_path

    responses: dict[str, dict[str, Any]] = {}

    async def capture(request_id: str, response: dict[str, Any] | None = None, error: str | None = None):
        responses[request_id] = {"response": response, "error": error}

    monkeypatch.setattr(handler, "_write_control_response", capture)

    await handler._handle_set_output_style({"outputStyle": "learning"}, "set_style")

    assert captured_style["name"] == "default"
    assert responses["set_style"]["response"]["output_style"] == "default"


@pytest.mark.asyncio
async def test_stdio_initialize_uses_project_output_language(monkeypatch, tmp_path):
    tools = [DummyTool("Read")]
    _patch_stdio_dependencies(monkeypatch, tools)

    config_dir = tmp_path / ".ripperdoc"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.local.json").write_text(
        json.dumps({"output_language": "Chinese"}),
        encoding="utf-8",
    )

    captured_language: dict[str, str] = {}

    def fake_build_system_prompt(*_args, **kwargs):
        captured_language["value"] = kwargs.get("output_language")
        return "system"

    monkeypatch.setattr(handler_config, "build_system_prompt", fake_build_system_prompt)

    handler = handler_module.StdioProtocolHandler()

    responses: dict[str, dict[str, Any]] = {}

    async def capture(request_id: str, response: dict[str, Any] | None = None, error: str | None = None):
        responses[request_id] = {"response": response, "error": error}

    monkeypatch.setattr(handler, "_write_control_response", capture)

    await handler._handle_initialize({"options": {"cwd": str(tmp_path)}}, "init")

    assert captured_language["value"] == "Chinese"
    assert responses["init"]["response"]["output_language"] == "Chinese"


@pytest.mark.asyncio
async def test_stdio_initialize_output_language_option_overrides_project(monkeypatch, tmp_path):
    tools = [DummyTool("Read")]
    _patch_stdio_dependencies(monkeypatch, tools)

    config_dir = tmp_path / ".ripperdoc"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.local.json").write_text(
        json.dumps({"output_language": "Chinese"}),
        encoding="utf-8",
    )

    captured_language: dict[str, str] = {}

    def fake_build_system_prompt(*_args, **kwargs):
        captured_language["value"] = kwargs.get("output_language")
        return "system"

    monkeypatch.setattr(handler_config, "build_system_prompt", fake_build_system_prompt)

    handler = handler_module.StdioProtocolHandler()

    responses: dict[str, dict[str, Any]] = {}

    async def capture(request_id: str, response: dict[str, Any] | None = None, error: str | None = None):
        responses[request_id] = {"response": response, "error": error}

    monkeypatch.setattr(handler, "_write_control_response", capture)

    await handler._handle_initialize(
        {"options": {"cwd": str(tmp_path), "output_language": "Japanese"}},
        "init",
    )

    assert captured_language["value"] == "Japanese"
    assert responses["init"]["response"]["output_language"] == "Japanese"
