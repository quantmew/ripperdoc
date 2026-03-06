"""Alignment tests for stdio control/request shape handling."""

from __future__ import annotations

from pathlib import Path
import asyncio
from functools import lru_cache
from typing import Any

import pytest

from ripperdoc.utils.mcp import McpServerInfo
from ripperdoc.utils.mcp import parse_mcp_config_option
from ripperdoc.utils.mcp import _SdkMcpSession, _coerce_sdk_schema

from ripperdoc.protocol.models import DEFAULT_PROTOCOL_VERSION, JsonRpcErrorCodes
from ripperdoc import __version__
from ripperdoc.protocol.stdio import handler as handler_module
from ripperdoc.protocol.stdio import handler_control as handler_control_module
from ripperdoc.protocol.stdio import handler_query as handler_query_module
from ripperdoc.protocol.stdio import handler_session as handler_session_module
from ripperdoc.utils.messaging.messages import create_assistant_message, create_user_message


@lru_cache(maxsize=1)
def _load_claude_sdk_symbols() -> tuple[type[Any], type[Any], type[Any]]:
    sdk_module = pytest.importorskip(
        "claude_agent_sdk",
        reason="claude_agent_sdk is required for SDK alignment tests",
    )
    transport_module = pytest.importorskip(
        "claude_agent_sdk._internal.transport.subprocess_cli",
        reason="claude_agent_sdk transport internals are required for SDK alignment tests",
    )
    query_module = pytest.importorskip(
        "claude_agent_sdk._internal.query",
        reason="claude_agent_sdk query internals are required for SDK alignment tests",
    )
    return (
        sdk_module.ClaudeAgentOptions,
        transport_module.SubprocessCLITransport,
        query_module.Query,
    )


def test_coerce_initialize_request_applies_strict_defaults() -> None:
    handler = handler_module.StdioProtocolHandler()

    request = handler._coerce_initialize_request(
        {
            "request": {
                "subtype": "initialize",
                "protocolVersion": "2025-11-25",
                "capabilities": {"sampling": {"tools": True}},
                "clientInfo": {"name": "test-client", "version": "9.9.9"},
                "unexpected": "ignored",
            }
        }
    )

    assert request["protocolVersion"] == "2025-11-25"
    assert request["capabilities"] == {"sampling": {"tools": True}}
    assert request["clientInfo"] == {"name": "test-client", "version": "9.9.9"}
    assert "unexpected" not in request


def test_coerce_initialize_request_defaults_from_options() -> None:
    handler = handler_module.StdioProtocolHandler()
    request = handler._coerce_initialize_request({"options": {}})

    assert request["protocolVersion"] == DEFAULT_PROTOCOL_VERSION
    assert request["capabilities"] == {}
    assert request["clientInfo"] == {
        "name": "ripperdoc",
        "version": __version__,
    }


def test_build_sdk_init_stream_message_matches_expected_shape(monkeypatch, tmp_path: Path) -> None:
    handler = handler_module.StdioProtocolHandler()
    handler._project_path = tmp_path
    handler._session_id = "session-1"
    handler._permission_mode = "default"
    handler._output_style = "default"
    handler._active_agent_names = ["Explore", "Plan"]
    handler._enabled_skill_names = ["debug", "simplify"]
    handler._plugin_payloads = [{"name": "demo-plugin", "path": str(tmp_path / ".plugins/demo")}]
    handler._sdk_betas = ["compact-2026-01-12"]
    handler._query_context = type("QueryContextStub", (), {"model": "glm-5"})()

    monkeypatch.setattr(
        handler_session_module,
        "list_slash_commands",
        lambda: [type("Cmd", (), {"name": "compact"})(), type("Cmd", (), {"name": "review"})()],
    )
    monkeypatch.setattr(
        handler_session_module,
        "list_custom_commands",
        lambda _path: [type("Cmd", (), {"name": "release-notes"})()],
    )

    tools = [
        type("Tool", (), {"name": "Read"})(),
        type("Tool", (), {"name": "mcp__localtools__echo_notes"})(),
    ]
    servers = [McpServerInfo(name="localtools", status="connected", type="sdk", tools=[], resources=[])]

    payload = handler._build_sdk_init_stream_message(tools=tools, servers=servers)

    assert payload["type"] == "system"
    assert payload["subtype"] == "init"
    assert payload["cwd"] == str(tmp_path)
    assert payload["session_id"] == "session-1"
    assert payload["tools"] == ["Read", "mcp__localtools__echo_notes"]
    assert payload["mcp_servers"] == [{"name": "localtools", "status": "connected"}]
    assert payload["model"] == "glm-5"
    assert payload["permissionMode"] == "default"
    assert payload["slash_commands"] == ["compact", "review", "release-notes"]
    assert payload["apiKeySource"] == "none"
    assert payload["betas"] == ["compact-2026-01-12"]
    assert payload["ripperdoc_version"] == __version__
    assert payload["output_style"] == "default"
    assert payload["agents"] == ["Explore", "Plan"]
    assert payload["skills"] == ["debug", "simplify"]
    assert payload["plugins"] == [{"name": "demo-plugin", "path": str(tmp_path / ".plugins/demo")}]
    assert payload["fast_mode_state"] == "off"
    assert isinstance(payload["uuid"], str) and payload["uuid"]


@pytest.mark.asyncio
async def test_handle_initialize_defaults_from_subtype_init_control_request(monkeypatch) -> None:
    handler = handler_module.StdioProtocolHandler()
    written: list[dict[str, Any]] = []
    monkeypatch.setattr(handler_session_module, "get_effective_model_profile", lambda _model: object())

    async def fake_write_control_response(
        request_id: str,
        response: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        written.append({"id": request_id, "response": response, "error": error})

    monkeypatch.setattr(handler, "_write_control_response", fake_write_control_response)
    await handler._handle_initialize(
        {
            "subtype": "initialize",
            "hooks": {"preToolUse": []},
            "agents": None,
        },
        "i-0",
    )

    assert written
    assert written[0]["response"] is not None
    assert written[0]["error"] is None
    assert written[0]["response"]["protocolVersion"] == DEFAULT_PROTOCOL_VERSION
    assert written[0]["response"]["capabilities"] == {"tools": {"listChanged": False}, "sampling": {"tools": True}}
    assert written[0]["response"]["serverInfo"]["name"] == "ripperdoc"


def test_parse_mcp_config_option_accepts_inline_json() -> None:
    parsed = parse_mcp_config_option(
        '{"mcpServers":{"localtools":{"type":"sdk","name":"example-local-tools"}}}'
    )

    assert "localtools" in parsed
    assert parsed["localtools"].type == "sdk"


def test_coerce_sdk_schema_converts_shorthand_object_shape() -> None:
    schema = _coerce_sdk_schema({"numbers": list, "label": str, "count": int})

    assert schema == {
        "type": "object",
        "properties": {
            "numbers": {"type": "array", "items": {}},
            "label": {"type": "string"},
            "count": {"type": "integer"},
        },
        "required": ["numbers", "label", "count"],
    }


@pytest.mark.asyncio
async def test_sdk_mcp_session_list_tools_normalizes_shorthand_input_schema() -> None:
    async def fake_sender(_server_name: str, message: dict[str, Any]) -> dict[str, Any]:
        method = message.get("method")
        if method == "tools/list":
            return {
                "result": {
                    "tools": [
                        {
                            "name": "add_numbers",
                            "description": "Add numbers",
                            "inputSchema": {"numbers": list},
                            "annotations": {"readOnlyHint": True},
                        }
                    ]
                }
            }
        return {"result": {}}

    session = _SdkMcpSession("calc", fake_sender)
    result = await session.list_tools()

    assert len(result.tools) == 1
    assert result.tools[0].inputSchema == {
        "type": "object",
        "properties": {
            "numbers": {"type": "array", "items": {}},
        },
        "required": ["numbers"],
    }


@pytest.mark.asyncio
async def test_initialize_applies_cli_mcp_config_and_sdk_sender(monkeypatch, tmp_path: Path) -> None:
    handler = handler_module.StdioProtocolHandler(
        default_options={
            "mcp_config": '{"mcpServers":{"localtools":{"type":"sdk","name":"example-local-tools"}}}'
        }
    )
    written: list[dict[str, Any]] = []

    monkeypatch.setattr(handler_session_module, "get_effective_model_profile", lambda _model: object())
    monkeypatch.setattr(handler_session_module, "get_project_config", lambda _path: object())
    monkeypatch.setattr(
        handler_session_module,
        "get_project_local_config",
        lambda _path: type("Cfg", (), {"output_style": "default", "output_language": "auto"})(),
    )
    monkeypatch.setattr(handler_session_module, "set_runtime_plugin_dirs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(handler_session_module, "set_runtime_task_scope", lambda **_kwargs: None)
    monkeypatch.setattr(handler_session_module, "load_agent_definitions", lambda **_kwargs: None)
    monkeypatch.setattr(handler_session_module, "discover_plugins", lambda **_kwargs: None)
    monkeypatch.setattr(handler_session_module, "list_slash_commands", lambda: [])
    monkeypatch.setattr(handler_session_module, "list_custom_commands", lambda _path: [])
    monkeypatch.setattr(
        handler_session_module,
        "load_all_output_styles",
        lambda _path: type("Styles", (), {"styles": [type("S", (), {"key": "default"})()]})(),
    )
    monkeypatch.setattr(
        handler_session_module,
        "load_mcp_servers_async",
        lambda *_args, **_kwargs: asyncio.sleep(
            0,
            result=[McpServerInfo(name="localtools", type="sdk", status="connected", tools=[], resources=[])],
        ),
    )
    monkeypatch.setattr(
        handler_session_module,
        "load_dynamic_mcp_tools_async",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=[]),
    )
    monkeypatch.setattr(
        handler_session_module,
        "format_mcp_instructions",
        lambda _servers: "",
    )

    async def fake_write_control_response(
        request_id: str,
        response: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        written.append({"id": request_id, "response": response, "error": error})

    monkeypatch.setattr(handler, "_write_control_response", fake_write_control_response)

    await handler._handle_initialize(
        {
            "protocolVersion": DEFAULT_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0"},
            "_meta": {"ripperdoc_options": {"cwd": str(tmp_path)}},
        },
        "init-sdk",
    )

    assert written[-1]["error"] is None
    assert handler._mcp_server_overrides is not None
    assert "localtools" in handler._mcp_server_overrides
    assert handler._mcp_server_overrides["localtools"].type == "sdk"
    assert handler._send_sdk_mcp_message is not None


def test_convert_user_tool_result_to_sdk_uses_content_blocks_shape() -> None:
    handler = handler_module.StdioProtocolHandler()
    handler._session_id = "session-1"

    message = create_user_message(
        [{"type": "tool_result", "tool_use_id": "call_1", "text": "done"}],
        tool_use_result={
            "server": "localtools",
            "tool": "echo_notes",
            "text": "done",
            "content_blocks": [{"type": "text", "text": "done"}],
            "token_estimate": 12,
        },
    )

    payload = handler._convert_message_to_sdk(message)

    assert payload is not None
    assert payload["type"] == "user"
    assert payload["message"]["content"] == [
        {
            "type": "tool_result",
            "tool_use_id": "call_1",
            "content": [{"type": "text", "text": "done"}],
        }
    ]
    assert payload["tool_use_result"] == [{"type": "text", "text": "done"}]



@pytest.mark.asyncio
async def test_initialize_validation_rejects_unknown_clientinfo_fields(monkeypatch) -> None:
    handler = handler_module.StdioProtocolHandler()
    written: list[dict[str, Any]] = []

    async def fake_write_control_response(
        request_id: str,
        response: dict | None = None,
        error: dict | None = None,
    ) -> None:
        written.append({"id": request_id, "response": response, "error": error})

    monkeypatch.setattr(handler, "_write_control_response", fake_write_control_response)

    await handler._handle_initialize(
        {
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": {
                "name": 123,
                "version": "1.2.3",
                "unexpected": "x",
            },
        },
        "i-1",
    )

    assert written
    assert written[0]["error"]["code"] == int(JsonRpcErrorCodes.InvalidParams)


@pytest.mark.asyncio
async def test_handle_initialize_rejects_missing_required_fields_without_control_payload(monkeypatch) -> None:
    handler = handler_module.StdioProtocolHandler()
    written: list[dict[str, Any]] = []

    async def fake_write_control_response(
        request_id: str,
        response: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        written.append({"id": request_id, "response": response, "error": error})

    monkeypatch.setattr(handler, "_write_control_response", fake_write_control_response)
    await handler._handle_initialize({}, "i-0")

    assert written
    assert written[0]["error"]["code"] == int(JsonRpcErrorCodes.InvalidParams)


@pytest.mark.asyncio
async def test_query_validation_rejects_bad_payload(monkeypatch) -> None:
    handler = handler_module.StdioProtocolHandler()
    handler._initialized = True
    captured: dict[str, int | None] = {}

    async def fake_write(request_id: str, response: dict | None = None, error: dict | None = None) -> None:
        captured["code"] = error["code"] if error else None

    monkeypatch.setattr(handler, "_write_control_response", fake_write)

    await handler._handle_query(
        {
            "messages": "not-a-list",
            "maxTokens": 128,
            "clientInfo": {"name": "x", "version": "1.0"},
        },
        "q-3",
    )

    assert captured["code"] == int(JsonRpcErrorCodes.InvalidParams)


@pytest.mark.asyncio
async def test_query_init_message_refreshes_dynamic_tools_before_emit(monkeypatch, tmp_path: Path) -> None:
    handler = handler_module.StdioProtocolHandler()
    handler._initialized = True
    handler._project_path = tmp_path
    handler._session_id = "session-1"
    handler._permission_mode = "default"
    handler._query_context = type(
        "QueryContextStub",
        (),
        {
            "tools": [],
            "pending_message_queue": None,
            "hook_scopes": [],
        },
    )()
    handler._conversation_messages = []

    monkeypatch.setattr(
        "ripperdoc.protocol.stdio.handler_query.load_mcp_servers_async",
        lambda *_args, **_kwargs: asyncio.sleep(
            0,
            result=[McpServerInfo(name="localtools", status="connected", type="sdk", tools=[], resources=[])],
        ),
    )
    monkeypatch.setattr(
        "ripperdoc.protocol.stdio.handler_query.format_mcp_instructions",
        lambda _servers: "",
    )
    monkeypatch.setattr(
        handler,
        "_resolve_system_prompt",
        lambda tools, *_args, **_kwargs: ",".join(getattr(tool, "name", "") for tool in tools),
    )
    monkeypatch.setattr(handler, "_emit_hook_notices", lambda *_args, **_kwargs: asyncio.sleep(0))
    monkeypatch.setattr(
        handler,
        "_collect_prepare_inputs",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=([], [], None)),
    )
    monkeypatch.setattr(
        handler,
        "_coerce_sampling_messages",
        lambda _messages: [create_user_message("hello")],
    )
    monkeypatch.setattr(handler, "_validate_tool_message_sequence", lambda _messages: True)
    monkeypatch.setattr(handler, "_extract_latest_user_prompt", lambda *_args, **_kwargs: "hello")
    monkeypatch.setattr(
        handler,
        "_ensure_session_history",
        lambda: type(
            "H",
            (),
            {
                "path": tmp_path / "session.jsonl",
                "append": lambda self, _m: None,
            },
        )(),
    )

    async def fake_refresh() -> None:
        handler._query_context.tools = [
            type("Tool", (), {"name": "mcp__localtools__echo_notes"})(),
            type("Tool", (), {"name": "mcp__localtools__project_brief"})(),
        ]

    monkeypatch.setattr(handler, "_refresh_query_context_dynamic_tools", fake_refresh)

    written: list[dict[str, Any]] = []

    async def fake_write_message_stream(message: dict[str, Any]) -> None:
        written.append(message)

    monkeypatch.setattr(handler, "_write_message_stream", fake_write_message_stream)

    prepared = await handler._prepare_query_stage(
        {
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
            "maxTokens": 128,
        },
        "q-init-refresh",
        handler_query_module._QueryRuntimeState(start_time=0.0),
    )

    assert prepared is not None
    assert written
    assert written[0]["type"] == "system"
    assert written[0]["subtype"] == "init"
    assert written[0]["tools"] == [
        "mcp__localtools__echo_notes",
        "mcp__localtools__project_brief",
    ]


def test_coerce_query_request_accepts_prompt_shortcut() -> None:
    handler = handler_module.StdioProtocolHandler()
    request = handler._coerce_query_request({"prompt": "hello", "maxTokens": 128})

    assert request["messages"][0]["role"] == "user"
    assert request["messages"][0]["content"][0]["text"] == "hello"
    assert request["maxTokens"] == 128


@pytest.mark.asyncio
async def test_control_request_routes_query_subtype(monkeypatch) -> None:
    handler = handler_module.StdioProtocolHandler()

    captured: dict[str, object] = {}

    async def fake_handle_query(payload: dict, request_id: str) -> None:
        captured["payload"] = payload
        captured["request_id"] = request_id

    async def fake_write(*_args, **_kwargs) -> None:  # pragma: no cover
        return None

    monkeypatch.setattr(handler, "_handle_query", fake_handle_query)
    monkeypatch.setattr(handler, "_write_control_response", fake_write)

    await handler._handle_control_request(
        {
            "type": "control_request",
            "request_id": "q-1",
            "request": {"subtype": "query", "prompt": "hello from control"},
        }
    )

    assert captured["request_id"] == "q-1"
    assert captured["payload"] == {"subtype": "query", "prompt": "hello from control"}


@pytest.mark.asyncio
async def test_control_request_routes_jsonrpc_query(monkeypatch) -> None:
    handler = handler_module.StdioProtocolHandler()
    captured: dict[str, Any] = {}

    async def fake_handle_query(payload: dict, request_id: str) -> None:
        captured["payload"] = payload
        captured["request_id"] = request_id

    async def fake_write(*_args, **_kwargs) -> None:  # pragma: no cover
        return None

    monkeypatch.setattr(handler, "_handle_query", fake_handle_query)
    monkeypatch.setattr(handler, "_write_control_response", fake_write)

    await handler._handle_control_request(
        {
            "jsonrpc": "2.0",
            "id": "jsonrpc-1",
            "method": "sampling/createMessage",
            "params": {"prompt": "from jsonrpc", "maxTokens": 256},
        }
    )

    assert captured["request_id"] == "jsonrpc-1"
    assert captured["payload"] == {"prompt": "from jsonrpc", "maxTokens": 256}


@pytest.mark.asyncio
async def test_control_request_routes_set_output_style_alias(monkeypatch) -> None:
    handler = handler_module.StdioProtocolHandler()
    captured: dict[str, Any] = {}

    async def fake_handle_set_output_style(request: dict, request_id: str) -> None:
        captured["request"] = request
        captured["request_id"] = request_id

    monkeypatch.setattr(handler, "_handle_set_output_style", fake_handle_set_output_style)
    monkeypatch.setattr(
        handler,
        "_write_control_response",
        lambda *_args, **_kwargs: None,
    )

    await handler._handle_control_request(
        {
            "type": "json-rpc",
            "id": "set-style-1",
            "method": "set_output_style",
            "params": {"output_style": "diff", "outputStyle": "markdown"},
        }
    )

    assert captured["request_id"] == "set-style-1"
    assert captured["request"] == {"output_style": "diff", "outputStyle": "markdown"}


@pytest.mark.asyncio
async def test_control_request_unknown_subtype_returns_method_not_found_error(monkeypatch) -> None:
    handler = handler_module.StdioProtocolHandler()
    written: list[dict] = []

    async def fake_write_control_response(request_id: str, response=None, error=None) -> None:
        written.append({"request_id": request_id, "response": response, "error": error})

    monkeypatch.setattr(handler, "_write_control_response", fake_write_control_response)

    await handler._handle_control_request(
        {
            "type": "control_request",
            "request_id": "q-2",
            "request": {"subtype": "not_supported"},
        }
    )

    assert written and written[0]["error"]["code"] == int(JsonRpcErrorCodes.MethodNotFound)


@pytest.mark.asyncio
async def test_control_request_handles_mcp_status(monkeypatch) -> None:
    handler = handler_module.StdioProtocolHandler()
    written: list[dict[str, Any]] = []

    async def fake_load_mcp_servers_async(_path, **_kwargs):
        return [
            McpServerInfo(
                name="connected-server",
                status="connected",
                type="stdio",
                tools=[],
                resources=[],
            ),
            McpServerInfo(name="failed-server", status="failed", tools=[], resources=[]),
        ]

    async def fake_write_control_response(
        request_id: str,
        response: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        written.append({"id": request_id, "response": response, "error": error})

    monkeypatch.setattr(handler, "_write_control_response", fake_write_control_response)
    monkeypatch.setattr(
        "ripperdoc.protocol.stdio.handler_control.load_mcp_servers_async",
        fake_load_mcp_servers_async,
    )

    await handler._handle_control_request(
        {
            "type": "control_request",
            "request_id": "mcp-1",
            "request": {"subtype": "mcp_status"},
        }
    )

    assert written
    response = written[0]["response"]
    assert response["status"] == "mcp_status"
    assert response["connected"] == ["connected-server"]
    assert response["failed"] == ["failed-server"]
    assert len(response["servers"]) == 2


@pytest.mark.asyncio
async def test_control_request_handles_sdk_mcp_status_shape(monkeypatch) -> None:
    handler = handler_module.StdioProtocolHandler()
    written: list[dict[str, Any]] = []

    async def fake_load_mcp_servers_async(_path, **_kwargs):
        return [
            McpServerInfo(
                name="localtools",
                status="connected",
                type="sdk",
                tools=[],
                resources=[],
            )
        ]

    async def fake_write_control_response(
        request_id: str,
        response: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        written.append({"id": request_id, "response": response, "error": error})

    monkeypatch.setattr(handler, "_write_control_response", fake_write_control_response)
    monkeypatch.setattr(
        "ripperdoc.protocol.stdio.handler_control.load_mcp_servers_async",
        fake_load_mcp_servers_async,
    )

    await handler._handle_mcp_status({}, "mcp-sdk")

    assert written[-1]["error"] is None
    assert written[-1]["response"]["connected"] == ["localtools"]
    assert written[-1]["response"]["servers"][0]["config"] == {"type": "sdk"}


@pytest.mark.asyncio
async def test_control_request_interrupt_cancels_other_tasks(monkeypatch) -> None:
    handler = handler_module.StdioProtocolHandler()
    written: list[dict[str, Any]] = []
    handler._query_context = type("QueryContextStub", (), {"abort_controller": asyncio.Event()})()

    async def fake_task() -> None:
        await asyncio.Future()

    other_task = asyncio.create_task(fake_task())
    handler._inflight_tasks.add(other_task)

    async def fake_write_control_response(
        request_id: str,
        response: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        written.append({"id": request_id, "response": response, "error": error})

    monkeypatch.setattr(handler, "_write_control_response", fake_write_control_response)

    await handler._handle_control_request(
        {
            "type": "control_request",
            "request_id": "interrupt-1",
            "request": {"subtype": "interrupt"},
        }
    )

    assert written
    assert written[0]["response"]["status"] == "interrupt"
    assert written[0]["response"]["interrupt_signaled"] is True
    assert written[0]["response"]["cancelled_tasks"] >= 1
    assert handler._query_context.abort_controller.is_set() is True
    other_task.cancel()


@pytest.mark.asyncio
async def test_control_request_interrupt_does_not_cancel_query_task(monkeypatch) -> None:
    handler = handler_module.StdioProtocolHandler()
    written: list[dict[str, Any]] = []
    handler._query_context = type("QueryContextStub", (), {"abort_controller": asyncio.Event()})()

    cancelled = asyncio.Event()

    async def fake_query_task() -> None:
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    query_task = asyncio.create_task(fake_query_task())
    handler._inflight_tasks.add(query_task)
    handler._request_tasks["query-1"] = query_task
    handler._request_subtypes["query-1"] = "query"

    async def fake_write_control_response(
        request_id: str,
        response: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        written.append({"id": request_id, "response": response, "error": error})

    monkeypatch.setattr(handler, "_write_control_response", fake_write_control_response)

    await handler._handle_control_request(
        {
            "type": "control_request",
            "request_id": "interrupt-2",
            "request": {"subtype": "interrupt"},
        }
    )
    await asyncio.sleep(0)

    assert written
    assert written[0]["response"]["status"] == "interrupt"
    assert written[0]["response"]["interrupt_signaled"] is True
    assert written[0]["response"]["cancelled_tasks"] == 0
    assert handler._query_context.abort_controller.is_set() is True
    assert cancelled.is_set() is False

    query_task.cancel()


def test_sdk_transport_builds_command_with_cli_path() -> None:
    ClaudeAgentOptions, SubprocessCLITransport, _ = _load_claude_sdk_symbols()
    cli_path = "ripperdoc"
    options = ClaudeAgentOptions(cli_path=Path(cli_path))
    transport = SubprocessCLITransport(prompt="", options=options)
    command = transport._build_command()

    assert command[0] == cli_path
    assert command[1:3] == ["--output-format", "stream-json"]
    assert "--verbose" in command
    assert command[-2:] == ["--input-format", "stream-json"]


def test_sdk_transport_respects_cli_path_option_as_str_or_path() -> None:
    ClaudeAgentOptions, SubprocessCLITransport, _ = _load_claude_sdk_symbols()
    for cli_path in ("ripperdoc", Path("ripperdoc")):
        options = ClaudeAgentOptions(cli_path=cli_path)
        transport = SubprocessCLITransport(prompt="", options=options)
        command = transport._build_command()

        assert command[0] == str(cli_path)
        assert "--system-prompt" in command


@pytest.mark.asyncio
async def test_sdk_query_controls_payload_shape_for_setters_and_initialize(monkeypatch) -> None:
    _, _, Query = _load_claude_sdk_symbols()
    captured: list[dict[str, Any]] = []

    async def fake_send_control_request(self, request: dict[str, Any], timeout: float = 60.0) -> dict[str, Any]:
        captured.append(dict(request))
        return {"status": "ok"}

    monkeypatch.setattr(Query, "_send_control_request", fake_send_control_request)

    query = Query(
        transport=object(),
        is_streaming_mode=True,
        agents={"reviewer": {"description": "x", "prompt": "y"}},
    )
    # Query methods are thin wrappers for SDK protocol control payloads.
    await query.set_permission_mode("acceptEdits")
    await query.set_model("custom-model")
    await query.rewind_files("msg-1")
    await query.initialize()

    assert captured[0] == {
        "subtype": "set_permission_mode",
        "mode": "acceptEdits",
    }
    assert captured[1] == {
        "subtype": "set_model",
        "model": "custom-model",
    }
    assert captured[2] == {
        "subtype": "rewind_files",
        "user_message_id": "msg-1",
    }
    assert captured[3] == {
        "subtype": "initialize",
        "hooks": None,
        "agents": {"reviewer": {"description": "x", "prompt": "y"}},
    }


@pytest.mark.asyncio
async def test_run_stdio_print_mode_forwards_append_system_prompt(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class FakeHandler:
        def __init__(self, input_format, output_format, default_options=None):
            captured["default_options"] = dict(default_options or {})

        async def _handle_initialize(self, request, request_id):
            captured["initialize"] = {"request": request, "request_id": request_id}

        async def _handle_query(self, request, request_id):
            captured["query"] = {"request": request, "request_id": request_id}

        async def flush_output(self):
            captured["flushed"] = True

    monkeypatch.setattr(handler_module, "StdioProtocolHandler", FakeHandler)
    monkeypatch.setattr("ripperdoc.protocol.stdio.command.StdioProtocolHandler", FakeHandler)

    from ripperdoc.protocol.stdio import command as stdio_command

    await stdio_command.run_stdio(
        input_format="stream-json",
        output_format="json",
        model=None,
        permission_mode="default",
        max_turns=None,
        system_prompt=None,
        append_system_prompt="Append these instructions.",
        print_mode=True,
        prompt="hello",
    )

    assert captured["default_options"]["append_system_prompt"] == "Append these instructions."
    init_options = captured["initialize"]["request"]["_meta"]["ripperdoc_options"]
    assert init_options["append_system_prompt"] == "Append these instructions."


@pytest.mark.asyncio
async def test_query_append_system_prompt_extends_session_base_prompt(monkeypatch) -> None:
    handler = handler_module.StdioProtocolHandler()
    handler._project_path = Path.cwd()
    handler._initialized = True
    handler._session_id = "session-append"
    handler._conversation_messages = []
    handler._custom_system_prompt = "Base system prompt"
    handler._append_system_prompt = "Session append"
    handler._query_context = type("QueryContextStub", (), {"tools": []})()

    monkeypatch.setattr(
        handler_query_module,
        "load_mcp_servers_async",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=[]),
    )
    monkeypatch.setattr(handler_query_module, "format_mcp_instructions", lambda _servers: "")

    async def fake_refresh_query_context_dynamic_tools() -> None:
        return None

    monkeypatch.setattr(handler, "_refresh_query_context_dynamic_tools", fake_refresh_query_context_dynamic_tools)
    monkeypatch.setattr(handler, "_collect_prepare_inputs", lambda _prompt: asyncio.sleep(0, result=([], [], None)))
    monkeypatch.setattr(handler, "_build_sdk_init_stream_message", lambda **_kwargs: {"type": "system"})
    monkeypatch.setattr(handler, "_write_message_stream", lambda _message: asyncio.sleep(0))
    monkeypatch.setattr(handler, "_emit_hook_notices", lambda _notices: asyncio.sleep(0))

    prepared = await handler._prepare_query_stage(
        request={
            "messages": [{"role": "user", "content": "hello"}],
            "maxTokens": 32,
            "appendSystemPrompt": "Request append",
        },
        request_id="query-1",
        state=handler_query_module._QueryRuntimeState(start_time=0.0),
    )

    assert prepared is not None
    assert prepared.system_prompt == "Base system prompt\n\nRequest append"


@pytest.mark.asyncio
async def test_send_control_request_timeout_emits_control_cancel_request(monkeypatch) -> None:
    handler = handler_module.StdioProtocolHandler()
    sent: list[dict[str, Any]] = []

    async def fake_write_message(message: dict[str, Any]) -> None:
        sent.append(dict(message))

    monkeypatch.setattr(handler, "_write_message", fake_write_message)

    with pytest.raises(asyncio.TimeoutError):
        await handler._send_control_request("can_use_tool", {}, timeout=0.01)

    assert sent[0]["type"] == "control_request"
    assert sent[-1]["type"] == "control_cancel_request"
    assert sent[-1]["request_id"] == sent[0]["request_id"]


@pytest.mark.asyncio
async def test_control_cancel_request_cancels_inflight_control_task(monkeypatch) -> None:
    handler = handler_module.StdioProtocolHandler()
    cancelled = asyncio.Event()

    async def fake_handle_control_request(_message: dict[str, Any]) -> None:
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    monkeypatch.setattr(handler, "_handle_control_request", fake_handle_control_request)

    handler._spawn_control_request_task(
        {"type": "control_request", "request_id": "cancel-me", "request": {"subtype": "query"}}
    )
    await asyncio.sleep(0)
    await handler._handle_control_cancel_request({"type": "control_cancel_request", "request_id": "cancel-me"})
    await asyncio.sleep(0)
    assert cancelled.is_set()
    assert "cancel-me" not in handler._request_tasks


@pytest.mark.asyncio
async def test_rewind_files_supports_dry_run_and_rewind(monkeypatch) -> None:
    handler = handler_module.StdioProtocolHandler()
    written: list[dict[str, Any]] = []

    user_1 = create_user_message("first")
    assistant_1 = create_assistant_message("reply-1")
    user_2 = create_user_message("second")
    assistant_2 = create_assistant_message("reply-2")
    handler._conversation_messages = [user_1, assistant_1, user_2, assistant_2]

    async def fake_write_control_response(
        request_id: str,
        response: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        written.append({"id": request_id, "response": response, "error": error})

    monkeypatch.setattr(handler, "_write_control_response", fake_write_control_response)

    await handler._handle_rewind_files(
        {"user_message_id": user_1.uuid, "dry_run": True},
        "rewind-dry",
    )
    assert written[-1]["response"]["canRewind"] is True
    assert written[-1]["response"]["dryRun"] is True
    assert len(handler._conversation_messages) == 4

    await handler._handle_rewind_files({"user_message_id": user_1.uuid}, "rewind-apply")
    assert written[-1]["response"]["canRewind"] is True
    assert written[-1]["response"]["dryRun"] is False
    assert len(handler._conversation_messages) == 1


@pytest.mark.asyncio
async def test_mcp_set_servers_and_toggle(monkeypatch) -> None:
    handler = handler_module.StdioProtocolHandler()
    written: list[dict[str, Any]] = []

    async def fake_write_control_response(
        request_id: str,
        response: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        written.append({"id": request_id, "response": response, "error": error})

    async def fake_reload() -> list[McpServerInfo]:
        return [McpServerInfo(name="svc", status="connected", type="stdio", tools=[], resources=[])]

    async def noop_refresh_tools() -> None:
        return None

    monkeypatch.setattr(handler, "_write_control_response", fake_write_control_response)
    monkeypatch.setattr(handler, "_reload_mcp_runtime", fake_reload)
    monkeypatch.setattr(handler, "_refresh_query_context_dynamic_tools", noop_refresh_tools)
    monkeypatch.setattr(
        handler_control_module,
        "load_mcp_server_configs",
        lambda _path: {"svc": McpServerInfo(name="svc", type="stdio", command="echo")},
    )

    await handler._handle_mcp_set_servers(
        {"servers": {"svc": {"command": "echo", "args": ["ok"]}}},
        "mcp-set",
    )
    assert written[-1]["response"]["status"] == "mcp_set_servers"
    assert "svc" in (handler._mcp_server_overrides or {})

    await handler._handle_mcp_toggle({"serverName": "svc", "enabled": False}, "mcp-toggle-off")
    assert written[-1]["response"]["status"] == "mcp_toggle"
    assert written[-1]["response"]["enabled"] is False
    assert "svc" in handler._mcp_disabled_servers

    await handler._handle_mcp_toggle({"serverName": "svc", "enabled": True}, "mcp-toggle-on")
    assert written[-1]["response"]["enabled"] is True
    assert "svc" not in handler._mcp_disabled_servers


@pytest.mark.asyncio
async def test_mcp_authenticate_and_clear_auth(monkeypatch) -> None:
    handler = handler_module.StdioProtocolHandler()
    written: list[dict[str, Any]] = []

    async def fake_write_control_response(
        request_id: str,
        response: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        written.append({"id": request_id, "response": response, "error": error})

    async def fake_reload() -> list[McpServerInfo]:
        return [McpServerInfo(name="svc", status="connected", type="http", tools=[], resources=[])]

    async def noop_refresh_tools() -> None:
        return None

    monkeypatch.setattr(handler, "_write_control_response", fake_write_control_response)
    monkeypatch.setattr(handler, "_reload_mcp_runtime", fake_reload)
    monkeypatch.setattr(handler, "_refresh_query_context_dynamic_tools", noop_refresh_tools)
    monkeypatch.setattr(
        handler_control_module,
        "load_mcp_server_configs",
        lambda _path: {
            "svc": McpServerInfo(
                name="svc",
                type="http",
                url="https://example.test/mcp",
                headers={},
            )
        },
    )

    await handler._handle_mcp_authenticate({"serverName": "svc", "token": "secret-token"}, "mcp-auth")
    assert written[-1]["response"]["status"] == "mcp_authenticate"
    assert written[-1]["response"]["requiresUserAction"] is False
    assert (handler._mcp_server_overrides or {})["svc"].headers.get("Authorization") == "Bearer secret-token"

    await handler._handle_mcp_clear_auth({"serverName": "svc"}, "mcp-clear-auth")
    assert written[-1]["response"]["status"] == "mcp_clear_auth"
    assert "Authorization" not in (handler._mcp_server_overrides or {})["svc"].headers


@pytest.mark.asyncio
async def test_mcp_reconnect_unknown_server_returns_error(monkeypatch) -> None:
    handler = handler_module.StdioProtocolHandler()
    written: list[dict[str, Any]] = []

    async def fake_write_control_response(
        request_id: str,
        response: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        written.append({"id": request_id, "response": response, "error": error})

    monkeypatch.setattr(handler, "_write_control_response", fake_write_control_response)
    monkeypatch.setattr(
        handler_control_module,
        "load_mcp_server_configs",
        lambda _path: {"svc": McpServerInfo(name="svc", type="stdio", command="echo")},
    )

    await handler._handle_mcp_reconnect({"serverName": "missing"}, "mcp-reconnect")
    assert written[-1]["error"]["code"] == int(JsonRpcErrorCodes.InvalidParams)
