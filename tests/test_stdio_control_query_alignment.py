"""Alignment tests for stdio control/request shape handling."""

from __future__ import annotations

from pathlib import Path
import asyncio
from typing import Any

import pytest

from claude_agent_sdk import ClaudeAgentOptions
from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport
from ripperdoc.utils.mcp import McpServerInfo

from ripperdoc.protocol.models import DEFAULT_PROTOCOL_VERSION, JsonRpcErrorCodes
from ripperdoc import __version__
from ripperdoc.protocol.stdio import handler as handler_module
from ripperdoc.protocol.stdio import handler_control as handler_control_module
from claude_agent_sdk._internal.query import Query
from ripperdoc.utils.messages import create_assistant_message, create_user_message


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


@pytest.mark.asyncio
async def test_handle_initialize_defaults_from_subtype_init_control_request(monkeypatch) -> None:
    handler = handler_module.StdioProtocolHandler()
    written: list[dict[str, Any]] = []

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

    async def fake_load_mcp_servers_async(_path):
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
async def test_control_request_interrupt_cancels_other_tasks(monkeypatch) -> None:
    handler = handler_module.StdioProtocolHandler()
    written: list[dict[str, Any]] = []

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
    assert written[0]["response"]["cancelled_tasks"] >= 1
    other_task.cancel()


def test_sdk_transport_builds_command_with_cli_path() -> None:
    cli_path = "ripperdoc"
    options = ClaudeAgentOptions(cli_path=Path(cli_path))
    transport = SubprocessCLITransport(prompt="", options=options)
    command = transport._build_command()

    assert command[0] == cli_path
    assert command[1:3] == ["--output-format", "stream-json"]
    assert "--verbose" in command
    assert command[-2:] == ["--input-format", "stream-json"]


def test_sdk_transport_respects_cli_path_option_as_str_or_path() -> None:
    for cli_path in ("ripperdoc", Path("ripperdoc")):
        options = ClaudeAgentOptions(cli_path=cli_path)
        transport = SubprocessCLITransport(prompt="", options=options)
        command = transport._build_command()

        assert command[0] == str(cli_path)
        assert "--system-prompt" in command


@pytest.mark.asyncio
async def test_sdk_query_controls_payload_shape_for_setters_and_initialize(monkeypatch) -> None:
    captured: list[dict[str, Any]] = []

    async def fake_send_control_request(self, request: dict[str, Any], timeout: float = 60.0) -> dict[str, Any]:
        captured.append(dict(request))
        return {"status": "ok"}

    monkeypatch.setattr(
        "claude_agent_sdk._internal.query.Query._send_control_request",
        fake_send_control_request,
    )

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
