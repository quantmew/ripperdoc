"""Compatibility tests for Claude-style `type=user` stdio input."""

from __future__ import annotations

import asyncio

import pytest

from ripperdoc.protocol.stdio import handler as handler_module
from ripperdoc.protocol.stdio import handler_runtime


def test_coerce_user_message_to_control_request() -> None:
    handler = handler_module.StdioProtocolHandler()

    message = {
        "type": "user",
        "uuid": "u1",
        "message": {"role": "user", "content": "hello world"},
    }
    control_message = handler._coerce_user_message_to_control_request(message)

    assert control_message is not None
    assert control_message["type"] == "control_request"
    assert control_message["request_id"] == "u1"
    assert control_message["request"]["subtype"] == "query"
    assert control_message["request"]["prompt"] == "hello world"

    block_message = {
        "type": "user",
        "message": {
            "role": "user",
            "content": [
                {"type": "text", "text": "first line"},
                {"type": "tool_result", "tool_use_id": "x", "content": "ignored"},
                {"type": "text", "text": "second line"},
            ],
        },
    }
    block_control_message = handler._coerce_user_message_to_control_request(block_message)
    assert block_control_message is not None
    assert block_control_message["request"]["prompt"] == "first line\nsecond line"

    invalid_message = {"type": "user", "message": {"role": "user", "content": []}}
    assert handler._coerce_user_message_to_control_request(invalid_message) is None

    invalid_type_message = {
        "type": "assistant",
        "message": {"role": "user", "content": "wrong type"},
    }
    assert handler._coerce_user_message_to_control_request(invalid_type_message) is None

    invalid_role_message = {
        "type": "user",
        "message": {"role": "assistant", "content": "wrong role"},
    }
    assert handler._coerce_user_message_to_control_request(invalid_role_message) is None


@pytest.mark.asyncio
async def test_run_dispatches_claude_style_user_message(monkeypatch) -> None:
    handler = handler_module.StdioProtocolHandler()
    captured_control_requests: list[dict] = []

    async def fake_read_messages():
        yield {
            "type": "user",
            "uuid": "dispatch-1",
            "message": {"role": "user", "content": "dispatch prompt"},
        }

    def fake_spawn_control_request_task(message: dict) -> None:
        captured_control_requests.append(message)

    async def noop_async(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr(handler, "_read_messages", fake_read_messages)
    monkeypatch.setattr(handler, "_spawn_control_request_task", fake_spawn_control_request_task)
    monkeypatch.setattr(handler, "flush_output", noop_async)
    monkeypatch.setattr(handler, "_run_session_end", noop_async)
    monkeypatch.setattr(handler_runtime, "shutdown_mcp_runtime", noop_async)
    monkeypatch.setattr(handler_runtime, "shutdown_lsp_manager", noop_async)
    monkeypatch.setattr(handler_runtime, "shutdown_background_shell", lambda force=True: None)

    await handler.run()

    assert len(captured_control_requests) == 1
    dispatched = captured_control_requests[0]
    assert dispatched["type"] == "control_request"
    assert dispatched["request_id"] == "dispatch-1"
    assert dispatched["request"]["subtype"] == "query"
    assert dispatched["request"]["prompt"] == "dispatch prompt"


@pytest.mark.asyncio
async def test_run_waits_for_inflight_user_query_on_eof(monkeypatch) -> None:
    handler = handler_module.StdioProtocolHandler()
    completed: list[str] = []

    async def fake_read_messages():
        yield {
            "type": "user",
            "uuid": "dispatch-2",
            "message": {"role": "user", "content": "delayed prompt"},
        }

    async def fake_handle_control_request(_message: dict) -> None:
        await asyncio.sleep(0.02)
        completed.append("done")

    async def noop_async(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr(handler, "_read_messages", fake_read_messages)
    monkeypatch.setattr(handler, "_handle_control_request", fake_handle_control_request)
    monkeypatch.setattr(handler, "flush_output", noop_async)
    monkeypatch.setattr(handler, "_run_session_end", noop_async)
    monkeypatch.setattr(handler_runtime, "shutdown_mcp_runtime", noop_async)
    monkeypatch.setattr(handler_runtime, "shutdown_lsp_manager", noop_async)
    monkeypatch.setattr(handler_runtime, "shutdown_background_shell", lambda force=True: None)

    await handler.run()

    assert completed == ["done"]
