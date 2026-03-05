from __future__ import annotations

import asyncio
import time
from typing import Any

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


def test_spawn_task_notification_followup_query_dispatches_control_request(monkeypatch) -> None:
    handler = handler_module.StdioProtocolHandler()
    captured_control_requests: list[dict] = []

    def _fake_spawn(message: dict) -> None:
        captured_control_requests.append(message)

    monkeypatch.setattr(handler, "_spawn_control_request_task", _fake_spawn)

    handler._spawn_task_notification_followup_query("background task finished")

    assert len(captured_control_requests) == 1
    dispatched = captured_control_requests[0]
    assert dispatched["type"] == "control_request"
    assert dispatched["request"]["subtype"] == "query"
    assert dispatched["request"]["prompt"] == "background task finished"


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
async def test_run_dispatches_control_request(monkeypatch) -> None:
    handler = handler_module.StdioProtocolHandler()
    captured: list[dict[str, Any]] = []

    async def fake_read_messages():
        yield {
            "type": "control_request",
            "request_id": "control-1",
            "request": {"subtype": "query", "prompt": "hello from control"},
        }

    def fake_spawn_control_request_task(message: dict[str, Any]) -> None:
        captured.append(message)

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

    assert len(captured) == 1
    assert captured[0]["type"] == "control_request"
    assert captured[0]["request"]["subtype"] == "query"
    assert captured[0]["request"]["prompt"] == "hello from control"


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


@pytest.mark.asyncio
async def test_run_auto_exits_after_idle_delay(monkeypatch) -> None:
    handler = handler_module.StdioProtocolHandler()
    monkeypatch.setenv("RIPPERDOC_EXIT_AFTER_STOP_DELAY", "30")

    async def fake_read_messages():
        while True:
            await asyncio.sleep(3600)
            yield {}

    async def noop_async(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr(handler, "_read_messages", fake_read_messages)
    monkeypatch.setattr(handler, "flush_output", noop_async)
    monkeypatch.setattr(handler, "_run_session_end", noop_async)
    monkeypatch.setattr(handler_runtime, "shutdown_mcp_runtime", noop_async)
    monkeypatch.setattr(handler_runtime, "shutdown_lsp_manager", noop_async)
    monkeypatch.setattr(handler_runtime, "shutdown_background_shell", lambda force=True: None)

    start = time.monotonic()
    await handler.run()
    elapsed = time.monotonic() - start

    assert elapsed < 1.0


@pytest.mark.asyncio
async def test_run_idle_exit_waits_for_inflight_completion(monkeypatch) -> None:
    handler = handler_module.StdioProtocolHandler()
    monkeypatch.setenv("RIPPERDOC_EXIT_AFTER_STOP_DELAY", "120")
    completed: list[str] = []

    async def fake_read_messages():
        yield {
            "type": "control_request",
            "request_id": "control-idle-exit",
            "request": {"subtype": "query", "prompt": "hello"},
        }
        while True:
            await asyncio.sleep(3600)
            yield {}

    async def fake_handle_control_request(_message: dict[str, Any]) -> None:
        await asyncio.sleep(0.05)
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


@pytest.mark.asyncio
async def test_run_replays_user_messages_when_enabled(monkeypatch) -> None:
    handler = handler_module.StdioProtocolHandler()
    handler._replay_user_messages = True
    captured_control_requests: list[dict[str, Any]] = []
    captured_stream: list[dict[str, Any]] = []

    async def fake_read_messages():
        yield {
            "type": "user",
            "uuid": "replay-user-1",
            "message": {"role": "user", "content": "dispatch prompt"},
        }

    def fake_spawn_control_request_task(message: dict[str, Any]) -> None:
        captured_control_requests.append(message)

    async def capture_stream(message: dict[str, Any]) -> None:
        captured_stream.append(message)

    async def noop_async(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr(handler, "_read_messages", fake_read_messages)
    monkeypatch.setattr(handler, "_spawn_control_request_task", fake_spawn_control_request_task)
    monkeypatch.setattr(handler, "_write_message_stream", capture_stream)
    monkeypatch.setattr(handler, "flush_output", noop_async)
    monkeypatch.setattr(handler, "_run_session_end", noop_async)
    monkeypatch.setattr(handler_runtime, "shutdown_mcp_runtime", noop_async)
    monkeypatch.setattr(handler_runtime, "shutdown_lsp_manager", noop_async)
    monkeypatch.setattr(handler_runtime, "shutdown_background_shell", lambda force=True: None)

    await handler.run()

    assert len(captured_control_requests) == 1
    assert captured_control_requests[0]["request_id"] == "replay-user-1"
    replay_payload = next((item for item in captured_stream if item.get("type") == "user"), None)
    assert replay_payload is not None
    assert replay_payload["uuid"] == "replay-user-1"
    assert replay_payload["isReplay"] is True


@pytest.mark.asyncio
async def test_run_replays_control_response_and_assistant_when_enabled(monkeypatch) -> None:
    handler = handler_module.StdioProtocolHandler()
    handler._replay_user_messages = True
    captured_stream: list[dict[str, Any]] = []

    async def fake_read_messages():
        yield {
            "type": "control_response",
            "response": {
                "request_id": "missing-pending",
                "subtype": "success",
                "response": {},
            },
        }
        yield {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "hello"}],
                "model": "main",
            },
        }

    async def capture_stream(message: dict[str, Any]) -> None:
        captured_stream.append(message)

    async def noop_async(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr(handler, "_read_messages", fake_read_messages)
    monkeypatch.setattr(handler, "_write_message_stream", capture_stream)
    monkeypatch.setattr(handler, "flush_output", noop_async)
    monkeypatch.setattr(handler, "_run_session_end", noop_async)
    monkeypatch.setattr(handler_runtime, "shutdown_mcp_runtime", noop_async)
    monkeypatch.setattr(handler_runtime, "shutdown_lsp_manager", noop_async)
    monkeypatch.setattr(handler_runtime, "shutdown_background_shell", lambda force=True: None)

    await handler.run()

    assert any(item.get("type") == "control_response" for item in captured_stream)
    assert any(item.get("type") == "assistant" for item in captured_stream)


@pytest.mark.asyncio
async def test_run_skips_duplicate_user_message_and_replays_ack(monkeypatch) -> None:
    handler = handler_module.StdioProtocolHandler()
    handler._replay_user_messages = True
    captured_control_requests: list[dict[str, Any]] = []
    captured_stream: list[dict[str, Any]] = []

    async def fake_read_messages():
        payload = {
            "type": "user",
            "uuid": "dup-user-1",
            "message": {"role": "user", "content": "same prompt"},
        }
        yield payload
        yield payload

    def fake_spawn_control_request_task(message: dict[str, Any]) -> None:
        captured_control_requests.append(message)

    async def capture_stream(message: dict[str, Any]) -> None:
        captured_stream.append(message)

    async def noop_async(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr(handler, "_read_messages", fake_read_messages)
    monkeypatch.setattr(handler, "_spawn_control_request_task", fake_spawn_control_request_task)
    monkeypatch.setattr(handler, "_write_message_stream", capture_stream)
    monkeypatch.setattr(handler, "flush_output", noop_async)
    monkeypatch.setattr(handler, "_run_session_end", noop_async)
    monkeypatch.setattr(handler_runtime, "shutdown_mcp_runtime", noop_async)
    monkeypatch.setattr(handler_runtime, "shutdown_lsp_manager", noop_async)
    monkeypatch.setattr(handler_runtime, "shutdown_background_shell", lambda force=True: None)

    await handler.run()

    assert len(captured_control_requests) == 1
    replay_messages = [item for item in captured_stream if item.get("type") == "user"]
    assert len(replay_messages) == 2
    assert all(item.get("isReplay") is True for item in replay_messages)
