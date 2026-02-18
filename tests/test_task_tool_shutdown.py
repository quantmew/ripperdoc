"""Shutdown-response handling tests for TaskTool."""

from __future__ import annotations

from ripperdoc.core.message_utils import tool_result_message
from ripperdoc.tools.task_tool import TaskTool
from ripperdoc.utils.messages import create_assistant_message


def test_extract_approved_shutdown_response_from_tool_result() -> None:
    assistant = create_assistant_message(
        [
            {
                "type": "tool_use",
                "id": "tool-1",
                "name": "SendMessage",
                "input": {
                    "type": "shutdown_response",
                    "request_id": "req_123",
                    "approve": True,
                    "content": "Task complete, shutting down.",
                },
            }
        ]
    )
    result_message = tool_result_message("tool-1", "ok", tool_use_result={"success": True})

    payload = TaskTool._extract_approved_shutdown_response([assistant], result_message)
    assert payload is not None
    assert payload.get("request_id") == "req_123"
    assert payload.get("content") == "Task complete, shutting down."


def test_extract_approved_shutdown_response_ignores_rejections() -> None:
    assistant = create_assistant_message(
        [
            {
                "type": "tool_use",
                "id": "tool-2",
                "name": "SendMessage",
                "input": {
                    "type": "shutdown_response",
                    "request_id": "req_456",
                    "approve": False,
                    "content": "Still processing final checks.",
                },
            }
        ]
    )
    result_message = tool_result_message("tool-2", "ok", tool_use_result={"success": True})

    payload = TaskTool._extract_approved_shutdown_response([assistant], result_message)
    assert payload is None
