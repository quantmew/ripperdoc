"""Tests for structured task notification helpers in task tool flows."""

from __future__ import annotations

import asyncio
import time

import pytest

from ripperdoc.core.query import QueryContext
from ripperdoc.tools.task_tool import AgentRunRecord, TaskTool
from ripperdoc.utils.messaging.pending_messages import PendingMessageQueue
from ripperdoc.utils.collaboration.task_notifications import parse_task_notification


def test_task_tool_background_completion_enqueues_task_notification() -> None:
    tool = TaskTool(lambda: [])
    queue = PendingMessageQueue()
    record = AgentRunRecord(
        agent_id="agent_test1234",
        agent_type="general-purpose",
        tools=[],
        system_prompt="system",
        history=[],
        missing_tools=[],
        model_used="main",
        start_time=time.time(),
        output_file="/tmp/agent_test1234.log",
        is_background=True,
        status="completed",
        result_text="Subagent finished successfully.",
        usage={
            "input_tokens": 12,
            "output_tokens": 34,
        },
    )

    tool._enqueue_background_completion_notification(  # type: ignore[attr-defined]
        record=record,
        queue=queue,
        parent_tool_use_id="toolu_test123",
    )

    pending = queue.drain()
    assert len(pending) == 1
    message = getattr(pending[0], "message", None)
    assert message is not None
    content = getattr(message, "content", "")
    metadata = getattr(message, "metadata", {})
    assert isinstance(content, str)
    assert "<task-notification>" in content
    assert "<task-id>agent_test1234</task-id>" in content
    assert metadata.get("notification_type") == "task_notification"
    assert metadata.get("task_id") == "agent_test1234"
    assert metadata.get("tool_use_id") == "toolu_test123"
    assert metadata.get("source") == "background_task"


def test_parse_task_notification_extracts_core_fields() -> None:
    payload = (
        "<task-notification>\n"
        "<task-id>agent_abc</task-id>\n"
        "<status>completed</status>\n"
        "<summary>finished</summary>\n"
        "<tool-use-id>toolu_1</tool-use-id>\n"
        "<output-file>/tmp/out.log</output-file>\n"
        "<usage>{\"total_tokens\":10}</usage>\n"
        "</task-notification>"
    )
    parsed = parse_task_notification(payload)
    assert parsed is not None
    assert parsed.get("task_id") == "agent_abc"
    assert parsed.get("status") == "completed"
    assert parsed.get("summary") == "finished"
    assert parsed.get("tool_use_id") == "toolu_1"
    assert parsed.get("output_file") == "/tmp/out.log"
    assert parsed.get("usage", {}).get("total_tokens") == 10


def test_query_context_uses_dedicated_task_notification_queue_by_default() -> None:
    context = QueryContext(tools=[])
    assert context.task_notification_queue is not context.pending_message_queue


@pytest.mark.asyncio
async def test_task_tool_handoff_foreground_run_to_background_starts_task(monkeypatch) -> None:
    tool = TaskTool(lambda: [])
    started = asyncio.Event()

    async def fake_run_background(
        record: AgentRunRecord,  # noqa: ARG001
        subagent_context: QueryContext,  # noqa: ARG001
        permission_checker,  # noqa: ANN001, ARG001
        *,
        notification_queue: PendingMessageQueue | None = None,  # noqa: ARG001
        parent_tool_use_id: str | None = None,  # noqa: ARG001
    ) -> None:
        started.set()

    monkeypatch.setattr(tool, "_run_subagent_background", fake_run_background)

    record = AgentRunRecord(
        agent_id="agent_handoff",
        agent_type="general-purpose",
        tools=[],
        system_prompt="system",
        history=[],
        missing_tools=[],
        model_used="main",
        start_time=time.time(),
    )
    context = QueryContext(tools=[])
    queue = PendingMessageQueue()

    ok = tool._handoff_foreground_run_to_background(  # type: ignore[attr-defined]
        record=record,
        subagent_context=context,
        permission_checker=None,
        notification_queue=queue,
        parent_tool_use_id="toolu_handoff",
    )

    assert ok is True
    assert record.task is not None
    await asyncio.wait_for(started.wait(), timeout=1.0)
    await asyncio.wait_for(record.task, timeout=1.0)
