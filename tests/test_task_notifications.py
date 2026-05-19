"""Tests for structured task notification helpers in task tool flows."""

from __future__ import annotations



from ripperdoc.core.query import QueryContext
from ripperdoc.utils.collaboration.task_notifications import parse_task_notification


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
