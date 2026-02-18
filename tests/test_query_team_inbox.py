"""Team inbox injection behavior tests."""

from __future__ import annotations

import json

from ripperdoc.core.query import loop as loop_mod
from ripperdoc.utils.pending_messages import PendingMessageQueue


def test_inject_team_inbox_prioritizes_shutdown_and_wraps_messages(monkeypatch):
    shutdown_payload = {
        "type": "shutdown_request",
        "requestId": "req_abc",
        "from": "team-lead",
        "reason": "Task complete.",
        "timestamp": "2026-01-01T00:00:00Z",
    }
    inbox_entries = [
        {
            "id": "msg_2",
            "team_name": "alpha",
            "recipient": "agent-a",
            "sender": "team-lead",
            "message_type": "message",
            "content": "Please summarize your progress.",
            "created_at": 2.0,
            "metadata": {"summary": "Status check"},
            "read": False,
        },
        {
            "id": "msg_1",
            "team_name": "alpha",
            "recipient": "agent-a",
            "sender": "team-lead",
            "message_type": "shutdown_request",
            "content": json.dumps(shutdown_payload, ensure_ascii=False),
            "created_at": 1.0,
            "metadata": {},
            "read": False,
        },
    ]

    monkeypatch.setattr(loop_mod, "drain_team_inbox_messages", lambda *_args, **_kwargs: inbox_entries)

    queue = PendingMessageQueue()
    injected = loop_mod._inject_team_inbox_messages("alpha", "agent-a", queue)

    assert injected == 2
    drained = queue.drain()
    assert len(drained) == 2
    assert drained[0].message.metadata.get("team_message_type") == "shutdown_request"
    assert isinstance(drained[0].message.content, str)
    assert "<teammate-message" in drained[0].message.content
    assert drained[0].message.metadata.get("request_id") == "req_abc"
    assert isinstance(drained[1].message.content, str)
    assert "<teammate-message" in drained[1].message.content
