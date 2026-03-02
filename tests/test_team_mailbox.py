"""Mailbox compatibility tests for team messaging."""

from __future__ import annotations

import json
from pathlib import Path

from ripperdoc.utils.pending_messages import PendingMessageQueue
from ripperdoc.utils.teams import (
    clear_mailbox,
    create_team,
    drain_team_inbox_messages,
    get_inbox_path,
    mark_message_as_read_by_index,
    mark_messages_as_read,
    read_mailbox,
    read_unread_messages,
    register_team_message_listener,
    send_team_message,
    unregister_team_message_listener,
)


def test_mailbox_read_and_mark_flow(tmp_path, monkeypatch):
    monkeypatch.setattr("ripperdoc.utils.tasks.Path.home", lambda: tmp_path)
    monkeypatch.setattr("ripperdoc.utils.teams.Path.home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)

    create_team(name="alpha")
    send_team_message(
        team_name="alpha",
        sender="team-lead",
        recipients=["dev-a"],
        message_type="message",
        content="first",
    )
    send_team_message(
        team_name="alpha",
        sender="team-lead",
        recipients=["dev-a"],
        message_type="message",
        content="second",
    )

    mailbox = read_mailbox("dev-a", "alpha")
    assert len(mailbox) == 2
    assert len(read_unread_messages("dev-a", "alpha")) == 2

    assert mark_message_as_read_by_index("dev-a", "alpha", 0) is True
    assert len(read_unread_messages("dev-a", "alpha")) == 1

    assert mark_messages_as_read("dev-a", "alpha") == 1
    assert len(read_unread_messages("dev-a", "alpha")) == 0


def test_clear_mailbox_keeps_inbox_file(tmp_path, monkeypatch):
    monkeypatch.setattr("ripperdoc.utils.tasks.Path.home", lambda: tmp_path)
    monkeypatch.setattr("ripperdoc.utils.teams.Path.home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)

    create_team(name="alpha")
    send_team_message(
        team_name="alpha",
        sender="team-lead",
        recipients=["dev-a"],
        message_type="message",
        content="hello",
    )

    clear_mailbox("dev-a", "alpha")
    assert read_mailbox("dev-a", "alpha") == []

    inbox_path = Path(get_inbox_path("dev-a", "alpha"))
    assert inbox_path.exists()
    assert json.loads(inbox_path.read_text(encoding="utf-8")) == []


def test_live_listener_delivery_persists_read_entry(tmp_path, monkeypatch):
    monkeypatch.setattr("ripperdoc.utils.tasks.Path.home", lambda: tmp_path)
    monkeypatch.setattr("ripperdoc.utils.teams.Path.home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)

    create_team(name="alpha")
    queue = PendingMessageQueue()
    register_team_message_listener("alpha", "dev-a", queue)
    try:
        send_team_message(
            team_name="alpha",
            sender="team-lead",
            recipients=["dev-a"],
            message_type="message",
            content="sync now",
        )
    finally:
        unregister_team_message_listener("alpha", "dev-a", queue)

    delivered = queue.drain()
    assert delivered
    mailbox = read_mailbox("dev-a", "alpha")
    assert len(mailbox) == 1
    assert mailbox[0].get("read") is True
    assert drain_team_inbox_messages("alpha", "dev-a") == []
