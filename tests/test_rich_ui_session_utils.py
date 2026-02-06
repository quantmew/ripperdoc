"""Focused utility tests for RichUI history/compaction helpers."""

from __future__ import annotations

from typing import Any, List

import pytest

from ripperdoc.cli.ui.rich_ui.session import RichUI
from ripperdoc.utils.messages import create_assistant_message, create_user_message


class _DummyConsole:
    def __init__(self) -> None:
        self.lines: List[str] = []

    def print(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        rendered = " ".join(str(part) for part in args)
        self.lines.append(rendered)

    def clear(self) -> None:
        return None


def _new_ui() -> RichUI:
    ui = RichUI.__new__(RichUI)
    ui.console = _DummyConsole()
    ui.session_id = "test-session"
    ui.conversation_messages = []
    ui._saved_conversation = None
    return ui


def test_build_history_candidates_skips_tool_result_only_user_message() -> None:
    ui = _new_ui()
    ui.conversation_messages = [
        create_user_message("hello world"),
        create_assistant_message("ok"),
        create_user_message([{"type": "tool_result", "tool_use_id": "t1", "text": "done"}]),
    ]

    candidates = ui._build_history_candidates()

    assert len(candidates) == 1
    assert candidates[0]["index"] == 0
    assert "hello world" in candidates[0]["preview"]


def test_resolve_turn_end_index_includes_tool_results_until_next_real_user_message() -> None:
    ui = _new_ui()
    ui.conversation_messages = [
        create_user_message("turn1"),
        create_assistant_message("thinking"),
        create_user_message([{"type": "tool_result", "tool_use_id": "t1", "text": "ok"}]),
        create_user_message("turn2"),
    ]

    assert ui._resolve_turn_end_index(0) == 2


def test_rollback_to_index_truncates_and_replays_from_turn_boundary() -> None:
    ui = _new_ui()
    ui.conversation_messages = [
        create_user_message("turn1"),
        create_assistant_message("thinking"),
        create_user_message([{"type": "tool_result", "tool_use_id": "t1", "text": "ok"}]),
        create_user_message("turn2"),
    ]

    replay_calls: list[int] = []
    ui._fork_session = lambda: ("old-session", "new-session")  # type: ignore[assignment]
    ui.replay_conversation = lambda messages: replay_calls.append(len(messages))  # type: ignore[assignment]

    ui._rollback_to_index(0)

    assert len(ui.conversation_messages) == 3
    assert replay_calls == [3]


@pytest.mark.asyncio
async def test_check_and_compact_messages_warns_when_auto_compact_disabled(monkeypatch: Any) -> None:
    ui = _new_ui()
    dummy_console = _DummyConsole()
    monkeypatch.setattr("ripperdoc.cli.ui.rich_ui.session.console", dummy_console)

    class _Micro:
        was_compacted = False
        messages: list[Any] = []
        tokens_before = 0
        tokens_after = 0
        tokens_saved = 0
        tools_compacted = 0
        trigger_type = "none"

    class _Usage:
        is_above_warning = True
        should_auto_compact = False
        percent_used = 91.2
        total_tokens = 912
        max_context_tokens = 1000

    monkeypatch.setattr(
        "ripperdoc.cli.ui.rich_ui.session.micro_compact_messages",
        lambda messages, context_limit, auto_compact_enabled, protocol: _Micro(),
    )
    monkeypatch.setattr(
        "ripperdoc.cli.ui.rich_ui.session.estimate_used_tokens",
        lambda messages, protocol: 912,
    )
    monkeypatch.setattr(
        "ripperdoc.cli.ui.rich_ui.session.get_context_usage_status",
        lambda used, maximum, enabled: _Usage(),
    )

    messages = [create_user_message("hello")]
    out = await ui._check_and_compact_messages(messages, 1000, False, "openai")

    assert out == messages
    rendered = "\n".join(dummy_console.lines)
    assert "Context usage is 91.2%" in rendered
    assert "Auto-compaction is disabled" in rendered
