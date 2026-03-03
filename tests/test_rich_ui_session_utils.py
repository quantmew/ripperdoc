"""Focused utility tests for RichUI history/compaction helpers."""

from __future__ import annotations

import asyncio
import threading
from typing import Any, List

import pytest

from pathlib import Path

from ripperdoc.cli.ui.rich_ui.session import RichUI, _replace_surrogate_codepoints
from ripperdoc.utils.pending_messages import PendingMessageQueue
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
    ui.model = "main"
    ui.project_path = Path.cwd()
    ui.conversation_messages = []
    ui._prompt_session = None
    ui._rprompt_context_cache_key = None
    ui._rprompt_context_fragment = ("class:rprompt-off", "Ctx --")
    ui._saved_conversation = None
    ui._pre_plan_mode = None
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


def test_replace_surrogate_codepoints_replaces_invalid_chars() -> None:
    raw = "A\ud800B\udfffC"
    assert _replace_surrogate_codepoints(raw) == "A�B�C"


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


def test_cycle_permission_mode_rotates_with_bypass_when_available(monkeypatch: Any) -> None:
    ui = _new_ui()
    ui.permission_mode = "default"
    ui.yolo_mode = False
    ui._permission_checker = object()
    ui._session_additional_working_dirs = set()
    ui.project_path = Path.cwd()
    ui.query_context = type(
        "Q",
        (),
        {
            "permission_mode": "default",
            "yolo_mode": False,
        },
    )()

    seen: list[str] = []
    monkeypatch.setattr(
        "ripperdoc.cli.ui.rich_ui.session.hook_manager.set_permission_mode",
        lambda mode: seen.append(mode),
    )

    ui._cycle_permission_mode()
    assert ui.permission_mode == "acceptEdits"
    assert ui.query_context.permission_mode == "acceptEdits"
    assert ui.query_context.yolo_mode is False

    ui._cycle_permission_mode()
    assert ui.permission_mode == "plan"
    assert ui.query_context.permission_mode == "plan"

    ui._cycle_permission_mode()
    assert ui.permission_mode == "bypassPermissions"
    assert ui.query_context.permission_mode == "bypassPermissions"
    assert ui.query_context.yolo_mode is True

    ui._cycle_permission_mode()
    assert ui.permission_mode == "default"
    assert ui.query_context.permission_mode == "default"
    assert ui.query_context.yolo_mode is False
    assert seen == ["acceptEdits", "plan", "bypassPermissions", "default"]
    assert ui.console.lines == []


def test_cycle_permission_mode_skips_bypass_when_unavailable(monkeypatch: Any) -> None:
    ui = _new_ui()
    ui.permission_mode = "default"
    ui.yolo_mode = False
    ui._permission_checker = object()
    ui._session_additional_working_dirs = set()
    ui.project_path = Path.cwd()
    ui.query_context = type(
        "Q",
        (),
        {
            "permission_mode": "default",
            "yolo_mode": False,
        },
    )()
    monkeypatch.setattr(RichUI, "_is_bypass_permissions_mode_available", lambda self: False)
    monkeypatch.setattr(
        "ripperdoc.cli.ui.rich_ui.session.hook_manager.set_permission_mode",
        lambda mode: None,
    )

    ui._cycle_permission_mode()
    assert ui.permission_mode == "acceptEdits"
    ui._cycle_permission_mode()
    assert ui.permission_mode == "plan"
    ui._cycle_permission_mode()
    assert ui.permission_mode == "default"


def test_apply_permission_mode_recreates_checker_when_leaving_bypass(monkeypatch: Any) -> None:
    ui = _new_ui()
    ui.permission_mode = "bypassPermissions"
    ui.yolo_mode = True
    ui._permission_checker = None
    ui._session_additional_working_dirs = set()
    ui.project_path = Path.cwd()
    ui.query_context = type(
        "Q",
        (),
        {
            "permission_mode": "bypassPermissions",
            "yolo_mode": True,
        },
    )()

    marker = object()
    monkeypatch.setattr(
        "ripperdoc.cli.ui.rich_ui.session.make_permission_checker",
        lambda *args, **kwargs: marker,
    )
    monkeypatch.setattr(
        "ripperdoc.cli.ui.rich_ui.session.hook_manager.set_permission_mode",
        lambda mode: None,
    )

    ui._apply_permission_mode("default", announce=False)

    assert ui.permission_mode == "default"
    assert ui.yolo_mode is False
    assert ui.query_context.permission_mode == "default"
    assert ui.query_context.yolo_mode is False
    assert ui._permission_checker is marker


def test_get_rprompt_keeps_permission_mode_label_when_thinking_unsupported(monkeypatch: Any) -> None:
    ui = _new_ui()
    ui.permission_mode = "plan"
    ui._thinking_mode_enabled = False
    monkeypatch.setattr(RichUI, "_supports_thinking_mode", lambda self: False)

    fragments = list(ui._get_rprompt())
    assert fragments == [("class:rprompt-mode-plan", "⏸ plan mode on")]
    assert ("class:rprompt-on", "⚡ Thinking") not in fragments


def test_get_bottom_toolbar_includes_context_percent(monkeypatch: Any) -> None:
    ui = _new_ui()
    ui.permission_mode = "default"
    ui._thinking_mode_enabled = False
    ui.model = "main"
    ui.project_path = Path.cwd()
    ui.conversation_messages = [create_user_message("hello")]

    monkeypatch.setattr(RichUI, "_supports_thinking_mode", lambda self: False)
    monkeypatch.setattr(
        "ripperdoc.cli.ui.rich_ui.session.estimate_used_tokens",
        lambda messages, protocol: 123,  # noqa: ARG005
    )
    monkeypatch.setattr(
        RichUI,
        "_resolve_query_runtime_settings",
        lambda self: (100_000, False, "openai"),
    )

    class _Usage:
        percent_used = 12.34
        should_auto_compact = False
        is_above_warning = False

    monkeypatch.setattr(
        "ripperdoc.cli.ui.rich_ui.session.get_context_usage_status",
        lambda used_tokens, max_context_tokens, auto_compact_enabled: _Usage(),  # noqa: ARG005
    )

    fragments = list(ui._get_bottom_toolbar())
    assert fragments == [("class:rprompt-ctx-ok", "Ctx 12.3%")]


def test_get_prompt_placeholder_returns_dimmed_suggestion() -> None:
    ui = _new_ui()
    ui._prompt_suggestion_lock = threading.Lock()
    ui._prompt_suggestion_text = "run the tests"

    placeholder = ui._get_prompt_placeholder()

    assert list(placeholder) == [("class:prompt-suggestion", "run the tests")]


def test_prompt_suggestion_skip_reason_detects_cache_cold(monkeypatch: Any) -> None:
    ui = _new_ui()
    ui._prompt_suggestion_enabled = True
    ui.permission_mode = "default"
    monkeypatch.setattr(
        "ripperdoc.cli.ui.rich_ui.session.sys.stdout",
        type("StdoutStub", (), {"isatty": lambda self: True})(),
    )
    first = create_assistant_message("initial response", input_tokens=10)
    cache_cold = create_assistant_message(
        "next response",
        input_tokens=100,
        cache_creation_tokens=120,
    )

    reason = ui._prompt_suggestion_skip_reason([first, cache_cold])

    assert reason == "cache_cold"


@pytest.mark.asyncio
async def test_on_exit_plan_mode_shift_tab_cycles_confirmation_mode(monkeypatch: Any) -> None:
    ui = _new_ui()
    ui.permission_mode = "plan"
    ui._pre_plan_mode = "default"
    ui._clear_context_after_turn = False
    ui._session_additional_working_dirs = set()
    ui.project_path = Path.cwd()
    ui.query_context = type("Q", (), {"permission_mode": "plan", "yolo_mode": False})()

    answers = iter(["__cycle_mode__", "yes_keep"])
    async def _fake_prompt_choice_async(*args: Any, **kwargs: Any) -> str:  # noqa: ARG001
        return next(answers)
    monkeypatch.setattr(
        "ripperdoc.cli.ui.rich_ui.session.prompt_choice_async",
        _fake_prompt_choice_async,
    )
    monkeypatch.setattr(RichUI, "_is_bypass_permissions_mode_available", lambda self: True)
    monkeypatch.setattr(
        "ripperdoc.cli.ui.rich_ui.session.hook_manager.set_permission_mode",
        lambda mode: None,
    )

    decision = await ui._on_exit_plan_mode(plan="plan text", is_agent=False)

    assert decision["approved"] is True
    assert decision["permission_mode"] == "acceptEdits"
    assert decision["clear_context"] is False
    assert ui.permission_mode == "acceptEdits"
    assert ui._clear_context_after_turn is False


@pytest.mark.asyncio
async def test_finalize_query_stream_clears_context_after_plan_exit() -> None:
    ui = _new_ui()
    ui._clear_context_after_turn = True
    ui._query_in_progress = True
    ui._active_spinner = object()
    ui._stop_interrupt_listener = lambda: None  # type: ignore[assignment]
    ui.project_path = Path.cwd()
    ui.session_id = "test-session"

    class _DummySpinner:
        def stop(self) -> None:
            return None

    messages = [
        create_user_message("turn1"),
        create_assistant_message("a1"),
        create_user_message("turn2"),
        create_assistant_message("a2"),
        create_user_message("turn3"),
        create_assistant_message("a3"),
    ]

    ui._finalize_query_stream(_DummySpinner(), messages)

    assert len(ui.conversation_messages) == 4
    assert ui._clear_context_after_turn is False
    assert ui._query_in_progress is False
    assert ui._active_spinner is None


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


@pytest.mark.asyncio
async def test_schedule_background_notification_response_triggers_process_query() -> None:
    ui = _new_ui()
    ui._loop = asyncio.get_running_loop()
    ui._background_notification_tasks = set()
    redraw_calls: list[str] = []
    ui._prompt_session = type(
        "PromptSessionStub",
        (),
        {"app": type("AppStub", (), {"invalidate": lambda self: redraw_calls.append("invalidate")})()},
    )()

    calls: list[tuple[str, dict[str, Any] | None, bool]] = []

    async def _fake_process_query(
        user_input: str,
        *,
        user_message_metadata: dict[str, Any] | None = None,
        append_prompt_history: bool = True,
    ) -> None:
        calls.append((user_input, user_message_metadata, append_prompt_history))

    ui.process_query = _fake_process_query  # type: ignore[assignment]

    ui._schedule_background_notification_response(
        agent_message="background finished",
        metadata={"notification_type": "task_notification", "task_id": "bash_1"},
    )

    await asyncio.sleep(0)
    tasks = list(ui._background_notification_tasks)
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    assert calls == [
        (
            "background finished",
            {"notification_type": "task_notification", "task_id": "bash_1"},
            False,
        )
    ]
    assert redraw_calls == ["invalidate"]


def test_finalize_query_stream_flushes_deferred_task_notifications() -> None:
    ui = _new_ui()
    ui._clear_context_after_turn = False
    ui._query_in_progress = True
    ui._active_spinner = object()
    ui._stop_interrupt_listener = lambda: None  # type: ignore[assignment]
    ui.project_path = Path.cwd()
    ui.session_id = "test-session"

    queue = PendingMessageQueue()
    queue.enqueue_text(
        "bash_2 [completed] done",
        metadata={
            "notification_type": "task_notification",
            "task_id": "bash_2",
            "status": "completed",
        },
    )
    queue.enqueue_text("keep me", metadata={"source": "manual"})
    ui.query_context = type("Q", (), {"pending_message_queue": queue})()

    scheduled: list[tuple[str, dict[str, Any]]] = []
    ui._schedule_background_notification_response = (  # type: ignore[assignment]
        lambda *, agent_message, metadata: scheduled.append((agent_message, dict(metadata)))
    )

    class _DummySpinner:
        def stop(self) -> None:
            return None

    messages = [create_user_message("turn")]
    ui._finalize_query_stream(_DummySpinner(), messages)

    assert len(scheduled) == 1
    assert scheduled[0][0] == "bash_2 [completed] done"
    assert scheduled[0][1].get("notification_type") == "task_notification"

    remaining = queue.drain()
    assert len(remaining) == 1
    remaining_content = getattr(getattr(remaining[0], "message", None), "content", None)
    assert remaining_content == "keep me"
