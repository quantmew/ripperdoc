"""Tests for prompt suggestion helpers and key handling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ripperdoc.cli.ui.rich_ui.input import build_prompt_session
from ripperdoc.cli.ui.rich_ui.prompt_suggestions import (
    build_suggestion_transcript,
    is_cache_cold,
    normalize_generated_suggestion,
    resolve_prompt_suggestion_enabled,
)
from ripperdoc.utils.messages import create_assistant_message, create_user_message


class _DummyUI:
    def __init__(self, suggestion: str | None = None) -> None:
        self.project_path = Path.cwd()
        self.session_id = "test-session"
        self._using_tty_input = False
        self._suggestion = suggestion
        self.toggle_count = 0
        self.cycle_count = 0
        self.insert_events: list[str] = []

    def _get_rprompt(self) -> str:
        return ""

    def _get_bottom_toolbar(self) -> str:
        return ""

    def _peek_prompt_suggestion(self) -> str | None:
        return self._suggestion

    def _take_prompt_suggestion(self) -> str | None:
        value = self._suggestion
        self._suggestion = None
        return value

    def _toggle_thinking_mode(self) -> None:
        self.toggle_count += 1

    def _cycle_permission_mode(self) -> None:
        self.cycle_count += 1

    def _on_prompt_text_insert(self, text: str) -> None:
        self.insert_events.append(text)


class _FakeBuffer:
    def __init__(self, text: str = "") -> None:
        self.text = text
        self.cursor_position = len(text)
        self.complete_state = None
        self.validate_calls = 0

    def validate_and_handle(self) -> None:
        self.validate_calls += 1

    def apply_completion(self, completion: Any) -> None:  # pragma: no cover - defensive
        return None

    def start_completion(self, select_first: bool = False) -> None:  # pragma: no cover - defensive
        return None


def _binding_handler(session: Any, keys: tuple[str, ...]) -> Any:
    for binding in session.key_bindings.bindings:
        if tuple(str(key) for key in binding.keys) == keys:
            return binding.handler
    raise AssertionError(f"binding not found for {keys}")


def test_resolve_prompt_suggestion_enabled_env_overrides() -> None:
    assert (
        resolve_prompt_suggestion_enabled(env_value="false", config_enabled=True, interactive=True)
        is False
    )
    assert (
        resolve_prompt_suggestion_enabled(env_value="1", config_enabled=False, interactive=False)
        is True
    )
    assert (
        resolve_prompt_suggestion_enabled(env_value=None, config_enabled=None, interactive=True) is True
    )
    assert (
        resolve_prompt_suggestion_enabled(env_value=None, config_enabled=None, interactive=False)
        is False
    )


def test_is_cache_cold_when_cache_creation_dominates() -> None:
    msg = create_assistant_message(
        "ok",
        input_tokens=100,
        cache_read_tokens=0,
        cache_creation_tokens=120,
    )
    assert is_cache_cold(msg) is True


def test_normalize_generated_suggestion_filters_assistant_voice() -> None:
    assert normalize_generated_suggestion("I'll do that next") is None
    assert normalize_generated_suggestion('"run the tests"') == "run the tests"


def test_build_suggestion_transcript_skips_tool_result_only_user_message() -> None:
    messages = [
        create_user_message("please fix the parser"),
        create_assistant_message("fixed parser, tests pass"),
        create_user_message([{"type": "tool_result", "tool_use_id": "x1", "text": "ok"}]),
    ]
    transcript = build_suggestion_transcript(messages)
    assert "User: please fix the parser" in transcript
    assert "[Tool result]" not in transcript


def test_tab_accepts_prompt_suggestion(monkeypatch: Any) -> None:
    monkeypatch.setattr("prompt_toolkit.application.run_in_terminal", lambda fn: fn())
    ui = _DummyUI("run the tests")
    session = build_prompt_session(ui, ignore_filter=None)
    tab_handler = _binding_handler(session, ("Keys.ControlI",))
    buf = _FakeBuffer("")
    event = type("Evt", (), {"current_buffer": buf})()

    tab_handler(event)

    assert buf.text == "run the tests"
    assert buf.cursor_position == len("run the tests")
    assert ui.toggle_count == 0


def test_enter_accepts_prompt_suggestion_and_submits() -> None:
    ui = _DummyUI("commit this")
    session = build_prompt_session(ui, ignore_filter=None)
    enter_handler = _binding_handler(session, ("Keys.ControlM",))
    buf = _FakeBuffer("")
    event = type("Evt", (), {"current_buffer": buf})()

    enter_handler(event)

    assert buf.text == "commit this"
    assert buf.validate_calls == 1


def test_tab_without_suggestion_toggles_thinking(monkeypatch: Any) -> None:
    monkeypatch.setattr("prompt_toolkit.application.run_in_terminal", lambda fn: fn())
    ui = _DummyUI(None)
    session = build_prompt_session(ui, ignore_filter=None)
    tab_handler = _binding_handler(session, ("Keys.ControlI",))
    buf = _FakeBuffer("")
    event = type("Evt", (), {"current_buffer": buf})()

    tab_handler(event)

    assert ui.toggle_count == 1


def test_build_prompt_session_wires_text_insert_callback() -> None:
    ui = _DummyUI(None)
    session = build_prompt_session(ui, ignore_filter=None)
    session.default_buffer.on_text_insert.sender = type("Buf", (), {"text": "a"})()
    session.default_buffer.on_text_insert.fire()

    assert ui.insert_events
    assert ui.insert_events[-1] == "a"
