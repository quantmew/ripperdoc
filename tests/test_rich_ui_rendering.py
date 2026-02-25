"""Tests for rich UI rendering helpers."""

from typing import Any

from ripperdoc.cli.ui.rich_ui.rendering import (
    handle_progress_message,
    simplify_progress_suffix,
)
from ripperdoc.utils.messages import create_assistant_message, create_progress_message


def test_simplify_progress_suffix_strips_stdout_prefix() -> None:
    """Spinner text should not include the raw stdout label."""
    assert simplify_progress_suffix("stdout: hello world") == "Working... hello world"


def test_simplify_progress_suffix_strips_stderr_prefix() -> None:
    """Spinner text should not include the raw stderr label."""
    assert simplify_progress_suffix("stderr: warning") == "Working... warning"


def test_simplify_progress_suffix_preserves_running_status_line() -> None:
    """Running timer line should be kept concise for spinner updates."""
    content = "Running... (3s)\nstdout: line 1\nstdout: line 2"
    assert simplify_progress_suffix(content) == "Running... (3s)"


class _DummySpinner:
    def __init__(self) -> None:
        self.calls: list[tuple[int, Any]] = []

    def update_tokens(self, out_tokens: int, suffix: Any = None) -> None:
        self.calls.append((out_tokens, suffix))

    def paused(self):  # pragma: no cover - this test path does not enter pause context
        class _NoopCtx:
            def __enter__(self) -> None:
                return None

            def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
                return False

        return _NoopCtx()


class _DummyUI:
    verbose = False

    def __init__(self) -> None:
        self.console = object()
        self.display_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def display_message(self, *args: Any, **kwargs: Any) -> None:
        self.display_calls.append((args, kwargs))
        return None


def test_handle_progress_message_does_not_set_spinner_suffix_for_tool_progress() -> None:
    """Non-stream tool progress should not be rendered into the spinner text."""
    message = create_progress_message(
        tool_use_id="tool1",
        sibling_tool_use_ids=set(),
        content="stdout: hello",
    )
    spinner = _DummySpinner()
    ui = _DummyUI()

    out_tokens = handle_progress_message(ui, message, spinner, output_token_est=42)

    assert out_tokens == 42
    assert spinner.calls == [(42, None)]


def test_handle_progress_message_renders_labeled_subagent_progress_payload() -> None:
    """Structured sender field should route progress to tool output."""
    message = create_progress_message(
        tool_use_id="tool1",
        sibling_tool_use_ids=set(),
        content="requesting Glob - pattern=*",
        progress_sender="Subagent(writer:agent_abcd1234)",
    )
    spinner = _DummySpinner()
    ui = _DummyUI()

    out_tokens = handle_progress_message(ui, message, spinner, output_token_est=0)

    assert out_tokens == 0
    assert spinner.calls == [(0, None)]
    assert len(ui.display_calls) == 1
    args, kwargs = ui.display_calls[0]
    assert args[0] == "Subagent(writer:agent_abcd1234)"
    assert args[1] == "requesting Glob - pattern=*"
    assert kwargs.get("is_tool") is True


def test_handle_progress_message_ignores_plain_progress_when_not_verbose() -> None:
    """Without typed sender and without verbose mode, plain progress is not printed."""
    message = create_progress_message(
        tool_use_id="tool1",
        sibling_tool_use_ids=set(),
        content="Subagent[reviewer:agent_00112233]: planning next step",
    )
    spinner = _DummySpinner()
    ui = _DummyUI()

    out_tokens = handle_progress_message(ui, message, spinner, output_token_est=7)

    assert out_tokens == 7
    assert spinner.calls == [(7, None)]
    assert ui.display_calls == []


def test_handle_progress_message_renders_structured_subagent_assistant_message() -> None:
    """Structured forwarded subagent assistant content should be summarized for display."""
    subagent_assistant = create_assistant_message(
        [{"type": "tool_use", "id": "t1", "name": "Glob", "input": {"pattern": "*"}}]
    )
    message = create_progress_message(
        tool_use_id="tool1",
        sibling_tool_use_ids=set(),
        content=subagent_assistant,
        progress_sender="Subagent(writer:agent_abcd1234)",
        is_subagent_message=True,
    )
    spinner = _DummySpinner()
    ui = _DummyUI()

    out_tokens = handle_progress_message(ui, message, spinner, output_token_est=3)

    assert out_tokens == 3
    assert spinner.calls == [(3, None)]
    assert len(ui.display_calls) == 1
    args, kwargs = ui.display_calls[0]
    assert args[0] == "Subagent(writer:agent_abcd1234)"
    assert args[1] == "requesting Glob"
    assert kwargs.get("is_tool") is True
