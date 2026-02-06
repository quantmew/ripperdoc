"""Tests for rich UI rendering helpers."""

from typing import Any

from ripperdoc.cli.ui.rich_ui.rendering import (
    handle_progress_message,
    simplify_progress_suffix,
)
from ripperdoc.utils.messages import create_progress_message


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

    def display_message(self, *_args: Any, **_kwargs: Any) -> None:
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
