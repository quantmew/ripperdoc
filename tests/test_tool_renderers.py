"""Tests for CLI tool result renderers."""

from rich.console import Console

from ripperdoc.cli.ui.tool_renderers import BashResultRenderer


def test_bash_renderer_hides_empty_stdout_and_stderr_sections() -> None:
    """When both streams are empty, do not render stdout/stderr placeholders."""
    console = Console(record=True, width=120)
    renderer = BashResultRenderer(console, verbose=False)

    tool_data = {
        "exit_code": 0,
        "stdout": "",
        "stderr": "",
        "duration_ms": 12.0,
        "timeout_ms": 120000,
    }
    renderer.render("", tool_data)

    output = console.export_text()
    assert "Exit code 0" in output
    assert "stdout:" not in output
    assert "stderr:" not in output
    assert "(no stdout)" not in output
    assert "(no stderr)" not in output


def test_bash_renderer_shows_only_present_stream_sections() -> None:
    """Render stream labels only for non-empty streams."""
    console = Console(record=True, width=120)
    renderer = BashResultRenderer(console, verbose=True)

    tool_data = {
        "exit_code": 0,
        "stdout": "line 1\nline 2",
        "stderr": "",
        "duration_ms": 10.0,
        "timeout_ms": 120000,
    }
    renderer.render("", tool_data)

    output = console.export_text()
    assert "stdout:" in output
    assert "line 1" in output
    assert "stderr:" not in output
