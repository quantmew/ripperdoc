"""Tests for CLI tool result renderers."""

from rich.console import Console

from ripperdoc.cli.ui.tool_renderers import BashResultRenderer, ToolResultRendererRegistry
from ripperdoc.utils.tasks import create_task


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


def test_task_graph_renderer_renders_panel_from_task_list_payload() -> None:
    """TaskList payloads should render in the same multiline style as Todo results."""
    console = Console(record=True, width=120)
    registry = ToolResultRendererRegistry(console, verbose=False)

    rendered = registry.render(
        "TaskList",
        "",
        {
            "tasks": [
                {
                    "id": "1",
                    "subject": "Fix parser error",
                    "status": "pending",
                    "owner": "alice",
                    "blockedBy": [],
                },
                {
                    "id": "2",
                    "subject": "Add regression tests",
                    "status": "in_progress",
                    "owner": None,
                    "blockedBy": ["1"],
                },
            ]
        },
    )

    output = console.export_text()
    assert rendered is True
    assert "Tasks updated (total 2; 1 pending, 1 in progress, 0 completed)." in output
    assert "○ Fix parser error @alice [1]" in output
    assert "◐ Add regression tests (blocked by 1) [2]" in output


def test_task_graph_renderer_loads_board_from_storage_for_task_update(
    tmp_path, monkeypatch
) -> None:
    """TaskCreate/TaskUpdate should auto-render board from persisted task state."""
    monkeypatch.setenv("RIPPERDOC_CONFIG_DIR", str(tmp_path / ".ripperdoc"))
    monkeypatch.chdir(tmp_path)

    create_task(
        subject="Investigate failing hook",
        description="Trace failing PreToolUse hook path",
        active_form="Investigating failing hook",
        status="pending",
    )

    console = Console(record=True, width=120)
    registry = ToolResultRendererRegistry(console, verbose=False)
    rendered = registry.render(
        "TaskUpdate",
        "Updated task '1' fields: status",
        {
            "success": True,
            "taskId": "1",
            "updatedFields": ["status"],
            "statusChange": {"from": "pending", "to": "in_progress"},
        },
    )

    output = console.export_text()
    assert rendered is True
    assert "Tasks updated (total 1; 1 pending, 0 in progress, 0 completed)." in output
    assert "○ Investigate failing hook [1]" in output
