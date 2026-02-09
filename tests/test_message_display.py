"""Tests for MessageDisplay tool argument formatting."""

from rich.console import Console

from ripperdoc.cli.ui.message_display import MessageDisplay


def test_file_tools_show_full_file_path_without_prefix() -> None:
    """Read/Write/Edit/MultiEdit should render raw file paths without key prefix."""
    display = MessageDisplay(Console(record=True, width=240))
    long_path = "/data2/wangjun/github/research_knight/baselines/qa/some/really/long/path/that/should/not/be/truncated/for/file_tools.py"

    for tool_name in ("Read", "Write", "Edit", "MultiEdit"):
        args = display.format_tool_args(tool_name, {"file_path": long_path})
        assert args == [long_path]


def test_non_file_tool_still_keeps_key_value_format() -> None:
    """Tools outside file ops should keep regular key/value argument format."""
    display = MessageDisplay(Console(record=True, width=240))

    args = display.format_tool_args("Task", {"description": "Investigate issue"})

    assert args == ["description: Investigate issue"]


def test_print_tool_call_renders_file_path_without_file_path_prefix() -> None:
    """Tool call output should not include `file_path:` for file tools."""
    console = Console(record=True, width=320)
    display = MessageDisplay(console)
    long_path = "/data2/wangjun/github/research_knight/baselines/qa/some/really/long/path/that/should/not/be/truncated/for/file_tools.py"

    display.print_tool_call("Read", "", {"file_path": long_path, "offset": 10, "limit": 20})

    output = console.export_text()
    assert f"Read({long_path}, offset: 10, limit: 20)" in output
    assert "file_path:" not in output
