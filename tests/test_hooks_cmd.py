"""Tests for the /hooks guided editor commands."""

import json
from pathlib import Path

from rich.console import Console

from ripperdoc.cli.commands.hooks_cmd import command as hooks_command


class _DummyUI:
    """Minimal UI stub for slash command handlers."""

    def __init__(self, console: Console, project_path: Path):
        self.console = console
        self.project_path = project_path


def _make_console_with_inputs(inputs):
    """Create a recording console with a predefined input sequence."""
    console = Console(record=True, width=120)
    answers = iter(inputs)
    console.input = lambda prompt="": next(answers)  # type: ignore[attr-defined]
    return console


def test_hooks_add_creates_local_file(tmp_path, monkeypatch):
    """Add should create a new hook entry with guided defaults."""
    monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)

    console = _make_console_with_inputs(
        [
            "",  # default location: local
            "",  # default event: PreToolUse
            "",  # matcher: match all
            "",  # hook type: command
            "echo hello",  # command
            "15",  # timeout
        ]
    )
    ui = _DummyUI(console, tmp_path)

    hooks_command.handler(ui, "add")

    hooks_path = tmp_path / ".ripperdoc" / "hooks.local.json"
    data = json.loads(hooks_path.read_text())
    pre_hooks = data["hooks"]["PreToolUse"][0]["hooks"]
    assert pre_hooks[0]["command"] == "echo hello"
    assert pre_hooks[0]["timeout"] == 15


def test_hooks_edit_updates_existing_entry(tmp_path, monkeypatch):
    """Edit should update an existing hook entry."""
    monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)

    hooks_path = tmp_path / ".ripperdoc" / "hooks.local.json"
    hooks_path.parent.mkdir(parents=True, exist_ok=True)
    hooks_path.write_text(
        json.dumps(
            {
                "hooks": {
                    "PreToolUse": [
                        {
                            "matcher": "*",
                            "hooks": [{"type": "command", "command": "echo before", "timeout": 20}],
                        }
                    ]
                }
            },
            indent=2,
        )
    )

    console = _make_console_with_inputs(
        [
            "",  # default location
            "1",  # event: PreToolUse
            "1",  # matcher: first
            "1",  # hook index
            "",  # hook type: keep command
            "echo after",  # new command
            "25",  # new timeout
        ]
    )
    ui = _DummyUI(console, tmp_path)

    hooks_command.handler(ui, "edit")

    data = json.loads(hooks_path.read_text())
    hook = data["hooks"]["PreToolUse"][0]["hooks"][0]
    assert hook["command"] == "echo after"
    assert hook["timeout"] == 25


def test_hooks_delete_removes_hook(tmp_path, monkeypatch):
    """Delete should remove the selected hook and keep others intact."""
    monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)

    hooks_path = tmp_path / ".ripperdoc" / "hooks.local.json"
    hooks_path.parent.mkdir(parents=True, exist_ok=True)
    hooks_path.write_text(
        json.dumps(
            {
                "hooks": {
                    "PreToolUse": [
                        {
                            "matcher": "*",
                            "hooks": [
                                {"type": "command", "command": "echo keep", "timeout": 10},
                                {"type": "command", "command": "echo remove", "timeout": 10},
                            ],
                        }
                    ]
                }
            },
            indent=2,
        )
    )

    console = _make_console_with_inputs(
        [
            "",  # default location
            "1",  # event: PreToolUse
            "1",  # matcher
            "2",  # hook to delete
            "",  # confirm delete
        ]
    )
    ui = _DummyUI(console, tmp_path)

    hooks_command.handler(ui, "delete")

    data = json.loads(hooks_path.read_text())
    hooks_list = data["hooks"]["PreToolUse"][0]["hooks"]
    assert len(hooks_list) == 1
    assert hooks_list[0]["command"] == "echo keep"
