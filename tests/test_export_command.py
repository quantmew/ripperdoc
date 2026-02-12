from pathlib import Path

from rich.console import Console

from ripperdoc.cli.commands import get_slash_command
from ripperdoc.cli.commands.export_cmd import command as export_command
from ripperdoc.utils.messages import create_assistant_message, create_user_message


class _DummyUI:
    def __init__(self, project_path: Path):
        self.console = Console(record=True, width=120)
        self.project_path = project_path
        self.session_id = "abc12345-test-session"
        self.conversation_messages = []


def _sample_messages():
    return [
        create_user_message("hello"),
        create_assistant_message("world"),
    ]


def test_export_command_registered():
    assert get_slash_command("export") is export_command


def test_export_command_empty_conversation(tmp_path):
    ui = _DummyUI(tmp_path)
    export_command.handler(ui, "")
    output = ui.console.export_text()
    assert "No conversation to export yet" in output


def test_export_command_cancelled(tmp_path, monkeypatch):
    ui = _DummyUI(tmp_path)
    ui.conversation_messages = _sample_messages()
    monkeypatch.setattr("ripperdoc.cli.commands.export_cmd._prompt_export_method", lambda: "__cancel__")

    export_command.handler(ui, "")

    output = ui.console.export_text()
    assert "Export cancelled" in output


def test_export_command_copy_to_clipboard_via_picker(tmp_path, monkeypatch):
    ui = _DummyUI(tmp_path)
    ui.conversation_messages = _sample_messages()
    monkeypatch.setattr("ripperdoc.cli.commands.export_cmd._prompt_export_method", lambda: "clipboard")
    monkeypatch.setattr(
        "ripperdoc.cli.commands.export_cmd._copy_to_clipboard",
        lambda text: (("User: hello" in text and "Assistant: world" in text), ""),
    )

    export_command.handler(ui, "")

    output = ui.console.export_text()
    assert "Copied conversation to clipboard" in output


def test_export_command_save_to_default_file(tmp_path, monkeypatch):
    ui = _DummyUI(tmp_path)
    ui.conversation_messages = _sample_messages()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("ripperdoc.cli.commands.export_cmd._prompt_export_method", lambda: "file")
    monkeypatch.setattr(
        "ripperdoc.cli.commands.export_cmd._default_export_path",
        lambda _ui: tmp_path / "conversation-export.md",
    )

    export_command.handler(ui, "")

    output = ui.console.export_text()
    assert "Saved conversation to" in output
    exported = (tmp_path / "conversation-export.md").read_text(encoding="utf-8")
    assert "User: hello" in exported
    assert "Assistant: world" in exported


def test_export_command_save_to_explicit_file_arg(tmp_path, monkeypatch):
    ui = _DummyUI(tmp_path)
    ui.conversation_messages = _sample_messages()
    monkeypatch.chdir(tmp_path)

    export_command.handler(ui, "file my-export.txt")

    output = ui.console.export_text()
    assert "Saved conversation to" in output
    assert (tmp_path / "my-export.txt").exists()


def test_export_command_clipboard_arg_bypasses_picker(tmp_path, monkeypatch):
    ui = _DummyUI(tmp_path)
    ui.conversation_messages = _sample_messages()
    called = {"picker": False}

    def _mark_picker():
        called["picker"] = True
        return "file"

    monkeypatch.setattr("ripperdoc.cli.commands.export_cmd._prompt_export_method", _mark_picker)
    monkeypatch.setattr("ripperdoc.cli.commands.export_cmd._copy_to_clipboard", lambda _text: (True, ""))

    export_command.handler(ui, "clipboard")

    output = ui.console.export_text()
    assert "Copied conversation to clipboard" in output
    assert called["picker"] is False


def test_export_command_invalid_args(tmp_path):
    ui = _DummyUI(tmp_path)
    ui.conversation_messages = _sample_messages()

    export_command.handler(ui, "wat")

    output = ui.console.export_text()
    assert "Usage: /export" in output
