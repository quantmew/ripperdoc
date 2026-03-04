from pathlib import Path

from rich.console import Console

from ripperdoc.cli.commands import get_slash_command
from ripperdoc.cli.commands.copy_cmd import command as copy_command
from ripperdoc.utils.messaging.messages import create_assistant_message, create_user_message


class _DummyUI:
    def __init__(self, project_path: Path):
        self.console = Console(record=True, width=120)
        self.project_path = project_path
        self.conversation_messages = []


def test_copy_command_registered():
    assert get_slash_command("copy") is copy_command


def test_copy_command_no_assistant_message(tmp_path):
    ui = _DummyUI(tmp_path)
    ui.conversation_messages = [create_user_message("hello")]
    copy_command.handler(ui, "")
    output = ui.console.export_text()
    assert "No assistant message to copy" in output


def test_copy_command_no_content(tmp_path):
    ui = _DummyUI(tmp_path)
    assistant = create_assistant_message([{"type": "text", "text": "x"}])
    assistant.message.content = []
    ui.conversation_messages = [assistant]

    copy_command.handler(ui, "")

    output = ui.console.export_text()
    assert "No content to copy" in output


def test_copy_command_no_text_content(tmp_path):
    ui = _DummyUI(tmp_path)
    ui.conversation_messages = [
        create_assistant_message([{"type": "tool_use", "id": "1", "name": "Bash", "input": {}}])
    ]

    copy_command.handler(ui, "")

    output = ui.console.export_text()
    assert "No text content to copy" in output


def test_copy_command_copies_last_assistant_response(tmp_path, monkeypatch):
    ui = _DummyUI(tmp_path)
    ui.conversation_messages = [
        create_user_message("hello"),
        create_assistant_message([{"type": "text", "text": "first"}, {"type": "text", "text": "second"}]),
    ]

    copied = {"text": ""}
    monkeypatch.setattr(
        "ripperdoc.cli.commands.copy_cmd._copy_to_clipboard",
        lambda text: (copied.update({"text": text}) or True, ""),
    )

    copy_command.handler(ui, "")

    output = ui.console.export_text()
    assert "Copied to clipboard" in output
    assert copied["text"] == "first\n\nsecond"


def test_copy_command_clipboard_failure(tmp_path, monkeypatch):
    ui = _DummyUI(tmp_path)
    ui.conversation_messages = [create_assistant_message("hello")]
    monkeypatch.setattr(
        "ripperdoc.cli.commands.copy_cmd._copy_to_clipboard",
        lambda _text: (False, "Failed to copy to clipboard."),
    )

    copy_command.handler(ui, "")

    output = ui.console.export_text()
    assert "Failed to copy to clipboard" in output
