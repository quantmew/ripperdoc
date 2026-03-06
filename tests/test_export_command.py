from pathlib import Path
import json

from rich.console import Console

from ripperdoc.cli.commands import get_slash_command
from ripperdoc.cli.commands.export_cmd import command as export_command
from ripperdoc.utils.messaging.messages import (
    create_assistant_message,
    create_plan_file_reference_attachment_message,
    create_plan_mode_attachment_message,
    create_user_message,
)


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
    assert "Conversation copied to clipboard" in output


def test_export_command_save_to_default_file_from_picker(tmp_path, monkeypatch):
    ui = _DummyUI(tmp_path)
    ui.conversation_messages = _sample_messages()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("ripperdoc.cli.commands.export_cmd._prompt_export_method", lambda: "file")
    monkeypatch.setattr(
        "ripperdoc.cli.commands.export_cmd._default_export_path",
        lambda _ui: tmp_path / "conversation-export.txt",
    )

    export_command.handler(ui, "")

    output = ui.console.export_text()
    assert "Conversation exported to: conversation-export.txt" in output
    exported = (tmp_path / "conversation-export.txt").read_text(encoding="utf-8")
    assert "User: hello" in exported
    assert "Assistant: world" in exported


def test_export_command_save_to_explicit_filename_arg(tmp_path, monkeypatch):
    ui = _DummyUI(tmp_path)
    ui.conversation_messages = _sample_messages()
    monkeypatch.chdir(tmp_path)

    export_command.handler(ui, "my-export")

    output = ui.console.export_text()
    assert "Conversation exported to: my-export.txt" in output
    assert (tmp_path / "my-export.txt").exists()


def test_export_command_filename_arg_bypasses_picker(tmp_path, monkeypatch):
    ui = _DummyUI(tmp_path)
    ui.conversation_messages = _sample_messages()
    monkeypatch.chdir(tmp_path)
    called = {"picker": False}

    def _mark_picker():
        called["picker"] = True
        return "clipboard"

    monkeypatch.setattr("ripperdoc.cli.commands.export_cmd._prompt_export_method", _mark_picker)

    export_command.handler(ui, "notes.md")

    output = ui.console.export_text()
    assert "Conversation exported to: notes.md" in output
    assert called["picker"] is False


def test_export_command_explicit_jsonl_arg_writes_machine_readable_export(tmp_path, monkeypatch):
    ui = _DummyUI(tmp_path)
    ui.conversation_messages = _sample_messages()
    monkeypatch.chdir(tmp_path)

    export_command.handler(ui, "conversation.jsonl")

    output = ui.console.export_text()
    assert "Conversation exported to: conversation.jsonl" in output
    exported_lines = (tmp_path / "conversation.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(exported_lines) == 2
    first_record = json.loads(exported_lines[0])
    second_record = json.loads(exported_lines[1])
    assert first_record["schema_version"] == 1
    assert first_record["record_type"] == "conversation_message"
    assert first_record["session_id"] == "abc12345-test-session"
    assert first_record["message"]["type"] == "user"
    assert second_record["schema_version"] == 1
    assert second_record["message"]["type"] == "assistant"


def test_export_command_picker_markdown_writes_markdown_export(tmp_path, monkeypatch):
    ui = _DummyUI(tmp_path)
    ui.conversation_messages = _sample_messages()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("ripperdoc.cli.commands.export_cmd._prompt_export_method", lambda: "file_md")
    monkeypatch.setattr(
        "ripperdoc.cli.commands.export_cmd._default_export_path",
        lambda _ui, extension=".txt": tmp_path / f"conversation-export{extension}",
    )

    export_command.handler(ui, "")

    output = ui.console.export_text()
    assert "Conversation exported to: conversation-export.md" in output
    exported = (tmp_path / "conversation-export.md").read_text(encoding="utf-8")
    assert "# Conversation Export" in exported
    assert "## Transcript" in exported
    assert "User: hello" in exported


def test_export_command_clipboard_failure(tmp_path, monkeypatch):
    ui = _DummyUI(tmp_path)
    ui.conversation_messages = _sample_messages()
    monkeypatch.setattr("ripperdoc.cli.commands.export_cmd._prompt_export_method", lambda: "clipboard")
    monkeypatch.setattr(
        "ripperdoc.cli.commands.export_cmd._copy_to_clipboard",
        lambda _text: (False, "Failed to copy to clipboard."),
    )

    export_command.handler(ui, "")

    output = ui.console.export_text()
    assert "Failed to copy to clipboard" in output


def test_export_command_includes_export_visible_attachments_but_hides_internal_ones(tmp_path, monkeypatch):
    ui = _DummyUI(tmp_path)
    ui.conversation_messages = [
        create_user_message("hello"),
        create_plan_mode_attachment_message(
            "Plan mode still active.",
            plan_file_path="/tmp/plan.md",
            reminder_type="sparse",
        ),
        create_plan_file_reference_attachment_message("/tmp/plan.md", "1. ship it"),
        create_assistant_message("world"),
    ]
    monkeypatch.chdir(tmp_path)

    export_command.handler(ui, "notes.txt")

    exported = (tmp_path / "notes.txt").read_text(encoding="utf-8")
    assert "User: hello" in exported
    assert "Assistant: world" in exported
    assert "System (plan_file_reference):" in exported
    assert "A plan file exists from plan mode at: /tmp/plan.md" in exported
    assert "Plan mode still active" not in exported
