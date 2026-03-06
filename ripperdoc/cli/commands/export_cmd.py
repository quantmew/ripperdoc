"""Export command - export current conversation to clipboard or file."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

from rich.markup import escape

from ripperdoc.cli.ui.choice import ChoiceOption, prompt_choice, theme_style
from ripperdoc.utils.clipboard import copy_to_clipboard
from ripperdoc.utils.messaging.message_formatting import render_transcript

from .base import SlashCommand

ExportFormat = Literal["txt", "md", "jsonl"]
JSONL_EXPORT_SCHEMA_VERSION = 1


def _build_export_content(ui: Any, messages: list[Any], export_format: ExportFormat) -> str:
    if export_format == "jsonl":
        lines = []
        for index, message in enumerate(messages):
            lines.append(
                json.dumps(
                    {
                        "schema_version": JSONL_EXPORT_SCHEMA_VERSION,
                        "record_type": "conversation_message",
                        "index": index,
                        "session_id": getattr(ui, "session_id", None),
                        "message": message.model_dump(mode="json") if hasattr(message, "model_dump") else str(message),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        return "\n".join(lines).rstrip() + "\n"

    transcript = render_transcript(
        messages,
        include_tool_details=True,
        attachment_mode="export",
    ).strip()

    if export_format == "md":
        exported_at = datetime.now().isoformat(timespec="seconds")
        header = [
            "# Conversation Export",
            "",
            f"- Session: `{getattr(ui, 'session_id', '')}`",
            f"- Exported: `{exported_at}`",
            "",
            "## Transcript",
            "",
        ]
        return "\n".join(header) + transcript.rstrip() + "\n"

    return transcript.rstrip() + "\n"


def _extract_first_prompt(messages: list[Any]) -> str:
    for message in messages:
        if getattr(message, "type", None) != "user":
            continue
        content = getattr(getattr(message, "message", None), "content", None)
        first_line = ""
        if isinstance(content, str):
            first_line = content.strip()
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text" and block.get("text"):
                        first_line = str(block.get("text", "")).strip()
                        break
                    continue
                if getattr(block, "type", None) == "text":
                    text = getattr(block, "text", None)
                    if text:
                        first_line = str(text).strip()
                        break

        first_line = (first_line.splitlines()[0] if first_line else "").strip()
        if len(first_line) > 50:
            first_line = first_line[:50] + "..."
        return first_line
    return ""


def _sanitize_filename(text: str) -> str:
    result = text.lower()
    result = re.sub(r"[^a-z0-9\s-]", "", result)
    result = re.sub(r"\s+", "-", result)
    result = re.sub(r"-+", "-", result)
    result = re.sub(r"^-|-$", "", result)
    return result


def _normalize_export_filename(raw_name: str, default_extension: str = ".txt") -> str:
    trimmed = raw_name.strip()
    lower_trimmed = trimmed.lower()
    if lower_trimmed.endswith((".txt", ".md", ".jsonl")):
        return trimmed
    return re.sub(r"\.[^.]+$", "", trimmed) + default_extension


def _default_export_path(ui: Any, extension: str = ".txt") -> Path:
    messages = list(getattr(ui, "conversation_messages", []) or [])
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    first_prompt = _extract_first_prompt(messages)
    if first_prompt:
        slug = _sanitize_filename(first_prompt)
        filename = f"{timestamp}-{slug}{extension}" if slug else f"conversation-{timestamp}{extension}"
    else:
        filename = f"conversation-{timestamp}{extension}"
    return Path.cwd() / filename


def _detect_export_format(path: str | Path) -> ExportFormat:
    suffix = Path(path).suffix.lower()
    if suffix == ".md":
        return "md"
    if suffix == ".jsonl":
        return "jsonl"
    return "txt"


def _resolve_export_path(ui: Any, raw_path: Optional[str], default_extension: str = ".txt") -> Path:
    if not raw_path or not raw_path.strip():
        return _default_export_path(ui, extension=default_extension)

    normalized = _normalize_export_filename(raw_path, default_extension=default_extension)
    candidate = Path(normalized).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate


def _copy_to_clipboard(text: str) -> tuple[bool, str]:
    return copy_to_clipboard(text)


def _save_to_file(path: Path, content: str) -> tuple[bool, str]:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return True, ""
    except (OSError, IOError) as exc:
        return False, str(exc)


def _prompt_export_method() -> str:
    options = [
        ChoiceOption(
            "clipboard",
            "<info>Copy to clipboard</info>  <dim>Copy the conversation to your system clipboard</dim>",
        ),
        ChoiceOption(
            "file_txt",
            "<info>Save text</info>          <dim>Save the conversation as plain text (.txt)</dim>",
        ),
        ChoiceOption(
            "file_md",
            "<info>Save markdown</info>      <dim>Save the conversation as Markdown (.md)</dim>",
        ),
        ChoiceOption(
            "file_jsonl",
            "<info>Save jsonl</info>         <dim>Save the conversation as machine-readable JSONL (.jsonl)</dim>",
        ),
    ]
    return str(
        prompt_choice(
            title="Export Conversation",
            message="<question>Select export method:</question>\n\n<dim>Esc to cancel</dim>",
            options=options,
            allow_esc=True,
            esc_value="__cancel__",
            style=theme_style(),
        )
    )


def _handle(ui: Any, arg: str) -> bool:
    messages = list(getattr(ui, "conversation_messages", []) or [])
    if not messages:
        ui.console.print("[yellow]No conversation to export yet.[/yellow]")
        return True

    file_arg = (arg or "").strip()
    export_format: ExportFormat = _detect_export_format(file_arg) if file_arg else "txt"
    content = _build_export_content(ui, messages, export_format)
    if not content.strip():
        ui.console.print("[yellow]No exportable conversation content found.[/yellow]")
        return True

    if file_arg:
        normalized_name = _normalize_export_filename(file_arg)
        path = _resolve_export_path(ui, file_arg)
        ok, error = _save_to_file(path, content)
        if ok:
            ui.console.print(f"[green]Conversation exported to: {escape(normalized_name)}[/green]")
        else:
            ui.console.print(f"[red]Failed to export conversation: {escape(error)}[/red]")
        return True

    action = _prompt_export_method()
    if action == "__cancel__":
        ui.console.print("[dim]Export cancelled.[/dim]")
        return True

    if action == "clipboard":
        clipboard_content = _build_export_content(ui, messages, "txt")
        ok, error = _copy_to_clipboard(clipboard_content)
        if ok:
            ui.console.print("[green]Conversation copied to clipboard[/green]")
        else:
            ui.console.print(f"[red]{escape(error)}[/red]")
        return True

    if action in {"file", "file_txt", "file_md", "file_jsonl"}:
        format_map: dict[str, tuple[ExportFormat, str]] = {
            "file": ("txt", ".txt"),
            "file_txt": ("txt", ".txt"),
            "file_md": ("md", ".md"),
            "file_jsonl": ("jsonl", ".jsonl"),
        }
        selected_format, extension = format_map[action]
        try:
            path = _default_export_path(ui, extension=extension)
        except TypeError:
            path = _default_export_path(ui)
        content = _build_export_content(ui, messages, selected_format)
        ok, error = _save_to_file(path, content)
        if ok:
            ui.console.print(f"[green]Conversation exported to: {escape(path.name)}[/green]")
        else:
            ui.console.print(f"[red]Failed to export conversation: {escape(error)}[/red]")
        return True

    ui.console.print("[red]Unknown export action.[/red]")
    return True


command = SlashCommand(
    name="export",
    description="Export the current conversation to a file or clipboard",
    handler=_handle,
)


__all__ = ["command"]
