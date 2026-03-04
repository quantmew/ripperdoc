"""Export command - export current conversation to clipboard or file."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from rich.markup import escape

from ripperdoc.cli.ui.choice import ChoiceOption, prompt_choice, theme_style
from ripperdoc.utils.clipboard import copy_to_clipboard
from ripperdoc.utils.messaging.message_formatting import render_transcript

from .base import SlashCommand


def _build_export_content(ui: Any, transcript: str) -> str:
    _ = ui
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


def _normalize_export_filename(raw_name: str) -> str:
    trimmed = raw_name.strip()
    if trimmed.endswith(".txt"):
        return trimmed
    return re.sub(r"\.[^.]+$", "", trimmed) + ".txt"


def _default_export_path(ui: Any) -> Path:
    messages = list(getattr(ui, "conversation_messages", []) or [])
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    first_prompt = _extract_first_prompt(messages)
    if first_prompt:
        slug = _sanitize_filename(first_prompt)
        filename = f"{timestamp}-{slug}.txt" if slug else f"conversation-{timestamp}.txt"
    else:
        filename = f"conversation-{timestamp}.txt"
    return Path.cwd() / filename


def _resolve_export_path(ui: Any, raw_path: Optional[str]) -> Path:
    if not raw_path or not raw_path.strip():
        return _default_export_path(ui)

    normalized = _normalize_export_filename(raw_path)
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
            "file",
            "<info>Save to file</info>       <dim>Save the conversation to a file in the current directory</dim>",
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

    transcript = render_transcript(messages, include_tool_details=True).strip()
    if not transcript:
        ui.console.print("[yellow]No exportable conversation content found.[/yellow]")
        return True

    content = _build_export_content(ui, transcript)
    file_arg = (arg or "").strip()

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
        ok, error = _copy_to_clipboard(content)
        if ok:
            ui.console.print("[green]Conversation copied to clipboard[/green]")
        else:
            ui.console.print(f"[red]{escape(error)}[/red]")
        return True

    if action == "file":
        path = _default_export_path(ui)
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
