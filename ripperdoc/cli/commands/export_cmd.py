"""Export command - export current conversation to clipboard or file."""

from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from rich.markup import escape

from ripperdoc.cli.ui.choice import ChoiceOption, prompt_choice, theme_style
from ripperdoc.utils.message_formatting import render_transcript

from .base import SlashCommand


def _build_export_content(ui: Any, transcript: str) -> str:
    session_id = str(getattr(ui, "session_id", "unknown"))
    project_path = Path(getattr(ui, "project_path", Path.cwd())).resolve()
    exported_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return (
        "# Ripperdoc Conversation Export\n\n"
        f"- Session: {session_id}\n"
        f"- Project: {project_path}\n"
        f"- Exported: {exported_at}\n\n"
        "---\n\n"
        f"{transcript.rstrip()}\n"
    )


def _default_export_path(ui: Any) -> Path:
    session_id = str(getattr(ui, "session_id", "session"))
    short_id = session_id.split("-")[0] or "session"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path.cwd() / f"ripperdoc-conversation-{short_id}-{timestamp}.md"


def _resolve_export_path(ui: Any, raw_path: Optional[str]) -> Path:
    if not raw_path:
        return _default_export_path(ui)
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate


def _copy_to_clipboard(text: str) -> tuple[bool, str]:
    if sys.platform == "darwin":
        commands = [["pbcopy"]]
    elif sys.platform.startswith("win"):
        commands = [["clip"], ["clip.exe"]]
    else:
        commands = [
            ["wl-copy"],
            ["xclip", "-selection", "clipboard"],
            ["xsel", "--clipboard", "--input"],
        ]

    for cmd in commands:
        if shutil.which(cmd[0]) is None:
            continue
        try:
            subprocess.run(cmd, input=text, text=True, check=True)
            return True, ""
        except (subprocess.SubprocessError, OSError):
            continue

    return (
        False,
        "No clipboard tool found. Install one of: pbcopy (macOS), wl-copy/xclip/xsel (Linux).",
    )


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


def _parse_args(trimmed_arg: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Parse export command args.

    Returns:
        (action, file_path, error_message)
    """
    if not trimmed_arg:
        return None, None, None
    try:
        parts = shlex.split(trimmed_arg)
    except ValueError as exc:
        return None, None, f"Invalid arguments: {exc}"

    if not parts:
        return None, None, None

    action = parts[0].lower()
    if action in {"clipboard", "copy", "clip"}:
        if len(parts) > 1:
            return None, None, "Usage: /export [clipboard|file [path]]"
        return "clipboard", None, None

    if action in {"file", "save"}:
        if len(parts) > 2:
            return None, None, "Usage: /export [clipboard|file [path]]"
        return "file", parts[1] if len(parts) == 2 else None, None

    return None, None, "Usage: /export [clipboard|file [path]]"


def _handle(ui: Any, arg: str) -> bool:
    messages = list(getattr(ui, "conversation_messages", []) or [])
    if not messages:
        ui.console.print("[yellow]No conversation to export yet.[/yellow]")
        return True

    transcript = render_transcript(messages, include_tool_details=True).strip()
    if not transcript:
        ui.console.print("[yellow]No exportable conversation content found.[/yellow]")
        return True

    action, file_arg, parse_error = _parse_args(arg.strip())
    if parse_error:
        ui.console.print(f"[red]{escape(parse_error)}[/red]")
        return True

    if action is None:
        action = _prompt_export_method()
        if action == "__cancel__":
            ui.console.print("[dim]Export cancelled.[/dim]")
            return True

    content = _build_export_content(ui, transcript)

    if action == "clipboard":
        ok, error = _copy_to_clipboard(content)
        if ok:
            ui.console.print("[green]✓ Copied conversation to clipboard.[/green]")
        else:
            ui.console.print(f"[red]Failed to copy to clipboard: {escape(error)}[/red]")
        return True

    if action == "file":
        path = _resolve_export_path(ui, file_arg)
        ok, error = _save_to_file(path, content)
        if ok:
            ui.console.print(f"[green]✓ Saved conversation to {escape(str(path))}[/green]")
        else:
            ui.console.print(f"[red]Failed to save export file: {escape(error)}[/red]")
        return True

    ui.console.print("[red]Unknown export action.[/red]")
    return True


command = SlashCommand(
    name="export",
    description="Export the current conversation to a file or clipboard",
    handler=_handle,
)


__all__ = ["command"]
