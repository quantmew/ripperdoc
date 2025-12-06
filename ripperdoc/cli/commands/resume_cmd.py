from typing import Any
from datetime import datetime
from typing import Optional

from rich.markup import escape

from ripperdoc.utils.session_history import (
    SessionSummary,
    list_session_summaries,
    load_session_messages,
)

from .base import SlashCommand


def _format_time(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M")


def _choose_session(ui: Any, arg: str) -> Optional[SessionSummary]:
    sessions = list_session_summaries(ui.project_path)
    if not sessions:
        ui.console.print("[yellow]No saved sessions found for this project.[/yellow]")
        return None

    # If a numeric arg is provided, try to resolve immediately.
    if arg.strip():
        if arg.isdigit():
            idx = int(arg)
            if 0 <= idx < len(sessions):
                return sessions[idx]
            ui.console.print(
                f"[red]Invalid session index {escape(str(idx))}. "
                f"Choose 0-{len(sessions) - 1}.[/red]"
            )
        else:
            # Treat arg as session id if it matches.
            match = next((s for s in sessions if s.session_id.startswith(arg.strip())), None)
            if match:
                return match
            ui.console.print(f"[red]No session found matching '{escape(arg)}'.[/red]")
            return None

    ui.console.print("\n[bold]Saved sessions:[/bold]")
    for idx, summary in enumerate(sessions):
        ui.console.print(
            f"  [{idx}] {summary.session_id} "
            f"({summary.message_count} messages, "
            f"{_format_time(summary.created_at)} → {_format_time(summary.updated_at)}) "
            f"{escape(summary.first_prompt)}",
            markup=False,
        )

    choice_text = ui.console.input("\nSelect a session index (Enter to cancel): ").strip()
    if not choice_text:
        return None
    if not choice_text.isdigit():
        ui.console.print("[red]Please enter a number.[/red]")
        return None

    idx = int(choice_text)
    if idx < 0 or idx >= len(sessions):
        ui.console.print(
            f"[red]Invalid session index {escape(str(idx))}. "
            f"Choose 0-{len(sessions) - 1}.[/red]"
        )
        return None
    return sessions[idx]


def _handle(ui: Any, arg: str) -> bool:
    summary = _choose_session(ui, arg)
    if not summary:
        return True

    messages = load_session_messages(ui.project_path, summary.session_id)
    if not messages:
        ui.console.print("[red]Failed to load messages for the selected session.[/red]")
        return True

    ui.conversation_messages = messages
    ui._saved_conversation = None
    ui._set_session(summary.session_id)
    ui.replay_conversation(messages)
    ui.console.print(
        f"[green]✓ Resumed session {escape(summary.session_id)} "
        f"with {len(messages)} messages.[/green]"
    )
    return True


command = SlashCommand(
    name="resume",
    description="Resume a previous session conversation",
    handler=_handle,
)


__all__ = ["command"]
