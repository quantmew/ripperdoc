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

# Number of sessions to display per page
PAGE_SIZE = 20


def _format_time(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M")


def _choose_session(ui: Any, arg: str) -> Optional[SessionSummary]:
    sessions = list_session_summaries(ui.project_path)
    if not sessions:
        ui.console.print("[yellow]No saved sessions found for this project.[/yellow]")
        return None

    # If arg is provided, treat it as session id prefix match
    if arg.strip():
        match = next((s for s in sessions if s.session_id.startswith(arg.strip())), None)
        if match:
            return match
        ui.console.print(f"[red]No session found matching '{escape(arg)}'.[/red]")
        return None

    # Pagination settings
    current_page = 0
    total_pages = (len(sessions) + PAGE_SIZE - 1) // PAGE_SIZE

    while True:
        start_idx = current_page * PAGE_SIZE
        end_idx = min(start_idx + PAGE_SIZE, len(sessions))
        page_sessions = sessions[start_idx:end_idx]

        ui.console.print(f"\n[bold]Saved sessions (Page {current_page + 1}/{total_pages}):[/bold]")
        for idx, summary in enumerate(page_sessions, start=start_idx):
            ui.console.print(
                f"  [{idx}] {summary.session_id} "
                f"({summary.message_count} messages, "
                f"{_format_time(summary.created_at)} → {_format_time(summary.updated_at)}) "
                f"{escape(summary.last_prompt)}",
                markup=False,
            )

        # Show navigation hints
        nav_hints = []
        if current_page > 0:
            nav_hints.append("'p' for previous page")
        if current_page < total_pages - 1:
            nav_hints.append("'n' for next page")
        nav_hints.append("Enter to cancel")

        prompt = "\nSelect session index"
        if nav_hints:
            prompt += f" ({', '.join(nav_hints)})"
        prompt += ": "

        choice_text = ui.console.input(prompt).strip().lower()

        if not choice_text:
            return None

        # Handle pagination commands
        if choice_text == "n":
            if current_page < total_pages - 1:
                current_page += 1
                continue
            else:
                ui.console.print("[yellow]Already at the last page.[/yellow]")
                continue
        elif choice_text == "p":
            if current_page > 0:
                current_page -= 1
                continue
            else:
                ui.console.print("[yellow]Already at the first page.[/yellow]")
                continue

        # Handle session selection
        if not choice_text.isdigit():
            ui.console.print(
                "[red]Please enter a session index number, 'n' for next page, or 'p' for previous page.[/red]"
            )
            continue

        idx = int(choice_text)
        if idx < 0 or idx >= len(sessions):
            ui.console.print(
                f"[red]Invalid session index {escape(str(idx))}. Choose 0-{len(sessions) - 1}.[/red]"
            )
            continue
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
