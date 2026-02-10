from typing import Any
from datetime import datetime
from typing import Optional

from rich.markup import escape

from ripperdoc.utils.session_history import (
    SessionSummary,
    list_session_summaries,
    load_session_messages,
)

from ripperdoc.cli.ui.choice import ChoiceOption, prompt_choice, theme_style

from .base import SlashCommand

# Number of sessions to display per page
PAGE_SIZE = 10


def _format_time(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M")


def _truncate_prompt(prompt: str, max_length: int = 50) -> str:
    """Truncate prompt text for display."""
    prompt = prompt.strip()
    if len(prompt) > max_length:
        return prompt[:max_length - 3] + "..."
    return prompt


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

        # Build choice options
        options = []
        for summary in page_sessions:
            prompt_preview = _truncate_prompt(summary.last_prompt)
            label = (
                f"<info>{summary.session_id[:8]}...</info> "
                f"({summary.message_count} msgs) "
                f"<dim>{_format_time(summary.created_at)}</dim> "
                f"<dim>{escape(prompt_preview)}</dim>"
            )
            options.append(ChoiceOption(summary.session_id, label))

        # Add navigation options
        nav_options = []
        if current_page > 0:
            nav_options.append(ChoiceOption(
                "__prev__",
                "<dim>←</dim> <dim>Previous page</dim>"
            ))
        if current_page < total_pages - 1:
            nav_options.append(ChoiceOption(
                "__next__",
                "<dim>→</dim> <dim>Next page</dim>"
            ))

        # Build the message
        message = f"<question>Select a session to resume</question> <dim>(Page {current_page + 1}/{total_pages})</dim>"

        # Combine options: sessions first, then navigation
        all_options = options + nav_options

        try:
            result = prompt_choice(
                message=message,
                options=all_options,
                title="Resume Session",
                allow_esc=True,
                esc_value="__cancel__",
                style=theme_style(),
            )
        except (EOFError, KeyboardInterrupt):
            return None

        # Handle the result
        if result == "__cancel__":
            return None
        elif result == "__next__":
            current_page += 1
            continue
        elif result == "__prev__":
            current_page -= 1
            continue
        else:
            # Find and return the selected session
            for s in sessions:
                if s.session_id == result:
                    return s
            return None


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
    if hasattr(ui, "_rebuild_session_usage_from_messages"):
        try:
            ui._rebuild_session_usage_from_messages(messages)
        except (TypeError, ValueError, RuntimeError, AttributeError):
            pass
    try:
        ui._run_session_start("resume")
    except (AttributeError, RuntimeError, ValueError):
        pass
    replay_limit = getattr(ui, "_resume_replay_max_messages", None)
    ui.replay_conversation(messages, max_messages=replay_limit)
    ui.console.print(
        f"[green]✓ Resumed session {escape(summary.session_id)} "
        f"with {len(messages)} messages.[/green]"
    )
    return True


command = SlashCommand(
    name="resume",
    description="Resume a previous session conversation",
    handler=_handle,
    aliases=("sessions",),
)


__all__ = ["command"]
