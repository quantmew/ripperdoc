"""Slash command to show background bash tasks."""

import textwrap

from rich import box
from rich.panel import Panel
from rich.table import Table

from ripperdoc.tools.background_shell import (
    get_background_status,
    list_background_tasks,
)

from .base import SlashCommand


def _format_duration(duration_ms) -> str:
    """Render milliseconds into a short human-readable duration."""
    if duration_ms is None:
        return "-"
    try:
        duration = float(duration_ms)
    except (TypeError, ValueError):
        return "-"
    if duration < 1000:
        return f"{int(duration)} ms"
    seconds = duration / 1000.0
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours, mins = divmod(minutes, 60)
    return f"{hours}h {mins}m"


def _format_status(status: dict) -> str:
    """Return a colored status label with exit codes when available."""
    state = status.get("status") or "unknown"
    if status.get("timed_out"):
        state = "failed"
    if status.get("killed"):
        state = "killed"

    exit_code = status.get("exit_code")
    label = state
    if exit_code is not None and state not in ("running", "killed"):
        label = f"{label} ({exit_code})"

    color = {
        "running": "yellow",
        "completed": "green",
        "failed": "red",
        "killed": "red",
    }.get(state)
    return f"[{color}]{label}[/{color}]" if color else label


def _handle(ui, _: str) -> bool:
    console = ui.console
    task_ids = list_background_tasks()

    if not task_ids:
        console.print(Panel("No tasks currently running", title="Background tasks", box=box.ROUNDED))
        return True

    table = Table(box=box.SIMPLE_HEAVY, expand=True)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Status", style="magenta", no_wrap=True)
    table.add_column("Command", style="white")
    table.add_column("Duration", style="dim", no_wrap=True)

    for task_id in sorted(task_ids):
        try:
            status = get_background_status(task_id, consume=False)
        except Exception as exc:
            table.add_row(task_id, "[red]error[/]", str(exc), "-")
            continue

        command = status.get("command") or ""
        command_display = textwrap.shorten(command, width=80, placeholder="...")
        table.add_row(
            task_id,
            _format_status(status),
            command_display,
            _format_duration(status.get("duration_ms")),
        )

    console.print(
        Panel(
            table,
            title="Background tasks",
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )
    console.print("[dim]Use BashOutput <id> to read output or KillBash to stop a running task.[/dim]")
    return True


command = SlashCommand(
    name="tasks",
    description="Show background tasks started with run_in_background",
    handler=_handle,
)


__all__ = ["command"]
