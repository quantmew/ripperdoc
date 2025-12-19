"""Slash command to inspect and manage background bash tasks."""

import asyncio
import textwrap

from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.markup import escape

from ripperdoc.tools.background_shell import (
    get_background_status,
    kill_background_task,
    list_background_tasks,
)
from ripperdoc.utils.log import get_logger

from typing import Any, Optional
from .base import SlashCommand


logger = get_logger()


def _format_duration(duration_ms: Optional[float]) -> str:
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


def _tail_lines(text: str, max_lines: int = 20, max_chars: int = 4000) -> str:
    """Return a shortened view of output for display."""
    if not text:
        return ""

    lines = text.splitlines()
    prefixes = []

    if len(lines) > max_lines:
        lines = lines[-max_lines:]
        prefixes.append(f"[dim]... showing last {max_lines} lines[/dim]")

    content = "\n".join(lines)
    if len(content) > max_chars:
        content = content[-max_chars:]
        prefixes.insert(0, f"[dim]... output truncated to last {max_chars} chars[/dim]")

    if prefixes:
        return "\n".join(prefixes + [content])
    return content


def _list_tasks(ui: Any) -> bool:
    console = ui.console
    task_ids = list_background_tasks()

    if not task_ids:
        console.print(
            Panel("No tasks currently running", title="Background tasks", box=box.ROUNDED)
        )
        return True

    table = Table(box=box.SIMPLE_HEAVY, expand=True)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Status", style="magenta", no_wrap=True)
    table.add_column("Command", style="white")
    table.add_column("Duration", style="dim", no_wrap=True)

    for task_id in sorted(task_ids):
        try:
            status = get_background_status(task_id, consume=False)
        except (KeyError, ValueError, RuntimeError, OSError) as exc:
            table.add_row(escape(task_id), "[red]error[/]", escape(str(exc)), "-")
            logger.warning(
                "[tasks_cmd] Failed to read background task status: %s: %s",
                type(exc).__name__,
                exc,
                extra={"task_id": task_id, "session_id": getattr(ui, "session_id", None)},
            )
            continue

        command = status.get("command") or ""
        command_display = textwrap.shorten(command, width=80, placeholder="...")
        table.add_row(
            escape(task_id),
            _format_status(status),
            escape(command_display),
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
    console.print(
        "[dim]Use /tasks show <id> to view details/output, /tasks kill <id> to stop a task, or BashOutput/KillBash tools directly.[/dim]"
    )
    return True


def _kill_task(ui: Any, task_id: str) -> bool:
    console = ui.console
    try:
        status = get_background_status(task_id, consume=False)
    except KeyError:
        console.print(f"[red]No task found with id '{escape(task_id)}'.[/red]")
        return True
    except (ValueError, RuntimeError, OSError) as exc:
        console.print(f"[red]Failed to read task '{escape(task_id)}': {escape(str(exc))}[/red]")
        logger.warning(
            "[tasks_cmd] Failed to read task before kill: %s: %s",
            type(exc).__name__,
            exc,
            extra={"task_id": task_id, "session_id": getattr(ui, "session_id", None)},
        )
        return True

    if status.get("status") != "running":
        console.print(
            f"[yellow]Task {escape(task_id)} is not running (status: {escape(str(status.get('status')))}).[/yellow]"
        )
        return True

    runner = getattr(ui, "run_async", None)

    try:
        if callable(runner):
            killed = runner(kill_background_task(task_id))
        else:
            killed = asyncio.run(kill_background_task(task_id))
    except (OSError, RuntimeError, asyncio.CancelledError) as exc:
        if isinstance(exc, asyncio.CancelledError):
            raise
        console.print(f"[red]Error stopping task {escape(task_id)}: {escape(str(exc))}[/red]")
        logger.warning(
            "[tasks_cmd] Error stopping background task: %s: %s",
            type(exc).__name__,
            exc,
            extra={"task_id": task_id, "session_id": getattr(ui, "session_id", None)},
        )
        return True

    if killed:
        console.print(
            f"[green]Killed task {escape(task_id)}[/green] — {escape(status.get('command') or '')}",
        )
    else:
        console.print(f"[red]Failed to kill task {escape(task_id)}[/red]")
    return True


def _show_task(ui: Any, task_id: str) -> bool:
    console = ui.console
    try:
        status = get_background_status(task_id, consume=False)
    except KeyError:
        console.print(f"[red]No task found with id '{escape(task_id)}'.[/red]")
        return True
    except (ValueError, RuntimeError, OSError) as exc:
        console.print(f"[red]Failed to read task '{escape(task_id)}': {escape(str(exc))}[/red]")
        logger.warning(
            "[tasks_cmd] Failed to read task for detail view: %s: %s",
            type(exc).__name__,
            exc,
            extra={"task_id": task_id, "session_id": getattr(ui, "session_id", None)},
        )
        return True

    details = Table(box=box.SIMPLE_HEAVY, show_header=False)
    details.add_row("ID", escape(task_id))
    details.add_row("Status", _format_status(status))
    details.add_row("Command", escape(status.get("command") or ""))
    details.add_row("Duration", _format_duration(status.get("duration_ms")))
    exit_code = status.get("exit_code")
    details.add_row("Exit code", str(exit_code) if exit_code is not None else "running")

    console.print(
        Panel(details, title=f"Task {escape(task_id)}", box=box.ROUNDED, padding=(1, 2)),
        markup=False,
    )

    stdout_block = _tail_lines(status.get("stdout") or "")
    stderr_block = _tail_lines(status.get("stderr") or "")

    console.print(
        Panel(
            escape(stdout_block) if stdout_block else "[dim]No stdout yet[/dim]",
            title="stdout (latest)",
            box=box.SIMPLE,
            padding=(1, 2),
        )
    )
    console.print(
        Panel(
            escape(stderr_block) if stderr_block else "[dim]No stderr yet[/dim]",
            title="stderr (latest)",
            box=box.SIMPLE,
            padding=(1, 2),
        )
    )
    return True


def _handle(ui: Any, args: str) -> bool:
    parts = args.split()
    logger.info(
        "[tasks_cmd] Handling /tasks command",
        extra={
            "session_id": getattr(ui, "session_id", None),
            "raw_args": args,
        },
    )
    if not parts:
        return _list_tasks(ui)

    command = parts[0].lower()
    if command in {"kill", "stop"}:
        if len(parts) < 2:
            ui.console.print("[red]Usage: /tasks kill <task_id>[/red]")
            return True
        return _kill_task(ui, parts[1])

    if command in {"show", "info", "view"}:
        if len(parts) < 2:
            ui.console.print("[red]Usage: /tasks show <task_id>[/red]")
            return True
        return _show_task(ui, parts[1])

    ui.console.print(
        "[red]Unknown subcommand. Use /tasks, /tasks show <id>, or /tasks kill <id>.[/red]"
    )
    return True


command = SlashCommand(
    name="tasks",
    description="List, inspect, and manage background tasks started with run_in_background",
    handler=_handle,
)


__all__ = ["command"]
