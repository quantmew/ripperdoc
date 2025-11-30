"""Slash command to display stored todos."""

from rich import box
from rich.panel import Panel

from ripperdoc.utils.todo import (
    format_todo_lines,
    format_todo_summary,
    get_next_actionable,
    load_todos,
)

from .base import SlashCommand


def _handle(ui, trimmed_arg: str) -> bool:
    console = ui.console
    todos = load_todos(ui.project_path)
    next_only = trimmed_arg.strip().lower() in ("next", "-n", "--next")

    if not todos:
        console.print("  ⎿  [dim]No todos currently tracked[/]")
        return True

    if next_only:
        next_todo = get_next_actionable(todos)
        if not next_todo:
            console.print("[yellow]No actionable todos (none pending or in_progress).[/yellow]")
            return True
        console.print(
            Panel(
                f"{next_todo.content}\n[id: {next_todo.id} | {next_todo.status} | {next_todo.priority}]",
                title="Next todo",
                box=box.ROUNDED,
            )
        )
        return True

    summary = format_todo_summary(todos)
    lines = format_todo_lines(todos)
    body = "\n".join(lines)
    panel = Panel(
        body or "No todos currently tracked",
        title="Todos",
        subtitle=summary,
        box=box.ROUNDED,
        padding=(1, 2),
    )
    console.print(panel)

    next_todo = get_next_actionable(todos)
    if next_todo:
        console.print(
            f"[dim]Next: {next_todo.content} (id: {next_todo.id}, status: {next_todo.status})[/dim]"
        )
    return True


command = SlashCommand(
    name="todos",
    description="Show the stored todo list for this project",
    handler=_handle,
    aliases=(),
)


__all__ = ["command"]
