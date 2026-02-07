"""Slash command to add additional working directories for the current session."""

from __future__ import annotations

import shlex
from typing import Any

from rich.markup import escape

from .base import SlashCommand


def _render_current_dirs(ui: Any) -> None:
    lister = getattr(ui, "list_additional_working_directories", None)
    if not callable(lister):
        ui.console.print("[dim]No session-scoped additional directories are configured.[/dim]")
        return

    dirs = list(lister())
    if not dirs:
        ui.console.print("[dim]No session-scoped additional directories are configured.[/dim]")
        return

    ui.console.print("[bold]Session Additional Directories:[/bold]")
    for directory in dirs:
        ui.console.print(f"  - {escape(directory)}")


def _handle(ui: Any, trimmed_arg: str) -> bool:
    adder = getattr(ui, "add_additional_working_directory", None)
    if not callable(adder):
        ui.console.print("[red]/add-dir is not available in this interface.[/red]")
        return True

    if not trimmed_arg.strip():
        ui.console.print("[red]Usage: /add-dir <path> [path2 ...][/red]")
        _render_current_dirs(ui)
        return True

    try:
        raw_paths = shlex.split(trimmed_arg)
    except ValueError as exc:
        ui.console.print(f"[red]Failed to parse paths: {escape(str(exc))}[/red]")
        return True

    if not raw_paths:
        ui.console.print("[red]Usage: /add-dir <path> [path2 ...][/red]")
        return True

    for raw_path in raw_paths:
        added, message = adder(raw_path)
        color = "green" if added else "yellow"
        ui.console.print(f"[{color}]{escape(message)}[/{color}]")

    return True


command = SlashCommand(
    name="add-dir",
    description="Add session-scoped additional working directories",
    handler=_handle,
)


__all__ = ["command"]
