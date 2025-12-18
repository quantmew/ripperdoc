"""Hooks command - Display configured hooks."""

from pathlib import Path
from typing import Any

from rich.markup import escape
from rich.table import Table

from .base import SlashCommand
from ripperdoc.core.hooks import (
    HookEvent,
    get_merged_hooks_config,
    get_global_hooks_path,
    get_project_hooks_path,
    get_project_local_hooks_path,
)


def _handle(ui: Any, _: str) -> bool:
    """Display all configured hooks."""
    project_path = getattr(ui, "project_path", None) or Path.cwd()
    config = get_merged_hooks_config(project_path)

    # Check config file locations
    global_path = get_global_hooks_path()
    project_path_hooks = get_project_hooks_path(project_path)
    local_path = get_project_local_hooks_path(project_path)

    ui.console.print()

    # Show config file status
    ui.console.print("[bold]Hook Configuration Files[/bold]")
    files_info = [
        ("Global", global_path),
        ("Project", project_path_hooks),
        ("Local", local_path),
    ]

    for label, path in files_info:
        if path.exists():
            ui.console.print(f"  [green]✓[/green] {label}: {path}")
        else:
            ui.console.print(f"  [dim]○[/dim] {label}: {path} [dim](not found)[/dim]")

    ui.console.print()

    # Count total hooks
    total_hooks = 0
    for matchers in config.hooks.values():
        for matcher in matchers:
            total_hooks += len(matcher.hooks)

    if not config.hooks:
        ui.console.print(
            "[yellow]No hooks configured.[/yellow]\n"
            "Add hooks to one of the configuration files above.\n"
            "See docs/hooks.md for configuration format."
        )
        return True

    ui.console.print(f"[bold]Registered Hooks[/bold] ({total_hooks} total)\n")

    # Display hooks by event
    for event in HookEvent:
        event_name = event.value
        if event_name not in config.hooks:
            continue

        matchers = config.hooks[event_name]
        if not matchers:
            continue

        # Create table for this event
        table = Table(
            title=f"[bold cyan]{event_name}[/bold cyan]",
            show_header=True,
            header_style="bold",
            expand=True,
            title_justify="left",
        )
        table.add_column("Matcher", style="yellow", width=20)
        table.add_column("Command", style="green")
        table.add_column("Timeout", style="dim", width=8, justify="right")

        for matcher in matchers:
            matcher_str = matcher.matcher or "*"
            for i, hook in enumerate(matcher.hooks):
                # Truncate long commands
                cmd = hook.command
                if len(cmd) > 60:
                    cmd = cmd[:57] + "..."

                # Only show matcher on first row of group
                if i == 0:
                    table.add_row(
                        escape(matcher_str),
                        escape(cmd),
                        f"{hook.timeout}s"
                    )
                else:
                    table.add_row(
                        "",
                        escape(cmd),
                        f"{hook.timeout}s"
                    )

        ui.console.print(table)
        ui.console.print()

    # Show quick reference
    ui.console.print("[dim]Tip: Hooks run in order. PreToolUse can block with exit code 2.[/dim]")

    return True


command = SlashCommand(
    name="hooks",
    description="Show configured hooks and their status",
    handler=_handle,
)


__all__ = ["command"]
