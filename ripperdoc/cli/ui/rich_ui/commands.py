"""Slash command handling helpers for the Rich UI."""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Callable, List, Optional

from rich.markup import escape

from ripperdoc.cli.commands import (
    get_custom_command,
    get_slash_command,
    expand_command_content,
)


def suggest_slash_commands(
    name: str,
    project_path: Optional[Path],
    completions_fn: Callable[[Optional[Path]], List[tuple[str, object]]],
) -> List[str]:
    """Return close matching slash commands for a mistyped name."""
    if not name:
        return []
    seen = set()
    candidates: List[str] = []
    for command_name, _cmd in completions_fn(project_path):
        if command_name not in seen:
            candidates.append(command_name)
            seen.add(command_name)
    return difflib.get_close_matches(name, candidates, n=3, cutoff=0.6)


def handle_slash_command(ui: object, user_input: str, suggest_fn: Callable[[str, Optional[Path]], List[str]]) -> bool | str:
    """Handle slash commands.

    Returns True if handled as a built-in, False if not a command,
    or a string if it's a custom command that should be sent to the AI.
    """
    if not user_input.startswith("/"):
        return False

    parts = user_input[1:].strip().split()
    if not parts:
        ui.console.print("[red]No command provided after '/'.[/red]")
        return True

    command_name = parts[0].lower()
    trimmed_arg = " ".join(parts[1:]).strip()

    # First, try built-in commands.
    command = get_slash_command(command_name)
    if command is not None:
        return command.handler(ui, trimmed_arg)

    # Then, try custom commands.
    custom_cmd = get_custom_command(command_name, ui.project_path)
    if custom_cmd is not None:
        # Expand the custom command content.
        expanded_content = expand_command_content(custom_cmd, trimmed_arg, ui.project_path)

        # Show a hint that this is from a custom command.
        ui.console.print(f"[dim]Running custom command: /{command_name}[/dim]")
        if custom_cmd.argument_hint and trimmed_arg:
            ui.console.print(f"[dim]Arguments: {trimmed_arg}[/dim]")

        # Return the expanded content to be processed as a query.
        return expanded_content

    suggestions = suggest_fn(command_name, ui.project_path)
    hint = ""
    if suggestions:
        hint = " [dim]Did you mean "
        hint += ", ".join(f"/{escape(s)}" for s in suggestions)
        hint += "?[/dim]"

    ui.console.print(f"[red]Unknown command: {escape(command_name)}[/red]{hint}")
    return True
