"""Output style command for controlling assistant response behavior."""

from __future__ import annotations

from typing import Any

from rich.markup import escape
from rich.table import Table

from ripperdoc.cli.ui.choice import ChoiceOption, prompt_choice, theme_style
from ripperdoc.core.config import get_project_local_config, save_project_local_config
from ripperdoc.core.output_styles import (
    OutputStyleDefinition,
    OutputStyleLocation,
    find_output_style,
    load_all_output_styles,
)

from .base import SlashCommand


def _current_style_key(ui: Any) -> str:
    style = getattr(ui, "output_style", "default")
    if isinstance(style, str) and style.strip():
        return style.strip()
    return "default"


def _style_source_label(style: OutputStyleDefinition) -> str:
    if style.location == OutputStyleLocation.BUILTIN:
        return "builtin"
    if style.location == OutputStyleLocation.PROJECT:
        return "project"
    return "user"


def _apply_style(ui: Any, style: OutputStyleDefinition) -> None:
    config = get_project_local_config(ui.project_path)
    config.output_style = style.key
    save_project_local_config(config, ui.project_path)

    setter = getattr(ui, "set_output_style", None)
    if callable(setter):
        setter(style.key)
    else:
        setattr(ui, "output_style", style.key)


def _list_styles(ui: Any) -> bool:
    current_key = _current_style_key(ui)
    result = load_all_output_styles(project_path=ui.project_path)

    table = Table(title="Available output styles", show_header=True)
    table.add_column("Current", justify="center", width=7)
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Name", style="bold")
    table.add_column("Source", style="dim", no_wrap=True)
    table.add_column("Description")

    for style in result.styles:
        marker = "✔" if style.key == current_key else ""
        description = style.description or ""
        table.add_row(marker, style.key, style.name, _style_source_label(style), description)

    ui.console.print()
    ui.console.print(table)

    if result.errors:
        ui.console.print("\n[yellow]Some output style files could not be loaded:[/yellow]")
        for error in result.errors:
            ui.console.print(f"  - {escape(str(error.path))}: {escape(error.reason)}")
    return True


def _interactive_select(ui: Any) -> bool:
    current_key = _current_style_key(ui)
    result = load_all_output_styles(project_path=ui.project_path)
    if not result.styles:
        ui.console.print("[red]No output styles available.[/red]")
        return True

    options = [
        ChoiceOption(
            style.key,
            f"<info>{style.name}</info>",
            style.description or f"Style key: {style.key}",
            is_default=(style.key == current_key),
        )
        for style in result.styles
    ]

    selected = prompt_choice(
        message="",
        title="Preferred output style",
        description="This changes how Ripperdoc communicates with you.",
        options=options,
        allow_esc=True,
        esc_value=current_key,
        style=theme_style(),
    )

    selected_style = next((style for style in result.styles if style.key == selected), None)
    if selected_style is None:
        ui.console.print("[red]No output style selected.[/red]")
        return True

    if selected_style.key == current_key:
        ui.console.print(f"[dim]Output style remains {escape(selected_style.name)}.[/dim]")
        return True

    _apply_style(ui, selected_style)
    ui.console.print(
        f"[green]✓ Output style set to {escape(selected_style.name)} ({escape(selected_style.key)}).[/green]"
    )
    return True


def _set_style_from_arg(ui: Any, raw_arg: str) -> bool:
    selected_style, result = find_output_style(raw_arg, project_path=ui.project_path)
    if selected_style is None:
        ui.console.print(f"[red]Unknown output style: {escape(raw_arg)}[/red]")
        available = ", ".join(style.key for style in result.styles)
        ui.console.print(f"[dim]Available styles: {escape(available)}[/dim]")
        return True

    _apply_style(ui, selected_style)
    ui.console.print(
        f"[green]✓ Output style set to {escape(selected_style.name)} ({escape(selected_style.key)}).[/green]"
    )
    if result.errors:
        ui.console.print("[yellow]Some style files had load errors; run /output-style list for details.[/yellow]")
    return True


def _handle(ui: Any, arg: str) -> bool:
    trimmed = arg.strip()
    if not trimmed:
        return _interactive_select(ui)

    lower = trimmed.lower()
    if lower in {"list", "ls"}:
        return _list_styles(ui)
    return _set_style_from_arg(ui, trimmed)


command = SlashCommand(
    name="output-style",
    description="Select response style (default/explanatory/learning/custom)",
    handler=_handle,
)


__all__ = ["command"]
