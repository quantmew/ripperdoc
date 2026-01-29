"""Theme management command for Ripperdoc.

Allows users to list, preview, and switch UI themes.
"""

from typing import Any

from rich.markup import escape
from rich.table import Table

from ripperdoc.core.config import get_global_config, save_global_config
from ripperdoc.core.theme import (
    BUILTIN_THEMES,
    Theme,
    get_theme_manager,
)

from .base import SlashCommand


def _show_current_theme(ui: Any) -> None:
    """Display current theme information."""
    manager = get_theme_manager()
    theme = manager.current
    primary = manager.get_color("primary")
    ui.console.print(f"\nCurrent theme: [bold {primary}]{theme.display_name}[/]")
    ui.console.print(f"  {theme.description}")


def _list_themes(ui: Any) -> None:
    """List all available themes."""
    manager = get_theme_manager()
    current_name = manager.current.name

    ui.console.print("\n[bold]Available Themes:[/bold]")

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Marker", width=2)
    table.add_column("Name", width=16)
    table.add_column("Description")

    for name, theme in BUILTIN_THEMES.items():
        is_current = name == current_name
        marker = f"[{manager.get_color('primary')}]→[/]" if is_current else " "
        display = (
            f"[bold {manager.get_color('primary')}]{theme.display_name}[/]"
            if is_current
            else theme.display_name
        )
        table.add_row(marker, display, theme.description)

    ui.console.print(table)
    ui.console.print("\n[dim]Usage: /themes <name> to switch[/dim]")


def _preview_theme(ui: Any, theme: Theme) -> None:
    """Preview a theme's color palette."""
    colors = theme.colors
    ui.console.print(f"\n[bold]Theme Preview:[/bold] {theme.display_name}\n")

    samples = [
        ("Primary", colors.primary, "Brand color, borders"),
        ("Secondary", colors.secondary, "Success, ready status"),
        ("Error", colors.error, "Error messages"),
        ("Warning", colors.warning, "Warning messages"),
        ("Info", colors.info, "Info messages"),
        ("Tool Call", colors.tool_call, "Tool invocations"),
        ("Spinner", colors.spinner, "Loading indicator"),
        ("Emphasis", colors.emphasis, "Highlighted text"),
    ]

    for label, color, desc in samples:
        ui.console.print(f"  [{color}]■[/] {label}: {desc}")


def _switch_theme(ui: Any, theme_name: str, preview_only: bool = False) -> None:
    """Switch to a different theme."""
    manager = get_theme_manager()

    if theme_name not in BUILTIN_THEMES:
        ui.console.print(f"[{manager.get_color('error')}]Unknown theme: {escape(theme_name)}[/]")
        available = ", ".join(BUILTIN_THEMES.keys())
        ui.console.print(f"[dim]Available themes: {available}[/dim]")
        return

    theme = BUILTIN_THEMES[theme_name]

    if preview_only:
        _preview_theme(ui, theme)
        return

    # Apply theme
    manager.set_theme(theme_name)

    # Persist to config
    try:
        config = get_global_config()
        config.theme = theme_name
        save_global_config(config)
    except Exception as e:
        ui.console.print(
            f"[{manager.get_color('warning')}]Theme applied but failed to save: {e}[/]"
        )
        return

    ui.console.print(f"[{manager.get_color('success')}]✓ Theme switched to {theme.display_name}[/]")


def _handle(ui: Any, arg: str) -> bool:
    """Handle the /themes command."""
    arg = arg.strip().lower()

    if not arg:
        _show_current_theme(ui)
        _list_themes(ui)
        return True

    parts = arg.split()

    if parts[0] == "preview" and len(parts) > 1:
        _switch_theme(ui, parts[1], preview_only=True)
        return True

    if parts[0] == "list":
        _list_themes(ui)
        return True

    # Direct theme switch
    _switch_theme(ui, parts[0])
    return True


command = SlashCommand(
    name="themes",
    description="List and switch UI themes",
    handler=_handle,
)

__all__ = ["command"]
