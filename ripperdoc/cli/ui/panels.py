"""UI panels and visual components for RichUI.

This module contains welcome panels, status bars, and other
visual UI elements.
"""

from typing import List, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich import box

from ripperdoc import __version__
from ripperdoc.cli.ui.helpers import get_profile_for_pointer


def create_welcome_panel() -> Panel:
    """Create a welcome panel for the CLI startup."""
    welcome_content = """
[bold cyan]Welcome to Ripperdoc![/bold cyan]

Ripperdoc is an AI-powered coding assistant that helps with software development tasks.
You can read files, edit code, run commands, and help with various programming tasks.

[dim]Type your questions below. Press Ctrl+C to exit.[/dim]
"""

    return Panel(
        welcome_content,
        title=f"Ripperdoc v{__version__}",
        border_style="cyan",
        box=box.ROUNDED,
        padding=(1, 2),
    )


def create_status_bar() -> Text:
    """Create a status bar with current model information."""
    profile = get_profile_for_pointer("main")
    model_name = profile.model if profile else "Not configured"

    status_text = Text()
    status_text.append("Ripperdoc", style="bold cyan")
    status_text.append(" • ")
    status_text.append(model_name, style="dim")
    status_text.append(" • ")
    status_text.append("Ready", style="green")

    return status_text


def print_shortcuts(console: Console) -> None:
    """Show common keyboard shortcuts and prefixes."""
    pairs: List[Tuple[str, str]] = [
        ("? for shortcuts", "! for bash mode"),
        ("/ for commands", "@ for file mention"),
    ]
    console.print("[dim]Shortcuts[/dim]")
    for left, right in pairs:
        left_text = f"  {left}".ljust(32)
        right_text = f"{right}" if right else ""
        console.print(f"{left_text}{right_text}")
