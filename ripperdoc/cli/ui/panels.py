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
from ripperdoc.core.theme import theme_color


def create_welcome_panel() -> Panel:
    """Create a welcome panel for the CLI startup."""
    import os

    primary = theme_color("primary")
    muted = theme_color("text_secondary")

    profile = get_profile_for_pointer("main")
    model_name = profile.model if profile else "Not configured"
    protocol = profile.protocol.value if profile else "unknown"
    cwd = os.getcwd()

    secondary = theme_color("secondary")
    welcome_content = f"""
[bold {primary}]Welcome to Ripperdoc![/bold {primary}]

[{muted}]model:     {model_name} • [{secondary}]Ready[/{secondary}]
protocol:  {protocol}
directory: {cwd}[/{muted}]
"""

    return Panel(
        welcome_content,
        title=f"Ripperdoc v{__version__}",
        border_style=theme_color("border"),
        box=box.ROUNDED,
        padding=(1, 2),
        expand=False,
    )


def create_status_bar() -> Text:
    """Create a status bar with current model information."""
    import os

    profile = get_profile_for_pointer("main")
    model_name = profile.model if profile else "Not configured"

    # Get current working directory
    cwd = os.getcwd()

    status_text = Text()
    status_text.append("Ripperdoc", style=f"bold {theme_color('primary')}")
    status_text.append(" • ")
    status_text.append(model_name, style=theme_color("text_secondary"))
    status_text.append(" • ")
    status_text.append("Ready", style=theme_color("secondary"))
    status_text.append(" • ")
    status_text.append(cwd, style=theme_color("text_secondary"))

    return status_text


def print_shortcuts(console: Console) -> None:
    """Show common keyboard shortcuts and prefixes."""
    pairs: List[Tuple[str, str]] = [
        ("? for shortcuts", "! for bash mode"),
        ("/ for commands", "@ for file mention"),
        ("Alt+Enter for newline", "Enter to submit"),
        ("Esc Esc for history", ""),
    ]
    muted = theme_color("text_secondary")
    console.print(f"[{muted}]Shortcuts[/{muted}]")
    for left, right in pairs:
        left_text = f"  {left}".ljust(32)
        right_text = f"{right}" if right else ""
        console.print(f"{left_text}{right_text}")
