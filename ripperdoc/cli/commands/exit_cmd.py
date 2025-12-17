from typing import Any
from .base import SlashCommand


def _handle(ui: Any, _: str) -> bool:
    ui.console.print("[yellow]Goodbye![/yellow]")
    ui._should_exit = True
    return True


command = SlashCommand(
    name="exit",
    description="Exit Ripperdoc",
    handler=_handle,
    aliases=("quit",),
)


__all__ = ["command"]
