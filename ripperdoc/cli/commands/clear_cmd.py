from typing import Any
from .base import SlashCommand


def _handle(ui: Any, _: str) -> bool:
    ui.conversation_messages = []
    ui.console.print("[green]✓ Conversation cleared[/green]")
    return True


command = SlashCommand(
    name="clear", description="Clear conversation history", handler=_handle, aliases=("new",)
)


__all__ = ["command"]
