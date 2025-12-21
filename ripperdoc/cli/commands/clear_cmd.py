from typing import Any
from .base import SlashCommand


def _handle(ui: Any, _: str) -> bool:
    try:
        ui._run_session_end("clear")
    except (AttributeError, RuntimeError, ValueError):
        pass
    ui.conversation_messages = []
    ui.console.print("[green]✓ Conversation cleared[/green]")
    try:
        ui._run_session_start("clear")
    except (AttributeError, RuntimeError, ValueError):
        pass
    return True


command = SlashCommand(
    name="clear", description="Clear conversation history", handler=_handle, aliases=("new",)
)


__all__ = ["command"]
