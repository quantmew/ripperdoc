from typing import Any
from uuid import uuid4

from .base import SlashCommand


def _handle(ui: Any, _: str) -> bool:
    try:
        ui._run_session_end("clear")
    except (AttributeError, RuntimeError, ValueError):
        pass

    try:
        set_session = getattr(ui, "_set_session", None)
        if callable(set_session):
            set_session(str(uuid4()))
    except (AttributeError, RuntimeError, ValueError, OSError, TypeError):
        pass

    ui.conversation_messages = []
    if hasattr(ui, "_saved_conversation"):
        ui._saved_conversation = None
    try:
        rebuild_usage = getattr(ui, "_rebuild_session_usage_from_messages", None)
        if callable(rebuild_usage):
            rebuild_usage(ui.conversation_messages)
    except (AttributeError, RuntimeError, ValueError, TypeError):
        pass

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
