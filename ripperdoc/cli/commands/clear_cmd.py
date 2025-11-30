from .base import SlashCommand


def _handle(ui, _: str) -> bool:
    ui.conversation_messages = []
    ui.console.print("[green]✓ Conversation cleared[/green]")
    return True


command = SlashCommand(
    name="clear",
    description="Clear conversation history",
    handler=_handle,
)


__all__ = ["command"]
