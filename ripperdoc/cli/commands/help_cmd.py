from typing import Any
from .base import SlashCommand


def _handle(ui: Any, _: str) -> bool:
    ui.console.print("\n[bold]Available Slash Commands:[/bold]")
    for cmd in ui.command_list:
        alias_text = f" (aliases: {', '.join(cmd.aliases)})" if cmd.aliases else ""
        ui.console.print(f"  /{cmd.name:<8} - {cmd.description}{alias_text}")
    return True


command = SlashCommand(
    name="help",
    description="Show available slash commands",
    handler=_handle,
)


__all__ = ["command"]
