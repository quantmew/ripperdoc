from typing import Any
from .base import SlashCommand


def _handle(ui: Any, _: str) -> bool:
    ui.console.print("\n[bold]Available Slash Commands:[/bold]")
    for cmd in ui.command_list:
        alias_text = f" (aliases: {', '.join(cmd.aliases)})" if cmd.aliases else ""
        ui.console.print(f"  /{cmd.name:<12} - {cmd.description}{alias_text}")

    # Show custom commands if any
    custom_cmds = getattr(ui, "_custom_command_list", [])
    if custom_cmds:
        ui.console.print("\n[bold]Custom Commands:[/bold]")
        for cmd in sorted(custom_cmds, key=lambda c: c.name):
            hint = f" {cmd.argument_hint}" if cmd.argument_hint else ""
            location = f" ({cmd.location.value})" if cmd.location else ""
            ui.console.print(f"  /{cmd.name:<12}{hint} - {cmd.description}{location}")

    return True


command = SlashCommand(
    name="help",
    description="Show available slash commands",
    handler=_handle,
)


__all__ = ["command"]
