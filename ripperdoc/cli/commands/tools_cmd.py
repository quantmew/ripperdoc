from typing import Any
from .base import SlashCommand


def _handle(ui: Any, _: str) -> bool:
    ui.console.print("\n[bold]Available Tools:[/bold]")
    for tool in ui.get_default_tools():
        ui.console.print(f"  • {tool.name}")
    return True


command = SlashCommand(
    name="tools",
    description="List available tools",
    handler=_handle,
)


__all__ = ["command"]
