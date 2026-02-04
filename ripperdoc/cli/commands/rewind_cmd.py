from typing import Any

from .base import SlashCommand


def _handle(ui: Any, _: str) -> bool:
    handler = getattr(ui, "_open_history_picker_async", None)
    if not callable(handler):
        console = getattr(ui, "console", None)
        if console is not None:
            console.print("[red]Rewind is not available in this UI.[/red]")
        return True

    runner = getattr(ui, "run_async", None)
    if callable(runner):
        runner(handler())
    else:
        import asyncio

        asyncio.run(handler())
    return True


command = SlashCommand(
    name="rewind",
    description="Open history picker to roll back the conversation",
    handler=_handle,
)


__all__ = ["command"]
