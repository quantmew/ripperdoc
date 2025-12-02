import asyncio

from typing import Any
from .base import SlashCommand


def _handle(ui: Any, trimmed_arg: str) -> bool:
    asyncio.run(ui._run_manual_compact(trimmed_arg))
    return True


command = SlashCommand(
    name="compact",
    description="Compact conversation history",
    handler=_handle,
)


__all__ = ["command"]
