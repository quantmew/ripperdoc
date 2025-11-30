"""Shared types for slash command handlers."""

from dataclasses import dataclass
from typing import Any, Callable, Tuple

Handler = Callable[[Any, str], bool]


@dataclass(frozen=True)
class SlashCommand:
    """A single slash command implementation."""

    name: str
    description: str
    handler: Handler
    aliases: Tuple[str, ...] = ()


__all__ = ["SlashCommand", "Handler"]
