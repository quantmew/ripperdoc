from dataclasses import dataclass
from typing import Dict, List, Optional

from ripperdoc.cli.commands import list_slash_commands


@dataclass(frozen=True)
class CommandDef:
    """Simple definition of a slash command."""

    name: str
    description: str


DEFAULT_COMMANDS: List[CommandDef] = []
for cmd in list_slash_commands():
    DEFAULT_COMMANDS.append(CommandDef(cmd.name, cmd.description))
    for alias in cmd.aliases:
        DEFAULT_COMMANDS.append(CommandDef(alias, cmd.description))

COMMAND_LOOKUP: Dict[str, CommandDef] = {cmd.name: cmd for cmd in DEFAULT_COMMANDS}


def get_command(name: str) -> Optional[CommandDef]:
    """Return a command definition by name."""

    return COMMAND_LOOKUP.get(name)


def list_commands() -> List[CommandDef]:
    """Return all available commands."""

    return DEFAULT_COMMANDS
