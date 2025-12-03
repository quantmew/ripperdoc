"""Slash command registry."""

from __future__ import annotations

from typing import Dict, List

from .base import SlashCommand
from .agents_cmd import command as agents_command
from .clear_cmd import command as clear_command
from .compact_cmd import command as compact_command
from .config_cmd import command as config_command
from .cost_cmd import command as cost_command
from .context_cmd import command as context_command
from .doctor_cmd import command as doctor_command
from .exit_cmd import command as exit_command
from .help_cmd import command as help_command
from .memory_cmd import command as memory_command
from .mcp_cmd import command as mcp_command
from .models_cmd import command as models_command
from .resume_cmd import command as resume_command
from .tasks_cmd import command as tasks_command
from .status_cmd import command as status_command
from .todos_cmd import command as todos_command
from .tools_cmd import command as tools_command


def _build_registry(commands: List[SlashCommand]) -> Dict[str, SlashCommand]:
    """Map command names and aliases to SlashCommand entries."""
    registry: Dict[str, SlashCommand] = {}
    for cmd in commands:
        registry[cmd.name] = cmd
        for alias in cmd.aliases:
            registry[alias] = cmd
    return registry


ALL_COMMANDS: List[SlashCommand] = [
    help_command,
    clear_command,
    config_command,
    tools_command,
    models_command,
    exit_command,
    status_command,
    doctor_command,
    memory_command,
    tasks_command,
    todos_command,
    mcp_command,
    cost_command,
    context_command,
    compact_command,
    resume_command,
    agents_command,
]

COMMAND_REGISTRY: Dict[str, SlashCommand] = _build_registry(ALL_COMMANDS)


def list_slash_commands() -> List[SlashCommand]:
    """Return the ordered list of base slash commands (no aliases)."""
    return ALL_COMMANDS


def get_slash_command(name: str) -> SlashCommand | None:
    """Return a command by name or alias."""
    return COMMAND_REGISTRY.get(name)


def slash_command_completions() -> List[tuple[str, SlashCommand]]:
    """Return (name, command) pairs for completion including aliases."""
    return list(COMMAND_REGISTRY.items())


__all__ = [
    "SlashCommand",
    "list_slash_commands",
    "get_slash_command",
    "slash_command_completions",
]
