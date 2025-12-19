"""Slash command registry with custom command support."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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
from .hooks_cmd import command as hooks_command
from .memory_cmd import command as memory_command
from .mcp_cmd import command as mcp_command
from .models_cmd import command as models_command
from .permissions_cmd import command as permissions_command
from .resume_cmd import command as resume_command
from .tasks_cmd import command as tasks_command
from .status_cmd import command as status_command
from .todos_cmd import command as todos_command
from .tools_cmd import command as tools_command

from ripperdoc.core.custom_commands import (
    CustomCommandDefinition,
    load_all_custom_commands,
    expand_command_content,
)


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
    permissions_command,
    tasks_command,
    todos_command,
    mcp_command,
    hooks_command,
    cost_command,
    context_command,
    compact_command,
    resume_command,
    agents_command,
]

COMMAND_REGISTRY: Dict[str, SlashCommand] = _build_registry(ALL_COMMANDS)

# Cache for custom commands
_custom_commands_cache: Optional[Tuple[Path, List[CustomCommandDefinition]]] = None


def _get_custom_commands(project_path: Optional[Path] = None) -> List[CustomCommandDefinition]:
    """Get custom commands with caching."""
    global _custom_commands_cache
    current_path = (project_path or Path.cwd()).resolve()

    # Return cached commands if same project
    if _custom_commands_cache and _custom_commands_cache[0] == current_path:
        return _custom_commands_cache[1]

    # Load and cache
    result = load_all_custom_commands(project_path=current_path)
    _custom_commands_cache = (current_path, result.commands)
    return result.commands


def refresh_custom_commands(project_path: Optional[Path] = None) -> List[CustomCommandDefinition]:
    """Force reload custom commands."""
    global _custom_commands_cache
    _custom_commands_cache = None
    return _get_custom_commands(project_path)


def list_slash_commands() -> List[SlashCommand]:
    """Return the ordered list of base slash commands (no aliases)."""
    return ALL_COMMANDS


def list_custom_commands(project_path: Optional[Path] = None) -> List[CustomCommandDefinition]:
    """Return all loaded custom commands."""
    return _get_custom_commands(project_path)


def get_slash_command(name: str) -> SlashCommand | None:
    """Return a built-in command by name or alias."""
    return COMMAND_REGISTRY.get(name)


def get_custom_command(
    name: str, project_path: Optional[Path] = None
) -> CustomCommandDefinition | None:
    """Return a custom command by name."""
    commands = _get_custom_commands(project_path)
    return next((cmd for cmd in commands if cmd.name == name), None)


def slash_command_completions(
    project_path: Optional[Path] = None,
) -> List[Tuple[str, Union[SlashCommand, CustomCommandDefinition]]]:
    """Return (name, command) pairs for completion including aliases and custom commands."""
    completions: List[Tuple[str, Union[SlashCommand, CustomCommandDefinition]]] = []

    # Add built-in commands
    completions.extend(list(COMMAND_REGISTRY.items()))

    # Add custom commands
    custom_cmds = _get_custom_commands(project_path)
    for cmd in custom_cmds:
        completions.append((cmd.name, cmd))

    return completions


__all__ = [
    "SlashCommand",
    "CustomCommandDefinition",
    "list_slash_commands",
    "list_custom_commands",
    "get_slash_command",
    "get_custom_command",
    "slash_command_completions",
    "refresh_custom_commands",
    "expand_command_content",
]
