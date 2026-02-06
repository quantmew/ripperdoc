"""Rich-based CLI interface for Ripperdoc."""

from __future__ import annotations

from typing import List, Optional
from pathlib import Path

from ripperdoc.cli.commands import slash_command_completions

from ripperdoc.cli.ui.rich_ui.commands import suggest_slash_commands
from ripperdoc.cli.ui.rich_ui.session import RichUI, check_onboarding_rich, main_rich


def suggest_slash_command_matches(name: str, project_path: Optional[Path]) -> List[str]:
    """Return close matching slash commands for a mistyped name."""
    def _completion_provider(path: Optional[Path]) -> List[tuple[str, object]]:
        return [(command_name, command_def) for command_name, command_def in slash_command_completions(path)]

    return suggest_slash_commands(name, project_path, _completion_provider)


__all__ = [
    "RichUI",
    "check_onboarding_rich",
    "main_rich",
    "suggest_slash_command_matches",
    "slash_command_completions",
]
