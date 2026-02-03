"""Rich-based CLI interface for Ripperdoc."""

from __future__ import annotations

from typing import List, Optional
from pathlib import Path

from ripperdoc.cli.commands import slash_command_completions

from ripperdoc.cli.ui.rich_ui.commands import suggest_slash_commands
from ripperdoc.cli.ui.rich_ui.session import RichUI, check_onboarding_rich, main_rich


def _suggest_slash_commands(name: str, project_path: Optional[Path]) -> List[str]:
    """Return close matching slash commands for a mistyped name."""
    return suggest_slash_commands(name, project_path, slash_command_completions)


__all__ = [
    "RichUI",
    "check_onboarding_rich",
    "main_rich",
    "_suggest_slash_commands",
    "slash_command_completions",
]
