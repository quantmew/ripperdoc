"""Tests for /themes command."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from ripperdoc.cli.commands.themes_cmd import command as themes_command
from ripperdoc.core.config import config_manager, get_global_config
from ripperdoc.core.theme import get_theme_manager


class _DummyUI:
    def __init__(self, console: Console, project_path: Path):
        self.console = console
        self.project_path = project_path


def test_themes_command_switch_shows_restart_notice(tmp_path: Path) -> None:
    original_global_path = config_manager.global_config_path
    original_global = config_manager._global_config
    original_project = config_manager._project_config
    original_project_local = config_manager._project_local_config
    manager = get_theme_manager()
    original_theme = manager.current.name

    try:
        config_manager.global_config_path = tmp_path / "global.json"
        config_manager._global_config = None
        config_manager._project_config = None
        config_manager._project_local_config = None
        manager.set_theme("dark")

        ui = _DummyUI(Console(record=True, width=120), tmp_path)
        themes_command.handler(ui, "light")

        output = ui.console.export_text()
        assert "Theme switched to Light" in output
        assert "Restart Ripperdoc" in output

        config = get_global_config()
        assert config.theme == "light"
    finally:
        manager.set_theme(original_theme)
        config_manager.global_config_path = original_global_path
        config_manager._global_config = original_global
        config_manager._project_config = original_project
        config_manager._project_local_config = original_project_local


def test_themes_command_preview_does_not_show_restart_notice(tmp_path: Path) -> None:
    manager = get_theme_manager()
    original_theme = manager.current.name
    try:
        manager.set_theme("dark")
        ui = _DummyUI(Console(record=True, width=120), tmp_path)
        themes_command.handler(ui, "preview light")

        output = ui.console.export_text()
        assert "Theme Preview" in output
        assert "Restart Ripperdoc" not in output
    finally:
        manager.set_theme(original_theme)

