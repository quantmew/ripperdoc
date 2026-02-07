"""Tests for /output-language command."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from ripperdoc.cli.commands.output_language_cmd import command as output_language_command
from ripperdoc.core.config import get_project_local_config


class _DummyUI:
    def __init__(self, console: Console, project_path: Path):
        self.console = console
        self.project_path = project_path
        self.output_language = "auto"

    def set_output_language(self, language: str) -> None:
        self.output_language = language


def test_output_language_command_sets_language(tmp_path: Path) -> None:
    ui = _DummyUI(Console(record=True, width=120), tmp_path)
    output_language_command.handler(ui, "Chinese")

    config = get_project_local_config(tmp_path)
    assert ui.output_language == "Chinese"
    assert config.output_language == "Chinese"


def test_output_language_command_resets_to_auto(tmp_path: Path) -> None:
    ui = _DummyUI(Console(record=True, width=120), tmp_path)
    output_language_command.handler(ui, "Chinese")
    output_language_command.handler(ui, "auto")

    config = get_project_local_config(tmp_path)
    assert ui.output_language == "auto"
    assert config.output_language == "auto"


def test_output_language_command_shows_current(tmp_path: Path) -> None:
    ui = _DummyUI(Console(record=True, width=120), tmp_path)
    output_language_command.handler(ui, "")

    output = ui.console.export_text()
    assert "Output language" in output
