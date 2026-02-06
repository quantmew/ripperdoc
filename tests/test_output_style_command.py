"""Tests for /output-style command."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from ripperdoc.cli.commands.output_style_cmd import command as output_style_command
from ripperdoc.core.config import get_project_local_config


class _DummyUI:
    def __init__(self, console: Console, project_path: Path):
        self.console = console
        self.project_path = project_path
        self.output_style = "default"

    def set_output_style(self, style_key: str) -> None:
        self.output_style = style_key


def test_output_style_command_direct_switch(tmp_path: Path) -> None:
    ui = _DummyUI(Console(record=True, width=120), tmp_path)
    output_style_command.handler(ui, "explanatory")

    config = get_project_local_config(tmp_path)
    assert ui.output_style == "explanatory"
    assert config.output_style == "explanatory"


def test_output_style_command_menu_switch(tmp_path: Path, monkeypatch) -> None:
    ui = _DummyUI(Console(record=True, width=120), tmp_path)
    monkeypatch.setattr("ripperdoc.cli.commands.output_style_cmd.prompt_choice", lambda **_kwargs: "learning")

    output_style_command.handler(ui, "")

    config = get_project_local_config(tmp_path)
    assert ui.output_style == "learning"
    assert config.output_style == "learning"


def test_output_style_command_switches_to_custom_style(tmp_path: Path) -> None:
    styles_dir = tmp_path / ".ripperdoc" / "output-styles"
    styles_dir.mkdir(parents=True)
    (styles_dir / "reviewer.md").write_text(
        """---
name: Reviewer
description: review style
---
Focus on risk review.
""",
        encoding="utf-8",
    )

    ui = _DummyUI(Console(record=True, width=120), tmp_path)
    output_style_command.handler(ui, "reviewer")

    config = get_project_local_config(tmp_path)
    assert ui.output_style == "reviewer"
    assert config.output_style == "reviewer"


def test_output_style_command_unknown_style(tmp_path: Path) -> None:
    ui = _DummyUI(Console(record=True, width=120), tmp_path)
    output_style_command.handler(ui, "missing-style")

    output = ui.console.export_text()
    assert "Unknown output style" in output
    assert ui.output_style == "default"
