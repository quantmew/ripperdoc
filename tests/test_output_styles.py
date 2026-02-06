"""Tests for output style loading and resolution."""

from __future__ import annotations

from pathlib import Path

from ripperdoc.core.output_styles import (
    OutputStyleLocation,
    find_output_style,
    load_all_output_styles,
    resolve_output_style,
)


def test_load_output_styles_includes_builtins(tmp_path: Path) -> None:
    result = load_all_output_styles(project_path=tmp_path, home=tmp_path)
    keys = {style.key for style in result.styles}
    assert {"default", "explanatory", "learning"}.issubset(keys)


def test_project_output_style_overrides_user(tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    user_dir = home / ".ripperdoc" / "output-styles"
    project_dir = project / ".ripperdoc" / "output-styles"
    user_dir.mkdir(parents=True)
    project_dir.mkdir(parents=True)

    (user_dir / "writer.md").write_text(
        """---
name: Writer
description: user version
keep-coding-instructions: false
---
User instructions
""",
        encoding="utf-8",
    )
    (project_dir / "writer.md").write_text(
        """---
name: Writer
description: project version
keep-coding-instructions: true
---
Project instructions
""",
        encoding="utf-8",
    )

    style, result = find_output_style("writer", project_path=project, home=home)
    assert style is not None
    assert style.location == OutputStyleLocation.PROJECT
    assert style.instructions == "Project instructions"
    assert style.keep_coding_instructions is True
    assert not result.errors


def test_resolve_output_style_falls_back_to_default(tmp_path: Path) -> None:
    style, _ = resolve_output_style("not-found-style", project_path=tmp_path, home=tmp_path)
    assert style.key == "default"


def test_find_output_style_by_display_name(tmp_path: Path) -> None:
    styles_dir = tmp_path / ".ripperdoc" / "output-styles"
    styles_dir.mkdir(parents=True)
    (styles_dir / "student-mode.md").write_text(
        """---
name: Student Mode
description: test
---
Use student mode.
""",
        encoding="utf-8",
    )

    style, _ = find_output_style("student mode", project_path=tmp_path, home=tmp_path)
    assert style is not None
    assert style.key == "student-mode"
