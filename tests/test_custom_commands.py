"""Tests for custom command loading and expansion."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Generator

import pytest

from ripperdoc.core.custom_commands import (
    CommandLocation,
    load_all_custom_commands,
    find_custom_command,
    expand_command_content,
    _split_frontmatter,
    _derive_command_name,
    _normalize_allowed_tools,
)


@pytest.fixture
def temp_project() -> Generator[Path, None, None]:
    """Create a temporary project directory with custom commands."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir)
        commands_dir = project_path / ".ripperdoc" / "commands"
        commands_dir.mkdir(parents=True)

        # Create a simple command
        simple_cmd = commands_dir / "simple.md"
        simple_cmd.write_text("This is a simple command")

        # Create a command with frontmatter
        full_cmd = commands_dir / "full.md"
        full_cmd.write_text(
            """---
description: Full featured command
allowed-tools: Bash(git:*), Read
argument-hint: [filename]
model: gpt-4
---
Process file $ARGUMENTS with these tools"""
        )

        # Create a nested command
        nested_dir = commands_dir / "git"
        nested_dir.mkdir()
        nested_cmd = nested_dir / "commit.md"
        nested_cmd.write_text(
            """---
description: Create a git commit
allowed-tools: Bash(git add:*), Bash(git commit:*)
---
Create a commit with message: $1"""
        )

        # Create a command with bash expansion
        bash_cmd = commands_dir / "status.md"
        bash_cmd.write_text(
            """---
description: Show git status
---
Current status: !`echo "test output"`"""
        )

        yield project_path


@pytest.fixture
def temp_home() -> Generator[Path, None, None]:
    """Create a temporary home directory with global commands."""
    with tempfile.TemporaryDirectory() as tmpdir:
        home_path = Path(tmpdir)
        commands_dir = home_path / ".ripperdoc" / "commands"
        commands_dir.mkdir(parents=True)

        # Create a global command
        global_cmd = commands_dir / "global.md"
        global_cmd.write_text(
            """---
description: Global helper command
---
This is a global command"""
        )

        yield home_path


class TestFrontmatterParsing:
    """Tests for YAML frontmatter parsing."""

    def test_split_frontmatter_with_valid_yaml(self) -> None:
        text = """---
name: test
description: A test
---
Body content here"""
        frontmatter, body = _split_frontmatter(text)
        assert frontmatter == {"name": "test", "description": "A test"}
        assert body.strip() == "Body content here"

    def test_split_frontmatter_without_frontmatter(self) -> None:
        text = "Just body content\nNo frontmatter"
        frontmatter, body = _split_frontmatter(text)
        assert frontmatter == {}
        assert body == text

    def test_split_frontmatter_with_invalid_yaml(self) -> None:
        text = """---
invalid: yaml: content
---
Body"""
        frontmatter, body = _split_frontmatter(text)
        assert "__error__" in frontmatter


class TestCommandNameDerivation:
    """Tests for deriving command names from file paths."""

    def test_simple_command_name(self, temp_project: Path) -> None:
        commands_dir = temp_project / ".ripperdoc" / "commands"
        path = commands_dir / "simple.md"
        name = _derive_command_name(path, commands_dir)
        assert name == "simple"

    def test_nested_command_name(self, temp_project: Path) -> None:
        commands_dir = temp_project / ".ripperdoc" / "commands"
        path = commands_dir / "git" / "commit.md"
        name = _derive_command_name(path, commands_dir)
        assert name == "git:commit"


class TestAllowedToolsNormalization:
    """Tests for normalizing allowed-tools values."""

    def test_normalize_string_list(self) -> None:
        result = _normalize_allowed_tools("Bash(git:*), Read, Write")
        assert result == ["Bash(git:*)", "Read", "Write"]

    def test_normalize_list(self) -> None:
        result = _normalize_allowed_tools(["Bash(git:*)", "Read"])
        assert result == ["Bash(git:*)", "Read"]

    def test_normalize_none(self) -> None:
        result = _normalize_allowed_tools(None)
        assert result == []

    def test_normalize_empty_string(self) -> None:
        result = _normalize_allowed_tools("")
        assert result == []


class TestLoadCustomCommands:
    """Tests for loading custom commands."""

    def test_load_project_commands(self, temp_project: Path) -> None:
        result = load_all_custom_commands(project_path=temp_project, home=temp_project)
        assert len(result.commands) >= 3  # simple, full, git:commit, status

    def test_load_global_commands(self, temp_home: Path, temp_project: Path) -> None:
        result = load_all_custom_commands(project_path=temp_project, home=temp_home)
        assert any(cmd.name == "global" for cmd in result.commands)
        global_cmd = next(cmd for cmd in result.commands if cmd.name == "global")
        assert global_cmd.location == CommandLocation.USER

    def test_project_overrides_global(self, temp_project: Path, temp_home: Path) -> None:
        # Create same-named command in both locations
        project_cmd = temp_project / ".ripperdoc" / "commands" / "global.md"
        project_cmd.write_text(
            """---
description: Project version of global
---
Project global command"""
        )

        result = load_all_custom_commands(project_path=temp_project, home=temp_home)
        global_cmd = next(cmd for cmd in result.commands if cmd.name == "global")
        assert global_cmd.location == CommandLocation.PROJECT
        assert "Project version" in global_cmd.description


class TestFindCustomCommand:
    """Tests for finding a specific custom command."""

    def test_find_existing_command(self, temp_project: Path) -> None:
        cmd = find_custom_command("simple", project_path=temp_project, home=temp_project)
        assert cmd is not None
        assert cmd.name == "simple"

    def test_find_nested_command(self, temp_project: Path) -> None:
        cmd = find_custom_command("git:commit", project_path=temp_project, home=temp_project)
        assert cmd is not None
        assert cmd.name == "git:commit"

    def test_find_nonexistent_command(self, temp_project: Path) -> None:
        cmd = find_custom_command("nonexistent", project_path=temp_project, home=temp_project)
        assert cmd is None


class TestExpandCommandContent:
    """Tests for command content expansion."""

    def test_expand_arguments(self, temp_project: Path) -> None:
        cmd = find_custom_command("full", project_path=temp_project, home=temp_project)
        assert cmd is not None
        expanded = expand_command_content(cmd, "test.txt", temp_project)
        assert "test.txt" in expanded

    def test_expand_positional_arguments(self, temp_project: Path) -> None:
        cmd = find_custom_command("git:commit", project_path=temp_project, home=temp_project)
        assert cmd is not None
        expanded = expand_command_content(cmd, "Initial commit", temp_project)
        assert "Initial" in expanded

    def test_expand_bash_command(self, temp_project: Path) -> None:
        cmd = find_custom_command("status", project_path=temp_project, home=temp_project)
        assert cmd is not None
        expanded = expand_command_content(cmd, "", temp_project)
        assert "test output" in expanded

    def test_expand_file_reference(self, temp_project: Path) -> None:
        # Create a file to reference
        test_file = temp_project / "test.txt"
        test_file.write_text("Test file content")

        # Create a command that references the file
        commands_dir = temp_project / ".ripperdoc" / "commands"
        ref_cmd = commands_dir / "ref.md"
        ref_cmd.write_text(
            """---
description: Reference test
---
Content from file: @test.txt"""
        )

        cmd = find_custom_command("ref", project_path=temp_project, home=temp_project)
        assert cmd is not None
        expanded = expand_command_content(cmd, "", temp_project)
        assert "Test file content" in expanded


class TestCommandDefinition:
    """Tests for CustomCommandDefinition dataclass."""

    def test_command_with_full_metadata(self, temp_project: Path) -> None:
        cmd = find_custom_command("full", project_path=temp_project, home=temp_project)
        assert cmd is not None
        assert cmd.description == "Full featured command"
        assert "Bash(git:*)" in cmd.allowed_tools
        assert cmd.argument_hint is not None
        assert "filename" in cmd.argument_hint  # YAML may parse [filename] as list
        assert cmd.model == "gpt-4"

    def test_command_without_frontmatter(self, temp_project: Path) -> None:
        cmd = find_custom_command("simple", project_path=temp_project, home=temp_project)
        assert cmd is not None
        assert cmd.description  # Should have auto-generated description
        assert cmd.allowed_tools == []
        assert cmd.argument_hint is None
