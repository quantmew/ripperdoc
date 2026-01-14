"""Tests for fuzzy slash command suggestions."""

from ripperdoc.cli.ui import rich_ui


def test_suggest_slash_commands_returns_close_matches(monkeypatch):
    """Mistyped commands should return close matches for hints."""

    def _fake_completions(_project_path=None):
        return [
            ("tasks", None),
            ("todos", None),
            ("tools", None),
        ]

    monkeypatch.setattr(rich_ui, "slash_command_completions", _fake_completions)

    suggestions = rich_ui._suggest_slash_commands("tsaks", None)

    assert suggestions and suggestions[0] == "tasks"
