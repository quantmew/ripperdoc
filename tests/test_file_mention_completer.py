"""Tests for FileMentionCompleter path navigation."""

from prompt_toolkit.document import Document

from ripperdoc.cli.ui.file_mention_completer import FileMentionCompleter
from ripperdoc.utils.path_ignore import build_ignore_filter


def _get_text_suggestions(completer, text: str):
    doc = Document(text=text, cursor_position=len(text))
    return list(completer.get_completions(doc, None))


def test_top_level_directory_completion(tmp_path, monkeypatch):
    """Typing a prefix should suggest top-level directories for stepwise navigation."""
    (tmp_path / "ripperdoc" / "cli").mkdir(parents=True)
    (tmp_path / "ripperdoc" / "__init__.py").write_text("")
    (tmp_path / "README.md").write_text("")

    ignore_filter = build_ignore_filter(tmp_path, include_defaults=False, include_gitignore=False)
    completer = FileMentionCompleter(tmp_path, ignore_filter)

    suggestions = _get_text_suggestions(completer, "@ripp")
    texts = [s.text for s in suggestions]
    assert "ripperdoc/" in texts


def test_nested_directory_completion(tmp_path, monkeypatch):
    """After completing a directory segment, show children in that directory."""
    (tmp_path / "ripperdoc" / "cli").mkdir(parents=True)
    (tmp_path / "ripperdoc" / "cli" / "__init__.py").write_text("")
    (tmp_path / "ripperdoc" / "cli" / "cli.py").write_text("")

    ignore_filter = build_ignore_filter(tmp_path, include_defaults=False, include_gitignore=False)
    completer = FileMentionCompleter(tmp_path, ignore_filter)

    suggestions = _get_text_suggestions(completer, "@ripperdoc/cli/")
    texts = [s.text for s in suggestions]
    assert "ripperdoc/cli/cli.py" in texts
    assert "ripperdoc/cli/__init__.py" in texts


def test_directory_name_also_surfaces_children(tmp_path, monkeypatch):
    """Typing a directory name without slash should still show its children."""
    (tmp_path / "ripperdoc" / "cli").mkdir(parents=True)
    (tmp_path / "ripperdoc" / "__init__.py").write_text("")
    (tmp_path / "ripperdoc" / "cli" / "__init__.py").write_text("")

    ignore_filter = build_ignore_filter(tmp_path, include_defaults=False, include_gitignore=False)
    completer = FileMentionCompleter(tmp_path, ignore_filter)

    suggestions = _get_text_suggestions(completer, "@ripperdoc")
    texts = [s.text for s in suggestions]
    assert "ripperdoc/cli/" in texts
    assert "ripperdoc/__init__.py" in texts


def test_completion_keeps_at_until_accept(tmp_path, monkeypatch):
    """Completion should not replace the @ marker while browsing suggestions."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")

    ignore_filter = build_ignore_filter(tmp_path, include_defaults=False, include_gitignore=False)
    completer = FileMentionCompleter(tmp_path, ignore_filter)

    query = "@pkg"
    doc = Document(text=query, cursor_position=len(query))
    completions = list(completer.get_completions(doc, None))
    assert completions  # sanity check
    # The implementation uses start_position = -(len(query) + 1) where query is "pkg"
    # So start_position should be -4, not -3
    assert any(c.start_position == -4 for c in completions)
