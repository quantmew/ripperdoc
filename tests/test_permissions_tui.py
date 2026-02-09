"""Tests for permissions TUI keyboard behavior."""

from pathlib import Path

from ripperdoc.cli.ui.permissions_tui.textual_app import PermissionsApp


class _FakeKeyEvent:
    def __init__(self, key: str, character: str | None = None) -> None:
        self.key = key
        self.character = character
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


def test_escape_clears_search_when_query_exists(monkeypatch) -> None:
    app = PermissionsApp(Path("."))
    app._search_query = "Read("

    updates: list[str] = []
    monkeypatch.setattr(app, "_update_search", lambda query: updates.append(query))

    exited = {"value": False}
    monkeypatch.setattr(app, "exit", lambda *args, **kwargs: exited.__setitem__("value", True))

    event = _FakeKeyEvent("escape")
    app.on_key(event)  # type: ignore[arg-type]

    assert updates == [""]
    assert exited["value"] is False
    assert event.stopped is True


def test_escape_exits_when_search_query_is_empty(monkeypatch) -> None:
    app = PermissionsApp(Path("."))
    app._search_query = ""

    exited = {"value": False}
    monkeypatch.setattr(app, "exit", lambda *args, **kwargs: exited.__setitem__("value", True))

    event = _FakeKeyEvent("escape")
    app.on_key(event)  # type: ignore[arg-type]

    assert exited["value"] is True
    assert event.stopped is True
