"""Tests for MCP TUI config helpers."""

from __future__ import annotations

import json

import pytest

from ripperdoc.cli.ui.mcp_tui.textual_app import AddServerDraft, _find_edit_draft, _save_mcp_server


def _draft(*, name: str = "context7", transport: str = "stdio") -> AddServerDraft:
    return AddServerDraft(
        name=name,
        scope="project",
        transport=transport,
        command="npx",
        args=["-y", "@upstash/context7-mcp"],
        url="",
        description="test server",
    )


def test_save_mcp_server_creates_project_config(tmp_path) -> None:
    path = tmp_path / ".ripperdoc" / "mcp.json"
    result = _save_mcp_server(path, _draft(), overwrite=False)

    assert result.path == path
    assert result.updated is False

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert "servers" in saved
    assert saved["servers"]["context7"]["command"] == "npx"
    assert saved["servers"]["context7"]["args"] == ["-y", "@upstash/context7-mcp"]


def test_save_mcp_server_preserves_mcpservers_key(tmp_path) -> None:
    path = tmp_path / ".ripperdoc" / "mcp.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"mcpServers":{"old":{"command":"node"}}}\n', encoding="utf-8")

    _save_mcp_server(path, _draft(name="new-server"), overwrite=False)
    saved = json.loads(path.read_text(encoding="utf-8"))

    assert "mcpServers" in saved
    assert "servers" not in saved
    assert saved["mcpServers"]["old"]["command"] == "node"
    assert saved["mcpServers"]["new-server"]["command"] == "npx"


def test_save_mcp_server_requires_overwrite_for_existing_name(tmp_path) -> None:
    path = tmp_path / ".ripperdoc" / "mcp.json"
    _save_mcp_server(path, _draft(), overwrite=False)

    with pytest.raises(FileExistsError):
        _save_mcp_server(path, _draft(), overwrite=False)


def test_find_edit_draft_reads_existing_project_server(tmp_path) -> None:
    path = tmp_path / ".ripperdoc" / "mcp.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"servers":{"context7":{"command":"npx","args":["-y","@upstash/context7-mcp"]}}}\n',
        encoding="utf-8",
    )

    draft = _find_edit_draft(tmp_path, "context7")
    assert draft is not None
    assert draft.name == "context7"
    assert draft.scope == "project"
    assert draft.transport == "stdio"
    assert draft.command == "npx"
    assert draft.args == ["-y", "@upstash/context7-mcp"]
    assert draft.target_path == path


def test_find_edit_draft_reads_top_level_user_server(tmp_path, monkeypatch) -> None:
    home = tmp_path / "home"
    monkeypatch.setattr("pathlib.Path.home", lambda: home)
    path = home / ".ripperdoc" / "mcp.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"agent-browser":{"command":"npx","args":["-y","github:quantmew/agent-browser-mcp"]}}\n',
        encoding="utf-8",
    )

    draft = _find_edit_draft(tmp_path / "repo", "agent-browser")
    assert draft is not None
    assert draft.name == "agent-browser"
    assert draft.scope == "user"
    assert draft.transport == "stdio"
    assert draft.command == "npx"
    assert draft.args == ["-y", "github:quantmew/agent-browser-mcp"]
    assert draft.target_path == path
