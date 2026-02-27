"""Tests for top-level `ripperdoc mcp` subcommands."""

from __future__ import annotations

import json

from click.testing import CliRunner

from ripperdoc.cli import cli as cli_module


def _run_cli(tmp_path, args: list[str]):
    runner = CliRunner()
    return runner.invoke(cli_module.cli, args, env={"HOME": str(tmp_path)})


def test_mcp_group_help_renders(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = _run_cli(tmp_path, ["mcp"])
    assert result.exit_code == 0
    assert "Configure and manage MCP servers" in result.output
    assert "reset-project-choices" in result.output


def test_mcp_add_list_get_remove_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    add_result = _run_cli(
        tmp_path,
        [
            "mcp",
            "add",
            "--scope",
            "project",
            "-e",
            "API_KEY=secret",
            "context7",
            "--",
            "npx",
            "-y",
            "@upstash/context7-mcp",
        ],
    )
    assert add_result.exit_code == 0
    assert "Added MCP server 'context7'" in add_result.output

    list_result = _run_cli(tmp_path, ["mcp", "list", "--json"])
    assert list_result.exit_code == 0
    rows = json.loads(list_result.output)
    assert len(rows) == 1
    assert rows[0]["name"] == "context7"
    assert rows[0]["transport"] == "stdio"
    assert rows[0]["scope"] == "project"

    get_result = _run_cli(tmp_path, ["mcp", "get", "context7", "--json"])
    assert get_result.exit_code == 0
    payload = json.loads(get_result.output)
    assert payload["name"] == "context7"
    assert payload["config"]["command"] == "npx"
    assert payload["config"]["args"] == ["-y", "@upstash/context7-mcp"]
    assert payload["config"]["env"]["API_KEY"] == "secret"

    remove_result = _run_cli(tmp_path, ["mcp", "remove", "--scope", "project", "context7"])
    assert remove_result.exit_code == 0
    assert "Removed MCP server 'context7'" in remove_result.output

    list_after_remove = _run_cli(tmp_path, ["mcp", "list", "--json"])
    assert list_after_remove.exit_code == 0
    assert json.loads(list_after_remove.output) == []


def test_mcp_add_json_preserves_mcpservers_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target_path = tmp_path / ".ripperdoc" / "mcp.json"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text('{"mcpServers":{}}\n', encoding="utf-8")

    result = _run_cli(
        tmp_path,
        [
            "mcp",
            "add-json",
            "--scope",
            "project",
            "acme-http",
            '{"type":"http","url":"https://example.com/mcp","headers":{"Authorization":"Bearer abc"}}',
        ],
    )
    assert result.exit_code == 0
    saved = json.loads(target_path.read_text(encoding="utf-8"))
    assert "servers" not in saved
    assert "mcpServers" in saved
    assert saved["mcpServers"]["acme-http"]["type"] == "http"

