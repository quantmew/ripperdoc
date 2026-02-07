from __future__ import annotations

import json
from pathlib import Path

import pytest

from ripperdoc.core.agents import AgentLocation, load_agent_definitions
from ripperdoc.core.custom_commands import CommandLocation, load_all_custom_commands
from ripperdoc.core.hooks.config import get_merged_hooks_config
from ripperdoc.core.hooks.events import HookEvent
from ripperdoc.core.plugins import (
    PluginSettingsScope,
    add_enabled_plugin_for_scope,
    clear_runtime_plugin_dirs,
    discover_plugins,
    list_enabled_plugin_entries_for_scope,
    remove_enabled_plugin_for_scope,
    set_runtime_plugin_dirs,
)
from ripperdoc.core.skills import SkillLocation, load_all_skills
from ripperdoc.utils.lsp import load_lsp_server_configs
from ripperdoc.utils.mcp import _load_server_configs


@pytest.fixture(autouse=True)
def _reset_runtime_plugin_dirs() -> None:
    clear_runtime_plugin_dirs()
    yield
    clear_runtime_plugin_dirs()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _create_plugin(plugin_dir: Path, plugin_name: str = "acme") -> Path:
    _write_json(
        plugin_dir / ".ripperdoc-plugin" / "plugin.json",
        {
            "name": plugin_name,
            "description": "Test plugin",
            "version": "1.0.0",
        },
    )
    (plugin_dir / "commands").mkdir(parents=True, exist_ok=True)
    (plugin_dir / "commands" / "hello.md").write_text(
        "---\n" "description: Hello command\n" "---\n" "Say hello to $ARGUMENTS\n",
        encoding="utf-8",
    )
    (plugin_dir / "skills" / "hello").mkdir(parents=True, exist_ok=True)
    # name intentionally omitted to validate directory-name fallback behavior.
    (plugin_dir / "skills" / "hello" / "SKILL.md").write_text(
        "---\n" "description: Hello skill\n" "---\n" "Use this skill when greeting users.\n",
        encoding="utf-8",
    )
    (plugin_dir / "agents").mkdir(parents=True, exist_ok=True)
    (plugin_dir / "agents" / "reviewer.md").write_text(
        "---\n"
        "name: reviewer\n"
        "description: Reviews code\n"
        "tools: Read, Grep\n"
        "---\n"
        "Review code for issues.\n",
        encoding="utf-8",
    )
    _write_json(
        plugin_dir / "hooks" / "hooks.json",
        {
            "hooks": {
                "PreToolUse": [
                    {
                        "matcher": "Write",
                        "hooks": [
                            {
                                "type": "command",
                                "command": "${RIPPERDOC_PLUGIN_ROOT}/scripts/check.sh",
                            }
                        ],
                    }
                ]
            }
        },
    )
    _write_json(
        plugin_dir / ".mcp.json",
        {
            "mcpServers": {
                "plugin-db": {
                    "command": "${CLAUDE_PLUGIN_ROOT}/bin/db-server",
                    "args": ["--mode", "test"],
                }
            }
        },
    )
    _write_json(
        plugin_dir / ".lsp.json",
        {
            "go": {
                "command": "gopls",
                "args": ["serve"],
                "extensionToLanguage": {".go": "go"},
            }
        },
    )
    return plugin_dir


def test_plugin_components_are_discovered_with_namespace(tmp_path: Path, monkeypatch) -> None:
    home_dir = tmp_path / "home"
    project_dir = tmp_path / "project"
    home_dir.mkdir(parents=True, exist_ok=True)
    project_dir.mkdir(parents=True, exist_ok=True)
    plugin_dir = _create_plugin(project_dir / "plugins" / "acme-plugin", "acme")

    _write_json(
        project_dir / ".ripperdoc" / "plugins.json",
        {"enabledPlugins": ["./plugins/acme-plugin"]},
    )

    command_result = load_all_custom_commands(project_path=project_dir, home=home_dir)
    commands = {cmd.name: cmd for cmd in command_result.commands}
    assert "acme:hello" in commands
    assert commands["acme:hello"].location == CommandLocation.PLUGIN
    assert commands["acme:hello"].plugin_name == "acme"

    skill_result = load_all_skills(project_path=project_dir, home=home_dir)
    skills = {skill.name: skill for skill in skill_result.skills}
    assert "acme:hello" in skills
    assert skills["acme:hello"].location == SkillLocation.PLUGIN
    assert skills["acme:hello"].plugin_name == "acme"

    agent_result = load_agent_definitions(project_path=project_dir, home=home_dir)
    plugin_agent = next(
        (a for a in agent_result.active_agents if a.agent_type == "acme:reviewer"), None
    )
    assert plugin_agent is not None
    assert plugin_agent.location == AgentLocation.PLUGIN
    assert plugin_agent.plugin_name == "acme"

    monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: home_dir)
    monkeypatch.setattr("ripperdoc.utils.mcp.Path.home", lambda: home_dir)
    monkeypatch.setattr("ripperdoc.utils.lsp.Path.home", lambda: home_dir)

    hooks = get_merged_hooks_config(project_path=project_dir)
    pre_tool_hooks = hooks.get_hooks_for_event(HookEvent.PRE_TOOL_USE, "Write")
    assert pre_tool_hooks
    assert pre_tool_hooks[0].command == f"{plugin_dir.resolve()}/scripts/check.sh"

    mcp_configs = _load_server_configs(project_dir)
    assert "plugin-db" in mcp_configs
    assert mcp_configs["plugin-db"].command == f"{plugin_dir.resolve()}/bin/db-server"

    lsp_configs = load_lsp_server_configs(project_dir)
    assert "go" in lsp_configs
    assert lsp_configs["go"].command == "gopls"


def test_runtime_plugin_dir_works_without_settings(tmp_path: Path) -> None:
    home_dir = tmp_path / "home"
    project_dir = tmp_path / "project"
    home_dir.mkdir(parents=True, exist_ok=True)
    project_dir.mkdir(parents=True, exist_ok=True)
    plugin_dir = _create_plugin(project_dir / "plugins" / "dev-plugin", "dev")

    clear_runtime_plugin_dirs()
    set_runtime_plugin_dirs([plugin_dir], base_dir=project_dir)
    try:
        command_result = load_all_custom_commands(project_path=project_dir, home=home_dir)
        names = {cmd.name for cmd in command_result.commands}
        assert "dev:hello" in names
    finally:
        clear_runtime_plugin_dirs()


def test_plugin_settings_add_and_remove(tmp_path: Path, monkeypatch) -> None:
    home_dir = tmp_path / "home"
    project_dir = tmp_path / "project"
    home_dir.mkdir(parents=True, exist_ok=True)
    project_dir.mkdir(parents=True, exist_ok=True)
    plugin_dir = _create_plugin(project_dir / "plugins" / "ops-plugin", "ops")

    settings_path = add_enabled_plugin_for_scope(
        plugin_dir,
        scope=PluginSettingsScope.PROJECT,
        project_path=project_dir,
        home=home_dir,
    )
    assert settings_path.exists()
    entries = list_enabled_plugin_entries_for_scope(
        PluginSettingsScope.PROJECT,
        project_path=project_dir,
        home=home_dir,
    )
    assert entries == ["./plugins/ops-plugin"]

    monkeypatch.setattr("ripperdoc.core.plugins.Path.home", lambda: home_dir)
    discovered = discover_plugins(project_path=project_dir, home=home_dir).plugins
    assert any(plugin.name == "ops" for plugin in discovered)

    _, removed = remove_enabled_plugin_for_scope(
        plugin_dir,
        scope=PluginSettingsScope.PROJECT,
        project_path=project_dir,
        home=home_dir,
    )
    assert removed is True
    entries_after = list_enabled_plugin_entries_for_scope(
        PluginSettingsScope.PROJECT,
        project_path=project_dir,
        home=home_dir,
    )
    assert entries_after == []
