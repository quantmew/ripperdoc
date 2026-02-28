"""Tests for top-level `ripperdoc agents` command."""

from __future__ import annotations

from click.testing import CliRunner

from ripperdoc.cli import cli as cli_module
from ripperdoc.core.agents import AgentDefinition, AgentLoadResult, AgentLocation


def _run_cli(tmp_path, args: list[str]):
    runner = CliRunner()
    return runner.invoke(cli_module.cli, args, env={"HOME": str(tmp_path)})


def _agent(name: str, location: AgentLocation, model: str | None = None) -> AgentDefinition:
    return AgentDefinition(
        agent_type=name,
        when_to_use="test",
        tools=["*"],
        system_prompt="test",
        location=location,
        model=model,
    )


def test_agents_help_renders(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = _run_cli(tmp_path, ["agents", "--help"])
    assert result.exit_code == 0
    assert "List configured agents" in result.output
    assert "--setting-sources" in result.output
    assert "setting sources to load" in result.output
    assert "(user," in result.output
    assert "project, local)." in result.output


def test_agents_lists_grouped_entries(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "ripperdoc.cli.agents_cli.load_agent_definitions",
        lambda project_path=None: AgentLoadResult(
            active_agents=[
                _agent("plugin:lint", AgentLocation.PLUGIN, "opus"),
                _agent("explore", AgentLocation.BUILT_IN, "haiku"),
                _agent("general-purpose", AgentLocation.BUILT_IN, None),
                _agent("user-helper", AgentLocation.USER, "main"),
                _agent("project-helper", AgentLocation.PROJECT, None),
            ],
            all_agents=[],
            failed_files=[],
        ),
    )

    result = _run_cli(tmp_path, ["agents"])
    assert result.exit_code == 0
    assert "5 active agents" in result.output
    assert "Plugin agents:" in result.output
    assert "Built-in agents:" in result.output
    assert "User agents:" in result.output
    assert "Project agents:" in result.output
    assert "  plugin:lint · opus" in result.output
    assert "  Explore · haiku" in result.output
    assert "  general-purpose · inherit" in result.output
    assert "  user-helper · main" in result.output
    assert "  project-helper · inherit" in result.output


def test_agents_filters_by_setting_sources(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "ripperdoc.cli.agents_cli.load_agent_definitions",
        lambda project_path=None: AgentLoadResult(
            active_agents=[
                _agent("general-purpose", AgentLocation.BUILT_IN, None),
                _agent("user-helper", AgentLocation.USER, "main"),
                _agent("project-helper", AgentLocation.PROJECT, None),
            ],
            all_agents=[],
            failed_files=[],
        ),
    )

    result = _run_cli(tmp_path, ["agents", "--setting-sources", "user"])
    assert result.exit_code == 0
    assert "2 active agents" in result.output
    assert "  general-purpose · inherit" in result.output
    assert "  user-helper · main" in result.output
    assert "project-helper" not in result.output


def test_agents_rejects_invalid_setting_sources(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = _run_cli(tmp_path, ["agents", "--setting-sources", "workspace"])
    assert result.exit_code != 0
    assert "Unsupported --setting-sources value(s): workspace" in result.output
