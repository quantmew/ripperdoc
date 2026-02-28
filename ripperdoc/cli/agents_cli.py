"""Top-level `ripperdoc agents` command."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import click

from ripperdoc.core.agents import AgentDefinition, AgentLocation, load_agent_definitions

_SETTING_SOURCE_USER = "user"
_SETTING_SOURCE_PROJECT = "project"
_SETTING_SOURCE_LOCAL = "local"
_VALID_SETTING_SOURCES = (
    _SETTING_SOURCE_USER,
    _SETTING_SOURCE_PROJECT,
    _SETTING_SOURCE_LOCAL,
)


def _parse_setting_sources(raw_sources: Optional[str]) -> Optional[set[str]]:
    if raw_sources is None:
        return None

    values = [item.strip().lower() for item in raw_sources.split(",") if item.strip()]
    if not values:
        raise click.ClickException(
            "--setting-sources requires a comma-separated list: user,project,local."
        )

    invalid = sorted({value for value in values if value not in _VALID_SETTING_SOURCES})
    if invalid:
        invalid_text = ", ".join(invalid)
        valid_text = ", ".join(_VALID_SETTING_SOURCES)
        raise click.ClickException(
            f"Unsupported --setting-sources value(s): {invalid_text}. "
            f"Supported values: {valid_text}."
        )

    return set(values)


def _is_agent_enabled_for_sources(agent: AgentDefinition, setting_sources: Optional[set[str]]) -> bool:
    if setting_sources is None:
        return True
    if agent.location == AgentLocation.USER:
        return _SETTING_SOURCE_USER in setting_sources
    if agent.location == AgentLocation.PROJECT:
        return (
            _SETTING_SOURCE_PROJECT in setting_sources
            or _SETTING_SOURCE_LOCAL in setting_sources
        )
    return True


def _display_agent_name(agent_type: str) -> str:
    if agent_type in ("explore", "plan"):
        return agent_type.title()
    return agent_type


def _display_model_label(agent: AgentDefinition) -> str:
    model = (agent.model or "").strip()
    return model if model else "inherit"


def _sorted_agents(agents: Iterable[AgentDefinition]) -> list[AgentDefinition]:
    return sorted(agents, key=lambda item: item.agent_type.lower())


@click.command(name="agents", help="List configured agents")
@click.option(
    "--setting-sources",
    type=str,
    default=None,
    help="Comma-separated list of setting sources to load (user, project, local).",
)
def agents_cmd(setting_sources: Optional[str]) -> None:
    selected_sources = _parse_setting_sources(setting_sources)
    result = load_agent_definitions(project_path=Path.cwd())
    active_agents = [
        agent
        for agent in result.active_agents
        if _is_agent_enabled_for_sources(agent, selected_sources)
    ]

    click.echo(f"{len(active_agents)} active agents")
    if active_agents:
        click.echo("")

    grouped_agents = [
        (
            "Plugin agents",
            [agent for agent in active_agents if agent.location == AgentLocation.PLUGIN],
        ),
        (
            "Built-in agents",
            [agent for agent in active_agents if agent.location == AgentLocation.BUILT_IN],
        ),
        (
            "User agents",
            [agent for agent in active_agents if agent.location == AgentLocation.USER],
        ),
        (
            "Project agents",
            [agent for agent in active_agents if agent.location == AgentLocation.PROJECT],
        ),
    ]

    printed_sections = 0
    for title, agents in grouped_agents:
        if not agents:
            continue
        if printed_sections:
            click.echo("")
        click.echo(f"{title}:")
        for agent in _sorted_agents(agents):
            click.echo(
                f"  {_display_agent_name(agent.agent_type)} · {_display_model_label(agent)}"
            )
        printed_sections += 1

    if result.failed_files:
        click.echo("")
        click.echo("Warnings:")
        for path, reason in result.failed_files:
            click.echo(f"  - {path}: {reason}")


__all__ = ["agents_cmd"]
