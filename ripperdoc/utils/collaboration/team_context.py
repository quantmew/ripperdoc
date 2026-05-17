"""Team context resolution helpers shared across team tools.

Team context resolution utilities for swarm-style multi-agent coordination.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

from ripperdoc.core.tool import ToolUseContext
from ripperdoc.utils.collaboration.teams import get_active_team_name, list_teams


_ACTIVE_TEAM_BY_AGENT: Dict[str, str] = {}


def context_key(context: ToolUseContext) -> str:
    return (context.agent_id or "__default__").strip() or "__default__"


def remember_active_team(context: ToolUseContext, team_name: str) -> None:
    _ACTIVE_TEAM_BY_AGENT[context_key(context)] = team_name


def resolve_active_team_name(
    context: ToolUseContext, *, allow_single_team_fallback: bool = True
) -> Optional[str]:
    context_team = (context.team_name or "").strip()
    if context_team:
        return context_team

    key = context_key(context)
    if key in _ACTIVE_TEAM_BY_AGENT:
        return _ACTIVE_TEAM_BY_AGENT[key]

    env_team = os.getenv("RIPPERDOC_TEAM_NAME")
    if env_team and env_team.strip():
        return env_team.strip()

    disk_team = get_active_team_name()
    if disk_team:
        return disk_team

    if allow_single_team_fallback:
        teams = list_teams()
        if len(teams) == 1:
            return teams[0].name
    return None


def sender_name(context: ToolUseContext, *, team_lead_name: str = "team-lead") -> str:
    teammate_name = (context.teammate_name or "").strip()
    if teammate_name:
        return teammate_name
    agent_id = (context.agent_id or "").strip()
    if agent_id:
        return agent_id
    return team_lead_name


def clear_agent_active_team(context: ToolUseContext) -> None:
    _ACTIVE_TEAM_BY_AGENT.pop(context_key(context), None)
