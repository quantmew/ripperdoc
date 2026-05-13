"""Helpers for session-scoped agent option parsing and resolution."""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Optional


SessionAgentSpec = Dict[str, str]


def normalize_agent_name(value: Any, *, source: str) -> Optional[str]:
    """Normalize an agent name from CLI/SDK settings."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{source} must be a string.")
    normalized = value.strip()
    return normalized or None


def parse_session_agents(raw_agents: Any, *, source: str) -> Dict[str, SessionAgentSpec]:
    """Parse custom session agents from JSON string/object payloads.

    Supports:
    - {"name": {"description": "...", "prompt": "..."}}
    - {"name": {"prompt": "..."}}
    - {"name": "prompt text"}
    """
    if raw_agents is None:
        return {}

    parsed_payload: Any = raw_agents
    if isinstance(raw_agents, str):
        payload = raw_agents.strip()
        if not payload:
            return {}
        try:
            parsed_payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source} must be valid JSON: {exc.msg}.") from exc

    if not isinstance(parsed_payload, Mapping):
        raise ValueError(f"{source} must be a JSON object mapping agent names to definitions.")

    parsed_agents: Dict[str, SessionAgentSpec] = {}
    for raw_name, raw_definition in parsed_payload.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError(f"{source} contains an invalid agent name: {raw_name!r}.")
        name = raw_name.strip()

        description = ""
        prompt: Any = None
        if isinstance(raw_definition, str):
            prompt = raw_definition
        elif isinstance(raw_definition, Mapping):
            prompt = (
                raw_definition.get("prompt")
                or raw_definition.get("system_prompt")
                or raw_definition.get("systemPrompt")
            )
            raw_description = raw_definition.get("description")
            if raw_description is not None and not isinstance(raw_description, str):
                raise ValueError(
                    f"{source}.{name}.description must be a string when provided."
                )
            description = str(raw_description or "").strip()
        else:
            raise ValueError(
                f"{source}.{name} must be a string or an object containing a prompt."
            )

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                f"{source}.{name}.prompt is required and must be a non-empty string."
            )

        parsed_agents[name] = {"prompt": prompt.strip(), "description": description}

    return parsed_agents


def merge_session_agents(
    *agent_layers: Mapping[str, SessionAgentSpec],
) -> Dict[str, SessionAgentSpec]:
    """Merge agent dictionaries in order, where later layers override earlier ones."""
    merged: Dict[str, SessionAgentSpec] = {}
    for layer in agent_layers:
        for name, definition in layer.items():
            merged[name] = dict(definition)
    return merged


def resolve_session_agent_prompt(
    agent_name: Optional[str],
    agents: Mapping[str, SessionAgentSpec],
    *,
    source: str = "agent",
) -> Optional[str]:
    """Resolve selected session agent prompt from a parsed agent map."""
    normalized = normalize_agent_name(agent_name, source=source)
    if normalized is None:
        return None

    agent_definition = agents.get(normalized)
    if agent_definition is None:
        available = ", ".join(sorted(agents)) if agents else "(none)"
        raise ValueError(f"Unknown agent '{normalized}'. Available agents: {available}.")

    prompt = str(agent_definition.get("prompt", "")).strip()
    if not prompt:
        raise ValueError(f"Agent '{normalized}' has an empty prompt definition.")
    return prompt


__all__ = [
    "SessionAgentSpec",
    "normalize_agent_name",
    "parse_session_agents",
    "merge_session_agents",
    "resolve_session_agent_prompt",
]
