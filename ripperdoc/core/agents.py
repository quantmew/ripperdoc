"""Agent definitions and helpers for Ripperdoc subagents."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

from ripperdoc.utils.log import get_logger


logger = get_logger()


AGENT_DIR_NAME = "agents"


class AgentLocation(str, Enum):
    """Where an agent definition is sourced from."""

    BUILT_IN = "built-in"
    USER = "user"
    PROJECT = "project"


@dataclass
class AgentDefinition:
    """A parsed agent definition."""

    agent_type: str
    when_to_use: str
    tools: List[str]
    system_prompt: str
    location: AgentLocation
    model: Optional[str] = None
    color: Optional[str] = None
    filename: Optional[str] = None


@dataclass
class AgentLoadResult:
    """Result of loading agent definitions."""

    active_agents: List[AgentDefinition]
    all_agents: List[AgentDefinition]
    failed_files: List[Tuple[Path, str]]


GENERAL_AGENT_PROMPT = """You are a general-purpose subagent for Ripperdoc. Work autonomously on the task provided by the parent agent. Use the allowed tools to research, edit files, and run commands as needed. When you finish, provide a concise report describing what you changed, what you investigated, and any follow-ups the parent agent should share with the user."""

CODE_REVIEW_AGENT_PROMPT = """You are a code review subagent. Inspect the code and summarize risks, bugs, missing tests, security concerns, and regressions. Do not make code changes. Provide clear, actionable feedback that the parent agent can relay to the user."""


def _built_in_agents() -> List[AgentDefinition]:
    return [
        AgentDefinition(
            agent_type="general-purpose",
            when_to_use=(
                "General-purpose agent for multi-step coding tasks, deep searches, and "
                "investigations that need their own context window."
            ),
            tools=["*"],
            system_prompt=GENERAL_AGENT_PROMPT,
            location=AgentLocation.BUILT_IN,
            color="cyan",
        ),
        AgentDefinition(
            agent_type="code-reviewer",
            when_to_use=(
                "Run after implementing non-trivial code changes to review for correctness, "
                "testing gaps, security issues, and regressions."
            ),
            tools=["View", "Glob", "Grep"],
            system_prompt=CODE_REVIEW_AGENT_PROMPT,
            location=AgentLocation.BUILT_IN,
            color="yellow",
        ),
    ]


def _agent_dirs() -> List[Tuple[Path, AgentLocation]]:
    home_dir = Path.home() / ".ripperdoc" / AGENT_DIR_NAME
    project_dir = Path.cwd() / ".ripperdoc" / AGENT_DIR_NAME
    return [
        (home_dir, AgentLocation.USER),
        (project_dir, AgentLocation.PROJECT),
    ]


def _agent_dir_for_location(location: AgentLocation) -> Path:
    for path, loc in _agent_dirs():
        if loc == location:
            return path
    raise ValueError(f"Unsupported agent location: {location}")


def _split_frontmatter(raw_text: str) -> Tuple[Dict[str, Any], str]:
    """Extract YAML frontmatter and body content."""
    lines = raw_text.splitlines()
    if len(lines) >= 3 and lines[0].strip() == "---":
        for idx in range(1, len(lines)):
            if lines[idx].strip() == "---":
                frontmatter_text = "\n".join(lines[1:idx])
                body = "\n".join(lines[idx + 1 :])
                try:
                    frontmatter = yaml.safe_load(frontmatter_text) or {}
                except Exception as exc:  # pragma: no cover - defensive
                    logger.exception("Invalid frontmatter in agent file", extra={"error": str(exc)})
                    return {"__error__": f"Invalid frontmatter: {exc}"}, body
                return frontmatter, body
    return {}, raw_text


def _normalize_tools(value: object) -> List[str]:
    if value is None:
        return ["*"]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()] or ["*"]
    if isinstance(value, Iterable):
        tools: List[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                tools.append(item.strip())
        return tools or ["*"]
    return ["*"]


def _parse_agent_file(
    path: Path, location: AgentLocation
) -> Tuple[Optional[AgentDefinition], Optional[str]]:
    """Parse a single agent file."""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.exception(
            "Failed to read agent file", extra={"error": str(exc), "path": str(path)}
        )
        return None, f"Failed to read agent file {path}: {exc}"

    frontmatter, body = _split_frontmatter(text)
    if "__error__" in frontmatter:
        return None, str(frontmatter["__error__"])

    agent_name = frontmatter.get("name")
    description = frontmatter.get("description")
    if not isinstance(agent_name, str) or not agent_name.strip():
        return None, 'Missing required "name" field in frontmatter'
    if not isinstance(description, str) or not description.strip():
        return None, 'Missing required "description" field in frontmatter'

    tools = _normalize_tools(frontmatter.get("tools"))
    model_value = frontmatter.get("model")
    color_value = frontmatter.get("color")
    model = model_value if isinstance(model_value, str) else None
    color = color_value if isinstance(color_value, str) else None

    agent = AgentDefinition(
        agent_type=agent_name.strip(),
        when_to_use=description.replace("\\n", "\n").strip(),
        tools=tools,
        system_prompt=body.strip(),
        location=location,
        model=model,
        color=color,
        filename=path.stem,
    )
    return agent, None


def _load_agent_dir(
    path: Path, location: AgentLocation
) -> Tuple[List[AgentDefinition], List[Tuple[Path, str]]]:
    agents: List[AgentDefinition] = []
    errors: List[Tuple[Path, str]] = []
    if not path.exists():
        return agents, errors

    for file_path in sorted(path.glob("*.md")):
        agent, error = _parse_agent_file(file_path, location)
        if agent:
            agents.append(agent)
        elif error:
            errors.append((file_path, error))
    return agents, errors


@lru_cache(maxsize=1)
def load_agent_definitions() -> AgentLoadResult:
    """Load built-in, user, and project agents."""
    built_ins = _built_in_agents()
    collected_agents = list(built_ins)
    errors: List[Tuple[Path, str]] = []

    for directory, location in _agent_dirs():
        loaded, dir_errors = _load_agent_dir(directory, location)
        collected_agents.extend(loaded)
        errors.extend(dir_errors)

    agent_map: Dict[str, AgentDefinition] = {}
    for agent in collected_agents:
        agent_map[agent.agent_type] = agent

    active_agents = list(agent_map.values())
    return AgentLoadResult(
        active_agents=active_agents,
        all_agents=collected_agents,
        failed_files=errors,
    )


def clear_agent_cache() -> None:
    """Reset cached agent definitions."""
    load_agent_definitions.cache_clear()  # type: ignore[attr-defined]


def summarize_agent(agent: AgentDefinition) -> str:
    """Short human-readable summary."""
    tool_label = "all tools" if "*" in agent.tools else ", ".join(agent.tools)
    location = getattr(agent.location, "value", agent.location)
    details = [f"tools: {tool_label}"]
    if agent.model:
        details.append(f"model: {agent.model}")
    return f"- {agent.agent_type} ({location}): {agent.when_to_use} [{'; '.join(details)}]"


def resolve_agent_tools(
    agent: AgentDefinition, available_tools: Iterable[object], task_tool_name: str
) -> Tuple[List[object], List[str]]:
    """Map tool names from an agent to Tool instances, filtering out the task tool itself."""
    tool_map: Dict[str, object] = {}
    ordered_tools: List[object] = []
    for tool in available_tools:
        name = getattr(tool, "name", None)
        if not name:
            continue
        if name == task_tool_name:
            continue
        tool_map[name] = tool
        ordered_tools.append(tool)

    if "*" in agent.tools:
        return ordered_tools, []

    resolved: List[object] = []
    missing: List[str] = []
    seen = set()
    for tool_name in agent.tools:
        if tool_name in seen:
            continue
        seen.add(tool_name)
        tool = tool_map.get(tool_name)
        if tool:
            resolved.append(tool)
        else:
            missing.append(tool_name)
    return resolved, missing


def save_agent_definition(
    agent_type: str,
    description: str,
    tools: List[str],
    system_prompt: str,
    location: AgentLocation = AgentLocation.USER,
    model: Optional[str] = None,
    color: Optional[str] = None,
    overwrite: bool = False,
) -> Path:
    """Persist an agent markdown file."""
    agent_dir = _agent_dir_for_location(location)
    agent_dir.mkdir(parents=True, exist_ok=True)
    target_path = agent_dir / f"{agent_type}.md"
    if target_path.exists() and not overwrite:
        raise FileExistsError(f"Agent file already exists: {target_path}")

    escaped_description = description.replace("\n", "\\n")
    lines = [
        "---",
        f"name: {agent_type}",
        f"description: {escaped_description}",
    ]

    if not (len(tools) == 1 and tools[0] == "*"):
        joined_tools = ", ".join(tools)
        lines.append(f"tools: {joined_tools}")
    if model:
        lines.append(f"model: {model}")
    if color:
        lines.append(f"color: {color}")
    lines.append("---")
    lines.append("")
    lines.append(system_prompt.strip())
    target_path.write_text("\n".join(lines), encoding="utf-8")
    clear_agent_cache()
    return target_path


def delete_agent_definition(agent_type: str, location: AgentLocation = AgentLocation.USER) -> Path:
    """Delete an agent markdown file."""
    agent_dir = _agent_dir_for_location(location)
    target_path = agent_dir / f"{agent_type}.md"
    if target_path.exists():
        target_path.unlink()
        clear_agent_cache()
        return target_path
    raise FileNotFoundError(f"Agent file not found: {target_path}")
