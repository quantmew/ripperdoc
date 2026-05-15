"""Agent definitions and helpers for Ripperdoc subagents."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

from ripperdoc.core.hooks.config import HooksConfig, parse_hooks_config
from ripperdoc.services.plugins import discover_plugins
from ripperdoc.utils.filesystem.config_paths import project_config_dir, user_config_dir
from ripperdoc.utils.coerce import parse_boolish
from ripperdoc.utils.log import get_logger
from ripperdoc.tools.ask_user_question import AskUserQuestionTool
from ripperdoc.tools.bash import BashTool
from ripperdoc.tools.enter_plan_mode import EnterPlanModeTool
from ripperdoc.tools.exit_plan_mode import ExitPlanModeTool
from ripperdoc.tools.file_edit import FileEditTool
from ripperdoc.tools.file_read import FileReadTool
from ripperdoc.tools.file_write import FileWriteTool
from ripperdoc.tools.glob import GlobTool
from ripperdoc.tools.grep import GrepTool
from ripperdoc.tools.ls import LSTool
from ripperdoc.tools.lsp import LspTool
from ripperdoc.tools.multi_edit import MultiEditTool
from ripperdoc.tools.notebook_edit import NotebookEditTool
from ripperdoc.tools.skill import SkillTool
from ripperdoc.tools.todo import TodoReadTool, TodoWriteTool
from ripperdoc.tools.tool_search import ToolSearchTool
from ripperdoc.tools.mcp import (
    ListMcpResourcesTool,
    ListMcpServersTool,
    ReadMcpResourceTool,
)


logger = get_logger()


def _safe_tool_name(factory: Any, fallback: str) -> str:
    try:
        name = getattr(factory(), "name", None)
        return str(name) if name else fallback
    except (TypeError, ValueError, RuntimeError, AttributeError):
        return fallback


GLOB_TOOL_NAME = _safe_tool_name(GlobTool, "Glob")
GREP_TOOL_NAME = _safe_tool_name(GrepTool, "Grep")
READ_TOOL_NAME = _safe_tool_name(FileReadTool, "Read")
FILE_EDIT_TOOL_NAME = _safe_tool_name(FileEditTool, "FileEdit")
MULTI_EDIT_TOOL_NAME = _safe_tool_name(MultiEditTool, "MultiEdit")
NOTEBOOK_EDIT_TOOL_NAME = _safe_tool_name(NotebookEditTool, "NotebookEdit")
FILE_WRITE_TOOL_NAME = _safe_tool_name(FileWriteTool, "FileWrite")
LS_TOOL_NAME = _safe_tool_name(LSTool, "LS")
BASH_TOOL_NAME = _safe_tool_name(BashTool, "Bash")
TODO_READ_TOOL_NAME = _safe_tool_name(TodoReadTool, "TodoRead")
TODO_WRITE_TOOL_NAME = _safe_tool_name(TodoWriteTool, "TodoWrite")
ASK_USER_QUESTION_TOOL_NAME = _safe_tool_name(AskUserQuestionTool, "AskUserQuestion")
ENTER_PLAN_MODE_TOOL_NAME = _safe_tool_name(EnterPlanModeTool, "EnterPlanMode")
EXIT_PLAN_MODE_TOOL_NAME = _safe_tool_name(ExitPlanModeTool, "ExitPlanMode")
TOOL_SEARCH_TOOL_NAME = _safe_tool_name(ToolSearchTool, "ToolSearch")
MCP_LIST_SERVERS_TOOL_NAME = _safe_tool_name(ListMcpServersTool, "ListMcpServers")
MCP_LIST_RESOURCES_TOOL_NAME = _safe_tool_name(ListMcpResourcesTool, "ListMcpResources")
MCP_READ_RESOURCE_TOOL_NAME = _safe_tool_name(ReadMcpResourceTool, "ReadMcpResource")
LSP_TOOL_NAME = _safe_tool_name(LspTool, "LSP")
SKILL_TOOL_NAME = _safe_tool_name(SkillTool, "Skill")
TASK_TOOL_NAME = "Task"


AGENT_DIR_NAME = "agents"


class AgentLocation(str, Enum):
    """Where an agent definition is sourced from."""

    BUILT_IN = "built-in"
    USER = "user"
    PROJECT = "project"
    PLUGIN = "plugin"


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
    fork_context: bool = False
    hooks: HooksConfig = field(default_factory=HooksConfig)
    plugin_name: Optional[str] = None
    disallowed_tools: List[str] = field(default_factory=list)
    permission_mode: Optional[str] = None
    max_turns: Optional[int] = None
    memory: Optional[str] = None  # "user", "project", "local"
    background: bool = False
    initial_prompt: Optional[str] = None
    omit_claude_md: bool = False


@dataclass
class AgentLoadResult:
    """Result of loading agent definitions."""

    active_agents: List[AgentDefinition]
    all_agents: List[AgentDefinition]
    failed_files: List[Tuple[Path, str]]


def _agent_dirs(
    project_path: Optional[Path] = None, home: Optional[Path] = None
) -> List[Tuple[Path, AgentLocation]]:
    home_dir = user_config_dir(home=home) / AGENT_DIR_NAME
    project_dir = project_config_dir(project_path) / AGENT_DIR_NAME
    return [
        (home_dir, AgentLocation.USER),
        (project_dir, AgentLocation.PROJECT),
    ]


def _agent_dir_for_location(
    location: AgentLocation,
    project_path: Optional[Path] = None,
    home: Optional[Path] = None,
) -> Path:
    for path, loc in _agent_dirs(project_path=project_path, home=home):
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
                except (
                    yaml.YAMLError,
                    ValueError,
                    TypeError,
                ) as exc:  # pragma: no cover - defensive
                    logger.warning(
                        "Invalid frontmatter in agent file: %s: %s",
                        type(exc).__name__,
                        exc,
                        extra={"error": str(exc)},
                    )
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


def _convert_stop_hook_to_subagent(hooks_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Stop hooks to SubagentStop for subagent-scoped hooks."""
    if "Stop" not in hooks_data:
        return hooks_data
    converted = dict(hooks_data)
    stop_matchers = converted.pop("Stop")
    existing = converted.get("SubagentStop")
    if existing is None:
        converted["SubagentStop"] = stop_matchers
    elif isinstance(existing, list) and isinstance(stop_matchers, list):
        converted["SubagentStop"] = existing + stop_matchers
    elif isinstance(stop_matchers, list):
        converted["SubagentStop"] = stop_matchers
    return converted


def _normalize_agent_hooks(raw_hooks: object) -> object:
    if not isinstance(raw_hooks, dict):
        return raw_hooks
    if "hooks" in raw_hooks and isinstance(raw_hooks.get("hooks"), dict):
        wrapped = dict(raw_hooks)
        wrapped["hooks"] = _convert_stop_hook_to_subagent(wrapped["hooks"])
        return wrapped
    return _convert_stop_hook_to_subagent(raw_hooks)


def _parse_agent_file(
    path: Path,
    location: AgentLocation,
    *,
    namespace_prefix: Optional[str] = None,
    plugin_name: Optional[str] = None,
) -> Tuple[Optional[AgentDefinition], Optional[str]]:
    """Parse a single agent file."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, IOError, UnicodeDecodeError) as exc:
        logger.warning(
            "Failed to read agent file: %s: %s",
            type(exc).__name__,
            exc,
            extra={"error": str(exc), "path": str(path)},
        )
        return None, f"Failed to read agent file {path}: {exc}"

    frontmatter, body = _split_frontmatter(text)
    error = frontmatter.get("__error__")
    if error is not None:
        return None, str(error)

    agent_name = frontmatter.get("name")
    description = frontmatter.get("description")
    resolved_agent_name = (
        agent_name.strip() if isinstance(agent_name, str) and agent_name.strip() else path.stem
    )
    if not resolved_agent_name:
        return None, 'Missing required "name" field in frontmatter'
    if not isinstance(description, str) or not description.strip():
        return None, 'Missing required "description" field in frontmatter'
    full_agent_name = (
        f"{namespace_prefix}:{resolved_agent_name}" if namespace_prefix else resolved_agent_name
    )

    tools = _normalize_tools(frontmatter.get("tools"))
    model_value = frontmatter.get("model")
    color_value = frontmatter.get("color")
    model = model_value if isinstance(model_value, str) else None
    color = color_value if isinstance(color_value, str) else None
    fork_context = parse_boolish(frontmatter.get("fork_context") or frontmatter.get("fork-context"))
    hooks = parse_hooks_config(
        _normalize_agent_hooks(frontmatter.get("hooks")), source=f"agent:{full_agent_name}"
    )
    disallowed_tools = _normalize_tools(frontmatter.get("disallowed_tools") or frontmatter.get("disallowedTools"))
    if not isinstance(disallowed_tools, list):
        disallowed_tools = []
    # Reset wildcard from _normalize_tools for disallowed list
    if disallowed_tools == ["*"]:
        disallowed_tools = []
    permission_mode = frontmatter.get("permissionMode") or frontmatter.get("permission_mode")
    if not isinstance(permission_mode, str):
        permission_mode = None
    max_turns_raw = frontmatter.get("maxTurns") or frontmatter.get("max_turns")
    max_turns = int(max_turns_raw) if isinstance(max_turns_raw, (int, float)) else None
    memory = frontmatter.get("memory")
    if not isinstance(memory, str):
        memory = None
    background = parse_boolish(frontmatter.get("background"))
    initial_prompt = frontmatter.get("initialPrompt") or frontmatter.get("initial_prompt")
    if not isinstance(initial_prompt, str):
        initial_prompt = None
    omit_claude_md = parse_boolish(frontmatter.get("omitClaudeMd") or frontmatter.get("omit_claude_md"))

    agent = AgentDefinition(
        agent_type=full_agent_name,
        when_to_use=description.replace("\\n", "\n").strip(),
        tools=tools,
        system_prompt=body.strip(),
        location=location,
        model=model,
        color=color,
        filename=path.stem,
        fork_context=fork_context,
        hooks=hooks,
        plugin_name=plugin_name,
        disallowed_tools=disallowed_tools,
        permission_mode=permission_mode,
        max_turns=max_turns,
        memory=memory,
        background=background,
        initial_prompt=initial_prompt,
        omit_claude_md=omit_claude_md,
    )
    return agent, None


def _load_agent_dir(
    path: Path,
    location: AgentLocation,
    *,
    namespace_prefix: Optional[str] = None,
    plugin_name: Optional[str] = None,
) -> Tuple[List[AgentDefinition], List[Tuple[Path, str]]]:
    agents: List[AgentDefinition] = []
    errors: List[Tuple[Path, str]] = []
    if not path.exists():
        return agents, errors

    for file_path in sorted(path.glob("*.md")):
        agent, error = _parse_agent_file(
            file_path,
            location,
            namespace_prefix=namespace_prefix,
            plugin_name=plugin_name,
        )
        if agent:
            agents.append(agent)
        elif error:
            errors.append((file_path, error))
    return agents, errors


def _load_agent_path(
    path: Path,
    location: AgentLocation,
    *,
    namespace_prefix: Optional[str] = None,
    plugin_name: Optional[str] = None,
) -> Tuple[List[AgentDefinition], List[Tuple[Path, str]]]:
    if path.is_file():
        if path.suffix.lower() != ".md":
            return [], []
        parsed, error = _parse_agent_file(
            path,
            location,
            namespace_prefix=namespace_prefix,
            plugin_name=plugin_name,
        )
        if parsed:
            return [parsed], []
        if error:
            return [], [(path, error)]
        return [], []
    if path.is_dir():
        return _load_agent_dir(
            path,
            location,
            namespace_prefix=namespace_prefix,
            plugin_name=plugin_name,
        )
    return [], []


def load_agent_definitions(
    project_path: Optional[Path] = None, home: Optional[Path] = None
) -> AgentLoadResult:
    """Load built-in, user, and project agents."""
    from ripperdoc.tools.agent._built_in import _built_in_agents

    built_ins = _built_in_agents()
    collected_agents = list(built_ins)
    errors: List[Tuple[Path, str]] = []

    for directory, location in _agent_dirs(project_path=project_path, home=home):
        loaded, dir_errors = _load_agent_dir(directory, location)
        collected_agents.extend(loaded)
        errors.extend(dir_errors)

    plugin_result = discover_plugins(project_path=project_path, home=home)
    for plugin_error in plugin_result.errors:
        errors.append((plugin_error.path, plugin_error.reason))
    for plugin in plugin_result.plugins:
        for agent_path in plugin.agents_paths:
            loaded, dir_errors = _load_agent_path(
                agent_path,
                AgentLocation.PLUGIN,
                namespace_prefix=plugin.name,
                plugin_name=plugin.name,
            )
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
    # No-op. Agent loading is intentionally uncached so plugin updates are visible immediately.
    return


def summarize_agent(agent: AgentDefinition) -> str:
    """Short human-readable agent summary."""
    if "*" in agent.tools:
        tools_description = "All tools"
    elif agent.disallowed_tools:
        disallowed = ", ".join(agent.disallowed_tools)
        tools_description = f"All tools except {disallowed}"
    else:
        tools_description = ", ".join(agent.tools) if agent.tools else "None"
    return f"- {agent.agent_type}: {agent.when_to_use} (Tools: {tools_description})"


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

    # Apply disallowed_tools filter
    disallowed = set(agent.disallowed_tools) if agent.disallowed_tools else set()
    if disallowed:
        ordered_tools = [t for t in ordered_tools if getattr(t, "name", None) not in disallowed]

    if "*" in agent.tools:
        return ordered_tools, []

    resolved: List[object] = []
    missing: List[str] = []
    seen = set()
    for tool_name in agent.tools:
        if tool_name in seen:
            continue
        if tool_name in disallowed:
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
