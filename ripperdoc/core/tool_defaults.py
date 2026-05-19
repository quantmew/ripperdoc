"""Shared factory for default tool instances."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, cast

from ripperdoc.core.tool import Tool

from ripperdoc.tools.bash import BashTool
from ripperdoc.tools.task_stop import TaskStopTool
from ripperdoc.tools.file_read import FileReadTool
from ripperdoc.tools.file_edit import FileEditTool
from ripperdoc.tools.notebook_edit import NotebookEditTool
from ripperdoc.tools.file_write import FileWriteTool
from ripperdoc.tools.glob import GlobTool
from ripperdoc.tools.ls import LSTool
from ripperdoc.tools.grep import GrepTool
from ripperdoc.tools.lsp import LspTool
from ripperdoc.tools.skill import SkillTool
from ripperdoc.tools.todo_write import TodoWriteTool
from ripperdoc.tools.todo_read import TodoReadTool
from ripperdoc.tools.task_create import TaskCreateTool
from ripperdoc.tools.task_get import TaskGetTool
from ripperdoc.tools.task_list import TaskListTool
from ripperdoc.tools.task_update import TaskUpdateTool
from ripperdoc.tools.send_message import SendMessageTool
from ripperdoc.tools.team_create import TeamCreateTool
from ripperdoc.tools.team_delete import TeamDeleteTool
from ripperdoc.tools.ask_user_question import AskUserQuestionTool
from ripperdoc.tools.enter_plan_mode import EnterPlanModeTool
from ripperdoc.tools.enter_worktree import EnterWorktreeTool
from ripperdoc.tools.exit_plan_mode import ExitPlanModeTool
from ripperdoc.tools.exit_worktree import ExitWorktreeTool
from ripperdoc.tools.memory import MemoryTool
from ripperdoc.tools.agent import AgentTool
from ripperdoc.tools.tool_search import ToolSearchTool
from ripperdoc.tools.sleep import SleepTool
from ripperdoc.tools.schedule_cron import CronCreateTool, CronDeleteTool, CronListTool
from ripperdoc.tools.mcp import (
    ListMcpResourcesTool,
    ListMcpServersTool,
    ReadMcpResourceTool,
)
from ripperdoc.tools.mcp.dynamic_mcp import (
    load_dynamic_mcp_tools_async,
    load_dynamic_mcp_tools_sync,
    merge_tools_with_dynamic,
)
from ripperdoc.utils.log import get_logger
from ripperdoc.utils.collaboration.tasks import is_task_system_enabled

logger = get_logger()

# Canonical tool names for --tools filtering
BUILTIN_TOOL_NAMES = [
    "Bash",
    "TaskStop",
    "Read",
    "Edit",

    "NotebookEdit",
    "Write",
    "Glob",
    "LS",
    "Grep",
    "LSP",
    "Skill",
    "TodoRead",
    "TodoWrite",
    "TaskCreate",
    "TaskGet",
    "TaskUpdate",
    "TaskList",
    "TeamCreate",
    "TeamDelete",
    "SendMessage",
    "AskUserQuestion",
    "EnterPlanMode",
    "EnterWorktree",
    "ExitPlanMode",
    "ExitWorktree",
    "Memory",
    "ToolSearch",
    "ListMcpServers",
    "ListMcpResources",
    "ReadMcpResource",
    "Sleep",
    "CronCreate",
    "CronDelete",
    "CronList",
    "Agent",
]


def filter_tools_by_names(
    tools: List[Tool[Any, Any]], tool_names: List[str]
) -> List[Tool[Any, Any]]:
    """Filter a tool list to only include tools with matching names.

    Args:
        tools: The full list of tools to filter.
        tool_names: List of tool names to include.

    Returns:
        Filtered list of tools. If Task is included, it's recreated with
        the filtered base tools.
    """
    if not tool_names:
        return []

    name_set = set(tool_names)
    filtered: List[Tool[Any, Any]] = []
    has_task = False

    for tool in tools:
        tool_name = getattr(tool, "name", tool.__class__.__name__)
        if tool_name in name_set:
            if tool_name == "Agent":
                has_task = True
            else:
                filtered.append(tool)

    # If Agent is requested, recreate it with the filtered base tools
    if has_task:

        def _filtered_base_provider() -> List[Tool[Any, Any]]:
            return [t for t in filtered if getattr(t, "name", None) != "Agent"]

        filtered.append(AgentTool(_filtered_base_provider))

    return filtered


def _build_base_tools() -> List[Tool[Any, Any]]:
    """Construct builtin tools without runtime-dependent MCP tool discovery."""
    tasks_enabled = is_task_system_enabled()
    base_tools: List[Tool[Any, Any]] = [
        BashTool(),
        TaskStopTool(),
        FileReadTool(),
        FileEditTool(),
        NotebookEditTool(),
        FileWriteTool(),
        GlobTool(),
        LSTool(),
        GrepTool(),
        LspTool(),
        SkillTool(),
        AskUserQuestionTool(),
        EnterPlanModeTool(),
        EnterWorktreeTool(),
        ExitPlanModeTool(),
        MemoryTool(),
        ToolSearchTool(),
        ExitWorktreeTool(),
        SleepTool(),
        CronCreateTool(),
        CronDeleteTool(),
        CronListTool(),
        ListMcpServersTool(),
        ListMcpResourcesTool(),
        ReadMcpResourceTool(),
    ]
    if tasks_enabled:
        base_tools.extend(
            [
                TaskCreateTool(),
                TaskGetTool(),
                TaskUpdateTool(),
                TaskListTool(),
                TeamCreateTool(),
                TeamDeleteTool(),
                SendMessageTool(),
            ]
        )
    else:
        base_tools.extend([TodoReadTool(), TodoWriteTool()])
    return base_tools


def _finalize_tool_list(
    base_tools: List[Tool[Any, Any]],
    *,
    allowed_tools: Optional[List[str]] = None,  # noqa: ARG001 – kept for API compat
    dynamic_tool_count: int = 0,
) -> List[Tool[Any, Any]]:
    """Append Task tool to the base tool list.

    ``allowed_tools`` is accepted for backward compatibility but is **not**
    used for tool-set filtering.  It controls auto-approval permissions and
    is handled by the permission engine.
    """
    task_tool = AgentTool(lambda: base_tools)
    all_tools = base_tools + [task_tool]

    logger.debug(
        "[tool_defaults] Built tool inventory",
        extra={
            "tasks_enabled": is_task_system_enabled(),
            "base_tools": len(base_tools),
            "dynamic_mcp_tools": dynamic_tool_count,
            "total_tools": len(all_tools),
        },
    )

    return all_tools


def get_default_tools(allowed_tools: Optional[List[str]] = None) -> List[Tool[Any, Any]]:
    """Construct the default tool set for synchronous callers."""
    base_tools = _build_base_tools()

    dynamic_tools: List[Tool[Any, Any]] = []
    try:
        mcp_tools = load_dynamic_mcp_tools_sync()
        # Filter to ensure only Tool instances are added
        for tool in mcp_tools:
            if isinstance(tool, Tool):
                base_tools.append(tool)
                dynamic_tools.append(tool)
    except (
        ImportError,
        ModuleNotFoundError,
        OSError,
        RuntimeError,
        ConnectionError,
        ValueError,
        TypeError,
    ) as exc:
        # If MCP runtime is not available, continue with base tools only.
        logger.warning(
            "[tool_defaults] Failed to load dynamic MCP tools: %s: %s",
            type(exc).__name__,
            exc,
        )
    return _finalize_tool_list(
        base_tools,
        allowed_tools=allowed_tools,
        dynamic_tool_count=len(dynamic_tools),
    )


async def get_default_tools_async(
    *,
    project_path: Optional[Path] = None,
    allowed_tools: Optional[List[str]] = None,
) -> List[Tool[Any, Any]]:
    """Construct the default tool set using the active event loop for MCP discovery."""
    base_tools = _build_base_tools()
    dynamic_tools: List[Tool[Any, Any]] = []
    try:
        mcp_tools = await load_dynamic_mcp_tools_async(project_path)
        typed_dynamic_tools = [tool for tool in mcp_tools if isinstance(tool, Tool)]
        if typed_dynamic_tools:
            base_tools = merge_tools_with_dynamic(base_tools, typed_dynamic_tools)[:-1]
            dynamic_tools = cast(List[Tool[Any, Any]], typed_dynamic_tools)
    except (
        ImportError,
        ModuleNotFoundError,
        OSError,
        RuntimeError,
        ConnectionError,
        ValueError,
        TypeError,
    ) as exc:
        logger.warning(
            "[tool_defaults] Failed to load dynamic MCP tools asynchronously: %s: %s",
            type(exc).__name__,
            exc,
        )
    return _finalize_tool_list(
        base_tools,
        allowed_tools=allowed_tools,
        dynamic_tool_count=len(dynamic_tools),
    )
