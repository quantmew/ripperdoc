"""Shared factory for default tool instances."""

from __future__ import annotations

from typing import Any, List

from ripperdoc.core.tool import Tool

from ripperdoc.tools.bash_tool import BashTool
from ripperdoc.tools.bash_output_tool import BashOutputTool
from ripperdoc.tools.kill_bash_tool import KillBashTool
from ripperdoc.tools.file_read_tool import FileReadTool
from ripperdoc.tools.file_edit_tool import FileEditTool
from ripperdoc.tools.multi_edit_tool import MultiEditTool
from ripperdoc.tools.notebook_edit_tool import NotebookEditTool
from ripperdoc.tools.file_write_tool import FileWriteTool
from ripperdoc.tools.glob_tool import GlobTool
from ripperdoc.tools.ls_tool import LSTool
from ripperdoc.tools.grep_tool import GrepTool
from ripperdoc.tools.skill_tool import SkillTool
from ripperdoc.tools.todo_tool import TodoReadTool, TodoWriteTool
from ripperdoc.tools.ask_user_question_tool import AskUserQuestionTool
from ripperdoc.tools.enter_plan_mode_tool import EnterPlanModeTool
from ripperdoc.tools.exit_plan_mode_tool import ExitPlanModeTool
from ripperdoc.tools.task_tool import TaskTool
from ripperdoc.tools.tool_search_tool import ToolSearchTool
from ripperdoc.tools.mcp_tools import (
    ListMcpResourcesTool,
    ListMcpServersTool,
    ReadMcpResourceTool,
    load_dynamic_mcp_tools_sync,
)
from ripperdoc.utils.log import get_logger

logger = get_logger()


def get_default_tools() -> List[Tool[Any, Any]]:
    """Construct the default tool set (base tools + Task subagent launcher)."""
    base_tools: List[Tool[Any, Any]] = [
        BashTool(),
        BashOutputTool(),
        KillBashTool(),
        FileReadTool(),
        FileEditTool(),
        MultiEditTool(),
        NotebookEditTool(),
        FileWriteTool(),
        GlobTool(),
        LSTool(),
        GrepTool(),
        SkillTool(),
        TodoReadTool(),
        TodoWriteTool(),
        AskUserQuestionTool(),
        EnterPlanModeTool(),
        ExitPlanModeTool(),
        ToolSearchTool(),
        ListMcpServersTool(),
        ListMcpResourcesTool(),
        ReadMcpResourceTool(),
    ]
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
            "[default_tools] Failed to load dynamic MCP tools: %s: %s",
            type(exc).__name__,
            exc,
        )

    task_tool = TaskTool(lambda: base_tools)
    all_tools = base_tools + [task_tool]
    logger.debug(
        "[default_tools] Built tool inventory",
        extra={
            "base_tools": len(base_tools),
            "dynamic_mcp_tools": len(dynamic_tools),
            "total_tools": len(all_tools),
        },
    )
    return all_tools
