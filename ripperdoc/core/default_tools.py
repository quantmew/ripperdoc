"""Shared factory for default tool instances."""

from __future__ import annotations

from typing import List

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
from ripperdoc.tools.todo_tool import TodoReadTool, TodoWriteTool
from ripperdoc.tools.task_tool import TaskTool
from ripperdoc.tools.mcp_tools import (
    ListMcpResourcesTool,
    ListMcpServersTool,
    ReadMcpResourceTool,
    load_dynamic_mcp_tools_sync,
)


def get_default_tools() -> List:
    """Construct the default tool set (base tools + Task subagent launcher)."""
    base_tools = [
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
        TodoReadTool(),
        TodoWriteTool(),
        ListMcpServersTool(),
        ListMcpResourcesTool(),
        ReadMcpResourceTool(),
    ]
    try:
        base_tools.extend(load_dynamic_mcp_tools_sync())
    except Exception:
        # If MCP runtime is not available, continue with base tools only.
        pass

    task_tool = TaskTool(lambda: base_tools)
    return base_tools + [task_tool]
