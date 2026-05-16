"""McpAuthTool - per-server MCP authentication pseudo-tools.

Provides a factory function ``create_mcp_auth_tool`` that generates
dynamically-created tools for MCP servers that require authentication.
Follows the reference ``McpAuthTool.ts`` pattern closely.

Usage:
    from ripperdoc.tools.mcp_auth import create_mcp_auth_tool, McpAuthOutput

    tool = create_mcp_auth_tool("myserver", server_config)
    async for result in tool.call(McpAuthInput(), context):
        ...
"""

from ripperdoc.tools.mcp_auth._tool import (
    DynamicMcpAuthTool,
    McpAuthInput,
    McpAuthOutput,
    create_mcp_auth_tool,
)

__all__ = [
    "DynamicMcpAuthTool",
    "McpAuthInput",
    "McpAuthOutput",
    "create_mcp_auth_tool",
]
