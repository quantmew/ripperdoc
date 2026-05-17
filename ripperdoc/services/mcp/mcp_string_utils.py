"""
Pure string utility functions for MCP tool/server name parsing.
Mirrors reference: services/mcp/mcpStringUtils.ts

This file has no heavy dependencies to keep it lightweight for
consumers that only need string parsing (e.g., permission validation).
"""

from typing import Optional


def mcp_info_from_string(tool_string: str) -> Optional[dict]:
    """Extract MCP server information from a tool name string.

    Expected format: ``mcp__serverName__toolName``

    Returns a dict with ``server_name`` and ``tool_name`` keys,
    or None if the string is not a valid MCP tool name.

    Known limitation: If a server name contains ``__``, parsing will
    be incorrect. This is rare since server names typically don't
    contain double underscores.
    """
    if not tool_string or not tool_string.startswith("mcp__"):
        return None

    # Strip the "mcp__" prefix
    rest = tool_string[5:]
    if not rest:
        return None

    # Split on first "__" to separate server name from tool name
    parts = rest.split("__", 1)
    server_name = parts[0]
    tool_name = parts[1] if len(parts) > 1 else None

    if not server_name:
        return None

    result: dict = {"server_name": server_name}
    if tool_name:
        result["tool_name"] = tool_name
    return result


def build_mcp_tool_name(server_name: str, tool_name: str) -> str:
    """Build a fully-qualified MCP tool name.

    Format: ``mcp__{sanitized_server_name}__{tool_name}``
    """
    from ripperdoc.services.mcp.normalization import normalize_name_for_mcp

    sanitized = normalize_name_for_mcp(server_name)
    return f"mcp__{sanitized}__{tool_name}"


def get_mcp_prefix(server_name: str) -> str:
    """Get the prefix used for all tools belonging to an MCP server.

    Format: ``mcp__{sanitized_server_name}__``
    """
    from ripperdoc.services.mcp.normalization import normalize_name_for_mcp

    sanitized = normalize_name_for_mcp(server_name)
    return f"mcp__{sanitized}__"
