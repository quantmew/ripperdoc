"""
MCP utility functions — mirrors reference: services/mcp/utils.ts.

Provides instructions formatting, token estimation, and resource lookup.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from ripperdoc.services.mcp.types import McpResourceInfo, McpServerInfo
from ripperdoc.utils.token_estimation import estimate_tokens


def _summarize_tools(server: McpServerInfo) -> str:
    """Build a comma-separated tool summary for a server."""
    if not server.tools:
        return "no tools"
    names = [tool.name for tool in server.tools[:6]]
    suffix = ", ".join(names)
    if len(server.tools) > 6:
        suffix += f", and {len(server.tools) - 6} more"
    return suffix


def format_mcp_instructions(servers: List[McpServerInfo]) -> str:
    """Build a concise MCP instruction block for the system prompt.

    Mirrors the reference's MCP formatting logic.
    """
    if not servers:
        return ""

    connected_count = len([s for s in servers if s.status == "connected"])
    lines: List[str] = []
    if connected_count > 0:
        lines.append("Connected MCP servers are available.")
    else:
        lines.append(
            "MCP servers are configured, but none are connected yet. "
            "Prefer non-MCP tools unless a server is [connected]."
        )
    lines.append(
        "Use ListMcpServers to inspect statuses and "
        "ListMcpResources/ReadMcpResource when a server exposes resources."
    )

    for server in servers:
        status = server.status or "unknown"
        prefix = f"- {server.name} [{status}]"
        if server.url:
            prefix += f" {server.url}"
        lines.append(prefix)

        if status == "connected":
            if server.instructions:
                trimmed = server.instructions.strip()
                if len(trimmed) > 260:
                    trimmed = trimmed[:257] + "..."
                lines.append(f"  Instructions: {trimmed}")
            tool_summary = _summarize_tools(server)
            lines.append(f"  Tools: {tool_summary}")
            if server.resources:
                lines.append(f"  Resources: {len(server.resources)} available")
        elif status == "connecting":
            lines.append("  Status: connecting (tool discovery in progress)")
        elif server.error:
            lines.append(f"  Error: {server.error}")

    return "\n".join(lines)


def estimate_mcp_tokens(servers: List[McpServerInfo]) -> int:
    """Estimate token usage for MCP instructions."""
    mcp_text = format_mcp_instructions(servers)
    return estimate_tokens(mcp_text)


def find_mcp_resource(
    servers: List[McpServerInfo], server_name: str, uri: str
) -> Optional[McpResourceInfo]:
    """Find an MCP resource by server name and URI."""
    server = next((s for s in servers if s.name == server_name), None)
    if not server:
        return None
    return next((r for r in server.resources if r.uri == uri), None)


def load_mcp_servers_async(
    project_path: Optional[Path] = None,
    *,
    wait_for_connections: bool = False,
    wait_timeout: Optional[float] = None,
) -> Any:
    """Load MCP servers, ensuring runtime initialization.

    Returns server snapshot from the runtime.
    """
    from ripperdoc.services.mcp.client import ensure_mcp_runtime

    async def _load() -> List[McpServerInfo]:
        runtime = await ensure_mcp_runtime(
            project_path,
            wait_for_connections=wait_for_connections,
            wait_timeout=wait_timeout,
        )
        return runtime.server_snapshot()

    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_load())
    if loop.is_running():
        return _load()
    return asyncio.run(_load())



def load_mcp_servers(
    project_path: Optional[Path] = None,
    *,
    wait_for_connections: bool = True,
    wait_timeout: Optional[float] = None,
) -> List[McpServerInfo]:
    """Synchronous wrapper primarily for legacy call sites."""
    import asyncio
    from ripperdoc.services.mcp.client import get_existing_mcp_runtime, shutdown_mcp_runtime

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            runtime = get_existing_mcp_runtime()
            if runtime and runtime.servers:
                return runtime.server_snapshot()
            return []
    except RuntimeError:
        pass

    async def _load_and_shutdown() -> List[McpServerInfo]:
        try:
            return await load_mcp_servers_async(  # type: ignore[no-any-return]
                project_path,
                wait_for_connections=wait_for_connections,
                wait_timeout=wait_timeout,
            )
        finally:
            await shutdown_mcp_runtime()

    import asyncio
    return asyncio.run(_load_and_shutdown())
