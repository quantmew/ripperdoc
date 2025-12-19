"""MCP configuration loader, connection manager, and prompt helpers."""

from __future__ import annotations

import asyncio
import contextvars
import json
import shlex
from contextlib import AsyncExitStack
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

from ripperdoc import __version__
from ripperdoc.utils.log import get_logger
from ripperdoc.utils.token_estimation import estimate_tokens

logger = get_logger()

try:
    import mcp.types as mcp_types  # type: ignore[import-not-found]
    from mcp.client.session import ClientSession  # type: ignore[import-not-found]
    from mcp.client.sse import sse_client  # type: ignore[import-not-found]
    from mcp.client.stdio import StdioServerParameters, stdio_client  # type: ignore[import-not-found]
    from mcp.client.streamable_http import streamablehttp_client  # type: ignore[import-not-found]

    MCP_AVAILABLE = True
except (ImportError, ModuleNotFoundError):  # pragma: no cover - handled gracefully at runtime
    MCP_AVAILABLE = False
    ClientSession = object  # type: ignore
    mcp_types = None  # type: ignore
    logger.debug("[mcp] MCP SDK not available at import time")


@dataclass
class McpToolInfo:
    name: str
    description: str = ""
    input_schema: Optional[Dict[str, Any]] = None
    annotations: Dict[str, Any] = field(default_factory=dict)


@dataclass
class McpResourceInfo:
    uri: str
    name: Optional[str] = None
    description: str = ""
    mime_type: Optional[str] = None
    size: Optional[int] = None
    text: Optional[str] = None


@dataclass
class McpServerInfo:
    name: str
    type: str = "stdio"
    url: Optional[str] = None
    description: str = ""
    command: Optional[str] = None
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    tools: List[McpToolInfo] = field(default_factory=list)
    resources: List[McpResourceInfo] = field(default_factory=list)
    status: str = "configured"
    error: Optional[str] = None
    instructions: Optional[str] = None
    server_version: Optional[str] = None
    capabilities: Dict[str, Any] = field(default_factory=dict)


def _load_json_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            return data
        return {}
    except (OSError, json.JSONDecodeError):
        logger.exception("Failed to load JSON", extra={"path": str(path)})
        return {}


def _ensure_str_dict(raw: object) -> Dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    result: Dict[str, str] = {}
    for key, value in raw.items():
        try:
            result[str(key)] = str(value)
        except (TypeError, ValueError) as exc:
            logger.warning(
                "[mcp] Failed to coerce env/header value to string: %s: %s",
                type(exc).__name__,
                exc,
                extra={"key": key},
            )
            continue
    return result


def _normalize_command(raw_command: Any, raw_args: Any) -> tuple[Optional[str], List[str]]:
    """Normalize MCP server command/args.

    Supports:
    - command as list -> first element is executable, rest are args
    - command as string with spaces -> shlex.split into executable/args (when args empty)
    - command as plain string -> used as-is
    """
    args: List[str] = []
    if isinstance(raw_args, list):
        args = [str(a) for a in raw_args]

    # Command provided as list: treat first token as command.
    if isinstance(raw_command, list):
        tokens = [str(t) for t in raw_command if str(t)]
        if not tokens:
            return None, args
        return tokens[0], tokens[1:] + args

    if not isinstance(raw_command, str):
        return None, args

    command_str = raw_command.strip()
    if not command_str:
        return None, args

    if not args and (" " in command_str or "\t" in command_str):
        try:
            tokens = shlex.split(command_str)
        except ValueError:
            tokens = [command_str]
        if tokens:
            return tokens[0], tokens[1:]

    return command_str, args


def _parse_server(name: str, raw: Dict[str, Any]) -> McpServerInfo:
    server_type = str(raw.get("type") or raw.get("transport") or "").strip().lower()
    command, args = _normalize_command(raw.get("command"), raw.get("args"))
    url = str(raw.get("url") or raw.get("uri") or "").strip() or None

    if not server_type:
        if url:
            server_type = "sse"
        elif command:
            server_type = "stdio"
        else:
            server_type = "stdio"

    description = str(raw.get("description") or "")
    env = _ensure_str_dict(raw.get("env"))
    headers = _ensure_str_dict(raw.get("headers"))
    instructions = raw.get("instructions")

    return McpServerInfo(
        name=name,
        type=server_type,
        url=url,
        description=description,
        command=command,
        args=[str(a) for a in args] if args else [],
        env=env,
        headers=headers,
        instructions=str(instructions) if isinstance(instructions, str) else None,
    )


def _parse_servers(data: Dict[str, Any]) -> Dict[str, McpServerInfo]:
    servers: Dict[str, McpServerInfo] = {}
    for key in ("servers", "mcpServers"):
        raw_servers = data.get(key)
        if not isinstance(raw_servers, dict):
            continue
        for name, raw in raw_servers.items():
            if not isinstance(raw, dict):
                continue
            server_name = str(name).strip()
            if not server_name:
                continue
            servers[server_name] = _parse_server(server_name, raw)
    return servers


def _load_server_configs(project_path: Optional[Path]) -> Dict[str, McpServerInfo]:
    project_path = project_path or Path.cwd()
    candidates = [
        Path.home() / ".ripperdoc" / "mcp.json",
        Path.home() / ".mcp.json",
        project_path / ".ripperdoc" / "mcp.json",
        project_path / ".mcp.json",
    ]

    merged: Dict[str, McpServerInfo] = {}
    for path in candidates:
        data = _load_json_file(path)
        merged.update(_parse_servers(data))
    logger.debug(
        "[mcp] Loaded MCP server configs",
        extra={
            "project_path": str(project_path),
            "server_count": len(merged),
            "candidates": [str(path) for path in candidates],
        },
    )
    return merged


class McpRuntime:
    """Manages live MCP connections for the current event loop."""

    def __init__(self, project_path: Path):
        self.project_path = project_path
        self._exit_stack = AsyncExitStack()
        self.sessions: Dict[str, ClientSession] = {}
        self.servers: List[McpServerInfo] = []
        self._closed = False

    async def connect(self, configs: Dict[str, McpServerInfo]) -> List[McpServerInfo]:
        logger.info(
            "[mcp] Connecting to MCP servers",
            extra={
                "project_path": str(self.project_path),
                "server_count": len(configs),
                "servers": list(configs.keys()),
            },
        )
        await self._exit_stack.__aenter__()
        if not MCP_AVAILABLE:
            for config in configs.values():
                self.servers.append(
                    replace(
                        config,
                        status="unavailable",
                        error="MCP Python SDK not installed; install `mcp[cli]` with Python 3.10+.",
                    )
                )
            return self.servers

        for config in configs.values():
            self.servers.append(await self._connect_server(config))
        logger.debug(
            "[mcp] MCP connection summary",
            extra={
                "connected": [s.name for s in self.servers if s.status == "connected"],
                "failed": [s.name for s in self.servers if s.status == "failed"],
                "unavailable": [s.name for s in self.servers if s.status == "unavailable"],
            },
        )
        return self.servers

    async def _list_roots_callback(self, *_: Any, **__: Any) -> Optional[Any]:
        if not mcp_types:
            return None
        return mcp_types.ListRootsResult(
            roots=[mcp_types.Root(uri=Path(self.project_path).resolve().as_uri())]  # type: ignore[arg-type]
        )

    async def _connect_server(self, config: McpServerInfo) -> McpServerInfo:
        info = replace(config, tools=[], resources=[])
        if not MCP_AVAILABLE or not mcp_types:
            info.status = "unavailable"
            info.error = "MCP Python SDK not installed."
            return info

        try:
            read_stream = None
            write_stream = None
            logger.debug(
                "[mcp] Connecting server",
                extra={
                    "server": config.name,
                    "type": config.type,
                    "command": config.command,
                    "url": config.url,
                },
            )

            if config.type in ("sse", "sse-ide"):
                if not config.url:
                    raise ValueError("SSE MCP server requires a 'url'.")
                read_stream, write_stream = await self._exit_stack.enter_async_context(
                    sse_client(config.url, headers=config.headers or None)
                )
            elif config.type in ("http", "streamable-http"):
                if not config.url:
                    raise ValueError("HTTP MCP server requires a 'url'.")
                read_stream, write_stream, _ = await self._exit_stack.enter_async_context(
                    streamablehttp_client(
                        url=config.url,
                        headers=config.headers or None,
                        terminate_on_close=True,
                    )
                )
            else:
                if not config.command:
                    raise ValueError("Stdio MCP server requires a 'command'.")
                stdio_params = StdioServerParameters(
                    command=config.command,
                    args=config.args,
                    env=config.env or None,
                    cwd=self.project_path,
                )
                read_stream, write_stream = await self._exit_stack.enter_async_context(
                    stdio_client(stdio_params)
                )

            if read_stream is None or write_stream is None:
                raise ValueError("Failed to create read/write streams for MCP server")

            session = await self._exit_stack.enter_async_context(
                ClientSession(
                    read_stream,
                    write_stream,
                    list_roots_callback=self._list_roots_callback,  # type: ignore[arg-type]
                    client_info=mcp_types.Implementation(name="ripperdoc", version=__version__),
                )
            )

            init_result = await session.initialize()
            capabilities = session.get_server_capabilities()
            if capabilities is None:
                capabilities = mcp_types.ServerCapabilities()

            info.status = "connected"
            info.instructions = init_result.instructions or info.instructions
            info.server_version = getattr(init_result.serverInfo, "version", None)
            info.capabilities = (
                capabilities.model_dump() if hasattr(capabilities, "model_dump") else {}
            )
            self.sessions[config.name] = session

            tools_result = await session.list_tools()
            info.tools = [
                McpToolInfo(
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=tool.inputSchema,
                    annotations=(tool.annotations.model_dump() if tool.annotations else {}),
                )
                for tool in tools_result.tools
            ]

            if capabilities and getattr(capabilities, "resources", None):
                resources_result = await session.list_resources()
                info.resources = [
                    McpResourceInfo(
                        uri=str(resource.uri),
                        name=resource.name,
                        description=resource.description or "",
                        mime_type=resource.mimeType,
                        size=resource.size,
                    )
                    for resource in resources_result.resources
                ]

            logger.info(
                "[mcp] Connected to MCP server",
                extra={
                    "server": config.name,
                    "status": info.status,
                    "tools": len(info.tools),
                    "resources": len(info.resources),
                    "capabilities": list(info.capabilities.keys()),
                },
            )
        except (
            OSError,
            RuntimeError,
            ConnectionError,
            ValueError,
            TimeoutError,
        ) as exc:  # pragma: no cover - network/process errors
            logger.warning(
                "Failed to connect to MCP server: %s: %s",
                type(exc).__name__,
                exc,
                extra={"server": config.name},
            )
            info.status = "failed"
            info.error = str(exc)

        return info

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        logger.debug(
            "[mcp] Shutting down MCP runtime",
            extra={"project_path": str(self.project_path), "session_count": len(self.sessions)},
        )
        try:
            await self._exit_stack.aclose()
        except BaseException as exc:  # pragma: no cover - defensive shutdown
            # Swallow noisy ExceptionGroups from stdio_client cancel scopes during exit.
            logger.debug(
                "[mcp] Suppressed MCP shutdown error",
                extra={"error": str(exc), "project_path": str(self.project_path)},
            )
        finally:
            self.sessions.clear()
            self.servers.clear()


_runtime_var: contextvars.ContextVar[Optional[McpRuntime]] = contextvars.ContextVar(
    "ripperdoc_mcp_runtime", default=None
)
# Fallback for synchronous contexts (e.g., run_until_complete) where contextvars
# don't propagate values back to the caller.
_global_runtime: Optional[McpRuntime] = None


def _get_runtime() -> Optional[McpRuntime]:
    runtime = _runtime_var.get()
    if runtime:
        return runtime
    return _global_runtime


def get_existing_mcp_runtime() -> Optional[McpRuntime]:
    """Return the current MCP runtime if it has already been initialized."""
    return _get_runtime()


async def ensure_mcp_runtime(project_path: Optional[Path] = None) -> McpRuntime:
    runtime = _get_runtime()
    project_path = project_path or Path.cwd()
    if runtime and not runtime._closed and runtime.project_path == project_path:
        _runtime_var.set(runtime)
        logger.debug(
            "[mcp] Reusing existing MCP runtime",
            extra={
                "project_path": str(project_path),
                "server_count": len(runtime.servers),
            },
        )
        return runtime

    if runtime:
        await runtime.aclose()

    runtime = McpRuntime(project_path)
    logger.debug(
        "[mcp] Creating MCP runtime",
        extra={"project_path": str(project_path)},
    )
    configs = _load_server_configs(project_path)
    await runtime.connect(configs)
    _runtime_var.set(runtime)
    # Keep a module-level reference so sync callers that hop event loops can reuse it.
    global _global_runtime
    _global_runtime = runtime
    return runtime


async def shutdown_mcp_runtime() -> None:
    runtime = _get_runtime()
    if not runtime:
        return
    try:
        await runtime.aclose()
    except BaseException as exc:  # pragma: no cover - defensive for ExceptionGroup
        logger.debug("[mcp] Suppressed MCP runtime shutdown error", extra={"error": str(exc)})
    _runtime_var.set(None)
    global _global_runtime
    _global_runtime = None


async def load_mcp_servers_async(project_path: Optional[Path] = None) -> List[McpServerInfo]:
    runtime = await ensure_mcp_runtime(project_path)
    return list(runtime.servers)


def _config_only_servers(project_path: Optional[Path]) -> List[McpServerInfo]:
    return list(_load_server_configs(project_path).values())


def load_mcp_servers(project_path: Optional[Path] = None) -> List[McpServerInfo]:
    """Synchronous wrapper primarily for legacy call sites."""
    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            runtime = _get_runtime()
            if runtime and runtime.servers:
                return list(runtime.servers)
            return _config_only_servers(project_path)
    except RuntimeError:
        pass

    return asyncio.run(load_mcp_servers_async(project_path))


def find_mcp_resource(
    servers: List[McpServerInfo],
    server_name: str,
    uri: str,
) -> Optional[McpResourceInfo]:
    server = next((s for s in servers if s.name == server_name), None)
    if not server:
        return None
    return next((r for r in server.resources if r.uri == uri), None)


def _summarize_tools(server: McpServerInfo) -> str:
    if not server.tools:
        return "no tools"
    names = [tool.name for tool in server.tools[:6]]
    suffix = ", ".join(names)
    if len(server.tools) > 6:
        suffix += f", and {len(server.tools) - 6} more"
    return suffix


def format_mcp_instructions(servers: List[McpServerInfo]) -> str:
    """Build a concise MCP instruction block for the system prompt."""
    if not servers:
        return ""

    lines: List[str] = [
        "MCP servers are available. Call tools via CallMcpTool by specifying server, tool, and arguments.",
        "Use ListMcpServers to inspect servers and ListMcpResources/ReadMcpResource when a server exposes resources.",
    ]

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
        elif server.error:
            lines.append(f"  Error: {server.error}")

    return "\n".join(lines)


def estimate_mcp_tokens(servers: List[McpServerInfo]) -> int:
    """Estimate token usage for MCP instructions."""
    mcp_text = format_mcp_instructions(servers)
    return estimate_tokens(mcp_text)


__all__ = [
    "McpServerInfo",
    "McpToolInfo",
    "McpResourceInfo",
    "get_existing_mcp_runtime",
    "load_mcp_servers",
    "load_mcp_servers_async",
    "ensure_mcp_runtime",
    "shutdown_mcp_runtime",
    "find_mcp_resource",
    "format_mcp_instructions",
    "estimate_mcp_tokens",
]
