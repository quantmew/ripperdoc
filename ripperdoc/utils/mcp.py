"""MCP configuration loader, connection manager, and prompt helpers."""

from __future__ import annotations

import asyncio
import contextvars
import json
import os
import re
import shlex
import subprocess
import sys
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

from ripperdoc import __version__
from ripperdoc.core.plugins import discover_plugins, expand_plugin_root_vars
from ripperdoc.utils.log import get_logger
from ripperdoc.utils.path_utils import sanitize_project_path
from ripperdoc.utils.token_estimation import estimate_tokens

logger = get_logger()
_MCP_STDERR_MODE_ENV = "RIPPERDOC_MCP_STDERR_MODE"
_MCP_STDERR_MODE_DEFAULT = "log"
_MCP_CONNECT_TIMEOUT_SEC_ENV = "RIPPERDOC_MCP_CONNECT_TIMEOUT_SEC"
_MCP_CONNECT_TIMEOUT_SEC_DEFAULT = 8.0
_MCP_CIRCUIT_BREAKER_FAILURES_ENV = "RIPPERDOC_MCP_CIRCUIT_BREAKER_FAILURES"
_MCP_CIRCUIT_BREAKER_FAILURES_DEFAULT = 2
_MCP_CIRCUIT_BREAKER_COOLDOWN_SEC_ENV = "RIPPERDOC_MCP_CIRCUIT_BREAKER_COOLDOWN_SEC"
_MCP_CIRCUIT_BREAKER_COOLDOWN_SEC_DEFAULT = 30.0


try:
    import mcp.types as mcp_types  # type: ignore[import-not-found]
    from mcp.client.session import ClientSession  # type: ignore[import-not-found]
    from mcp.client.sse import sse_client  # type: ignore[import-not-found]
    from mcp.client.stdio import StdioServerParameters, stdio_client  # type: ignore[import-not-found]
    from mcp.client.streamable_http import streamable_http_client  # type: ignore[import-not-found]

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
    if servers:
        return servers

    # Support direct top-level map of server_name -> config.
    for name, raw in data.items():
        if not isinstance(raw, dict):
            continue
        if not any(key in raw for key in ("command", "args", "url", "uri", "type", "transport")):
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

    plugin_result = discover_plugins(project_path=project_path)
    for plugin_error in plugin_result.errors:
        logger.warning(
            "[mcp] Plugin discovery warning: %s (%s)",
            plugin_error.path,
            plugin_error.reason,
        )

    for plugin in plugin_result.plugins:
        for mcp_path in plugin.mcp_paths:
            resolved_path = mcp_path
            if resolved_path.is_dir():
                if (resolved_path / ".mcp.json").exists():
                    resolved_path = resolved_path / ".mcp.json"
                elif (resolved_path / "mcp.json").exists():
                    resolved_path = resolved_path / "mcp.json"
            data = _load_json_file(resolved_path)
            if not data:
                continue
            expanded = expand_plugin_root_vars(data, plugin.root)
            if isinstance(expanded, dict):
                merged.update(_parse_servers(expanded))

        for inline_entry in plugin.mcp_inline:
            expanded_inline = expand_plugin_root_vars(inline_entry, plugin.root)
            if isinstance(expanded_inline, dict):
                merged.update(_parse_servers(expanded_inline))

    logger.debug(
        "[mcp] Loaded MCP server configs",
        extra={
            "project_path": str(project_path),
            "server_count": len(merged),
            "candidates": [str(path) for path in candidates],
        },
    )
    return merged


def _mcp_stderr_mode() -> str:
    raw = os.getenv(_MCP_STDERR_MODE_ENV, _MCP_STDERR_MODE_DEFAULT)
    mode = str(raw or _MCP_STDERR_MODE_DEFAULT).strip().lower()
    if mode in {"inherit", "stderr", "log", "silent", "off", "devnull"}:
        return mode
    return _MCP_STDERR_MODE_DEFAULT


def _sanitize_server_filename(server_name: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", server_name.strip())
    return value or "unknown-server"


def _mcp_stderr_log_path(project_path: Path, server_name: str) -> Path:
    safe_project = sanitize_project_path(project_path)
    base_dir = Path.home() / ".ripperdoc" / "logs" / "mcp_stderr" / safe_project
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / f"{_sanitize_server_filename(server_name)}.log"


def get_mcp_stderr_mode() -> str:
    """Return effective MCP stdio stderr routing mode."""
    return _mcp_stderr_mode()


def get_mcp_stderr_log_path(project_path: Path, server_name: str) -> Path:
    """Return log file path used for a server's stdio stderr stream."""
    return _mcp_stderr_log_path(project_path, server_name)


@dataclass
class _McpCircuitState:
    failure_count: int = 0
    open_until_monotonic: float = 0.0
    last_error: Optional[str] = None


_mcp_circuit_states: Dict[str, _McpCircuitState] = {}


def _read_positive_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _read_positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _mcp_connect_timeout_sec() -> float:
    return _read_positive_float_env(_MCP_CONNECT_TIMEOUT_SEC_ENV, _MCP_CONNECT_TIMEOUT_SEC_DEFAULT)


def _mcp_circuit_failures_threshold() -> int:
    return _read_positive_int_env(
        _MCP_CIRCUIT_BREAKER_FAILURES_ENV,
        _MCP_CIRCUIT_BREAKER_FAILURES_DEFAULT,
    )


def _mcp_circuit_cooldown_sec() -> float:
    return _read_positive_float_env(
        _MCP_CIRCUIT_BREAKER_COOLDOWN_SEC_ENV,
        _MCP_CIRCUIT_BREAKER_COOLDOWN_SEC_DEFAULT,
    )


def _mcp_circuit_key(project_path: Path, server_name: str) -> str:
    try:
        project_token = str(project_path.resolve())
    except (OSError, RuntimeError):
        project_token = str(project_path)
    return f"{project_token}::{server_name}"


def _mcp_circuit_open_remaining_sec(project_path: Path, server_name: str) -> float:
    state = _mcp_circuit_states.get(_mcp_circuit_key(project_path, server_name))
    if not state:
        return 0.0
    return max(state.open_until_monotonic - time.monotonic(), 0.0)


def _record_mcp_server_success(project_path: Path, server_name: str) -> None:
    _mcp_circuit_states.pop(_mcp_circuit_key(project_path, server_name), None)


def _record_mcp_server_failure(
    project_path: Path,
    server_name: str,
    error: str,
    *,
    timeout: bool,
) -> None:
    key = _mcp_circuit_key(project_path, server_name)
    state = _mcp_circuit_states.get(key) or _McpCircuitState()
    state.failure_count += 1
    state.last_error = error
    threshold = _mcp_circuit_failures_threshold()
    if timeout or state.failure_count >= threshold:
        state.open_until_monotonic = time.monotonic() + _mcp_circuit_cooldown_sec()
    _mcp_circuit_states[key] = state


class McpRuntime:
    """Manages live MCP connections for the current event loop."""

    def __init__(self, project_path: Path):
        self.project_path = project_path
        self._exit_stack = AsyncExitStack()
        self._exit_stack_lock = asyncio.Lock()
        self.sessions: Dict[str, ClientSession] = {}
        self.servers: List[McpServerInfo] = []
        self._servers_lock = asyncio.Lock()
        self._connection_tasks: Dict[str, asyncio.Task[None]] = {}
        self._connect_started = False
        self._all_connections_finished = asyncio.Event()
        self._all_connections_finished.set()
        self._closed = False
        # Track MCP streams for proper cleanup ordering
        # We need to close write streams BEFORE exiting the stdio_client context
        # to allow the internal tasks to exit cleanly
        self._mcp_write_streams: List[Any] = []
        # Track the underlying async generators from @asynccontextmanager wrappers
        # These need to be explicitly closed after exit stack cleanup to prevent
        # shutdown_asyncgens() from trying to close them in a different task
        self._raw_async_generators: List[Any] = []
        # Keep opened stderr log handles for stdio MCP servers.
        self._mcp_stderr_logs: List[TextIO] = []

    async def connect(
        self,
        configs: Dict[str, McpServerInfo],
        *,
        wait_for_connections: bool = False,
        wait_timeout: Optional[float] = None,
    ) -> List[McpServerInfo]:
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
            self._all_connections_finished.set()
            for config in configs.values():
                self.servers.append(
                    replace(
                        config,
                        status="unavailable",
                        error="MCP Python SDK not installed; install `mcp[cli]` with Python 3.10+.",
                    )
                )
            return self.server_snapshot()

        self._start_connecting(configs)

        if wait_for_connections:
            await self.wait_for_connections(timeout=wait_timeout)

        logger.debug(
            "[mcp] MCP connection summary",
            extra={
                "connected": [s.name for s in self.servers if s.status == "connected"],
                "connecting": [s.name for s in self.servers if s.status == "connecting"],
                "failed": [s.name for s in self.servers if s.status == "failed"],
                "unavailable": [s.name for s in self.servers if s.status == "unavailable"],
            },
        )
        return self.server_snapshot()

    def _start_connecting(self, configs: Dict[str, McpServerInfo]) -> None:
        if self._connect_started:
            return
        self._connect_started = True
        self._all_connections_finished.clear()
        self.servers = [
            replace(
                config,
                tools=[],
                resources=[],
                status="connecting",
                error=None,
                capabilities={},
            )
            for config in configs.values()
        ]

        if not configs:
            self._all_connections_finished.set()
            return

        for config in configs.values():
            task = asyncio.create_task(self._connect_single_server(config))
            self._connection_tasks[config.name] = task
            task.add_done_callback(lambda done_task, server_name=config.name: self._on_connect_done(server_name, done_task))

    def _on_connect_done(self, server_name: str, task: asyncio.Task[None]) -> None:
        self._connection_tasks.pop(server_name, None)
        if not self._connection_tasks:
            self._all_connections_finished.set()
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.warning(
                "[mcp] Server connection task failed unexpectedly: %s: %s",
                type(error).__name__,
                error,
                extra={"server": server_name},
            )

    async def _connect_single_server(self, config: McpServerInfo) -> None:
        info = await self._connect_server_with_policy(config)
        async with self._servers_lock:
            for idx, server in enumerate(self.servers):
                if server.name == info.name:
                    self.servers[idx] = info
                    break
            else:
                self.servers.append(info)

    async def _connect_server_with_policy(self, config: McpServerInfo) -> McpServerInfo:
        remaining_open = _mcp_circuit_open_remaining_sec(self.project_path, config.name)
        if remaining_open > 0:
            state = _mcp_circuit_states.get(_mcp_circuit_key(self.project_path, config.name))
            last_error = state.last_error if state else None
            message = f"Circuit breaker open ({remaining_open:.1f}s remaining)."
            if last_error:
                message = f"{message} Last error: {last_error}"
            return replace(
                config,
                tools=[],
                resources=[],
                capabilities={},
                status="failed",
                error=message,
            )

        timeout_sec = _mcp_connect_timeout_sec()
        try:
            if timeout_sec > 0:
                info = await asyncio.wait_for(self._connect_server(config), timeout=timeout_sec)
            else:
                info = await self._connect_server(config)
        except asyncio.TimeoutError:
            message = f"Connection timed out after {timeout_sec:.3g}s."
            _record_mcp_server_failure(
                self.project_path,
                config.name,
                message,
                timeout=True,
            )
            logger.warning(
                "[mcp] MCP server connect timeout",
                extra={
                    "server": config.name,
                    "timeout_sec": timeout_sec,
                },
            )
            return replace(
                config,
                tools=[],
                resources=[],
                capabilities={},
                status="failed",
                error=message,
            )
        except Exception as exc:  # noqa: BLE001 - isolate single-server failures
            if isinstance(exc, asyncio.CancelledError):
                raise
            message = str(exc)
            _record_mcp_server_failure(
                self.project_path,
                config.name,
                message,
                timeout=False,
            )
            logger.warning(
                "[mcp] Unexpected MCP server connect error: %s: %s",
                type(exc).__name__,
                exc,
                extra={"server": config.name},
            )
            return replace(
                config,
                tools=[],
                resources=[],
                capabilities={},
                status="failed",
                error=message,
            )

        if info.status == "connected":
            _record_mcp_server_success(self.project_path, config.name)
        elif info.status == "failed":
            lower_error = (info.error or "").lower()
            _record_mcp_server_failure(
                self.project_path,
                config.name,
                info.error or "Connection failed.",
                timeout="timed out" in lower_error or "timeout" in lower_error,
            )
        return info

    async def wait_for_connections(self, timeout: Optional[float] = None) -> None:
        tasks = [task for task in self._connection_tasks.values() if not task.done()]
        if not tasks:
            return
        if timeout is None:
            await asyncio.gather(*tasks, return_exceptions=True)
            return
        done, pending = await asyncio.wait(tasks, timeout=timeout)
        for task in done:
            try:
                task.result()
            except Exception:  # pragma: no cover - logged in callback
                continue
        if pending:
            logger.debug(
                "[mcp] wait_for_connections timeout",
                extra={"pending_servers": len(pending), "timeout": timeout},
            )

    def server_snapshot(self) -> List[McpServerInfo]:
        return list(self.servers)

    async def _list_roots_callback(self, *_: Any, **__: Any) -> Optional[Any]:
        if not mcp_types:
            return None
        return mcp_types.ListRootsResult(
            roots=[mcp_types.Root(uri=Path(self.project_path).resolve().as_uri())]  # type: ignore[arg-type]
        )

    def _stdio_errlog_target(self, server_name: str) -> Any:
        mode = _mcp_stderr_mode()
        if mode in {"inherit", "stderr"}:
            return sys.stderr
        if mode in {"silent", "off", "devnull"}:
            return subprocess.DEVNULL

        path = _mcp_stderr_log_path(self.project_path, server_name)
        try:
            handle = path.open("a", encoding="utf-8", buffering=1)
        except (OSError, IOError, RuntimeError) as exc:
            logger.warning(
                "[mcp] Failed to open stderr log; falling back to /dev/null: %s: %s",
                type(exc).__name__,
                exc,
                extra={"server": server_name, "path": str(path)},
            )
            return subprocess.DEVNULL
        self._mcp_stderr_logs.append(handle)
        logger.debug(
            "[mcp] Redirecting stdio server stderr to log file",
            extra={"server": server_name, "path": str(path)},
        )
        return handle

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
                cm = sse_client(config.url, headers=config.headers or None)
                # Track the underlying async generator for explicit cleanup
                if hasattr(cm, "gen"):
                    self._raw_async_generators.append(cm.gen)
                async with self._exit_stack_lock:
                    read_stream, write_stream = await self._exit_stack.enter_async_context(cm)
                self._mcp_write_streams.append(write_stream)
            elif config.type in ("http", "streamable-http"):
                if not config.url:
                    raise ValueError("HTTP MCP server requires a 'url'.")
                cm = streamable_http_client(  # type: ignore[call-arg]
                    url=config.url,
                    terminate_on_close=True,
                )
                # Track the underlying async generator for explicit cleanup
                if hasattr(cm, "gen"):
                    self._raw_async_generators.append(cm.gen)
                async with self._exit_stack_lock:
                    read_stream, write_stream, _ = await self._exit_stack.enter_async_context(cm)
                self._mcp_write_streams.append(write_stream)
            else:
                if not config.command:
                    raise ValueError("Stdio MCP server requires a 'command'.")
                stdio_params = StdioServerParameters(
                    command=config.command,
                    args=config.args,
                    env=config.env or None,
                    cwd=self.project_path,
                )
                cm = stdio_client(stdio_params, errlog=self._stdio_errlog_target(config.name))
                # Track the underlying async generator for explicit cleanup
                if hasattr(cm, "gen"):
                    self._raw_async_generators.append(cm.gen)
                async with self._exit_stack_lock:
                    read_stream, write_stream = await self._exit_stack.enter_async_context(cm)
                self._mcp_write_streams.append(write_stream)

            if read_stream is None or write_stream is None:
                raise ValueError("Failed to create read/write streams for MCP server")

            async with self._exit_stack_lock:
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

        connection_tasks = [task for task in self._connection_tasks.values() if not task.done()]
        for task in connection_tasks:
            task.cancel()
        if connection_tasks:
            await asyncio.gather(*connection_tasks, return_exceptions=True)
        self._connection_tasks.clear()
        self._all_connections_finished.set()

        # CRITICAL: Close all MCP write streams FIRST to signal internal tasks to stop.
        for write_stream in self._mcp_write_streams:
            try:
                await write_stream.aclose()
            except BaseException:  # pragma: no cover
                pass
        self._mcp_write_streams.clear()

        # Small delay to allow internal tasks to notice stream closure and exit
        await asyncio.sleep(0.1)

        # CRITICAL: Close the raw async generators BEFORE the exit stack cleanup.
        # This prevents asyncio's shutdown_asyncgens() from trying to close them
        # later, which would cause the "cancel scope in different task" error.
        for gen in self._raw_async_generators:
            try:
                await gen.aclose()
            except BaseException:  # pragma: no cover
                pass
        self._raw_async_generators.clear()

        for errlog in self._mcp_stderr_logs:
            try:
                errlog.close()
            except BaseException:  # pragma: no cover
                pass
        self._mcp_stderr_logs.clear()

        # Now close the exit stack
        try:
            await self._exit_stack.aclose()
        except BaseException as exc:  # pragma: no cover - defensive shutdown
            logger.debug(
                "[mcp] Suppressed MCP shutdown error during exit_stack.aclose()",
                extra={"error": str(exc), "project_path": str(self.project_path)},
            )

        self.sessions.clear()
        self.servers.clear()


_runtime_var: contextvars.ContextVar[Optional[McpRuntime]] = contextvars.ContextVar(
    "ripperdoc_mcp_runtime", default=None
)
# Fallback for synchronous contexts (e.g., run_until_complete) where contextvars
# don't propagate values back to the caller.
_global_runtime: Optional[McpRuntime] = None
_runtime_init_task: Optional[asyncio.Task[McpRuntime]] = None
_runtime_init_project: Optional[Path] = None


def _get_runtime() -> Optional[McpRuntime]:
    runtime = _runtime_var.get()
    if runtime:
        return runtime
    return _global_runtime


def get_existing_mcp_runtime() -> Optional[McpRuntime]:
    """Return the current MCP runtime if it has already been initialized."""
    return _get_runtime()


async def ensure_mcp_runtime(
    project_path: Optional[Path] = None,
    *,
    wait_for_connections: bool = False,
    wait_timeout: Optional[float] = None,
) -> McpRuntime:
    global _runtime_init_task, _runtime_init_project
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
        if wait_for_connections:
            await runtime.wait_for_connections(timeout=wait_timeout)
        return runtime

    # If an initialization task is already in flight for this project and loop, await it.
    if _runtime_init_task is not None and not _runtime_init_task.done():
        if (
            _runtime_init_project == project_path
            and _runtime_init_task.get_loop() is asyncio.get_running_loop()
        ):
            runtime = await _runtime_init_task
            _runtime_var.set(runtime)
            if wait_for_connections:
                await runtime.wait_for_connections(timeout=wait_timeout)
            return runtime

    async def _initialize_runtime() -> McpRuntime:
        existing = _get_runtime()
        if existing and not existing._closed:
            await existing.aclose()

        initialized_runtime = McpRuntime(project_path)
        try:
            logger.debug(
                "[mcp] Creating MCP runtime",
                extra={"project_path": str(project_path)},
            )
            configs = _load_server_configs(project_path)
            await initialized_runtime.connect(configs, wait_for_connections=False)
            _runtime_var.set(initialized_runtime)
            # Keep a module-level reference so sync callers that hop event loops can reuse it.
            global _global_runtime
            _global_runtime = initialized_runtime

            # Install custom exception handler to suppress MCP asyncgen cleanup errors.
            # These errors occur due to anyio cancel scope issues when stdio_client async
            # generators are finalized by Python's asyncgen hooks. The errors are harmless
            # but noisy, so we suppress them here.
            loop = asyncio.get_running_loop()
            original_handler = loop.get_exception_handler()

            def mcp_exception_handler(loop: asyncio.AbstractEventLoop, context: Dict[str, Any]) -> None:
                asyncgen = context.get("asyncgen")
                # Suppress MCP stdio_client asyncgen cleanup errors
                if asyncgen and "stdio_client" in str(asyncgen):
                    logger.debug("[mcp] Suppressed asyncgen cleanup error for stdio_client")
                    return
                # Call original handler for other errors
                if original_handler:
                    original_handler(loop, context)
                else:
                    loop.default_exception_handler(context)

            loop.set_exception_handler(mcp_exception_handler)
            logger.debug("[mcp] Installed custom exception handler for asyncgen cleanup")
            return initialized_runtime
        except BaseException:
            # Ensure partially connected runtimes are cleaned up when initialization is cancelled/failed.
            try:
                await initialized_runtime.aclose()
            except BaseException:
                pass
            raise

    init_task = asyncio.create_task(_initialize_runtime())
    _runtime_init_task = init_task
    _runtime_init_project = project_path
    try:
        runtime = await init_task
        _runtime_var.set(runtime)
        if wait_for_connections:
            await runtime.wait_for_connections(timeout=wait_timeout)
        return runtime
    finally:
        if _runtime_init_task is init_task:
            _runtime_init_task = None
            _runtime_init_project = None


async def shutdown_mcp_runtime() -> None:
    global _runtime_init_task, _runtime_init_project
    if _runtime_init_task is not None and not _runtime_init_task.done():
        if _runtime_init_task.get_loop() is asyncio.get_running_loop():
            _runtime_init_task.cancel()
            try:
                await _runtime_init_task
            except (asyncio.CancelledError, RuntimeError, OSError, ConnectionError, ValueError):
                pass
        _runtime_init_task = None
        _runtime_init_project = None

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


async def load_mcp_servers_async(
    project_path: Optional[Path] = None,
    *,
    wait_for_connections: bool = False,
    wait_timeout: Optional[float] = None,
) -> List[McpServerInfo]:
    runtime = await ensure_mcp_runtime(
        project_path,
        wait_for_connections=wait_for_connections,
        wait_timeout=wait_timeout,
    )
    return runtime.server_snapshot()


def _config_only_servers(project_path: Optional[Path]) -> List[McpServerInfo]:
    return list(_load_server_configs(project_path).values())


def load_mcp_servers(
    project_path: Optional[Path] = None,
    *,
    wait_for_connections: bool = True,
    wait_timeout: Optional[float] = None,
) -> List[McpServerInfo]:
    """Synchronous wrapper primarily for legacy call sites."""
    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            runtime = _get_runtime()
            if runtime and runtime.servers:
                return runtime.server_snapshot()
            return _config_only_servers(project_path)
    except RuntimeError:
        pass

    return asyncio.run(
        load_mcp_servers_async(
            project_path,
            wait_for_connections=wait_for_connections,
            wait_timeout=wait_timeout,
        )
    )


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

    connected_count = len([server for server in servers if server.status == "connected"])
    lines: List[str] = []
    if connected_count > 0:
        lines.append(
            "Connected MCP servers are available. Call tools via CallMcpTool by specifying server, tool, and arguments."
        )
    else:
        lines.append(
            "MCP servers are configured, but none are connected yet. Prefer non-MCP tools unless a server is [connected]."
        )
    lines.append(
        "Use ListMcpServers to inspect statuses and ListMcpResources/ReadMcpResource when a server exposes resources."
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
    "get_mcp_stderr_mode",
    "get_mcp_stderr_log_path",
]
