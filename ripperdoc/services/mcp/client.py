"""
MCP client connection management — mirrors reference: services/mcp/client.ts.

Manages the lifecycle of MCP server connections: connect, reconnect,
discover tools/resources, circuit breaker, and session management.
"""

from __future__ import annotations

import asyncio
import contextvars
import os
import subprocess
import sys
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TextIO, Union, cast

from ripperdoc import __version__
from ripperdoc.services.mcp import config as _mcp_config_loader
from ripperdoc.services.mcp.types import (
    McpResourceInfo,
    McpServerInfo,
    McpToolInfo,
)
from ripperdoc.utils.filesystem.config_paths import config_dir_for_scope
from ripperdoc.utils.filesystem.path_utils import sanitize_project_path
from ripperdoc.utils.log import get_logger

logger = get_logger()

# Environment variable keys
_MCP_STDERR_MODE_ENV = "RIPPERDOC_MCP_STDERR_MODE"
_MCP_STDERR_MODE_DEFAULT = "log"
_MCP_CONNECT_TIMEOUT_SEC_ENV = "RIPPERDOC_MCP_CONNECT_TIMEOUT_SEC"
_MCP_CONNECT_TIMEOUT_SEC_DEFAULT = 8.0
_MCP_CIRCUIT_BREAKER_FAILURES_ENV = "RIPPERDOC_MCP_CIRCUIT_BREAKER_FAILURES"
_MCP_CIRCUIT_BREAKER_FAILURES_DEFAULT = 2
_MCP_CIRCUIT_BREAKER_COOLDOWN_SEC_ENV = "RIPPERDOC_MCP_CIRCUIT_BREAKER_COOLDOWN_SEC"
_MCP_CIRCUIT_BREAKER_COOLDOWN_SEC_DEFAULT = 30.0

# SDK MCP request sender (for in-process SDK-backed servers)
_sdk_mcp_request_sender_var: contextvars.ContextVar[
    Optional[Callable[[str, Dict[str, Any]], Any]]
] = contextvars.ContextVar("ripperdoc_sdk_mcp_request_sender", default=None)
_global_sdk_mcp_request_sender: (
    Optional[Callable[[str, Dict[str, Any]], Any]]
) = None

# Conditional MCP SDK import
from ripperdoc.services.mcp import types as _mcp_types
try:
    import mcp.types as mcp_types
    from mcp.client.session import ClientSession
    from mcp.client.sse import sse_client
    from mcp.client.stdio import StdioServerParameters, stdio_client
    from mcp.client.streamable_http import streamable_http_client
    _mcp_types.MCP_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    ClientSession = object  # type: ignore
MCP_AVAILABLE = _mcp_types.MCP_AVAILABLE


@dataclass
class _SdkMcpCallToolResult:
    """Result from an SDK-backed MCP tool call."""

    content: List[Dict[str, Any]]
    structuredContent: Optional[Dict[str, Any]] = None
    isError: bool = False


@dataclass
class _SdkMcpToolDefinition:
    """Tool definition from an SDK-backed MCP server."""

    name: str
    description: str = ""
    inputSchema: Optional[Dict[str, Any]] = None
    annotations: Optional[Dict[str, Any]] = None


@dataclass
class _SdkMcpListToolsResult:
    """Result from listing tools on an SDK-backed MCP server."""

    tools: list[_SdkMcpToolDefinition]


class _SdkMcpSession:
    """Minimal MCP client for SDK-backed in-process servers."""

    def __init__(
        self,
        server_name: str,
        sender: Callable[[str, dict[str, Any]], Any],
    ) -> None:
        self.server_name = server_name
        self._sender = sender
        self._request_id = 0

    async def _send_message(self, message: dict[str, Any]) -> dict[str, Any]:
        response = await self._sender(self.server_name, message)
        payload = response.get("mcp_response", response)
        if not isinstance(payload, dict):
            raise RuntimeError(f"Invalid MCP response for server '{self.server_name}'")
        error = payload.get("error")
        if isinstance(error, dict):
            raise RuntimeError(str(error.get("message") or "Unknown MCP error"))
        return payload

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    async def initialize(self) -> dict[str, Any]:
        rid = self._next_id()
        resp = await self._send_message(
            {
                "jsonrpc": "2.0",
                "id": rid,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "ripperdoc", "version": __version__},
                },
            }
        )
        await self._send_message(
            {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
        )
        return cast(dict[str, Any], resp.get("result") or {})

    async def list_tools(self) -> _SdkMcpListToolsResult:
        resp = await self._send_message(
            {"jsonrpc": "2.0", "id": self._next_id(), "method": "tools/list", "params": {}}
        )
        result = resp.get("result") or {}
        raw_tools = result.get("tools") or []
        tools: list[_SdkMcpToolDefinition] = []
        for tool in raw_tools:
            if not isinstance(tool, dict):
                continue
            tools.append(
                _SdkMcpToolDefinition(
                    name=str(tool.get("name") or ""),
                    description=str(tool.get("description") or ""),
                    inputSchema=_coerce_sdk_schema(tool.get("inputSchema")),
                    annotations=(
                        tool.get("annotations") if isinstance(tool.get("annotations"), dict) else {}
                    ),
                )
            )
        return _SdkMcpListToolsResult(tools=tools)

    async def call_tool(
        self, name: str, arguments: Optional[Dict[str, Any]] = None
    ) -> _SdkMcpCallToolResult:
        resp = await self._send_message(
            {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments or {}},
            }
        )
        result = resp.get("result") or {}
        content = result.get("content")
        return _SdkMcpCallToolResult(
            content=content if isinstance(content, list) else [],
            structuredContent=(
                result.get("structuredContent") if isinstance(result.get("structuredContent"), dict) else None
            ),
            isError=bool(result.get("is_error") or result.get("isError")),
        )


def _coerce_sdk_schema(value: Any) -> dict[str, Any]:
    """Coerce Agent SDK shorthand schemas into JSON Schema."""
    if value is None:
        return {}
    # Extract from _ripperdoc_ helpers
    from ripperdoc.utils.mcp import _coerce_sdk_schema as _coerce
    return _coerce(value)


# ── SDK request sender management ──────────────────────────────────────


def get_sdk_mcp_request_sender() -> (
    Optional[Callable[[str, Dict[str, Any]], Any]]
):
    sender = _sdk_mcp_request_sender_var.get()
    if sender is not None:
        return sender
    return _global_sdk_mcp_request_sender


def set_sdk_mcp_request_sender(
    sender: Optional[Callable[[str, Dict[str, Any]], Any]],
) -> None:
    global _global_sdk_mcp_request_sender
    _global_sdk_mcp_request_sender = sender
    _sdk_mcp_request_sender_var.set(sender)


def clear_sdk_mcp_request_sender() -> None:
    set_sdk_mcp_request_sender(None)


# ── Stderr management ─────────────────────────────────────────────────


def _mcp_stderr_mode() -> str:
    raw = os.getenv(_MCP_STDERR_MODE_ENV, _MCP_STDERR_MODE_DEFAULT)
    mode = str(raw or _MCP_STDERR_MODE_DEFAULT).strip().lower()
    return mode if mode in {"inherit", "stderr", "log", "silent", "off", "devnull"} else _MCP_STDERR_MODE_DEFAULT


def _sanitize_server_filename(server_name: str) -> str:
    import re
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", server_name.strip())
    return value or "unknown-server"


def _mcp_stderr_log_path(project_path: Path, server_name: str) -> Path:
    safe_project = sanitize_project_path(project_path)
    base_dir = config_dir_for_scope("user") / "logs" / "mcp_stderr" / safe_project
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / f"{_sanitize_server_filename(server_name)}.log"


def get_mcp_stderr_mode() -> str:
    return _mcp_stderr_mode()


def get_mcp_stderr_log_path(project_path: Path, server_name: str) -> Path:
    return _mcp_stderr_log_path(project_path, server_name)


# ── Circuit breaker ────────────────────────────────────────────────────


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
    return _read_positive_float_env(
        _MCP_CONNECT_TIMEOUT_SEC_ENV, _MCP_CONNECT_TIMEOUT_SEC_DEFAULT
    )


def _mcp_circuit_failures_threshold() -> int:
    return _read_positive_int_env(
        _MCP_CIRCUIT_BREAKER_FAILURES_ENV, _MCP_CIRCUIT_BREAKER_FAILURES_DEFAULT
    )


def _mcp_circuit_cooldown_sec() -> float:
    return _read_positive_float_env(
        _MCP_CIRCUIT_BREAKER_COOLDOWN_SEC_ENV, _MCP_CIRCUIT_BREAKER_COOLDOWN_SEC_DEFAULT
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
    project_path: Path, server_name: str, error: str, *, timeout: bool
) -> None:
    key = _mcp_circuit_key(project_path, server_name)
    state = _mcp_circuit_states.get(key) or _McpCircuitState()
    state.failure_count += 1
    state.last_error = error
    threshold = _mcp_circuit_failures_threshold()
    if timeout or state.failure_count >= threshold:
        state.open_until_monotonic = time.monotonic() + _mcp_circuit_cooldown_sec()
    _mcp_circuit_states[key] = state


# ── McpRuntime ──────────────────────────────────────────────────────────


class McpRuntime:
    """Manages live MCP connections for the current event loop."""

    def __init__(self, project_path: Path):
        self.project_path = project_path
        self._owner_loop = asyncio.get_running_loop()
        self._exit_stack = AsyncExitStack()
        self._exit_stack_lock = asyncio.Lock()
        self.sessions: Dict[str, Any] = {}
        self.servers: List[McpServerInfo] = []
        self._servers_lock = asyncio.Lock()
        self._connection_tasks: Dict[str, asyncio.Task[None]] = {}
        self._connect_started = False
        self._all_connections_finished = asyncio.Event()
        self._all_connections_finished.set()
        self._closed = False
        self._mcp_write_streams: List[Any] = []
        self._raw_async_generators: List[Any] = []
        self._mcp_stderr_logs: List[TextIO] = []

    def belongs_to_loop(self, loop: asyncio.AbstractEventLoop) -> bool:
        return self._owner_loop is loop

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
        if not _mcp_types.MCP_AVAILABLE:
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
            "[mcp] Connection summary",
            extra={
                "connected": [s.name for s in self.servers if s.status == "connected"],
                "failed": [s.name for s in self.servers if s.status == "failed"],
            },
        )
        return self.server_snapshot()

    def _start_connecting(self, configs: Dict[str, McpServerInfo]) -> None:
        if self._connect_started:
            return
        self._connect_started = True
        self._all_connections_finished.clear()
        self.servers = [
            replace(config, tools=[], resources=[], status="connecting", error=None, capabilities={})
            for config in configs.values()
        ]
        if not configs:
            self._all_connections_finished.set()
            return
        for config in configs.values():
            task = asyncio.create_task(self._connect_single_server(config))
            self._connection_tasks[config.name] = task
            task.add_done_callback(lambda _t, name=config.name: self._on_connect_done(name))

    def _on_connect_done(self, server_name: str) -> None:
        self._connection_tasks.pop(server_name, None)
        if not self._connection_tasks:
            self._all_connections_finished.set()

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
        remaining = _mcp_circuit_open_remaining_sec(self.project_path, config.name)
        if remaining > 0:
            state = _mcp_circuit_states.get(
                _mcp_circuit_key(self.project_path, config.name)
            )
            last_error = state.last_error if state else None
            msg = f"Circuit breaker open ({remaining:.1f}s remaining)."
            if last_error:
                msg = f"{msg} Last error: {last_error}"
            return replace(config, tools=[], resources=[], capabilities={}, status="failed", error=msg)

        timeout_sec = _mcp_connect_timeout_sec()
        try:
            if timeout_sec > 0:
                info = await asyncio.wait_for(self._connect_server(config), timeout=timeout_sec)
            else:
                info = await self._connect_server(config)
        except asyncio.TimeoutError:
            msg = f"Connection timed out after {timeout_sec:.3g}s."
            _record_mcp_server_failure(self.project_path, config.name, msg, timeout=True)
            return replace(config, tools=[], resources=[], capabilities={}, status="failed", error=msg)
        except Exception as exc:  # noqa: BLE001
            if isinstance(exc, asyncio.CancelledError):
                raise
            msg = str(exc)
            _record_mcp_server_failure(self.project_path, config.name, msg, timeout=False)
            return replace(config, tools=[], resources=[], capabilities={}, status="failed", error=msg)

        if info.status == "connected":
            _record_mcp_server_success(self.project_path, config.name)
        elif info.status == "failed":
            err_lower = (info.error or "").lower()
            _record_mcp_server_failure(
                self.project_path, config.name, info.error or "Connection failed.",
                timeout="timed out" in err_lower or "timeout" in err_lower,
            )
        return info

    async def wait_for_connections(self, timeout: Optional[float] = None) -> None:
        tasks = [t for t in self._connection_tasks.values() if not t.done()]
        if not tasks:
            return
        if timeout is None:
            await asyncio.gather(*tasks, return_exceptions=True)
            return
        done, pending = await asyncio.wait(tasks, timeout=timeout)
        for t in done:
            try:
                t.result()
            except Exception:
                continue

    def server_snapshot(self) -> List[McpServerInfo]:
        return list(self.servers)

    async def _list_roots_callback(self, *_: Any, **__: Any) -> Any:
        if not mcp_types:
            return None
        return mcp_types.ListRootsResult(
            roots=[mcp_types.Root(uri=Path(self.project_path).resolve().as_uri())]
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
        except (OSError, IOError, RuntimeError):
            return subprocess.DEVNULL
        self._mcp_stderr_logs.append(handle)
        return handle

    async def _connect_server(self, config: McpServerInfo) -> McpServerInfo:
        info = replace(config, tools=[], resources=[])
        if not _mcp_types.MCP_AVAILABLE or not mcp_types:
            info.status = "unavailable"
            info.error = "MCP Python SDK not installed."
            return info

        try:
            read_stream = write_stream = None

            if config.type == "sdk":
                sender = get_sdk_mcp_request_sender()
                if sender is None:
                    raise RuntimeError("SDK MCP transport is not available")
                session = _SdkMcpSession(config.name, sender)
                init_result = await session.initialize()
                info.status = "connected"
                info.instructions = cast(Optional[str], init_result.get("instructions")) or info.instructions
                server_info = init_result.get("serverInfo")
                if isinstance(server_info, dict):
                    info.server_version = str(server_info.get("version", "")) or None
                capabilities = init_result.get("capabilities")
                info.capabilities = capabilities if isinstance(capabilities, dict) else {}
                self.sessions[config.name] = session

                tools_result = await session.list_tools()
                info.tools = [
                    McpToolInfo(name=t.name, description=t.description, input_schema=t.inputSchema, annotations=t.annotations)
                    for t in tools_result.tools if t.name
                ]
                return info

            elif config.type in ("sse", "sse-ide"):
                if not config.url:
                    raise ValueError("SSE MCP server requires a 'url'.")
                cm = sse_client(config.url, headers=config.headers or None)
                if hasattr(cm, "gen"):
                    self._raw_async_generators.append(cm.gen)
                async with self._exit_stack_lock:
                    read_stream, write_stream = await self._exit_stack.enter_async_context(cm)
                self._mcp_write_streams.append(write_stream)

            elif config.type in ("http", "streamable-http"):
                if not config.url:
                    raise ValueError("HTTP MCP server requires a 'url'.")
                cm = streamable_http_client(url=config.url, terminate_on_close=True)
                if hasattr(cm, "gen"):
                    self._raw_async_generators.append(cm.gen)
                async with self._exit_stack_lock:
                    read_stream, write_stream, _ = await self._exit_stack.enter_async_context(cm)
                self._mcp_write_streams.append(write_stream)

            else:
                if not config.command:
                    raise ValueError("Stdio MCP server requires a 'command'.")
                stdio_params = StdioServerParameters(
                    command=config.command, args=config.args,
                    env=config.env or None, cwd=self.project_path,
                )
                cm = stdio_client(stdio_params, errlog=self._stdio_errlog_target(config.name))
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
                        read_stream, write_stream,
                        list_roots_callback=self._list_roots_callback,
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
            info.capabilities = capabilities.model_dump() if hasattr(capabilities, "model_dump") else {}
            self.sessions[config.name] = session

            tools_result = await session.list_tools()
            info.tools = [
                McpToolInfo(name=t.name, description=t.description, input_schema=t.inputSchema,
                            annotations=t.annotations.model_dump() if t.annotations else {})
                for t in tools_result.tools
            ]
            info.tools_discovered = True

            if capabilities and getattr(capabilities, "resources", None):
                try:
                    resources_result = await session.list_resources()
                    info.resources = [
                        McpResourceInfo(uri=str(r.uri), name=r.name,
                                        description=r.description or "", mime_type=r.mimeType, size=r.size)
                        for r in resources_result.resources
                    ]
                except Exception:  # noqa: BLE001
                    pass

            logger.info(
                "[mcp] Connected to MCP server",
                extra={"server": config.name, "tools": len(info.tools), "resources": len(info.resources)},
            )
        except (OSError, RuntimeError, ConnectionError, ValueError, TimeoutError) as exc:
            logger.warning("Failed to connect to MCP server: %s: %s", type(exc).__name__, exc,
                           extra={"server": config.name})
            info.status = "failed"
            info.error = str(exc)

        return info

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        logger.debug("[mcp] Shutting down MCP runtime", extra={"project_path": str(self.project_path)})

        tasks = [t for t in self._connection_tasks.values() if not t.done()]
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._connection_tasks.clear()
        self._all_connections_finished.set()

        for ws in self._mcp_write_streams:
            try:
                await ws.aclose()
            except BaseException:
                pass
        self._mcp_write_streams.clear()
        await asyncio.sleep(0.1)

        for gen in self._raw_async_generators:
            try:
                await gen.aclose()
            except BaseException:
                pass
        self._raw_async_generators.clear()

        for el in self._mcp_stderr_logs:
            try:
                el.close()
            except BaseException:
                pass
        self._mcp_stderr_logs.clear()

        try:
            await self._exit_stack.aclose()
        except BaseException as exc:
            logger.debug("[mcp] Suppressed shutdown error", extra={"error": str(exc)})

        self.sessions.clear()
        self.servers.clear()


# ── Global runtime management ──────────────────────────────────────────

_runtime_var: contextvars.ContextVar[Optional[McpRuntime]] = contextvars.ContextVar(
    "ripperdoc_mcp_runtime", default=None
)
_global_runtime: Optional[McpRuntime] = None
_runtime_init_task: Optional[asyncio.Task[McpRuntime]] = None
_runtime_init_project: Optional[Path] = None


def _current_loop_or_none() -> Optional[asyncio.AbstractEventLoop]:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


def _runtime_matches_current_loop(runtime: McpRuntime) -> bool:
    loop = _current_loop_or_none()
    return loop is not None and runtime.belongs_to_loop(loop)


def _clear_foreign_global_runtime_reference() -> None:
    global _global_runtime
    if _global_runtime is not None and not _runtime_matches_current_loop(_global_runtime):
        _global_runtime = None


def _get_runtime(*, require_current_loop: bool = False) -> Optional[McpRuntime]:
    runtime = _runtime_var.get()
    if runtime and (not require_current_loop or _runtime_matches_current_loop(runtime)):
        return runtime
    if _global_runtime and (not require_current_loop or _runtime_matches_current_loop(_global_runtime)):
        return _global_runtime
    return None


def get_existing_mcp_runtime(*, require_current_loop: bool = False) -> Optional[McpRuntime]:
    """Return the current MCP runtime if it has already been initialized."""
    return _get_runtime(require_current_loop=require_current_loop)


async def ensure_mcp_runtime(
    project_path: Optional[Path] = None,
    *,
    wait_for_connections: bool = False,
    wait_timeout: Optional[float] = None,
) -> McpRuntime:
    """Ensure an MCP runtime exists for the given project path, creating one if needed."""
    global _runtime_init_task, _runtime_init_project
    _clear_foreign_global_runtime_reference()
    runtime = _get_runtime(require_current_loop=True)
    project_path = project_path or Path.cwd()
    if runtime and not runtime._closed and runtime.project_path == project_path:
        _runtime_var.set(runtime)
        if wait_for_connections:
            await runtime.wait_for_connections(timeout=wait_timeout)
        return runtime

    if _runtime_init_task is not None and not _runtime_init_task.done():
        if _runtime_init_project == project_path and _runtime_init_task.get_loop() is asyncio.get_running_loop():
            runtime = await _runtime_init_task
            _runtime_var.set(runtime)
            if wait_for_connections:
                await runtime.wait_for_connections(timeout=wait_timeout)
            return runtime

    async def _initialize_runtime() -> McpRuntime:
        existing = _get_runtime(require_current_loop=True)
        if existing and not existing._closed:
            await existing.aclose()
        initialized = McpRuntime(project_path)
        try:
            configs = _mcp_config_loader.load_server_configs(project_path)
            await initialized.connect(configs, wait_for_connections=False)
            _runtime_var.set(initialized)
            global _global_runtime
            _global_runtime = initialized

            loop = asyncio.get_running_loop()
            original_handler = loop.get_exception_handler()

            def _mcp_exc_handler(loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
                asyncgen = context.get("asyncgen")
                if asyncgen and "stdio_client" in str(asyncgen):
                    logger.debug("[mcp] Suppressed asyncgen cleanup error")
                    return
                if original_handler:
                    original_handler(loop, context)
                else:
                    loop.default_exception_handler(context)

            loop.set_exception_handler(_mcp_exc_handler)
            return initialized
        except BaseException:
            try:
                await initialized.aclose()
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
    """Shut down the current MCP runtime."""
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

    _clear_foreign_global_runtime_reference()
    runtime = _get_runtime(require_current_loop=True)
    if not runtime:
        return
    try:
        await runtime.aclose()
    except BaseException:
        pass
    _runtime_var.set(None)
    global _global_runtime
    _global_runtime = None
