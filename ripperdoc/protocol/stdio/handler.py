"""Stdio protocol handler implementation."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from ripperdoc.core.hooks.config import HooksConfig
from ripperdoc.core.query import QueryContext
from ripperdoc.utils.sessions.session_history import SessionHistory
from ripperdoc.utils.mcp import McpServerInfo

from .handler_config import StdioConfigMixin
from .handler_control import StdioControlMixin
from .handler_io import StdioIOMixin
from .handler_message import StdioMessageMixin
from .handler_query import StdioQueryMixin
from .handler_runtime import StdioRuntimeMixin
from .handler_session import StdioSessionMixin
from .handler_io import _SDKWebSocketTransport

logger = logging.getLogger(__name__)


class StdioProtocolHandler(
    StdioIOMixin,
    StdioConfigMixin,
    StdioMessageMixin,
    StdioSessionMixin,
    StdioQueryMixin,
    StdioControlMixin,
    StdioRuntimeMixin,
):
    """Handler for stdio-based JSON Control Protocol.

    This class manages bidirectional communication with the SDK:
    - Reads JSON commands from stdin
    - Parses control requests (initialize, query, etc.)
    - Executes core query logic
    - Writes JSON responses to stdout

    Following Claude SDK's elegant patterns:
    - JSON messages separated by newlines
    - Control requests/responses for protocol management
    - Message streaming for query results
    """

    _PERMISSION_MODES = {"default", "acceptEdits", "plan", "bypassPermissions", "dontAsk"}

    def __init__(
        self,
        input_format: str = "stream-json",
        output_format: str = "stream-json",
        default_options: dict[str, Any] | None = None,
    ):
        """Initialize the protocol handler.

        Args:
            input_format: Input format ("stream-json" or "auto")
            output_format: Output format ("stream-json" or "json")
            default_options: Default options applied if initialize request omits them.
        """
        if input_format not in {"stream-json", "auto"}:
            logger.warning("[stdio] Unsupported input_format %r; falling back to stream-json", input_format)
            input_format = "stream-json"
        if output_format not in {"stream-json", "json"}:
            logger.warning("[stdio] Unsupported output_format %r; falling back to stream-json", output_format)
            output_format = "stream-json"

        self._input_format = input_format
        self._output_format = output_format
        self._default_options = default_options or {}
        self._initialized = False
        self._session_id: str | None = None
        self._project_path: Path = Path.cwd()
        self._query_context: QueryContext | None = None
        self._can_use_tool: Any | None = None
        self._local_can_use_tool: Any | None = None
        self._sdk_can_use_tool_enabled: bool = False
        self._sdk_can_use_tool_supported: bool = True
        raw_sdk_url = self._default_options.get("sdk_url")
        self._sdk_url: str | None = (
            raw_sdk_url.strip() if isinstance(raw_sdk_url, str) and raw_sdk_url.strip() else None
        )
        self._pre_plan_mode: str | None = None
        self._clear_context_after_turn: bool = False
        self._hooks: dict[str, list[dict[str, Any]]] = {}
        self._sdk_hook_scope: HooksConfig | None = None
        self._pending_requests: dict[str, Any] = {}
        self._request_tasks: dict[str, asyncio.Task[None]] = {}
        self._request_subtypes: dict[str, str] = {}
        self._request_lock = asyncio.Lock()
        self._inflight_tasks: set[asyncio.Task[None]] = set()
        self._custom_system_prompt: str | None = None
        self._append_system_prompt: str | None = None
        self._skill_instructions: str | None = None
        self._output_style: str = "default"
        self._output_language: str = "auto"
        self._output_buffer: list[dict[str, Any]] = []
        self._allowed_tools: list[str] | None = None
        self._disallowed_tools: list[str] | None = None
        self._sdk_transport: _SDKWebSocketTransport | None = None
        self._tools_list: list[str] | None = None
        self._tools_preset: str | None = None
        self._session_additional_working_dirs: set[str] = set()
        self._fallback_model: str | None = None
        self._max_budget_usd: float | None = None
        self._json_schema: dict[str, Any] | None = None
        self._session_agent_name: str | None = None
        self._session_agents: dict[str, dict[str, str]] = {}
        self._session_agent_prompt: str | None = None
        self._active_agent_names: list[str] = []
        self._enabled_skill_names: list[str] = []
        self._plugin_payloads: list[dict[str, str]] = []
        self._disable_slash_commands: bool = False
        self._replay_user_messages: bool = False
        self._session_persistence_enabled: bool = True
        self._mcp_server_overrides: dict[str, McpServerInfo] | None = None
        self._mcp_disabled_servers: set[str] = set()
        self._permission_mode: str = "default"
        self._init_stream_message_sent: bool = False
        self._sdk_betas: list[str] = []

        # Conversation history for multi-turn queries
        self._conversation_messages: list[Any] = []
        self._seen_user_message_ids: set[str] = set()
        self._session_started = False
        self._session_start_time: float | None = None
        self._session_end_sent = False
        self._session_hook_messages: list[Any] = []
        self._session_history: SessionHistory | None = None
        self._query_in_progress: bool = False
        self._task_notification_task: asyncio.Task[None] | None = None
        self._resolved_tool_use_ids: set[str] = set()
        self._idle_exit_delay_ms: int | None = None
        self._idle_exit_task: asyncio.Task[None] | None = None
        self._runtime_task: asyncio.Task[Any] | None = None
        self._idle_exit_triggered: bool = False

        # Ensure each stdio handler starts with a clean MCP runtime override state.
        from ripperdoc.utils.mcp import clear_mcp_runtime_overrides, clear_sdk_mcp_request_sender

        clear_mcp_runtime_overrides(self._project_path)
        clear_sdk_mcp_request_sender()


__all__ = ["StdioProtocolHandler"]
