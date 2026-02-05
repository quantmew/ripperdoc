"""Stdio protocol handler implementation."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import sys
import time
import traceback
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from ripperdoc import __version__
from ripperdoc.core.agents import load_agent_definitions
from ripperdoc.core.config import get_project_config, get_effective_model_profile
from ripperdoc.core.default_tools import filter_tools_by_names, get_default_tools
from ripperdoc.core.hooks.llm_callback import build_hook_llm_callback
from ripperdoc.core.hooks.manager import hook_manager
from ripperdoc.core.hooks.config import HookDefinition, HookMatcher, HooksConfig, DEFAULT_HOOK_TIMEOUT
from ripperdoc.core.hooks.events import HookOutput
from pydantic import ValidationError
from ripperdoc.core.permissions import PermissionResult, make_permission_checker
from ripperdoc.core.query import QueryContext, query
from ripperdoc.core.hooks.state import bind_pending_message_queue, bind_hook_scopes
from ripperdoc.core.query_utils import format_pydantic_errors, resolve_model_profile
from ripperdoc.core.system_prompt import build_system_prompt
from ripperdoc.cli.commands import list_custom_commands, list_slash_commands
from ripperdoc.protocol.models import (
    AssistantMessageData,
    AssistantStreamMessage,
    ControlResponseError,
    ControlResponseMessage,
    ControlResponseSuccess,
    InitializeResponseData,
    MCPServerInfo,
    MCPServerStatusInfo,
    PermissionResponseAllow,
    PermissionResponseDeny,
    ResultMessage,
    SystemStreamMessage,
    UsageInfo,
    UserMessageData,
    UserStreamMessage,
    model_to_dict,
)
from ripperdoc.tools.background_shell import shutdown_background_shell
from ripperdoc.tools.mcp_tools import load_dynamic_mcp_tools_async, merge_tools_with_dynamic
from ripperdoc.utils.lsp import shutdown_lsp_manager
from ripperdoc.utils.mcp import format_mcp_instructions, load_mcp_servers_async, shutdown_mcp_runtime
from ripperdoc.utils.messages import (
    create_hook_notice_payload,
    create_user_message,
    is_hook_notice_payload,
)
from ripperdoc.utils.asyncio_compat import asyncio_timeout
from ripperdoc.utils.memory import build_memory_instructions
from ripperdoc.utils.session_history import SessionHistory

from .timeouts import (
    STDIO_HOOK_TIMEOUT_SEC,
    STDIO_QUERY_TIMEOUT_SEC,
    STDIO_READ_TIMEOUT_SEC,
    STDIO_WATCHDOG_INTERVAL_SEC,
)
from .watchdog import OperationWatchdog

logger = logging.getLogger(__name__)


class StdioProtocolHandler:
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

    _PERMISSION_MODES = {"default", "acceptEdits", "plan", "bypassPermissions"}

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
        self._hooks: dict[str, list[dict[str, Any]]] = {}
        self._sdk_hook_scope: HooksConfig | None = None
        self._pending_requests: dict[str, Any] = {}
        self._request_lock = asyncio.Lock()
        self._inflight_tasks: set[asyncio.Task[None]] = set()
        self._custom_system_prompt: str | None = None
        self._skill_instructions: str | None = None
        self._output_buffer: list[dict[str, Any]] = []
        self._allowed_tools: list[str] | None = None
        self._disallowed_tools: list[str] | None = None
        self._tools_list: list[str] | None = None

        # Conversation history for multi-turn queries
        self._conversation_messages: list[Any] = []
        self._session_started = False
        self._session_start_time: float | None = None
        self._session_end_sent = False
        self._session_hook_contexts: list[str] = []
        self._session_history: SessionHistory | None = None

    def _normalize_permission_mode(self, mode: Any) -> str:
        """Normalize permission mode to a supported value."""
        if isinstance(mode, str) and mode in self._PERMISSION_MODES:
            return mode
        return "default"

    def _normalize_tool_list(self, value: Any) -> list[str] | None:
        """Normalize tool list inputs from SDK/CLI options."""
        if value is None:
            return None
        if isinstance(value, str):
            raw = value.strip()
            if raw == "":
                return []
            return [item.strip() for item in raw.split(",") if item.strip()]
        if isinstance(value, (list, tuple, set)):
            names: list[str] = []
            for item in value:
                if item is None:
                    continue
                name = str(item).strip()
                if name:
                    names.append(name)
            return names
        return None

    def _apply_tool_filters(
        self,
        tools: list[Any],
        *,
        allowed_tools: list[str] | None,
        disallowed_tools: list[str] | None,
        tools_list: list[str] | None,
    ) -> list[Any]:
        """Apply SDK tool filters while keeping Task tool consistent."""
        if tools_list is None and allowed_tools is None and not disallowed_tools:
            return tools

        tool_names = [getattr(tool, "name", tool.__class__.__name__) for tool in tools]
        allow_set: set[str] | None = None

        if tools_list is not None:
            allow_set = set(tools_list)
        if allowed_tools is not None:
            allow_set = set(allowed_tools) if allow_set is None else allow_set & set(allowed_tools)

        if disallowed_tools:
            if allow_set is None:
                allow_set = set(tool_names)
            allow_set -= set(disallowed_tools)

        if allow_set is None:
            return tools

        return filter_tools_by_names(tools, list(allow_set))

    def _apply_permission_mode(self, mode: str) -> None:
        """Apply permission mode across query context, hooks, and permissions."""
        yolo_mode = mode == "bypassPermissions"
        if self._query_context:
            self._query_context.yolo_mode = yolo_mode
            self._query_context.permission_mode = mode

        hook_manager.set_permission_mode(mode)

        if yolo_mode:
            self._can_use_tool = None
        else:
            self._can_use_tool = make_permission_checker(self._project_path, yolo_mode=False)

    def _collect_hook_contexts(self, hook_result: Any) -> list[str]:
        contexts: list[str] = []
        additional_context = getattr(hook_result, "additional_context", None)
        if additional_context:
            contexts.append(str(additional_context))
        return contexts

    def _build_hook_notice_stream_message(
        self,
        text: str,
        hook_event: str,
        *,
        tool_name: str | None = None,
        level: str = "info",
    ) -> UserStreamMessage:
        return UserStreamMessage(
            session_id=self._session_id,
            message=UserMessageData(
                content=text,
                metadata={
                    "hook_notice": True,
                    "hook_event": hook_event,
                    "tool_name": tool_name,
                    "level": level,
                },
            ),
        )

    def _resolve_system_prompt(
        self,
        tools: list[Any],
        prompt: str,
        mcp_instructions: str | None,
        hook_instructions: list[str] | None = None,
    ) -> str:
        """Resolve the system prompt for the current session/query."""
        if self._custom_system_prompt:
            return self._custom_system_prompt

        additional_instructions: list[str] = []
        if self._skill_instructions:
            additional_instructions.append(self._skill_instructions)
        memory_instructions = build_memory_instructions()
        if memory_instructions:
            additional_instructions.append(memory_instructions)
        if hook_instructions:
            additional_instructions.extend([text for text in hook_instructions if text])

        return build_system_prompt(
            tools,
            prompt,
            {},
            additional_instructions=additional_instructions or None,
            mcp_instructions=mcp_instructions,
        )

    async def _write_message(self, message: dict[str, Any]) -> None:
        """Write a JSON message to stdout.

        Args:
            message: The message dictionary to write.
        """
        msg_type = message.get("type", "unknown")
        if self._output_format == "stream-json":
            json_data = json.dumps(message, ensure_ascii=False)
            logger.debug(f"[stdio] Writing message: type={msg_type}, json_length={len(json_data)}")
            sys.stdout.write(json_data + "\n")
            sys.stdout.flush()
            logger.debug(f"[stdio] Flushed message: type={msg_type}")
            return

        if self._output_format == "json":
            logger.debug(f"[stdio] Buffering message: type={msg_type}")
            self._output_buffer.append(message)
            return

        json_data = json.dumps(message, ensure_ascii=False)
        logger.warning(
            "[stdio] Unknown output_format %r; falling back to stream-json",
            self._output_format,
        )
        sys.stdout.write(json_data + "\n")
        sys.stdout.flush()

    async def flush_output(self) -> None:
        """Flush buffered output if using non-stream output formats."""
        if self._output_format != "json":
            return
        if not self._output_buffer:
            return
        json_data = json.dumps(self._output_buffer, ensure_ascii=False)
        sys.stdout.write(json_data + "\n")
        sys.stdout.flush()
        self._output_buffer.clear()

    async def _write_control_response(
        self,
        request_id: str,
        response: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """Write a control response message.

        Args:
            request_id: The request ID this responds to.
            response: The response data (for success).
            error: The error message (for failure).
        """
        if error:
            response_data: ControlResponseSuccess | ControlResponseError = ControlResponseError(  # type: ignore[assignment]
                request_id=request_id,
                error=error,
            )
        else:
            response_data = ControlResponseSuccess(
                request_id=request_id,
                response=response,
            )

        message = ControlResponseMessage(response=response_data)

        await self._write_message(model_to_dict(message))

    async def _handle_control_response(self, message: dict[str, Any]) -> None:
        """Handle control_response messages from the SDK."""
        response = message.get("response") or {}
        request_id = response.get("request_id")
        if not request_id:
            logger.warning("[stdio] control_response missing request_id")
            return
        future = self._pending_requests.pop(request_id, None)
        if future is None:
            logger.debug("[stdio] No pending request for control_response %s", request_id)
            return
        if response.get("subtype") == "error":
            error_msg = response.get("error", "Unknown error")
            future.set_exception(RuntimeError(error_msg))
            return
        future.set_result(response.get("response"))

    async def _send_control_request(
        self,
        request: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> Any:
        """Send a control_request to the SDK and await the response."""
        if self._output_format != "stream-json":
            raise RuntimeError("control_request requires stream-json output mode")
        request_id = f"cli_{uuid.uuid4().hex}"
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        self._pending_requests[request_id] = future
        await self._write_message(
            {
                "type": "control_request",
                "request_id": request_id,
                "request": request,
            }
        )
        try:
            return await asyncio.wait_for(
                future, timeout=timeout if timeout is not None else STDIO_HOOK_TIMEOUT_SEC
            )
        except asyncio.TimeoutError:
            self._pending_requests.pop(request_id, None)
            raise

    def _build_sdk_hook_scope(self, hooks: Any) -> HooksConfig:
        """Convert SDK hook config into a HooksConfig scope."""
        if not hooks or not isinstance(hooks, dict):
            return HooksConfig()
        parsed: dict[str, list[HookMatcher]] = {}
        for event_name, matchers in hooks.items():
            if not isinstance(matchers, list):
                continue
            parsed_matchers: list[HookMatcher] = []
            for matcher in matchers:
                if not isinstance(matcher, dict):
                    continue
                callback_ids = (
                    matcher.get("hookCallbackIds")
                    or matcher.get("hook_callback_ids")
                    or []
                )
                if not isinstance(callback_ids, list) or not callback_ids:
                    continue
                timeout = matcher.get("timeout", DEFAULT_HOOK_TIMEOUT)
                hook_defs: list[HookDefinition] = []
                for callback_id in callback_ids:
                    if not isinstance(callback_id, str) or not callback_id:
                        continue
                    hook_defs.append(
                        HookDefinition(
                            type="callback",
                            callback_id=callback_id,
                            timeout=timeout,
                        )
                    )
                if hook_defs:
                    parsed_matchers.append(
                        HookMatcher(matcher=matcher.get("matcher"), hooks=hook_defs)
                    )
            if parsed_matchers:
                parsed[event_name] = parsed_matchers
        return HooksConfig(hooks=parsed)

    def _configure_sdk_hooks(self, hooks: Any) -> None:
        """Register SDK-provided hooks for this session."""
        self._hooks = hooks or {}
        self._sdk_hook_scope = self._build_sdk_hook_scope(self._hooks)
        if self._query_context:
            self._query_context.add_hook_scope("sdk_hooks", self._sdk_hook_scope)
        if self._sdk_hook_scope and self._sdk_hook_scope.hooks:
            hook_manager.set_hook_callback(self._run_sdk_hook_callback)
        else:
            hook_manager.set_hook_callback(None)

    async def _run_sdk_hook_callback(
        self,
        callback_id: str,
        input_data: dict[str, Any],
        tool_use_id: str | None,
        timeout: float | None,
    ) -> HookOutput:
        """Invoke an SDK hook callback via control protocol."""
        safe_input = self._sanitize_for_json(input_data)
        try:
            response = await self._send_control_request(
                {
                    "subtype": "hook_callback",
                    "callback_id": callback_id,
                    "input": safe_input,
                    "tool_use_id": tool_use_id,
                },
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("[stdio] SDK hook callback timed out")
            return HookOutput.from_raw("", "", 1, timed_out=True)
        except Exception as exc:
            logger.error("[stdio] SDK hook callback failed: %s", exc)
            return HookOutput(error=str(exc), exit_code=1)

        if response is None:
            return HookOutput()
        if isinstance(response, dict):
            return HookOutput.from_raw(json.dumps(response), "", 0)
        return HookOutput()

    async def _write_message_stream(
        self,
        message_dict: dict[str, Any],
    ) -> None:
        """Write a regular message to the output stream.

        Args:
            message_dict: The message dictionary to write.
        """
        await self._write_message(message_dict)

    async def _read_line(self) -> str | None:
        """Read a single line from stdin with timeout.

        Returns:
            The line content, or None if EOF.
        """
        while True:
            try:
                # Wrap the blocking readline with timeout (idle timeout should not close the session)
                if STDIO_READ_TIMEOUT_SEC <= 0:
                    line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
                else:
                    line = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline),
                        timeout=STDIO_READ_TIMEOUT_SEC,
                    )
                if not line:
                    return None
                return line.rstrip("\n\r")  # type: ignore[no-any-return]
            except asyncio.TimeoutError:
                logger.debug(
                    f"[stdio] stdin read timed out after {STDIO_READ_TIMEOUT_SEC}s; continuing to wait"
                )
                continue
            except (OSError, IOError) as e:
                logger.error(f"Error reading from stdin: {e}")
                return None

    async def _read_messages(self) -> AsyncIterator[dict[str, Any]]:
        """Read and parse JSON messages from stdin with comprehensive error handling.

        Yields:
            Parsed JSON message dictionaries.
        """
        json_buffer = ""
        decoder = json.JSONDecoder()
        consecutive_empty_lines = 0
        max_empty_lines = 100  # Prevent infinite loop on empty input

        try:
            while True:
                line = await self._read_line()
                if line is None:
                    logger.debug("[stdio] EOF reached, stopping message reader")
                    break

                line = line.strip()
                if not line:
                    consecutive_empty_lines += 1
                    if consecutive_empty_lines > max_empty_lines:
                        logger.warning(
                            f"[stdio] Too many empty lines ({max_empty_lines}), stopping"
                        )
                        break
                    continue

                consecutive_empty_lines = 0  # Reset counter on non-empty line

                # Handle JSON that may span multiple lines
                json_lines = line.split("\n")
                for json_line in json_lines:
                    json_line = json_line.strip()
                    if not json_line:
                        continue

                    if self._input_format == "auto" and not json_buffer:
                        if json_line[:1] not in '{["':
                            for msg in self._coerce_auto_messages(json_line):
                                yield msg
                            continue

                    json_buffer += json_line

                    # Limit buffer size to prevent memory issues
                    if len(json_buffer) > 10_000_000:  # 10MB limit
                        logger.error("[stdio] JSON buffer too large, resetting")
                        json_buffer = ""
                        continue

                    # Attempt to parse as many JSON objects as possible
                    while json_buffer:
                        try:
                            data, index = decoder.raw_decode(json_buffer)
                        except json.JSONDecodeError as decode_error:
                            buffer_stripped = json_buffer.lstrip()
                            starts_like_json = buffer_stripped[:1] in '{["'
                            # If error is at/near the end and the buffer looks like JSON, keep buffering
                            if decode_error.pos >= len(json_buffer) - 1 and starts_like_json:
                                break
                            if self._input_format == "auto" and not starts_like_json:
                                for msg in self._coerce_auto_messages(buffer_stripped):
                                    yield msg
                                json_buffer = ""
                                break
                            # Otherwise treat as invalid JSON and reset buffer to recover
                            logger.warning(
                                "[stdio] Invalid JSON encountered, resetting buffer",
                                exc_info=False,
                            )
                            json_buffer = ""
                            break
                        else:
                            json_buffer = json_buffer[index:].lstrip()
                            if self._input_format == "auto":
                                for msg in self._coerce_auto_messages(data):
                                    yield msg
                            else:
                                if isinstance(data, list):
                                    logger.warning(
                                        "[stdio] Received JSON array in stream-json mode; skipping"
                                    )
                                    continue
                                if not isinstance(data, dict):
                                    logger.warning(
                                        "[stdio] Received non-object JSON in stream-json mode; skipping"
                                    )
                                    continue
                                logger.debug(
                                    f"[stdio] Successfully parsed message, type={data.get('type', 'unknown')}"
                                )
                                yield data

        except asyncio.CancelledError:
            logger.info("[stdio] Message reader cancelled")
            raise
        except Exception as e:
            logger.error(f"[stdio] Error in message reader: {type(e).__name__}: {e}", exc_info=True)
            raise

    def _generate_auto_request_id(self) -> str:
        """Generate a request id for auto input format."""
        return f"auto_{uuid.uuid4().hex}"

    def _coerce_auto_messages(self, data: Any) -> list[dict[str, Any]]:
        """Coerce auto input into control_request messages."""
        messages: list[dict[str, Any]] = []

        if isinstance(data, list):
            for item in data:
                messages.extend(self._coerce_auto_messages(item))
            return messages

        if isinstance(data, dict):
            if "type" in data:
                return [data]

            request_id = data.get("request_id") or self._generate_auto_request_id()
            if "request" in data:
                request_payload = data.get("request") or {}
                return [
                    {
                        "type": "control_request",
                        "request_id": request_id,
                        "request": request_payload,
                    }
                ]

            if any(key in data for key in ("subtype", "prompt", "options", "mode", "model")):
                request_payload = {k: v for k, v in data.items() if k != "request_id"}
                return [
                    {
                        "type": "control_request",
                        "request_id": request_id,
                        "request": request_payload,
                    }
                ]

            return messages

        if isinstance(data, str):
            prompt = data.strip()
            if not prompt:
                return messages
            return [
                {
                    "type": "control_request",
                    "request_id": self._generate_auto_request_id(),
                    "request": {"subtype": "query", "prompt": prompt},
                }
            ]

        return messages

    async def _handle_initialize(self, request: dict[str, Any], request_id: str) -> None:
        """Handle initialize request from SDK.

        Args:
            request: The initialize request data.
            request_id: The request ID.
        """
        if self._initialized:
            await self._write_control_response(request_id, error="Already initialized")
            return

        try:
            # Extract options from request
            request_options = request.get("options", {}) or {}
            options = {**self._default_options, **request_options}
            self._session_id = options.get("session_id") or str(uuid.uuid4())
            self._custom_system_prompt = options.get("system_prompt")
            raw_max_turns = options.get("max_turns")
            max_turns: int | None = None
            if raw_max_turns is not None:
                try:
                    max_turns = int(raw_max_turns)
                except (TypeError, ValueError):
                    logger.warning(
                        "[stdio] Invalid max_turns %r; ignoring",
                        raw_max_turns,
                    )

            # Setup working directory
            cwd = options.get("cwd")
            if cwd:
                self._project_path = Path(cwd)
            else:
                self._project_path = Path.cwd()

            # Initialize project config
            get_project_config(self._project_path)

            # Parse tool options
            self._allowed_tools = self._normalize_tool_list(options.get("allowed_tools"))
            self._disallowed_tools = self._normalize_tool_list(options.get("disallowed_tools"))
            self._tools_list = self._normalize_tool_list(options.get("tools"))

            # Get the tool list (apply SDK filters)
            tools = get_default_tools()
            tools = self._apply_tool_filters(
                tools,
                allowed_tools=self._allowed_tools,
                disallowed_tools=self._disallowed_tools,
                tools_list=self._tools_list,
            )

            # Parse permission mode
            permission_mode = self._normalize_permission_mode(
                options.get("permission_mode", "default")
            )
            yolo_mode = permission_mode == "bypassPermissions"

            # Setup model
            model = options.get("model") or "main"

            # 验证模型配置是否有效
            model_profile = get_effective_model_profile(model)
            if model_profile is None:
                error_msg = (
                    f"No valid model configuration found for '{model}'. "
                    f"Please set RIPPERDOC_BASE_URL environment variable or complete onboarding."
                )
                logger.error(f"[stdio] {error_msg}")
                await self._write_control_response(request_id, error=error_msg)
                return

            # Create query context
            self._query_context = QueryContext(
                tools=tools,
                yolo_mode=yolo_mode,
                verbose=options.get("verbose", False),
                model=model,
                max_turns=max_turns,
                permission_mode=permission_mode,
            )

            # Initialize hook manager
            hook_manager.set_project_dir(self._project_path)
            hook_manager.set_session_id(self._session_id)
            hook_manager.set_llm_callback(build_hook_llm_callback())
            self._session_history = SessionHistory(
                self._project_path, self._session_id or str(uuid.uuid4())
            )
            hook_manager.set_transcript_path(str(self._session_history.path))

            # Configure SDK-provided hooks (if any)
            hooks = request.get("hooks", None)
            if hooks is None:
                hooks = options.get("hooks", {})
            self._configure_sdk_hooks(hooks)

            session_start_notices: list[str] = []
            # Run SessionStart hooks during initialize (once per session)
            queue = self._query_context.pending_message_queue if self._query_context else None
            hook_scopes = self._query_context.hook_scopes if self._query_context else []
            with bind_pending_message_queue(queue), bind_hook_scopes(hook_scopes):
                try:
                    async with asyncio_timeout(STDIO_HOOK_TIMEOUT_SEC):
                        session_start_result = await hook_manager.run_session_start_async("startup")
                    if getattr(session_start_result, "system_message", None):
                        session_start_notices.append(str(session_start_result.system_message))
                    self._session_hook_contexts = self._collect_hook_contexts(
                        session_start_result
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        f"[stdio] Session start hook timed out after {STDIO_HOOK_TIMEOUT_SEC}s"
                    )
                except Exception as e:
                    logger.warning(f"[stdio] Session start hook failed: {e}")
                finally:
                    self._session_started = True
                    if self._session_start_time is None:
                        self._session_start_time = time.time()
                    self._session_end_sent = False

            # Load MCP servers and dynamic tools
            servers = await load_mcp_servers_async(self._project_path)
            dynamic_tools = await load_dynamic_mcp_tools_async(self._project_path)
            if dynamic_tools:
                tools = merge_tools_with_dynamic(tools, dynamic_tools)
                tools = self._apply_tool_filters(
                    tools,
                    allowed_tools=self._allowed_tools,
                    disallowed_tools=self._disallowed_tools,
                    tools_list=self._tools_list,
                )
                self._query_context.tools = tools

            mcp_instructions = format_mcp_instructions(servers)

            # Build system prompt components
            from ripperdoc.core.skills import load_all_skills, build_skill_summary

            skill_result = load_all_skills(self._project_path)
            self._skill_instructions = build_skill_summary(skill_result.skills)

            agent_result = load_agent_definitions()

            system_prompt = self._resolve_system_prompt(
                tools,
                "",  # Will be set per query
                mcp_instructions,
                self._session_hook_contexts,
            )

            # Apply permission mode to runtime state (checker + query context)
            self._apply_permission_mode(permission_mode)

            # Mark as initialized
            self._initialized = True

            # Send success response with available tools
            # Use simple list format for Claude SDK compatibility
            def _display_agent_name(agent_type: str) -> str:
                if agent_type in ("explore", "plan"):
                    return agent_type.title()
                return agent_type

            agent_names = [
                _display_agent_name(agent.agent_type) for agent in agent_result.active_agents
            ]
            skill_names = [skill.name for skill in skill_result.skills] if skill_result.skills else []

            slash_commands = [cmd.name for cmd in list_slash_commands()]
            for custom_cmd in list_custom_commands(self._project_path):
                if custom_cmd.name not in slash_commands:
                    slash_commands.append(custom_cmd.name)

            resolved_model_profile = resolve_model_profile(model or "main")
            resolved_model = resolved_model_profile.model if resolved_model_profile else (model or "main")

            init_response = InitializeResponseData(
                session_id=self._session_id or "",
                system_prompt=system_prompt,
                tools=[t.name for t in tools],
                mcp_servers=[MCPServerInfo(name=s.name) for s in servers] if servers else [],
                slash_commands=slash_commands,
                claude_code_version=__version__,
                agents=agent_names,
                skills=skill_names,
                plugins=[],
            )

            # Emit a system/init stream message first (Claude CLI compatibility)
            try:
                system_message = SystemStreamMessage(
                    uuid=str(uuid.uuid4()),
                    session_id=self._session_id or "",
                    api_key_source=init_response.apiKeySource,
                    cwd=str(self._project_path),
                    tools=[t.name for t in tools],
                    mcp_servers=[
                        MCPServerStatusInfo(name=s.name, status=getattr(s, "status", "unknown"))
                        for s in servers
                    ]
                    if servers
                    else [],
                    model=resolved_model,
                    permission_mode=permission_mode,
                    slash_commands=slash_commands,
                    claude_code_version=init_response.claude_code_version,
                    output_style=init_response.output_style,
                    agents=agent_names,
                    skills=skill_names,
                    plugins=[],
                )
                await self._write_message_stream(model_to_dict(system_message))
            except Exception as e:
                logger.warning(f"[stdio] Failed to emit system init message: {e}")

            for notice_text in session_start_notices:
                stream_message = self._build_hook_notice_stream_message(
                    notice_text,
                    "SessionStart",
                    tool_name=None,
                    level="info",
                )
                await self._write_message_stream(model_to_dict(stream_message))

            await self._write_control_response(
                request_id,
                response=model_to_dict(init_response),
            )

        except Exception as e:
            logger.error(f"Initialize failed: {e}", exc_info=True)
            await self._write_control_response(request_id, error=str(e))

    async def _handle_query(self, request: dict[str, Any], request_id: str) -> None:
        """Handle query request from SDK with comprehensive timeout and error handling.

        This method ensures ResultMessage is ALWAYS sent on any error/exception/timeout,
        including detailed stack traces for debugging.

        Args:
            request: The query request data.
            request_id: The request ID.
        """
        start_time = time.time()
        result_message_sent = [False]  # [bool] - Track if we've sent ResultMessage
        # Variables to track query state (using lists for mutable reference across async contexts)
        num_turns = [0]  # [int]
        is_error = [False]  # [bool]
        final_result_text = [None]  # [str | None]

        # Track token usage
        total_input_tokens = [0]  # [int]
        total_output_tokens = [0]  # [int]
        total_cache_read_tokens = [0]  # [int]
        total_cache_creation_tokens = [0]  # [int]

        async def send_final_result(result: ResultMessage) -> None:
            """Send ResultMessage and mark as sent."""
            if result_message_sent[0]:
                logger.warning("[stdio] ResultMessage already sent, skipping duplicate")
                return
            logger.debug("[stdio] Sending ResultMessage")
            try:
                await self._write_message_stream(model_to_dict(result))
                result_message_sent[0] = True
                logger.debug("[stdio] ResultMessage sent successfully")
            except Exception as e:
                logger.error(f"[stdio] Failed to send ResultMessage: {e}", exc_info=True)
                result_message_sent[0] = True  # Mark as sent to avoid retries

        async def send_error_result(error_msg: str, exc: Exception | None = None) -> None:
            """Send error ResultMessage with full stack trace."""
            if exc:
                tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                error_detail = f"{type(exc).__name__}: {error_msg}\n\nStack trace:\n{tb_str}"
            else:
                error_detail = error_msg

            result = ResultMessage(
                duration_ms=int((time.time() - start_time) * 1000),
                duration_api_ms=0,
                is_error=True,
                subtype="error_during_execution",
                num_turns=num_turns[0],
                session_id=self._session_id or "",
                total_cost_usd=None,
                usage=None,
                result=error_detail[:50000] if len(error_detail) > 50000 else error_detail,  # Limit size
                structured_output=None,
            )
            await send_final_result(result)

        async def fail_request(error_msg: str, exc: Exception | None = None) -> None:
            """Send control error response and always follow with ResultMessage."""
            try:
                await self._write_control_response(request_id, error=error_msg)
            except Exception as write_error:
                logger.error(
                    f"[stdio] Failed to send control response: {write_error}",
                    exc_info=True,
                )
            await send_error_result(error_msg, exc)

        if not self._initialized:
            await fail_request("Not initialized")
            return

        try:
            prompt = request.get("prompt", "")
            if not prompt:
                await fail_request("Prompt is required")
                return

            logger.info(
                "[stdio] Starting query handling",
                extra={
                    "request_id": request_id,
                    "prompt_length": len(prompt),
                    "session_id": self._session_id,
                    "conversation_messages": len(self._conversation_messages),
                    "query_timeout": STDIO_QUERY_TIMEOUT_SEC,
                },
            )

            # Create session history (one per session)
            if self._session_history is None:
                self._session_history = SessionHistory(
                    self._project_path, self._session_id or str(uuid.uuid4())
                )
            session_history = self._session_history
            hook_manager.set_transcript_path(str(session_history.path))

            # Create initial user message
            user_message = create_user_message(prompt)
            self._conversation_messages.append(user_message)
            session_history.append(user_message)

            # Use the conversation history for messages
            messages = list(self._conversation_messages)

            # Build system prompt
            additional_instructions: list[str] = []
            hook_notices: list[dict[str, Any]] = []

            queue = self._query_context.pending_message_queue if self._query_context else None
            hook_scopes = self._query_context.hook_scopes if self._query_context else []
            with bind_pending_message_queue(queue), bind_hook_scopes(hook_scopes):
                if self._session_hook_contexts:
                    additional_instructions.extend(self._session_hook_contexts)

                # Run prompt submit hooks with timeout
                try:
                    async with asyncio_timeout(STDIO_HOOK_TIMEOUT_SEC):
                        prompt_hook_result = await hook_manager.run_user_prompt_submit_async(prompt)
                        if hasattr(prompt_hook_result, "should_block") and prompt_hook_result.should_block:
                            reason = (
                                prompt_hook_result.block_reason
                                if hasattr(prompt_hook_result, "block_reason")
                                else "Prompt blocked by hook."
                            )
                            await fail_request(str(reason))
                            return
                        if (
                            hasattr(prompt_hook_result, "system_message")
                            and prompt_hook_result.system_message
                        ):
                            hook_notices.append(
                                create_hook_notice_payload(
                                    text=str(prompt_hook_result.system_message),
                                    hook_event="UserPromptSubmit",
                                )
                            )
                        if (
                            hasattr(prompt_hook_result, "additional_context")
                            and prompt_hook_result.additional_context
                        ):
                            additional_instructions.append(str(prompt_hook_result.additional_context))
                except asyncio.TimeoutError:
                    logger.warning(f"[stdio] Prompt submit hook timed out after {STDIO_HOOK_TIMEOUT_SEC}s")
                except Exception as e:
                    logger.warning(f"[stdio] Prompt submit hook failed: {e}")

            # Build final system prompt
            servers = await load_mcp_servers_async(self._project_path)
            mcp_instructions = format_mcp_instructions(servers)

            system_prompt = self._resolve_system_prompt(
                self._query_context.tools if self._query_context else [],
                prompt,
                mcp_instructions,
                additional_instructions,
            )

            # Send acknowledgment that query is starting
            await self._write_control_response(
                request_id, response={"status": "querying", "session_id": self._session_id}
            )
            for notice in hook_notices:
                stream_message = self._build_hook_notice_stream_message(
                    str(notice.get("text", "")),
                    str(notice.get("hook_event", "")),
                    tool_name=notice.get("tool_name"),
                    level=notice.get("level") or "info",
                )
                await self._write_message_stream(model_to_dict(stream_message))

            # Execute query with comprehensive timeout and error handling
            try:
                context: dict[str, Any] = {}

                logger.debug(
                    "[stdio] Preparing query execution",
                    extra={
                        "messages_count": len(messages),
                        "system_prompt_length": len(system_prompt),
                        "query_timeout": STDIO_QUERY_TIMEOUT_SEC,
                    },
                )

                # Create watchdog for monitoring query progress
                async with OperationWatchdog(
                    timeout_sec=STDIO_QUERY_TIMEOUT_SEC, check_interval=STDIO_WATCHDOG_INTERVAL_SEC
                ) as watchdog:
                    # Execute query with overall timeout
                    async with asyncio_timeout(STDIO_QUERY_TIMEOUT_SEC):
                        async for message in query(
                            messages,
                            system_prompt,
                            context,
                            self._query_context or {},  # type: ignore[arg-type]
                            self._can_use_tool,
                        ):
                            watchdog.ping()
                            msg_type = getattr(message, "type", None)
                            logger.debug(
                                f"[stdio] Received message of type: {msg_type}, "
                                f"num_turns={num_turns[0]}, "
                                f"elapsed_ms={int((time.time() - start_time) * 1000)}"
                            )
                            num_turns[0] += 1

                            # Handle progress messages
                            if msg_type == "progress":
                                notice_content = getattr(message, "content", None)
                                if is_hook_notice_payload(notice_content):
                                    stream_message = self._build_hook_notice_stream_message(
                                        str(notice_content.get("text", "")),
                                        str(notice_content.get("hook_event", "")),
                                        tool_name=notice_content.get("tool_name"),
                                        level=notice_content.get("level") or "info",
                                    )
                                    await self._write_message_stream(model_to_dict(stream_message))
                                    continue
                                # Check if this is a subagent message that should be forwarded to SDK
                                is_subagent_msg = getattr(message, "is_subagent_message", False)
                                if is_subagent_msg:
                                    # Extract the subagent message from content
                                    subagent_message = getattr(message, "content", None)
                                    if subagent_message and hasattr(subagent_message, "type"):
                                        logger.debug(
                                            f"[stdio] Forwarding subagent message: type={getattr(subagent_message, 'type', 'unknown')}"
                                        )
                                        # Convert and forward the subagent message to SDK
                                        message_dict = self._convert_message_to_sdk(subagent_message)
                                        if message_dict:
                                            await self._write_message_stream(message_dict)

                                            # Add subagent messages to conversation history
                                            subagent_msg_type = getattr(subagent_message, "type", "")
                                            if subagent_msg_type == "assistant":
                                                self._conversation_messages.append(subagent_message)

                                                # Track token usage from subagent assistant messages
                                                total_input_tokens[0] += getattr(subagent_message, "input_tokens", 0)
                                                total_output_tokens[0] += getattr(subagent_message, "output_tokens", 0)
                                                total_cache_read_tokens[0] += getattr(subagent_message, "cache_read_tokens", 0)
                                                total_cache_creation_tokens[0] += getattr(
                                                    subagent_message, "cache_creation_tokens", 0
                                                )
                                # Continue to filter out normal progress messages
                                continue

                            # Track token usage from assistant messages
                            if msg_type == "assistant":
                                total_input_tokens[0] += getattr(message, "input_tokens", 0)
                                total_output_tokens[0] += getattr(message, "output_tokens", 0)
                                total_cache_read_tokens[0] += getattr(message, "cache_read_tokens", 0)
                                total_cache_creation_tokens[0] += getattr(
                                    message, "cache_creation_tokens", 0
                                )

                                msg_content = getattr(message, "message", None)
                                if msg_content:
                                    content = getattr(msg_content, "content", None)
                                    if content:
                                        # Extract text blocks for result field
                                        if isinstance(content, str):
                                            final_result_text[0] = content
                                        elif isinstance(content, list):
                                            text_parts = []
                                            for block in content:
                                                if isinstance(block, dict):
                                                    if block.get("type") == "text":
                                                        text_parts.append(block.get("text", ""))
                                                    elif block.get("type") == "tool_use":
                                                        text_parts.clear()
                                                elif hasattr(block, "type"):
                                                    if block.type == "text":
                                                        text_parts.append(getattr(block, "text", ""))
                                            if text_parts:
                                                final_result_text[0] = "\n".join(text_parts)

                            # Convert message to SDK format
                            message_dict = self._convert_message_to_sdk(message)
                            if message_dict is None:
                                continue
                            await self._write_message_stream(message_dict)

                            # Add to conversation history
                            if msg_type in ("assistant", "user"):
                                self._conversation_messages.append(message)

                            # Add to local history and session history
                            messages.append(message)  # type: ignore[arg-type]
                            session_history.append(message)  # type: ignore[arg-type]

                logger.debug("[stdio] Query loop ended successfully")

            except asyncio.TimeoutError:
                logger.error(f"[stdio] Query execution timed out after {STDIO_QUERY_TIMEOUT_SEC}s")
                await send_error_result(f"Query timed out after {STDIO_QUERY_TIMEOUT_SEC}s")
            except asyncio.CancelledError:
                logger.warning("[stdio] Query was cancelled")
                await send_error_result("Query was cancelled")
            except Exception as query_error:
                is_error[0] = True
                logger.error(f"[stdio] Query execution error: {type(query_error).__name__}: {query_error}", exc_info=True)
                await send_error_result(str(query_error), query_error)

            # Build and send normal completion result if no error occurred
            if not is_error[0] and not result_message_sent[0]:
                logger.debug("[stdio] Building usage info")

                # Calculate cost
                cost_per_million_input = 3.0
                cost_per_million_output = 15.0
                total_cost_usd = (total_input_tokens[0] * cost_per_million_input / 1_000_000) + (
                    total_output_tokens[0] * cost_per_million_output / 1_000_000
                )

                # Build usage info
                usage_info = None
                if total_input_tokens[0] or total_output_tokens[0]:
                    usage_info = UsageInfo(
                        input_tokens=total_input_tokens[0],
                        cache_creation_input_tokens=total_cache_creation_tokens[0],
                        cache_read_input_tokens=total_cache_read_tokens[0],
                        output_tokens=total_output_tokens[0],
                    )

                duration_ms = int((time.time() - start_time) * 1000)
                duration_api_ms = duration_ms

                result_message = ResultMessage(
                    duration_ms=duration_ms,
                    duration_api_ms=duration_api_ms,
                    is_error=is_error[0],
                    subtype="success",
                    num_turns=num_turns[0],
                    session_id=self._session_id or "",
                    total_cost_usd=round(total_cost_usd, 8) if total_cost_usd > 0 else None,
                    usage=usage_info,
                    result=final_result_text[0],
                    structured_output=None,
                )
                await send_final_result(result_message)

            logger.info(
                "[stdio] Query completed",
                extra={
                    "request_id": request_id,
                    "duration_ms": int((time.time() - start_time) * 1000),
                    "num_turns": num_turns[0],
                    "is_error": is_error[0],
                    "input_tokens": total_input_tokens[0],
                    "output_tokens": total_output_tokens[0],
                    "result_sent": result_message_sent[0],
                },
            )

        except Exception as e:
            logger.error(f"[stdio] Handle query failed: {type(e).__name__}: {e}", exc_info=True)
            await self._write_control_response(request_id, error=str(e))

            # Ensure ResultMessage is sent even if everything else fails
            if not result_message_sent[0]:
                try:
                    await send_error_result(str(e), e)
                except Exception as send_error:
                    logger.error(
                        f"[stdio] Critical: Failed to send error ResultMessage: {send_error}",
                        exc_info=True,
                    )

    def _convert_message_to_sdk(self, message: Any) -> dict[str, Any] | None:
        """Convert internal message to SDK format.

        Args:
            message: The internal message object.

        Returns:
            A dictionary in SDK message format, or None if message should be skipped.
        """
        msg_type = getattr(message, "type", None)

        # Filter out progress messages (internal implementation detail)
        if msg_type == "progress":
            return None

        if msg_type == "assistant":
            content_blocks = []
            msg_content = getattr(message, "message", None)
            if msg_content:
                content = getattr(msg_content, "content", None)
                if content:
                    if isinstance(content, str):
                        content_blocks.append({"type": "text", "text": content})
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict):
                                if block.get("type") == "tool_use":
                                    content_blocks.append(self._normalize_tool_use_block(block))
                                else:
                                    content_blocks.append(block)
                            else:
                                # Convert MessageContent (Pydantic model) to dict
                                block_dict = self._convert_content_block(block)
                                if block_dict:
                                    content_blocks.append(block_dict)

            # Resolve model pointer to actual model name for SDK
            # The message may have model=None (unset), so fall back to QueryContext.model
            # Then resolve any pointer (e.g., "main") to the actual model name
            model_pointer = getattr(message, "model", None) or (
                self._query_context.model if self._query_context else None
            )  # type: ignore[union-attr]
            model_profile = resolve_model_profile(
                str(model_pointer) if model_pointer else "claude-opus-4-5-20251101"
            )
            actual_model = (
                model_profile.model
                if model_profile
                else (model_pointer or "claude-opus-4-5-20251101")
            )

            stream_message = AssistantStreamMessage(
                message=AssistantMessageData(
                    content=content_blocks,
                    model=actual_model,
                ),
                session_id=self._session_id,
                parent_tool_use_id=getattr(message, "parent_tool_use_id", None),
                uuid=getattr(message, "uuid", None),
            )
            return model_to_dict(stream_message)

        if msg_type == "user":
            msg_content = getattr(message, "message", None)
            content = getattr(msg_content, "content", "") if msg_content else ""
            tool_result_text: str | None = None
            tool_result_is_error = False

            # If content is a list of MessageContent objects (e.g., tool results),
            # convert it to SDK content blocks
            if isinstance(content, list):
                content_blocks = []
                for block in content:
                    if isinstance(block, dict):
                        block_type = block.get("type", "")
                        if block_type == "tool_result":
                            tool_use_id = block.get("tool_use_id") or block.get("id") or ""
                            text_value = self._normalize_tool_result_text(
                                block.get("text"), block.get("content")
                            )
                            normalized_block: dict[str, Any] = {
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": text_value,
                            }
                            if "is_error" in block:
                                normalized_block["is_error"] = block.get("is_error")
                                tool_result_is_error = bool(block.get("is_error"))
                            content_blocks.append(normalized_block)
                            if tool_result_text is None:
                                tool_result_text = str(text_value) if text_value is not None else ""
                        else:
                            if block_type == "tool_use":
                                content_blocks.append(self._normalize_tool_use_block(block))
                            else:
                                content_blocks.append(block)
                    else:
                        block_dict = self._convert_content_block(block)
                        if block_dict:
                            if block_dict.get("type") == "tool_result":
                                if tool_result_text is None:
                                    tool_result_text = str(block_dict.get("content") or "")
                                tool_result_is_error = bool(block_dict.get("is_error", False))
                            content_blocks.append(block_dict)
                content = content_blocks

            stream_message: UserStreamMessage | AssistantStreamMessage = UserStreamMessage(  # type: ignore[assignment,no-redef]
                message=UserMessageData(content=content),
                uuid=getattr(message, "uuid", None),
                session_id=self._session_id,
                parent_tool_use_id=getattr(message, "parent_tool_use_id", None),
                tool_use_result=(
                    (
                        self._format_tool_use_result(tool_result_text, tool_result_is_error)
                        if isinstance(content, list)
                        and tool_result_is_error
                        and tool_result_text is not None
                        else self._sanitize_for_json(getattr(message, "tool_use_result", None))
                    )
                ),
            )
            return model_to_dict(stream_message)

        # Unknown message type - return None to skip
        return None

    def _sanitize_for_json(self, obj: Any) -> Any:
        """Recursively convert objects to JSON-serializable types.

        This function ensures Pydantic models and other objects are converted
        to dictionaries/lists/primitives that can be JSON serialized.

        Args:
            obj: The object to sanitize.

        Returns:
            A JSON-serializable version of the object.
        """
        # None values
        if obj is None:
            return None

        # Primitives
        if isinstance(obj, (str, int, float, bool)):
            return obj

        # Lists and tuples
        if isinstance(obj, (list, tuple)):
            return [self._sanitize_for_json(item) for item in obj]

        # Dictionaries
        if isinstance(obj, dict):
            return {key: self._sanitize_for_json(value) for key, value in obj.items()}

        # Pydantic models
        if hasattr(obj, "model_dump"):
            try:
                dumped = obj.model_dump(exclude_none=True)
                return self._sanitize_for_json(dumped)
            except Exception:
                pass

        # Objects with dict() method
        if hasattr(obj, "dict"):
            try:
                dumped = obj.dict(exclude_none=True)
                return self._sanitize_for_json(dumped)
            except Exception:
                pass

        # Fallback: try to convert to string
        try:
            return str(obj)
        except Exception:
            return None

    def _normalize_tool_result_text(self, text_value: Any, content_value: Any) -> str:
        if isinstance(text_value, str):
            return text_value
        if isinstance(content_value, str):
            return content_value
        if isinstance(content_value, list):
            for item in content_value:
                if isinstance(item, dict) and item.get("type") == "text":
                    return str(item.get("text") or "")
            if content_value:
                return str(content_value[0])
        if content_value is None:
            return ""
        return str(content_value)

    def _format_tool_use_result(self, text_value: str | None, is_error: bool) -> str | None:
        if text_value is None:
            return None
        if is_error and not text_value.startswith("Error: "):
            return f"Error: {text_value}"
        return text_value

    def _summarize_task_prompt(self, prompt: str) -> str:
        line = prompt.strip().splitlines()[0] if prompt else ""
        if len(line) > 120:
            return f"{line[:117]}..."
        return line

    def _normalize_task_tool_input(self, input_data: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(input_data)
        subagent_type = normalized.get("subagent_type")
        if isinstance(subagent_type, str) and subagent_type in ("explore", "plan"):
            normalized["subagent_type"] = subagent_type.title()
        if "description" not in normalized:
            prompt = normalized.get("prompt")
            if isinstance(prompt, str) and prompt.strip():
                normalized["description"] = self._summarize_task_prompt(prompt)
        return normalized

    def _normalize_tool_use_block(self, block: dict[str, Any]) -> dict[str, Any]:
        tool_id = block.get("id") or block.get("tool_use_id") or ""
        name = block.get("name") or ""
        input_value = block.get("input") or {}
        if hasattr(input_value, "model_dump"):
            input_value = input_value.model_dump()
        elif hasattr(input_value, "dict"):
            input_value = input_value.dict()
        if not isinstance(input_value, dict):
            input_value = {"value": str(input_value)}
        if name == "Task":
            input_value = self._normalize_task_tool_input(input_value)
        normalized = dict(block)
        normalized.update(
            {
                "type": "tool_use",
                "id": tool_id,
                "name": name,
                "input": input_value,
            }
        )
        return normalized

    def _convert_content_block(self, block: Any) -> dict[str, Any] | None:
        """Convert a MessageContent block to dictionary.

        Uses the same logic as _content_block_to_api in messages.py
        to ensure consistency and proper field mapping.

        Args:
            block: The MessageContent object.

        Returns:
            A dictionary representation of the block.
        """
        block_type = getattr(block, "type", None)

        if block_type == "text":
            return {
                "type": "text",
                "text": getattr(block, "text", None) or "",
            }

        if block_type == "thinking":
            return {
                "type": "thinking",
                "thinking": getattr(block, "thinking", None) or getattr(block, "text", None) or "",
                "signature": getattr(block, "signature", None),
            }

        if block_type == "tool_use":
            # Use the same id extraction logic as _content_block_to_api
            # Try id first, then tool_use_id, then empty string
            tool_id = getattr(block, "id", None) or getattr(block, "tool_use_id", None) or ""
            name = getattr(block, "name", None) or ""
            input_value = getattr(block, "input", None) or {}
            if hasattr(input_value, "model_dump"):
                input_value = input_value.model_dump()
            elif hasattr(input_value, "dict"):
                input_value = input_value.dict()
            if not isinstance(input_value, dict):
                input_value = {"value": str(input_value)}
            if name == "Task" and isinstance(input_value, dict):
                input_value = self._normalize_task_tool_input(input_value)
            return {
                "type": "tool_use",
                "id": tool_id,
                "name": name,
                "input": input_value,
            }

        if block_type == "tool_result":
            text_value = (
                getattr(block, "text", None)
                or self._normalize_tool_result_text(None, getattr(block, "content", None))
                or ""
            )
            result_block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": getattr(block, "tool_use_id", None)
                or getattr(block, "id", None)
                or "",
                "content": text_value,
            }
            if getattr(block, "is_error", None) is not None:
                result_block["is_error"] = getattr(block, "is_error", None)
            return result_block

        if block_type == "image":
            return {
                "type": "image",
                "source": {
                    "type": getattr(block, "source_type", None) or "base64",
                    "media_type": getattr(block, "media_type", None) or "image/jpeg",
                    "data": getattr(block, "image_data", None) or "",
                },
            }

        # Unknown block type - try to convert with generic approach
        block_dict = {}
        if hasattr(block, "type"):
            block_dict["type"] = block.type
        if hasattr(block, "text"):
            block_dict["text"] = block.text
        if hasattr(block, "id"):
            block_dict["id"] = block.id
        if hasattr(block, "name"):
            block_dict["name"] = block.name
        if hasattr(block, "input"):
            block_dict["input"] = block.input
        if hasattr(block, "content"):
            block_dict["content"] = block.content
        if hasattr(block, "is_error"):
            block_dict["is_error"] = block.is_error
        return block_dict if block_dict else None

    async def _handle_control_request(self, message: dict[str, Any]) -> None:
        """Handle a control request from the SDK.

        Args:
            message: The control request message.
        """
        request = message.get("request", {})
        request_id = message.get("request_id", "")
        request_subtype = request.get("subtype", "")

        async with self._request_lock:
            try:
                if request_subtype == "initialize":
                    await self._handle_initialize(request, request_id)

                elif request_subtype == "query":
                    if not self._initialized and self._input_format == "auto":
                        await self._handle_initialize({"options": {}}, f"{request_id}_init")
                    await self._handle_query(request, request_id)

                elif request_subtype == "set_permission_mode":
                    await self._handle_set_permission_mode(request, request_id)

                elif request_subtype == "set_model":
                    await self._handle_set_model(request, request_id)

                elif request_subtype == "rewind_files":
                    await self._handle_rewind_files(request, request_id)

                elif request_subtype == "hook_callback":
                    await self._handle_hook_callback(request, request_id)

                elif request_subtype == "can_use_tool":
                    await self._handle_can_use_tool(request, request_id)

                else:
                    await self._write_control_response(
                        request_id, error=f"Unknown request subtype: {request_subtype}"
                    )

            except Exception as e:
                logger.error(f"Error handling control request: {e}", exc_info=True)
                await self._write_control_response(request_id, error=str(e))

    async def _handle_set_permission_mode(self, request: dict[str, Any], request_id: str) -> None:
        """Handle set_permission_mode request from SDK.

        Args:
            request: The set_permission_mode request data.
            request_id: The request ID.
        """
        mode = self._normalize_permission_mode(request.get("mode", "default"))
        self._apply_permission_mode(mode)

        await self._write_control_response(
            request_id, response={"status": "permission_mode_set", "mode": mode}
        )

    async def _handle_set_model(self, request: dict[str, Any], request_id: str) -> None:
        """Handle set_model request from SDK.

        Args:
            request: The set_model request data.
            request_id: The request ID.
        """
        model = request.get("model")
        # Update the model in the query context
        if self._query_context:
            self._query_context.model = model or "main"

        await self._write_control_response(
            request_id, response={"status": "model_set", "model": model}
        )

    async def _handle_rewind_files(self, _request: dict[str, Any], request_id: str) -> None:
        """Handle rewind_files request from SDK.

        Note: File checkpointing is not currently supported.
        This method exists for Claude SDK API compatibility.

        Args:
            _request: The rewind_files request data.
            request_id: The request ID.
        """
        await self._write_control_response(
            request_id, error="File checkpointing and rewind_files are not currently supported"
        )

    async def _handle_hook_callback(self, request: dict[str, Any], request_id: str) -> None:
        """Handle hook_callback request from SDK.

        Args:
            request: The hook_callback request data.
            request_id: The request ID.
        """
        logger.warning("[stdio] hook_callback requests are not supported by the CLI")
        await self._write_control_response(
            request_id,
            error="hook_callback requests must be initiated by the CLI (SDK hooks).",
        )

    async def _handle_can_use_tool(self, request: dict[str, Any], request_id: str) -> None:
        """Handle can_use_tool request from SDK.

        Args:
            request: The can_use_tool request data.
            request_id: The request ID.
        """
        tool_name = request.get("tool_name") or request.get("toolName") or ""
        tool_input = request.get("input")
        if tool_input is None:
            tool_input = request.get("tool_input", {})
        if tool_input is None:
            tool_input = {}

        if not tool_name:
            await self._write_control_response(request_id, error="Missing tool_name")
            return

        if not self._query_context:
            await self._write_control_response(request_id, error="Session not initialized")
            return

        tool = self._query_context.tool_registry.get(tool_name)
        if tool is None:
            perm_response = PermissionResponseDeny(message=f"Tool '{tool_name}' not found")
            await self._write_control_response(request_id, response=model_to_dict(perm_response))
            return

        if tool_input and hasattr(tool_input, "model_dump"):
            tool_input = tool_input.model_dump()
        elif tool_input and hasattr(tool_input, "dict") and callable(getattr(tool_input, "dict")):
            tool_input = tool_input.dict()
        if not isinstance(tool_input, dict):
            tool_input = {"value": str(tool_input)}
        if tool_name == "Task" and isinstance(tool_input, dict):
            tool_input = self._normalize_task_tool_input(tool_input)

        try:
            parsed_input = tool.input_schema(**tool_input)
        except ValidationError as ve:
            detail = format_pydantic_errors(ve)
            perm_response = PermissionResponseDeny(
                message=f"Invalid input for tool '{tool_name}': {detail}"
            )
            await self._write_control_response(request_id, response=model_to_dict(perm_response))
            return
        except (TypeError, ValueError) as exc:
            perm_response = PermissionResponseDeny(
                message=f"Invalid input for tool '{tool_name}': {exc}"
            )
            await self._write_control_response(request_id, response=model_to_dict(perm_response))
            return

        # Use the permission checker if available
        if self._can_use_tool:
            try:
                result = self._can_use_tool(tool, parsed_input)
                if inspect.isawaitable(result):
                    result = await result

                allowed = False
                message = None
                updated_input = None

                if isinstance(result, PermissionResult):
                    allowed = bool(result.result)
                    message = result.message
                    updated_input = result.updated_input
                elif isinstance(result, dict) and "result" in result:
                    allowed = bool(result.get("result"))
                    message = result.get("message")
                    updated_input = result.get("updated_input") or result.get("updatedInput")
                elif isinstance(result, tuple) and len(result) == 2:
                    allowed = bool(result[0])
                    message = result[1]
                else:
                    allowed = bool(result)

                if allowed:
                    normalized_input = tool_input if updated_input is None else updated_input
                    if normalized_input and hasattr(normalized_input, "model_dump"):
                        normalized_input = normalized_input.model_dump()
                    elif normalized_input and hasattr(normalized_input, "dict") and callable(
                        getattr(normalized_input, "dict")
                    ):
                        normalized_input = normalized_input.dict()
                    if not isinstance(normalized_input, dict):
                        normalized_input = {"value": str(normalized_input)}
                    if tool_name == "Task" and isinstance(normalized_input, dict):
                        normalized_input = self._normalize_task_tool_input(normalized_input)
                    perm_response = PermissionResponseAllow(
                        updatedInput=normalized_input,
                    )
                else:
                    perm_response = PermissionResponseDeny(message=message or "")

                await self._write_control_response(
                    request_id,
                    response=model_to_dict(perm_response),
                )
            except Exception as e:
                logger.error(f"Error in permission check: {e}")
                await self._write_control_response(request_id, error=str(e))
        else:
            # No permission checker, allow by default
            perm_response = PermissionResponseAllow(
                updatedInput=tool_input,
            )
            await self._write_control_response(
                request_id,
                response=model_to_dict(perm_response),
            )

    async def _run_session_end(self, reason: str) -> None:
        if self._session_end_sent or not self._session_started:
            return
        duration = 0.0
        if self._session_start_time is not None:
            duration = max(time.time() - self._session_start_time, 0.0)
        message_count = len(self._conversation_messages)
        hook_scopes = self._query_context.hook_scopes if self._query_context else []
        logger.debug("[stdio] Running session end hooks")
        try:
            with bind_hook_scopes(hook_scopes):
                async with asyncio_timeout(STDIO_HOOK_TIMEOUT_SEC):
                    await hook_manager.run_session_end_async(
                        reason,
                        duration_seconds=duration,
                        message_count=message_count,
                    )
            logger.debug("[stdio] Session end hooks completed")
        except asyncio.TimeoutError:
            logger.warning(
                f"[stdio] Session end hook timed out after {STDIO_HOOK_TIMEOUT_SEC}s"
            )
        except Exception as e:
            logger.warning(f"[stdio] Session end hook failed: {e}")
        finally:
            self._session_end_sent = True

    async def run(self) -> None:
        """Main run loop for the stdio protocol handler with graceful shutdown.

        Reads messages from stdin and handles them until EOF.
        """
        logger.info("Stdio protocol handler starting")

        try:
            async for message in self._read_messages():
                msg_type = message.get("type")

                if msg_type == "control_response":
                    await self._handle_control_response(message)
                    continue

                if msg_type == "control_request":
                    task = asyncio.create_task(self._handle_control_request(message))
                    self._inflight_tasks.add(task)

                    def _cleanup_task(t: asyncio.Task[None]) -> None:
                        self._inflight_tasks.discard(t)
                        if t.cancelled():
                            return
                        exc = t.exception()
                        if exc:
                            logger.error(
                                "[stdio] control_request task failed: %s: %s",
                                type(exc).__name__,
                                exc,
                            )

                    task.add_done_callback(_cleanup_task)
                    continue

                # Unknown message type
                logger.warning(f"Unknown message type: {msg_type}")

        except (OSError, IOError, json.JSONDecodeError) as e:
            logger.error(f"Error in stdio loop: {e}", exc_info=True)
        except asyncio.CancelledError:
            logger.info("Stdio protocol handler cancelled")
        except Exception as e:
            logger.error(f"Unexpected error in stdio loop: {type(e).__name__}: {e}", exc_info=True)
        finally:
            # Comprehensive cleanup with timeout
            logger.info("Stdio protocol handler shutting down...")

            try:
                await self.flush_output()
            except Exception as e:
                logger.error(f"[cleanup] Error flushing output: {e}")

            if self._inflight_tasks:
                for task in list(self._inflight_tasks):
                    task.cancel()
                try:
                    await asyncio.gather(*self._inflight_tasks, return_exceptions=True)
                except Exception:
                    pass

            try:
                await self._run_session_end("other")
            except Exception as e:
                logger.warning(f"[cleanup] Session end hook failed: {e}")

            cleanup_tasks = []

            # Add MCP runtime shutdown
            async def cleanup_mcp():
                try:
                    async with asyncio_timeout(10):
                        await shutdown_mcp_runtime()
                except asyncio.TimeoutError:
                    logger.warning("[cleanup] MCP runtime shutdown timed out")
                except Exception as e:
                    logger.error(f"[cleanup] Error shutting down MCP runtime: {e}")

            cleanup_tasks.append(asyncio.create_task(cleanup_mcp()))

            # Add LSP manager shutdown
            async def cleanup_lsp():
                try:
                    async with asyncio_timeout(10):
                        await shutdown_lsp_manager()
                except asyncio.TimeoutError:
                    logger.warning("[cleanup] LSP manager shutdown timed out")
                except Exception as e:
                    logger.error(f"[cleanup] Error shutting down LSP manager: {e}")

            cleanup_tasks.append(asyncio.create_task(cleanup_lsp()))

            # Add background shell shutdown
            async def cleanup_shell():
                try:
                    shutdown_background_shell(force=True)
                except Exception:
                    pass  # Background shell cleanup is best-effort

            cleanup_tasks.append(asyncio.create_task(cleanup_shell()))

            # Wait for all cleanup tasks with overall timeout
            try:
                async with asyncio_timeout(30):
                    results = await asyncio.gather(*cleanup_tasks, return_exceptions=True)
                    # Check for any exceptions that occurred
                    for i, result in enumerate(results):
                        if isinstance(result, Exception):
                            logger.error(f"[cleanup] Task {i} failed: {result}")
            except asyncio.TimeoutError:
                logger.warning("[cleanup] Cleanup tasks timed out after 30s")
            except Exception as e:
                logger.error(f"[cleanup] Error during cleanup: {e}")

            logger.info("Stdio protocol handler shutdown complete")


__all__ = ["StdioProtocolHandler"]
