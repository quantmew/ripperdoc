"""Control request handling for stdio protocol handler."""

from __future__ import annotations

import inspect
import logging
import asyncio
from dataclasses import replace
from typing import Any

from pydantic import ValidationError

from ripperdoc.core.hooks.manager import hook_manager
from ripperdoc.core.tool_defaults import get_default_tools
from ripperdoc.core.message_utils import format_pydantic_errors
from ripperdoc.core.output_styles import resolve_output_style
from ripperdoc.core.permission_engine import PermissionResult
from ripperdoc.protocol.models import (
    JsonRpcErrorCodes,
    PermissionResponseAllow,
    PermissionResponseDeny,
    ToolCallRequest,
    model_to_dict,
)
from ripperdoc.utils.permissions import ToolRule
from ripperdoc.tools.dynamic_mcp_tool import (
    load_dynamic_mcp_tools_async,
    merge_tools_with_dynamic,
)
from ripperdoc.utils.mcp import (
    McpServerInfo,
    load_mcp_server_configs,
    load_mcp_servers_async,
    parse_mcp_server_configs,
    set_mcp_runtime_overrides,
    shutdown_mcp_runtime,
)
from .error_codes import resolve_protocol_request_error_code
from .timeouts import STDIO_HOOK_TIMEOUT_SEC

logger = logging.getLogger("ripperdoc.protocol.stdio.handler")


class _UnknownControlMethodError(RuntimeError):
    """Raised when a control request method is unsupported."""


class StdioControlMixin:
    def _coerce_tool_request(self, request: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Normalize JSON-RPC payload into (name, arguments)."""
        name = str(request.get("name") or "").strip()
        arguments: dict[str, Any] = request.get("arguments") or {}
        if not isinstance(arguments, dict):
            arguments = {"value": arguments}
        return name, arguments

    async def _handle_control_request(self, message: dict[str, Any]) -> None:
        """Handle inbound control requests from SDK."""
        request_id = message.get("id")
        if request_id is None:
            request_id = message.get("request_id")
        if request_id is None:
            logger.warning("[stdio] control request missing id: %s", message)
            return

        method = message.get("method")
        if isinstance(method, str):
            method = method.strip()
        else:
            method = ""
        request: Any = message.get("params")
        request_type = message.get("type")

        if request_type == "control_request":
            request_payload = message.get("request")
            subtype = str((request_payload or {}).get("subtype") or "").strip()
            method = subtype
            request = request_payload
        if not method:
            method = ""

        if request is None:
            request = {}
        if not isinstance(request, dict):
            request = {"value": request}
        request = dict(request)

        async with self._request_lock:
            try:
                if method == "initialize":
                    await self._handle_initialize(request, str(request_id))
                elif method in {"sampling/createMessage", "query"}:
                    await self._handle_query(request, str(request_id))
                elif method == "ping":
                    await self._write_control_response(str(request_id), response={})
                elif method == "tools/call":
                    await self._handle_tools_call(request, str(request_id))
                elif method == "mcp_status":
                    await self._handle_mcp_status(request, str(request_id))
                elif method == "interrupt":
                    await self._handle_interrupt(request, str(request_id))
                elif method == "set_permission_mode":
                    await self._handle_set_permission_mode(request, str(request_id))
                elif method == "set_model":
                    await self._handle_set_model(request, str(request_id))
                elif method == "set_output_style":
                    await self._handle_set_output_style(request, str(request_id))
                elif method == "rewind_files":
                    await self._handle_rewind_files(request, str(request_id))
                elif method == "mcp_set_servers":
                    await self._handle_mcp_set_servers(request, str(request_id))
                elif method == "mcp_reconnect":
                    await self._handle_mcp_reconnect(request, str(request_id))
                elif method == "mcp_toggle":
                    await self._handle_mcp_toggle(request, str(request_id))
                elif method == "mcp_authenticate":
                    await self._handle_mcp_authenticate(request, str(request_id))
                elif method == "mcp_clear_auth":
                    await self._handle_mcp_clear_auth(request, str(request_id))
                elif method == "can_use_tool":
                    await self._handle_can_use_tool(request, str(request_id))
                else:
                    raise _UnknownControlMethodError(f"Unknown control method '{method}'")
            except Exception as e:
                logger.error("Error handling control request: %s", e, exc_info=True)
                error_code = (
                    int(JsonRpcErrorCodes.MethodNotFound)
                    if isinstance(e, _UnknownControlMethodError)
                    else resolve_protocol_request_error_code(
                        e,
                        default=JsonRpcErrorCodes.InternalError,
                    )
                )
                await self._write_control_response(
                    str(request_id),
                    error={
                        "code": error_code,
                        "message": str(e),
                    },
                )

    async def _handle_tools_call(self, request: dict[str, Any], request_id: str) -> None:
        """Handle SDK-driven `tools/call` callbacks used by control integrations."""
        try:
            validated = ToolCallRequest.model_validate(request)
        except ValidationError as exc:
            await self._write_control_response(
                request_id,
                error={
                    "code": int(JsonRpcErrorCodes.InvalidParams),
                    "message": f"Invalid tools/call request: {format_pydantic_errors(exc)}",
                },
            )
            return

        tool_name, tool_input = self._coerce_tool_request(validated.model_dump())

        if not tool_name:
            await self._write_control_response(
                request_id,
                error={
                    "code": int(JsonRpcErrorCodes.InvalidParams),
                    "message": "Missing tool name in tools/call",
                },
            )
            return

        if tool_name == "ripperdoc.can_use_tool":
            tool_name_in = tool_input.get("tool_name")
            input_payload = tool_input.get("input") or {}
            if not tool_name_in:
                await self._write_control_response(
                    request_id,
                    error={
                        "code": int(JsonRpcErrorCodes.InvalidParams),
                        "message": "Missing tool_name argument",
                    },
                )
                return
            request_payload = {
                "tool_name": tool_name_in,
                "input": input_payload,
                "force_prompt": bool(tool_input.get("force_prompt")),
            }
            await self._handle_can_use_tool(request_payload, request_id)
            return

        if tool_name == "ripperdoc.hook_callback":
            await self._handle_hook_callback(tool_input, request_id)
            return

        await self._write_control_response(
            request_id,
            response=model_to_dict(
                PermissionResponseDeny(message=f"Tool '{tool_name}' is not available")
            ),
        )

    async def _handle_set_permission_mode(self, request: dict[str, Any], request_id: str) -> None:
        mode = self._normalize_permission_mode(
            request.get("mode")
            if request.get("mode") is not None
            else request.get("permission_mode", "default")
        )
        self._apply_permission_mode(mode)
        applied_mode = (
            getattr(self._query_context, "permission_mode", mode)
            if getattr(self, "_query_context", None) is not None
            else mode
        )

        await self._write_control_response(
            request_id,
            response={"status": "permission_mode_set", "mode": applied_mode},
        )

    async def _handle_set_model(self, request: dict[str, Any], request_id: str) -> None:
        model = request.get("model")
        if self._query_context:
            self._query_context.model = model or "main"
        await self._write_control_response(
            request_id,
            response={"status": "model_set", "model": model},
        )

    def _resolve_mcp_server_name(self, requested_name: Any, available_names: list[str]) -> str | None:
        target = str(requested_name or "").strip()
        if not target:
            return None
        if target in available_names:
            return target
        lowered_matches = [name for name in available_names if name.lower() == target.lower()]
        if len(lowered_matches) == 1:
            return lowered_matches[0]
        return None

    def _sync_mcp_runtime_overrides(self) -> None:
        disabled = self._mcp_disabled_servers if self._mcp_disabled_servers else None
        set_mcp_runtime_overrides(
            self._project_path,
            servers=self._mcp_server_overrides,
            disabled=disabled,
        )

    async def _reload_mcp_runtime(self) -> list[McpServerInfo]:
        self._sync_mcp_runtime_overrides()
        await shutdown_mcp_runtime()
        return await load_mcp_servers_async(self._project_path)

    def _clone_mcp_config_map(self, configs: dict[str, McpServerInfo]) -> dict[str, McpServerInfo]:
        return {name: replace(info) for name, info in configs.items()}

    def _ensure_mutable_mcp_configs(self) -> dict[str, McpServerInfo]:
        if self._mcp_server_overrides is None:
            self._mcp_server_overrides = self._clone_mcp_config_map(
                load_mcp_server_configs(self._project_path)
            )
        return self._clone_mcp_config_map(self._mcp_server_overrides)

    async def _refresh_query_context_dynamic_tools(self) -> None:
        if not self._query_context:
            return
        tools = get_default_tools()
        if self._disable_slash_commands:
            tools = [tool for tool in tools if getattr(tool, "name", None) != "Skill"]
        tools = self._apply_tool_filters(
            tools,
            allowed_tools=self._allowed_tools,
            disallowed_tools=self._disallowed_tools,
            tools_list=self._tools_list,
        )
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

    def _format_mcp_server_status(self, server: McpServerInfo) -> dict[str, Any]:
        config: dict[str, Any]
        if server.type in {"sse", "http", "streamable-http"}:
            config = {
                "type": server.type,
                "url": server.url,
                "headers": server.headers or None,
            }
        else:
            config = {
                "type": "stdio",
                "command": server.command,
                "args": server.args or [],
            }

        tools_payload = [
            {
                "name": tool.name,
                "annotations": dict(getattr(tool, "annotations", {}) or {}),
            }
            for tool in (server.tools or [])
        ]

        return {
            "name": server.name,
            "status": server.status,
            "type": server.type,
            "error": server.error,
            "serverInfo": (
                {"version": server.server_version}
                if server.server_version
                else None
            ),
            "config": config,
            "tools": tools_payload,
            "resources": len(server.resources),
            "commands": [tool.name for tool in server.tools],
        }

    def _build_mcp_status_response(
        self,
        servers: list[McpServerInfo],
        *,
        status: str = "mcp_status",
    ) -> dict[str, Any]:
        connected = [server.name for server in servers if server.status == "connected"]
        failed = [server.name for server in servers if server.status == "failed"]
        unavailable = [server.name for server in servers if server.status == "unavailable"]
        configuring = [server.name for server in servers if server.status == "connecting"]
        configured = [self._format_mcp_server_status(server) for server in servers]
        return {
            "status": status,
            "connected": connected,
            "failed": failed,
            "unavailable": unavailable,
            "connecting": configuring,
            "servers": configured,
            # Claude-style field alias
            "mcpServers": configured,
        }

    async def _handle_mcp_status(self, _request: dict[str, Any], request_id: str) -> None:
        """Return MCP runtime status and server summaries."""
        try:
            servers = await load_mcp_servers_async(self._project_path)
            await self._write_control_response(
                request_id,
                response=self._build_mcp_status_response(servers, status="mcp_status"),
            )
        except Exception as exc:
            logger.error("[stdio] Failed to read MCP status: %s", exc)
            await self._write_control_response(
                request_id,
                error={
                    "code": int(JsonRpcErrorCodes.InternalError),
                    "message": f"Failed to load MCP status: {exc}",
                },
            )

    async def _handle_mcp_set_servers(self, request: dict[str, Any], request_id: str) -> None:
        raw_servers = request.get("servers")
        if raw_servers is None and "mcpServers" in request:
            raw_servers = request.get("mcpServers")
        if raw_servers is None:
            await self._write_control_response(
                request_id,
                error={
                    "code": int(JsonRpcErrorCodes.InvalidParams),
                    "message": "Missing MCP server payload: expected 'servers'",
                },
            )
            return
        if not isinstance(raw_servers, (dict, list)):
            await self._write_control_response(
                request_id,
                error={
                    "code": int(JsonRpcErrorCodes.InvalidParams),
                    "message": "Invalid 'servers' payload: expected object or list",
                },
            )
            return

        parsed_configs = parse_mcp_server_configs(raw_servers)
        self._mcp_server_overrides = self._clone_mcp_config_map(parsed_configs)
        self._mcp_disabled_servers = {
            name for name in self._mcp_disabled_servers if name in parsed_configs
        }
        servers = await self._reload_mcp_runtime()
        await self._refresh_query_context_dynamic_tools()
        response = self._build_mcp_status_response(servers, status="mcp_set_servers")
        response["applied"] = sorted(parsed_configs.keys())
        await self._write_control_response(request_id, response=response)

    async def _handle_mcp_reconnect(self, request: dict[str, Any], request_id: str) -> None:
        requested_name = request.get("serverName")
        if requested_name is None:
            requested_name = request.get("server_name")

        available = list((self._mcp_server_overrides or load_mcp_server_configs(self._project_path)).keys())
        resolved_name = None
        if requested_name is not None:
            resolved_name = self._resolve_mcp_server_name(requested_name, available)
            if resolved_name is None:
                await self._write_control_response(
                    request_id,
                    error={
                        "code": int(JsonRpcErrorCodes.InvalidParams),
                        "message": f"Server not found: {requested_name}",
                    },
                )
                return

        servers = await self._reload_mcp_runtime()
        await self._refresh_query_context_dynamic_tools()
        response = self._build_mcp_status_response(servers, status="mcp_reconnect")
        if resolved_name is not None:
            response["serverName"] = resolved_name
        await self._write_control_response(request_id, response=response)

    async def _handle_mcp_toggle(self, request: dict[str, Any], request_id: str) -> None:
        requested_name = request.get("serverName")
        if requested_name is None:
            requested_name = request.get("server_name")
        enabled = bool(request.get("enabled", True))

        configs = self._ensure_mutable_mcp_configs()
        resolved_name = self._resolve_mcp_server_name(requested_name, list(configs.keys()))
        if resolved_name is None:
            await self._write_control_response(
                request_id,
                error={
                    "code": int(JsonRpcErrorCodes.InvalidParams),
                    "message": f"Server not found: {requested_name}",
                },
            )
            return

        self._mcp_server_overrides = configs
        if enabled:
            self._mcp_disabled_servers.discard(resolved_name)
        else:
            self._mcp_disabled_servers.add(resolved_name)

        servers = await self._reload_mcp_runtime()
        await self._refresh_query_context_dynamic_tools()
        response = self._build_mcp_status_response(servers, status="mcp_toggle")
        response["serverName"] = resolved_name
        response["enabled"] = enabled
        await self._write_control_response(request_id, response=response)

    async def _handle_mcp_authenticate(self, request: dict[str, Any], request_id: str) -> None:
        requested_name = request.get("serverName")
        if requested_name is None:
            requested_name = request.get("server_name")

        headers_payload = request.get("headers")
        header_updates: dict[str, str] = {}
        if isinstance(headers_payload, dict):
            header_updates = {
                str(key).strip(): str(value)
                for key, value in headers_payload.items()
                if str(key).strip()
            }
        token = request.get("token")
        if token is None:
            token = request.get("authToken")
        if token is None:
            token = request.get("auth_token")
        if token is not None and str(token).strip():
            header_updates["Authorization"] = f"Bearer {str(token).strip()}"

        if not header_updates:
            await self._write_control_response(
                request_id,
                response={
                    "status": "mcp_authenticate",
                    "serverName": requested_name,
                    "requiresUserAction": True,
                    "authUrl": request.get("authUrl") or request.get("auth_url"),
                },
            )
            return

        configs = self._ensure_mutable_mcp_configs()
        resolved_name = self._resolve_mcp_server_name(requested_name, list(configs.keys()))
        if resolved_name is None:
            await self._write_control_response(
                request_id,
                error={
                    "code": int(JsonRpcErrorCodes.InvalidParams),
                    "message": f"Server not found: {requested_name}",
                },
            )
            return

        target = configs[resolved_name]
        if target.type not in {"sse", "http", "streamable-http"}:
            await self._write_control_response(
                request_id,
                error={
                    "code": int(JsonRpcErrorCodes.InvalidParams),
                    "message": f'Server type "{target.type}" does not support authentication updates',
                },
            )
            return

        merged_headers = dict(target.headers or {})
        merged_headers.update(header_updates)
        configs[resolved_name] = replace(target, headers=merged_headers)
        self._mcp_server_overrides = configs
        self._mcp_disabled_servers.discard(resolved_name)

        await self._reload_mcp_runtime()
        await self._refresh_query_context_dynamic_tools()
        await self._write_control_response(
            request_id,
            response={
                "status": "mcp_authenticate",
                "serverName": resolved_name,
                "requiresUserAction": False,
            },
        )

    async def _handle_mcp_clear_auth(self, request: dict[str, Any], request_id: str) -> None:
        requested_name = request.get("serverName")
        if requested_name is None:
            requested_name = request.get("server_name")

        configs = self._ensure_mutable_mcp_configs()
        resolved_name = self._resolve_mcp_server_name(requested_name, list(configs.keys()))
        if resolved_name is None:
            await self._write_control_response(
                request_id,
                error={
                    "code": int(JsonRpcErrorCodes.InvalidParams),
                    "message": f"Server not found: {requested_name}",
                },
            )
            return

        target = configs[resolved_name]
        if target.type not in {"sse", "http", "streamable-http"}:
            await self._write_control_response(
                request_id,
                error={
                    "code": int(JsonRpcErrorCodes.InvalidParams),
                    "message": f'Cannot clear auth for server type "{target.type}"',
                },
            )
            return

        auth_header_keys = {
            "authorization",
            "proxy-authorization",
            "x-api-key",
            "api-key",
            "x-auth-token",
        }
        filtered_headers = {
            key: value
            for key, value in (target.headers or {}).items()
            if key.lower() not in auth_header_keys
        }
        configs[resolved_name] = replace(target, headers=filtered_headers)
        self._mcp_server_overrides = configs

        await self._reload_mcp_runtime()
        await self._refresh_query_context_dynamic_tools()
        await self._write_control_response(
            request_id,
            response={
                "status": "mcp_clear_auth",
                "serverName": resolved_name,
            },
        )

    async def _handle_interrupt(self, _request: dict[str, Any], request_id: str) -> None:
        """Attempt to interrupt in-flight tasks."""
        current_task = asyncio.current_task()
        cancelled = 0
        for task in list(self._inflight_tasks):
            if task is current_task:
                continue
            task.cancel()
            cancelled += 1
        await self._write_control_response(
            request_id,
            response={"status": "interrupt", "cancelled_tasks": cancelled},
        )

    async def _handle_set_output_style(self, request: dict[str, Any], request_id: str) -> None:
        requested_style = (
            request.get("style")
            if request.get("style") is not None
            else request.get("output_style")
            or "default"
        )
        resolved_style, _ = resolve_output_style(
            str(requested_style),
            project_path=getattr(self, "_project_path", None),
        )
        self._output_style = resolved_style.key
        await self._write_control_response(
            request_id,
            response={"status": "output_style_set", "output_style": self._output_style},
        )

    def _resolve_rewind_target_index(self, user_message_id: str) -> int | None:
        if not user_message_id:
            return None
        matches = [
            idx
            for idx, msg in enumerate(self._conversation_messages)
            if str(getattr(msg, "uuid", "")).strip() == user_message_id
            and getattr(msg, "type", "") == "user"
        ]
        if not matches:
            return None
        return matches[-1]

    async def _handle_rewind_files(self, request: dict[str, Any], request_id: str) -> None:
        user_message_id = str(request.get("user_message_id") or "").strip()
        dry_run = bool(request.get("dry_run"))
        if not user_message_id:
            await self._write_control_response(
                request_id,
                error={
                    "code": int(JsonRpcErrorCodes.InvalidParams),
                    "message": "Missing user_message_id",
                },
            )
            return

        target_index = self._resolve_rewind_target_index(user_message_id)
        if target_index is None:
            await self._write_control_response(
                request_id,
                response={
                    "canRewind": False,
                    "error": "No file checkpoint found for this message.",
                },
            )
            return

        if dry_run:
            await self._write_control_response(
                request_id,
                response={
                    "canRewind": True,
                    "filesChanged": 0,
                    "insertions": 0,
                    "deletions": 0,
                    "dryRun": True,
                },
            )
            return

        self._conversation_messages = list(self._conversation_messages[: target_index + 1])
        await self._write_control_response(
            request_id,
            response={
                "canRewind": True,
                "dryRun": False,
                "rewoundTo": user_message_id,
            },
        )

    async def _handle_hook_callback(self, request: dict[str, Any], request_id: str) -> None:
        callback_id = request.get("callback_id")
        if not callback_id:
            await self._write_control_response(
                request_id,
                error={
                    "code": int(JsonRpcErrorCodes.InvalidParams),
                    "message": "Missing callback_id in hook callback payload",
                },
            )
            return

        hook_input = request.get("input") or {}
        tool_use_id = request.get("tool_use_id")
        timeout = request.get("timeout")
        try:
            response = await self._run_sdk_hook_callback(
                callback_id=str(callback_id),
                input_data=hook_input,
                tool_use_id=tool_use_id,
                timeout=timeout,
            )
            await self._write_control_response(request_id, response=response.model_dump())
        except Exception as e:
            logger.error("[stdio] SDK hook callback failed: %s", e)
            await self._write_control_response(
                request_id,
                error={
                    "code": int(JsonRpcErrorCodes.InternalError),
                    "message": f"SDK hook callback failed: {e}",
                },
            )

    def _serialize_permission_suggestions(
        self, rule_suggestions: Any
    ) -> list[dict[str, Any]] | None:
        """Serialize rule suggestions into SDK-compatible format.

        Converts ToolRule objects and other suggestion formats into
        dictionaries suitable for JSON serialization and transmission
        to the SDK.

        Args:
            rule_suggestions: Rule suggestions from PermissionDecision

        Returns:
            List of suggestion dicts or None if empty
        """
        if not rule_suggestions:
            return None
        suggestions: list[dict[str, Any]] = []
        for suggestion in rule_suggestions:
            if isinstance(suggestion, ToolRule):
                suggestions.append(
                    {
                        "tool_name": suggestion.tool_name,
                        "rule": suggestion.rule_content,
                        "behavior": suggestion.behavior,
                    }
                )
            elif isinstance(suggestion, dict):
                suggestions.append(suggestion)
            else:
                suggestions.append({"rule": str(suggestion)})
        return suggestions or None

    def _extract_blocked_path(self, decision: Any) -> str | None:
        """Extract blocked path from permission decision.

        Args:
            decision: PermissionDecision or similar object

        Returns:
            Blocked path string or None
        """
        if decision is None:
            return None
        # Check for explicit blocked_path attribute
        if hasattr(decision, "blocked_path"):
            return decision.blocked_path
        # Check decision_reason for path info
        decision_reason = getattr(decision, "decision_reason", None)
        if isinstance(decision_reason, dict):
            return decision_reason.get("path") or decision_reason.get("blocked_path")
        return None

    def _serialize_decision_reason(self, decision: Any) -> dict[str, Any] | None:
        """Serialize decision reason for SDK transmission.

        Args:
            decision: PermissionDecision or similar object

        Returns:
            Decision reason dict or None
        """
        if decision is None:
            return None
        decision_reason = getattr(decision, "decision_reason", None)
        if decision_reason is None:
            # Build a default decision reason
            behavior = getattr(decision, "behavior", "unknown")
            message = getattr(decision, "message", None)
            return {
                "type": "permission_decision",
                "behavior": behavior,
                "message": message,
            }
        if isinstance(decision_reason, dict):
            return decision_reason
        if hasattr(decision_reason, "model_dump"):
            return decision_reason.model_dump()
        return {"type": str(decision_reason)}

    async def _run_permission_hook_race(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_use_id: str | None,
        permission_suggestions: list[dict[str, Any]] | None,
        blocked_path: str | None,
        decision_reason: dict[str, Any] | None,
        agent_id: str | None,
    ) -> dict[str, Any] | None:
        """Run PermissionRequest hooks concurrently with SDK request.

        Implements Claude Code's Promise.race pattern where hooks and
        SDK permission requests compete. First decisive result wins.

        Args:
            tool_name: Name of the tool requesting permission
            tool_input: Input for the tool
            tool_use_id: Unique ID for this tool use
            permission_suggestions: Suggested permission rules
            blocked_path: Path that was blocked (if any)
            decision_reason: Reason for the decision
            agent_id: Agent ID (if applicable)

        Returns:
            Hook decision dict with 'behavior', 'updated_input', etc. or None
        """
        if not self._sdk_can_use_tool_enabled:
            return None

        # Create abort controller pattern like Claude Code
        abort_event = asyncio.Event()

        async def run_hook_decision() -> dict[str, Any] | None:
            """Run PermissionRequest hooks and return first decisive result."""
            try:
                hook_input = {
                    "hook_event_name": "PermissionRequest",
                    "tool_name": tool_name,
                    "tool_input": tool_input,
                    "permission_suggestions": permission_suggestions,
                    "tool_use_id": tool_use_id,
                }
                hook_results = await hook_manager.run_hook_async(
                    "PermissionRequest",
                    hook_input,
                    timeout=STDIO_HOOK_TIMEOUT_SEC,
                )
                for hook_result in hook_results:
                    if abort_event.is_set():
                        return None
                    decision = getattr(hook_result, "decision", None)
                    if decision is None:
                        continue
                    behavior = getattr(decision, "behavior", None)
                    if behavior in ("allow", "deny"):
                        return {
                            "source": "hook",
                            "behavior": behavior,
                            "message": getattr(decision, "message", None),
                            "updated_input": getattr(decision, "updated_input", None),
                            "updated_permissions": getattr(decision, "updated_permissions", None),
                        }
                return None
            except asyncio.CancelledError:
                return None
            except Exception as e:
                logger.debug("[stdio] PermissionRequest hook error: %s", e)
                return None

        async def run_sdk_request() -> dict[str, Any]:
            """Send can_use_tool request to SDK and await response."""
            try:
                sdk_request = {
                    "subtype": "can_use_tool",
                    "tool_name": tool_name,
                    "input": tool_input,
                    "tool_use_id": tool_use_id,
                    "agent_id": agent_id,
                    "permission_suggestions": permission_suggestions,
                    "blocked_path": blocked_path,
                    "decision_reason": decision_reason,
                }
                response = await self._send_control_request(
                    subtype="can_use_tool",
                    request=sdk_request,
                    timeout=STDIO_HOOK_TIMEOUT_SEC,
                )
                return {
                    "source": "sdk",
                    "response": response,
                }
            except asyncio.CancelledError:
                return {"source": "sdk", "error": "cancelled"}
            except Exception as e:
                return {"source": "sdk", "error": str(e)}

        # Run both concurrently, first to complete wins
        hook_task = asyncio.create_task(run_hook_decision())
        sdk_task = asyncio.create_task(run_sdk_request())

        try:
            done, pending = await asyncio.wait(
                {hook_task, sdk_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Cancel pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # Process completed task
            for task in done:
                result = task.result()
                if result is None:
                    continue

                if result.get("source") == "hook":
                    # Hook returned decisive result
                    abort_event.set()
                    behavior = result.get("behavior")
                    if behavior in ("allow", "deny"):
                        return result
                    # Hook didn't decide, wait for SDK
                    continue

                if result.get("source") == "sdk":
                    response = result.get("response")
                    if response:
                        return {
                            "source": "sdk",
                            "behavior": response.get("behavior", "ask"),
                            "message": response.get("message"),
                            "updated_input": response.get("updatedInput"),
                            "tool_use_id": response.get("toolUseID"),
                        }
                    error = result.get("error")
                    if error:
                        return {
                            "source": "sdk",
                            "behavior": "deny",
                            "message": f"SDK permission request failed: {error}",
                        }

            return None
        except asyncio.CancelledError:
            abort_event.set()
            hook_task.cancel()
            sdk_task.cancel()
            raise

    async def _handle_can_use_tool(self, request: dict[str, Any], request_id: str) -> None:
        tool_name = request.get("tool_name") or ""
        tool_input = request.get("input")
        if tool_input is None:
            tool_input = {}

        if not tool_name:
            await self._write_control_response(
                request_id,
                error={
                    "code": int(JsonRpcErrorCodes.InvalidParams),
                    "message": "Missing tool_name",
                },
            )
            return

        if not self._query_context:
            await self._write_control_response(
                request_id,
                error={
                    "code": int(JsonRpcErrorCodes.InvalidRequest),
                    "message": "Session not initialized",
                },
            )
            return

        tool = self._query_context.tool_registry.get(tool_name)
        if tool is None:
            perm_response = PermissionResponseDeny(message=f"Tool '{tool_name}' not found")
            await self._write_control_response(
                request_id,
                response=model_to_dict(perm_response),
            )
            return

        if tool_input and hasattr(tool_input, "model_dump"):
            tool_input = tool_input.model_dump()
        elif tool_input and hasattr(tool_input, "dict") and callable(getattr(tool_input, "dict")):
            tool_input = tool_input.dict()
        if not isinstance(tool_input, dict):
            tool_input = {"value": str(tool_input)}
        if tool_name == "Task" and isinstance(tool_input, dict):
            tool_input = self._apply_tool_input_aliases(tool, tool_input)

        try:
            parsed_input = tool.input_schema(**tool_input)
        except ValidationError as ve:
            detail = format_pydantic_errors(ve)
            perm_response = PermissionResponseDeny(message=f"Invalid input for tool '{tool_name}': {detail}")
            await self._write_control_response(request_id, response=model_to_dict(perm_response))
            return
        except (TypeError, ValueError) as exc:
            perm_response = PermissionResponseDeny(message=f"Invalid input for tool '{tool_name}': {exc}")
            await self._write_control_response(request_id, response=model_to_dict(perm_response))
            return

        permission_checker = self._can_use_tool or self._local_can_use_tool
        force_prompt = bool(request.get("force_prompt"))
        tool_use_id = request.get("tool_use_id")
        agent_id = request.get("agent_id")

        try:
            result = permission_checker(tool, parsed_input)
            if inspect.isawaitable(result):
                result = await result

            allowed = False
            message = None
            updated_input = None
            decision = None

            if isinstance(result, PermissionResult):
                allowed = bool(result.result)
                message = result.message
                updated_input = result.updated_input
                decision = result.decision
            elif isinstance(result, tuple) and len(result) == 2:
                allowed = bool(result[0])
                message = result[1]
            else:
                allowed = bool(result)

            # If not definitively allowed or denied, run hook/SDK race
            if not allowed and message is None and self._sdk_can_use_tool_enabled:
                permission_suggestions = self._serialize_permission_suggestions(
                    getattr(decision, "rule_suggestions", None)
                    if decision
                    else None
                )
                blocked_path = self._extract_blocked_path(decision)
                decision_reason = self._serialize_decision_reason(decision)

                race_result = await self._run_permission_hook_race(
                    tool_name=tool_name,
                    tool_input=tool_input,
                    tool_use_id=tool_use_id,
                    permission_suggestions=permission_suggestions,
                    blocked_path=blocked_path,
                    decision_reason=decision_reason,
                    agent_id=agent_id,
                )

                if race_result is not None:
                    behavior = race_result.get("behavior")
                    if behavior == "allow":
                        allowed = True
                        updated_input = race_result.get("updated_input") or updated_input
                        message = race_result.get("message")
                    elif behavior == "deny":
                        allowed = False
                        message = race_result.get("message") or message

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
                    normalized_input = self._apply_tool_input_aliases(tool, normalized_input)
                perm_response = PermissionResponseAllow(
                    updatedInput=normalized_input,
                    toolUseID=tool_use_id,
                    decisionReason=self._serialize_decision_reason(decision),
                )
                await self._write_control_response(request_id, response=model_to_dict(perm_response))
            else:
                perm_response = PermissionResponseDeny(
                    message=message or "",
                    toolUseID=tool_use_id,
                    decisionReason=self._serialize_decision_reason(decision),
                )
                await self._write_control_response(request_id, response=model_to_dict(perm_response))
        except Exception as e:
            logger.error("Error in permission check: %s", e)
            await self._write_control_response(
                request_id,
                error={
                    "code": int(JsonRpcErrorCodes.InternalError),
                    "message": str(e),
                },
            )
