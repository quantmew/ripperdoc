"""Control request handling for stdio protocol handler."""

from __future__ import annotations

import inspect
import logging
from typing import Any

from pydantic import ValidationError

from ripperdoc.core.output_styles import resolve_output_style
from ripperdoc.core.permissions import PermissionResult
from ripperdoc.core.query_utils import format_pydantic_errors
from ripperdoc.protocol.models import PermissionResponseAllow, PermissionResponseDeny, model_to_dict

logger = logging.getLogger("ripperdoc.protocol.stdio.handler")


class StdioControlMixin:
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

                elif request_subtype == "set_output_style":
                    await self._handle_set_output_style(request, request_id)

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

    async def _handle_set_output_style(self, request: dict[str, Any], request_id: str) -> None:
        """Handle set_output_style request from SDK."""
        requested_style = (
            request.get("style")
            or request.get("output_style")
            or request.get("outputStyle")
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
        perm_response: PermissionResponseAllow | PermissionResponseDeny
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

        # Use local permission checker for inbound SDK compatibility requests.
        permission_checker = self._local_can_use_tool or self._can_use_tool
        if permission_checker:
            try:
                result = permission_checker(tool, parsed_input)
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
