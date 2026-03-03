"""IO and control message helpers for stdio protocol handler."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import uuid
from collections.abc import AsyncIterator
from typing import Any

from ripperdoc.protocol.models import (
    DEFAULT_PROTOCOL_VERSION,
    JsonRpcError,
    JsonRpcErrorCodes,
    JsonRpcResponse,
    JsonRpcResponseError,
    model_to_dict,
)

from .timeouts import STDIO_HOOK_TIMEOUT_SEC, STDIO_READ_TIMEOUT_SEC

logger = logging.getLogger("ripperdoc.protocol.stdio.handler")

# Maximum number of resolved tool_use_ids to track for duplicate detection
_MAX_RESOLVED_TOOL_USE_IDS = 1000


class StdioIOMixin:
    _resolved_tool_use_ids: set[str]

    def _track_resolved_tool_use_id(self, tool_use_id: str) -> None:
        """Track a resolved tool_use_id with LRU-style cleanup when exceeding limit."""
        self._resolved_tool_use_ids.add(tool_use_id)
        # LRU-style cleanup: remove oldest entries when exceeding limit
        while len(self._resolved_tool_use_ids) > _MAX_RESOLVED_TOOL_USE_IDS:
            # Remove the first (oldest) entry
            oldest = next(iter(self._resolved_tool_use_ids), None)
            if oldest:
                self._resolved_tool_use_ids.discard(oldest)
            else:
                break

    def _extract_tool_use_id_from_control_response(
        self,
        response_payload: dict[str, Any] | None,
    ) -> str | None:
        response_data = (response_payload or {}).get("response")
        if isinstance(response_data, dict):
            for key in ("toolUseID", "tool_use_id"):
                candidate = response_data.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()
        return None

    def _normalize_control_response_subtype(
        self,
        response_payload: dict[str, Any],
    ) -> str:
        subtype = response_payload.get("subtype")
        if isinstance(subtype, str) and subtype.strip():
            return subtype.strip()

        if response_payload.get("error") is not None:
            return "error"

        if "response" in response_payload or response_payload.get("result") is not None:
            return "success"

        return "success"

    async def _send_control_cancel_request(self, request_id: str | int) -> None:
        """Send a best-effort control cancel envelope for an in-flight request id."""
        if request_id is None:
            return
        try:
            await self._write_message(
                {
                    "type": "control_cancel_request",
                    "request_id": str(request_id),
                }
            )
        except Exception as exc:  # noqa: BLE001 - best-effort cancellation signal
            logger.debug("[stdio] Failed to send control cancel request id=%s: %s", request_id, exc)

    async def _handle_control_cancel_request(self, message: dict[str, Any]) -> None:
        """Handle inbound control cancel messages for pending requests/task handlers."""
        request_id = message.get("request_id")
        if request_id is None:
            request_id = message.get("id")
        if request_id is None:
            logger.warning("[stdio] control_cancel_request missing request_id: %s", message)
            return

        request_key = str(request_id)
        future = self._pending_requests.pop(request_key, None)
        if future is not None and not future.done():
            future.set_exception(
                JsonRpcResponseError(
                    code=int(JsonRpcErrorCodes.RequestTimeout),
                    message="Control request cancelled by peer",
                )
            )

        task = self._request_tasks.pop(request_key, None)
        if task is not None and not task.done():
            task.cancel()

    async def _write_message(self, message: dict[str, Any]) -> None:
        """Write a JSON message to stdout."""

        msg_type = message.get("type", "json-rpc")
        if self._output_format == "stream-json":
            json_data = json.dumps(message, ensure_ascii=False)
            logger.debug(f"[stdio] Writing message: type={msg_type}, json_length={len(json_data)}")
            sys.stdout.write(json_data + "\n")
            sys.stdout.flush()
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
        """Flush buffered output if using non-stream output format."""

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
        request_id: str | int,
        response: dict[str, Any] | None = None,
        error: str | JsonRpcError | dict[str, Any] | JsonRpcResponseError | None = None,
    ) -> None:
        """Write a control protocol response envelope."""

        if request_id is None:
            logger.debug("[stdio] Skip response write: missing request_id")
            return

        control_response: dict[str, Any] = {
            "type": "control_response",
            "response": {
                "request_id": str(request_id),
                "subtype": "success",
                "response": response or {},
            },
        }

        if isinstance(error, JsonRpcResponseError):
            error_payload = JsonRpcError(
                code=int(error.code),
                message=error.message,
                data=error.data,
            )
            control_response["response"]["subtype"] = "error"
            control_response["response"]["error"] = model_to_dict(error_payload)
            control_response["response"].pop("response", None)
        elif isinstance(error, dict):
            code = error.get("code")
            message = error.get("message")
            data = error.get("data")
            if not isinstance(code, int):
                code = int(JsonRpcErrorCodes.InternalError)
            if not isinstance(message, str) or not message:
                message = "Request failed"
            control_response["response"]["subtype"] = "error"
            control_response["response"]["error"] = {
                "code": code,
                "message": message,
                "data": data,
            }
            control_response["response"].pop("response", None)
        elif isinstance(error, JsonRpcError):
            control_response["response"]["subtype"] = "error"
            control_response["response"]["error"] = model_to_dict(error)
            control_response["response"].pop("response", None)
        elif isinstance(error, str):
            control_response["response"]["subtype"] = "error"
            control_response["response"]["error"] = {
                "code": int(JsonRpcErrorCodes.InternalError),
                "message": error,
            }
            control_response["response"].pop("response", None)

        await self._write_message(control_response)

    async def _handle_control_response(self, message: dict[str, Any]) -> None:
        """Handle protocol responses from the SDK and resolve awaiters."""

        request_id: str | None = None
        response_payload: dict[str, Any] | None = None
        if message.get("type") == "control_response":
            payload = message.get("response")
            if isinstance(payload, dict):
                request_id = payload.get("request_id")
                response_payload = payload
        elif message.get("jsonrpc") == "2.0" and message.get("id") is not None:
            request_id = message.get("id")
            response_payload = {
                "request_id": str(request_id),
                "subtype": "error" if message.get("error") is not None else "success",
                "response": message.get("result"),
                "error": message.get("error"),
            }

        if request_id is None:
            logger.warning("[stdio] control response missing id")
            return

        request_key = str(request_id)
        future = self._pending_requests.pop(request_key, None)
        if future is None:
            tool_use_id = self._extract_tool_use_id_from_control_response(response_payload)
            if tool_use_id is None:
                logger.debug("[stdio] No pending request for response id=%s", request_id)
                return

            if tool_use_id in self._resolved_tool_use_ids:
                logger.debug(
                    "[stdio] Ignoring duplicate control_response for already-resolved toolUseID=%s request_id=%s",
                    tool_use_id,
                    request_id,
                )
                return

            # Track resolved tool_use_id with LRU-style cleanup (max 1000 entries)
            self._track_resolved_tool_use_id(tool_use_id)
            logger.debug(
                "[stdio] Tracking orphan control_response toolUseID=%s for request_id=%s",
                tool_use_id,
                request_id,
            )
            return

        if response_payload is None:
            logger.warning("[stdio] control response payload missing data for request_id=%s", request_key)
            return

        response_payload["subtype"] = self._normalize_control_response_subtype(response_payload)
        if response_payload["subtype"] == "error":
            error_data = response_payload.get("error")
            if isinstance(error_data, dict):
                future.set_exception(
                    JsonRpcResponseError(
                        code=int(error_data.get("code") or JsonRpcErrorCodes.InternalError),
                        message=str(error_data.get("message") or "Unknown protocol error"),
                        data=error_data.get("data"),
                    )
                )
            else:
                future.set_exception(
                    JsonRpcResponseError(
                        code=int(JsonRpcErrorCodes.InternalError),
                        message="Unknown protocol error",
                        data=error_data,
                    )
                    )
            return

        tool_use_id = self._extract_tool_use_id_from_control_response(response_payload)
        if tool_use_id:
            self._track_resolved_tool_use_id(tool_use_id)
        future.set_result((response_payload or {}).get("response"))

    async def _send_control_request(
        self,
        *args: Any,
        timeout: float | None = None,
        request_id: str | None = None,
        subtype: str | None = None,
        request: dict[str, Any] | None = None,
    ) -> Any:
        """Send a control request and await the response."""
        if args:
            if isinstance(args[0], str):
                subtype = str(args[0])
                if len(args) > 1:
                    request = args[1]  # type: ignore[assignment]
            elif isinstance(args[0], dict):
                request = args[0]
                if len(args) > 1 and subtype is None:
                    subtype = str(args[1])
            else:
                raise TypeError("Invalid _send_control_request positional arguments")
        if subtype is None:
            if request is not None and isinstance(request, dict):
                request_name = request.get("name")
                if request_name in {"ripperdoc.can_use_tool", "ripperdoc.hook_callback"}:
                    subtype = "tools/call"
            if subtype is None:
                raise TypeError("_send_control_request requires a subtype")
        if request is None:
            request = {}

        if self._output_format != "stream-json":
            raise RuntimeError("Control request requires stream-json output mode")

        request_id = request_id or f"cli_{uuid.uuid4().hex}"
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        self._pending_requests[str(request_id)] = future

        await self._write_message(
            {
                "type": "control_request",
                "request_id": request_id,
                "request": {"subtype": str(subtype), **(request or {})},
            }
        )

        try:
            effective_timeout = STDIO_HOOK_TIMEOUT_SEC if timeout is None else timeout
            if effective_timeout <= 0:
                return await future
            return await asyncio.wait_for(future, timeout=effective_timeout)
        except asyncio.TimeoutError:
            self._pending_requests.pop(str(request_id), None)
            await self._send_control_cancel_request(str(request_id))
            raise
        except asyncio.CancelledError:
            self._pending_requests.pop(str(request_id), None)
            await self._send_control_cancel_request(str(request_id))
            raise

    async def _write_message_stream(self, message_dict: dict[str, Any]) -> None:
        """Write a regular message to the output stream."""

        await self._write_message(message_dict)

    async def _read_line(self) -> str | None:
        """Read a single line from stdin with timeout."""

        while True:
            try:
                if STDIO_READ_TIMEOUT_SEC <= 0:
                    line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
                else:
                    line = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline),
                        timeout=STDIO_READ_TIMEOUT_SEC,
                    )
                if not line:
                    return None
                return line.rstrip("\n\r")
            except asyncio.TimeoutError:
                logger.debug(
                    "[stdio] stdin read timed out after %ds; continuing to wait",
                    STDIO_READ_TIMEOUT_SEC,
                )
                continue
            except (OSError, IOError) as e:
                logger.error("Error reading from stdin: %s", e)
                return None

    async def _read_messages(self) -> AsyncIterator[dict[str, Any]]:
        """Read and parse JSON messages from stdin with incremental buffering."""

        json_buffer = ""
        decoder = json.JSONDecoder()
        consecutive_empty_lines = 0
        max_empty_lines = 100

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
                            "[stdio] Too many empty lines (%d), stopping",
                            max_empty_lines,
                        )
                        break
                    continue

                consecutive_empty_lines = 0
                for json_line in line.split("\n"):
                    json_line = json_line.strip()
                    if not json_line:
                        continue

                    if self._input_format == "auto" and not json_buffer:
                        if json_line[:1] not in "{[":
                            for msg in self._coerce_auto_messages(json_line):
                                yield msg
                            continue

                    json_buffer += json_line
                    if len(json_buffer) > 10_000_000:
                        logger.error("[stdio] JSON buffer too large, resetting")
                        json_buffer = ""
                        continue

                    while json_buffer:
                        try:
                            data, index = decoder.raw_decode(json_buffer)
                        except json.JSONDecodeError:
                            if json_buffer.lstrip().startswith(("{", "[")):
                                break
                            json_buffer = ""
                            break
                        else:
                            json_buffer = json_buffer[index:].lstrip()
                            if self._input_format == "auto":
                                for msg in self._coerce_auto_messages(data):
                                    yield msg
                            else:
                                if isinstance(data, dict):
                                    yield data
                                elif isinstance(data, list):
                                    for item in data:
                                        if isinstance(item, dict):
                                            yield item
                                else:
                                    logger.warning(
                                        "[stdio] Received non-object JSON in stream-json mode; skipping"
                                    )

        except asyncio.CancelledError:
            logger.info("[stdio] Message reader cancelled")
            raise
        except Exception as e:
            logger.error(
                "[stdio] Error in message reader: %s: %s",
                type(e).__name__,
                e,
                exc_info=True,
            )
            raise

    def _generate_auto_request_id(self) -> str:
        """Generate a request id for auto input format."""

        return f"auto_{uuid.uuid4().hex}"

    def _coerce_auto_messages(self, data: Any) -> list[dict[str, Any]]:
        """Coerce permissive auto input into control request messages."""

        messages: list[dict[str, Any]] = []

        if isinstance(data, list):
            for item in data:
                messages.extend(self._coerce_auto_messages(item))
            return messages

        if isinstance(data, dict):
            if data.get("type") == "control_request":
                if isinstance(data.get("request"), dict):
                    return [data]

            if "method" in data:
                return [data]

            if "type" in data and data.get("type") == "user":
                message_payload = self._extract_prompt_from_user_content_for_auto(data)
                if message_payload:
                    request_id = data.get("uuid") or self._generate_auto_request_id()
                    messages.append(
                        self._build_query_control_request(
                            "\n".join(
                                block.get("text", "").strip()
                                for block in message_payload
                                if block.get("type") == "text" and isinstance(block.get("text"), str)
                            )
                            or "",
                            request_id=request_id,
                        )
                    )
                return messages

            if any(key in data for key in ("prompt", "options", "mode", "model", "subtype")):
                prompt = str(data.get("prompt") or "").strip()
                if prompt:
                    request_id = data.get("request_id") or self._generate_auto_request_id()
                    messages.append(
                        self._build_query_control_request(
                            prompt,
                            request_id=request_id,
                            max_tokens=int(data.get("maxTokens", 0) or 1024),
                        )
                    )

            return messages

        if isinstance(data, str):
            prompt = data.strip()
            if not prompt:
                return messages
            messages.append(self._build_query_control_request(prompt))

        return messages

    def _extract_prompt_from_user_content_for_auto(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert legacy `type=user` payload content into sampling messages."""

        inner_message = message.get("message")
        if not isinstance(inner_message, dict):
            return []

        content = inner_message.get("content")
        if isinstance(content, str):
            value = content.strip()
            if not value:
                return []
            return [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": value,
                        },
                    ],
                }
            ]

        if isinstance(content, list):
            normalized: list[dict[str, Any]] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                text = block.get("text")
                if block.get("type") == "text" and isinstance(text, str) and text.strip():
                    normalized.append({"type": "text", "text": text.strip()})
            if normalized:
                return [
                    {
                        "role": "user",
                        "content": normalized,
                    }
                ]

        return []

    def _build_query_control_request(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        """Build a control query request from user prompt input."""
        return {
            "type": "control_request",
            "request_id": request_id or self._generate_auto_request_id(),
            "request": {
                "subtype": "query",
                "prompt": str(prompt),
                "maxTokens": max_tokens,
            },
        }
