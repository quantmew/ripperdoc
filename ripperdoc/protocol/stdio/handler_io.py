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
    ControlResponseError,
    ControlResponseMessage,
    ControlResponseSuccess,
    model_to_dict,
)

from .timeouts import STDIO_HOOK_TIMEOUT_SEC, STDIO_READ_TIMEOUT_SEC

logger = logging.getLogger("ripperdoc.protocol.stdio.handler")


class StdioIOMixin:
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
            effective_timeout = STDIO_HOOK_TIMEOUT_SEC if timeout is None else timeout
            if effective_timeout <= 0:
                return await future
            return await asyncio.wait_for(future, timeout=effective_timeout)
        except asyncio.TimeoutError:
            self._pending_requests.pop(request_id, None)
            raise

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
