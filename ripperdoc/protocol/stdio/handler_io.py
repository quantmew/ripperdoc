"""IO and control message helpers for stdio protocol handler."""

from __future__ import annotations

import asyncio
import contextlib
import json
import importlib
import logging
import random
import os
import sys
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator
from urllib.parse import urlparse, urlunparse
from typing import Any, Callable, cast

from ripperdoc.protocol.models import (
    JsonRpcError,
    JsonRpcErrorCodes,
    JsonRpcResponseError,
    model_to_dict,
)
from ripperdoc.utils.coerce import parse_boolish

from .timeouts import STDIO_HOOK_TIMEOUT_SEC, STDIO_READ_TIMEOUT_SEC

_httpx_module: Any | None
try:
    _httpx_module = importlib.import_module("httpx")
except Exception:  # pragma: no cover - optional dependency for hybrid send mode
    _httpx_module = None

_websockets_module: Any | None
try:
    _websockets_module = importlib.import_module("websockets")
except Exception:  # pragma: no cover - compatibility fallback for missing optional dependency
    _websockets_module = None

_httpx: Any | None = _httpx_module
_websockets: Any | None = _websockets_module

logger = logging.getLogger("ripperdoc.protocol.stdio.handler")

# Maximum number of resolved tool_use_ids to track for duplicate detection
_MAX_RESOLVED_TOOL_USE_IDS = 1000

_SDK_RECONNECT_BASE_DELAY_SEC = 1.0
_SDK_RECONNECT_MAX_DELAY_SEC = 30.0
_SDK_RECONNECT_BUDGET_SEC = 600.0
_SDK_PING_INTERVAL_SEC = 10.0
_SDK_PING_TIMEOUT_SEC = 5.0
_SDK_KEEPALIVE_INTERVAL_SEC = 300.0
_SDK_MESSAGE_BUFFER_MAX = 1000
_SDK_PERMANENT_CLOSE_CODES = {1002, 4001, 4003}
_SDK_RECONNECT_SLEEP_GAP_RESET_SEC = 60.0
_SDK_HYBRID_POST_RETRY_COUNT = 10
_SDK_HYBRID_POST_RETRY_BASE_SEC = 0.5
_SDK_HYBRID_POST_RETRY_MAX_SEC = 8.0


class _SDKWebSocketTransport:
    """Minimal async websocket transport with reconnect, keepalive, and replay buffer."""

    def __init__(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        auto_reconnect: bool = True,
        refresh_headers: Callable[[], dict[str, str] | None] | None = None,
    ) -> None:
        self.url = url
        self.headers = headers or {}
        self._refresh_headers = refresh_headers
        self.auto_reconnect = auto_reconnect
        self._state = "idle"
        self._ws: Any | None = None
        self._running = False
        self._loop_task: asyncio.Task[None] | None = None
        self._ping_task: asyncio.Task[None] | None = None
        self._keepalive_task: asyncio.Task[None] | None = None
        self._incoming_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._outgoing_queue: asyncio.Queue[dict[str, Any] | object | None] = asyncio.Queue()
        self._last_sent_id: str | None = None
        self._last_confirmed_id: str | None = None
        self._message_buffer = deque[dict[str, Any]](maxlen=_SDK_MESSAGE_BUFFER_MAX)
        self._stop_sentinel: object = object()
        self._last_reconnect_attempt_time: float | None = None
        self._reconnect_start_time: float | None = None
        self._reconnect_attempts = 0
        self._on_close_callback: Callable[[int | None], None] | None = None
        self._pong_received = True

    async def start(self) -> None:
        if self._running:
            return
        if _websockets is None:
            raise RuntimeError("websockets package is required for --sdk-url support.")
        self._running = True
        self._state = "idle"
        self._loop_task = asyncio.create_task(self._run_loop())

    async def close(self) -> None:
        self._running = False
        self._state = "closing"
        await self._stop_ping_loop()
        await self._stop_keepalive_loop()
        await self._close_socket()
        await self._outgoing_queue.put(self._stop_sentinel)

        if self._loop_task is not None:
            self._loop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._loop_task
            self._loop_task = None

        # Ensure any readers stop.
        await self._incoming_queue.put(None)
        self._state = "closed"

    async def write(self, message: dict[str, Any]) -> None:
        if not self._running:
            await self.start()

        tracked_id = self._extract_tracking_id(message)
        if tracked_id:
            self._message_buffer.append(message)
            self._last_sent_id = tracked_id

        await self._outgoing_queue.put(message)

    async def read_line(self) -> str | None:
        line = await self._incoming_queue.get()
        if line is None:
            return None
        return line

    def setOnClose(self, callback: Callable[[int | None], None] | None) -> None:
        self._on_close_callback = callback

    def _collect_connect_headers(self, *, include_last_request_id: bool = True) -> dict[str, str]:
        headers = dict(self.headers)
        if self._refresh_headers:
            try:
                refreshed = self._refresh_headers() or {}
                if isinstance(refreshed, dict):
                    headers.update(refreshed)
                    self.headers.update(refreshed)
            except Exception as exc:  # noqa: BLE001
                logger.debug("[stdio-sdk] Failed refreshing SDK headers", exc_info=exc)

        if include_last_request_id and self._last_sent_id:
            headers["X-Last-Request-Id"] = self._last_sent_id
            self.headers["X-Last-Request-Id"] = self._last_sent_id

        return headers

    def _refresh_headers_if_needed(self, *, force: bool = False) -> bool:
        if not self._refresh_headers:
            return False

        refreshed = self._refresh_headers() or {}
        if not isinstance(refreshed, dict):
            return False

        previous_auth_snapshot = (
            self.headers.get("Authorization"),
            self.headers.get("Cookie"),
            self.headers.get("X-Organization-Uuid"),
        )
        self.headers.update(refreshed)
        if force:
            return (
                self.headers.get("Authorization") is not None
                or self.headers.get("Cookie") is not None
            )
        if "Authorization" in refreshed or "Cookie" in refreshed:
            current_auth_snapshot = (
                self.headers.get("Authorization"),
                self.headers.get("Cookie"),
                self.headers.get("X-Organization-Uuid"),
            )
            return current_auth_snapshot != previous_auth_snapshot
        return False

    async def _run_loop(self) -> None:
        self._reconnect_attempts = 0
        self._reconnect_start_time = None
        try:
            while self._running:
                delay = await self._connect_and_run()
                if not self._running:
                    return
                if delay is None:
                    return
                await asyncio.sleep(delay)
        finally:
            await self._incoming_queue.put(None)

    async def _connect_and_run(self) -> float | None:
        if self._state not in {"idle", "reconnecting"}:
            logger.error(
                "[stdio-sdk] WebSocket connect rejected; current state is %s",
                self._state,
            )
            return None

        self._state = "reconnecting"
        logger.info("[stdio-sdk] Connecting to SDK WebSocket endpoint: %s", self.url)

        try:
            websockets_client = _websockets
            if websockets_client is None:
                raise RuntimeError("websockets package is required for --sdk-url support.")
            ws = await self._open_websocket(websockets_client)
        except Exception as exc:
            return await self._handle_connection_close(None, exc)

        self._ws = ws
        self._state = "connected"
        logger.info("[stdio-sdk] WebSocket transport connected")
        self._last_reconnect_attempt_time = None
        self._reconnect_attempts = 0
        self._reconnect_start_time = None
        self._pong_received = True

        await self._replay_buffer(self._last_confirmed_id)

        ping_task = asyncio.create_task(self._ping_loop(ws))
        self._ping_task = ping_task
        keepalive_task = asyncio.create_task(self._keepalive_loop(ws))
        self._keepalive_task = keepalive_task
        receiver_task = asyncio.create_task(self._receiver_loop(ws))
        sender_task = asyncio.create_task(self._sender_loop(ws))

        done, pending = await asyncio.wait(
            [ping_task, keepalive_task, receiver_task, sender_task],
            return_when=asyncio.FIRST_EXCEPTION,
        )

        for task in pending:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        self._ping_task = None
        self._keepalive_task = None

        if not self._running:
            await self._close_socket()
            return None

        exception: BaseException | None = None
        close_code = getattr(ws, "close_code", None)
        for task in done:
            if task.cancelled():
                continue
            task_exception = task.exception()
            if task_exception is not None:
                exception = task_exception
                close_code = getattr(task_exception, "code", close_code)
                break

        delay = await self._handle_connection_close(close_code, exception)
        await self._close_socket()
        return delay

    async def _sender_loop(self, ws: Any) -> None:
        while self._running:
            item = await self._outgoing_queue.get()
            if item is self._stop_sentinel:
                break
            if item is None:
                break
            if not isinstance(item, dict):
                continue
            await self._send_message(ws, cast(dict[str, Any], item))

    async def _receiver_loop(self, ws: Any) -> None:
        while self._running and self._state == "connected":
            payload = await ws.recv()
            if isinstance(payload, bytes):
                payload = payload.decode("utf-8", errors="replace")
            for raw_line in str(payload).splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                self._process_incoming_ack(line)
                await self._incoming_queue.put(line)

    async def _send_message(self, ws: Any, message: dict[str, Any]) -> None:
        data = json.dumps(message, ensure_ascii=False) + "\n"
        await ws.send(data)

    async def _send_frame(self, ws: Any, frame: str) -> None:
        await ws.send(frame)

    async def _ping_loop(self, ws: Any) -> None:
        self._pong_received = True
        while self._running:
            await asyncio.sleep(_SDK_PING_INTERVAL_SEC)
            if self._state != "connected" or ws is None:
                continue

            if not self._pong_received:
                logger.error("[stdio-sdk] No pong received, connection appears dead")
                raise RuntimeError("No pong received")

            self._pong_received = False
            try:
                ping_result = ws.ping()
                if hasattr(ping_result, "__await__"):
                    await asyncio.wait_for(ping_result, timeout=_SDK_PING_TIMEOUT_SEC)
                self._pong_received = True
            except asyncio.TimeoutError:
                logger.error("[stdio-sdk] WebSocket ping timeout")
                raise
            except OSError as exc:
                logger.debug("[stdio-sdk] WebSocket ping failed: %s", exc)
                raise

    async def _keepalive_loop(self, ws: Any) -> None:
        if parse_boolish(
            os.getenv("RIPPERDOC_REMOTE_CONTROL"),
            default=False,
        ):
            return
        while self._running:
            await asyncio.sleep(_SDK_KEEPALIVE_INTERVAL_SEC)
            if self._state != "connected" or ws is None:
                continue
            await self._send_frame(
                ws,
                json.dumps({"type": "keep_alive"}) + "\n",
            )

    async def _stop_ping_loop(self) -> None:
        if self._ping_task is None:
            return
        self._ping_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._ping_task
        self._ping_task = None

    async def _stop_keepalive_loop(self) -> None:
        if self._keepalive_task is None:
            return
        self._keepalive_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._keepalive_task
        self._keepalive_task = None

    async def _handle_connection_close(
        self,
        close_code: int | None,
        exception: BaseException | None,
    ) -> float | None:
        await self._stop_ping_loop()
        await self._stop_keepalive_loop()
        if self._state in {"closing", "closed"}:
            return None

        if exception is not None:
            logger.debug("[stdio-sdk] WebSocket session ended with error", exc_info=exception)

        if close_code is not None:
            logger.info("[stdio-sdk] Disconnected from %s (code %s)", self.url, close_code)

        headers_refreshed = False
        if close_code == 4003 and self._refresh_headers:
            headers_refreshed = self._refresh_headers_if_needed(force=True)
            if headers_refreshed:
                logger.debug("[stdio-sdk] 4003 received but headers refreshed; scheduling reconnect")

        if close_code in _SDK_PERMANENT_CLOSE_CODES and not headers_refreshed:
            logger.error("[stdio-sdk] Permanent close code %s, not reconnecting", close_code)
            self.auto_reconnect = False
            self._state = "closed"
            if self._on_close_callback is not None:
                self._on_close_callback(close_code)
            return None

        if not self.auto_reconnect:
            self._state = "closed"
            if self._on_close_callback is not None:
                self._on_close_callback(close_code)
            return None

        now = time.monotonic()
        if self._reconnect_start_time is None:
            self._reconnect_start_time = now

        if (
            self._last_reconnect_attempt_time is not None
            and now - self._last_reconnect_attempt_time > _SDK_RECONNECT_SLEEP_GAP_RESET_SEC
        ):
            logger.info("[stdio-sdk] System sleep gap detected; reset reconnect budget")
            self._reconnect_start_time = now
            self._reconnect_attempts = 0

        self._last_reconnect_attempt_time = now
        elapsed = now - self._reconnect_start_time
        if elapsed >= _SDK_RECONNECT_BUDGET_SEC:
            logger.error("[stdio-sdk] Reconnect budget exhausted; stopping transport")
            self._state = "closed"
            if self._on_close_callback is not None:
                self._on_close_callback(close_code)
            return None

        if not headers_refreshed and self._refresh_headers:
            self._refresh_headers_if_needed()

        self._state = "reconnecting"
        self._reconnect_attempts += 1
        delay = self._reconnect_delay(self._reconnect_attempts)
        logger.warning(
            "[stdio-sdk] Reconnecting in %.2fs (attempt %d, %.2fs elapsed)",
            delay,
            self._reconnect_attempts,
            elapsed,
        )
        return delay

    async def _open_websocket(self, websockets_client: Any) -> Any:
        """Open websocket with headers while supporting multiple websockets versions."""
        connect_headers = self._collect_connect_headers()
        connect_kwargs: dict[str, Any] = {
            "ping_interval": None,
            "ping_timeout": None,
        }
        try:
            # websockets >= 14 uses `additional_headers`
            return await websockets_client.connect(
                self.url,
                additional_headers=connect_headers,
                **connect_kwargs,
            )
        except TypeError as exc:
            # websockets <= 13 uses `extra_headers`
            if "additional_headers" not in str(exc):
                raise
            return await websockets_client.connect(
                self.url,
                extra_headers=connect_headers,
                **connect_kwargs,
            )

    async def _replay_buffer(self, last_request_id: str | None = None) -> None:
        if not self._message_buffer:
            return

        messages = list(self._message_buffer)
        if last_request_id:
            start_index = 0
            for index, message in enumerate(messages):
                if self._extract_tracking_id(message) == last_request_id:
                    start_index = index + 1
                    break
            if start_index > 0:
                messages = messages[start_index:]
                if not messages:
                    logger.debug(
                        "[stdio-sdk] All buffered messages confirmed by %s; clearing buffer",
                        last_request_id,
                    )
                    self._message_buffer.clear()
                    self._last_sent_id = None
                    return

        self._message_buffer = deque(messages, maxlen=_SDK_MESSAGE_BUFFER_MAX)
        if self._ws is None:
            return
        for message in self._message_buffer:
            await self._send_message(self._ws, message)

    async def _close_socket(self) -> None:
        if self._ws is None:
            return
        try:
            await self._ws.close()
        finally:
            self._ws = None
            self._state = "closed"

    def _extract_tracking_id(self, message: dict[str, Any]) -> str | None:
        for key in ("uuid", "request_id", "id"):
            value = message.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _process_incoming_ack(self, raw_line: str) -> None:
        try:
            message = json.loads(raw_line)
        except json.JSONDecodeError:
            return
        if not isinstance(message, dict):
            return

        request_id = None
        if message.get("type") == "control_response":
            response = message.get("response")
            if isinstance(response, dict):
                request_id = response.get("request_id")
        elif message.get("jsonrpc") == "2.0":
            request_id = message.get("id")

        if not isinstance(request_id, str):
            return
        self._acknowledge(request_id)

    def _acknowledge(self, request_id: str) -> None:
        if not self._message_buffer:
            return

        messages = list(self._message_buffer)
        confirmed_index = None
        for index, message in enumerate(messages):
            if self._extract_tracking_id(message) == request_id:
                confirmed_index = index
                break
        if confirmed_index is None:
            return

        remaining_messages = messages[confirmed_index + 1 :]
        self._message_buffer = deque(remaining_messages, maxlen=_SDK_MESSAGE_BUFFER_MAX)

        self._last_confirmed_id = request_id
        if not remaining_messages:
            self._last_sent_id = None

    def _reconnect_delay(self, attempt: int) -> float:
        delay: float = float(_SDK_RECONNECT_BASE_DELAY_SEC) * (2 ** (attempt - 1))
        jitter = delay * random.uniform(-0.25, 0.25)
        max_delay: float = float(_SDK_RECONNECT_MAX_DELAY_SEC)
        delay = delay + jitter
        if delay > max_delay:
            delay = max_delay
        if delay < 0.0:
            return 0.0
        return delay


class _SDKHybridWebSocketTransport(_SDKWebSocketTransport):
    """HTTP relay variant for session ingress while retaining websocket read path."""

    def __init__(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        auto_reconnect: bool = True,
        refresh_headers: Callable[[], dict[str, str] | None] | None = None,
    ) -> None:
        super().__init__(
            url,
            headers=headers,
            auto_reconnect=auto_reconnect,
            refresh_headers=refresh_headers,
        )
        self._post_url = self._resolve_post_url(url)

    @staticmethod
    def _resolve_post_url(url: str) -> str:
        parsed = urlparse(url)
        scheme = parsed.scheme
        if scheme == "wss":
            scheme = "https"
        elif scheme == "ws":
            scheme = "http"
        path = parsed.path
        if not path.startswith("/"):
            path = f"/{path}"

        if "/session_ingress/ws/" in path:
            path = path.replace("/session_ingress/ws/", "/session_ingress/session/", 1)
        elif "/session_ingress/session/" not in path:
            path = path.replace("/ws/", "/session/")

        if not path.endswith("/events"):
            path = path.rstrip("/") + "/events"
        return urlunparse(
            (scheme, parsed.netloc, path, parsed.params, parsed.query, parsed.fragment),
        )

    async def _send_message(self, ws: Any, message: dict[str, Any]) -> None:
        event_type = message.get("type", "message")
        await self._post_events([message], event_type=event_type)

    async def _post_events(self, events: list[dict[str, Any]], event_type: str) -> None:
        if _httpx is None:
            raise RuntimeError("httpx package is required for SDK hybrid transport mode.")

        request_headers = self._collect_connect_headers(include_last_request_id=False)
        if not request_headers.get("Authorization") and not request_headers.get("Cookie"):
            logger.debug("[stdio-sdk] Hybrid transport skipped: no session auth header")
            return

        request_headers.setdefault("Content-Type", "application/json")

        for attempt in range(1, _SDK_HYBRID_POST_RETRY_COUNT + 1):
            try:
                async with _httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        self._post_url,
                        json={"events": events},
                        headers=request_headers,
                    )
                if response.status_code in {200, 201}:
                    return
                if 400 <= response.status_code < 500 and response.status_code != 429:
                    logger.debug(
                        "[stdio-sdk] Hybrid POST returned %s for %s (client error, stop retrying)",
                        response.status_code,
                        event_type,
                    )
                    return
                logger.debug(
                    "[stdio-sdk] Hybrid POST returned %s for %s (attempt %d/%d)",
                    response.status_code,
                    event_type,
                    attempt,
                    _SDK_HYBRID_POST_RETRY_COUNT,
                )
            except (_httpx.RequestError, OSError) as exc:
                logger.debug(
                    "[stdio-sdk] Hybrid POST for %s failed (attempt %d/%d): %s",
                    event_type,
                    attempt,
                    _SDK_HYBRID_POST_RETRY_COUNT,
                    exc,
                )

            if attempt == _SDK_HYBRID_POST_RETRY_COUNT:
                logger.debug(
                    "[stdio-sdk] Hybrid POST failed after %d attempts",
                    _SDK_HYBRID_POST_RETRY_COUNT,
                )
                return

            delay = min(
                _SDK_HYBRID_POST_RETRY_BASE_SEC * (2 ** (attempt - 1)),
                _SDK_HYBRID_POST_RETRY_MAX_SEC,
            )
            await asyncio.sleep(delay)


class StdioIOMixin:
    _resolved_tool_use_ids: set[str]
    _sdk_transport: _SDKWebSocketTransport | None
    _mcp_server_overrides: dict[str, Any] | None

    def _is_sdk_transport_enabled(self) -> bool:
        return bool(getattr(self, "_sdk_url", None))

    @staticmethod
    def _readable_stripped_env(*env_keys: str) -> str | None:
        for env_key in env_keys:
            value = os.getenv(env_key, "")
            if value:
                value = value.strip()
                if value:
                    return value
        return None

    def _build_sdk_transport_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}

        session_access_token = self._readable_stripped_env(
            "RIPPERDOC_SESSION_ACCESS_TOKEN",
            "RIPPERDOC_REMOTE_CONTROL_ACCESS_TOKEN",
            "RIPPERDOC_AUTH_TOKEN",
            "RIPPERDOC_API_KEY",
        )
        if session_access_token:
            if session_access_token.startswith("sk-ant-sid"):
                headers["Cookie"] = f"sessionKey={session_access_token}"
                if organization_uuid := self._readable_stripped_env(
                    "RIPPERDOC_ORGANIZATION_UUID",
                ):
                    headers["X-Organization-Uuid"] = organization_uuid
            else:
                headers["Authorization"] = f"Bearer {session_access_token}"

        if runner_version := self._readable_stripped_env(
            "RIPPERDOC_ENVIRONMENT_RUNNER_VERSION",
        ):
            headers["x-environment-runner-version"] = runner_version

        if session_id := getattr(self, "_session_id", None):
            headers["x-ripperdoc-session-id"] = str(session_id)

        return headers

    async def _start_sdk_transport(self) -> None:
        if not self._is_sdk_transport_enabled():
            return

        if getattr(self, "_sdk_transport", None) is not None:
            return

        transport_cls = _SDKWebSocketTransport
        if parse_boolish(
            os.getenv("RIPPERDOC_POST_FOR_SESSION_INGRESS_V2"),
            default=False,
        ):
            transport_cls = _SDKHybridWebSocketTransport

        transport = transport_cls(
            str(getattr(self, "_sdk_url")),
            headers=self._build_sdk_transport_headers(),
            auto_reconnect=True,
            refresh_headers=self._build_sdk_transport_headers,
        )
        self._sdk_transport = transport
        await transport.start()

    async def _stop_sdk_transport(self) -> None:
        transport = getattr(self, "_sdk_transport", None)
        if transport is None:
            return

        self._sdk_transport = None
        try:
            await transport.close()
        except Exception as exc:  # noqa: BLE001
            logger.debug("[stdio] Failed to stop SDK transport: %s", exc)

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

        if self._is_sdk_transport_enabled():
            await self._start_sdk_transport()
            transport = self._sdk_transport
            if transport is None:
                raise RuntimeError("SDK transport is not available")
            await transport.write(message)
            return

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

        if self._is_sdk_transport_enabled():
            await self._start_sdk_transport()
            transport = self._sdk_transport
            if transport is None:
                return None
            return await transport.read_line()

        while True:
            try:
                if STDIO_READ_TIMEOUT_SEC <= 0:
                    line = cast(
                        str,
                        await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline),
                    )
                else:
                    line = cast(
                        str,
                        await asyncio.wait_for(
                            asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline),
                            timeout=STDIO_READ_TIMEOUT_SEC,
                        ),
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
