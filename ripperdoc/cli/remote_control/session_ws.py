# mypy: disable-error-code=misc

"""Session-scoped websocket manager for remote control / REPL bridge."""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, cast

from ripperdoc.utils.log import get_logger

from .constants import (
    SESSIONS_WS_MAX_RECONNECTS,
    SESSIONS_WS_PERMANENT_CLOSE_CODES,
    SESSIONS_WS_PING_INTERVAL_SEC,
    SESSIONS_WS_RECONNECT_DELAY_SEC,
)

logger = get_logger()

if TYPE_CHECKING:
    from websockets.sync.client import ClientConnection as WebSocketConnection
else:
    WebSocketConnection = Any

WebSocketClosed: type[Exception]
websocket_connect: Any | None

try:
    from websockets.exceptions import ConnectionClosed as _WebSocketClosed
    from websockets.sync.client import connect as _websocket_connect

    WebSocketClosed = _WebSocketClosed
    websocket_connect = _websocket_connect
except Exception:  # pragma: no cover - optional runtime dependency
    WebSocketClosed = Exception
    websocket_connect = None


def _is_valid_message_type(message: Any) -> bool:
    return isinstance(message, dict) and isinstance(message.get("type"), str)


@dataclass
class SessionsWebSocketCallbacks:
    on_message: Callable[[dict[str, Any]], None] | None = None
    on_connected: Callable[[], None] | None = None
    on_close: Callable[[], None] | None = None
    on_error: Callable[[Exception], None] | None = None


class SessionsWebSocketManager:
    """Manage websocket lifecycle for a remote session stream."""

    def __init__(
        self,
        session_id: str,
        org_uuid: str,
        access_token: str,
        callbacks: SessionsWebSocketCallbacks,
        *,
        base_api_url: str,
    ) -> None:
        self.session_id = session_id
        self.org_uuid = org_uuid
        self.access_token = access_token
        self.callbacks = callbacks
        self.base_api_url = base_api_url.rstrip("/")

        self._state = "closed"
        self._reconnect_attempts = 0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._ws: WebSocketConnection | None = None
        self._ws_lock = threading.Lock()

    def _build_ws_url(self) -> str:
        base = self.base_api_url.replace("https://", "wss://").replace("http://", "ws://")
        return (
            f"{base}/v1/sessions/ws/{self.session_id}/subscribe"
            f"?organization_uuid={self.org_uuid}"
        )

    def connect(self) -> None:
        if self._state in {"connecting", "connected"}:
            return
        self._stop_event.clear()
        self._state = "connecting"
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="ripperdoc-session-ws")
        self._thread.start()

    def _run_loop(self) -> None:
        if websocket_connect is None:
            err = RuntimeError("websockets.sync.client is unavailable")
            logger.error("[SessionsWebSocket] %s", err)
            self.callbacks.on_error(err) if self.callbacks.on_error else None
            self._state = "closed"
            return

        while not self._stop_event.is_set():
            url = self._build_ws_url()
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "anthropic-version": "2023-06-01",
            }
            try:
                logger.info("[SessionsWebSocket] Connecting to %s", url)
                with websocket_connect(
                    url,
                    additional_headers=headers,
                    ping_interval=SESSIONS_WS_PING_INTERVAL_SEC,
                ) as ws:
                    with self._ws_lock:
                        self._ws = cast("WebSocketConnection", ws)
                    self._state = "connected"
                    self._reconnect_attempts = 0
                    if self.callbacks.on_connected:
                        self.callbacks.on_connected()

                    for raw in ws:
                        if self._stop_event.is_set():
                            break
                        if isinstance(raw, str):
                            self._handle_message(raw)
            except WebSocketClosed as exc:
                close_code = getattr(exc, "code", None)
                logger.warning("[SessionsWebSocket] Closed code=%s", close_code)
                if isinstance(close_code, int) and close_code in SESSIONS_WS_PERMANENT_CLOSE_CODES:
                    break
                if not self._maybe_schedule_reconnect():
                    break
            except Exception as exc:  # noqa: BLE001
                logger.error("[SessionsWebSocket] Error: %s", exc)
                if self.callbacks.on_error:
                    self.callbacks.on_error(exc if isinstance(exc, Exception) else Exception(str(exc)))
                if not self._maybe_schedule_reconnect():
                    break
            finally:
                with self._ws_lock:
                    self._ws = None
                self._state = "closed"

        if self.callbacks.on_close:
            self.callbacks.on_close()

    def _maybe_schedule_reconnect(self) -> bool:
        if self._stop_event.is_set():
            return False
        if self._reconnect_attempts >= SESSIONS_WS_MAX_RECONNECTS:
            return False
        self._reconnect_attempts += 1
        time.sleep(SESSIONS_WS_RECONNECT_DELAY_SEC)
        self._state = "connecting"
        return True

    def _handle_message(self, raw_message: str) -> None:
        try:
            payload = json.loads(raw_message)
        except Exception as exc:  # noqa: BLE001
            logger.debug("[SessionsWebSocket] Failed to parse message: %s", exc)
            return
        if not _is_valid_message_type(payload):
            return
        if self.callbacks.on_message:
            self.callbacks.on_message(payload)

    def _send(self, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False)
        with self._ws_lock:
            ws = self._ws
            if ws is None or self._state != "connected":
                raise RuntimeError("websocket not connected")
            ws.send(encoded)

    def send_control_response(self, payload: dict[str, Any]) -> None:
        self._send(payload)

    def send_control_request(
        self,
        payload: dict[str, Any],
        *,
        request_id: str | None = None,
    ) -> str:
        resolved_request_id = request_id or str(uuid.uuid4())
        envelope = {
            "type": "control_request",
            "request_id": resolved_request_id,
            "request": payload,
        }
        self._send(envelope)
        return resolved_request_id

    def is_connected(self) -> bool:
        return self._state == "connected"

    def close(self) -> None:
        self._stop_event.set()
        with self._ws_lock:
            ws = self._ws
            self._ws = None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        self._state = "closed"

    def reconnect(self) -> None:
        self.close()
        self._reconnect_attempts = 0
        self.connect()
