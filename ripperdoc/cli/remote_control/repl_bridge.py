"""REPL/session bridge manager built on top of the session websocket manager."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from ripperdoc.utils.log import get_logger

from .session_ws import SessionsWebSocketCallbacks, SessionsWebSocketManager

logger = get_logger()


@dataclass(frozen=True)
class RemoteSessionConfig:
    """Configuration required to manage one remote session bridge."""

    session_id: str
    access_token: str
    org_uuid: str
    base_api_url: str
    has_initial_prompt: bool = False


@dataclass
class RemoteSessionCallbacks:
    """Lifecycle and message callbacks for remote REPL bridge usage."""

    on_message: Callable[[dict[str, Any]], None] | None = None
    on_connected: Callable[[], None] | None = None
    on_disconnected: Callable[[], None] | None = None
    on_error: Callable[[Exception], None] | None = None
    on_permission_request: Callable[[dict[str, Any], str], None] | None = None


@dataclass
class _PendingRequest:
    event: threading.Event
    response: dict[str, Any] | None = None


class RemoteSessionBridgeManager:
    """High-level manager that handles control messages and permission requests."""

    def __init__(
        self,
        config: RemoteSessionConfig,
        callbacks: RemoteSessionCallbacks,
        *,
        on_control_response_fallback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.config = config
        self.callbacks = callbacks
        self._on_control_response_fallback = on_control_response_fallback
        self.websocket: SessionsWebSocketManager | None = None
        self.pending_permission_requests: dict[str, dict[str, Any]] = {}
        self._pending_outbound: dict[str, _PendingRequest] = {}
        self._pending_lock = threading.Lock()

    def connect(self) -> None:
        logger.info("[RemoteSessionManager] Connecting to session %s", self.config.session_id)
        ws_callbacks = SessionsWebSocketCallbacks(
            on_message=self._handle_message,
            on_connected=self._on_connected,
            on_close=self._on_close,
            on_error=self._on_error,
        )
        self.websocket = SessionsWebSocketManager(
            self.config.session_id,
            self.config.org_uuid,
            self.config.access_token,
            ws_callbacks,
            base_api_url=self.config.base_api_url,
        )
        self.websocket.connect()

    def _on_connected(self) -> None:
        logger.info("[RemoteSessionManager] Connected")
        if self.callbacks.on_connected is not None:
            self.callbacks.on_connected()

    def _on_close(self) -> None:
        logger.info("[RemoteSessionManager] Disconnected")
        if self.callbacks.on_disconnected is not None:
            self.callbacks.on_disconnected()

    def _on_error(self, error: Exception) -> None:
        logger.error("[RemoteSessionManager] Error: %s", error)
        if self.callbacks.on_error is not None:
            self.callbacks.on_error(error)

    def _handle_message(self, message: dict[str, Any]) -> None:
        message_type = str(message.get("type") or "")
        if message_type == "control_request":
            self._handle_control_request(message)
            return
        if message_type == "control_response":
            self._handle_control_response(message)
            return
        if self.callbacks.on_message is not None:
            self.callbacks.on_message(message)

    def _handle_control_response(self, message: dict[str, Any]) -> None:
        response_raw = message.get("response")
        response = response_raw if isinstance(response_raw, dict) else {}
        request_id = str(response.get("request_id") or message.get("request_id") or "").strip()
        if not request_id:
            logger.debug("[RemoteSessionManager] Ignoring control response without request_id")
            return

        with self._pending_lock:
            pending = self._pending_outbound.get(request_id)
            if pending is None:
                logger.debug(
                    "[RemoteSessionManager] control_response for unknown request_id=%s",
                    request_id,
                )
                return
            pending.response = response
            pending.event.set()

    def _handle_control_request(self, message: dict[str, Any]) -> None:
        request_id = str(message.get("request_id") or "").strip()
        request_raw = message.get("request")
        request = request_raw if isinstance(request_raw, dict) else {}
        subtype = str(request.get("subtype") or "").strip()

        if request_id and subtype == "can_use_tool":
            self.pending_permission_requests[request_id] = request
            logger.info(
                "[RemoteSessionManager] Permission request for tool: %s",
                request.get("tool_name") or "<unknown>",
            )
            if self.callbacks.on_permission_request is not None:
                self.callbacks.on_permission_request(request, request_id)
            return

        logger.info(
            "[RemoteSessionManager] Unsupported control request subtype: %s",
            subtype or "<empty>",
        )
        if request_id:
            self.send_control_response(
                {
                    "type": "control_response",
                    "response": {
                        "subtype": "error",
                        "request_id": request_id,
                        "error": (
                            f"Unsupported control request subtype: {subtype or '<empty>'}"
                        ),
                    },
                }
            )

    def send_control_response(self, payload: dict[str, Any]) -> None:
        if self.websocket is None:
            logger.error("[RemoteSessionManager] No websocket available for response")
            self._emit_control_response_fallback(payload, reason="no websocket")
            return
        try:
            self.websocket.send_control_response(payload)
        except Exception as exc:  # noqa: BLE001
            logger.debug("[RemoteSessionManager] send_control_response failed: %s", exc)
            self._emit_control_response_fallback(payload, reason=str(exc))

    def _emit_control_response_fallback(self, payload: dict[str, Any], *, reason: str) -> None:
        if self._on_control_response_fallback is None:
            return
        try:
            self._on_control_response_fallback(payload)
            logger.debug(
                "[RemoteSessionManager] control_response fallback dispatched session=%s reason=%s",
                self.config.session_id,
                reason,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("[RemoteSessionManager] control_response fallback failed: %s", exc)

    def forward_control_request(
        self,
        request: dict[str, Any],
        *,
        timeout_sec: float = 60.0,
    ) -> dict[str, Any] | None:
        if self.websocket is None:
            return None

        request_id = str(uuid.uuid4())
        pending = _PendingRequest(event=threading.Event())
        with self._pending_lock:
            self._pending_outbound[request_id] = pending

        try:
            self.websocket.send_control_request(request, request_id=request_id)
        except Exception:
            with self._pending_lock:
                self._pending_outbound.pop(request_id, None)
            raise

        completed = pending.event.wait(timeout_sec)
        with self._pending_lock:
            resolved = self._pending_outbound.pop(request_id, pending)
        if not completed:
            logger.warning(
                "[RemoteSessionManager] Timed out waiting for control response request_id=%s",
                request_id,
            )
            return None
        return resolved.response

    def respond_to_permission_request(
        self,
        request_id: str,
        response: dict[str, Any],
    ) -> None:
        if request_id not in self.pending_permission_requests:
            logger.error(
                "[RemoteSessionManager] No pending permission request with ID: %s",
                request_id,
            )
            return

        self.pending_permission_requests.pop(request_id, None)
        behavior = str(response.get("behavior") or "").strip().lower()
        response_payload: dict[str, Any]
        if behavior == "allow":
            response_payload = {
                "behavior": "allow",
                "updatedInput": response.get("updatedInput"),
            }
        else:
            response_payload = {
                "behavior": "deny",
                "message": str(response.get("message") or "Request denied"),
            }

        self.send_control_response(
            {
                "type": "control_response",
                "response": {
                    "subtype": "success",
                    "request_id": request_id,
                    "response": response_payload,
                },
            }
        )

    def is_connected(self) -> bool:
        return self.websocket.is_connected() if self.websocket is not None else False

    def cancel_session(self) -> None:
        if self.websocket is None:
            return
        self.websocket.send_control_request({"subtype": "interrupt"})

    def get_session_id(self) -> str:
        return self.config.session_id

    def disconnect(self) -> None:
        if self.websocket is not None:
            self.websocket.close()
            self.websocket = None
        self.pending_permission_requests.clear()
        with self._pending_lock:
            for pending in self._pending_outbound.values():
                pending.event.set()
            self._pending_outbound.clear()

    def reconnect(self) -> None:
        if self.websocket is not None:
            self.websocket.reconnect()
