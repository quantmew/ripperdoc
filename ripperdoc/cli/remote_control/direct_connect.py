"""Direct-connect session manager for connecting to a remote Claude Code server.

Provides:
- DirectConnectSessionManager: WebSocket-based session manager
- create_direct_connect_session: Create a session on a direct-connect server
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from ripperdoc.utils.log import get_logger

logger = get_logger()

try:
    import websockets.sync.client as _ws_client
    from websockets.exceptions import ConnectionClosed as _WSConnectionClosed

    _HAS_WEBSOCKETS = True
except Exception:  # pragma: no cover
    _HAS_WEBSOCKETS = False


@dataclass
class DirectConnectConfig:
    """Configuration for a direct-connect session."""

    server_url: str
    session_id: str
    ws_url: str
    auth_token: str | None = None


@dataclass
class RemotePermissionResponse:
    """Permission response for direct-connect sessions."""

    behavior: str  # "allow" or "deny"
    updated_input: dict[str, Any] | None = None
    message: str | None = None


@dataclass
class DirectConnectCallbacks:
    """Callbacks for direct-connect session lifecycle."""

    on_message: Callable[[dict[str, Any]], None]
    on_permission_request: Callable[[dict[str, Any], str], None]
    on_connected: Callable[[], None] | None = None
    on_disconnected: Callable[[], None] | None = None
    on_error: Callable[[Exception], None] | None = None


def _is_stdout_message(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and "type" in value
        and isinstance(value["type"], str)
    )


class DirectConnectSessionManager:
    """WebSocket-based direct-connect session manager."""

    def __init__(
        self,
        config: DirectConnectConfig,
        callbacks: DirectConnectCallbacks,
    ) -> None:
        self._config = config
        self._callbacks = callbacks
        self._ws: Any = None

    def connect(self) -> None:
        """Connect to the direct-connect server via WebSocket."""
        if not _HAS_WEBSOCKETS:
            err = RuntimeError("websockets package is required for direct-connect")
            if self._callbacks.on_error:
                self._callbacks.on_error(err)
            return

        headers: dict[str, str] = {}
        if self._config.auth_token:
            headers["authorization"] = f"Bearer {self._config.auth_token}"

        try:
            self._ws = _ws_client.connect(
                self._config.ws_url,
                additional_headers=headers,
            )
            if self._callbacks.on_connected:
                self._callbacks.on_connected()

            # Read messages in a loop
            for raw in self._ws:
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="replace")
                if not isinstance(raw, str):
                    continue

                for line in raw.split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        parsed = json.loads(line)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue

                    if not _is_stdout_message(parsed):
                        continue

                    msg_type = str(parsed.get("type") or "")

                    if msg_type == "control_request":
                        request = parsed.get("request")
                        if isinstance(request, dict) and request.get("subtype") == "can_use_tool":
                            request_id = str(parsed.get("request_id") or "")
                            self._callbacks.on_permission_request(request, request_id)
                        else:
                            # Send error for unsupported subtypes
                            request_id = str(parsed.get("request_id") or "")
                            self._send_error_response(request_id, f"Unsupported control request subtype")
                        continue

                    # Forward non-control messages
                    if msg_type not in (
                        "control_response",
                        "keep_alive",
                        "control_cancel_request",
                    ):
                        self._callbacks.on_message(parsed)

        except _WSConnectionClosed:
            pass
        except Exception as exc:  # noqa: BLE001
            if self._callbacks.on_error:
                self._callbacks.on_error(exc if isinstance(exc, Exception) else Exception(str(exc)))
        finally:
            if self._callbacks.on_disconnected:
                self._callbacks.on_disconnected()

    def send_message(self, content: dict[str, Any]) -> bool:
        """Send a user message to the direct-connect session."""
        if self._ws is None:
            return False
        try:
            message = json.dumps({
                "type": "user",
                "message": {"role": "user", "content": content},
                "parent_tool_use_id": None,
                "session_id": "",
            })
            self._ws.send(message)
            return True
        except Exception:  # noqa: BLE001
            return False

    def respond_to_permission_request(
        self,
        request_id: str,
        result: RemotePermissionResponse,
    ) -> None:
        """Respond to a permission request from the server."""
        if self._ws is None:
            return
        response_payload: dict[str, Any] = {"behavior": result.behavior}
        if result.behavior == "allow":
            if result.updated_input:
                response_payload["updatedInput"] = result.updated_input
        else:
            response_payload["message"] = result.message or "Request denied"

        response = json.dumps({
            "type": "control_response",
            "response": {
                "subtype": "success",
                "request_id": request_id,
                "response": response_payload,
            },
        })
        try:
            self._ws.send(response)
        except Exception:  # noqa: BLE001
            pass

    def send_interrupt(self) -> None:
        """Send an interrupt signal to cancel the current request."""
        if self._ws is None:
            return
        request = json.dumps({
            "type": "control_request",
            "request_id": str(uuid.uuid4()),
            "request": {"subtype": "interrupt"},
        })
        try:
            self._ws.send(request)
        except Exception:  # noqa: BLE001
            pass

    def _send_error_response(self, request_id: str, error: str) -> None:
        if self._ws is None:
            return
        response = json.dumps({
            "type": "control_response",
            "response": {"subtype": "error", "request_id": request_id, "error": error},
        })
        try:
            self._ws.send(response)
        except Exception:  # noqa: BLE001
            pass

    def disconnect(self) -> None:
        """Disconnect from the server."""
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None

    def is_connected(self) -> bool:
        """Check if connected to the server."""
        return self._ws is not None


class DirectConnectError(Exception):
    """Error thrown by create_direct_connect_session."""


def create_direct_connect_session(
    server_url: str,
    *,
    auth_token: str | None = None,
    cwd: str | None = None,
    dangerously_skip_permissions: bool = False,
) -> tuple[DirectConnectConfig, str | None]:
    """Create a session on a direct-connect server.

    Posts to {server_url}/sessions, validates the response, and returns
    a DirectConnectConfig ready for use.

    Returns (config, work_dir).
    Raises DirectConnectError on failure.
    """
    import urllib.request
    import urllib.error

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    payload: dict[str, Any] = {"cwd": cwd or "."}
    if dangerously_skip_permissions:
        payload["dangerously_skip_permissions"] = True

    url = f"{server_url.rstrip('/')}/sessions"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers=headers,
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise DirectConnectError(f"Failed to create session: {exc.code} {exc.reason}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise DirectConnectError(f"Failed to connect to server at {server_url}: {exc}") from exc

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise DirectConnectError(f"Invalid session response: {exc}") from exc

    if not isinstance(data, dict):
        raise DirectConnectError("Invalid session response: expected JSON object")

    session_id = data.get("session_id")
    ws_url = data.get("ws_url")
    if not isinstance(session_id, str) or not session_id.strip():
        raise DirectConnectError("Invalid session response: missing session_id")
    if not isinstance(ws_url, str) or not ws_url.strip():
        raise DirectConnectError("Invalid session response: missing ws_url")

    work_dir = data.get("work_dir")
    if not isinstance(work_dir, str):
        work_dir = None

    return (
        DirectConnectConfig(
            server_url=server_url,
            session_id=session_id.strip(),
            ws_url=ws_url.strip(),
            auth_token=auth_token,
        ),
        work_dir,
    )
