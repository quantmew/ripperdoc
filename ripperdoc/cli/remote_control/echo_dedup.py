"""Echo deduplication and ingress message handling for bridge transport.

Provides BoundedUUIDSet (FIFO-bounded ring buffer for UUID dedup) and
handle_ingress_message (parse + route inbound WebSocket messages with
echo/re-delivery filtering).
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Set


class BoundedUUIDSet:
    """FIFO-bounded set backed by a circular buffer.

    Evicts the oldest entry when capacity is reached, keeping memory
    usage constant at O(capacity). Messages are added in chronological
    order, so evicted entries are always the oldest.
    """

    def __init__(self, capacity: int = 2000) -> None:
        self._capacity = max(1, capacity)
        self._ring: List[Optional[str]] = [None] * self._capacity
        self._set: Set[str] = set()
        self._write_idx = 0

    def add(self, uuid: str) -> None:
        if uuid in self._set:
            return
        evicted = self._ring[self._write_idx]
        if evicted is not None:
            self._set.discard(evicted)
        self._ring[self._write_idx] = uuid
        self._set.add(uuid)
        self._write_idx = (self._write_idx + 1) % self._capacity

    def has(self, uuid: str) -> bool:
        return uuid in self._set

    def clear(self) -> None:
        self._set.clear()
        self._ring = [None] * self._capacity
        self._write_idx = 0


# --- Type guards ---


def _is_dict_with(value: Any, *keys: str) -> bool:
    return (
        isinstance(value, dict)
        and all(k in value for k in keys)
    )


def is_sdk_control_response(value: Any) -> bool:
    return (
        _is_dict_with(value, "type", "response")
        and value["type"] == "control_response"
    )


def is_sdk_control_request(value: Any) -> bool:
    return (
        _is_dict_with(value, "type", "request_id", "request")
        and value["type"] == "control_request"
    )


def is_sdk_message(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and "type" in value
        and isinstance(value["type"], str)
    )


# --- Ingress routing ---


def handle_ingress_message(
    data: str,
    recent_posted_uuids: BoundedUUIDSet,
    recent_inbound_uuids: BoundedUUIDSet,
    on_inbound_message: Optional[Callable[[Dict[str, Any]], None]] = None,
    on_permission_response: Optional[Callable[[Dict[str, Any]], None]] = None,
    on_control_request: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> None:
    """Parse an ingress WebSocket message and route it to the appropriate handler.

    Ignores messages whose UUID is in recent_posted_uuids (echoes of what we sent)
    or in recent_inbound_uuids (re-deliveries we've already forwarded).
    """
    try:
        parsed: Any = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return

    if not isinstance(parsed, dict):
        return

    # control_response is not an SDKMessage -- check before the type guard
    if is_sdk_control_response(parsed):
        if on_permission_response is not None:
            on_permission_response(parsed)
        return

    # control_request from the server (initialize, set_model, can_use_tool).
    if is_sdk_control_request(parsed):
        if on_control_request is not None:
            on_control_request(parsed)
        return

    if not is_sdk_message(parsed):
        return

    # Check for UUID to detect echoes of our own messages
    uuid_val = parsed.get("uuid")
    if isinstance(uuid_val, str) and uuid_val:
        if recent_posted_uuids.has(uuid_val):
            return

        # Defensive dedup: drop inbound prompts we've already forwarded
        if recent_inbound_uuids.has(uuid_val):
            return

    msg_type = str(parsed.get("type") or "")

    if msg_type == "user":
        if isinstance(uuid_val, str) and uuid_val:
            recent_inbound_uuids.add(uuid_val)
        if on_inbound_message is not None:
            on_inbound_message(parsed)
