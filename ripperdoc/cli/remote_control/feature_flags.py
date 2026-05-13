"""Feature flag configuration for bridge features.

Reads flags from environment variables with sensible defaults.
Allows enabling/disabling bridge features without code changes.
"""

from __future__ import annotations

import os
from typing import Any

from ripperdoc.utils.coerce import parse_boolish
from ripperdoc.utils.log import get_logger

logger = get_logger()

# Feature flag env var names
_V2_ENV = "RIPPERDOC_BRIDGE_REPL_V2"
_HEARTBEAT_ENV = "RIPPERDOC_BRIDGE_HEARTBEAT"
_MULTI_SESSION_ENV = "RIPPERDOC_BRIDGE_MULTI_SESSION"
_TRUSTED_DEVICE_ENV = "RIPPERDOC_BRIDGE_TRUSTED_DEVICE"
_VIEWER_MODE_ENV = "RIPPERDOC_BRIDGE_VIEWER_MODE"
_DIRECT_CONNECT_ENV = "RIPPERDOC_BRIDGE_DIRECT_CONNECT"

# Poll config env var names
_POLL_INTERVAL_MS_ENV = "RIPPERDOC_BRIDGE_POLL_INTERVAL_MS"
_POLL_BLOCK_MS_ENV = "RIPPERDOC_BRIDGE_POLL_BLOCK_MS"
_CONNECT_TIMEOUT_MS_ENV = "RIPPERDOC_BRIDGE_CONNECT_TIMEOUT_MS"
_HEARTBEAT_INTERVAL_MS_ENV = "RIPPERDOC_BRIDGE_HEARTBEAT_INTERVAL_MS"
_UUID_DEDUP_BUFFER_ENV = "RIPPERDOC_BRIDGE_UUID_DEDUP_BUFFER"


class BridgeFeatureFlags:
    """Feature flag configuration with fallback to env vars."""

    def __init__(self) -> None:
        self._load_from_env()

    def _load_from_env(self) -> None:
        self._v2 = parse_boolish(os.getenv(_V2_ENV), default=False)
        self._heartbeat = parse_boolish(os.getenv(_HEARTBEAT_ENV), default=True)
        self._multi_session = parse_boolish(os.getenv(_MULTI_SESSION_ENV), default=False)
        self._trusted_device = parse_boolish(os.getenv(_TRUSTED_DEVICE_ENV), default=False)
        self._viewer_mode = parse_boolish(os.getenv(_VIEWER_MODE_ENV), default=False)
        self._direct_connect = parse_boolish(os.getenv(_DIRECT_CONNECT_ENV), default=False)

    def is_v2_enabled(self) -> bool:
        return self._v2

    def is_heartbeat_enabled(self) -> bool:
        return self._heartbeat

    def is_multi_session_enabled(self) -> bool:
        return self._multi_session

    def is_trusted_device_enabled(self) -> bool:
        return self._trusted_device

    def is_viewer_mode(self) -> bool:
        return self._viewer_mode

    def is_direct_connect_enabled(self) -> bool:
        return self._direct_connect

    def get_poll_config(self) -> dict[str, Any]:
        """Return dynamic poll configuration from environment."""
        return {
            "poll_interval_ms": _env_int(_POLL_INTERVAL_MS_ENV, 1000),
            "poll_block_ms": _env_int(_POLL_BLOCK_MS_ENV, 900),
            "connect_timeout_ms": _env_int(_CONNECT_TIMEOUT_MS_ENV, 15000),
            "heartbeat_interval_ms": _env_int(_HEARTBEAT_INTERVAL_MS_ENV, 20000),
            "uuid_dedup_buffer": _env_int(_UUID_DEDUP_BUFFER_ENV, 2000),
        }


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        return default
