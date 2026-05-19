"""Configuration for the v2 env-less bridge path."""

from __future__ import annotations

import os
from dataclasses import dataclass



def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(0.0, float(raw))
    except ValueError:
        return default


@dataclass(frozen=True)
class EnvLessBridgeConfig:
    """Static configuration for the v2 env-less bridge."""

    http_timeout_ms: int = _env_int("RIPPERDOC_BRIDGE_HTTP_TIMEOUT_MS", 15_000)
    heartbeat_interval_ms: int = _env_int("RIPPERDOC_BRIDGE_V2_HEARTBEAT_MS", 20_000)
    heartbeat_jitter_fraction: float = _env_float("RIPPERDOC_BRIDGE_V2_HEARTBEAT_JITTER", 0.1)
    token_refresh_buffer_ms: int = _env_int("RIPPERDOC_BRIDGE_TOKEN_REFRESH_BUFFER_MS", 300_000)
    init_retry_max_attempts: int = _env_int("RIPPERDOC_BRIDGE_INIT_RETRY_MAX", 3)
    init_retry_base_delay_ms: int = _env_int("RIPPERDOC_BRIDGE_INIT_RETRY_BASE_MS", 500)
    init_retry_max_delay_ms: int = _env_int("RIPPERDOC_BRIDGE_INIT_RETRY_MAX_MS", 10_000)
    init_retry_jitter_fraction: float = _env_float("RIPPERDOC_BRIDGE_INIT_RETRY_JITTER", 0.25)
    uuid_dedup_buffer_size: int = _env_int("RIPPERDOC_BRIDGE_UUID_DEDUP_BUFFER", 2000)
    connect_timeout_ms: int = _env_int("RIPPERDOC_BRIDGE_CONNECT_TIMEOUT_MS", 15_000)
    teardown_archive_timeout_ms: int = _env_int("RIPPERDOC_BRIDGE_ARCHIVE_TIMEOUT_MS", 1500)
