"""Trusted device token source for bridge (remote-control) sessions.

Bridge sessions have SecurityTier=ELEVATED on the server (CCR v2).
When the enforcement flag is on, the server requires a trusted device
token sent as X-Trusted-Device-Token header.

Enrollment (POST /auth/trusted_devices) is gated server-side by
account_session.created_at < 10min, so it must happen during /login.
Token is persistent (90d rolling expiry) and stored in secure storage.
"""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path
from typing import Any

from ripperdoc.utils.log import get_logger

logger = get_logger()

TRUSTED_DEVICE_ENV = "RIPPERDOC_TRUSTED_DEVICE_TOKEN"
TRUSTED_DEVICE_GATE_ENV = "RIPPERDOC_TRUSTED_DEVICE_ENABLED"

_cached_token: str | None = None


def _get_auth_storage_path() -> Path:
    """Return the path to the auth storage file."""
    home = Path.home()
    base = home / ".ripperdoc"
    env_dir = os.getenv("RIPPERDOC_PROJECTS_DIR", "").strip()
    if env_dir:
        base = Path(env_dir)
    return base / "auth.json"


def _read_auth_storage() -> dict[str, Any] | None:
    """Read the auth storage file. Returns None on any failure."""
    path = _get_auth_storage_path()
    try:
        if not path.exists():
            return None
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def _write_auth_storage(data: dict[str, Any]) -> bool:
    """Write the auth storage file. Returns True on success."""
    path = _get_auth_storage_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return True
    except OSError as exc:
        logger.debug("[trusted-device] Failed to write auth storage: %s", exc)
        return False


def _is_gate_enabled() -> bool:
    """Check whether the trusted device feature gate is enabled."""
    val = os.getenv(TRUSTED_DEVICE_GATE_ENV, "").strip().lower()
    return val in ("1", "true", "yes")


def get_trusted_device_token() -> str | None:
    """Return the trusted device token if the gate is enabled and a token exists."""
    global _cached_token

    if not _is_gate_enabled():
        return None

    if _cached_token is not None:
        return _cached_token

    # Env var takes precedence for testing/canary
    env_token = os.getenv(TRUSTED_DEVICE_ENV, "").strip()
    if env_token:
        _cached_token = env_token
        return _cached_token

    # Check secure storage
    storage = _read_auth_storage()
    if storage and isinstance(storage.get("trusted_device_token"), str):
        _cached_token = storage["trusted_device_token"]
        return _cached_token

    return None


def clear_trusted_device_token_cache() -> None:
    """Clear the in-memory cache."""
    global _cached_token
    _cached_token = None


def clear_trusted_device_token() -> None:
    """Clear the stored trusted device token from secure storage and cache."""
    global _cached_token
    _cached_token = None

    if not _is_gate_enabled():
        return

    storage = _read_auth_storage()
    if storage and "trusted_device_token" in storage:
        del storage["trusted_device_token"]
        _write_auth_storage(storage)


def enroll_trusted_device(access_token: str, base_url: str) -> None:
    """Enroll this device via POST /auth/trusted_devices and persist the token.

    Best-effort -- logs and returns on failure so callers don't block.
    """
    global _cached_token

    if not _is_gate_enabled():
        return

    # If env var is set, skip enrollment
    if os.getenv(TRUSTED_DEVICE_ENV, "").strip():
        return

    if not access_token:
        return

    display_name = f"Ripperdoc on {platform.node()} ({platform.system()})"

    try:
        import urllib.request
        import urllib.error

        url = f"{base_url.rstrip('/')}/api/auth/trusted_devices"
        payload = json.dumps({"display_name": display_name}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                status = int(getattr(response, "status", 200))
                raw = response.read()
        except urllib.error.HTTPError as exc:
            logger.debug("[trusted-device] Enrollment failed with status %s", exc.code)
            return
        except (urllib.error.URLError, OSError) as exc:
            logger.debug("[trusted-device] Enrollment request failed: %s", exc)
            return

        if status not in {200, 201}:
            logger.debug("[trusted-device] Enrollment failed with status %s", status)
            return

        try:
            data = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return

        token = data.get("device_token")
        if not isinstance(token, str) or not token.strip():
            return

        # Persist to storage
        storage = _read_auth_storage() or {}
        storage["trusted_device_token"] = token
        if _write_auth_storage(storage):
            _cached_token = token
            logger.info("[trusted-device] Enrolled device_id=%s", data.get("device_id", "unknown"))
    except Exception as exc:  # noqa: BLE001
        logger.debug("[trusted-device] Enrollment error: %s", exc)
