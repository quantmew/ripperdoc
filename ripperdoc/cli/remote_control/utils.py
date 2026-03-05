"""Utility helpers for remote-control bridge components."""

from __future__ import annotations

import base64
import json
import os
import random
from datetime import datetime
from urllib.parse import urlparse

from .constants import CONNECT_TEMPLATE_ENV, IDENTIFIER_RE
from .models import WorkSecret


def validate_identifier(value: str, name: str) -> str:
    """Validate identifier shape against bridge-safe character rules."""
    candidate = (value or "").strip()
    if not candidate or not IDENTIFIER_RE.match(candidate):
        raise ValueError(f"Invalid {name}: contains unsafe characters")
    return candidate


def jitter_delay(base_delay_sec: float) -> float:
    """Apply +/-25% jitter to retry delay to reduce synchronized retries."""
    bounded = max(0.0, float(base_delay_sec))
    return max(0.0, bounded + bounded * 0.25 * (2.0 * random.random() - 1.0))


def base64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def decode_work_secret(encoded_secret: str) -> WorkSecret:
    """Decode and validate a base64url work secret payload."""
    if not encoded_secret:
        raise ValueError("empty work secret")
    raw = base64url_decode(encoded_secret).decode("utf-8")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("work secret payload must be a JSON object")

    version = payload.get("version")
    if version != 1:
        raise ValueError(f"unsupported work secret version: {version!r}")

    session_ingress_token = payload.get("session_ingress_token")
    if not isinstance(session_ingress_token, str) or not session_ingress_token.strip():
        raise ValueError("missing or empty session_ingress_token")

    api_base_url = payload.get("api_base_url")
    if api_base_url is not None and not isinstance(api_base_url, str):
        raise ValueError("api_base_url must be a string when provided")

    return WorkSecret(
        version=1,
        session_ingress_token=session_ingress_token.strip(),
        api_base_url=api_base_url.strip() if isinstance(api_base_url, str) and api_base_url.strip() else None,
    )


def build_session_ingress_ws_url(base_url: str, session_id: str) -> str:
    """Build session ingress websocket URL using the bridge ingress URL shape."""
    normalized = base_url.strip()
    parsed = urlparse(normalized)
    hostname = (parsed.hostname or "").strip().lower()
    localhost = hostname in {"localhost", "127.0.0.1"} or hostname.endswith(".localhost")
    ws_scheme = "ws" if localhost else "wss"
    api_version = "v2" if localhost else "v1"
    if parsed.netloc:
        domain = parsed.netloc
        path_prefix = parsed.path.strip("/")
        if path_prefix:
            domain = f"{domain}/{path_prefix}"
    else:
        domain = normalized.replace("http://", "").replace("https://", "").strip("/")
    return f"{ws_scheme}://{domain}/{api_version}/session_ingress/ws/{session_id}"


def build_connect_url(base_url: str, bridge_id: str) -> str:
    """Return human-facing connect URL for a bridge id."""
    template = os.getenv(CONNECT_TEMPLATE_ENV, "").strip()
    if template:
        try:
            return template.format(base_url=base_url.rstrip("/"), bridge_id=bridge_id)
        except Exception:
            pass

    parsed = urlparse(base_url.strip())
    scheme = parsed.scheme or "https"
    hostname = (parsed.hostname or "").strip()
    if not hostname:
        return f"{base_url.rstrip('/')}/code?bridge={bridge_id}"

    app_host = hostname
    for prefix in ("api.", "api-", "gateway."):
        if app_host.startswith(prefix):
            app_host = app_host[len(prefix) :]
            break

    default_port = 443 if scheme == "https" else 80
    if parsed.port and parsed.port != default_port:
        app_host = f"{app_host}:{parsed.port}"

    return f"{scheme}://{app_host}/code?bridge={bridge_id}"


def get_token_expiration_epoch(token: str) -> int | None:
    """Decode JWT-like token and return exp timestamp if present."""
    token_value = token.strip()
    if not token_value:
        return None
    if token_value.startswith("sk-ant-si-"):
        token_value = token_value[10:]
    token_parts = token_value.split(".")
    if len(token_parts) != 3 or not token_parts[1]:
        return None
    try:
        payload = json.loads(base64url_decode(token_parts[1]).decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    exp = payload.get("exp") if isinstance(payload, dict) else None
    if isinstance(exp, int):
        return exp
    if isinstance(exp, float):
        return int(exp)
    return None


def format_iso_from_epoch(epoch_sec: int) -> str:
    """Best-effort ISO timestamp formatter for logs."""
    try:
        return datetime.fromtimestamp(epoch_sec).isoformat()
    except Exception:
        return str(epoch_sec)
