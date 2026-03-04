"""Managed settings loading with policy-priority precedence."""

from __future__ import annotations

import getpass
import json
import os
import plistlib
import sys
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ripperdoc.utils.filesystem.config_paths import config_file_for_scope
from ripperdoc.utils.log import get_logger

logger = get_logger()

_WIN_POLICY_KEY = r"SOFTWARE\Policies\Ripperdoc"
_WIN_POLICY_VALUE_NAME = "Settings"
_MACOS_MDM_DOMAIN = "com.ripperdoc.agent"


@dataclass(frozen=True)
class ManagedSettingsSnapshot:
    data: Dict[str, Any]
    sources: Tuple[str, ...]


def _merge_dict_layers(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _merge_dict_layers(existing, value)
        else:
            merged[key] = value
    return merged


def _coerce_payload(raw: Any) -> Optional[Dict[str, Any]]:
    """Normalize payload into a JSON object (dict) when possible."""
    if isinstance(raw, dict):
        if "Settings" in raw:
            parsed_settings = _coerce_payload(raw.get("Settings"))
            if parsed_settings is not None:
                return parsed_settings
        if "settings" in raw and len(raw) == 1:
            parsed_settings = _coerce_payload(raw.get("settings"))
            if parsed_settings is not None:
                return parsed_settings
        return raw

    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    return None


def _load_json_file(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, IOError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.warning(
            "[managed] Failed to read JSON settings %s: %s: %s",
            path,
            type(exc).__name__,
            exc,
        )
        return None
    return _coerce_payload(parsed)


def _load_server_managed_settings() -> Optional[Dict[str, Any]]:
    inline = os.getenv("RIPPERDOC_SERVER_MANAGED_SETTINGS_JSON")
    if inline and inline.strip():
        payload = _coerce_payload(inline)
        if payload is not None:
            return payload
        logger.warning("[managed] RIPPERDOC_SERVER_MANAGED_SETTINGS_JSON is invalid JSON object")

    path_raw = os.getenv("RIPPERDOC_SERVER_MANAGED_SETTINGS_PATH")
    if path_raw and path_raw.strip():
        payload = _load_json_file(Path(path_raw).expanduser())
        if payload is not None:
            return payload

    # Optional remote endpoint support. Disabled unless explicitly configured.
    url = (os.getenv("RIPPERDOC_SERVER_MANAGED_SETTINGS_URL") or "").strip()
    if not url:
        return None
    headers = {
        "Accept": "application/json",
    }
    token = (os.getenv("RIPPERDOC_SERVER_MANAGED_SETTINGS_TOKEN") or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            body = response.read().decode("utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[managed] Failed loading server-managed settings from %s: %s: %s",
            url,
            type(exc).__name__,
            exc,
        )
        return None
    payload = _coerce_payload(body)
    if payload is None:
        logger.warning("[managed] Server-managed response is not a JSON object")
    return payload


def _load_macos_mdm_settings() -> Optional[Dict[str, Any]]:
    if sys.platform != "darwin":
        return None

    username = ""
    try:
        username = getpass.getuser().strip()
    except Exception:  # noqa: BLE001
        username = ""

    candidates = []
    if username:
        candidates.append(
            Path("/Library/Managed Preferences") / username / f"{_MACOS_MDM_DOMAIN}.plist"
        )
    candidates.append(Path("/Library/Managed Preferences") / f"{_MACOS_MDM_DOMAIN}.plist")

    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            with candidate.open("rb") as handle:
                payload = plistlib.load(handle)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[managed] Failed reading macOS managed preferences %s: %s: %s",
                candidate,
                type(exc).__name__,
                exc,
            )
            continue
        parsed = _coerce_payload(payload)
        if parsed is not None:
            return parsed
    return None


def _read_windows_registry_payload(root: Any) -> Optional[Dict[str, Any]]:
    if sys.platform != "win32":
        return None
    try:
        import winreg  # type: ignore
    except ImportError:
        return None

    try:
        with winreg.OpenKey(root, _WIN_POLICY_KEY) as key:
            value, value_type = winreg.QueryValueEx(key, _WIN_POLICY_VALUE_NAME)
    except OSError:
        return None

    if value_type == getattr(winreg, "REG_EXPAND_SZ", -1) and isinstance(value, str):
        value = os.path.expandvars(value)
    return _coerce_payload(value)


def _load_windows_registry_settings() -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if sys.platform != "win32":
        return None, None
    try:
        import winreg  # type: ignore
    except ImportError:
        return None, None

    hklm = _read_windows_registry_payload(winreg.HKEY_LOCAL_MACHINE)
    hkcu = _read_windows_registry_payload(winreg.HKEY_CURRENT_USER)
    return hklm, hkcu


def _load_file_managed_settings() -> Optional[Dict[str, Any]]:
    return _load_json_file(config_file_for_scope("managed", "managed-settings.json"))


@lru_cache(maxsize=1)
def _load_managed_settings_snapshot() -> ManagedSettingsSnapshot:
    file_payload = _load_file_managed_settings()
    mdm_payload = _load_macos_mdm_settings()
    hklm_payload, hkcu_payload = _load_windows_registry_settings()
    server_payload = _load_server_managed_settings()

    # HKCU is explicitly lowest and only active when no admin-level source exists.
    admin_present = any(
        payload is not None
        for payload in (file_payload, mdm_payload, hklm_payload, server_payload)
    )

    layers: list[tuple[str, Dict[str, Any]]] = []
    if hkcu_payload is not None and not admin_present:
        layers.append(("windows-user-policy", hkcu_payload))
    if file_payload is not None:
        layers.append(("file-managed-settings", file_payload))
    if mdm_payload is not None:
        layers.append(("mdm-os-policy", mdm_payload))
    if hklm_payload is not None:
        layers.append(("windows-admin-policy", hklm_payload))
    if server_payload is not None:
        layers.append(("server-managed-settings", server_payload))

    merged: Dict[str, Any] = {}
    sources: list[str] = []
    for source_name, layer in layers:
        merged = _merge_dict_layers(merged, layer)
        sources.append(source_name)

    return ManagedSettingsSnapshot(data=merged, sources=tuple(sources))


def load_managed_settings_snapshot() -> ManagedSettingsSnapshot:
    """Return merged managed settings and source metadata."""
    return _load_managed_settings_snapshot()


def load_managed_settings() -> Dict[str, Any]:
    """Return merged managed settings object."""
    return dict(load_managed_settings_snapshot().data)


def get_managed_setting(key: str, default: Any = None) -> Any:
    """Get one managed setting by key."""
    if not key:
        return default
    return load_managed_settings_snapshot().data.get(key, default)


def has_managed_settings() -> bool:
    """Whether any managed source is currently active."""
    return bool(load_managed_settings_snapshot().sources)


def reset_managed_settings_cache() -> None:
    """Clear managed settings cache (for tests/runtime refresh)."""
    _load_managed_settings_snapshot.cache_clear()
