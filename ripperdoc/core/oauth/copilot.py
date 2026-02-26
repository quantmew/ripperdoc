"""GitHub Copilot OAuth device-flow helpers."""

from __future__ import annotations

import time
import urllib.parse
import webbrowser
from typing import Callable, Optional, Tuple

import httpx

from ripperdoc.core.oauth import OAuthToken, OAuthTokenType
from ripperdoc.utils.user_agent import build_user_agent

COPILOT_OAUTH_CLIENT_ID = "Ov23li8tweQw6odWQebz"  # OpenCode OAuth client ID
COPILOT_DEFAULT_DOMAIN = "github.com"
_POLLING_SAFETY_MARGIN_SEC = 3


class CopilotOAuthError(RuntimeError):
    """Raised for Copilot OAuth login failures."""


def _normalize_domain(raw: str) -> str:
    value = (raw or "").strip()
    if not value:
        return COPILOT_DEFAULT_DOMAIN
    parsed = urllib.parse.urlparse(value if "://" in value else f"https://{value}")
    host = (parsed.hostname or "").strip().lower()
    if not host:
        raise CopilotOAuthError("Invalid GitHub domain.")
    return host


def _urls_for_domain(domain: str) -> tuple[str, str]:
    return (
        f"https://{domain}/login/device/code",
        f"https://{domain}/login/oauth/access_token",
    )


def _extract_error_message(response: httpx.Response) -> str:
    text = response.text
    try:
        payload = response.json()
    except ValueError:
        return text or f"HTTP {response.status_code}"
    if isinstance(payload, dict):
        for field in ("error_description", "error", "message"):
            value = payload.get(field)
            if isinstance(value, str) and value:
                return value
    return text or f"HTTP {response.status_code}"


def login_copilot_with_device_code(
    *,
    github_domain: str = COPILOT_DEFAULT_DOMAIN,
    timeout_sec: int = 600,
    open_browser: bool = True,
    notify: Optional[Callable[[str], None]] = None,
) -> Tuple[OAuthToken, str]:
    """Login with GitHub Copilot via OAuth device flow."""
    domain = _normalize_domain(github_domain)
    device_code_url, access_token_url = _urls_for_domain(domain)

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": build_user_agent(),
    }

    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            device_code_url,
            headers=headers,
            json={
                "client_id": COPILOT_OAUTH_CLIENT_ID,
                "scope": "read:user",
            },
        )
    if response.status_code >= 400:
        raise CopilotOAuthError(
            f"Device authorization failed ({response.status_code}): {_extract_error_message(response)}"
        )

    payload = response.json()
    if not isinstance(payload, dict):
        raise CopilotOAuthError("Device authorization returned unexpected payload.")

    verification_uri = payload.get("verification_uri")
    user_code = payload.get("user_code")
    device_code = payload.get("device_code")
    interval_raw = payload.get("interval")
    if not isinstance(verification_uri, str) or not verification_uri:
        raise CopilotOAuthError("Device authorization response missing verification_uri.")
    if not isinstance(user_code, str) or not user_code:
        raise CopilotOAuthError("Device authorization response missing user_code.")
    if not isinstance(device_code, str) or not device_code:
        raise CopilotOAuthError("Device authorization response missing device_code.")
    interval = (
        int(interval_raw)
        if isinstance(interval_raw, (int, float, str)) and str(interval_raw).isdigit()
        else 5
    )
    interval = max(1, interval)

    if notify:
        notify(f"Open {verification_uri} and enter code: {user_code}")
    if open_browser:
        try:
            webbrowser.open(verification_uri, new=2)
        except Exception:  # noqa: BLE001
            pass

    deadline = time.time() + timeout_sec
    with httpx.Client(timeout=30.0) as client:
        while time.time() < deadline:
            poll = client.post(
                access_token_url,
                headers=headers,
                json={
                    "client_id": COPILOT_OAUTH_CLIENT_ID,
                    "device_code": device_code,
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                },
            )
            if poll.status_code >= 400:
                raise CopilotOAuthError(
                    f"Device polling failed ({poll.status_code}): {_extract_error_message(poll)}"
                )
            poll_payload = poll.json()
            if not isinstance(poll_payload, dict):
                raise CopilotOAuthError("Device polling returned unexpected payload.")

            access_token = poll_payload.get("access_token")
            if isinstance(access_token, str) and access_token:
                enterprise_domain: Optional[str] = None
                if domain != COPILOT_DEFAULT_DOMAIN:
                    enterprise_domain = domain
                token = OAuthToken(
                    type=OAuthTokenType.COPILOT,
                    access_token=access_token,
                    refresh_token=access_token,
                    expires_at=None,
                    account_id=enterprise_domain,
                )
                return token, user_code

            error = poll_payload.get("error")
            if error == "authorization_pending":
                time.sleep(interval + _POLLING_SAFETY_MARGIN_SEC)
                continue
            if error == "slow_down":
                server_interval = poll_payload.get("interval")
                if isinstance(server_interval, (int, float)) and int(server_interval) > 0:
                    interval = int(server_interval)
                else:
                    interval += 5
                time.sleep(interval + _POLLING_SAFETY_MARGIN_SEC)
                continue
            if isinstance(error, str) and error:
                raise CopilotOAuthError(f"Device authorization failed: {error}")
            time.sleep(interval + _POLLING_SAFETY_MARGIN_SEC)

    raise CopilotOAuthError("Device authorization timed out.")


__all__ = [
    "COPILOT_DEFAULT_DOMAIN",
    "CopilotOAuthError",
    "login_copilot_with_device_code",
]
