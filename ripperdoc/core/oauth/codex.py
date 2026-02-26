"""Codex OAuth login and refresh helpers."""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import threading
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, cast

import httpx

from ripperdoc.core.oauth import OAuthToken, OAuthTokenType

CODEX_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_OAUTH_ISSUER = "https://auth.openai.com"
CODEX_OAUTH_DEFAULT_CALLBACK_PORT = 1455
CODEX_OAUTH_CALLBACK_PATH = "/auth/callback"
_DEVICE_POLL_SAFETY_MARGIN_SEC = 3


class CodexOAuthError(RuntimeError):
    """Raised for Codex OAuth login/refresh failures."""


@dataclass(frozen=True)
class CodexBrowserAuthContext:
    """Context needed to complete browser OAuth from callback URL."""

    auth_url: str
    redirect_uri: str
    state: str
    code_verifier: str


class CodexOAuthPendingCallback(CodexOAuthError):
    """Raised when browser flow needs manual callback URL completion."""

    def __init__(self, context: CodexBrowserAuthContext):
        self.context = context
        super().__init__(
            "OAuth callback timed out. Paste callback URL to finish manually."
        )


def _base64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _generate_pkce_verifier(length: int = 64) -> str:
    raw = secrets.token_urlsafe(length)
    # PKCE verifier max length is 128.
    return raw[:128]


def _pkce_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("utf-8")).digest()
    return _base64url(digest)


def _generate_state() -> str:
    return _base64url(secrets.token_bytes(32))


def _build_authorize_url(
    *,
    redirect_uri: str,
    code_challenge: str,
    state: str,
) -> str:
    params = urllib.parse.urlencode(
        {
            "response_type": "code",
            "client_id": CODEX_OAUTH_CLIENT_ID,
            "redirect_uri": redirect_uri,
            "scope": "openid profile email offline_access",
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "id_token_add_organizations": "true",
            "codex_cli_simplified_flow": "true",
            "state": state,
            "originator": "ripperdoc",
        }
    )
    return f"{CODEX_OAUTH_ISSUER}/oauth/authorize?{params}"


def _extract_code_from_callback_url(
    *,
    callback_url: str,
    expected_state: str,
    expected_path: str = CODEX_OAUTH_CALLBACK_PATH,
) -> str:
    parsed = urllib.parse.urlparse((callback_url or "").strip())
    if parsed.scheme not in {"http", "https"}:
        raise CodexOAuthError("Callback URL must start with http:// or https://")

    if parsed.path and expected_path and parsed.path != expected_path:
        raise CodexOAuthError(
            f"Callback URL path mismatch: expected '{expected_path}', got '{parsed.path}'"
        )

    query = urllib.parse.parse_qs(parsed.query)
    error = query.get("error", [None])[0]
    if isinstance(error, str) and error:
        description = query.get("error_description", [None])[0]
        raise CodexOAuthError(str(description or error))

    state = query.get("state", [None])[0]
    if state != expected_state:
        raise CodexOAuthError("Invalid callback state.")

    code = query.get("code", [None])[0]
    if not isinstance(code, str) or not code:
        raise CodexOAuthError("Callback URL does not include authorization code.")
    return code


def _extract_error_message(response: httpx.Response) -> str:
    text = response.text
    try:
        payload = response.json()
    except ValueError:
        return text or f"HTTP {response.status_code}"

    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict):
            detail = err.get("message") or err.get("error_description")
            if isinstance(detail, str) and detail:
                return detail
        for field in ("error_description", "message", "error"):
            value = payload.get(field)
            if isinstance(value, str) and value:
                return value

    return text or f"HTTP {response.status_code}"


def parse_jwt_claims(token: str) -> Optional[Dict[str, object]]:
    """Decode JWT claims without verifying signature."""
    parts = token.split(".")
    if len(parts) != 3:
        return None
    payload = parts[1]
    padded = payload + ("=" * ((4 - len(payload) % 4) % 4))
    try:
        decoded = base64.urlsafe_b64decode(padded.encode("ascii"))
        obj = json.loads(decoded.decode("utf-8"))
        if isinstance(obj, dict):
            return cast(Dict[str, object], obj)
    except (ValueError, TypeError, UnicodeDecodeError):
        return None
    return None


def extract_account_id_from_claims(claims: Dict[str, object]) -> Optional[str]:
    """Extract ChatGPT account/workspace id from JWT claims."""
    direct = claims.get("chatgpt_account_id")
    if isinstance(direct, str) and direct:
        return direct

    nested = claims.get("https://api.openai.com/auth")
    if isinstance(nested, dict):
        nested_account = nested.get("chatgpt_account_id")
        if isinstance(nested_account, str) and nested_account:
            return nested_account

    orgs = claims.get("organizations")
    if isinstance(orgs, list) and orgs:
        first = orgs[0]
        if isinstance(first, dict):
            org_id = first.get("id")
            if isinstance(org_id, str) and org_id:
                return org_id

    return None


def extract_account_id(tokens: Dict[str, object]) -> Optional[str]:
    """Extract account id from id/access token claims."""
    for key in ("id_token", "access_token"):
        raw = tokens.get(key)
        if not isinstance(raw, str) or not raw:
            continue
        claims = parse_jwt_claims(raw)
        if not claims:
            continue
        account_id = extract_account_id_from_claims(claims)
        if account_id:
            return account_id
    return None


def _exchange_authorization_code(
    *,
    code: str,
    redirect_uri: str,
    code_verifier: str,
    timeout_sec: float = 30.0,
) -> Dict[str, object]:
    with httpx.Client(timeout=timeout_sec) as client:
        response = client.post(
            f"{CODEX_OAUTH_ISSUER}/oauth/token",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
                "client_id": CODEX_OAUTH_CLIENT_ID,
                "code_verifier": code_verifier,
            },
        )
    if response.status_code >= 400:
        raise CodexOAuthError(
            f"OAuth token exchange failed ({response.status_code}): {_extract_error_message(response)}"
        )
    payload = response.json()
    if not isinstance(payload, dict):
        raise CodexOAuthError("OAuth token exchange returned unexpected payload.")
    return cast(Dict[str, object], payload)


def refresh_codex_access_token(token: OAuthToken, *, timeout_sec: float = 30.0) -> OAuthToken:
    """Refresh Codex OAuth access token using refresh token."""
    refresh_token = (token.refresh_token or "").strip()
    if not refresh_token:
        raise CodexOAuthError("Refresh token is missing.")

    with httpx.Client(timeout=timeout_sec) as client:
        response = client.post(
            f"{CODEX_OAUTH_ISSUER}/oauth/token",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": CODEX_OAUTH_CLIENT_ID,
            },
        )
    if response.status_code >= 400:
        raise CodexOAuthError(
            f"OAuth refresh failed ({response.status_code}): {_extract_error_message(response)}"
        )
    payload = response.json()
    if not isinstance(payload, dict):
        raise CodexOAuthError("OAuth refresh returned unexpected payload.")

    access = payload.get("access_token")
    if not isinstance(access, str) or not access:
        raise CodexOAuthError("OAuth refresh response did not include access_token.")

    refreshed = payload.get("refresh_token")
    refresh = refreshed if isinstance(refreshed, str) and refreshed else refresh_token
    expires_in = payload.get("expires_in")
    ttl = int(expires_in) if isinstance(expires_in, (int, float)) and int(expires_in) > 0 else 3600
    expires_at = int(time.time() * 1000) + ttl * 1000
    account_id = extract_account_id(payload) or token.account_id
    return OAuthToken(
        type=OAuthTokenType.CODEX,
        access_token=access,
        refresh_token=refresh,
        expires_at=expires_at,
        account_id=account_id,
    )


def _token_from_payload(payload: Dict[str, object]) -> OAuthToken:
    access = payload.get("access_token")
    if not isinstance(access, str) or not access:
        raise CodexOAuthError("OAuth response missing access_token.")

    refresh = payload.get("refresh_token")
    refresh_token = refresh if isinstance(refresh, str) and refresh else None
    expires_in = payload.get("expires_in")
    ttl = int(expires_in) if isinstance(expires_in, (int, float)) and int(expires_in) > 0 else 3600
    expires_at = int(time.time() * 1000) + ttl * 1000
    return OAuthToken(
        type=OAuthTokenType.CODEX,
        access_token=access,
        refresh_token=refresh_token,
        expires_at=expires_at,
        account_id=extract_account_id(payload),
    )


def complete_codex_browser_auth_from_callback_url(
    context: CodexBrowserAuthContext, callback_url: str
) -> OAuthToken:
    """Complete browser login by parsing callback URL and exchanging code."""
    code = _extract_code_from_callback_url(
        callback_url=callback_url,
        expected_state=context.state,
        expected_path=urllib.parse.urlparse(context.redirect_uri).path or CODEX_OAUTH_CALLBACK_PATH,
    )
    payload = _exchange_authorization_code(
        code=code,
        redirect_uri=context.redirect_uri,
        code_verifier=context.code_verifier,
    )
    return _token_from_payload(payload)


def start_codex_browser_auth(
    *,
    callback_port: int = CODEX_OAUTH_DEFAULT_CALLBACK_PORT,
    open_browser: bool = True,
    notify: Optional[Callable[[str], None]] = None,
) -> CodexBrowserAuthContext:
    """Create browser auth context and open authorization URL."""
    code_verifier = _generate_pkce_verifier()
    challenge = _pkce_challenge(code_verifier)
    state = _generate_state()
    redirect_uri = f"http://localhost:{callback_port}{CODEX_OAUTH_CALLBACK_PATH}"
    auth_url = _build_authorize_url(
        redirect_uri=redirect_uri,
        code_challenge=challenge,
        state=state,
    )
    context = CodexBrowserAuthContext(
        auth_url=auth_url,
        redirect_uri=redirect_uri,
        state=state,
        code_verifier=code_verifier,
    )
    if notify:
        notify(f"Open this URL to authorize Codex: {auth_url}")
    if open_browser:
        try:
            webbrowser.open(auth_url, new=2)
        except Exception:  # noqa: BLE001
            pass
    return context


def login_codex_with_browser(
    *,
    callback_port: int = CODEX_OAUTH_DEFAULT_CALLBACK_PORT,
    timeout_sec: int = 300,
    open_browser: bool = True,
    notify: Optional[Callable[[str], None]] = None,
) -> OAuthToken:
    """Login with browser-based local callback flow."""
    context = start_codex_browser_auth(
        callback_port=callback_port,
        open_browser=open_browser,
        notify=notify,
    )
    redirect_uri = context.redirect_uri
    state = context.state
    code_verifier = context.code_verifier

    result: Dict[str, object] = {}
    ready = threading.Event()

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *args: object) -> None:  # noqa: A003
            return

        def do_GET(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path != CODEX_OAUTH_CALLBACK_PATH:
                self.send_response(404)
                self.end_headers()
                return

            query = urllib.parse.parse_qs(parsed.query)
            error = query.get("error", [None])[0]
            error_description = query.get("error_description", [None])[0]
            callback_state = query.get("state", [None])[0]
            code = query.get("code", [None])[0]

            if error:
                result["error"] = str(error_description or error)
                html = "<h1>Authorization failed</h1><p>You can close this window.</p>"
                self.send_response(400)
            elif callback_state != state:
                result["error"] = "Invalid OAuth state."
                html = "<h1>Authorization failed</h1><p>Invalid state.</p>"
                self.send_response(400)
            elif not code:
                result["error"] = "Missing authorization code."
                html = "<h1>Authorization failed</h1><p>Missing code.</p>"
                self.send_response(400)
            else:
                try:
                    payload = _exchange_authorization_code(
                        code=code,
                        redirect_uri=redirect_uri,
                        code_verifier=code_verifier,
                    )
                except Exception as exc:  # noqa: BLE001
                    result["error"] = str(exc)
                    html = "<h1>Authorization failed</h1><p>Token exchange failed.</p>"
                    self.send_response(400)
                else:
                    result["token_payload"] = payload
                    html = (
                        "<h1>Authorization successful</h1>"
                        "<p>You can close this window and return to terminal.</p>"
                        "<script>setTimeout(() => window.close(), 1800)</script>"
                    )
                    self.send_response(200)

            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))
            ready.set()

    try:
        server = ThreadingHTTPServer(("127.0.0.1", callback_port), _Handler)
    except OSError as exc:
        raise CodexOAuthError(
            f"Failed to start OAuth callback server on localhost:{callback_port}: {exc}"
        ) from exc

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        if not ready.wait(timeout=timeout_sec):
            raise CodexOAuthPendingCallback(context)

        if "error" in result:
            raise CodexOAuthError(str(result["error"]))
        payload = result.get("token_payload")
        if not isinstance(payload, dict):
            raise CodexOAuthError("OAuth callback did not return token payload.")
        return _token_from_payload(cast(Dict[str, object], payload))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


def login_codex_with_device_code(
    *,
    timeout_sec: int = 600,
    open_browser: bool = True,
    notify: Optional[Callable[[str], None]] = None,
) -> Tuple[OAuthToken, str]:
    """Login with device code flow (headless-friendly)."""
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            f"{CODEX_OAUTH_ISSUER}/api/accounts/deviceauth/usercode",
            headers={"Content-Type": "application/json"},
            json={"client_id": CODEX_OAUTH_CLIENT_ID},
        )
    if response.status_code >= 400:
        raise CodexOAuthError(
            f"Device auth init failed ({response.status_code}): {_extract_error_message(response)}"
        )
    payload = response.json()
    if not isinstance(payload, dict):
        raise CodexOAuthError("Device auth init returned unexpected payload.")

    device_auth_id = payload.get("device_auth_id")
    user_code = payload.get("user_code")
    interval_raw = payload.get("interval")
    if not isinstance(device_auth_id, str) or not device_auth_id:
        raise CodexOAuthError("Device auth response missing device_auth_id.")
    if not isinstance(user_code, str) or not user_code:
        raise CodexOAuthError("Device auth response missing user_code.")

    interval = (
        int(interval_raw)
        if isinstance(interval_raw, (int, float, str)) and str(interval_raw).isdigit()
        else 5
    )
    interval = max(1, interval)
    verify_url = f"{CODEX_OAUTH_ISSUER}/codex/device"

    if notify:
        notify(f"Open {verify_url} and enter code: {user_code}")
    if open_browser:
        try:
            webbrowser.open(verify_url, new=2)
        except Exception:  # noqa: BLE001
            pass

    deadline = time.time() + timeout_sec
    with httpx.Client(timeout=30.0) as client:
        while time.time() < deadline:
            poll = client.post(
                f"{CODEX_OAUTH_ISSUER}/api/accounts/deviceauth/token",
                headers={"Content-Type": "application/json"},
                json={
                    "device_auth_id": device_auth_id,
                    "user_code": user_code,
                },
            )
            if poll.status_code == 200:
                poll_payload = poll.json()
                if not isinstance(poll_payload, dict):
                    raise CodexOAuthError("Device auth token response has invalid payload.")
                auth_code = poll_payload.get("authorization_code")
                code_verifier = poll_payload.get("code_verifier")
                if not isinstance(auth_code, str) or not auth_code:
                    raise CodexOAuthError("Device auth token response missing authorization_code.")
                if not isinstance(code_verifier, str) or not code_verifier:
                    raise CodexOAuthError("Device auth token response missing code_verifier.")

                token_payload = _exchange_authorization_code(
                    code=auth_code,
                    redirect_uri=f"{CODEX_OAUTH_ISSUER}/deviceauth/callback",
                    code_verifier=code_verifier,
                )
                return _token_from_payload(token_payload), user_code

            # Pending authorization: server currently returns 403/404 in this flow.
            if poll.status_code not in (403, 404):
                raise CodexOAuthError(
                    f"Device auth polling failed ({poll.status_code}): {_extract_error_message(poll)}"
                )
            time.sleep(interval + _DEVICE_POLL_SAFETY_MARGIN_SEC)

    raise CodexOAuthError("Device auth timed out.")
