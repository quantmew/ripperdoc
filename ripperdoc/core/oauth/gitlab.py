"""GitLab OAuth login and refresh helpers."""

from __future__ import annotations

import base64
import hashlib
import os
import secrets
import threading
import time
import urllib.parse
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Dict, Optional, cast

import httpx

from ripperdoc.core.oauth import OAuthToken, OAuthTokenType

GITLAB_DEFAULT_INSTANCE_URL = "https://gitlab.com"
GITLAB_DEFAULT_CALLBACK_PORT = 8080
GITLAB_CALLBACK_PATH = "/callback"
GITLAB_OAUTH_SCOPE = "api"
GITLAB_BUNDLED_CLIENT_ID = (
    "1d89f9fdb23ee96d4e603201f6861dab6e143c5c3c00469a018a2d94bdc03d4e"
)


class GitLabOAuthError(RuntimeError):
    """Raised for GitLab OAuth failures."""


@dataclass(frozen=True)
class GitLabBrowserAuthContext:
    """Context needed to complete browser OAuth from callback URL."""

    auth_url: str
    redirect_uri: str
    state: str
    code_verifier: str
    instance_url: str
    client_id: str


class GitLabOAuthPendingCallback(GitLabOAuthError):
    """Raised when browser flow needs manual callback URL completion."""

    def __init__(self, context: GitLabBrowserAuthContext):
        self.context = context
        super().__init__("OAuth callback timed out. Paste callback URL to finish manually.")


def _base64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _generate_pkce_verifier(length: int = 64) -> str:
    raw = secrets.token_urlsafe(length)
    return raw[:128]


def _pkce_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("utf-8")).digest()
    return _base64url(digest)


def _generate_state() -> str:
    return _base64url(secrets.token_bytes(32))


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


def normalize_gitlab_instance_url(raw_url: str) -> str:
    """Normalize GitLab instance URL to scheme+host."""
    value = (raw_url or "").strip() or GITLAB_DEFAULT_INSTANCE_URL
    parsed = urllib.parse.urlparse(value if "://" in value else f"https://{value}")
    scheme = (parsed.scheme or "https").lower()
    host = (parsed.hostname or "").strip().lower()
    if not host:
        raise GitLabOAuthError("Invalid GitLab instance URL.")
    port = parsed.port
    if port and ((scheme == "https" and port != 443) or (scheme == "http" and port != 80)):
        return f"{scheme}://{host}:{port}"
    return f"{scheme}://{host}"


def resolve_gitlab_oauth_client_id(explicit_client_id: Optional[str] = None) -> str:
    """Resolve GitLab OAuth client id from explicit value, env, or bundled fallback."""
    candidate = (explicit_client_id or os.getenv("GITLAB_OAUTH_CLIENT_ID") or "").strip()
    return candidate or GITLAB_BUNDLED_CLIENT_ID


def _build_authorize_url(
    *,
    instance_url: str,
    client_id: str,
    redirect_uri: str,
    code_challenge: str,
    state: str,
) -> str:
    params = urllib.parse.urlencode(
        {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "state": state,
            "scope": GITLAB_OAUTH_SCOPE,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
    )
    return f"{instance_url}/oauth/authorize?{params}"


def _extract_code_from_callback_url(
    *,
    callback_url: str,
    expected_state: str,
    expected_path: str = GITLAB_CALLBACK_PATH,
) -> str:
    parsed = urllib.parse.urlparse((callback_url or "").strip())
    if parsed.scheme not in {"http", "https"}:
        raise GitLabOAuthError("Callback URL must start with http:// or https://")
    if parsed.path and expected_path and parsed.path != expected_path:
        raise GitLabOAuthError(
            f"Callback URL path mismatch: expected '{expected_path}', got '{parsed.path}'"
        )

    query = urllib.parse.parse_qs(parsed.query)
    error = query.get("error", [None])[0]
    if isinstance(error, str) and error:
        description = query.get("error_description", [None])[0]
        raise GitLabOAuthError(str(description or error))

    state = query.get("state", [None])[0]
    if state != expected_state:
        raise GitLabOAuthError("Invalid callback state.")

    code = query.get("code", [None])[0]
    if not isinstance(code, str) or not code:
        raise GitLabOAuthError("Callback URL does not include authorization code.")
    return code


def _exchange_authorization_code(
    *,
    instance_url: str,
    client_id: str,
    code: str,
    redirect_uri: str,
    code_verifier: str,
    timeout_sec: float = 30.0,
) -> Dict[str, object]:
    with httpx.Client(timeout=timeout_sec) as client:
        response = client.post(
            f"{instance_url}/oauth/token",
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            },
            data={
                "client_id": client_id,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": redirect_uri,
                "code_verifier": code_verifier,
            },
        )
    if response.status_code >= 400:
        raise GitLabOAuthError(
            f"OAuth token exchange failed ({response.status_code}): {_extract_error_message(response)}"
        )
    payload = response.json()
    if not isinstance(payload, dict):
        raise GitLabOAuthError("OAuth token exchange returned unexpected payload.")
    return cast(Dict[str, object], payload)


def _exchange_refresh_token(
    *,
    instance_url: str,
    client_id: str,
    refresh_token: str,
    timeout_sec: float = 30.0,
) -> Dict[str, object]:
    with httpx.Client(timeout=timeout_sec) as client:
        response = client.post(
            f"{instance_url}/oauth/token",
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            },
            data={
                "client_id": client_id,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            },
        )
    if response.status_code >= 400:
        raise GitLabOAuthError(
            f"OAuth refresh failed ({response.status_code}): {_extract_error_message(response)}"
        )
    payload = response.json()
    if not isinstance(payload, dict):
        raise GitLabOAuthError("OAuth refresh returned unexpected payload.")
    return cast(Dict[str, object], payload)


def _token_from_payload(payload: Dict[str, object], *, instance_url: str) -> OAuthToken:
    access = payload.get("access_token")
    if not isinstance(access, str) or not access:
        raise GitLabOAuthError("OAuth response missing access_token.")

    refresh_raw = payload.get("refresh_token")
    refresh = refresh_raw if isinstance(refresh_raw, str) and refresh_raw else None
    expires_in = payload.get("expires_in")
    ttl = (
        int(expires_in)
        if isinstance(expires_in, (int, float)) and int(expires_in) > 0
        else 3600
    )
    expires_at = int(time.time() * 1000) + ttl * 1000

    return OAuthToken(
        type=OAuthTokenType.GITLAB,
        access_token=access,
        refresh_token=refresh,
        expires_at=expires_at,
        account_id=instance_url,
    )


def complete_gitlab_browser_auth_from_callback_url(
    context: GitLabBrowserAuthContext, callback_url: str
) -> OAuthToken:
    """Complete browser login by parsing callback URL and exchanging code."""
    code = _extract_code_from_callback_url(
        callback_url=callback_url,
        expected_state=context.state,
        expected_path=urllib.parse.urlparse(context.redirect_uri).path or GITLAB_CALLBACK_PATH,
    )
    payload = _exchange_authorization_code(
        instance_url=context.instance_url,
        client_id=context.client_id,
        code=code,
        redirect_uri=context.redirect_uri,
        code_verifier=context.code_verifier,
    )
    return _token_from_payload(payload, instance_url=context.instance_url)


def start_gitlab_browser_auth(
    *,
    instance_url: str = GITLAB_DEFAULT_INSTANCE_URL,
    client_id: Optional[str] = None,
    callback_port: int = GITLAB_DEFAULT_CALLBACK_PORT,
    open_browser: bool = True,
    notify: Optional[Callable[[str], None]] = None,
) -> GitLabBrowserAuthContext:
    """Create browser auth context and open GitLab authorization URL."""
    normalized_instance = normalize_gitlab_instance_url(instance_url)
    resolved_client_id = resolve_gitlab_oauth_client_id(client_id)
    code_verifier = _generate_pkce_verifier()
    challenge = _pkce_challenge(code_verifier)
    state = _generate_state()
    redirect_uri = f"http://127.0.0.1:{callback_port}{GITLAB_CALLBACK_PATH}"
    auth_url = _build_authorize_url(
        instance_url=normalized_instance,
        client_id=resolved_client_id,
        redirect_uri=redirect_uri,
        code_challenge=challenge,
        state=state,
    )
    context = GitLabBrowserAuthContext(
        auth_url=auth_url,
        redirect_uri=redirect_uri,
        state=state,
        code_verifier=code_verifier,
        instance_url=normalized_instance,
        client_id=resolved_client_id,
    )
    if notify:
        notify(f"Open this URL to authorize GitLab: {auth_url}")
    if open_browser:
        try:
            webbrowser.open(auth_url, new=2)
        except Exception:  # noqa: BLE001
            pass
    return context


def login_gitlab_with_browser(
    *,
    instance_url: str = GITLAB_DEFAULT_INSTANCE_URL,
    client_id: Optional[str] = None,
    callback_port: int = GITLAB_DEFAULT_CALLBACK_PORT,
    timeout_sec: int = 300,
    open_browser: bool = True,
    notify: Optional[Callable[[str], None]] = None,
) -> OAuthToken:
    """Login with browser-based local callback flow."""
    context = start_gitlab_browser_auth(
        instance_url=instance_url,
        client_id=client_id,
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
            if parsed.path != GITLAB_CALLBACK_PATH:
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
                        instance_url=context.instance_url,
                        client_id=context.client_id,
                        code=str(code),
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
        raise GitLabOAuthError(
            f"Failed to start OAuth callback server on 127.0.0.1:{callback_port}: {exc}"
        ) from exc

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        if not ready.wait(timeout=timeout_sec):
            raise GitLabOAuthPendingCallback(context)
        if "error" in result:
            raise GitLabOAuthError(str(result["error"]))
        payload = result.get("token_payload")
        if not isinstance(payload, dict):
            raise GitLabOAuthError("OAuth callback did not return token payload.")
        return _token_from_payload(cast(Dict[str, object], payload), instance_url=context.instance_url)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


def refresh_gitlab_access_token(
    token: OAuthToken,
    *,
    instance_url: Optional[str] = None,
    client_id: Optional[str] = None,
    timeout_sec: float = 30.0,
) -> OAuthToken:
    """Refresh GitLab OAuth access token."""
    refresh_token = (token.refresh_token or "").strip()
    if not refresh_token:
        raise GitLabOAuthError("Refresh token is missing.")

    resolved_instance = normalize_gitlab_instance_url(
        instance_url or token.account_id or GITLAB_DEFAULT_INSTANCE_URL
    )
    resolved_client_id = resolve_gitlab_oauth_client_id(client_id)
    payload = _exchange_refresh_token(
        instance_url=resolved_instance,
        client_id=resolved_client_id,
        refresh_token=refresh_token,
        timeout_sec=timeout_sec,
    )
    refreshed = _token_from_payload(payload, instance_url=resolved_instance)
    if not refreshed.refresh_token:
        refreshed = refreshed.model_copy(update={"refresh_token": refresh_token})
    return refreshed


__all__ = [
    "GITLAB_DEFAULT_CALLBACK_PORT",
    "GITLAB_DEFAULT_INSTANCE_URL",
    "GitLabBrowserAuthContext",
    "GitLabOAuthError",
    "GitLabOAuthPendingCallback",
    "complete_gitlab_browser_auth_from_callback_url",
    "login_gitlab_with_browser",
    "normalize_gitlab_instance_url",
    "refresh_gitlab_access_token",
    "resolve_gitlab_oauth_client_id",
    "start_gitlab_browser_auth",
]
