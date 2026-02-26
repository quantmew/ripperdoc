"""Tests for GitLab OAuth helpers."""

from __future__ import annotations

import urllib.parse

from ripperdoc.core.oauth import OAuthTokenType
from ripperdoc.core.oauth.gitlab import (
    GITLAB_BUNDLED_CLIENT_ID,
    GitLabBrowserAuthContext,
    complete_gitlab_browser_auth_from_callback_url,
    normalize_gitlab_instance_url,
    resolve_gitlab_oauth_client_id,
    start_gitlab_browser_auth,
)


def test_normalize_gitlab_instance_url_handles_scheme_and_host() -> None:
    assert normalize_gitlab_instance_url("gitlab.company.com") == "https://gitlab.company.com"
    assert (
        normalize_gitlab_instance_url("https://gitlab.company.com:8443/path")
        == "https://gitlab.company.com:8443"
    )


def test_resolve_gitlab_oauth_client_id_priority(monkeypatch) -> None:
    monkeypatch.delenv("GITLAB_OAUTH_CLIENT_ID", raising=False)
    assert resolve_gitlab_oauth_client_id("explicit-client") == "explicit-client"
    monkeypatch.setenv("GITLAB_OAUTH_CLIENT_ID", "env-client")
    assert resolve_gitlab_oauth_client_id() == "env-client"
    monkeypatch.delenv("GITLAB_OAUTH_CLIENT_ID", raising=False)
    assert resolve_gitlab_oauth_client_id() == GITLAB_BUNDLED_CLIENT_ID


def test_start_gitlab_browser_auth_builds_context(monkeypatch) -> None:
    opened: list[str] = []
    notices: list[str] = []

    monkeypatch.setattr(
        "ripperdoc.core.oauth.gitlab.webbrowser.open",
        lambda url, new=2: opened.append(url),  # noqa: ARG005
    )

    context = start_gitlab_browser_auth(
        instance_url="https://gitlab.company.com",
        client_id="client_123",
        callback_port=8080,
        open_browser=True,
        notify=notices.append,
    )

    assert context.redirect_uri == "http://127.0.0.1:8080/callback"
    assert context.auth_url.startswith("https://gitlab.company.com/oauth/authorize?")
    assert opened and opened[0] == context.auth_url
    assert notices and "authorize GitLab" in notices[0]

    parsed = urllib.parse.urlparse(context.auth_url)
    params = urllib.parse.parse_qs(parsed.query)
    assert params.get("state", [None])[0] == context.state
    assert params.get("redirect_uri", [None])[0] == context.redirect_uri


def test_complete_gitlab_browser_auth_from_callback_url(monkeypatch) -> None:
    context = GitLabBrowserAuthContext(
        auth_url="https://gitlab.company.com/oauth/authorize?x=1",
        redirect_uri="http://127.0.0.1:8080/callback",
        state="state123",
        code_verifier="verifier",
        instance_url="https://gitlab.company.com",
        client_id="client_123",
    )
    monkeypatch.setattr(
        "ripperdoc.core.oauth.gitlab._exchange_authorization_code",
        lambda **kwargs: {  # noqa: ARG005
            "access_token": "access-123",
            "refresh_token": "refresh-123",
            "expires_in": 3600,
        },
    )

    token = complete_gitlab_browser_auth_from_callback_url(
        context,
        "http://127.0.0.1:8080/callback?code=abc&state=state123",
    )

    assert token.type == OAuthTokenType.GITLAB
    assert token.access_token == "access-123"
    assert token.refresh_token == "refresh-123"
