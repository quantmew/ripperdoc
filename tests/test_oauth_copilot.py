"""Tests for Copilot OAuth helpers."""

from __future__ import annotations

import httpx

from ripperdoc.core.oauth import OAuthTokenType
from ripperdoc.core.oauth.copilot import login_copilot_with_device_code


def test_login_copilot_with_device_code_success(monkeypatch) -> None:
    calls = {"count": 0}

    class _FakeClient:
        def __init__(self, timeout: float = 0.0) -> None:  # noqa: ARG002
            return

        def __enter__(self) -> "_FakeClient":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
            return False

        def post(self, url: str, *, headers: dict, json: dict) -> httpx.Response:  # noqa: A002, ARG002
            calls["count"] += 1
            if calls["count"] == 1:
                return httpx.Response(
                    status_code=200,
                    request=httpx.Request("POST", url),
                    json={
                        "verification_uri": "https://github.com/login/device",
                        "user_code": "ABCD-1234",
                        "device_code": "device_123",
                        "interval": 0,
                    },
                )
            if calls["count"] == 2:
                return httpx.Response(
                    status_code=200,
                    request=httpx.Request("POST", url),
                    json={"error": "authorization_pending"},
                )
            return httpx.Response(
                status_code=200,
                request=httpx.Request("POST", url),
                json={"access_token": "copilot_token_123"},
            )

    monkeypatch.setattr("ripperdoc.core.oauth.copilot.httpx.Client", _FakeClient)
    monkeypatch.setattr("ripperdoc.core.oauth.copilot.time.sleep", lambda _seconds: None)
    monkeypatch.setattr(
        "ripperdoc.core.oauth.copilot.webbrowser.open",
        lambda _url, new=2: True,  # noqa: ARG005
    )

    token, code = login_copilot_with_device_code(
        github_domain="ghe.company.com",
        timeout_sec=5,
    )

    assert code == "ABCD-1234"
    assert token.type == OAuthTokenType.COPILOT
    assert token.access_token == "copilot_token_123"
    assert token.account_id == "ghe.company.com"
