"""SDK transport environment compatibility tests."""

from __future__ import annotations

from ripperdoc.protocol.stdio.handler import StdioProtocolHandler


def test_build_sdk_transport_headers_accepts_ripperdoc_session_access_token(monkeypatch) -> None:
    monkeypatch.setenv("RIPPERDOC_SESSION_ACCESS_TOKEN", "token-abc")

    handler = StdioProtocolHandler(default_options={"sdk_url": "wss://example.test/ws/123"})
    handler._session_id = "session-1"

    headers = handler._build_sdk_transport_headers()

    assert headers["Authorization"] == "Bearer token-abc"
    assert headers["x-ripperdoc-session-id"] == "session-1"


def test_build_sdk_transport_headers_ignores_claude_code_session_access_token(monkeypatch) -> None:
    monkeypatch.delenv("RIPPERDOC_SESSION_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("RIPPERDOC_REMOTE_CONTROL_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("RIPPERDOC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("RIPPERDOC_API_KEY", raising=False)
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ACCESS_TOKEN", "token-legacy")

    handler = StdioProtocolHandler(default_options={"sdk_url": "wss://example.test/ws/123"})
    handler._session_id = "session-1"

    headers = handler._build_sdk_transport_headers()

    assert "Authorization" not in headers
    assert "Cookie" not in headers
    assert headers["x-ripperdoc-session-id"] == "session-1"
