"""Tests for OAuth token storage helpers."""

from __future__ import annotations

import pytest

from ripperdoc.core.oauth import (
    OAuthToken,
    OAuthTokenType,
    add_oauth_token,
    delete_oauth_token,
    get_oauth_token,
    list_oauth_tokens,
)


def test_oauth_token_store_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr("ripperdoc.core.oauth.Path.home", lambda: tmp_path)

    token = OAuthToken(
        type=OAuthTokenType.CODEX,
        access_token="abcd1234efgh5678",
        refresh_token="refresh123",
        expires_at=1234567890,
        account_id="acct_123",
    )
    add_oauth_token("codex-main", token)

    loaded = list_oauth_tokens()
    assert "codex-main" in loaded
    assert loaded["codex-main"].type == OAuthTokenType.CODEX
    assert loaded["codex-main"].refresh_token == "refresh123"
    assert get_oauth_token("codex-main") is not None

    delete_oauth_token("codex-main")
    assert get_oauth_token("codex-main") is None


def test_delete_missing_oauth_token_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("ripperdoc.core.oauth.Path.home", lambda: tmp_path)
    with pytest.raises(KeyError):
        delete_oauth_token("missing")
