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

@pytest.mark.parametrize(
    ("token_name", "token_type"),
    [
        ("codex-main", OAuthTokenType.CODEX),
        ("copilot-main", OAuthTokenType.COPILOT),
        ("gitlab-main", OAuthTokenType.GITLAB),
    ],
)
def test_oauth_token_store_roundtrip(tmp_path, monkeypatch, token_name, token_type):
    monkeypatch.setattr("ripperdoc.core.oauth.Path.home", lambda: tmp_path)

    token = OAuthToken(
        type=token_type,
        access_token="abcd1234efgh5678",
        refresh_token="refresh123",
        expires_at=1234567890,
        account_id="acct_123",
    )
    add_oauth_token(token_name, token)

    loaded = list_oauth_tokens()
    assert token_name in loaded
    assert loaded[token_name].type == token_type
    assert loaded[token_name].refresh_token == "refresh123"
    assert get_oauth_token(token_name) is not None

    delete_oauth_token(token_name)
    assert get_oauth_token(token_name) is None


def test_delete_missing_oauth_token_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("ripperdoc.core.oauth.Path.home", lambda: tmp_path)
    with pytest.raises(KeyError):
        delete_oauth_token("missing")
