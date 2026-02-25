"""Tests for Codex OAuth helpers."""

from __future__ import annotations

import base64
import json
import urllib.parse

from ripperdoc.core.oauth_codex import (
    extract_account_id,
    extract_account_id_from_claims,
    parse_jwt_claims,
    start_codex_browser_auth,
)


def _jwt_with_payload(payload: dict[str, object]) -> str:
    header = base64.urlsafe_b64encode(json.dumps({"alg": "none"}).encode("utf-8")).decode("ascii").rstrip("=")
    body = base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).decode("ascii").rstrip("=")
    return f"{header}.{body}.sig"


def test_parse_jwt_claims_reads_payload() -> None:
    token = _jwt_with_payload({"chatgpt_account_id": "acct_123"})
    claims = parse_jwt_claims(token)
    assert claims is not None
    assert claims.get("chatgpt_account_id") == "acct_123"


def test_extract_account_id_from_claims_variants() -> None:
    assert extract_account_id_from_claims({"chatgpt_account_id": "acct_direct"}) == "acct_direct"
    assert extract_account_id_from_claims(
        {"https://api.openai.com/auth": {"chatgpt_account_id": "acct_nested"}}
    ) == "acct_nested"
    assert extract_account_id_from_claims({"organizations": [{"id": "org_1"}]}) == "org_1"


def test_extract_account_id_from_tokens() -> None:
    access = _jwt_with_payload({"organizations": [{"id": "org_abc"}]})
    assert extract_account_id({"access_token": access}) == "org_abc"


def test_start_codex_browser_auth_builds_context(monkeypatch) -> None:
    opened: list[str] = []
    notices: list[str] = []

    monkeypatch.setattr(
        "ripperdoc.core.oauth_codex.webbrowser.open",
        lambda url, new=2: opened.append(url),  # noqa: ARG005
    )

    context = start_codex_browser_auth(
        callback_port=1455,
        open_browser=True,
        notify=notices.append,
    )

    assert context.redirect_uri == "http://localhost:1455/auth/callback"
    assert context.auth_url.startswith("https://auth.openai.com/oauth/authorize?")
    assert opened and opened[0] == context.auth_url
    assert notices and "authorize Codex" in notices[0]

    parsed = urllib.parse.urlparse(context.auth_url)
    params = urllib.parse.parse_qs(parsed.query)
    assert params.get("state", [None])[0] == context.state
    assert params.get("redirect_uri", [None])[0] == context.redirect_uri
