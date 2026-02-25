"""Tests for /oauth command login paths."""

from __future__ import annotations

from rich.console import Console

from ripperdoc.cli.commands import oauth_cmd
from ripperdoc.core.oauth import OAuthToken, OAuthTokenType
from ripperdoc.core.oauth_codex import CodexBrowserAuthContext, CodexOAuthPendingCallback


class _DummyUI:
    def __init__(self, console: Console):
        self.console = console


def test_oauth_login_subcommand_dispatches_args(monkeypatch) -> None:
    captured: dict[str, str] = {}

    def _fake_run(console, *, token_name: str, mode: str):  # noqa: ANN001
        captured["token_name"] = token_name
        captured["mode"] = mode
        return True

    monkeypatch.setattr(oauth_cmd, "_run_codex_login", _fake_run)
    ui = _DummyUI(Console(record=True, width=120))

    result = oauth_cmd.command.handler(ui, "login codex my-token device")

    assert result is True
    assert captured == {"token_name": "my-token", "mode": "device"}


def test_run_codex_login_saves_token(monkeypatch) -> None:
    saved: dict[str, OAuthToken] = {}

    monkeypatch.setattr(oauth_cmd, "list_oauth_tokens", lambda: {})
    monkeypatch.setattr(
        oauth_cmd,
        "login_codex_with_browser",
        lambda notify=None: OAuthToken(  # noqa: ARG005
            type=OAuthTokenType.CODEX,
            access_token="abcd1234efgh",
            refresh_token="refresh",
            expires_at=123456,
            account_id="acct_1",
        ),
    )
    monkeypatch.setattr(
        oauth_cmd,
        "add_oauth_token",
        lambda name, token, overwrite=True: saved.update({name: token}),  # noqa: ARG005
    )

    console = Console(record=True, width=120)
    result = oauth_cmd._run_codex_login(console, token_name="codex", mode="browser")

    assert result is True
    assert "codex" in saved
    assert saved["codex"].type == OAuthTokenType.CODEX


def test_run_codex_login_manual_callback_flow(monkeypatch) -> None:
    saved: dict[str, OAuthToken] = {}
    context = CodexBrowserAuthContext(
        auth_url="https://auth.openai.com/oauth/authorize?x=1",
        redirect_uri="http://localhost:1455/auth/callback",
        state="state123",
        code_verifier="verifier",
    )

    def _raise_pending(notify=None):  # noqa: ANN001, ARG001
        raise CodexOAuthPendingCallback(context)

    monkeypatch.setattr(oauth_cmd, "list_oauth_tokens", lambda: {})
    monkeypatch.setattr(oauth_cmd, "login_codex_with_browser", _raise_pending)
    monkeypatch.setattr(
        oauth_cmd,
        "complete_codex_browser_auth_from_callback_url",
        lambda ctx, callback_url: OAuthToken(  # noqa: ARG005
            type=OAuthTokenType.CODEX,
            access_token="abcd1234efgh",
            refresh_token="refresh",
            expires_at=123456,
            account_id="acct_1",
        ),
    )
    monkeypatch.setattr(
        oauth_cmd,
        "add_oauth_token",
        lambda name, token, overwrite=True: saved.update({name: token}),  # noqa: ARG005
    )

    console = Console(record=True, width=120)
    monkeypatch.setattr(
        console,
        "input",
        lambda prompt="": "http://localhost:1455/auth/callback?code=abc&state=state123",  # noqa: ARG005
    )

    result = oauth_cmd._run_codex_login(console, token_name="codex", mode="browser")

    assert result is True
    assert "codex" in saved


def test_run_codex_login_rejects_invalid_token_name(monkeypatch) -> None:
    monkeypatch.setattr(oauth_cmd, "list_oauth_tokens", lambda: {})
    console = Console(record=True, width=120)
    result = oauth_cmd._run_codex_login(console, token_name="bad name", mode="browser")
    assert result is True
