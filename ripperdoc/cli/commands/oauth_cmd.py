"""Slash command for managing OAuth token storage."""

from __future__ import annotations

import sys
import re
from typing import Any, Optional

from rich.markup import escape

from ripperdoc.core.oauth import (
    OAuthToken,
    OAuthTokenType,
    add_oauth_token,
    delete_oauth_token,
    get_oauth_tokens_path,
    list_oauth_tokens,
)
from ripperdoc.core.oauth_codex import (
    CodexOAuthPendingCallback,
    CodexOAuthError,
    complete_codex_browser_auth_from_callback_url,
    login_codex_with_browser,
    login_codex_with_device_code,
)
from ripperdoc.utils.prompt import prompt_secret

from .base import SlashCommand


def _mask_secret(value: Optional[str]) -> str:
    if not value:
        return "unset"
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}…{value[-2:]}"


def _parse_optional_int(raw: str) -> Optional[int]:
    value = (raw or "").strip()
    if not value:
        return None
    return int(value)


def _prompt_token_type(console: Any, default: OAuthTokenType = OAuthTokenType.CODEX) -> OAuthTokenType:
    options = ", ".join(token_type.value for token_type in OAuthTokenType)
    raw = (
        console.input(f"Token type ({options}) [{default.value}]: ").strip().lower()
        or default.value
    )
    try:
        return OAuthTokenType(raw)
    except ValueError:
        console.print(
            f"[yellow]Unknown token type '{escape(raw)}', using '{default.value}'.[/yellow]"
        )
        return default


def _collect_add_token_input(
    console: Any,
    *,
    existing: Optional[OAuthToken] = None,
) -> Optional[OAuthToken]:
    token_type = _prompt_token_type(
        console, existing.type if existing else OAuthTokenType.CODEX
    )

    access_prompt = "Access token (required)"
    if existing and existing.access_token:
        access_prompt += " (blank=keep)"
    access_raw = prompt_secret(access_prompt).strip()
    if existing and not access_raw:
        access_token = existing.access_token
    else:
        access_token = access_raw
    if not access_token:
        console.print("[red]Access token is required.[/red]")
        return None

    refresh_prompt = "Refresh token (optional)"
    if existing:
        refresh_prompt += " (blank=keep, '-'=clear)"
    refresh_raw = prompt_secret(refresh_prompt).strip()
    if existing:
        if refresh_raw == "-":
            refresh_token = None
        elif refresh_raw:
            refresh_token = refresh_raw
        else:
            refresh_token = existing.refresh_token
    else:
        refresh_token = refresh_raw or None

    expires_default = str(existing.expires_at) if existing and existing.expires_at else ""
    expires_raw = console.input(
        f"Expires at (unix ms, optional){f' [{expires_default}]' if expires_default else ''}: "
    ).strip()
    try:
        if existing and not expires_raw:
            expires_at = existing.expires_at
        elif expires_raw in {"-", "clear"}:
            expires_at = None
        else:
            expires_at = _parse_optional_int(expires_raw)
    except ValueError:
        console.print("[red]Expires at must be an integer timestamp.[/red]")
        return None

    account_default = existing.account_id if existing else None
    account_raw = console.input(
        f"Account ID (optional){f' [{account_default}]' if account_default else ''}: "
    ).strip()
    if existing:
        if account_raw in {"-", "clear"}:
            account_id = None
        elif account_raw:
            account_id = account_raw
        else:
            account_id = existing.account_id
    else:
        account_id = account_raw or None

    return OAuthToken(
        type=token_type,
        access_token=access_token,
        refresh_token=refresh_token,
        expires_at=expires_at,
        account_id=account_id,
    )


def _render_oauth_plain(console: Any) -> None:
    tokens = list_oauth_tokens()
    if not tokens:
        console.print("  • No OAuth tokens configured")
        console.print(f"[dim]Path: {escape(str(get_oauth_tokens_path()))}[/dim]")
        return

    console.print("\n[bold]Configured OAuth Tokens:[/bold]")
    for name, token in tokens.items():
        console.print(f"  • {escape(name)}", markup=False)
        console.print(f"      type: {token.type.value}", markup=False)
        console.print(f"      access: {_mask_secret(token.access_token)}", markup=False)
        console.print(f"      refresh: {_mask_secret(token.refresh_token)}", markup=False)
        if token.expires_at is not None:
            console.print(f"      expires_at: {token.expires_at}", markup=False)
        if token.account_id:
            console.print(f"      account_id: {token.account_id}", markup=False)
    console.print(f"[dim]Path: {escape(str(get_oauth_tokens_path()))}[/dim]")


def _confirm_action(console: Any, prompt_text: str, *, default: bool = False) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    raw = console.input(f"{prompt_text} {suffix}: ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes")


def _parse_login_args(args: list[str]) -> tuple[str, str, str]:
    provider = "codex"
    token_name = "codex"
    mode = "browser"
    remaining = list(args)

    if remaining and remaining[0].lower() in {"codex"}:
        provider = remaining.pop(0).lower()

    if remaining and remaining[0].lower() in {"browser", "device"}:
        mode = remaining.pop(0).lower()
    elif remaining:
        token_name = remaining.pop(0)

    if remaining and remaining[0].lower() in {"browser", "device"}:
        mode = remaining.pop(0).lower()
    elif remaining and token_name == "codex":
        token_name = remaining.pop(0)

    return provider, token_name, mode


def _run_codex_login(
    console: Any,
    *,
    token_name: str,
    mode: str,
) -> bool:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", token_name or ""):
        console.print(
            "[red]Token name must match [A-Za-z0-9._-]+ (no spaces).[/red]"
        )
        return True
    normalized_mode = (mode or "browser").strip().lower()
    if normalized_mode not in {"browser", "device"}:
        console.print(
            f"[red]Unknown login mode '{escape(normalized_mode)}'. Use browser or device.[/red]"
        )
        return True

    existing = list_oauth_tokens().get(token_name)
    if existing and not _confirm_action(
        console, f"OAuth token '{token_name}' exists. Overwrite with new login?"
    ):
        return True

    try:
        if normalized_mode == "device":
            token, code = login_codex_with_device_code(
                notify=lambda msg: console.print(f"[cyan]{escape(msg)}[/cyan]")
            )
            console.print(f"[dim]Device code used: {escape(code)}[/dim]")
        else:
            token = login_codex_with_browser(
                notify=lambda msg: console.print(f"[cyan]{escape(msg)}[/cyan]")
            )
    except CodexOAuthPendingCallback as pending:
        console.print(
            "[yellow]Local callback timed out. If browser redirected to localhost and failed,"
            " copy that full URL and paste it below.[/yellow]"
        )
        console.print(f"[dim]Auth URL: {escape(pending.context.auth_url)}[/dim]")
        callback_url = console.input("Callback URL (or blank to cancel): ").strip()
        if not callback_url:
            console.print("[yellow]Login cancelled.[/yellow]")
            return True
        try:
            token = complete_codex_browser_auth_from_callback_url(
                pending.context, callback_url
            )
        except CodexOAuthError as exc:
            console.print(
                f"[red]Manual callback completion failed: {escape(str(exc))}[/red]"
            )
            return True
    except CodexOAuthError as exc:
        console.print(f"[red]Codex OAuth login failed: {escape(str(exc))}[/red]")
        return True

    try:
        add_oauth_token(token_name, token, overwrite=True)
    except (ValueError, OSError, IOError, PermissionError) as exc:
        console.print(f"[red]Failed to save OAuth token: {escape(str(exc))}[/red]")
        return True

    account_display = token.account_id or "-"
    console.print(
        f"[green]✓ Logged in and saved token '{escape(token_name)}' "
        f"(type={token.type.value}, account={escape(account_display)})[/green]"
    )
    return True


def _handle_oauth_tui(ui: Any) -> bool:
    console = ui.console
    if not sys.stdin.isatty():
        console.print("[yellow]Interactive UI requires a TTY. Showing plain list instead.[/yellow]")
        _render_oauth_plain(console)
        return True

    try:
        from ripperdoc.cli.ui.oauth_tui import run_oauth_tui
    except (ImportError, ModuleNotFoundError) as exc:
        console.print(
            f"[yellow]Textual OAuth UI not available ({escape(str(exc))}). Showing plain list.[/yellow]"
        )
        _render_oauth_plain(console)
        console.print("[dim]Use /oauth login|add|edit|delete subcommands in this environment.[/dim]")
        return True

    try:
        return bool(run_oauth_tui())
    except Exception as exc:  # noqa: BLE001 - fail safe
        console.print(f"[red]OAuth TUI failed: {escape(str(exc))}[/red]")
        _render_oauth_plain(console)
        return True


def _handle(ui: Any, trimmed_arg: str) -> bool:
    console = ui.console
    tokens = trimmed_arg.split()
    subcmd = tokens[0].lower() if tokens else ""

    def print_usage() -> None:
        console.print("[bold]/oauth[/bold] — open OAuth token TUI")
        console.print("[bold]/oauth tui[/bold] — open OAuth token TUI")
        console.print("[bold]/oauth list[/bold] — list configured OAuth tokens")
        console.print(
            "[bold]/oauth login codex [name] [browser|device][/bold] — login and save token"
        )
        console.print("[bold]/oauth add <name>[/bold] — add an OAuth token")
        console.print("[bold]/oauth edit <name>[/bold] — edit an OAuth token")
        console.print("[bold]/oauth delete <name>[/bold] — delete an OAuth token")

    if subcmd in {"", "tui", "ui"}:
        return _handle_oauth_tui(ui)
    if subcmd in {"help", "-h", "--help"}:
        print_usage()
        return True
    if subcmd in {"list", "ls"}:
        _render_oauth_plain(console)
        return True
    if subcmd in {"login", "auth"}:
        provider, token_name, mode = _parse_login_args(tokens[1:])
        if provider != "codex":
            console.print(
                f"[red]Unsupported OAuth provider '{escape(provider)}'. Only 'codex' is supported currently.[/red]"
            )
            return True
        return _run_codex_login(console, token_name=token_name, mode=mode)
    if subcmd in {"add", "create"}:
        name = tokens[1] if len(tokens) > 1 else console.input("Token name: ").strip()
        if not name:
            console.print("[red]OAuth token name is required.[/red]")
            return True
        existing = list_oauth_tokens().get(name)
        if existing and not _confirm_action(console, f"OAuth token '{name}' exists. Overwrite?"):
            return True
        token = _collect_add_token_input(console, existing=existing)
        if not token:
            return True
        try:
            add_oauth_token(name, token, overwrite=bool(existing))
            console.print(f"[green]✓ OAuth token '{escape(name)}' saved[/green]")
        except (ValueError, OSError, IOError, PermissionError) as exc:
            console.print(f"[red]Failed to save token: {escape(str(exc))}[/red]")
        return True
    if subcmd in {"edit", "update"}:
        name = tokens[1] if len(tokens) > 1 else console.input("Token to edit: ").strip()
        existing = list_oauth_tokens().get(name or "")
        if not name or not existing:
            console.print("[red]OAuth token not found.[/red]")
            return True
        token = _collect_add_token_input(console, existing=existing)
        if not token:
            return True
        try:
            add_oauth_token(name, token, overwrite=True)
            console.print(f"[green]✓ OAuth token '{escape(name)}' updated[/green]")
        except (ValueError, OSError, IOError, PermissionError) as exc:
            console.print(f"[red]Failed to update token: {escape(str(exc))}[/red]")
        return True
    if subcmd in {"delete", "del", "remove"}:
        name = tokens[1] if len(tokens) > 1 else console.input("Token to delete: ").strip()
        if not name:
            console.print("[red]OAuth token name is required.[/red]")
            return True
        if not _confirm_action(console, f"Delete OAuth token '{name}'?"):
            return True
        try:
            delete_oauth_token(name)
            console.print(f"[green]✓ Deleted OAuth token '{escape(name)}'[/green]")
        except KeyError as exc:
            console.print(f"[yellow]{escape(str(exc))}[/yellow]")
        except (OSError, IOError, PermissionError) as exc:
            console.print(f"[red]Failed to delete token: {escape(str(exc))}[/red]")
        return True

    print_usage()
    _render_oauth_plain(console)
    return True


command = SlashCommand(
    name="oauth",
    description="Manage OAuth tokens for OAuth-backed model providers",
    handler=_handle,
)


__all__ = ["command", "_run_codex_login"]
