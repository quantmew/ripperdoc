"""Textual app for managing OAuth tokens."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

from rich import box
from rich.panel import Panel
from rich.table import Table

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Select,
    Static,
)

from ripperdoc.core.oauth import (
    OAuthToken,
    OAuthTokenType,
    add_oauth_token,
    delete_oauth_token,
    get_oauth_tokens_path,
    list_oauth_tokens,
)
from ripperdoc.core.oauth_codex import (
    CodexOAuthError,
    complete_codex_browser_auth_from_callback_url,
    login_codex_with_device_code,
    start_codex_browser_auth,
)


def _mask_secret(value: Optional[str]) -> str:
    if not value:
        return "unset"
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}…{value[-2:]}"


@dataclass
class OAuthFormResult:
    name: str
    token: OAuthToken


@dataclass
class OAuthLoginResult:
    token_name: str
    mode: str


class ConfirmScreen(ModalScreen[bool]):
    """Simple confirmation dialog."""

    def __init__(self, message: str) -> None:
        super().__init__()
        self._message = message

    def compose(self) -> ComposeResult:
        with Container(id="confirm_dialog"):
            yield Static(self._message, id="confirm_message")
            with Horizontal(id="confirm_buttons"):
                yield Button("Yes", id="confirm_yes", variant="primary")
                yield Button("No", id="confirm_no")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "confirm_yes":
            self.dismiss(True)
        else:
            self.dismiss(False)


class OAuthFormScreen(ModalScreen[Optional[OAuthFormResult]]):
    """Modal form for add/edit token."""

    def __init__(
        self,
        mode: str,
        *,
        existing_name: Optional[str] = None,
        existing_token: Optional[OAuthToken] = None,
    ) -> None:
        super().__init__()
        self._mode = mode
        self._existing_name = existing_name
        self._existing_token = existing_token

    def compose(self) -> ComposeResult:
        title = "Add OAuth token" if self._mode == "add" else "Edit OAuth token"
        with Container(id="form_dialog"):
            yield Static(title, id="form_title")
            with VerticalScroll(id="form_fields"):
                if self._mode == "add":
                    yield Static("Token name", classes="field_label")
                    yield Input(placeholder="Token name", id="name_input")
                else:
                    yield Static("Token name", classes="field_label")
                    yield Static(self._existing_name or "", classes="field_value")

                token_type_default = (
                    self._existing_token.type.value
                    if self._existing_token
                    else OAuthTokenType.CODEX.value
                )
                token_type_options = [(x.value, x.value) for x in OAuthTokenType]
                yield Static("Token type", classes="field_label")
                yield Select(token_type_options, value=token_type_default, id="token_type_select")

                access_hint = "[set]" if (self._existing_token and self._existing_token.access_token) else "[not set]"
                yield Static("Access token", classes="field_label")
                yield Input(
                    placeholder=f"Access token {access_hint} (blank=keep)",
                    password=True,
                    id="access_input",
                )

                refresh_hint = (
                    "[set]" if (self._existing_token and self._existing_token.refresh_token) else "[not set]"
                )
                yield Static("Refresh token", classes="field_label")
                yield Input(
                    placeholder=f"Refresh token {refresh_hint} (blank=keep, '-'=clear)",
                    password=True,
                    id="refresh_input",
                )

                expires_default = (
                    str(self._existing_token.expires_at)
                    if self._existing_token and self._existing_token.expires_at is not None
                    else ""
                )
                yield Static("Expires at (unix ms)", classes="field_label")
                yield Input(
                    value=expires_default,
                    placeholder="Optional, blank=keep, '-'=clear",
                    id="expires_input",
                )

                account_default = self._existing_token.account_id if self._existing_token else ""
                yield Static("Account ID", classes="field_label")
                yield Input(
                    value=account_default or "",
                    placeholder="Optional, blank=keep, '-'=clear",
                    id="account_input",
                )

            with Horizontal(id="form_buttons"):
                yield Button("Save", id="form_save", variant="primary")
                yield Button("Cancel", id="form_cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "form_cancel":
            self.dismiss(None)
            return
        if event.button.id != "form_save":
            return

        name_input = self.query_one("#name_input", Input) if self._mode == "add" else None
        type_select = self.query_one("#token_type_select", Select)
        access_input = self.query_one("#access_input", Input)
        refresh_input = self.query_one("#refresh_input", Input)
        expires_input = self.query_one("#expires_input", Input)
        account_input = self.query_one("#account_input", Input)

        name = self._existing_name or ""
        if self._mode == "add":
            name = (name_input.value or "").strip() if name_input else ""
            if not name:
                self._set_error("Token name is required.")
                return

        type_raw = type_select.value
        type_value = type_raw.strip() if isinstance(type_raw, str) else OAuthTokenType.CODEX.value
        try:
            token_type = OAuthTokenType(type_value)
        except ValueError:
            self._set_error("Invalid token type.")
            return

        access_raw = (access_input.value or "").strip()
        if self._existing_token and not access_raw:
            access_token = self._existing_token.access_token
        else:
            access_token = access_raw
        if not access_token:
            self._set_error("Access token is required.")
            return

        refresh_raw = (refresh_input.value or "").strip()
        if self._existing_token:
            if refresh_raw == "-":
                refresh_token = None
            elif refresh_raw:
                refresh_token = refresh_raw
            else:
                refresh_token = self._existing_token.refresh_token
        else:
            refresh_token = refresh_raw or None

        expires_raw = (expires_input.value or "").strip()
        try:
            if self._existing_token and not expires_raw:
                expires_at = self._existing_token.expires_at
            elif expires_raw in {"-", "clear"}:
                expires_at = None
            elif expires_raw:
                expires_at = int(expires_raw)
            else:
                expires_at = None
        except ValueError:
            self._set_error("Expires at must be an integer.")
            return

        account_raw = (account_input.value or "").strip()
        if self._existing_token:
            if account_raw in {"-", "clear"}:
                account_id = None
            elif account_raw:
                account_id = account_raw
            else:
                account_id = self._existing_token.account_id
        else:
            account_id = account_raw or None

        self.dismiss(
            OAuthFormResult(
                name=name,
                token=OAuthToken(
                    type=token_type,
                    access_token=access_token,
                    refresh_token=refresh_token,
                    expires_at=expires_at,
                    account_id=account_id,
                ),
            )
        )

    def _set_error(self, message: str) -> None:
        app = getattr(self, "app", None)
        if app:
            app.notify(message, title="Validation error", severity="error", timeout=6)


class OAuthLoginScreen(ModalScreen[Optional[OAuthLoginResult]]):
    """Modal form for Codex OAuth login."""

    def compose(self) -> ComposeResult:
        with Container(id="form_dialog"):
            yield Static("Codex OAuth Login", id="form_title")
            with VerticalScroll(id="form_fields"):
                yield Static("Token name", classes="field_label")
                yield Input(value="codex", placeholder="Token name", id="login_name_input")
                yield Static("Mode", classes="field_label")
                yield Select(
                    [("browser", "browser"), ("device", "device")],
                    value="browser",
                    id="login_mode_select",
                )
            with Horizontal(id="form_buttons"):
                yield Button("Login", id="login_submit", variant="primary")
                yield Button("Cancel", id="login_cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "login_cancel":
            self.dismiss(None)
            return
        if event.button.id != "login_submit":
            return

        name = (self.query_one("#login_name_input", Input).value or "").strip()
        mode_raw = self.query_one("#login_mode_select", Select).value
        mode = mode_raw.strip().lower() if isinstance(mode_raw, str) else "browser"
        if not name:
            self._set_error("Token name is required.")
            return
        if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
            self._set_error("Token name must match [A-Za-z0-9._-]+ (no spaces).")
            return
        if mode not in {"browser", "device"}:
            self._set_error("Login mode must be browser or device.")
            return
        self.dismiss(OAuthLoginResult(token_name=name, mode=mode))

    def _set_error(self, message: str) -> None:
        app = getattr(self, "app", None)
        if app:
            app.notify(message, title="Validation error", severity="error", timeout=6)


class OAuthCallbackUrlScreen(ModalScreen[Optional[str]]):
    """Modal prompt for manual OAuth callback URL completion."""

    def __init__(self, auth_url: str) -> None:
        super().__init__()
        self._auth_url = auth_url

    def compose(self) -> ComposeResult:
        with Container(id="form_dialog"):
            yield Static("Browser OAuth Login", id="form_title")
            with VerticalScroll(id="form_fields"):
                yield Static(
                    "1) Open Authorization URL and complete login.\n"
                    "2) Browser will try to open localhost and may fail.\n"
                    "3) Copy the full failed URL from address bar and paste below.",
                    classes="field_label",
                )
                yield Static("Authorization URL", classes="field_label")
                yield Static(self._auth_url, classes="field_value")
                yield Static("Callback URL", classes="field_label")
                yield Input(
                    placeholder="http://localhost:1455/auth/callback?code=...&state=...",
                    id="callback_url_input",
                )
            with Horizontal(id="form_buttons"):
                yield Button("Complete", id="callback_submit", variant="primary")
                yield Button("Cancel", id="callback_cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "callback_cancel":
            self.dismiss(None)
            return
        if event.button.id != "callback_submit":
            return
        callback_url = (self.query_one("#callback_url_input", Input).value or "").strip()
        if not callback_url:
            self._set_error("Callback URL is required.")
            return
        self.dismiss(callback_url)

    def _set_error(self, message: str) -> None:
        app = getattr(self, "app", None)
        if app:
            app.notify(message, title="Validation error", severity="error", timeout=6)


class OAuthApp(App[None]):
    """Textual OAuth manager app."""

    CSS = """
    #status_bar {
        height: 1;
        color: $text-muted;
        padding: 0 1;
    }

    #body {
        layout: horizontal;
        height: 1fr;
    }

    #tokens_table {
        width: 48%;
        min-width: 42;
    }

    #details_panel {
        width: 52%;
        padding: 0 1;
    }

    #form_dialog, #confirm_dialog {
        width: 76;
        max-height: 90%;
        background: $panel;
        border: round $accent;
        padding: 1 2;
    }

    #form_title {
        text-style: bold;
        padding: 0 0 1 0;
    }

    #form_fields Input, #form_fields Select {
        margin: 0 0 1 0;
    }

    #form_fields {
        height: 1fr;
        overflow: auto;
    }

    .field_label {
        color: $text-muted;
        padding: 0 0 0 0;
    }

    .field_value {
        color: $accent;
        text-style: bold;
        padding: 0 0 1 0;
    }

    #form_buttons, #confirm_buttons {
        align-horizontal: right;
        padding-top: 1;
        height: auto;
    }

    #confirm_message {
        padding: 0 0 1 0;
    }
    """

    BINDINGS = [
        ("escape", "quit", "Quit"),
        ("a", "add", "Add"),
        ("e", "edit", "Edit"),
        ("d", "delete", "Delete"),
        ("l", "login", "Login"),
        ("r", "refresh", "Refresh"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._selected_name: Optional[str] = None
        self._row_names: list[str] = []
        self._login_in_progress = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Static("", id="status_bar")
        with Container(id="body"):
            yield DataTable(id="tokens_table")
            yield Static(id="details_panel")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#tokens_table", DataTable)
        table.add_columns("#", "Name", "Type", "Access", "Refresh")
        try:
            table.cursor_type = "row"
            table.zebra_stripes = True
        except Exception:
            pass
        self._refresh_tokens(select_first=True)

    def action_refresh(self) -> None:
        self._refresh_tokens(select_first=False)
        self._set_status("Refreshed.")

    def action_add(self) -> None:
        self.push_screen(OAuthFormScreen("add"), self._handle_add_result)

    def action_edit(self) -> None:
        token = self._selected_token()
        if not token or not self._selected_name:
            self._set_status("No token selected.")
            return
        self.push_screen(
            OAuthFormScreen(
                "edit",
                existing_name=self._selected_name,
                existing_token=token,
            ),
            self._handle_edit_result,
        )

    def action_delete(self) -> None:
        if not self._selected_name:
            self._set_status("No token selected.")
            return
        self.push_screen(
            ConfirmScreen(f"Delete OAuth token '{self._selected_name}'?"),
            self._handle_delete_confirm,
        )

    def action_login(self) -> None:
        if self._login_in_progress:
            self._set_status("Login already in progress.")
            return
        self.push_screen(OAuthLoginScreen(), self._handle_login_result)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        self._select_by_index(int(event.cursor_row))

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        self._select_by_index(int(event.cursor_row))
        self.action_edit()

    def _handle_add_result(self, result: Optional[OAuthFormResult]) -> None:
        if not result:
            return
        try:
            add_oauth_token(result.name, result.token, overwrite=False)
        except (ValueError, OSError, IOError, PermissionError) as exc:
            self._set_status(str(exc))
            return
        self._selected_name = result.name
        self._set_status(f"Saved {result.name}.")
        self._refresh_tokens(select_first=False)

    def _handle_edit_result(self, result: Optional[OAuthFormResult]) -> None:
        if not result:
            return
        try:
            add_oauth_token(result.name, result.token, overwrite=True)
        except (ValueError, OSError, IOError, PermissionError) as exc:
            self._set_status(str(exc))
            return
        self._selected_name = result.name
        self._set_status(f"Updated {result.name}.")
        self._refresh_tokens(select_first=False)

    def _handle_delete_confirm(self, confirmed: Optional[bool]) -> None:
        if not confirmed or not self._selected_name:
            return
        try:
            delete_oauth_token(self._selected_name)
        except (KeyError, OSError, IOError, PermissionError) as exc:
            self._set_status(str(exc))
            return
        self._set_status(f"Deleted {self._selected_name}.")
        self._selected_name = None
        self._refresh_tokens(select_first=True)

    def _handle_login_result(self, result: Optional[OAuthLoginResult]) -> None:
        if not result:
            return
        self._login_in_progress = True
        self._set_status(f"Starting {result.mode} login for token '{result.token_name}'...")
        self.run_worker(
            self._run_login_flow(result),
            exclusive=True,
            group="oauth_login",
        )

    async def _run_login_flow(self, result: OAuthLoginResult) -> None:
        def _notify(message: str) -> None:
            self.call_from_thread(self._set_status, message)

        try:
            if result.mode == "device":
                token, code = await asyncio.to_thread(
                    login_codex_with_device_code,
                    notify=_notify,
                )
                self._set_status(f"Device code used: {code}")
            else:
                context = await asyncio.to_thread(
                    start_codex_browser_auth,
                    notify=_notify,
                )
                self._set_status("Paste callback URL to complete login.")
                callback_url = await self._prompt_callback_url(context.auth_url)
                if not callback_url:
                    self._set_status("Login cancelled.")
                    return
                token = await asyncio.to_thread(
                    complete_codex_browser_auth_from_callback_url,
                    context,
                    callback_url,
                )

            await asyncio.to_thread(
                lambda: add_oauth_token(result.token_name, token, overwrite=True)
            )
        except CodexOAuthError as exc:
            self._set_status(f"Login failed: {exc}")
        except (ValueError, OSError, IOError, PermissionError) as exc:
            self._set_status(f"Failed to save token: {exc}")
        else:
            self._selected_name = result.token_name
            self._set_status(f"Logged in and saved {result.token_name}.")
            self._refresh_tokens(select_first=False)
        finally:
            self._login_in_progress = False

    async def _prompt_callback_url(self, auth_url: str) -> Optional[str]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Optional[str]] = loop.create_future()

        def _on_result(value: Optional[str]) -> None:
            if not future.done():
                future.set_result(value)

        self.push_screen(OAuthCallbackUrlScreen(auth_url), _on_result)
        return await future

    def _refresh_tokens(self, select_first: bool) -> None:
        tokens = list_oauth_tokens()
        table = self.query_one("#tokens_table", DataTable)
        table.clear(columns=False)
        self._row_names = []

        for idx, (name, token) in enumerate(tokens.items(), start=1):
            self._row_names.append(name)
            table.add_row(
                str(idx),
                name,
                token.type.value,
                _mask_secret(token.access_token),
                _mask_secret(token.refresh_token),
            )

        if not tokens:
            self._selected_name = None
            self._update_details()
            return

        if self._selected_name and self._selected_name in tokens:
            try:
                row_index = self._row_names.index(self._selected_name)
            except ValueError:
                row_index = 0
            self._move_cursor(table, row_index)
            self._update_details()
            return

        if select_first:
            self._selected_name = next(iter(tokens.keys()))
            self._move_cursor(table, 0)
            self._update_details()

    def _select_by_index(self, row_index: int) -> None:
        if row_index < 0 or row_index >= len(self._row_names):
            return
        self._selected_name = self._row_names[row_index]
        self._update_details()

    def _move_cursor(self, table: DataTable, row_index: int) -> None:
        try:
            table.move_cursor(row=row_index)
        except Exception:
            pass

    def _selected_token(self) -> Optional[OAuthToken]:
        if not self._selected_name:
            return None
        return list_oauth_tokens().get(self._selected_name)

    def _update_details(self) -> None:
        details = self.query_one("#details_panel", Static)
        if not self._selected_name:
            details.update(f"No OAuth token selected.\n\nPath: {get_oauth_tokens_path()}")
            return
        token = self._selected_token()
        if not token:
            details.update(f"No OAuth token selected.\n\nPath: {get_oauth_tokens_path()}")
            return

        table = Table.grid(padding=(0, 2))
        table.add_column(style="cyan", no_wrap=True)
        table.add_column()
        table.add_row("Name", self._selected_name)
        table.add_row("Type", token.type.value)
        table.add_row("Access", _mask_secret(token.access_token))
        table.add_row("Refresh", _mask_secret(token.refresh_token))
        table.add_row(
            "Expires",
            str(token.expires_at) if token.expires_at is not None else "-",
        )
        table.add_row("Account", token.account_id or "-")
        table.add_row("Path", str(get_oauth_tokens_path()))
        table.add_row("Login", "Codex browser/device via key 'L'")
        details.update(Panel(table, title=f"OAuth: {self._selected_name}", box=box.ROUNDED))

    def _set_status(self, message: str) -> None:
        status = self.query_one("#status_bar", Static)
        status.update(message)


def run_oauth_tui(on_exit: Optional[Callable[[], Any]] = None) -> bool:
    """Run the Textual OAuth TUI."""
    app = OAuthApp()
    app.run()
    if on_exit:
        on_exit()
    return True
