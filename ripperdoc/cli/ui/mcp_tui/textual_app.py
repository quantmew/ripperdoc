"""Textual app for inspecting and adding MCP servers."""

from __future__ import annotations

import datetime as dt
import json
import shlex
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from textual import events
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, OptionList, Select, Static
from textual.widgets.option_list import Option
from textual.worker import Worker, WorkerState

from ripperdoc.utils.mcp import (
    McpServerInfo,
    get_mcp_stderr_log_path,
    get_mcp_stderr_mode,
    load_mcp_servers_async,
    shutdown_mcp_runtime,
)

_DEFAULT_LOG_LINES = 80
_MAX_TOOLS_PREVIEW = 10
_MAX_RESOURCES_PREVIEW = 10
_PROJECT_SCOPE = "project"
_USER_SCOPE = "user"
_SUPPORTED_TRANSPORTS = {"stdio", "sse", "http", "streamable-http"}
_TOP_LEVEL_SERVERS_KEY = "__top_level_servers__"


@dataclass(frozen=True)
class AddServerDraft:
    name: str
    scope: str
    transport: str
    command: str
    args: list[str]
    url: str
    description: str
    target_path: Optional[Path] = None
    original_name: Optional[str] = None


@dataclass(frozen=True)
class SaveResult:
    path: Path
    updated: bool


def _truncate(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _tail_text(path: Path, line_count: int) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            lines = deque(handle, maxlen=line_count)
    except OSError as exc:
        return f"(failed to read log: {exc})"
    return "".join(lines).rstrip("\n")


def _format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024.0:.1f} KB"
    return f"{size_bytes / (1024.0 * 1024.0):.1f} MB"


def _format_time(timestamp: float) -> str:
    return dt.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def _config_path_for_scope(project_path: Path, scope: str) -> Path:
    if scope == _USER_SCOPE:
        return Path.home().expanduser() / ".ripperdoc" / "mcp.json"
    return project_path / ".ripperdoc" / "mcp.json"


def _scope_for_path(project_path: Path, path: Path) -> str:
    user_paths = {
        (Path.home().expanduser() / ".ripperdoc" / "mcp.json").resolve(),
        (Path.home().expanduser() / ".mcp.json").resolve(),
    }
    return _USER_SCOPE if path.resolve() in user_paths else _PROJECT_SCOPE


def _candidate_edit_paths(project_path: Path) -> list[Path]:
    return [
        (project_path / ".ripperdoc" / "mcp.json").resolve(),
        (project_path / ".mcp.json").resolve(),
        (Path.home().expanduser() / ".ripperdoc" / "mcp.json").resolve(),
        (Path.home().expanduser() / ".mcp.json").resolve(),
    ]


def _serialize_args_for_input(args: list[str]) -> str:
    if not args:
        return ""
    return " ".join(shlex.quote(arg) for arg in args)


def _draft_from_server_entry(
    *,
    name: str,
    scope: str,
    entry: dict[str, Any],
    source_path: Path,
) -> AddServerDraft:
    raw_type = str(entry.get("type") or entry.get("transport") or "").strip().lower()
    command = ""
    args: list[str] = []
    url = str(entry.get("url") or entry.get("uri") or "").strip()
    description = str(entry.get("description") or "")

    raw_command = entry.get("command")
    if isinstance(raw_command, str):
        command = raw_command.strip()
    elif isinstance(raw_command, list):
        tokens = [str(token) for token in raw_command if str(token)]
        if tokens:
            command = tokens[0]
            args.extend(tokens[1:])

    raw_args = entry.get("args")
    if isinstance(raw_args, list):
        args.extend(str(arg) for arg in raw_args)

    transport = raw_type
    if not transport:
        if command:
            transport = "stdio"
        elif url:
            transport = "sse"
        else:
            transport = "stdio"
    if transport not in _SUPPORTED_TRANSPORTS:
        transport = "stdio"

    return AddServerDraft(
        name=name,
        scope=scope,
        transport=transport,
        command=command,
        args=args,
        url=url,
        description=description,
        target_path=source_path,
        original_name=name,
    )


def _find_edit_draft(project_path: Path, server_name: str) -> Optional[AddServerDraft]:
    for candidate in _candidate_edit_paths(project_path):
        if not candidate.exists():
            continue
        try:
            data, servers_key = _load_mcp_json(candidate)
        except ValueError:
            continue
        servers_raw: Any
        if servers_key == _TOP_LEVEL_SERVERS_KEY:
            servers_raw = data
        else:
            servers_raw = data.get(servers_key)
        if not isinstance(servers_raw, dict):
            continue
        entry = servers_raw.get(server_name)
        if not isinstance(entry, dict):
            continue
        scope = _scope_for_path(project_path, candidate)
        return _draft_from_server_entry(
            name=server_name,
            scope=scope,
            entry=entry,
            source_path=candidate,
        )
    return None


def _load_mcp_json(path: Path) -> tuple[dict[str, Any], str]:
    if not path.exists():
        return {}, "servers"

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid MCP config JSON at {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError(f"Invalid MCP config format at {path}: root object must be JSON object")

    if "servers" in raw:
        if not isinstance(raw["servers"], dict):
            raise ValueError(f"Invalid MCP config at {path}: 'servers' must be object")
        return raw, "servers"
    if "mcpServers" in raw:
        if not isinstance(raw["mcpServers"], dict):
            raise ValueError(f"Invalid MCP config at {path}: 'mcpServers' must be object")
        return raw, "mcpServers"

    # Legacy/alternate format: direct top-level map of server_name -> config.
    has_top_level_server_entries = any(
        isinstance(value, dict)
        and any(key in value for key in ("command", "args", "url", "uri", "type", "transport"))
        for value in raw.values()
    )
    if has_top_level_server_entries:
        return raw, _TOP_LEVEL_SERVERS_KEY

    return raw, "servers"


def _save_mcp_server(path: Path, draft: AddServerDraft, *, overwrite: bool) -> SaveResult:
    data, servers_key = _load_mcp_json(path)
    servers: dict[str, Any]
    if servers_key == _TOP_LEVEL_SERVERS_KEY:
        servers = data
    else:
        servers_raw = data.get(servers_key)
        if isinstance(servers_raw, dict):
            servers = servers_raw
        elif servers_raw is None:
            servers = {}
            data[servers_key] = servers
        else:
            raise ValueError(f"Invalid MCP config at {path}: '{servers_key}' must be object")

    existing = draft.name in servers
    if existing and not overwrite:
        raise FileExistsError(f"Server '{draft.name}' already exists in {path}.")

    entry: dict[str, Any] = {}
    if draft.description:
        entry["description"] = draft.description

    if draft.transport == "stdio":
        entry["command"] = draft.command
        if draft.args:
            entry["args"] = draft.args
    else:
        entry["type"] = draft.transport
        entry["url"] = draft.url

    servers[draft.name] = entry
    if servers_key != _TOP_LEVEL_SERVERS_KEY:
        data[servers_key] = servers

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return SaveResult(path=path, updated=existing)


class ConfirmScreen(ModalScreen[bool]):
    """Simple confirmation dialog."""

    def __init__(self, message: str) -> None:
        super().__init__()
        self._message = message

    def compose(self) -> ComposeResult:
        with Container(id="confirm_dialog"):
            yield Static(self._message, id="confirm_message")
            with Horizontal(id="confirm_buttons"):
                yield Button("Overwrite", id="confirm_yes", variant="warning")
                yield Button("Cancel", id="confirm_no")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "confirm_yes")


class AddServerScreen(ModalScreen[Optional[AddServerDraft]]):
    """Modal form for adding an MCP server."""

    def __init__(self, *, mode: str = "add", existing: Optional[AddServerDraft] = None) -> None:
        super().__init__()
        self._mode = mode
        self._existing = existing

    def compose(self) -> ComposeResult:
        title = "Add MCP Server" if self._mode == "add" else "Edit MCP Server"
        submit_label = "Add" if self._mode == "add" else "Save"
        existing = self._existing
        scope_value = existing.scope if existing else _PROJECT_SCOPE
        transport_value = existing.transport if existing else "stdio"
        command_value = existing.command if existing else ""
        args_value = _serialize_args_for_input(existing.args) if existing else ""
        url_value = existing.url if existing else ""
        description_value = existing.description if existing else ""

        with Container(id="add_dialog"):
            yield Static(title, id="add_title")
            yield Static(
                "For stdio: fill command/args. For sse/http/streamable-http: fill url.",
                id="add_hint",
            )
            with VerticalScroll(id="add_fields"):
                if self._mode == "add":
                    yield Static("Name", classes="field_label")
                    yield Input(placeholder="e.g., context7", id="name_input")
                    yield Static("Scope", classes="field_label")
                    yield Select(
                        [
                            ("Project (.ripperdoc/mcp.json)", _PROJECT_SCOPE),
                            ("User (~/.ripperdoc/mcp.json)", _USER_SCOPE),
                        ],
                        value=scope_value,
                        id="scope_select",
                    )
                else:
                    display_name = existing.name if existing else ""
                    display_scope = scope_value
                    yield Static("Name", classes="field_label")
                    yield Static(display_name, classes="field_value")
                    yield Static("Scope", classes="field_label")
                    yield Static(display_scope, classes="field_value")
                yield Static("Transport", classes="field_label")
                yield Select(
                    [
                        ("stdio", "stdio"),
                        ("sse", "sse"),
                        ("http", "http"),
                        ("streamable-http", "streamable-http"),
                    ],
                    value=transport_value,
                    id="transport_select",
                )
                yield Static("Command (stdio)", classes="field_label")
                yield Input(
                    value=command_value,
                    placeholder="e.g., npx -y @upstash/context7-mcp",
                    id="command_input",
                )
                yield Static("Args (optional)", classes="field_label")
                yield Input(
                    value=args_value,
                    placeholder='e.g., --port 8080 "--flag value"',
                    id="args_input",
                )
                yield Static("URL (sse/http)", classes="field_label")
                yield Input(value=url_value, placeholder="e.g., https://example.com/mcp", id="url_input")
                yield Static("Description (optional)", classes="field_label")
                yield Input(value=description_value, placeholder="Optional description", id="description_input")
            with Horizontal(id="add_buttons"):
                yield Button(submit_label, id="add_submit", variant="primary")
                yield Button("Cancel", id="add_cancel")

    def on_mount(self) -> None:
        if self._mode == "add":
            self.query_one("#name_input", Input).focus()
            return
        self.query_one("#command_input", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "add_cancel":
            self.dismiss(None)
            return
        if event.button.id != "add_submit":
            return
        draft, error = self._build_draft()
        if error:
            self._set_error(error)
            return
        self.dismiss(draft)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if not event.input.id:
            return
        draft, error = self._build_draft()
        if error:
            self._set_error(error)
            return
        self.dismiss(draft)

    def _set_error(self, message: str) -> None:
        app = getattr(self, "app", None)
        if app and hasattr(app, "notify"):
            app.notify(message, title="Validation error", severity="error", timeout=6)

    def _build_draft(self) -> tuple[Optional[AddServerDraft], Optional[str]]:
        if self._mode == "edit":
            if not self._existing:
                return None, "Missing existing server metadata."
            name = self._existing.name
            scope = self._existing.scope
            target_path = self._existing.target_path
            original_name = self._existing.original_name or self._existing.name
        else:
            name = (self.query_one("#name_input", Input).value or "").strip()
            if not name:
                return None, "Name is required."
            if " " in name:
                return None, "Name must not contain spaces."

            scope_value = self.query_one("#scope_select", Select).value
            scope = scope_value.strip().lower() if isinstance(scope_value, str) else ""
            if not scope:
                scope = _PROJECT_SCOPE
            if scope.startswith("proj"):
                scope = _PROJECT_SCOPE
            elif scope.startswith(("usr", "global")):
                scope = _USER_SCOPE
            if scope not in {_PROJECT_SCOPE, _USER_SCOPE}:
                return None, "Scope must be 'project' or 'user'."
            target_path = None
            original_name = name

        transport_value = self.query_one("#transport_select", Select).value
        transport = transport_value.strip().lower() if isinstance(transport_value, str) else ""
        if not transport:
            transport = "stdio"
        if transport not in _SUPPORTED_TRANSPORTS:
            allowed = ", ".join(sorted(_SUPPORTED_TRANSPORTS))
            return None, f"Transport must be one of: {allowed}."

        command = (self.query_one("#command_input", Input).value or "").strip()
        args_raw = (self.query_one("#args_input", Input).value or "").strip()
        url = (self.query_one("#url_input", Input).value or "").strip()
        description = (self.query_one("#description_input", Input).value or "").strip()

        args: list[str] = []
        if args_raw:
            try:
                args = shlex.split(args_raw)
            except ValueError as exc:
                return None, f"Args parse error: {exc}"

        if transport == "stdio":
            if not command:
                return None, "Command is required for stdio transport."
        else:
            if not url:
                return None, f"URL is required for {transport} transport."

        return (
            AddServerDraft(
                name=name,
                scope=scope,
                transport=transport,
                command=command,
                args=args,
                url=url,
                description=description,
                target_path=target_path,
                original_name=original_name,
            ),
            None,
        )


class McpApp(App[None]):
    CSS = """
    #title {
        text-style: bold;
        padding: 1 1 0 1;
    }

    #description {
        color: $text-muted;
        padding: 0 1;
    }

    #content {
        height: 1fr;
        margin: 0 1;
    }

    #servers_list {
        width: 42%;
        min-width: 36;
        margin-right: 1;
        height: 1fr;
    }

    #details_scroll {
        width: 58%;
        height: 1fr;
        border: round $surface;
        padding: 1;
    }

    #details_panel {
        width: 1fr;
    }

    #status_line {
        color: $text-muted;
        padding: 0 1;
    }

    #hint {
        color: $text-muted;
        padding: 0 1 1 1;
    }

    #confirm_dialog, #add_dialog {
        width: 84;
        max-height: 90%;
        background: $panel;
        border: round $accent;
        padding: 1 2;
    }

    #add_title {
        text-style: bold;
        padding: 0 0 1 0;
    }

    #add_hint {
        color: $text-muted;
        padding: 0 0 1 0;
    }

    .field_label {
        color: $text-muted;
        padding: 0;
    }

    .field_value {
        color: $text;
        padding: 0 0 1 0;
    }

    #add_fields Input, #add_fields Select {
        width: 1fr;
    }

    #add_buttons, #confirm_buttons {
        align-horizontal: right;
        padding-top: 1;
        height: auto;
    }
    """

    BINDINGS = [
        ("escape", "close", "Close"),
        ("q", "close", "Close"),
        ("r", "refresh", "Refresh"),
        ("a", "add", "Add"),
        ("e", "edit", "Edit"),
        ("l", "toggle_logs", "Logs"),
    ]

    def __init__(self, project_path: Optional[Path], log_lines: int = _DEFAULT_LOG_LINES) -> None:
        super().__init__()
        self._project_path = (project_path or Path.cwd()).resolve()
        self._log_lines = max(1, int(log_lines))
        self._servers: list[McpServerInfo] = []
        self._servers_by_name: dict[str, McpServerInfo] = {}
        self._visible_server_names: list[str] = []
        self._highlighted_server_name: Optional[str] = None
        self._show_logs = False
        self._updating_list = False
        self._refresh_worker: Optional[Worker[list[McpServerInfo]]] = None
        self._pending_add: Optional[AddServerDraft] = None

    def compose(self) -> ComposeResult:
        yield Static("MCP Servers", id="title")
        yield Static(
            "Inspect server status, tool/resource discovery, and stderr logs.",
            id="description",
        )
        with Horizontal(id="content"):
            yield OptionList(id="servers_list", markup=False)
            with VerticalScroll(id="details_scroll"):
                yield Static(id="details_panel")
        yield Static("", id="status_line")
        yield Static(
            "a add | e edit | space/enter/l logs | r refresh | esc/q close",
            id="hint",
        )

    def on_mount(self) -> None:
        self.query_one("#servers_list", OptionList).focus()
        self._start_refresh(reset_runtime=False)

    def on_key(self, event: events.Key) -> None:
        if event.key == "space":
            self.action_toggle_logs()
            event.stop()

    def action_close(self) -> None:
        self.exit()

    def action_refresh(self) -> None:
        self._start_refresh(reset_runtime=False)

    def action_add(self) -> None:
        screen = AddServerScreen()
        self.push_screen(screen, self._handle_add_result)

    def action_edit(self) -> None:
        server = self._selected_server()
        if not server:
            self._set_status("No server selected.")
            return
        draft = _find_edit_draft(self._project_path, server.name)
        if not draft:
            self._set_status(
                "This server is not in editable project/user MCP config (it may come from plugin/runtime)."
            )
            return
        screen = AddServerScreen(mode="edit", existing=draft)
        self.push_screen(screen, self._handle_edit_result)

    def action_toggle_logs(self) -> None:
        if not self._selected_server():
            self._set_status("No server selected.")
            return
        self._show_logs = not self._show_logs
        self._update_details(self._selected_server())
        self._set_status("Showing logs." if self._show_logs else "Showing server details.")

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        worker = event.worker
        if worker.name != "mcp_refresh":
            return
        if event.state == WorkerState.ERROR:
            error = worker.error or "Unknown error."
            self._set_status(f"Failed to load MCP servers: {error}")
            return
        if event.state != WorkerState.SUCCESS:
            return

        servers = sorted(worker.result or [], key=lambda s: s.name.lower())
        self._servers = servers
        self._servers_by_name = {server.name: server for server in servers}
        self._refresh_view()
        if not servers:
            self._set_status("No MCP servers configured.")
            return
        connected = len([server for server in servers if server.status == "connected"])
        failed = len([server for server in servers if server.status == "failed"])
        self._set_status(
            f"Loaded {len(servers)} server(s): {connected} connected, {failed} failed."
        )

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if self._updating_list:
            return
        option_id = event.option.id or ""
        if not option_id.startswith("server:"):
            return
        server_name = option_id.split(":", 1)[1]
        if server_name == self._highlighted_server_name:
            return
        self._highlighted_server_name = server_name
        self._update_details(self._selected_server())

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        option_id = event.option.id or ""
        if not option_id.startswith("server:"):
            return
        server_name = option_id.split(":", 1)[1]
        self._highlighted_server_name = server_name
        self.action_toggle_logs()

    async def _load_servers(self, reset_runtime: bool = False) -> list[McpServerInfo]:
        if reset_runtime:
            await shutdown_mcp_runtime()
        return await load_mcp_servers_async(self._project_path, wait_for_connections=False)

    def _start_refresh(self, *, reset_runtime: bool) -> None:
        if self._refresh_worker and self._refresh_worker.state in (
            WorkerState.PENDING,
            WorkerState.RUNNING,
        ):
            self._set_status("Refresh already in progress.")
            return
        self._set_status("Refreshing MCP servers...")
        self._refresh_worker = self.run_worker(
            self._load_servers(reset_runtime=reset_runtime),
            name="mcp_refresh",
            group="mcp",
            exit_on_error=False,
        )

    def _handle_add_result(self, draft: Optional[AddServerDraft]) -> None:
        if not draft:
            return
        path = draft.target_path or _config_path_for_scope(self._project_path, draft.scope)
        try:
            result = _save_mcp_server(path, draft, overwrite=False)
        except FileExistsError:
            self._pending_add = draft
            self.push_screen(
                ConfirmScreen(f"Server '{draft.name}' already exists. Overwrite?"),
                self._handle_add_confirm,
            )
            return
        except (OSError, ValueError, RuntimeError) as exc:
            self._set_status(f"Failed to save MCP server: {exc}")
            return

        action = "Updated" if result.updated else "Added"
        self._set_status(f"{action} MCP server '{draft.name}' in {result.path}.")
        self._show_logs = False
        self._start_refresh(reset_runtime=True)

    def _handle_edit_result(self, draft: Optional[AddServerDraft]) -> None:
        if not draft:
            return
        path = draft.target_path or _config_path_for_scope(self._project_path, draft.scope)
        try:
            result = _save_mcp_server(path, draft, overwrite=True)
        except (OSError, ValueError, RuntimeError, FileExistsError) as exc:
            self._set_status(f"Failed to update MCP server: {exc}")
            return
        self._set_status(f"Updated MCP server '{draft.name}' in {result.path}.")
        self._show_logs = False
        self._start_refresh(reset_runtime=True)

    def _handle_add_confirm(self, confirmed: Optional[bool]) -> None:
        if not confirmed:
            self._pending_add = None
            return
        draft = self._pending_add
        self._pending_add = None
        if not draft:
            return
        path = draft.target_path or _config_path_for_scope(self._project_path, draft.scope)
        try:
            result = _save_mcp_server(path, draft, overwrite=True)
        except (OSError, ValueError, RuntimeError) as exc:
            self._set_status(f"Failed to overwrite MCP server: {exc}")
            return
        action = "Updated" if result.updated else "Added"
        self._set_status(f"{action} MCP server '{draft.name}' in {result.path}.")
        self._show_logs = False
        self._start_refresh(reset_runtime=True)

    def _set_status(self, message: str) -> None:
        self.query_one("#status_line", Static).update(message)

    def _refresh_view(self) -> None:
        self._refresh_list(preserve_highlight=True)
        self._update_details(self._selected_server())

    def _refresh_list(self, preserve_highlight: bool = False) -> None:
        option_list = self.query_one("#servers_list", OptionList)
        previous = self._highlighted_server_name if preserve_highlight else None
        if previous is None and self._visible_server_names:
            idx = option_list.highlighted or 0
            if 0 <= idx < len(self._visible_server_names):
                previous = self._visible_server_names[idx]

        filtered = list(self._servers)
        self._updating_list = True
        option_list.clear_options()
        self._visible_server_names = []
        if not filtered:
            option_list.add_option(Option("No MCP servers configured.", id="empty"))
            option_list.highlighted = 0
            self._highlighted_server_name = None
            self._updating_list = False
            return

        target_highlight = previous if previous in {s.name for s in filtered} else filtered[0].name
        highlighted_idx = 0
        for server in filtered:
            idx = len(self._visible_server_names)
            self._visible_server_names.append(server.name)
            state = server.status or "unknown"
            type_label = server.type or "stdio"
            tools_count = len(server.tools)
            resources_count = len(server.resources)
            prefix = "> " if server.name == target_highlight else "  "
            label = (
                f"{prefix}[{state}] {server.name} ({type_label})"
                f"  tools:{tools_count} resources:{resources_count}"
            )
            option_list.add_option(Option(label, id=f"server:{server.name}"))
            if server.name == target_highlight:
                highlighted_idx = idx
        option_list.highlighted = highlighted_idx
        self._highlighted_server_name = self._visible_server_names[highlighted_idx]
        self._updating_list = False

    def _selected_server(self) -> Optional[McpServerInfo]:
        if not self._highlighted_server_name:
            return None
        return self._servers_by_name.get(self._highlighted_server_name)

    def _update_details(self, server: Optional[McpServerInfo]) -> None:
        panel = self.query_one("#details_panel", Static)
        if not server:
            panel.update(Panel("No MCP server selected.", title="Details", border_style="cyan"))
            return
        renderable = self._build_log_view(server) if self._show_logs else self._build_info_view(server)
        panel.update(renderable)

    def _build_info_view(self, server: McpServerInfo) -> Group:
        meta = Table.grid(padding=(0, 1))
        meta.add_column(style="bold")
        meta.add_column()
        meta.add_row("Name", server.name)
        meta.add_row("Status", server.status or "unknown")
        meta.add_row("Type", server.type or "stdio")
        if server.url:
            meta.add_row("URL", server.url)
        if server.command:
            command = " ".join([server.command, *server.args]) if server.args else server.command
            meta.add_row("Command", _truncate(command, 140))
        if server.server_version:
            meta.add_row("Version", server.server_version)
        if server.env:
            meta.add_row("Env vars", str(len(server.env)))
        if server.headers:
            meta.add_row("Headers", str(len(server.headers)))
        if server.error:
            meta.add_row("Error", _truncate(server.error, 220))

        notes = Table.grid(padding=(0, 1))
        notes.add_column(style="bold")
        notes.add_column()
        description = server.description.strip() if server.description else "(none)"
        instructions = server.instructions.strip() if server.instructions else "(none)"
        notes.add_row("Description", _truncate(description, 220))
        notes.add_row("Instructions", _truncate(instructions, 220))

        tool_lines = ["(none)"]
        if server.tools:
            tool_lines = []
            for idx, tool in enumerate(server.tools):
                if idx >= _MAX_TOOLS_PREVIEW:
                    remaining = len(server.tools) - _MAX_TOOLS_PREVIEW
                    tool_lines.append(f"... and {remaining} more")
                    break
                detail = _truncate(tool.description, 72) if tool.description else ""
                suffix = f" - {detail}" if detail else ""
                tool_lines.append(f"- {tool.name}{suffix}")

        resource_lines = ["(none)"]
        if server.resources:
            resource_lines = []
            for idx, resource in enumerate(server.resources):
                if idx >= _MAX_RESOURCES_PREVIEW:
                    remaining = len(server.resources) - _MAX_RESOURCES_PREVIEW
                    resource_lines.append(f"... and {remaining} more")
                    break
                label = resource.name or resource.uri
                if resource.name:
                    label = f"{resource.name} ({resource.uri})"
                resource_lines.append(f"- {label}")

        return Group(
            Panel(meta, title="Server", border_style="cyan"),
            Panel(notes, title="Notes"),
            Panel(Text("\n".join(tool_lines), overflow="fold"), title=f"Tools ({len(server.tools)})"),
            Panel(
                Text("\n".join(resource_lines), overflow="fold"),
                title=f"Resources ({len(server.resources)})",
            ),
        )

    def _build_log_view(self, server: McpServerInfo) -> Group:
        path = get_mcp_stderr_log_path(self._project_path, server.name)
        mode = get_mcp_stderr_mode()

        details = Table.grid(padding=(0, 1))
        details.add_column(style="bold")
        details.add_column()
        details.add_row("Server", server.name)
        details.add_row("stderr mode", mode)
        details.add_row("Path", str(path))

        if path.exists():
            try:
                stat = path.stat()
            except OSError:
                details.add_row("Status", "unreadable")
                content = "(failed to stat log file)"
            else:
                details.add_row("Status", "ready" if stat.st_size > 0 else "empty")
                details.add_row("Size", _format_size(stat.st_size))
                details.add_row("Updated", _format_time(stat.st_mtime))
                content = _tail_text(path, self._log_lines)
        else:
            details.add_row("Status", "missing")
            content = "(log file does not exist yet)"

        body = Text(content if content else "(log file is empty)", overflow="fold")
        return Group(
            Panel(details, title="Log target", border_style="cyan"),
            Panel(body, title=f"Last {self._log_lines} lines"),
        )


def run_mcp_tui(project_path: Optional[Path]) -> bool:
    """Run the Textual MCP UI."""
    app = McpApp(project_path)
    app.run()
    return True
