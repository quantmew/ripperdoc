"""Textual app for managing custom commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

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
    TextArea,
)

from ripperdoc.core.custom_commands import (
    CommandLocation,
    CustomCommandDefinition,
    command_name_to_path,
    load_all_custom_commands,
    load_custom_commands_by_scope,
    normalize_command_name,
    parse_command_markdown,
    render_command_markdown,
    validate_command_name,
)

KNOWN_FRONTMATTER_KEYS = {
    "description",
    "argument-hint",
    "argument_hint",
    "allowed-tools",
    "allowed_tools",
    "model",
    "thinking-mode",
    "thinking_mode",
}


@dataclass
class CommandRow:
    command: CustomCommandDefinition
    active: bool
    shadowed_by: Optional[CommandLocation] = None


@dataclass
class CommandFormResult:
    name: str
    location: CommandLocation
    description: str
    argument_hint: str
    allowed_tools: List[str]
    model: str
    thinking_mode: str
    content: str
    extras: Dict[str, Any]


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


class CommandFormScreen(ModalScreen[Optional[CommandFormResult]]):
    """Modal form for adding/editing commands."""

    def __init__(
        self,
        mode: str,
        *,
        existing: Optional[CommandFormResult] = None,
        default_location: CommandLocation = CommandLocation.PROJECT,
    ) -> None:
        super().__init__()
        self._mode = mode
        self._existing = existing
        self._default_location = default_location
        self._error_text: Optional[str] = None

    def compose(self) -> ComposeResult:
        title = "Add command" if self._mode == "add" else "Edit command"
        with Container(id="form_dialog"):
            yield Static(title, id="form_title")
            with VerticalScroll(id="form_fields"):
                if self._mode == "add":
                    yield Static("Command name", classes="field_label")
                    yield Input(placeholder="e.g., sprint-summary", id="name_input")
                    yield Static("Scope", classes="field_label")
                    scope_options = [
                        ("Project (.ripperdoc/commands)", "project"),
                        ("User (~/.ripperdoc/commands)", "user"),
                    ]
                    default_scope = (
                        "user" if self._default_location == CommandLocation.USER else "project"
                    )
                    yield Select(scope_options, value=default_scope, id="scope_select")
                else:
                    name_display = self._existing.name if self._existing else ""
                    location_display = (
                        self._existing.location.value if self._existing else CommandLocation.PROJECT.value
                    )
                    yield Static("Command name", classes="field_label")
                    yield Static(f"/{name_display}", classes="field_value")
                    yield Static("Scope", classes="field_label")
                    yield Static(location_display, classes="field_value")

                description = self._existing.description if self._existing else ""
                yield Static("Description", classes="field_label")
                yield Input(value=description, placeholder="What this command does", id="desc_input")

                argument_hint = self._existing.argument_hint if self._existing else ""
                yield Static("Argument hint", classes="field_label")
                yield Input(
                    value=argument_hint,
                    placeholder="e.g., <branch> or <file>",
                    id="arg_input",
                )

                tools_value = ", ".join(self._existing.allowed_tools) if self._existing else ""
                yield Static("Allowed tools", classes="field_label")
                yield Input(
                    value=tools_value,
                    placeholder="Comma-separated (blank = no limit)",
                    id="tools_input",
                )

                model_value = self._existing.model if self._existing else ""
                yield Static("Model", classes="field_label")
                yield Input(value=model_value, placeholder="Optional model override", id="model_input")

                thinking_value = self._existing.thinking_mode if self._existing else ""
                yield Static("Thinking mode", classes="field_label")
                yield Input(
                    value=thinking_value,
                    placeholder="think / think hard / ultrathink (optional)",
                    id="thinking_input",
                )

                content_value = self._existing.content if self._existing else ""
                yield Static("Command content", classes="field_label")
                yield TextArea(
                    text=content_value,
                    id="content_input",
                    placeholder="Prompt text sent to the AI",
                )

            with Horizontal(id="form_buttons"):
                yield Button("Save", id="form_save", variant="primary")
                yield Button("Cancel", id="form_cancel")

    def on_mount(self) -> None:
        if self._mode == "add":
            self.query_one("#name_input", Input).focus()
        else:
            self.query_one("#desc_input", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "form_cancel":
            self.dismiss(None)
            return
        if event.button.id != "form_save":
            return
        result = self._build_result()
        if result is None:
            return
        self.dismiss(result)

    def _set_error(self, message: str) -> None:
        app = getattr(self, "app", None)
        if app:
            app.notify(message, title="Validation error", severity="error", timeout=6)

    def _build_result(self) -> Optional[CommandFormResult]:
        if self._mode == "add":
            name_input = (self.query_one("#name_input", Input).value or "").strip()
            error = validate_command_name(name_input)
            if error:
                self._set_error(error)
                return None
            name = normalize_command_name(name_input)
            scope_raw = self.query_one("#scope_select", Select).value or "project"
            location = (
                CommandLocation.USER if scope_raw == "user" else CommandLocation.PROJECT
            )
            extras: Dict[str, Any] = {}
        else:
            if not self._existing:
                self._set_error("Missing command data.")
                return None
            name = self._existing.name
            location = self._existing.location
            extras = self._existing.extras

        description = (self.query_one("#desc_input", Input).value or "").strip()
        argument_hint = (self.query_one("#arg_input", Input).value or "").strip()
        tools_raw = (self.query_one("#tools_input", Input).value or "").strip()
        allowed_tools = _parse_allowed_tools(tools_raw)
        model = (self.query_one("#model_input", Input).value or "").strip()
        thinking_mode = (self.query_one("#thinking_input", Input).value or "").strip()
        content = (self.query_one("#content_input", TextArea).text or "").strip("\n")

        return CommandFormResult(
            name=name,
            location=location,
            description=description,
            argument_hint=argument_hint,
            allowed_tools=allowed_tools,
            model=model,
            thinking_mode=thinking_mode,
            content=content,
            extras=extras,
        )


class CommandsApp(App):
    """Textual application for managing custom commands."""

    CSS = """
    Screen {
        layout: vertical;
    }
    #status_bar {
        padding: 0 1;
        height: 1;
        color: $text-muted;
    }
    #body {
        layout: horizontal;
        height: 1fr;
    }
    #commands_table {
        width: 45%;
        min-width: 36;
    }
    #details_panel {
        width: 1fr;
        padding: 1 2;
    }
    #form_dialog, #confirm_dialog {
        background: $panel;
        border: round $primary;
        padding: 1 2;
        width: 80%;
        max-width: 90;
        height: 85%;
    }
    #confirm_dialog {
        height: auto;
    }
    #form_title, #confirm_message {
        text-style: bold;
        padding: 0 0 1 0;
    }
    #form_fields Input, #form_fields TextArea, #form_fields Select {
        width: 1fr;
    }
    .field_label {
        color: $text-muted;
        padding: 0;
    }
    .field_value {
        color: $text;
        padding: 0 0 1 0;
    }
    #content_input {
        height: 9;
    }
    #form_buttons, #confirm_buttons {
        height: auto;
        align: right middle;
        padding-top: 1;
    }
    """

    BINDINGS = [
        ("escape", "quit", "Quit"),
        ("a", "add", "Add"),
        ("e", "edit", "Edit"),
        ("d", "delete", "Delete"),
        ("r", "refresh", "Refresh"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self, project_path: Path) -> None:
        super().__init__()
        self._project_path = project_path
        self._rows: List[CommandRow] = []
        self._selected_index: Optional[int] = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Static("", id="status_bar")
        with Container(id="body"):
            yield DataTable(id="commands_table")
            yield Static(id="details_panel")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#commands_table", DataTable)
        table.add_columns("#", "Name", "Scope", "Status", "Description")
        try:
            table.cursor_type = "row"
            table.zebra_stripes = True
        except Exception:
            pass
        self._refresh_commands(select_first=True)

    def action_refresh(self) -> None:
        self._refresh_commands(select_first=False)
        self._set_status("Refreshed.")

    def action_add(self) -> None:
        screen = CommandFormScreen("add")
        self.push_screen(screen, self._handle_add_result)

    def action_edit(self) -> None:
        row = self._selected_row()
        if not row:
            self._set_status("No command selected.")
            return
        draft = self._build_form_result(row.command)
        if draft is None:
            return
        screen = CommandFormScreen("edit", existing=draft)
        self.push_screen(screen, self._handle_edit_result)

    def action_delete(self) -> None:
        row = self._selected_row()
        if not row:
            self._set_status("No command selected.")
            return
        screen = ConfirmScreen(f"Delete /{row.command.name} ({row.command.location.value})?")
        self.push_screen(screen, self._handle_delete_confirm)

    def action_quit(self) -> None:
        self.exit()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        self._select_by_index(int(event.cursor_row))

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        self._select_by_index(int(event.cursor_row))
        self.action_edit()

    def _handle_add_result(self, result: Optional[CommandFormResult]) -> None:
        if not result:
            return
        error = self._save_command(result, overwrite=False)
        if error:
            self._set_status(error)
            return
        self._set_status(f"Saved /{result.name}.")
        self._refresh_commands(select_first=False, prefer_name=result.name)

    def _handle_edit_result(self, result: Optional[CommandFormResult]) -> None:
        if not result:
            return
        error = self._save_command(result, overwrite=True)
        if error:
            self._set_status(error)
            return
        self._set_status(f"Updated /{result.name}.")
        self._refresh_commands(select_first=False, prefer_name=result.name)

    def _handle_delete_confirm(self, confirmed: bool) -> None:
        if not confirmed:
            return
        row = self._selected_row()
        if not row:
            self._set_status("No command selected.")
            return
        try:
            row.command.path.unlink()
        except FileNotFoundError:
            self._set_status("Command already removed.")
            return
        except (OSError, IOError, PermissionError) as exc:
            self._set_status(str(exc))
            return
        self._refresh_custom_command_cache()
        self._set_status(f"Deleted /{row.command.name}.")
        self._refresh_commands(select_first=True)

    def _refresh_commands(self, *, select_first: bool, prefer_name: Optional[str] = None) -> None:
        table = self.query_one("#commands_table", DataTable)
        table.clear()
        self._rows = self._load_rows()
        for idx, row in enumerate(self._rows, start=1):
            cmd = row.command
            if row.active:
                status = "active"
            elif row.shadowed_by:
                status = f"shadowed by {row.shadowed_by.value}"
            else:
                status = "inactive"
            table.add_row(str(idx), f"/{cmd.name}", cmd.location.value, status, cmd.description)
        self._selected_index = None
        if not self._rows:
            self._update_details(None)
            return
        if prefer_name:
            for idx, row in enumerate(self._rows):
                if row.command.name == prefer_name:
                    self._select_by_index(idx)
                    return
        if select_first:
            self._select_by_index(0)

    def _select_by_index(self, row_index: int) -> None:
        if row_index < 0 or row_index >= len(self._rows):
            return
        self._selected_index = row_index
        self._update_details(self._rows[row_index])

    def _selected_row(self) -> Optional[CommandRow]:
        if self._selected_index is None:
            return None
        if self._selected_index >= len(self._rows):
            return None
        return self._rows[self._selected_index]

    def _load_rows(self) -> List[CommandRow]:
        catalog = load_custom_commands_by_scope(project_path=self._project_path)
        merged = load_all_custom_commands(project_path=self._project_path)
        active_by_name = {cmd.name: cmd.location for cmd in merged.commands}
        rows: List[CommandRow] = []
        for cmd in catalog.commands:
            active_loc = active_by_name.get(cmd.name)
            active = active_loc == cmd.location
            rows.append(
                CommandRow(
                    command=cmd,
                    active=active,
                    shadowed_by=None if active else active_loc,
                )
            )
        rows.sort(key=lambda r: (r.command.name, _location_rank(r.command.location)))
        return rows

    def _update_details(self, row: Optional[CommandRow]) -> None:
        panel = self.query_one("#details_panel", Static)
        if not row:
            panel.update(Panel("No commands found.", title="Details"))
            return
        cmd = row.command
        info = Table.grid(padding=(0, 1))
        info.add_column(style="bold")
        info.add_column()
        info.add_row("Name", f"/{cmd.name}")
        info.add_row("Scope", cmd.location.value)
        if row.active:
            status_text = "active"
        elif row.shadowed_by:
            status_text = f"shadowed by {row.shadowed_by.value}"
        else:
            status_text = "inactive"
        info.add_row("Status", status_text)
        info.add_row("Path", str(cmd.path))
        if cmd.argument_hint:
            info.add_row("Args", cmd.argument_hint)
        if cmd.allowed_tools:
            info.add_row("Tools", ", ".join(cmd.allowed_tools))
        if cmd.model:
            info.add_row("Model", cmd.model)
        if cmd.thinking_mode:
            info.add_row("Thinking", cmd.thinking_mode)

        description = cmd.description or ""
        content_preview = cmd.content.strip() if cmd.content else "(empty)"
        if len(content_preview) > 420:
            content_preview = content_preview[:420] + "..."

        renderable = Group(
            Panel(info, title="Metadata", border_style="cyan"),
            Panel(Text(description or "(no description)", overflow="fold"), title="Description"),
            Panel(Text(content_preview, overflow="fold"), title="Content"),
        )
        panel.update(renderable)

    def _set_status(self, message: str) -> None:
        self.query_one("#status_bar", Static).update(message)

    def _build_form_result(self, cmd: CustomCommandDefinition) -> Optional[CommandFormResult]:
        try:
            raw_text = cmd.path.read_text(encoding="utf-8")
        except (OSError, IOError, PermissionError) as exc:
            self._set_status(str(exc))
            return None

        frontmatter, body, parse_error = parse_command_markdown(raw_text)
        if parse_error:
            self._set_status(parse_error)

        extras = _extract_frontmatter_extras(frontmatter)
        description = (frontmatter.get("description") or cmd.description or "").strip()
        argument_hint = (
            frontmatter.get("argument-hint")
            or frontmatter.get("argument_hint")
            or cmd.argument_hint
            or ""
        ).strip()
        model = (frontmatter.get("model") or cmd.model or "").strip()
        thinking_mode = (
            frontmatter.get("thinking-mode") or frontmatter.get("thinking_mode") or cmd.thinking_mode or ""
        ).strip()

        return CommandFormResult(
            name=cmd.name,
            location=cmd.location,
            description=description,
            argument_hint=argument_hint,
            allowed_tools=cmd.allowed_tools,
            model=model,
            thinking_mode=thinking_mode,
            content=body,
            extras=extras,
        )

    def _save_command(self, result: CommandFormResult, *, overwrite: bool) -> Optional[str]:
        frontmatter: Dict[str, Any] = {}
        if result.description:
            frontmatter["description"] = result.description
        if result.argument_hint:
            frontmatter["argument-hint"] = result.argument_hint
        if result.allowed_tools:
            frontmatter["allowed-tools"] = result.allowed_tools
        if result.model:
            frontmatter["model"] = result.model
        if result.thinking_mode:
            frontmatter["thinking-mode"] = result.thinking_mode
        frontmatter.update(result.extras)

        base_dir = _command_dir(self._project_path, result.location)
        try:
            path = command_name_to_path(result.name, base_dir)
        except ValueError as exc:
            return str(exc)

        if path.exists() and not overwrite:
            return f"Command already exists: {path}"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(render_command_markdown(frontmatter, result.content), encoding="utf-8")
        except (OSError, IOError, PermissionError) as exc:
            return str(exc)

        self._refresh_custom_command_cache()
        return None

    def _refresh_custom_command_cache(self) -> None:
        try:
            from ripperdoc.cli import commands as commands_registry

            commands_registry.refresh_custom_commands(self._project_path)
        except Exception:
            pass


def _command_dir(project_path: Path, location: CommandLocation) -> Path:
    if location == CommandLocation.USER:
        return Path.home().expanduser() / ".ripperdoc" / "commands"
    return project_path.resolve() / ".ripperdoc" / "commands"


def _location_rank(location: CommandLocation) -> int:
    if location == CommandLocation.PROJECT:
        return 0
    if location == CommandLocation.USER:
        return 1
    return 2


def _parse_allowed_tools(raw: str) -> List[str]:
    raw = raw.strip()
    if not raw:
        return []
    if "," in raw:
        parts = raw.split(",")
    else:
        parts = raw.split()
    return [item.strip() for item in parts if item.strip()]


def _extract_frontmatter_extras(frontmatter: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in frontmatter.items() if k not in KNOWN_FRONTMATTER_KEYS}


def run_commands_tui(project_path: Path, on_exit: Optional[Callable[[], Any]] = None) -> bool:
    """Run the Textual commands TUI."""
    app = CommandsApp(project_path)
    app.run()
    if on_exit:
        on_exit()
    return True
