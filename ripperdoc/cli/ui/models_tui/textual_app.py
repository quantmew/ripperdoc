"""Textual app for managing model profiles."""

from __future__ import annotations

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
    Checkbox,
    DataTable,
    Footer,
    Header,
    Input,
    Select,
    Static,
)

from ripperdoc.core.config import (
    ModelProfile,
    ProtocolType,
    add_model_profile,
    delete_model_profile,
    get_global_config,
    model_supports_vision,
    set_model_pointer,
)
from ripperdoc.cli.ui.helpers import get_profile_for_pointer


@dataclass
class ModelFormResult:
    name: str
    profile: ModelProfile
    set_as_main: bool = False
    set_as_quick: bool = False


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


class ModelFormScreen(ModalScreen[Optional[ModelFormResult]]):
    """Modal form for adding/editing models."""

    def __init__(
        self,
        mode: str,
        *,
        existing_name: Optional[str] = None,
        existing_profile: Optional[ModelProfile] = None,
        default_set_main: bool = False,
        default_set_quick: bool = False,
    ) -> None:
        super().__init__()
        self._mode = mode
        self._existing_name = existing_name
        self._existing_profile = existing_profile
        self._default_set_main = default_set_main
        self._default_set_quick = default_set_quick
        self._error_text: Optional[str] = None

    def compose(self) -> ComposeResult:
        title = "Add model" if self._mode == "add" else "Edit model"
        with Container(id="form_dialog"):
            yield Static(title, id="form_title")
            with VerticalScroll(id="form_fields"):
                if self._mode == "add":
                    yield Static("Profile name", classes="field_label")
                    yield Input(placeholder="Profile name", id="name_input")
                else:
                    name_display = self._existing_name or ""
                    yield Static("Profile name", classes="field_label")
                    yield Static(f"{name_display}", id="name_static", classes="field_value")

                current_profile = get_profile_for_pointer("main")
                protocol_default = (
                    self._existing_profile.protocol.value
                    if self._existing_profile
                    else (
                        current_profile.protocol.value
                        if current_profile
                        else ProtocolType.ANTHROPIC.value
                    )
                )
                provider_options = [(p.value, p.value) for p in ProtocolType]
                yield Static("Protocol", classes="field_label")
                yield Select(provider_options, value=protocol_default, id="provider_select")

                model_default = (
                    self._existing_profile.model if self._existing_profile else ""
                )
                yield Static("Model name", classes="field_label")
                yield Input(value=model_default, placeholder="Model name", id="model_input")

                api_key_placeholder = "[set]" if (self._existing_profile and self._existing_profile.api_key) else "[not set]"
                yield Static("API key", classes="field_label")
                yield Input(
                    placeholder=f"API key {api_key_placeholder} (blank=keep, '-'=clear)",
                    password=True,
                    id="api_key_input",
                )

                auth_placeholder = "[set]" if (self._existing_profile and self._existing_profile.auth_token) else "[not set]"
                yield Static("Auth token (Anthropic only)", classes="field_label")
                yield Input(
                    placeholder=f"Auth token (Anthropic only) {auth_placeholder} (blank=keep, '-'=clear)",
                    password=True,
                    id="auth_token_input",
                )

                api_base_default = self._existing_profile.api_base if self._existing_profile else ""
                yield Static("API base", classes="field_label")
                yield Input(
                    value=api_base_default or "",
                    placeholder="API base (optional)",
                    id="api_base_input",
                )

                max_input_tokens_default = (
                    self._existing_profile.max_input_tokens
                    if self._existing_profile and self._existing_profile.max_input_tokens is not None
                    else ""
                )
                yield Static("Max input tokens", classes="field_label")
                yield Input(
                    value=str(max_input_tokens_default),
                    placeholder="Max input tokens (optional)",
                    id="max_input_tokens_input",
                )

                max_output_tokens_default = (
                    self._existing_profile.max_output_tokens
                    if self._existing_profile and self._existing_profile.max_output_tokens is not None
                    else ""
                )
                yield Static("Max output tokens", classes="field_label")
                yield Input(
                    value=str(max_output_tokens_default),
                    placeholder="Max output tokens (optional)",
                    id="max_output_tokens_input",
                )

                max_tokens_default = self._existing_profile.max_tokens if self._existing_profile else 4096
                yield Static("Max tokens", classes="field_label")
                yield Input(
                    value=str(max_tokens_default),
                    placeholder="Max tokens",
                    id="max_tokens_input",
                )

                temp_default = self._existing_profile.temperature if self._existing_profile else 1.0
                yield Static("Temperature", classes="field_label")
                yield Input(
                    value=str(temp_default),
                    placeholder="Temperature",
                    id="temperature_input",
                )

                input_price_default = (
                    self._existing_profile.price.input if self._existing_profile else 0.0
                )
                output_price_default = (
                    self._existing_profile.price.output if self._existing_profile else 0.0
                )
                currency_default = (
                    (self._existing_profile.currency if self._existing_profile else "USD") or "USD"
                )
                yield Static("Input price / 1M tokens", classes="field_label")
                yield Input(
                    value=str(input_price_default),
                    placeholder="Input price per 1M tokens",
                    id="input_price_input",
                )
                yield Static("Output price / 1M tokens", classes="field_label")
                yield Input(
                    value=str(output_price_default),
                    placeholder="Output price per 1M tokens",
                    id="output_price_input",
                )
                yield Static("Currency", classes="field_label")
                yield Input(
                    value=currency_default,
                    placeholder="Currency (e.g. USD)",
                    id="currency_input",
                )

                yield Static("Supports vision", classes="field_label")
                supports_default = (
                    "auto"
                    if not self._existing_profile or self._existing_profile.supports_vision is None
                    else ("yes" if self._existing_profile.supports_vision else "no")
                )
                supports_options = [
                    ("auto (detect)", "auto"),
                    ("yes (image input)", "yes"),
                    ("no (text-only)", "no"),
                ]
                yield Select(supports_options, value=supports_default, id="vision_select")

                set_main_value = self._default_set_main
                set_quick_value = self._default_set_quick
                if self._mode == "edit" and self._existing_name:
                    config = get_global_config()
                    set_main_value = getattr(config.model_pointers, "main", "") == self._existing_name
                    set_quick_value = (
                        getattr(config.model_pointers, "quick", "") == self._existing_name
                    )

                yield Static("Set as main", classes="field_label")
                yield Checkbox("Set as main", value=set_main_value, id="set_main")
                yield Static("Set as quick", classes="field_label")
                yield Checkbox("Set as quick", value=set_quick_value, id="set_quick")

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
        provider_select = self.query_one("#provider_select", Select)
        model_input = self.query_one("#model_input", Input)
        api_key_input = self.query_one("#api_key_input", Input)
        auth_token_input = self.query_one("#auth_token_input", Input)
        api_base_input = self.query_one("#api_base_input", Input)
        max_input_tokens_input = self.query_one("#max_input_tokens_input", Input)
        max_output_tokens_input = self.query_one("#max_output_tokens_input", Input)
        max_tokens_input = self.query_one("#max_tokens_input", Input)
        temperature_input = self.query_one("#temperature_input", Input)
        input_price_input = self.query_one("#input_price_input", Input)
        output_price_input = self.query_one("#output_price_input", Input)
        currency_input = self.query_one("#currency_input", Input)
        vision_select = self.query_one("#vision_select", Select)

        name = self._existing_name or ""
        if self._mode == "add":
            name = (name_input.value or "").strip() if name_input else ""
            if not name:
                self._set_error("Profile name is required.")
                return

        protocol_raw = provider_select.value
        protocol_value = protocol_raw.strip() if isinstance(protocol_raw, str) else ""
        try:
            protocol = ProtocolType(protocol_value)
        except ValueError:
            self._set_error("Invalid protocol.")
            return

        model_name = (model_input.value or "").strip()
        if not model_name:
            self._set_error("Model name is required.")
            return
        inferred_profile = ModelProfile(protocol=protocol, model=model_name)

        api_key_raw = (api_key_input.value or "").strip()
        if self._existing_profile:
            if api_key_raw == "-":
                api_key = None
            elif api_key_raw:
                api_key = api_key_raw
            else:
                api_key = self._existing_profile.api_key
        else:
            api_key = api_key_raw or None

        auth_token_raw = (auth_token_input.value or "").strip()
        if protocol == ProtocolType.ANTHROPIC or (
            self._existing_profile and self._existing_profile.protocol == ProtocolType.ANTHROPIC
        ):
            if self._existing_profile:
                if auth_token_raw == "-":
                    auth_token = None
                elif auth_token_raw:
                    auth_token = auth_token_raw
                else:
                    auth_token = self._existing_profile.auth_token
            else:
                auth_token = auth_token_raw or None
        else:
            auth_token = None

        api_base = (api_base_input.value or "").strip() or None

        max_input_tokens_default = (
            self._existing_profile.max_input_tokens
            if self._existing_profile
            else inferred_profile.max_input_tokens
        )
        max_input_tokens = self._parse_int(
            max_input_tokens_input.value,
            "Max input tokens",
            default_value=max_input_tokens_default,
        )

        max_output_tokens_default = (
            self._existing_profile.max_output_tokens
            if self._existing_profile
            else inferred_profile.max_output_tokens
        )
        max_output_tokens = self._parse_int(
            max_output_tokens_input.value,
            "Max output tokens",
            default_value=max_output_tokens_default,
        )

        max_tokens_default = (
            self._existing_profile.max_tokens if self._existing_profile else inferred_profile.max_tokens
        )
        max_tokens = self._parse_int(
            max_tokens_input.value,
            "Max tokens",
            default_value=max_tokens_default,
        )
        if max_tokens is None:
            return
        if max_output_tokens is not None and max_tokens > max_output_tokens:
            self._set_error("Max tokens must be less than or equal to max output tokens.")
            return

        temperature = self._parse_float(temperature_input.value, "Temperature")
        if temperature is None:
            return

        input_price_default = (
            self._existing_profile.price.input if self._existing_profile else inferred_profile.price.input
        )
        input_price = self._parse_float(
            input_price_input.value,
            "Input price",
            default_value=input_price_default,
        )
        if input_price is None:
            return
        output_price_default = (
            self._existing_profile.price.output
            if self._existing_profile
            else inferred_profile.price.output
        )
        output_price = self._parse_float(
            output_price_input.value,
            "Output price",
            default_value=output_price_default,
        )
        if output_price is None:
            return
        currency_default = (
            self._existing_profile.currency if self._existing_profile else inferred_profile.currency
        )
        currency = ((currency_input.value or "").strip().upper() or currency_default or "USD")

        vision_raw = vision_select.value
        supports_value = (
            vision_raw.strip().lower() if isinstance(vision_raw, str) and vision_raw else "auto"
        )
        if supports_value == "yes":
            supports_vision = True
        elif supports_value == "no":
            supports_vision = False
        else:
            supports_vision = (
                self._existing_profile.supports_vision
                if self._existing_profile
                else inferred_profile.supports_vision
            )

        set_as_main = bool(self.query_one("#set_main", Checkbox).value)
        set_as_quick = bool(self.query_one("#set_quick", Checkbox).value)

        profile = ModelProfile(
            protocol=protocol,
            model=model_name,
            api_key=api_key,
            auth_token=auth_token,
            api_base=api_base,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            supports_vision=supports_vision,
            price={"input": input_price, "output": output_price},
            currency=currency,
        )

        self.dismiss(
            ModelFormResult(
                name=name,
                profile=profile,
                set_as_main=set_as_main,
                set_as_quick=set_as_quick,
            )
        )

    def _set_error(self, message: str) -> None:
        app = getattr(self, "app", None)
        if app:
            app.notify(message, title="Validation error", severity="error", timeout=6)

    def _parse_int(self, raw: str, label: str, default_value: Optional[int] = None) -> Optional[int]:
        raw = (raw or "").strip()
        if not raw:
            if default_value is not None:
                return default_value
            if label == "Max tokens":
                return 4096
            return None
        try:
            return int(raw)
        except ValueError:
            self._set_error(f"Invalid number for {label}.")
            return None

    def _parse_float(self, raw: str, label: str, default_value: Optional[float] = None) -> Optional[float]:
        raw = (raw or "").strip()
        if not raw:
            if default_value is not None:
                return default_value
            if self._existing_profile and label == "Temperature":
                return self._existing_profile.temperature
            if self._existing_profile and label == "Input price":
                return self._existing_profile.price.input
            if self._existing_profile and label == "Output price":
                return self._existing_profile.price.output
            if label == "Temperature":
                return 1.0
            if label in {"Input price", "Output price"}:
                return 0.0
            return None
        try:
            return float(raw)
        except ValueError:
            self._set_error(f"Invalid number for {label}.")
            return None


class ModelsApp(App[None]):
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

    #models_table {
        width: 42%;
        min-width: 40;
    }

    #details_panel {
        width: 58%;
        padding: 0 1;
    }

    #form_dialog, #confirm_dialog {
        width: 72;
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
        ("m", "set_main", "Set main"),
        ("k", "set_quick", "Set quick"),
        ("r", "refresh", "Refresh"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._selected_name: Optional[str] = None
        self._row_names: list[str] = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Static("", id="status_bar")
        with Container(id="body"):
            yield DataTable(id="models_table")
            yield Static(id="details_panel")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#models_table", DataTable)
        table.add_columns("#", "Name", "Ptr", "Protocol", "Model")
        try:
            table.cursor_type = "row"
            table.zebra_stripes = True
        except Exception:
            pass
        self._refresh_models(select_first=True)

    def action_refresh(self) -> None:
        self._refresh_models(select_first=False)
        self._set_status("Refreshed.")

    def action_add(self) -> None:
        config = get_global_config()
        default_set_main = (
            not config.model_profiles
            or getattr(config.model_pointers, "main", "") not in config.model_profiles
        )
        screen = ModelFormScreen(
            "add",
            default_set_main=default_set_main,
            default_set_quick=False,
        )
        self.push_screen(screen, self._handle_add_result)

    def action_edit(self) -> None:
        profile = self._selected_profile()
        if not profile:
            self._set_status("No model selected.")
            return
        screen = ModelFormScreen(
            "edit",
            existing_name=self._selected_name,
            existing_profile=profile,
        )
        self.push_screen(screen, self._handle_edit_result)

    def action_delete(self) -> None:
        profile = self._selected_profile()
        if not profile:
            self._set_status("No model selected.")
            return
        screen = ConfirmScreen(f"Delete model '{self._selected_name}'?")
        self.push_screen(screen, self._handle_delete_confirm)

    def action_set_main(self) -> None:
        if not self._selected_name:
            self._set_status("No model selected.")
            return
        try:
            set_model_pointer("main", self._selected_name)
            self._set_status(f"Main -> {self._selected_name}")
            self._refresh_models(select_first=False)
        except (ValueError, KeyError, OSError, IOError, PermissionError) as exc:
            self._set_status(str(exc))

    def action_set_quick(self) -> None:
        if not self._selected_name:
            self._set_status("No model selected.")
            return
        try:
            set_model_pointer("quick", self._selected_name)
            self._set_status(f"Quick -> {self._selected_name}")
            self._refresh_models(select_first=False)
        except (ValueError, KeyError, OSError, IOError, PermissionError) as exc:
            self._set_status(str(exc))

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        self._select_by_index(int(event.cursor_row))

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        self._select_by_index(int(event.cursor_row))
        self.action_edit()

    def _handle_add_result(self, result: Optional[ModelFormResult]) -> None:
        if not result:
            return
        try:
            add_model_profile(
                result.name,
                result.profile,
                overwrite=False,
                set_as_main=result.set_as_main,
            )
            if result.set_as_quick:
                set_model_pointer("quick", result.name)
        except (OSError, IOError, ValueError, TypeError, PermissionError) as exc:
            self._set_status(str(exc))
            return
        self._selected_name = result.name
        self._set_status(f"Saved {result.name}.")
        self._refresh_models(select_first=False)

    def _handle_edit_result(self, result: Optional[ModelFormResult]) -> None:
        if not result:
            return
        try:
            add_model_profile(
                result.name,
                result.profile,
                overwrite=True,
                set_as_main=False,
            )
            if result.set_as_main:
                set_model_pointer("main", result.name)
            if result.set_as_quick:
                set_model_pointer("quick", result.name)
        except (OSError, IOError, ValueError, TypeError, PermissionError) as exc:
            self._set_status(str(exc))
            return
        self._selected_name = result.name
        self._set_status(f"Updated {result.name}.")
        self._refresh_models(select_first=False)

    def _handle_delete_confirm(self, confirmed: Optional[bool]) -> None:
        if not confirmed or not self._selected_name:
            return
        try:
            delete_model_profile(self._selected_name)
        except (OSError, IOError, KeyError, PermissionError) as exc:
            self._set_status(str(exc))
            return
        self._set_status(f"Deleted {self._selected_name}.")
        self._selected_name = None
        self._refresh_models(select_first=True)

    def _refresh_models(self, select_first: bool) -> None:
        config = get_global_config()
        table = self.query_one("#models_table", DataTable)
        table.clear(columns=False)
        pointer_map = config.model_pointers.model_dump()
        self._row_names = []

        for idx, (name, profile) in enumerate(config.model_profiles.items(), start=1):
            self._row_names.append(name)
            markers = [ptr for ptr, value in pointer_map.items() if value == name]
            pointer_label = ",".join(markers) if markers else "-"
            table.add_row(
                str(idx),
                name,
                pointer_label,
                profile.protocol.value,
                profile.model,
            )

        if not config.model_profiles:
            self._selected_name = None
            self._update_details()
            return

        if self._selected_name and self._selected_name in config.model_profiles:
            try:
                row_index = self._row_names.index(self._selected_name)
            except ValueError:
                row_index = 0
            self._move_cursor(table, row_index)
            self._update_details()
            return

        if select_first:
            first_name = next(iter(config.model_profiles))
            self._selected_name = first_name
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

    def _selected_profile(self) -> Optional[ModelProfile]:
        if not self._selected_name:
            return None
        config = get_global_config()
        return config.model_profiles.get(self._selected_name)

    def _update_details(self) -> None:
        details = self.query_one("#details_panel", Static)
        if not self._selected_name:
            details.update("No model selected.")
            return
        config = get_global_config()
        profile = config.model_profiles.get(self._selected_name)
        if not profile:
            details.update("No model selected.")
            return
        pointer_map = config.model_pointers.model_dump()
        markers = [ptr for ptr, value in pointer_map.items() if value == self._selected_name]
        marker_text = ", ".join(markers) if markers else "-"
        vision_display = self._vision_display(profile)

        table = Table.grid(padding=(0, 2))
        table.add_column(style="cyan", no_wrap=True)
        table.add_column()
        table.add_row("Profile", self._selected_name)
        table.add_row("Pointers", marker_text)
        table.add_row("Protocol", profile.protocol.value)
        table.add_row("Model", profile.model)
        table.add_row("API base", profile.api_base or "-")
        table.add_row(
            "Max input tokens",
            str(profile.max_input_tokens) if profile.max_input_tokens else "auto",
        )
        table.add_row(
            "Max output tokens",
            str(profile.max_output_tokens) if profile.max_output_tokens else "auto",
        )
        table.add_row("Max tokens", str(profile.max_tokens))
        table.add_row("Temperature", str(profile.temperature))
        table.add_row(
            "Price",
            f"in={profile.price.input}/1M, out={profile.price.output}/1M ({profile.currency})",
        )
        table.add_row("Vision", vision_display)
        table.add_row("API key", "set" if profile.api_key else "unset")
        if profile.protocol == ProtocolType.ANTHROPIC:
            table.add_row(
                "Auth token",
                "set" if getattr(profile, "auth_token", None) else "unset",
            )
        if profile.openai_tool_mode:
            table.add_row("OpenAI tool mode", profile.openai_tool_mode)
        if profile.thinking_mode:
            table.add_row("Thinking mode", profile.thinking_mode)

        details.update(Panel(table, title=f"Model: {self._selected_name}", box=box.ROUNDED))

    def _vision_display(self, profile: ModelProfile) -> str:
        if profile.supports_vision is None:
            detected = model_supports_vision(profile)
            return f"auto (detected {'yes' if detected else 'no'})"
        return "yes" if profile.supports_vision else "no"

    def _set_status(self, message: str) -> None:
        status = self.query_one("#status_bar", Static)
        status.update(message)


def run_models_tui(on_exit: Optional[Callable[[], Any]] = None) -> bool:
    """Run the Textual models TUI."""
    app = ModelsApp()
    app.run()
    if on_exit:
        on_exit()
    return True
