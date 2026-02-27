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
    rename_model_profile,
    set_model_pointer,
)
from ripperdoc.core.oauth import (
    OAuthTokenType,
    list_oauth_tokens,
    oauth_models_for_type,
)
from ripperdoc.cli.ui.helpers import get_profile_for_pointer

_KNOWN_THINKING_MODES = {
    "deepseek",
    "openrouter",
    "qwen",
    "gemini_openai",
    "openai",
    "off",
    "disabled",
}
_THINKING_MODE_CUSTOM = "__custom__"


def _next_copied_profile_name(source_name: str, existing_names: set[str]) -> str:
    """Return the next available copied profile name, e.g. `x (1)`."""
    index = 1
    while True:
        candidate = f"{source_name} ({index})"
        if candidate not in existing_names:
            return candidate
        index += 1


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


class RenameScreen(ModalScreen[Optional[str]]):
    """Prompt for a new model profile name."""

    def __init__(self, current_name: str) -> None:
        super().__init__()
        self._current_name = current_name

    def compose(self) -> ComposeResult:
        with Container(id="confirm_dialog"):
            yield Static(f"Rename '{self._current_name}' to:", id="confirm_message")
            yield Input(value=self._current_name, id="rename_input", placeholder="New profile name")
            with Horizontal(id="confirm_buttons"):
                yield Button("Rename", id="rename_save", variant="primary")
                yield Button("Cancel", id="rename_cancel")

    def on_mount(self) -> None:
        try:
            rename_input = self.query_one("#rename_input", Input)
            rename_input.focus()
            rename_input.select_all()
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "rename_cancel":
            self.dismiss(None)
            return
        if event.button.id != "rename_save":
            return
        self._submit()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "rename_input":
            self._submit()

    def _submit(self) -> None:
        try:
            rename_input = self.query_one("#rename_input", Input)
            self.dismiss((rename_input.value or "").strip())
        except Exception:
            self.dismiss(None)


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

    def _oauth_token_select_options(
        self, selected_name: Optional[str]
    ) -> list[tuple[str, str]]:
        tokens = list_oauth_tokens()
        options: list[tuple[str, str]] = []
        for name, token in tokens.items():
            masked = (
                f"{token.access_token[:4]}…{token.access_token[-2:]}"
                if len(token.access_token) > 8
                else "***"
            )
            options.append((f"{name} ({token.type.value}, {masked})", name))
        if not options and selected_name:
            options.append((f"{selected_name} (missing)", selected_name))
        return options

    def _oauth_model_select_options(
        self,
        token_type: OAuthTokenType,
    ) -> list[tuple[str, str]]:
        return [
            (option.label, option.model)
            for option in oauth_models_for_type(token_type)
        ]

    def _selected_oauth_token_type(self) -> OAuthTokenType:
        try:
            oauth_token_select = self.query_one("#oauth_token_select", Select)
        except Exception:
            return OAuthTokenType.CODEX
        token_raw = oauth_token_select.value
        token_name = token_raw.strip() if isinstance(token_raw, str) else ""
        token = list_oauth_tokens().get(token_name)
        if token is not None:
            return token.type
        if self._existing_profile and self._existing_profile.oauth_token_type:
            return self._existing_profile.oauth_token_type
        return OAuthTokenType.CODEX

    def _refresh_oauth_model_options(self) -> None:
        try:
            oauth_model_select = self.query_one("#oauth_model_select", Select)
        except Exception:
            return

        token_type = self._selected_oauth_token_type()
        model_options = self._oauth_model_select_options(token_type)
        current_raw = oauth_model_select.value
        current_value = current_raw.strip() if isinstance(current_raw, str) else ""
        existing_default = (
            self._existing_profile.model
            if self._existing_profile and self._existing_profile.protocol == ProtocolType.OAUTH
            else ""
        )
        preferred = current_value or existing_default
        if model_options:
            selected_value = (
                preferred
                if any(value == preferred for _label, value in model_options)
                else model_options[0][1]
            )
            oauth_model_select.set_options(model_options)
            oauth_model_select.value = selected_value
            return
        oauth_model_select.set_options([("No OAuth models", Select.BLANK)])
        oauth_model_select.value = Select.BLANK

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
                yield Static("Model name", classes="field_label", id="model_input_label")
                yield Input(value=model_default, placeholder="Model name", id="model_input")
                oauth_token_options = self._oauth_token_select_options(
                    self._existing_profile.oauth_token_name if self._existing_profile else None
                )
                oauth_token_value = oauth_token_options[0][1] if oauth_token_options else Select.BLANK
                if self._existing_profile and self._existing_profile.oauth_token_name:
                    existing_name = self._existing_profile.oauth_token_name
                    if any(value == existing_name for _label, value in oauth_token_options):
                        oauth_token_value = existing_name
                selected_oauth_token = list_oauth_tokens().get(
                    oauth_token_value if isinstance(oauth_token_value, str) else ""
                )
                selected_oauth_token_type = (
                    selected_oauth_token.type
                    if selected_oauth_token is not None
                    else (
                        self._existing_profile.oauth_token_type
                        if self._existing_profile and self._existing_profile.oauth_token_type
                        else OAuthTokenType.CODEX
                    )
                )
                oauth_model_options = [
                    (option.label, option.model)
                    for option in oauth_models_for_type(selected_oauth_token_type)
                ]
                oauth_model_default = (
                    model_default
                    if any(value == model_default for _label, value in oauth_model_options)
                    else (oauth_model_options[0][1] if oauth_model_options else Select.BLANK)
                )
                yield Select(
                    oauth_model_options if oauth_model_options else [("No OAuth models", Select.BLANK)],
                    value=oauth_model_default,
                    id="oauth_model_select",
                )

                api_key_placeholder = (
                    "[set]" if (self._existing_profile and self._existing_profile.api_key) else "[not set]"
                )
                yield Static("API key", classes="field_label", id="api_key_label")
                yield Input(
                    placeholder=f"API key {api_key_placeholder} (blank=keep, '-'=clear)",
                    password=True,
                    id="api_key_input",
                )

                auth_placeholder = (
                    "[set]"
                    if (self._existing_profile and self._existing_profile.auth_token)
                    else "[not set]"
                )
                yield Static(
                    "Auth token (Anthropic only)",
                    classes="field_label",
                    id="auth_token_label",
                )
                yield Input(
                    placeholder=f"Auth token (Anthropic only) {auth_placeholder} (blank=keep, '-'=clear)",
                    password=True,
                    id="auth_token_input",
                )

                api_base_default = self._existing_profile.api_base if self._existing_profile else ""
                yield Static("API base", classes="field_label", id="api_base_label")
                yield Input(
                    value=api_base_default or "",
                    placeholder="API base (optional)",
                    id="api_base_input",
                )

                yield Static("OAuth token", classes="field_label", id="oauth_token_label")
                yield Select(
                    oauth_token_options if oauth_token_options else [("No OAuth tokens", Select.BLANK)],
                    value=oauth_token_value,
                    id="oauth_token_select",
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

                thinking_mode_default = (
                    self._existing_profile.thinking_mode if self._existing_profile else ""
                )
                select_default, custom_default = self._thinking_mode_form_defaults(
                    thinking_mode_default
                )
                yield Static("Thinking mode override", classes="field_label")
                thinking_options = [
                    ("auto (detect)", "auto"),
                    ("deepseek", "deepseek"),
                    ("openrouter", "openrouter"),
                    ("qwen", "qwen"),
                    ("gemini_openai", "gemini_openai"),
                    ("openai", "openai"),
                    ("off", "off"),
                    ("disabled", "disabled"),
                    ("custom", _THINKING_MODE_CUSTOM),
                ]
                yield Select(thinking_options, value=select_default, id="thinking_mode_select")
                yield Input(
                    value=custom_default,
                    placeholder="Custom thinking mode (only used when 'custom' is selected)",
                    id="thinking_mode_custom_input",
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

    def on_mount(self) -> None:
        self._refresh_provider_fields()
        self._refresh_oauth_model_options()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "provider_select":
            self._refresh_provider_fields()
            if isinstance(event.value, str) and event.value == ProtocolType.OAUTH.value:
                self._refresh_oauth_model_options()
            return
        if event.select.id == "oauth_token_select":
            self._refresh_oauth_model_options()

    def _set_widget_visible(self, selector: str, visible: bool) -> None:
        try:
            widget = self.query_one(selector)
        except Exception:
            return
        try:
            widget.display = visible
        except Exception:
            try:
                widget.styles.display = "block" if visible else "none"
            except Exception:
                pass

    def _refresh_provider_fields(self) -> None:
        provider_select = self.query_one("#provider_select", Select)
        provider_raw = provider_select.value
        provider_value = provider_raw.strip() if isinstance(provider_raw, str) else ""
        try:
            protocol = ProtocolType(provider_value)
        except ValueError:
            protocol = ProtocolType.OPENAI_COMPATIBLE

        is_oauth = protocol == ProtocolType.OAUTH
        is_anthropic = protocol == ProtocolType.ANTHROPIC

        self._set_widget_visible("#model_input", not is_oauth)
        self._set_widget_visible("#oauth_model_select", is_oauth)
        self._set_widget_visible("#oauth_token_label", is_oauth)
        self._set_widget_visible("#oauth_token_select", is_oauth)

        self._set_widget_visible("#api_key_label", not is_oauth)
        self._set_widget_visible("#api_key_input", not is_oauth)
        self._set_widget_visible("#api_base_label", not is_oauth)
        self._set_widget_visible("#api_base_input", not is_oauth)
        self._set_widget_visible("#auth_token_label", (not is_oauth) and is_anthropic)
        self._set_widget_visible("#auth_token_input", (not is_oauth) and is_anthropic)

        try:
            model_label = self.query_one("#model_input_label", Static)
            model_label.update("Model (OAuth preset)" if is_oauth else "Model name")
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "form_cancel":
            self.dismiss(None)
            return
        if event.button.id != "form_save":
            return

        name_input = self.query_one("#name_input", Input) if self._mode == "add" else None
        provider_select = self.query_one("#provider_select", Select)
        model_input = self.query_one("#model_input", Input)
        oauth_model_select = self.query_one("#oauth_model_select", Select)
        oauth_token_select = self.query_one("#oauth_token_select", Select)
        api_key_input = self.query_one("#api_key_input", Input)
        auth_token_input = self.query_one("#auth_token_input", Input)
        api_base_input = self.query_one("#api_base_input", Input)
        max_input_tokens_input = self.query_one("#max_input_tokens_input", Input)
        max_output_tokens_input = self.query_one("#max_output_tokens_input", Input)
        max_tokens_input = self.query_one("#max_tokens_input", Input)
        temperature_input = self.query_one("#temperature_input", Input)
        thinking_mode_select = self.query_one("#thinking_mode_select", Select)
        thinking_mode_custom_input = self.query_one("#thinking_mode_custom_input", Input)
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

        oauth_token_name: Optional[str] = None
        oauth_token_type: Optional[OAuthTokenType] = None
        if protocol == ProtocolType.OAUTH:
            oauth_model_raw = oauth_model_select.value
            model_name = (
                oauth_model_raw.strip()
                if isinstance(oauth_model_raw, str)
                else ""
            )
            oauth_token_raw = oauth_token_select.value
            oauth_token_name = (
                oauth_token_raw.strip()
                if isinstance(oauth_token_raw, str)
                else ""
            )
            token = list_oauth_tokens().get(oauth_token_name or "")
            if not oauth_token_name or token is None:
                self._set_error("Select a valid OAuth token first.")
                return
            oauth_token_type = token.type
        else:
            model_name = (model_input.value or "").strip()
        if not model_name:
            self._set_error("Model name is required.")
            return
        inferred_profile = ModelProfile(
            protocol=protocol,
            model=model_name,
            oauth_token_name=oauth_token_name,
            oauth_token_type=oauth_token_type,
        )

        if protocol == ProtocolType.OAUTH:
            api_key = None
            auth_token = None
            api_base = None
        else:
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

        select_raw = thinking_mode_select.value
        select_value = select_raw.strip().lower() if isinstance(select_raw, str) else "auto"
        thinking_mode = self._resolve_thinking_mode(
            selected_value=select_value,
            custom_value=thinking_mode_custom_input.value,
        )
        if thinking_mode is None and select_value == _THINKING_MODE_CUSTOM:
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
        supports_vision: bool | None
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
            oauth_token_name=oauth_token_name,
            oauth_token_type=oauth_token_type,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            thinking_mode=thinking_mode,
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

    def _thinking_mode_form_defaults(self, thinking_mode: Optional[str]) -> tuple[str, str]:
        if not thinking_mode:
            return "auto", ""
        lowered = thinking_mode.strip().lower()
        if lowered in _KNOWN_THINKING_MODES:
            return lowered, ""
        return _THINKING_MODE_CUSTOM, lowered

    def _resolve_thinking_mode(
        self,
        *,
        selected_value: str,
        custom_value: str,
    ) -> Optional[str]:
        lowered = (selected_value or "auto").strip().lower()
        if lowered == "auto":
            return None
        if lowered == _THINKING_MODE_CUSTOM:
            custom = (custom_value or "").strip().lower()
            if not custom:
                self._set_error("Custom thinking mode cannot be empty.")
                return None
            return custom
        return lowered


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
        ("c", "copy", "Copy"),
        ("n", "rename", "Rename"),
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

    def action_copy(self) -> None:
        profile = self._selected_profile()
        if not profile or not self._selected_name:
            self._set_status("No model selected.")
            return

        source_name = self._selected_name
        config = get_global_config()
        existing_names = set(config.model_profiles.keys())
        new_name = _next_copied_profile_name(source_name, existing_names)

        try:
            add_model_profile(
                new_name,
                profile.model_copy(deep=True),
                overwrite=False,
                set_as_main=False,
            )
        except (OSError, IOError, ValueError, TypeError, PermissionError) as exc:
            self._set_status(str(exc))
            return

        self._selected_name = new_name
        self._set_status(f"Copied {source_name} as {new_name}.")
        self._refresh_models(select_first=False)

    def action_rename(self) -> None:
        if not self._selected_name:
            self._set_status("No model selected.")
            return
        screen = RenameScreen(self._selected_name)
        self.push_screen(screen, self._handle_rename_result)

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

    def _handle_rename_result(self, new_name: Optional[str]) -> None:
        if new_name is None or not self._selected_name:
            return
        if new_name == self._selected_name:
            self._set_status("Name unchanged.")
            return
        try:
            rename_model_profile(self._selected_name, new_name)
        except (OSError, IOError, KeyError, ValueError, PermissionError) as exc:
            message = str(exc)
            self._set_status(message)
            try:
                self.notify(
                    message,
                    title="Rename failed",
                    severity="error",
                    timeout=6,
                )
            except Exception:
                pass
            return
        self._selected_name = new_name
        self._set_status(f"Renamed profile to {new_name}.")
        self._refresh_models(select_first=False)

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
        api_base_display = "-" if profile.protocol == ProtocolType.OAUTH else (profile.api_base or "-")
        table.add_row("API base", api_base_display)
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
        if profile.protocol == ProtocolType.OAUTH:
            token_type = (
                profile.oauth_token_type.value
                if profile.oauth_token_type is not None
                else OAuthTokenType.CODEX.value
            )
            table.add_row("OAuth token", f"{profile.oauth_token_name or 'unset'} ({token_type})")
        else:
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
