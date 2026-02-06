"""Textual app for managing hooks configuration."""

from __future__ import annotations

import json
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from textual.app import App, ComposeResult
from textual import events
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Footer, Header, Input, OptionList, Select, Static, TextArea
from textual.widgets.option_list import Option

from ripperdoc.core.default_tools import BUILTIN_TOOL_NAMES
from ripperdoc.core.hooks import (
    HookEvent,
    get_global_hooks_path,
    get_project_hooks_path,
    get_project_local_hooks_path,
)
from ripperdoc.core.agents import load_agent_definitions
from ripperdoc.core.hooks.config import (
    DEFAULT_HOOK_TIMEOUT,
    PROMPT_SUPPORTED_EVENTS,
    ALWAYS_MATCHER_EVENTS,
    MATCHER_VALUE_OPTIONS,
    TEXT_MATCHER_EVENTS,
    TOOL_MATCHER_EVENTS,
)

EVENT_DESCRIPTIONS: Dict[str, str] = {
    HookEvent.PRE_TOOL_USE.value: "Before a tool runs (can block or edit input)",
    HookEvent.PERMISSION_REQUEST.value: "When a permission dialog is shown",
    HookEvent.POST_TOOL_USE.value: "After a tool finishes running",
    HookEvent.POST_TOOL_USE_FAILURE.value: "After a tool fails",
    HookEvent.USER_PROMPT_SUBMIT.value: "When you submit a prompt",
    HookEvent.NOTIFICATION.value: "When Ripperdoc sends a notification",
    HookEvent.STOP.value: "When the agent stops responding",
    HookEvent.SUBAGENT_START.value: "Before a Task/subagent starts",
    HookEvent.SUBAGENT_STOP.value: "When a Task/subagent stops",
    HookEvent.PRE_COMPACT.value: "Before compacting the conversation",
    HookEvent.SESSION_START.value: "When a session starts or resumes",
    HookEvent.SESSION_END.value: "When a session ends",
    HookEvent.SETUP.value: "When repository setup/maintenance runs",
}

MATCHER_FIELD_NAMES: Dict[str, str] = {
    HookEvent.PRE_TOOL_USE.value: "tool_name",
    HookEvent.PERMISSION_REQUEST.value: "tool_name",
    HookEvent.POST_TOOL_USE.value: "tool_name",
    HookEvent.POST_TOOL_USE_FAILURE.value: "tool_name",
    HookEvent.USER_PROMPT_SUBMIT.value: "prompt",
    HookEvent.NOTIFICATION.value: "notification_type",
    HookEvent.PRE_COMPACT.value: "trigger",
    HookEvent.SESSION_START.value: "source",
    HookEvent.SESSION_END.value: "reason",
    HookEvent.SETUP.value: "trigger",
    HookEvent.SUBAGENT_START.value: "subagent_type",
    HookEvent.SUBAGENT_STOP.value: "subagent_type",
}


def _matcher_mode(event_name: str) -> str:
    if event_name in TOOL_MATCHER_EVENTS:
        return "tool"
    if event_name in MATCHER_VALUE_OPTIONS:
        return "enum"
    if event_name in TEXT_MATCHER_EVENTS:
        return "text"
    if event_name in ALWAYS_MATCHER_EVENTS:
        return "always"
    return "none"


def _matcher_choices(event_name: str) -> List[str]:
    return list(MATCHER_VALUE_OPTIONS.get(event_name, []))


def _matcher_field_name(event_name: str) -> str:
    return MATCHER_FIELD_NAMES.get(event_name, "matcher")


@dataclass(frozen=True)
class HookConfigTarget:
    key: str
    label: str
    path: Path
    description: str


@dataclass(frozen=True)
class HookFormResult:
    hook: Dict[str, Any]


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


class SaveTargetScreen(ModalScreen[Optional[int]]):
    """Modal screen for choosing where to save a new hook."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, targets: List["HookConfigTarget"], current_index: int) -> None:
        super().__init__()
        self._targets = targets
        self._current_index = current_index
        self._option_ids: List[str] = []

    def compose(self) -> ComposeResult:
        with Container(id="target_dialog"):
            yield Static("Save hook configuration", id="target_title")
            yield Static("Where should this hook be saved?", id="target_subtitle")
            yield OptionList(id="target_list")
            yield Static("Enter to confirm | Esc to cancel", id="target_hint")

    def on_mount(self) -> None:
        option_list = self.query_one("#target_list", OptionList)
        option_list.clear_options()
        self._option_ids = []
        for idx, target in enumerate(self._targets, start=1):
            current = " (current)" if idx - 1 == self._current_index else ""
            label = f"{idx}. {target.label}{current}\n{target.description}"
            option_id = f"target:{idx - 1}"
            self._option_ids.append(option_id)
            option_list.add_option(Option(label, id=option_id))
        if option_list.option_count:
            option_list.highlighted = min(self._current_index, option_list.option_count - 1)
        option_list.focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        option_id = event.option.id or ""
        if option_id.startswith("target:"):
            self.dismiss(int(option_id.split(":", 1)[1]))
            return
        self.dismiss(None)


class MatcherScreen(ModalScreen[Optional[str]]):
    """Modal screen for adding/editing matchers."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, event_name: str, existing: Optional[str] = None) -> None:
        super().__init__()
        self._event_name = event_name
        self._existing = existing or ""
        self._mode = _matcher_mode(event_name)
        self._choices = _matcher_choices(event_name)

    def compose(self) -> ComposeResult:
        title = f"{'Edit' if self._existing else 'Add'} matcher for {self._event_name}"
        tools_hint = self._tool_name_hint()
        agent_hint = self._agent_type_hint()
        with Container(id="form_dialog"):
            yield Static(title, id="form_title")
            with VerticalScroll(id="form_fields"):
                if tools_hint:
                    yield Static("Possible matcher values for field tool_name:", classes="field_label")
                    yield Static(tools_hint, classes="field_value")
                if agent_hint:
                    yield Static(
                        "Possible matcher values for field subagent_type:",
                        classes="field_label",
                    )
                    yield Static(agent_hint, classes="field_value")

                field_name = _matcher_field_name(self._event_name)
                if self._mode == "enum" and self._choices:
                    yield Static(f"Matcher ({field_name})", classes="field_label")
                    options, default_value = self._build_select_options()
                    yield Select(options, value=default_value, id="matcher_select")
                else:
                    yield Static("Matcher (blank for '*')", classes="field_label")
                    yield Input(
                        value=self._existing,
                        placeholder="Matcher pattern",
                        id="matcher_input",
                    )
                    if self._mode == "tool":
                        yield Static("Example matchers:", classes="field_label")
                        yield Static(
                            "Write (single tool) | Write|Edit (multiple) | Web.* (regex)",
                            classes="field_value",
                        )
                    elif self._mode == "text":
                        yield Static(
                            f"Matcher values map to {field_name}. Leave blank for all.",
                            classes="field_value",
                        )
                    elif self._mode == "always":
                        yield Static(
                            "Matcher is ignored for this event; all hooks will run.",
                            classes="field_value",
                        )

            with Horizontal(id="form_buttons"):
                yield Button("Save", id="form_save", variant="primary")
                yield Button("Cancel", id="form_cancel")

    def on_mount(self) -> None:
        if self._mode == "enum" and self._choices:
            self.query_one("#matcher_select", Select).focus()
        else:
            self.query_one("#matcher_input", Input).focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "form_cancel":
            self.dismiss(None)
            return
        if event.button.id != "form_save":
            return
        if self._mode == "enum" and self._choices:
            selected_value = self.query_one("#matcher_select", Select).value
            selected = selected_value.strip() if isinstance(selected_value, str) else ""
            self.dismiss(selected or "*")
            return
        raw = (self.query_one("#matcher_input", Input).value or "").strip()
        self.dismiss(raw or "*")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "matcher_input":
            return
        raw = (event.value or "").strip()
        self.dismiss(raw or "*")

    def _tool_name_hint(self) -> str:
        if self._event_name not in TOOL_MATCHER_EVENTS:
            return ""
        tools = ", ".join(sorted(BUILTIN_TOOL_NAMES))
        return textwrap.fill(tools, width=84)

    def _agent_type_hint(self) -> str:
        if self._event_name not in TEXT_MATCHER_EVENTS:
            return ""
        try:
            agents = load_agent_definitions().active_agents
        except Exception:
            return ""
        names = sorted({agent.agent_type for agent in agents if agent.agent_type})
        if not names:
            return ""
        return textwrap.fill(", ".join(names), width=84)

    def _build_select_options(self) -> tuple[list[tuple[str, str]], str]:
        options: list[tuple[str, str]] = [("All (*)", "*")]
        for value in self._choices:
            options.append((value, value))
        existing = (self._existing or "").strip()
        if existing and existing not in {value for _, value in options}:
            options.append((existing, existing))
        default_value = existing or "*"
        return options, default_value


class HookFormScreen(ModalScreen[Optional[HookFormResult]]):
    """Modal form for adding/editing hooks."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(
        self,
        event_name: str,
        hook_type: str,
        *,
        existing_hook: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self._event_name = event_name
        self._hook_type = hook_type
        self._existing_hook = existing_hook or {}

    def compose(self) -> ComposeResult:
        title = "Add hook" if not self._existing_hook else "Edit hook"
        with Container(id="form_dialog"):
            yield Static(title, id="form_title")
            with VerticalScroll(id="form_fields"):
                if self._hook_type in ("prompt", "agent"):
                    prompt_default = self._existing_hook.get("prompt", "")
                    yield Static("Prompt template (use $ARGUMENTS for JSON input)", classes="field_label")
                    yield TextArea(
                        text=prompt_default,
                        placeholder="Prompt template",
                        id="prompt_input",
                    )
                    if self._hook_type == "agent":
                        model_default = self._existing_hook.get("model", "")
                        yield Static("Model pointer (optional)", classes="field_label")
                        yield Input(
                            value=model_default,
                            placeholder="quick",
                            id="model_input",
                        )
                else:
                    command_default = self._existing_hook.get("command", "")
                    yield Static("Command to run", classes="field_label")
                    yield Input(
                        value=command_default,
                        placeholder="Command to run",
                        id="command_input",
                    )

                # Options section
                yield Static("", classes="field_label")
                yield Static("Options", classes="field_label")
                if self._hook_type == "command":
                    yield Checkbox(
                        label="Run asynchronously (don't block)",
                        value=self._existing_hook.get("async", False),
                        id="async_checkbox",
                    )
                if self._hook_type in ("command", "prompt"):
                    once_hint = " Skills only" if self._hook_type == "prompt" else ""
                    yield Checkbox(
                        label=f"Run once per session{once_hint}",
                        value=self._existing_hook.get("once", False),
                        id="once_checkbox",
                    )

                status_message_default = self._existing_hook.get("statusMessage", "")
                yield Static("Status message (optional)", classes="field_label")
                yield Input(
                    value=status_message_default,
                    placeholder="Custom spinner message while hook runs",
                    id="status_message_input",
                )

                timeout_default = str(
                    self._existing_hook.get("timeout", DEFAULT_HOOK_TIMEOUT)
                )
                yield Static("Timeout seconds", classes="field_label")
                yield Input(
                    value=timeout_default,
                    placeholder=str(DEFAULT_HOOK_TIMEOUT),
                    id="timeout_input",
                )

            with Horizontal(id="form_buttons"):
                yield Button("Save", id="form_save", variant="primary")
                yield Button("Cancel", id="form_cancel")

    def on_mount(self) -> None:
        if self._hook_type in ("prompt", "agent"):
            self.query_one("#prompt_input", TextArea).focus()
        else:
            self.query_one("#command_input", Input).focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "form_cancel":
            self.dismiss(None)
            return
        if event.button.id != "form_save":
            return

        if self._hook_type in ("prompt", "agent"):
            prompt_text = (self.query_one("#prompt_input", TextArea).text or "").strip()
            if not prompt_text:
                self._set_error("Prompt text is required.")
                return
            hook_type = "prompt" if self._hook_type == "prompt" else "agent"
            hook: Dict[str, Any] = {"type": hook_type, "prompt": prompt_text}
            if hook_type == "agent":
                model_text = (self.query_one("#model_input", Input).value or "").strip()
                if model_text:
                    hook["model"] = model_text
        else:
            command_text = (self.query_one("#command_input", Input).value or "").strip()
            if not command_text:
                self._set_error("Command is required.")
                return
            hook = {"type": "command", "command": command_text}

        # Handle async option (command only)
        if self._hook_type == "command":
            async_checkbox = self.query_one("#async_checkbox", Checkbox)
            if async_checkbox.value:
                hook["async"] = True

        # Handle once option (command and prompt)
        if self._hook_type in ("command", "prompt"):
            once_checkbox = self.query_one("#once_checkbox", Checkbox)
            if once_checkbox.value:
                hook["once"] = True

        # Handle statusMessage (all types)
        status_message = (self.query_one("#status_message_input", Input).value or "").strip()
        if status_message:
            hook["statusMessage"] = status_message

        timeout_raw = (self.query_one("#timeout_input", Input).value or "").strip()
        if timeout_raw:
            if not timeout_raw.isdigit() or int(timeout_raw) <= 0:
                self._set_error("Timeout must be a positive integer.")
                return
            hook["timeout"] = int(timeout_raw)
        else:
            hook["timeout"] = DEFAULT_HOOK_TIMEOUT

        self.dismiss(HookFormResult(hook=hook))

    def _set_error(self, message: str) -> None:
        app = getattr(self, "app", None)
        if app:
            app.notify(message, title="Validation error", severity="error", timeout=6)


class HooksApp(App[None]):
    CSS = """
    #title_bar {
        height: auto;
        padding: 0 1 0 1;
    }

    #subtitle {
        color: $text-muted;
        padding: 0 1 0 1;
    }

    #status_message {
        color: $text-muted;
        padding: 0 1 0 1;
    }

    #hooks_list {
        margin: 0 1 1 1;
        height: 1fr;
    }

    #hint_bar {
        color: $text-muted;
        padding: 0 1 1 1;
    }

    #form_dialog, #confirm_dialog {
        width: 86;
        max-height: 90%;
        background: $panel;
        border: round $accent;
        padding: 1 2;
    }

    #target_dialog {
        width: 86;
        max-height: 90%;
        background: $panel;
        border: round $accent;
        padding: 1 2;
    }

    #form_title {
        text-style: bold;
        padding: 0 0 1 0;
    }

    #form_fields Input, #form_fields TextArea {
        margin: 0 0 1 0;
    }

    #form_buttons, #confirm_buttons {
        align-horizontal: right;
        padding-top: 1;
        height: auto;
    }

    .field_label {
        color: $text-muted;
        padding: 0 0 0 0;
    }

    .field_value {
        color: $accent;
        padding: 0 0 1 0;
    }

    #confirm_message {
        padding: 0 0 1 0;
    }

    #target_title {
        text-style: bold;
        padding: 0 0 1 0;
    }

    #target_subtitle {
        color: $text-muted;
        padding: 0 0 1 0;
    }

    #target_list {
        margin: 0 0 1 0;
        height: 1fr;
    }

    #target_hint {
        color: $text-muted;
        padding: 0 0 0 0;
    }
    """

    BINDINGS = [
        ("escape", "back", "Back"),
        ("s", "scope", "Scope"),
        ("a", "all_scopes", "All scopes"),
        ("r", "refresh", "Refresh"),
        ("d", "delete", "Delete"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self, project_path: Path) -> None:
        super().__init__()
        self._project_path = project_path
        self._targets = _get_targets(project_path)
        self._target_index = _default_target_index(self._targets)
        self._hooks: Dict[str, List[Dict[str, Any]]] = {}
        self._merged_matcher_refs: Dict[str, List[Dict[str, int]]] = {}
        self._view_mode = "events"
        self._event_name: Optional[str] = None
        self._matcher_index: Optional[int] = None
        self._option_ids: List[str] = []
        self._status_text = ""
        self._show_merged: bool = True
        self._pending_matcher: Optional[str] = None
        self._last_select_option_id: Optional[str] = None
        self._last_select_time: float = 0.0
        self._next_select_is_keyboard: bool = False
        self._double_click_threshold = 0.45

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Static("", id="title_bar")
        yield Static("", id="subtitle")
        yield Static("", id="status_message")
        yield OptionList(id="hooks_list")
        yield Static("", id="hint_bar")
        yield Footer()

    def on_mount(self) -> None:
        self._load_current_scope()
        self._refresh_view()
        self.query_one("#hooks_list", OptionList).focus()

    def action_refresh(self) -> None:
        self._load_current_scope()
        self._set_status("Reloaded hooks.")
        self._refresh_view()

    def action_all_scopes(self) -> None:
        self._show_merged = not self._show_merged
        self._view_mode = "events"
        self._event_name = None
        self._matcher_index = None
        label = "All scopes (merged)" if self._show_merged else self._current_target().label
        self._set_status(f"Scope: {label}")
        self._load_current_scope()
        self._refresh_view()

    def action_scope(self) -> None:
        self._target_index = (self._target_index + 1) % len(self._targets)
        self._load_current_scope()
        self._view_mode = "events"
        self._event_name = None
        self._matcher_index = None
        if self._show_merged:
            self._set_status(f"Default scope: {self._current_target().label}")
        else:
            self._set_status(f"Scope: {self._current_target().label}")
        self._refresh_view()

    async def action_back(self) -> None:
        if len(self.screen_stack) > 1:
            return
        if self._view_mode == "hooks":
            self._view_mode = "matchers"
            self._matcher_index = None
            self._set_status("")
            self._refresh_view()
            return
        if self._view_mode == "matchers":
            self._view_mode = "events"
            self._event_name = None
            self._set_status("")
            self._refresh_view()
            return
        self.exit()

    def action_delete(self) -> None:
        if len(self.screen_stack) > 1:
            return
        option_id = self._current_option_id()
        if not option_id:
            return
        if self._view_mode == "matchers" and option_id.startswith("matcher:"):
            matcher_index = int(option_id.split(":", 1)[1])
            self._delete_matcher(matcher_index)
            return
        if self._view_mode == "hooks" and option_id.startswith("hook:"):
            hook_index = int(option_id.split(":", 1)[1])
            self._delete_hook(hook_index)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id != "hooks_list":
            return
        option_id = event.option.id or ""
        if not self._should_activate_option(option_id):
            return
        if self._view_mode == "events":
            if option_id.startswith("event:"):
                self._event_name = option_id.split(":", 1)[1]
                self._view_mode = "matchers"
                self._set_status("")
                self._refresh_view()
            return

        if self._view_mode == "matchers":
            self._handle_matcher_selection(option_id)
            return

        if self._view_mode == "hooks":
            self._handle_hook_selection(option_id)

    def on_key(self, event: events.Key) -> None:
        if event.key == "enter" and self._hooks_list_focused():
            self._next_select_is_keyboard = True

    def _hooks_list_focused(self) -> bool:
        focused = getattr(self.screen, "focused", None)
        return getattr(focused, "id", None) == "hooks_list"

    def _should_activate_option(self, option_id: str) -> bool:
        if not option_id:
            return False
        if self._next_select_is_keyboard:
            self._next_select_is_keyboard = False
            return True
        now = time.monotonic()
        if (
            option_id == self._last_select_option_id
            and (now - self._last_select_time) <= self._double_click_threshold
        ):
            self._last_select_option_id = None
            self._last_select_time = 0.0
            return True
        self._last_select_option_id = option_id
        self._last_select_time = now
        return False

    def _handle_matcher_selection(self, option_id: str) -> None:
        if not self._event_name:
            return
        if self._show_merged:
            if option_id == "action:add_matcher":
                self.push_screen(
                    MatcherScreen(self._event_name),
                    self._handle_add_matcher_result,
                )
                return
            if option_id.startswith("matcher:"):
                self._matcher_index = int(option_id.split(":", 1)[1])
                self._view_mode = "hooks"
                self._refresh_view()
            return
        if option_id == "action:add_matcher":
            self.push_screen(
                MatcherScreen(self._event_name),
                self._handle_add_matcher_result,
            )
            return
        if option_id == "action:match_all":
            matcher_index = self._match_all_index()
            if matcher_index is None:
                matcher_index = self._create_match_all_matcher()
            self._matcher_index = matcher_index
            self._view_mode = "hooks"
            self._refresh_view()
            return
        if option_id.startswith("matcher:"):
            self._matcher_index = int(option_id.split(":", 1)[1])
            self._view_mode = "hooks"
            self._refresh_view()

    def _handle_hook_selection(self, option_id: str) -> None:
        if option_id == "action:add_command":
            self._open_hook_form("command")
            return
        if option_id == "action:add_prompt":
            self._open_hook_form("prompt")
            return
        if option_id == "action:add_agent":
            self._open_hook_form("agent")
            return
        if option_id.startswith("hook:"):
            hook_index = int(option_id.split(":", 1)[1])
            hook = self._current_hook(hook_index)
            if hook is None:
                return
            hook_type = hook.get("type", "command")
            self._open_hook_form(hook_type, hook_index)

    def _open_hook_form(self, hook_type: str, hook_index: Optional[int] = None) -> None:
        if not self._event_name:
            return
        existing_hook = None
        if hook_index is not None:
            existing_hook = self._current_hook(hook_index)
        screen = HookFormScreen(
            self._event_name,
            hook_type,
            existing_hook=existing_hook,
        )
        self.push_screen(
            screen,
            lambda result: self._handle_hook_form_result(result, hook_index),
        )

    def _handle_add_matcher_result(self, matcher: Optional[str]) -> None:
        if not matcher or not self._event_name:
            return
        if self._show_merged:
            if _matcher_mode(self._event_name) == "none":
                self._set_status("Matchers are fixed to match-all for this event.")
                self._refresh_view()
                return
            self._prompt_matcher_target(matcher, self._target_index)
            return
        matchers = self._hooks.setdefault(self._event_name, [])
        for idx, existing in enumerate(matchers):
            if self._matcher_matches_pattern(existing.get("matcher"), matcher):
                self._matcher_index = idx
                self._view_mode = "hooks"
                self._set_status(f"Matcher '{matcher}' already exists.")
                self._refresh_view()
                return
        matchers.append({"matcher": matcher, "hooks": []})
        if not self._save_current_scope():
            return
        self._matcher_index = len(matchers) - 1
        self._view_mode = "hooks"
        self._set_status(f"Added matcher '{matcher}'.")
        self._refresh_view()

    def _handle_hook_form_result(
        self, result: Optional[HookFormResult], hook_index: Optional[int]
    ) -> None:
        if not result or not self._event_name:
            return
        if hook_index is None:
            self._add_hook_in_current_matcher(result.hook)
            return

        if self._show_merged:
            self._replace_hook_in_scope(result.hook, hook_index)
            return

        matcher = self._current_matcher()
        if matcher is None:
            return
        hooks = matcher.setdefault("hooks", [])
        if 0 <= hook_index < len(hooks):
            hooks[hook_index] = result.hook
            self._set_status("Updated hook.")
        if not self._save_current_scope():
            return
        self._refresh_view()

    def _replace_hook_in_scope(self, hook: Dict[str, Any], hook_index: int) -> None:
        ref = self._current_matcher_ref()
        if ref is None or not self._event_name:
            return
        scope_index = ref["scope_index"]
        matcher_index = ref["matcher_index"]
        target = self._targets[scope_index]
        hooks_data = _load_hooks_json(target.path)
        matchers = hooks_data.get(self._event_name, [])
        if matcher_index < 0 or matcher_index >= len(matchers):
            self._set_status("Selected matcher no longer exists.")
            self._load_current_scope()
            self._refresh_view()
            return
        hooks_list = matchers[matcher_index].get("hooks", [])
        if hook_index < 0 or hook_index >= len(hooks_list):
            self._set_status("Selected hook no longer exists.")
            self._load_current_scope()
            self._refresh_view()
            return
        hooks_list[hook_index] = hook
        if not _save_hooks_json(target.path, hooks_data):
            self._set_status(f"Failed to save hooks to {target.path}")
            self._refresh_view()
            return
        self._set_status(f"Updated hook in {target.label}.")
        self._load_current_scope()
        self._refresh_view()

    def _current_matcher_pattern(self) -> str:
        matcher = self._current_matcher()
        if matcher is None:
            return "*"
        pattern = matcher.get("matcher")
        return str(pattern) if pattern not in (None, "") else "*"

    def _add_hook_in_current_matcher(self, hook: Dict[str, Any]) -> None:
        if not self._event_name:
            return
        matcher = self._current_matcher()
        if matcher is None:
            self._set_status("No matcher selected.")
            self._refresh_view()
            return
        if self._show_merged:
            ref = self._current_matcher_ref()
            if ref is None:
                self._set_status("No matcher selected.")
                self._refresh_view()
                return
            scope_index = ref["scope_index"]
            matcher_index = ref["matcher_index"]
            target = self._targets[scope_index]
            hooks_data = _load_hooks_json(target.path)
            matchers = hooks_data.get(self._event_name, [])
            if matcher_index < 0 or matcher_index >= len(matchers):
                self._set_status("Selected matcher no longer exists.")
                self._load_current_scope()
                self._refresh_view()
                return
            hooks_list = matchers[matcher_index].setdefault("hooks", [])
            hooks_list.append(hook)
            if not _save_hooks_json(target.path, hooks_data):
                self._set_status(f"Failed to save hooks to {target.path}")
                self._refresh_view()
                return
            self._set_status(f"Added hook in {target.label}.")
            self._load_current_scope()
            self._refresh_view()
            return

        hooks_list = matcher.setdefault("hooks", [])
        hooks_list.append(hook)
        self._set_status("Added hook.")
        if not self._save_current_scope():
            return
        self._refresh_view()

    def _prompt_matcher_target(self, matcher: str, preferred_index: int) -> None:
        self._pending_matcher = matcher
        self.push_screen(
            SaveTargetScreen(self._targets, preferred_index),
            self._handle_matcher_target_result,
        )

    def _handle_matcher_target_result(self, target_index: Optional[int]) -> None:
        matcher = self._pending_matcher
        self._pending_matcher = None
        if matcher is None or not self._event_name:
            return
        if target_index is None:
            self._set_status("Matcher creation canceled.")
            self._refresh_view()
            return
        ok, created, hooks_data, matcher_index = self._apply_matcher_to_target(
            matcher, target_index
        )
        if not ok:
            self._refresh_view()
            return
        self._target_index = target_index
        if self._show_merged:
            self._load_current_scope()
            merged_index = self._merged_index_for_ref(
                self._event_name, target_index, matcher_index
            )
            self._matcher_index = merged_index
        else:
            self._hooks = hooks_data
            self._matcher_index = matcher_index
        self._view_mode = "hooks"
        if created:
            self._set_status(f"Added matcher in {self._current_target().label}.")
        else:
            self._set_status(f"Matcher already exists in {self._current_target().label}.")
        self._refresh_view()

    def _apply_matcher_to_target(
        self, matcher_pattern: str, target_index: int
    ) -> tuple[bool, bool, Dict[str, List[Dict[str, Any]]], int]:
        if not self._event_name:
            return False, False, {}, 0
        target = self._targets[target_index]
        hooks_data = _load_hooks_json(target.path)
        matchers = hooks_data.setdefault(self._event_name, [])
        for idx, matcher in enumerate(matchers):
            if self._matcher_matches_pattern(matcher.get("matcher"), matcher_pattern):
                return True, False, hooks_data, idx
        matcher_value = "*" if self._is_match_all(matcher_pattern) else matcher_pattern
        matchers.append({"matcher": matcher_value, "hooks": []})
        matcher_index = len(matchers) - 1
        if not _save_hooks_json(target.path, hooks_data):
            self._set_status(f"Failed to save hooks to {target.path}")
            return False, True, hooks_data, matcher_index
        return True, True, hooks_data, matcher_index

    def _delete_matcher(self, matcher_index: int) -> None:
        if not self._event_name:
            return
        matchers = self._hooks.get(self._event_name, [])
        if matcher_index < 0 or matcher_index >= len(matchers):
            return
        matcher_label = self._matcher_label(matchers[matcher_index])
        self.push_screen(
            ConfirmScreen(f"Delete matcher '{matcher_label}'?"),
            lambda confirmed: self._handle_delete_matcher_confirmed(
                confirmed, matcher_index
            ),
        )

    def _handle_delete_matcher_confirmed(self, confirmed: bool, matcher_index: int) -> None:
        if not confirmed or not self._event_name:
            return
        if self._show_merged:
            self._delete_matcher_in_scope(matcher_index)
            return
        matchers = self._hooks.get(self._event_name, [])
        if matcher_index < 0 or matcher_index >= len(matchers):
            return
        matchers.pop(matcher_index)
        if not matchers:
            self._hooks.pop(self._event_name, None)
        if not self._save_current_scope():
            return
        self._matcher_index = None
        self._view_mode = "matchers"
        self._set_status("Matcher deleted.")
        self._refresh_view()

    def _delete_hook(self, hook_index: int) -> None:
        matcher = self._current_matcher()
        if matcher is None:
            return
        hooks = matcher.get("hooks", [])
        if hook_index < 0 or hook_index >= len(hooks):
            return
        summary = _summarize_hook(hooks[hook_index])
        self.push_screen(
            ConfirmScreen(f"Delete hook '{summary}'?"),
            lambda confirmed: self._handle_delete_hook_confirmed(confirmed, hook_index),
        )

    def _handle_delete_hook_confirmed(self, confirmed: bool, hook_index: int) -> None:
        if not confirmed:
            return
        if self._show_merged:
            self._delete_hook_in_scope(hook_index)
            return
        matcher = self._current_matcher()
        if matcher is None:
            return
        hooks = matcher.get("hooks", [])
        if hook_index < 0 or hook_index >= len(hooks):
            return
        hooks.pop(hook_index)
        if not hooks:
            self._delete_empty_matcher()
        if not self._save_current_scope():
            return
        self._set_status("Hook deleted.")
        self._refresh_view()

    def _delete_hook_in_scope(self, hook_index: int) -> None:
        ref = self._current_matcher_ref()
        if ref is None or not self._event_name:
            return
        scope_index = ref["scope_index"]
        matcher_index = ref["matcher_index"]
        target = self._targets[scope_index]
        hooks_data = _load_hooks_json(target.path)
        matchers = hooks_data.get(self._event_name, [])
        if matcher_index < 0 or matcher_index >= len(matchers):
            self._set_status("Selected matcher no longer exists.")
            self._load_current_scope()
            self._refresh_view()
            return
        hooks_list = matchers[matcher_index].get("hooks", [])
        if hook_index < 0 or hook_index >= len(hooks_list):
            self._set_status("Selected hook no longer exists.")
            self._load_current_scope()
            self._refresh_view()
            return
        hooks_list.pop(hook_index)
        if not hooks_list:
            matchers.pop(matcher_index)
            if not matchers:
                hooks_data.pop(self._event_name, None)
            self._matcher_index = None
            self._view_mode = "matchers"
        if not _save_hooks_json(target.path, hooks_data):
            self._set_status(f"Failed to save hooks to {target.path}")
            self._refresh_view()
            return
        self._set_status(f"Hook deleted from {target.label}.")
        self._load_current_scope()
        self._refresh_view()

    def _delete_matcher_in_scope(self, matcher_index: int) -> None:
        if not self._event_name:
            return
        ref = self._merged_ref_for_index(matcher_index)
        if ref is None:
            return
        scope_index = ref["scope_index"]
        source_index = ref["matcher_index"]
        target = self._targets[scope_index]
        hooks_data = _load_hooks_json(target.path)
        matchers = hooks_data.get(self._event_name, [])
        if source_index < 0 or source_index >= len(matchers):
            self._set_status("Selected matcher no longer exists.")
            self._load_current_scope()
            self._refresh_view()
            return
        matchers.pop(source_index)
        if not matchers:
            hooks_data.pop(self._event_name, None)
        if not _save_hooks_json(target.path, hooks_data):
            self._set_status(f"Failed to save hooks to {target.path}")
            self._refresh_view()
            return
        self._matcher_index = None
        self._view_mode = "matchers"
        self._set_status(f"Matcher deleted from {target.label}.")
        self._load_current_scope()
        self._refresh_view()

    def _delete_empty_matcher(self) -> None:
        if not self._event_name:
            return
        matchers = self._hooks.get(self._event_name, [])
        if not matchers or self._matcher_index is None:
            return
        if 0 <= self._matcher_index < len(matchers) and not matchers[self._matcher_index].get(
            "hooks"
        ):
            matchers.pop(self._matcher_index)
            if not matchers:
                self._hooks.pop(self._event_name, None)
            self._matcher_index = None
            self._view_mode = "matchers"

    def _refresh_view(self) -> None:
        self._update_title_bar()
        self._update_subtitle()
        self._update_status_message()
        self._refresh_list()
        self._update_hint_bar()

    def _update_title_bar(self) -> None:
        title_bar = self.query_one("#title_bar", Static)
        if self._view_mode == "events":
            title_bar.update("Hooks")
            return
        if self._view_mode == "matchers" and self._event_name:
            matcher_mode = _matcher_mode(self._event_name)
            if matcher_mode == "tool":
                title = f"{self._event_name} - Tool Matchers"
            elif matcher_mode == "none":
                title = f"{self._event_name} - Hook Groups"
            else:
                title = f"{self._event_name} - Matchers"
            title_bar.update(title)
            return
        if self._view_mode == "hooks" and self._event_name:
            matcher_label = self._current_matcher_label()
            title_bar.update(f"{self._event_name} - Hooks ({matcher_label})")
            return

    def _update_subtitle(self) -> None:
        subtitle = self.query_one("#subtitle", Static)
        if self._view_mode == "events":
            target = self._current_target()
            total_hooks = _count_hooks(self._hooks)
            if self._show_merged:
                paths = [
                    f"Local: {self._targets[0].path}",
                    f"Project: {self._targets[1].path}",
                    f"User: {self._targets[2].path}",
                ]
                subtitle.update(
                    "\n".join(
                        [
                            f"{total_hooks} hooks | Scope: All scopes (merged)",
                            f"Default save scope: {target.label}",
                            *paths,
                        ]
                    )
                )
            else:
                subtitle.update(
                    f"{total_hooks} hooks | Scope: {target.label}\n{target.path}"
                )
            return

        if not self._event_name:
            subtitle.update("")
            return

        lines: List[str] = []
        description = EVENT_DESCRIPTIONS.get(self._event_name, "")
        if description:
            lines.append(description)
        if self._show_merged:
            lines.append("Merged view: entries include Project/Project (Local)/User settings labels.")
        matcher_mode = _matcher_mode(self._event_name)
        if matcher_mode == "tool":
            lines.append("Matcher: tool_name (regex ok). Blank/'*' matches all tools.")
        elif matcher_mode == "enum":
            choices = _matcher_choices(self._event_name)
            if choices:
                lines.append(
                    "Matcher values: " + ", ".join(choices) + " (blank/'*' matches all)."
                )
        elif matcher_mode == "text":
            lines.append(
                f"Matcher: {_matcher_field_name(self._event_name)} (blank/'*' matches all)."
            )
        elif matcher_mode == "always":
            lines.append("Matcher: ignored for this event (all match).")
        else:
            lines.append("Matcher: match-all only for this event.")
        lines.extend(
            [
                "Input to command is JSON of hook input.",
                "Exit code 0 - stdout/stderr not shown",
                "Exit code 2 - show stderr to model and block hook",
                "Other exit codes - show stderr to user only but continue",
            ]
        )
        subtitle.update("\n".join(lines))

    def _update_status_message(self) -> None:
        self.query_one("#status_message", Static).update(self._status_text)

    def _update_hint_bar(self) -> None:
        hint = self.query_one("#hint_bar", Static)
        if self._view_mode == "events":
            hint.update("Double-click/Enter to open | S to switch scope | A to toggle all | Q to quit")
        elif self._view_mode == "matchers":
            hint.update("Double-click/Enter to open | D to delete matcher | A to toggle all | Esc to go back")
        else:
            hint.update("Double-click/Enter to edit | D to delete | A to toggle all | Esc to go back")

    def _refresh_list(self) -> None:
        option_list = self.query_one("#hooks_list", OptionList)
        option_list.clear_options()
        self._option_ids = []

        if self._view_mode == "events":
            self._render_events(option_list)
        elif self._view_mode == "matchers":
            self._render_matchers(option_list)
        else:
            self._render_hooks(option_list)

        if option_list.option_count:
            option_list.highlighted = 0

    def _render_events(self, option_list: OptionList) -> None:
        for idx, event in enumerate(HookEvent, start=1):
            event_name = event.value
            desc = EVENT_DESCRIPTIONS.get(event_name, "")
            label = f"{idx}. {event_name}"
            if desc:
                label = f"{label} - {desc}"
            option_id = f"event:{event_name}"
            self._option_ids.append(option_id)
            option_list.add_option(Option(label, id=option_id))

    def _render_matchers(self, option_list: OptionList) -> None:
        if not self._event_name:
            return
        matchers = self._hooks.get(self._event_name, [])
        allow_matchers = _matcher_mode(self._event_name) != "none"
        if self._show_merged:
            display_index = 1
            if allow_matchers:
                option_list.add_option(Option("1. + Add new matcher...", id="action:add_matcher"))
                self._option_ids.append("action:add_matcher")
                display_index += 1

            if not matchers:
                if not self._status_text:
                    self._set_status("No matchers configured in any scope.")
                return

            for idx, matcher in enumerate(matchers):
                scope_label = self._merged_scope_label(idx)
                label = (
                    f"{display_index}. {self._matcher_label(matcher)}"
                    f" - {self._scope_settings_label(scope_label)}"
                )
                option_id = f"matcher:{idx}"
                self._option_ids.append(option_id)
                option_list.add_option(Option(label, id=option_id))
                display_index += 1
            return
        if allow_matchers:
            option_list.add_option(Option("1. + Add new matcher...", id="action:add_matcher"))
            self._option_ids.append("action:add_matcher")
            option_list.add_option(Option("2. + Match all (no filter)", id="action:match_all"))
            self._option_ids.append("action:match_all")
            offset = 3
        else:
            option_list.add_option(Option("1. Match all (no filter)", id="action:match_all"))
            self._option_ids.append("action:match_all")
            offset = 2

        if not matchers:
            if not self._status_text:
                self._set_status("No matchers configured yet.")
            return

        display_index = offset
        has_display = False
        for matcher_index, matcher in enumerate(matchers):
            if self._is_match_all(matcher.get("matcher")):
                continue
            label = f"{display_index}. {self._matcher_label(matcher)}"
            option_id = f"matcher:{matcher_index}"
            self._option_ids.append(option_id)
            option_list.add_option(Option(label, id=option_id))
            display_index += 1
            has_display = True

        if not has_display and not self._status_text:
            self._set_status("Only match-all matcher configured.")

    def _render_hooks(self, option_list: OptionList) -> None:
        matcher = self._current_matcher()
        if matcher is None:
            self._set_status("No matcher selected.")
            return
        option_index = 1
        option_list.add_option(Option(f"{option_index}. + Add command hook...", id="action:add_command"))
        self._option_ids.append("action:add_command")
        option_index += 1
        if self._event_name and self._event_name in PROMPT_SUPPORTED_EVENTS:
            option_list.add_option(
                Option(f"{option_index}. + Add prompt hook...", id="action:add_prompt")
            )
            self._option_ids.append("action:add_prompt")
            option_index += 1
            option_list.add_option(
                Option(f"{option_index}. + Add agent hook...", id="action:add_agent")
            )
            self._option_ids.append("action:add_agent")
            option_index += 1

        hooks = matcher.get("hooks", [])
        if not hooks:
            if not self._status_text:
                self._set_status("No hooks configured yet.")
            return

        for idx, hook in enumerate(hooks, start=option_index):
            label = f"{idx}. {_summarize_hook(hook)}"
            option_id = f"hook:{idx - option_index}"
            self._option_ids.append(option_id)
            option_list.add_option(Option(label, id=option_id))

    def _current_option_id(self) -> Optional[str]:
        option_list = self.query_one("#hooks_list", OptionList)
        if option_list.highlighted is None:
            return None
        idx = option_list.highlighted
        if idx < 0 or idx >= len(self._option_ids):
            return None
        return self._option_ids[idx]

    def _current_target(self) -> HookConfigTarget:
        return self._targets[self._target_index]

    def _current_matcher(self) -> Optional[Dict[str, Any]]:
        if not self._event_name:
            return None
        matchers = self._hooks.get(self._event_name, [])
        if not matchers:
            return None
        if self._matcher_index is None:
            matcher_index = self._match_all_index()
            if matcher_index is None:
                matcher_index = 0
            self._matcher_index = matcher_index
        if 0 <= self._matcher_index < len(matchers):
            return matchers[self._matcher_index]
        return None

    def _current_hook(self, hook_index: int) -> Optional[Dict[str, Any]]:
        matcher = self._current_matcher()
        if matcher is None:
            return None
        hooks = matcher.get("hooks", [])
        if 0 <= hook_index < len(hooks):
            hook = hooks[hook_index]
            if isinstance(hook, dict):
                return hook
        return None

    def _current_matcher_label(self) -> str:
        matcher = self._current_matcher()
        if matcher is None:
            return "*"
        label = self._matcher_label(matcher)
        if self._show_merged:
            scope_label = self._current_matcher_scope_label()
            return f"{label} - {self._scope_settings_label(scope_label)}"
        return label

    def _current_matcher_scope_label(self) -> str:
        scope_index = self._current_matcher_scope_index()
        if scope_index is None:
            return self._scope_label_for_index(self._target_index)
        return self._scope_label_for_index(scope_index)

    def _merged_scope_label(self, matcher_index: int) -> str:
        if not self._event_name:
            return self._scope_label_for_index(self._target_index)
        refs = self._merged_matcher_refs.get(self._event_name, [])
        if matcher_index < 0 or matcher_index >= len(refs):
            return self._scope_label_for_index(self._target_index)
        scope_index = refs[matcher_index]["scope_index"]
        return self._scope_label_for_index(scope_index)

    def _current_matcher_scope_index(self) -> Optional[int]:
        ref = self._current_matcher_ref()
        if ref is None:
            return None
        return ref["scope_index"]

    def _current_matcher_ref(self) -> Optional[Dict[str, int]]:
        if not self._show_merged or not self._event_name:
            return None
        refs = self._merged_matcher_refs.get(self._event_name, [])
        if self._matcher_index is None:
            return None
        if 0 <= self._matcher_index < len(refs):
            return refs[self._matcher_index]
        return None

    def _merged_ref_for_index(self, matcher_index: int) -> Optional[Dict[str, int]]:
        if not self._show_merged or not self._event_name:
            return None
        refs = self._merged_matcher_refs.get(self._event_name, [])
        if 0 <= matcher_index < len(refs):
            return refs[matcher_index]
        return None

    def _merged_index_for_ref(
        self, event_name: str, scope_index: int, matcher_index: int
    ) -> Optional[int]:
        refs = self._merged_matcher_refs.get(event_name, [])
        for idx, ref in enumerate(refs):
            if ref["scope_index"] == scope_index and ref["matcher_index"] == matcher_index:
                return idx
        return None

    def _scope_label_for_index(self, scope_index: int) -> str:
        key = self._targets[scope_index].key
        if key in ("global", "user"):
            return "User"
        if key == "local":
            return "Project (Local)"
        return "Project"

    @staticmethod
    def _scope_settings_label(scope_label: str) -> str:
        if scope_label == "Project (Local)":
            return "Project Settings (Local)"
        return f"{scope_label} Settings"

    def _matcher_label(self, matcher: Dict[str, Any]) -> str:
        label = matcher.get("matcher")
        return str(label) if label not in (None, "") else "*"

    def _match_all_index(self) -> Optional[int]:
        if not self._event_name:
            return None
        matchers = self._hooks.get(self._event_name, [])
        for idx, matcher in enumerate(matchers):
            if self._is_match_all(matcher.get("matcher")):
                return idx
        return None

    def _create_match_all_matcher(self) -> int:
        if not self._event_name:
            return 0
        matchers = self._hooks.setdefault(self._event_name, [])
        matchers.append({"matcher": "*", "hooks": []})
        self._save_current_scope()
        return len(matchers) - 1

    @staticmethod
    def _is_match_all(pattern: Optional[str]) -> bool:
        return pattern in (None, "", "*")

    @staticmethod
    def _matcher_matches_pattern(existing: Optional[str], desired: str) -> bool:
        if HooksApp._is_match_all(desired):
            return HooksApp._is_match_all(existing)
        return existing == desired

    def _load_current_scope(self) -> None:
        if self._show_merged:
            self._load_merged_scopes()
        else:
            self._hooks = _load_hooks_json(self._current_target().path)
            self._merged_matcher_refs = {}

    def _save_current_scope(self) -> bool:
        if self._show_merged:
            self._set_status("Switch to a single scope to save changes.")
            return False
        target = self._current_target()
        ok = _save_hooks_json(target.path, self._hooks)
        if not ok:
            self._set_status(f"Failed to save hooks to {target.path}")
        return ok

    def _load_merged_scopes(self) -> None:
        merged_hooks: Dict[str, List[Dict[str, Any]]] = {}
        merged_refs: Dict[str, List[Dict[str, int]]] = {}
        for scope_index, target in enumerate(self._targets):
            hooks_data = _load_hooks_json(target.path)
            for event_name, matchers in hooks_data.items():
                for matcher_index, matcher in enumerate(matchers):
                    merged_hooks.setdefault(event_name, []).append(
                        {
                            "matcher": matcher.get("matcher"),
                            "hooks": list(matcher.get("hooks", [])),
                        }
                    )
                    merged_refs.setdefault(event_name, []).append(
                        {
                            "scope_index": scope_index,
                            "matcher_index": matcher_index,
                        }
                    )
        self._hooks = merged_hooks
        self._merged_matcher_refs = merged_refs

    def _set_status(self, message: str) -> None:
        self._status_text = message


def _get_targets(project_path: Path) -> List[HookConfigTarget]:
    return [
        HookConfigTarget(
            key="local",
            label="Local (.ripperdoc/hooks.local.json)",
            path=get_project_local_hooks_path(project_path),
            description="Git-ignored hooks for this project",
        ),
        HookConfigTarget(
            key="project",
            label="Project (.ripperdoc/hooks.json)",
            path=get_project_hooks_path(project_path),
            description="Shared hooks committed to the repo",
        ),
        HookConfigTarget(
            key="user",
            label="User (~/.ripperdoc/hooks.json)",
            path=get_global_hooks_path(),
            description="Applies to all projects on this machine",
        ),
    ]


def _default_target_index(targets: List[HookConfigTarget]) -> int:
    for idx, target in enumerate(targets):
        if target.path.exists():
            return idx
    return 0


def _load_hooks_json(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    if not path.exists():
        return {}

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, IOError, PermissionError):
        return {}

    hooks_section = {}
    if isinstance(raw, dict):
        hooks_section = raw.get("hooks", raw)

    if not isinstance(hooks_section, dict):
        return {}

    hooks: Dict[str, List[Dict[str, Any]]] = {}
    for event_name, matchers in hooks_section.items():
        if not isinstance(matchers, list):
            continue
        cleaned_matchers: List[Dict[str, Any]] = []
        for matcher in matchers:
            if not isinstance(matcher, dict):
                continue
            hooks_list = matcher.get("hooks", [])
            if not isinstance(hooks_list, list):
                continue
            cleaned_hooks = [h for h in hooks_list if isinstance(h, dict)]
            cleaned_matchers.append({"matcher": matcher.get("matcher"), "hooks": cleaned_hooks})
        if cleaned_matchers:
            hooks[event_name] = cleaned_matchers

    return hooks


def _save_hooks_json(path: Path, hooks: Dict[str, List[Dict[str, Any]]]) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps({"hooks": hooks}, indent=2, ensure_ascii=False)
        path.write_text(serialized, encoding="utf-8")
        return True
    except (OSError, IOError, PermissionError):
        return False


def _summarize_hook(hook: Dict[str, Any]) -> str:
    hook_type = hook.get("type", "command")
    if hook_type == "prompt":
        text = hook.get("prompt", "") or "(prompt missing)"
        label = "prompt"
    elif hook_type == "agent":
        text = hook.get("prompt", "") or "(prompt missing)"
        label = "agent"
    else:
        text = hook.get("command", "") or "(command missing)"
        label = "command"

    text = text.replace("\n", "\\n")
    if len(text) > 60:
        text = text[:57] + "..."

    timeout = hook.get("timeout", DEFAULT_HOOK_TIMEOUT)
    model = hook.get("model") if hook_type == "agent" else None
    is_async = hook.get("async", False)
    run_once = hook.get("once", False)
    status_msg = hook.get("statusMessage", "")

    # Build options suffix
    options = []
    if is_async:
        options.append("async")
    if run_once:
        options.append("once")
    if status_msg:
        truncated_msg = status_msg[:15] + "..." if len(status_msg) > 15 else status_msg
        options.append(f"msg:{truncated_msg}")

    # Build base label
    base = f"{label}: {text} ({timeout}s)"
    if model:
        base += f", model={model}"

    # Append options if present
    if options:
        return f"{base}, [{', '.join(options)}]"
    return base


def _count_hooks(hooks: Dict[str, List[Dict[str, Any]]]) -> int:
    total = 0
    for matchers in hooks.values():
        for matcher in matchers:
            total += len(matcher.get("hooks", []))
    return total


def run_hooks_tui(project_path: Path, on_exit: Optional[Callable[[], Any]] = None) -> bool:
    """Run the Textual hooks TUI."""
    app = HooksApp(project_path)
    app.run()
    if on_exit:
        on_exit()
    return True
