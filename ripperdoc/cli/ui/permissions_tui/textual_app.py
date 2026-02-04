"""Textual app for managing permission rules."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from rich.text import Text

from textual import events
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Header, Input, OptionList, Static
from textual.widgets.option_list import Option

from ripperdoc.core.config import (
    UserConfig,
    ProjectConfig,
    ProjectLocalConfig,
    get_global_config,
    get_project_config,
    get_project_local_config,
    save_global_config,
    save_project_config,
    save_project_local_config,
)


ScopeType = str
RuleType = str


@dataclass
class RuleSelection:
    rule: str


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


class AddRuleScreen(ModalScreen[Optional[str]]):
    """Modal screen for adding permission rules."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, rule_type: str) -> None:
        super().__init__()
        self._rule_type = rule_type

    def compose(self) -> ComposeResult:
        title = f"Add {self._rule_type} permission rule"
        with Container(id="add_dialog"):
            yield Static(title, id="add_title")
            yield Static(
                "Permission rules are a tool name, optionally followed by a specifier in parentheses.",
                id="add_hint",
            )
            yield Static("e.g., WebFetch or Bash(ls:*)", id="add_example")
            yield Static("", id="add_error")
            with VerticalScroll(id="add_fields"):
                yield Input(placeholder="Enter permission rule...", id="rule_input")
            with Horizontal(id="add_buttons"):
                yield Button("Add", id="add_submit", variant="primary")
                yield Button("Cancel", id="add_cancel")

    def on_mount(self) -> None:
        self.query_one("#rule_input", Input).focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "add_cancel":
            self.dismiss(None)
            return
        if event.button.id != "add_submit":
            return
        raw = (self.query_one("#rule_input", Input).value or "").strip()
        if not raw:
            self._set_error("Rule cannot be empty.")
            return
        self.dismiss(raw)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "rule_input":
            return
        raw = (event.value or "").strip()
        if not raw:
            self._set_error("Rule cannot be empty.")
            return
        self.dismiss(raw)

    def _set_error(self, message: str) -> None:
        self.query_one("#add_error", Static).update(message)


class PermissionsApp(App[None]):
    CSS = """
    #status_bar {
        height: auto;
        padding: 0 1;
    }

    #description {
        color: $text-muted;
        padding: 0 1 0 1;
    }

    #status_message {
        color: $text-muted;
        padding: 0 1 0 1;
    }

    #search_input {
        margin: 0 1 0 1;
    }

    #rules_list {
        margin: 0 1 1 1;
        height: 1fr;
    }

    #hint_bar {
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

    #add_hint, #add_example {
        color: $text-muted;
        padding: 0 0 1 0;
    }

    #add_error {
        color: $error;
        padding: 0 0 1 0;
    }

    #add_buttons, #confirm_buttons {
        align-horizontal: right;
        padding-top: 1;
        height: auto;
    }
    """

    BINDINGS = [
        ("left", "prev_type", "Prev type"),
        ("right", "next_type", "Next type"),
        ("tab", "next_type", "Next type"),
        ("shift+tab", "prev_type", "Prev type"),
        ("shift+left", "prev_scope", "Prev scope"),
        ("shift+right", "next_scope", "Next scope"),
        ("q", "quit", "Quit"),
    ]

    _TYPE_ORDER = ("allow", "ask", "deny")
    _SCOPE_ORDER = ("project", "user", "local")

    def __init__(self, project_path: Path) -> None:
        super().__init__()
        self._project_path = project_path
        self._rule_type: RuleType = "allow"
        self._scope: ScopeType = "project"
        self._search_query: str = ""
        self._rule_map: dict[str, str] = {}

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Static("", id="status_bar")
        yield Static("", id="description")
        yield Static("", id="status_message")
        yield Input(placeholder="Search...", id="search_input")
        yield OptionList(id="rules_list")
        yield Static(
            "Press ↑↓ to navigate · Enter to select · Type to search · Esc to clear search",
            id="hint_bar",
        )
        yield Footer()

    def on_mount(self) -> None:
        search_input = self.query_one("#search_input", Input)
        search_input.disabled = True
        self._refresh_view()
        self.query_one("#rules_list", OptionList).focus()

    def on_key(self, event: events.Key) -> None:
        if len(self.screen_stack) > 1:
            return
        if event.key == "tab":
            self.action_next_type()
            event.stop()
            return
        if event.key == "shift+tab":
            self.action_prev_type()
            event.stop()
            return
        if event.key == "escape":
            if self._search_query:
                self._update_search("")
                event.stop()
            return
        if event.key == "backspace":
            if self._search_query:
                self._update_search(self._search_query[:-1])
                event.stop()
            return
        if event.character and event.character.isprintable():
            if event.character not in ("\t", "\n"):
                self._update_search(self._search_query + event.character)
                event.stop()

    def action_next_type(self) -> None:
        idx = self._TYPE_ORDER.index(self._rule_type)
        self._rule_type = self._TYPE_ORDER[(idx + 1) % len(self._TYPE_ORDER)]
        self._refresh_view()

    def action_prev_type(self) -> None:
        idx = self._TYPE_ORDER.index(self._rule_type)
        self._rule_type = self._TYPE_ORDER[(idx - 1) % len(self._TYPE_ORDER)]
        self._refresh_view()

    def action_next_scope(self) -> None:
        idx = self._SCOPE_ORDER.index(self._scope)
        self._scope = self._SCOPE_ORDER[(idx + 1) % len(self._SCOPE_ORDER)]
        self._refresh_view()

    def action_prev_scope(self) -> None:
        idx = self._SCOPE_ORDER.index(self._scope)
        self._scope = self._SCOPE_ORDER[(idx - 1) % len(self._SCOPE_ORDER)]
        self._refresh_view()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        option_id = event.option.id or ""
        if option_id == "__add__":
            self.push_screen(AddRuleScreen(self._rule_type), self._handle_add_result)
            return
        if option_id.startswith("rule:"):
            rule = self._rule_map.get(option_id)
            if not rule:
                return
            self.push_screen(
                ConfirmScreen(f"Remove rule '{self._format_rule_display(rule)}'?"),
                lambda confirmed: self._handle_remove_result(confirmed, rule),
            )

    def _handle_add_result(self, rule_input: Optional[str]) -> None:
        if not rule_input:
            return
        rule = self._normalize_rule_input(rule_input)
        if not rule:
            self._set_status("Rule cannot be empty.")
            return
        if self._add_rule(self._scope, self._rule_type, rule):
            self._set_status(f"Added {self._rule_type} rule.")
        else:
            self._set_status("Rule already exists.")
        self._refresh_view()

    def _handle_remove_result(self, confirmed: bool, rule: str) -> None:
        if not confirmed:
            return
        if self._remove_rule(self._scope, self._rule_type, rule):
            self._set_status("Rule removed.")
        else:
            self._set_status("Rule not found.")
        self._refresh_view()

    def _refresh_view(self) -> None:
        self._update_status_bar()
        self._update_description()
        self._update_search_input()
        self._refresh_list()

    def _update_status_bar(self) -> None:
        status = self.query_one("#status_bar", Static)
        type_parts = []
        for item in self._TYPE_ORDER:
            label = item.capitalize() if item != "ask" else "Ask"
            if item == self._rule_type:
                label = f"[reverse bold]{label}[/reverse bold]"
            type_parts.append(label)
        status.update(
            f"Permissions:  {'   '.join(type_parts)}  (←/→ or tab to cycle)"
        )

    def _update_description(self) -> None:
        description = self.query_one("#description", Static)
        if self._rule_type == "allow":
            text = "Ripperdoc won't ask before using allowed tools."
        elif self._rule_type == "deny":
            text = "Ripperdoc will block matching tools."
        else:
            text = "Ripperdoc will always ask for matching tools."
        description.update(text)

    def _update_search_input(self) -> None:
        search_input = self.query_one("#search_input", Input)
        search_input.value = self._search_query

    def _refresh_list(self) -> None:
        option_list = self.query_one("#rules_list", OptionList)
        option_list.clear_options()
        self._rule_map = {}

        allow_rules, deny_rules, ask_rules = self._get_rules_for_scope(self._scope)
        if self._rule_type == "allow":
            rules = allow_rules
        elif self._rule_type == "deny":
            rules = deny_rules
        else:
            rules = ask_rules
        filtered = self._filter_rules(rules, self._search_query)

        option_list.add_option(Option("1. Add a new rule...", id="__add__"))
        for idx, rule in enumerate(filtered, start=2):
            display = self._format_rule_display(rule)
            option_id = f"rule:{idx}"
            self._rule_map[option_id] = rule
            option_list.add_option(Option(f"{idx}. {display}", id=option_id))

        if option_list.option_count:
            option_list.highlighted = 0

    def _update_search(self, query: str) -> None:
        self._search_query = query
        self._refresh_view()

    @staticmethod
    def _filter_rules(rules: list[str], query: str) -> list[str]:
        if not query:
            return list(rules)
        needle = query.lower()
        filtered = []
        for rule in rules:
            display = PermissionsApp._format_rule_display(rule).lower()
            if needle in rule.lower() or needle in display:
                filtered.append(rule)
        return filtered

    @staticmethod
    def _scope_label(scope: ScopeType) -> str:
        if scope == "project":
            return "Workspace"
        if scope == "user":
            return "User"
        return "Local"

    @staticmethod
    def _parse_tool_rule(rule: str) -> tuple[Optional[str], Optional[str]]:
        match = re.match(r"\s*([A-Za-z0-9_-]+)\s*\((.*)\)\s*$", rule)
        if not match:
            return None, None
        return match.group(1), match.group(2)

    @staticmethod
    def _normalize_rule_input(rule: str) -> str:
        rule = rule.strip()
        tool, inner = PermissionsApp._parse_tool_rule(rule)
        if tool and inner is not None:
            if tool.lower() == "bash":
                return inner.strip()
            return rule
        return rule

    @staticmethod
    def _format_rule_display(rule: str) -> str:
        tool, _ = PermissionsApp._parse_tool_rule(rule)
        if tool:
            return rule
        return f"Bash({rule})"

    def _get_rules_for_scope(self, scope: ScopeType) -> tuple[list[str], list[str], list[str]]:
        if scope == "user":
            user_config: UserConfig = get_global_config()
            return (
                list(user_config.user_allow_rules),
                list(user_config.user_deny_rules),
                list(user_config.user_ask_rules),
            )
        if scope == "project":
            project_config: ProjectConfig = get_project_config(self._project_path)
            return (
                list(project_config.bash_allow_rules),
                list(project_config.bash_deny_rules),
                list(project_config.bash_ask_rules),
            )
        local_config: ProjectLocalConfig = get_project_local_config(self._project_path)
        return (
            list(local_config.local_allow_rules),
            list(local_config.local_deny_rules),
            list(local_config.local_ask_rules),
        )

    def _add_rule(self, scope: ScopeType, rule_type: RuleType, rule: str) -> bool:
        if scope == "user":
            user_config: UserConfig = get_global_config()
            if rule_type == "allow":
                rules = user_config.user_allow_rules
            elif rule_type == "deny":
                rules = user_config.user_deny_rules
            else:
                rules = user_config.user_ask_rules
            if rule in rules:
                return False
            rules.append(rule)
            save_global_config(user_config)
            return True
        if scope == "project":
            project_config: ProjectConfig = get_project_config(self._project_path)
            if rule_type == "allow":
                rules = project_config.bash_allow_rules
            elif rule_type == "deny":
                rules = project_config.bash_deny_rules
            else:
                rules = project_config.bash_ask_rules
            if rule in rules:
                return False
            rules.append(rule)
            save_project_config(project_config, self._project_path)
            return True
        local_config: ProjectLocalConfig = get_project_local_config(self._project_path)
        if rule_type == "allow":
            rules = local_config.local_allow_rules
        elif rule_type == "deny":
            rules = local_config.local_deny_rules
        else:
            rules = local_config.local_ask_rules
        if rule in rules:
            return False
        rules.append(rule)
        save_project_local_config(local_config, self._project_path)
        return True

    def _remove_rule(self, scope: ScopeType, rule_type: RuleType, rule: str) -> bool:
        if scope == "user":
            user_config: UserConfig = get_global_config()
            if rule_type == "allow":
                rules = user_config.user_allow_rules
            elif rule_type == "deny":
                rules = user_config.user_deny_rules
            else:
                rules = user_config.user_ask_rules
            if rule not in rules:
                return False
            rules.remove(rule)
            save_global_config(user_config)
            return True
        if scope == "project":
            project_config: ProjectConfig = get_project_config(self._project_path)
            if rule_type == "allow":
                rules = project_config.bash_allow_rules
            elif rule_type == "deny":
                rules = project_config.bash_deny_rules
            else:
                rules = project_config.bash_ask_rules
            if rule not in rules:
                return False
            rules.remove(rule)
            save_project_config(project_config, self._project_path)
            return True
        local_config: ProjectLocalConfig = get_project_local_config(self._project_path)
        if rule_type == "allow":
            rules = local_config.local_allow_rules
        elif rule_type == "deny":
            rules = local_config.local_deny_rules
        else:
            rules = local_config.local_ask_rules
        if rule not in rules:
            return False
        rules.remove(rule)
        save_project_local_config(local_config, self._project_path)
        return True

    def _set_status(self, message: str) -> None:
        status = self.query_one("#status_message", Static)
        scope_label = self._scope_label(self._scope)
        if scope_label != "Workspace" and message:
            status.update(f"{message}  Scope: {scope_label}")
        elif scope_label != "Workspace":
            status.update(f"Scope: {scope_label}")
        else:
            status.update(message)


def run_permissions_tui(
    project_path: Path, on_exit: Optional[Callable[[], Any]] = None
) -> bool:
    """Run the Textual permissions TUI."""
    app = PermissionsApp(project_path)
    app.run()
    if on_exit:
        on_exit()
    return True
