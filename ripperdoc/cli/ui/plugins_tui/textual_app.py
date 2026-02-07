"""Textual app for plugin discovery and management."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import List, Optional

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Input, OptionList, Static
from textual.widgets.option_list import Option

from ripperdoc.core.plugin_marketplaces import (
    MarketplaceSource,
    PluginCatalogEntry,
    add_marketplace,
    discover_marketplace_plugins,
    install_marketplace_plugin,
    load_marketplaces,
    remove_marketplace,
)
from ripperdoc.core.plugins import PluginSettingsScope, discover_plugins


@dataclass(frozen=True)
class _InstalledPlugin:
    name: str
    path: Path
    description: str
    version: Optional[str]


class _PluginDetailsScreen(ModalScreen[Optional[str]]):
    def __init__(self, title: str, body: str, *, installed: bool) -> None:
        super().__init__()
        self._title = title
        self._body = body
        self._installed = installed

    def compose(self) -> ComposeResult:
        with Container(id="details_dialog"):
            yield Static(self._title, id="details_title")
            yield Static(self._body, id="details_body")
            with Horizontal(id="details_buttons"):
                if self._installed:
                    yield Button("Uninstall", id="details_uninstall", variant="warning")
                else:
                    yield Button("Install", id="details_install", variant="primary")
                yield Button("Close", id="details_close", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "details_install":
            self.dismiss("install")
            return
        if event.button.id == "details_uninstall":
            self.dismiss("uninstall")
            return
        if event.button.id == "details_close":
            self.dismiss("close")


class _MarketplaceDetailsScreen(ModalScreen[Optional[str]]):
    def __init__(self, title: str, body: str, *, can_remove: bool) -> None:
        super().__init__()
        self._title = title
        self._body = body
        self._can_remove = can_remove

    def compose(self) -> ComposeResult:
        with Container(id="details_dialog"):
            yield Static(self._title, id="details_title")
            yield Static(self._body, id="details_body")
            with Horizontal(id="details_buttons"):
                yield Button("Update", id="details_update", variant="primary")
                if self._can_remove:
                    yield Button("Uninstall", id="details_remove", variant="warning")
                yield Button("Close", id="details_close", variant="default")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "details_update":
            self.dismiss("update")
            return
        if event.button.id == "details_remove":
            self.dismiss("remove")
            return
        if event.button.id == "details_close":
            self.dismiss("close")


class _AddMarketplaceScreen(ModalScreen[Optional[str]]):
    def compose(self) -> ComposeResult:
        with Container(id="add_marketplace_dialog"):
            yield Static("Add Marketplace", id="add_marketplace_title")
            yield Static(
                "Enter marketplace source:\n"
                "• owner/repo (GitHub)\n"
                "• git@github.com:owner/repo.git (SSH)\n"
                "• https://example.com/marketplace.json\n"
                "• ./path/to/marketplace",
                id="add_marketplace_help",
            )
            yield Input(placeholder="Marketplace source", id="add_marketplace_input")
            with Horizontal(id="add_marketplace_buttons"):
                yield Button("Add", id="add_marketplace_submit", variant="primary")
                yield Button("Cancel", id="add_marketplace_cancel")

    def on_mount(self) -> None:
        self.query_one("#add_marketplace_input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "add_marketplace_input":
            return
        value = event.value.strip()
        self.dismiss(value or None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "add_marketplace_cancel":
            self.dismiss(None)
            return
        if event.button.id == "add_marketplace_submit":
            raw = self.query_one("#add_marketplace_input", Input).value.strip()
            self.dismiss(raw or None)


class PluginsApp(App[None]):
    CSS = """
    Screen {
        layout: vertical;
    }

    #header {
        text-style: bold;
        padding: 1 1 0 1;
    }

    #tabs_row {
        height: auto;
        padding: 0 1;
    }

    .tab-btn {
        margin-right: 1;
        min-width: 16;
    }

    .tab-active {
        text-style: bold;
    }

    #subtitle {
        padding: 0 1;
        color: $text;
    }

    #search {
        margin: 1 1 0 1;
    }

    #content_row {
        layout: horizontal;
        height: 1fr;
        margin: 0 1 0 1;
    }

    #plugin_list {
        width: 60%;
        height: 1fr;
        margin-right: 1;
        border: round $surface;
    }

    #details_panel {
        width: 40%;
        height: 1fr;
        border: round $surface;
        padding: 1;
        color: $text;
    }

    #hint {
        color: $text-muted;
        padding: 0 1;
    }

    #details_dialog, #add_marketplace_dialog {
        background: $panel;
        border: round $primary;
        padding: 1 2;
        width: 85%;
        max-width: 110;
        height: auto;
    }

    #details_body {
        padding: 1 0;
    }

    #details_buttons, #add_marketplace_buttons {
        align: right middle;
        padding-top: 1;
    }
    """

    BINDINGS = [
        ("tab", "next_tab", "Next tab"),
        ("shift+tab", "prev_tab", "Prev tab"),
        ("left", "prev_tab", "Prev tab"),
        ("right", "next_tab", "Next tab"),
        ("space", "toggle_selected", "Install/Remove"),
        ("a", "add_marketplace", "Add marketplace"),
        ("r", "remove_marketplace", "Remove marketplace"),
        ("u", "update_marketplace", "Update marketplace"),
        ("escape", "close", "Close"),
        ("q", "close", "Close"),
    ]

    _TAB_ORDER = ("discover", "installed", "marketplaces")
    _TAB_LABELS = {
        "discover": "Discover",
        "installed": "Installed",
        "marketplaces": "Marketplaces",
    }

    def __init__(self, project_path: Path) -> None:
        super().__init__()
        self._project_path = project_path.resolve()
        self._tab = "discover"
        self._search_query = ""
        self._discover_entries: List[PluginCatalogEntry] = []
        self._discover_errors: list[str] = []
        self._installed_plugins: List[_InstalledPlugin] = []
        self._marketplaces: List[MarketplaceSource] = []
        self._visible_ids: List[str] = []
        self._highlighted_id: Optional[str] = None
        self._last_selected_id: Optional[str] = None
        self._last_selected_at: float = 0.0

    def compose(self) -> ComposeResult:
        yield Static("Plugins", id="header")
        with Horizontal(id="tabs_row"):
            yield Button("Discover", id="tab_discover", classes="tab-btn")
            yield Button("Installed", id="tab_installed", classes="tab-btn")
            yield Button("Marketplaces", id="tab_marketplaces", classes="tab-btn")
        yield Static("", id="subtitle")
        yield Input(placeholder="Search...", id="search")
        with Horizontal(id="content_row"):
            yield OptionList(id="plugin_list", markup=False)
            yield Static("", id="details_panel")
        yield Static("", id="hint")
        yield Footer()

    def on_mount(self) -> None:
        self._reload_all_data(force_discover=False)
        self._refresh_view()
        self.query_one("#plugin_list", OptionList).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id == "tab_discover":
            self._switch_tab("discover")
            return
        if button_id == "tab_installed":
            self._switch_tab("installed")
            return
        if button_id == "tab_marketplaces":
            self._switch_tab("marketplaces")

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "search":
            return
        self._search_query = event.value
        self._refresh_list(preserve_highlight=True)
        self._refresh_details_panel()

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        option_id = event.option.id or ""
        self._highlighted_id = option_id
        self._refresh_details_panel()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        option_id = event.option.id or ""
        if not self._is_double_select(option_id):
            return
        if option_id.startswith("discover:"):
            idx = int(option_id.split(":", 1)[1])
            self._show_discover_details(idx)
            return
        if option_id.startswith("installed:"):
            idx = int(option_id.split(":", 1)[1])
            self._show_installed_details(idx)
            return
        if option_id.startswith("marketplace:"):
            idx = int(option_id.split(":", 1)[1])
            self._show_marketplace_details(idx)

    def action_close(self) -> None:
        self.exit()

    def action_next_tab(self) -> None:
        current = self._TAB_ORDER.index(self._tab)
        self._switch_tab(self._TAB_ORDER[(current + 1) % len(self._TAB_ORDER)])

    def action_prev_tab(self) -> None:
        current = self._TAB_ORDER.index(self._tab)
        self._switch_tab(self._TAB_ORDER[(current - 1) % len(self._TAB_ORDER)])

    def action_add_marketplace(self) -> None:
        if self._tab != "marketplaces":
            return
        self.push_screen(_AddMarketplaceScreen(), self._handle_add_marketplace_result)

    def action_remove_marketplace(self) -> None:
        if self._tab != "marketplaces":
            return
        selected = self._selected_marketplace()
        if not selected:
            self.notify("No marketplace selected.", severity="warning")
            return
        removed, _ = remove_marketplace(selected.marketplace_id)
        if removed:
            self.notify(f"Removed marketplace: {selected.marketplace_id}", severity="information")
            self._reload_all_data(force_discover=False)
            self._refresh_view()
        else:
            self.notify(
                f"Marketplace '{selected.marketplace_id}' cannot be removed.",
                severity="warning",
            )

    def action_update_marketplace(self) -> None:
        if self._tab != "marketplaces":
            return
        selected = self._selected_marketplace()
        if not selected:
            self.notify("No marketplace selected.", severity="warning")
            return
        self._reload_all_data(force_discover=True, target_marketplace_id=selected.marketplace_id)
        self._refresh_view()
        self.notify(f"Updated marketplace: {selected.marketplace_id}", severity="information")

    def action_toggle_selected(self) -> None:
        if self._tab == "discover":
            self._toggle_discover_selected()
            return
        if self._tab == "installed":
            self._remove_installed_selected()

    def _switch_tab(self, tab: str) -> None:
        if tab not in self._TAB_ORDER:
            return
        self._tab = tab
        self._refresh_view()

    def _handle_add_marketplace_result(self, value: Optional[str]) -> None:
        if not value:
            return
        try:
            created, _ = add_marketplace(value)
        except Exception as exc:  # noqa: BLE001
            self.notify(f"Failed to add marketplace: {exc}", severity="error")
            return
        self.notify(f"Added marketplace: {created.marketplace_id}", severity="information")
        self._reload_all_data(force_discover=False)
        self._refresh_view()

    def _reload_all_data(
        self,
        *,
        force_discover: bool,
        target_marketplace_id: Optional[str] = None,
    ) -> None:
        self._marketplaces = load_marketplaces()
        discover_result = discover_marketplace_plugins(
            (
                self._marketplaces
                if target_marketplace_id is None
                else [mp for mp in self._marketplaces if mp.marketplace_id == target_marketplace_id]
            ),
            force_refresh=force_discover,
        )
        if target_marketplace_id is not None:
            baseline = discover_marketplace_plugins(self._marketplaces, force_refresh=False)
            refreshed_keys = {
                (entry.marketplace_id, entry.name) for entry in discover_result.entries
            }
            merged = [
                entry
                for entry in baseline.entries
                if (entry.marketplace_id, entry.name) not in refreshed_keys
            ]
            merged.extend(discover_result.entries)
            self._discover_entries = sorted(merged, key=lambda item: item.name.lower())
            self._discover_errors = [
                error.reason for error in baseline.errors + discover_result.errors
            ]
        else:
            self._discover_entries = discover_result.entries
            self._discover_errors = [error.reason for error in discover_result.errors]

        plugin_result = discover_plugins(project_path=self._project_path)
        self._installed_plugins = [
            _InstalledPlugin(
                name=item.name,
                path=item.root,
                description=item.description,
                version=item.version,
            )
            for item in plugin_result.plugins
        ]

    def _refresh_view(self) -> None:
        self._render_tab_buttons()
        self.query_one("#subtitle", Static).update(self._subtitle_text())
        self.query_one("#hint", Static).update(self._hint_text())

        search = self.query_one("#search", Input)
        search.display = self._tab in ("discover", "installed")
        if not search.display:
            search.value = ""
            self._search_query = ""

        self._refresh_list(preserve_highlight=False)
        self._refresh_details_panel()

    def _render_tab_buttons(self) -> None:
        for tab in self._TAB_ORDER:
            button = self.query_one(f"#tab_{tab}", Button)
            if tab == self._tab:
                button.variant = "primary"
                button.add_class("tab-active")
            else:
                button.variant = "default"
                button.remove_class("tab-active")

    def _subtitle_text(self) -> str:
        if self._tab == "discover":
            return f"Discover plugins ({len(self._discover_entries)})"
        if self._tab == "installed":
            return f"Installed plugins ({len(self._installed_plugins)})"
        return f"Manage marketplaces ({len(self._marketplaces)})"

    def _hint_text(self) -> str:
        if self._tab == "discover":
            return "←/→ or Tab switch tabs · type to search · Space install/remove · double-click/double-Enter details"
        if self._tab == "installed":
            return "←/→ or Tab switch tabs · type to search · Space disable · double-click/double-Enter details"
        return "←/→ or Tab switch tabs · a add · u update · r remove · double-click/double-Enter details"

    def _refresh_list(self, preserve_highlight: bool) -> None:
        option_list = self.query_one("#plugin_list", OptionList)
        previous = self._highlighted_id if preserve_highlight else None
        option_list.clear_options()
        self._visible_ids = []

        if self._tab == "discover":
            installed_names = {item.name for item in self._installed_plugins}
            entries = self._filter_discover()
            if not entries:
                option_list.add_option(Option("No plugins match current filters.", id="empty"))
                option_list.highlighted = 0
                self._highlighted_id = None
                return
            for idx, entry in enumerate(entries):
                marker = "●" if entry.name in installed_names else "◯"
                community = " [Community Managed]" if entry.community_managed else ""
                installs = f" · {entry.installs} installs" if entry.installs else ""
                line = (
                    f"{marker} {entry.name} · {entry.source_label}{community}{installs}\n"
                    f"    {entry.description or '(no description)'}"
                )
                option_id = f"discover:{idx}"
                self._visible_ids.append(option_id)
                option_list.add_option(Option(line, id=option_id))
        elif self._tab == "installed":
            entries = self._filter_installed()
            if not entries:
                option_list.add_option(Option("No installed plugins.", id="empty"))
                option_list.highlighted = 0
                self._highlighted_id = None
                return
            for idx, item in enumerate(entries):
                version = f" · v{item.version}" if item.version else ""
                line = f"● {item.name}{version}\n    {item.path}"
                option_id = f"installed:{idx}"
                self._visible_ids.append(option_id)
                option_list.add_option(Option(line, id=option_id))
        else:
            if not self._marketplaces:
                option_list.add_option(Option("No marketplaces configured.", id="empty"))
                option_list.highlighted = 0
                self._highlighted_id = None
                return
            for idx, marketplace in enumerate(self._marketplaces):
                enabled = "●" if marketplace.enabled else "◯"
                count = len(
                    [
                        entry
                        for entry in self._discover_entries
                        if entry.marketplace_id == marketplace.marketplace_id
                    ]
                )
                line = (
                    f"{enabled} {marketplace.marketplace_id}\n"
                    f"    {marketplace.source}\n"
                    f"    {count} available"
                )
                option_id = f"marketplace:{idx}"
                self._visible_ids.append(option_id)
                option_list.add_option(Option(line, id=option_id))

        if previous and previous in self._visible_ids:
            idx = self._visible_ids.index(previous)
            option_list.highlighted = idx
            self._highlighted_id = previous
        elif self._visible_ids:
            option_list.highlighted = 0
            self._highlighted_id = self._visible_ids[0]
        else:
            self._highlighted_id = None

    def _refresh_details_panel(self) -> None:
        panel = self.query_one("#details_panel", Static)
        panel.update(self._details_text_for_current())

    def _details_text_for_current(self) -> str:
        if self._tab == "discover":
            entry = self._selected_discover_entry()
            if not entry:
                return "Select a plugin to view details."
            installed = any(item.name == entry.name for item in self._installed_plugins)
            status = "Installed" if installed else "Not installed"
            return (
                f"Name: {entry.name}\n"
                f"Status: {status}\n"
                f"Marketplace: {entry.marketplace_id}\n"
                f"Source: {entry.source_label}\n"
                f"Plugin Source: {entry.plugin_source}\n"
                f"Author: {entry.author or 'Unknown'}\n"
                f"Updated: {entry.updated_at or 'Unknown'}\n"
                f"Installs: {entry.installs or 'Unknown'}\n\n"
                f"{entry.description or '(no description)'}"
            )

        if self._tab == "installed":
            item = self._selected_installed_plugin()
            if not item:
                return "Select an installed plugin to view details."
            return (
                f"Name: {item.name}\n"
                f"Version: {item.version or 'Unknown'}\n"
                f"Path: {item.path}\n\n"
                f"{item.description or '(no description)'}"
            )

        marketplace = self._selected_marketplace()
        if not marketplace:
            return "Select a marketplace to view details."
        count = len(
            [
                entry
                for entry in self._discover_entries
                if entry.marketplace_id == marketplace.marketplace_id
            ]
        )
        return (
            f"ID: {marketplace.marketplace_id}\n"
            f"Source: {marketplace.source}\n"
            f"Enabled: {marketplace.enabled}\n"
            f"Available plugins: {count}\n"
            f"Updated: {marketplace.updated_at or 'Unknown'}"
        )

    def _filter_discover(self) -> List[PluginCatalogEntry]:
        if not self._search_query:
            return list(self._discover_entries)
        needle = self._search_query.lower()
        return [
            item
            for item in self._discover_entries
            if needle in item.name.lower()
            or needle in item.description.lower()
            or needle in item.source_label.lower()
        ]

    def _filter_installed(self) -> List[_InstalledPlugin]:
        if not self._search_query:
            return list(self._installed_plugins)
        needle = self._search_query.lower()
        return [
            item
            for item in self._installed_plugins
            if needle in item.name.lower()
            or needle in str(item.path).lower()
            or needle in item.description.lower()
        ]

    def _selected_discover_entry(self) -> Optional[PluginCatalogEntry]:
        selected = self._highlighted_id or ""
        if not selected.startswith("discover:"):
            return None
        idx = int(selected.split(":", 1)[1])
        filtered = self._filter_discover()
        if 0 <= idx < len(filtered):
            return filtered[idx]
        return None

    def _selected_installed_plugin(self) -> Optional[_InstalledPlugin]:
        selected = self._highlighted_id or ""
        if not selected.startswith("installed:"):
            return None
        idx = int(selected.split(":", 1)[1])
        filtered = self._filter_installed()
        if 0 <= idx < len(filtered):
            return filtered[idx]
        return None

    def _selected_marketplace(self) -> Optional[MarketplaceSource]:
        selected = self._highlighted_id or ""
        if not selected.startswith("marketplace:"):
            return None
        idx = int(selected.split(":", 1)[1])
        if 0 <= idx < len(self._marketplaces):
            return self._marketplaces[idx]
        return None

    def _is_double_select(self, option_id: str) -> bool:
        if not option_id or option_id == "empty":
            return False
        now = time.monotonic()
        is_double = self._last_selected_id == option_id and (now - self._last_selected_at) <= 0.5
        self._last_selected_id = option_id
        self._last_selected_at = now
        return is_double

    def _toggle_discover_selected(self) -> None:
        entry = self._selected_discover_entry()
        if not entry:
            return
        installed = next(
            (item for item in self._installed_plugins if item.name == entry.name), None
        )
        if installed is not None:
            from ripperdoc.core.plugin_marketplaces import uninstall_plugin_by_path

            ok, message = uninstall_plugin_by_path(
                installed.path,
                project_path=self._project_path,
                scope=PluginSettingsScope.PROJECT,
            )
            self.notify(message, severity="information" if ok else "warning")
            self._reload_all_data(force_discover=False)
            self._refresh_view()
            return

        ok, message, _ = install_marketplace_plugin(
            entry,
            project_path=self._project_path,
            scope=PluginSettingsScope.PROJECT,
        )
        self.notify(message, severity="information" if ok else "error")
        self._reload_all_data(force_discover=False)
        self._refresh_view()

    def _remove_installed_selected(self) -> None:
        selected = self._selected_installed_plugin()
        if not selected:
            return
        from ripperdoc.core.plugin_marketplaces import uninstall_plugin_by_path

        ok, message = uninstall_plugin_by_path(
            selected.path,
            project_path=self._project_path,
            scope=PluginSettingsScope.PROJECT,
        )
        self.notify(message, severity="information" if ok else "warning")
        self._reload_all_data(force_discover=False)
        self._refresh_view()

    def _show_discover_details(self, idx: int) -> None:
        filtered = self._filter_discover()
        if not (0 <= idx < len(filtered)):
            return
        entry = filtered[idx]
        installed = any(item.name == entry.name for item in self._installed_plugins)
        body = (
            f"Name: {entry.name}\n"
            f"Marketplace: {entry.marketplace_id}\n"
            f"Source: {entry.source_label}\n"
            f"Plugin Source: {entry.plugin_source}\n"
            f"Author: {entry.author or 'Unknown'}\n"
            f"Updated: {entry.updated_at or 'Unknown'}\n\n"
            f"{entry.description or '(no description)'}"
        )
        self.push_screen(
            _PluginDetailsScreen("Plugin Details", body, installed=installed),
            lambda action: self._handle_discover_details_action(entry, action),
        )

    def _show_installed_details(self, idx: int) -> None:
        filtered = self._filter_installed()
        if not (0 <= idx < len(filtered)):
            return
        item = filtered[idx]
        body = (
            f"Name: {item.name}\n"
            f"Path: {item.path}\n"
            f"Version: {item.version or 'Unknown'}\n\n"
            f"{item.description or '(no description)'}"
        )
        self.push_screen(
            _PluginDetailsScreen("Installed Plugin", body, installed=True),
            lambda action: self._handle_installed_details_action(item, action),
        )

    def _show_marketplace_details(self, idx: int) -> None:
        if not (0 <= idx < len(self._marketplaces)):
            return
        marketplace = self._marketplaces[idx]
        count = len(
            [
                entry
                for entry in self._discover_entries
                if entry.marketplace_id == marketplace.marketplace_id
            ]
        )
        body = (
            f"ID: {marketplace.marketplace_id}\n"
            f"Source: {marketplace.source}\n"
            f"Enabled: {marketplace.enabled}\n"
            f"Available plugins: {count}\n"
            f"Updated: {marketplace.updated_at or 'Unknown'}"
        )
        self.push_screen(
            _MarketplaceDetailsScreen(
                "Marketplace Details",
                body,
                can_remove=marketplace.marketplace_id != "claude-plugins-official",
            ),
            lambda action: self._handle_marketplace_details_action(marketplace, action),
        )

    def _handle_discover_details_action(
        self, entry: PluginCatalogEntry, action: Optional[str]
    ) -> None:
        if action == "install":
            ok, message, _ = install_marketplace_plugin(
                entry,
                project_path=self._project_path,
                scope=PluginSettingsScope.PROJECT,
            )
            self.notify(message, severity="information" if ok else "error")
            self._reload_all_data(force_discover=False)
            self._refresh_view()
            return
        if action == "uninstall":
            installed = next(
                (item for item in self._installed_plugins if item.name == entry.name),
                None,
            )
            if not installed:
                self.notify(f"Plugin '{entry.name}' is not installed.", severity="warning")
                return
            from ripperdoc.core.plugin_marketplaces import uninstall_plugin_by_path

            ok, message = uninstall_plugin_by_path(
                installed.path,
                project_path=self._project_path,
                scope=PluginSettingsScope.PROJECT,
            )
            self.notify(message, severity="information" if ok else "warning")
            self._reload_all_data(force_discover=False)
            self._refresh_view()

    def _handle_installed_details_action(
        self, item: _InstalledPlugin, action: Optional[str]
    ) -> None:
        if action != "uninstall":
            return
        from ripperdoc.core.plugin_marketplaces import uninstall_plugin_by_path

        ok, message = uninstall_plugin_by_path(
            item.path,
            project_path=self._project_path,
            scope=PluginSettingsScope.PROJECT,
        )
        self.notify(message, severity="information" if ok else "warning")
        self._reload_all_data(force_discover=False)
        self._refresh_view()

    def _handle_marketplace_details_action(
        self, marketplace: MarketplaceSource, action: Optional[str]
    ) -> None:
        if action == "update":
            self._reload_all_data(
                force_discover=True,
                target_marketplace_id=marketplace.marketplace_id,
            )
            self._refresh_view()
            self.notify(
                f"Updated marketplace: {marketplace.marketplace_id}",
                severity="information",
            )
            return
        if action == "remove":
            removed, _ = remove_marketplace(marketplace.marketplace_id)
            if removed:
                self.notify(
                    f"Removed marketplace: {marketplace.marketplace_id}",
                    severity="information",
                )
                self._reload_all_data(force_discover=False)
                self._refresh_view()
            else:
                self.notify(
                    f"Marketplace '{marketplace.marketplace_id}' cannot be removed.",
                    severity="warning",
                )


def run_plugins_tui(project_path: Optional[Path]) -> bool:
    app = PluginsApp((project_path or Path.cwd()).resolve())
    app.run()
    return True
