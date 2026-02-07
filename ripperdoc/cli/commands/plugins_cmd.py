"""Plugins command - inspect and manage plugin configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Optional

from rich import box
from rich.markup import escape
from rich.table import Table

from ripperdoc.core.hooks.manager import hook_manager
from ripperdoc.core.plugins import (
    PluginSettingsScope,
    add_enabled_plugin_for_scope,
    discover_plugins,
    list_enabled_plugin_entries_for_scope,
    remove_enabled_plugin_for_scope,
)

from .base import SlashCommand


@dataclass(frozen=True)
class _ScopeOption:
    key: str
    description: str
    scope: PluginSettingsScope


_SCOPE_OPTIONS = [
    _ScopeOption(
        key="project",
        description=".ripperdoc/plugins.json (shared)",
        scope=PluginSettingsScope.PROJECT,
    ),
    _ScopeOption(
        key="local",
        description=".ripperdoc/plugins.local.json (gitignored)",
        scope=PluginSettingsScope.LOCAL,
    ),
    _ScopeOption(
        key="user",
        description="~/.ripperdoc/plugins.json (all projects)",
        scope=PluginSettingsScope.USER,
    ),
]


def _print_usage(console: Any) -> None:
    console.print("[bold]/plugins[/bold] — open interactive plugins UI")
    console.print("[bold]/plugin[/bold] — alias of /plugins")
    console.print("[bold]/plugins tui[/bold] — open interactive plugins UI")
    console.print("[bold]/plugins list[/bold] — list active plugins")
    console.print("[bold]/plugins add <path> [scope][/bold] — enable plugin directory")
    console.print(
        "[bold]/plugins remove <name-or-path> [scope][/bold] — disable plugin by name/path"
    )
    console.print("[bold]/plugins entries [scope][/bold] — show raw configured entries")
    console.print("[bold]/plugins reload[/bold] — reload plugin-driven hooks/commands")
    console.print("[dim]Scopes: project (default), local, user[/dim]")


def _normalize_scope(raw: Optional[str]) -> PluginSettingsScope:
    if not raw:
        return PluginSettingsScope.PROJECT
    value = raw.strip().lower()
    if value.startswith("loc"):
        return PluginSettingsScope.LOCAL
    if value.startswith(("usr", "global")):
        return PluginSettingsScope.USER
    return PluginSettingsScope.PROJECT


def _scope_from_key(raw: Optional[str]) -> Optional[PluginSettingsScope]:
    if raw is None:
        return None
    value = raw.strip().lower()
    if value.startswith("proj"):
        return PluginSettingsScope.PROJECT
    if value.startswith("loc"):
        return PluginSettingsScope.LOCAL
    if value.startswith(("usr", "global")):
        return PluginSettingsScope.USER
    return None


def _refresh_plugin_runtime(ui: Any) -> None:
    project_path = getattr(ui, "project_path", Path.cwd())
    try:
        from ripperdoc.cli import commands as command_registry

        command_registry.refresh_custom_commands(project_path)
    except Exception:
        pass
    hook_manager.set_project_dir(project_path)
    hook_manager.reload_config()


def _render_plugins(ui: Any) -> bool:
    console = ui.console
    project_path = getattr(ui, "project_path", Path.cwd())
    result = discover_plugins(project_path=project_path)

    console.print("\n[bold]Active Plugins[/bold]")
    _print_usage(console)

    if not result.plugins:
        console.print("  • None")
    else:
        table = Table(box=box.SIMPLE_HEAVY)
        table.add_column("Name", style="cyan")
        table.add_column("Version", style="dim")
        table.add_column("Description")
        table.add_column("Path", style="dim")
        for plugin in result.plugins:
            table.add_row(
                plugin.name,
                plugin.version or "-",
                escape(plugin.description or "-"),
                escape(str(plugin.root)),
            )
        console.print(table)

    if result.errors:
        console.print("\n[yellow]Plugin load warnings:[/yellow]")
        for error in result.errors:
            console.print(f"  • {escape(str(error.path))}: {escape(error.reason)}")
    return True


def _render_entries(ui: Any, scope: Optional[PluginSettingsScope]) -> bool:
    console = ui.console
    project_path = getattr(ui, "project_path", Path.cwd())
    scopes = [scope] if scope is not None else [option.scope for option in _SCOPE_OPTIONS]
    console.print("\n[bold]Configured Plugin Entries[/bold]")
    for selected in scopes:
        entries = list_enabled_plugin_entries_for_scope(selected, project_path=project_path)
        label = next(
            (option.description for option in _SCOPE_OPTIONS if option.scope == selected),
            selected.value,
        )
        console.print(f"\n[bold]{selected.value}[/bold] [dim]({escape(label)})[/dim]")
        if not entries:
            console.print("  • None")
            continue
        for entry in entries:
            console.print(f"  • {escape(entry)}")
    return True


def _handle_add(ui: Any, tokens: list[str]) -> bool:
    console = ui.console
    if not tokens:
        console.print("[red]Usage: /plugins add <path> [scope][/red]")
        return True

    raw_path = tokens[0]
    scope = _normalize_scope(tokens[1] if len(tokens) > 1 else None)
    project_path = getattr(ui, "project_path", Path.cwd())
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = (project_path / candidate).resolve()

    if not candidate.exists() or not candidate.is_dir():
        console.print(f"[red]Plugin directory not found: {escape(str(candidate))}[/red]")
        return True

    settings_path = add_enabled_plugin_for_scope(candidate, scope=scope, project_path=project_path)
    _refresh_plugin_runtime(ui)
    console.print(
        f"[green]Enabled plugin[/green] {escape(str(candidate))} "
        f"[dim]({scope.value}: {escape(str(settings_path))})[/dim]"
    )
    return True


def _resolve_selector_to_path(ui: Any, selector: str) -> Optional[Path]:
    project_path = getattr(ui, "project_path", Path.cwd())
    candidate = Path(selector).expanduser()
    if candidate.is_absolute() or selector.startswith(".") or "/" in selector:
        if not candidate.is_absolute():
            candidate = (project_path / candidate).resolve()
        return candidate.resolve()

    plugins = discover_plugins(project_path=project_path).plugins
    by_name = next((plugin for plugin in plugins if plugin.name == selector), None)
    if by_name:
        return by_name.root
    return None


def _handle_remove(ui: Any, tokens: list[str]) -> bool:
    console = ui.console
    if not tokens:
        console.print("[red]Usage: /plugins remove <name-or-path> [scope][/red]")
        return True

    selector = tokens[0]
    raw_scope = tokens[1] if len(tokens) > 1 else None
    scope = _normalize_scope(raw_scope)
    project_path = getattr(ui, "project_path", Path.cwd())

    target = _resolve_selector_to_path(ui, selector)
    if target is None:
        console.print(f"[red]Could not resolve plugin '{escape(selector)}'.[/red]")
        return True

    settings_path, removed = remove_enabled_plugin_for_scope(
        target, scope=scope, project_path=project_path
    )
    _refresh_plugin_runtime(ui)
    if removed:
        console.print(
            f"[green]Removed plugin[/green] {escape(str(target))} "
            f"[dim]({scope.value}: {escape(str(settings_path))})[/dim]"
        )
    else:
        console.print(
            f"[yellow]Plugin was not configured in {scope.value} scope:[/yellow] "
            f"{escape(str(target))}"
        )
    return True


def _handle(ui: Any, trimmed_arg: str) -> bool:
    args = trimmed_arg.strip().split()
    subcmd = args[0].lower() if args else ""

    if subcmd in ("help", "-h", "--help"):
        _print_usage(ui.console)
        return True
    if subcmd in ("", "tui", "ui", "discover", "installed", "marketplaces"):
        if not sys.stdin.isatty():
            ui.console.print(
                "[yellow]Interactive UI requires a TTY. Showing plain list instead.[/yellow]"
            )
            return _render_plugins(ui)
        try:
            from ripperdoc.cli.ui.plugins_tui import run_plugins_tui
        except (ImportError, ModuleNotFoundError) as exc:
            ui.console.print(
                f"[yellow]Textual UI not available ({escape(str(exc))}). Showing plain list.[/yellow]"
            )
            return _render_plugins(ui)
        try:
            result = bool(run_plugins_tui(getattr(ui, "project_path", Path.cwd())))
            _refresh_plugin_runtime(ui)
            return result
        except Exception as exc:  # noqa: BLE001
            ui.console.print(f"[red]Plugins TUI failed: {escape(str(exc))}[/red]")
            return _render_plugins(ui)
    if subcmd in ("list", "ls"):
        return _render_plugins(ui)
    if subcmd in ("reload", "refresh"):
        _refresh_plugin_runtime(ui)
        ui.console.print("[green]Plugin state reloaded.[/green]")
        return True
    if subcmd in ("entries", "config"):
        scope = _scope_from_key(args[1]) if len(args) > 1 else None
        return _render_entries(ui, scope)
    if subcmd in ("add", "enable", "install"):
        return _handle_add(ui, args[1:])
    if subcmd in ("remove", "rm", "uninstall", "disable"):
        return _handle_remove(ui, args[1:])

    ui.console.print(f"[red]Unknown action: {escape(subcmd)}[/red]")
    _print_usage(ui.console)
    return True


command = SlashCommand(
    name="plugins",
    description="List and manage plugins",
    handler=_handle,
    aliases=("plugin",),
)


__all__ = ["command"]
