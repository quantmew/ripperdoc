"""Slash command to manage permission rules."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal

from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from ripperdoc.core.config import (
    GlobalConfig,
    ProjectConfig,
    ProjectLocalConfig,
    get_global_config,
    get_project_config,
    get_project_local_config,
    save_global_config,
    save_project_config,
    save_project_local_config,
)

from .base import SlashCommand


ScopeType = Literal["user", "project", "local"]


def _shorten_path(path: Path, project_path: Path) -> str:
    """Return a short, user-friendly path."""
    try:
        return str(path.resolve().relative_to(project_path.resolve()))
    except (ValueError, OSError):
        pass

    home = Path.home()
    try:
        rel_home = path.resolve().relative_to(home)
        return f"~/{rel_home}"
    except (ValueError, OSError):
        return str(path)


def _get_scope_info(scope: ScopeType, project_path: Path) -> tuple[str, str]:
    """Return (heading, config_path) for a given scope."""
    if scope == "user":
        return "User (global)", str(Path.home() / ".ripperdoc.json")
    elif scope == "project":
        return "Project (shared)", str(project_path / ".ripperdoc" / "config.json")
    else:  # local
        return "Local (private)", str(project_path / ".ripperdoc" / "config.local.json")


def _get_rules_for_scope(scope: ScopeType, project_path: Path) -> tuple[List[str], List[str]]:
    """Return (allow_rules, deny_rules) for a given scope."""
    if scope == "user":
        user_config: GlobalConfig = get_global_config()
        return list(user_config.user_allow_rules), list(user_config.user_deny_rules)
    elif scope == "project":
        project_config: ProjectConfig = get_project_config(project_path)
        return list(project_config.bash_allow_rules), list(project_config.bash_deny_rules)
    else:  # local
        local_config: ProjectLocalConfig = get_project_local_config(project_path)
        return list(local_config.local_allow_rules), list(local_config.local_deny_rules)


def _add_rule(
    scope: ScopeType,
    rule_type: Literal["allow", "deny"],
    rule: str,
    project_path: Path,
) -> bool:
    """Add a rule to the specified scope. Returns True if added, False if already exists."""
    if scope == "user":
        user_config: GlobalConfig = get_global_config()
        rules = (
            user_config.user_allow_rules if rule_type == "allow" else user_config.user_deny_rules
        )
        if rule in rules:
            return False
        rules.append(rule)
        save_global_config(user_config)
    elif scope == "project":
        project_config: ProjectConfig = get_project_config(project_path)
        rules = (
            project_config.bash_allow_rules
            if rule_type == "allow"
            else project_config.bash_deny_rules
        )
        if rule in rules:
            return False
        rules.append(rule)
        save_project_config(project_config, project_path)
    else:  # local
        local_config: ProjectLocalConfig = get_project_local_config(project_path)
        rules = (
            local_config.local_allow_rules
            if rule_type == "allow"
            else local_config.local_deny_rules
        )
        if rule in rules:
            return False
        rules.append(rule)
        save_project_local_config(local_config, project_path)
    return True


def _remove_rule(
    scope: ScopeType,
    rule_type: Literal["allow", "deny"],
    rule: str,
    project_path: Path,
) -> bool:
    """Remove a rule from the specified scope. Returns True if removed, False if not found."""
    if scope == "user":
        user_config: GlobalConfig = get_global_config()
        rules = (
            user_config.user_allow_rules if rule_type == "allow" else user_config.user_deny_rules
        )
        if rule not in rules:
            return False
        rules.remove(rule)
        save_global_config(user_config)
    elif scope == "project":
        project_config: ProjectConfig = get_project_config(project_path)
        rules = (
            project_config.bash_allow_rules
            if rule_type == "allow"
            else project_config.bash_deny_rules
        )
        if rule not in rules:
            return False
        rules.remove(rule)
        save_project_config(project_config, project_path)
    else:  # local
        local_config: ProjectLocalConfig = get_project_local_config(project_path)
        rules = (
            local_config.local_allow_rules
            if rule_type == "allow"
            else local_config.local_deny_rules
        )
        if rule not in rules:
            return False
        rules.remove(rule)
        save_project_local_config(local_config, project_path)
    return True


def _render_all_rules(console: Any, project_path: Path) -> None:
    """Display all permission rules from all scopes."""
    table = Table(title="Permission Rules", show_header=True, header_style="bold cyan")
    table.add_column("Scope", style="bold")
    table.add_column("Type", style="dim")
    table.add_column("Rule")

    has_rules = False

    for scope in ("user", "project", "local"):
        allow_rules, deny_rules = _get_rules_for_scope(scope, project_path)  # type: ignore

        for rule in allow_rules:
            table.add_row(scope, "[green]allow[/green]", escape(rule))
            has_rules = True

        for rule in deny_rules:
            table.add_row(scope, "[red]deny[/red]", escape(rule))
            has_rules = True

    if has_rules:
        console.print(table)
    else:
        console.print("[yellow]No permission rules configured yet.[/yellow]")

    console.print()
    console.print("[dim]Scopes (in priority order):[/dim]")
    console.print("[dim]  - user: Global rules (~/.ripperdoc.json)[/dim]")
    console.print("[dim]  - project: Shared project rules (.ripperdoc/config.json)[/dim]")
    console.print("[dim]  - local: Private project rules (.ripperdoc/config.local.json)[/dim]")


def _render_scope_rules(console: Any, scope: ScopeType, project_path: Path) -> None:
    """Display rules for a specific scope."""
    heading, config_path = _get_scope_info(scope, project_path)
    allow_rules, deny_rules = _get_rules_for_scope(scope, project_path)

    table = Table(title=f"{heading} Permission Rules", show_header=True, header_style="bold cyan")
    table.add_column("Type", style="dim")
    table.add_column("Rule")

    has_rules = False
    for rule in allow_rules:
        table.add_row("[green]allow[/green]", escape(rule))
        has_rules = True

    for rule in deny_rules:
        table.add_row("[red]deny[/red]", escape(rule))
        has_rules = True

    if has_rules:
        console.print(table)
    else:
        console.print(f"[yellow]No {scope} rules configured.[/yellow]")

    console.print(f"[dim]Config file: {escape(config_path)}[/dim]")


def _handle(ui: Any, trimmed_arg: str) -> bool:
    project_path = getattr(ui, "project_path", Path.cwd())
    args = trimmed_arg.strip().split()

    # No args: show all rules
    if not args:
        _render_all_rules(ui.console, project_path)
        ui.console.print()
        ui.console.print("[dim]Usage:[/dim]")
        ui.console.print("[dim]  /permissions                     - Show all rules[/dim]")
        ui.console.print("[dim]  /permissions <scope>             - Show rules for scope[/dim]")
        ui.console.print("[dim]  /permissions add <scope> <type> <rule>   - Add a rule[/dim]")
        ui.console.print("[dim]  /permissions remove <scope> <type> <rule> - Remove a rule[/dim]")
        ui.console.print("[dim]Scopes: user, project, local[/dim]")
        ui.console.print("[dim]Types: allow, deny[/dim]")
        return True

    # Parse command
    action = args[0].lower()
    scope_aliases: Dict[str, ScopeType] = {
        "user": "user",
        "global": "user",
        "project": "project",
        "workspace": "project",
        "local": "local",
        "private": "local",
    }

    # Single scope display
    if action in scope_aliases:
        display_scope: ScopeType = scope_aliases[action]
        _render_scope_rules(ui.console, display_scope, project_path)
        return True

    # Add rule
    if action == "add":
        if len(args) < 4:
            ui.console.print("[red]Usage: /permissions add <scope> <type> <rule>[/red]")
            ui.console.print("[dim]Example: /permissions add local allow npm test[/dim]")
            return True

        scope_arg = args[1].lower()
        if scope_arg not in scope_aliases:
            ui.console.print(f"[red]Unknown scope: {escape(scope_arg)}[/red]")
            ui.console.print("[dim]Available scopes: user, project, local[/dim]")
            return True
        scope: ScopeType = scope_aliases[scope_arg]

        rule_type_arg = args[2].lower()
        if rule_type_arg not in ("allow", "deny"):
            ui.console.print(f"[red]Unknown rule type: {escape(rule_type_arg)}[/red]")
            ui.console.print("[dim]Available types: allow, deny[/dim]")
            return True
        rule_type: Literal["allow", "deny"] = rule_type_arg  # type: ignore[assignment]

        rule = " ".join(args[3:])
        if _add_rule(scope, rule_type, rule, project_path):
            ui.console.print(
                Panel(
                    f"Added [{'green' if rule_type == 'allow' else 'red'}]{rule_type}[/] rule to {scope}:\n{escape(rule)}",
                    title="/permissions",
                )
            )
        else:
            ui.console.print(f"[yellow]Rule already exists in {scope}.[/yellow]")
        return True

    # Remove rule
    if action in ("remove", "rm", "delete", "del"):
        if len(args) < 4:
            ui.console.print("[red]Usage: /permissions remove <scope> <type> <rule>[/red]")
            ui.console.print("[dim]Example: /permissions remove local allow npm test[/dim]")
            return True

        scope_arg = args[1].lower()
        if scope_arg not in scope_aliases:
            ui.console.print(f"[red]Unknown scope: {escape(scope_arg)}[/red]")
            ui.console.print("[dim]Available scopes: user, project, local[/dim]")
            return True
        scope = scope_aliases[scope_arg]  # type: ignore

        rule_type_arg = args[2].lower()
        if rule_type_arg not in ("allow", "deny"):
            ui.console.print(f"[red]Unknown rule type: {escape(rule_type_arg)}[/red]")
            ui.console.print("[dim]Available types: allow, deny[/dim]")
            return True
        remove_rule_type: Literal["allow", "deny"] = rule_type_arg  # type: ignore[assignment]

        rule = " ".join(args[3:])
        if _remove_rule(scope, remove_rule_type, rule, project_path):
            ui.console.print(
                Panel(
                    f"Removed [{'green' if remove_rule_type == 'allow' else 'red'}]{remove_rule_type}[/] rule from {scope}:\n{escape(rule)}",
                    title="/permissions",
                )
            )
        else:
            ui.console.print(f"[yellow]Rule not found in {scope}.[/yellow]")
        return True

    # Unknown command
    ui.console.print(f"[red]Unknown action: {escape(action)}[/red]")
    ui.console.print("[dim]Available actions: add, remove, or a scope name[/dim]")
    return True


command = SlashCommand(
    name="permissions",
    description="Manage permission rules for tools",
    handler=_handle,
    aliases=(),
)


__all__ = ["command"]
