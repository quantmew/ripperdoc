"""Commands command - manage custom slash commands."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich import box
from rich.markup import escape
from rich.table import Table

from .base import SlashCommand
from ripperdoc.core.custom_commands import (
    COMMAND_DIR_NAME,
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
from ripperdoc.utils.log import get_logger

logger = get_logger()

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


@dataclass(frozen=True)
class CommandConfigTarget:
    """Target directory for storing commands."""

    key: str
    label: str
    path: Path
    description: str
    location: CommandLocation


def _print_usage(console: Any) -> None:
    console.print("[bold]/commands[/bold] — open interactive commands UI")
    console.print("[bold]/commands tui[/bold] — open interactive commands UI")
    console.print("[bold]/commands list[/bold] — show custom commands (plain)")
    console.print("[bold]/commands add <name> [scope][/bold] — create a command")
    console.print("[bold]/commands edit <name> [scope][/bold] — edit a command")
    console.print("[bold]/commands delete <name> [scope][/bold] — delete a command (alias: del)")
    console.print(
        "[dim]Scopes: project=.ripperdoc/commands (shared), "
        "user=~/.ripperdoc/commands (all projects)[/dim]"
    )


def _get_targets(project_path: Path) -> List[CommandConfigTarget]:
    project_dir = project_path.resolve()
    user_dir = Path.home().expanduser() / ".ripperdoc" / COMMAND_DIR_NAME
    return [
        CommandConfigTarget(
            key="project",
            label="Project (.ripperdoc/commands)",
            path=project_dir / ".ripperdoc" / COMMAND_DIR_NAME,
            description="Shared commands committed to the repo",
            location=CommandLocation.PROJECT,
        ),
        CommandConfigTarget(
            key="user",
            label="User (~/.ripperdoc/commands)",
            path=user_dir,
            description="Applies to all projects on this machine",
            location=CommandLocation.USER,
        ),
    ]


def _normalize_scope(scope_hint: Optional[str]) -> Optional[str]:
    if not scope_hint:
        return None
    normalized = scope_hint.strip().lower()
    if normalized.startswith("proj"):
        return "project"
    if normalized.startswith(("user", "global")):
        return "user"
    return None


def _select_target(
    console: Any, project_path: Path, scope_hint: Optional[str]
) -> Optional[CommandConfigTarget]:
    targets = _get_targets(project_path)
    normalized = _normalize_scope(scope_hint)
    if normalized:
        match = next((t for t in targets if t.key == normalized), None)
        if match:
            return match

    default_idx = 0
    console.print("\n[bold]Where should this command live?[/bold]")
    while True:
        for idx, target in enumerate(targets, start=1):
            status = "[green]✓[/green]" if target.path.exists() else "[dim]○[/dim]"
            console.print(
                f"  [{idx}] {target.label} {status}\n"
                f"      {escape(str(target.path))}\n"
                f"      [dim]{target.description}[/dim]"
            )

        choice = console.input(f"Location [1-{len(targets)}, default {default_idx + 1}]: ").strip()
        if not choice:
            return targets[default_idx]
        for idx, target in enumerate(targets, start=1):
            if choice == str(idx) or choice.lower() == target.key:
                return target
        console.print("[red]Please choose a valid location number or key.[/red]")


def _parse_allowed_tools(raw: str) -> List[str]:
    raw = raw.strip()
    if not raw:
        return []
    if "," in raw:
        parts = raw.split(",")
    else:
        parts = raw.split()
    return [item.strip() for item in parts if item.strip()]


def _format_tools(tools: List[str]) -> str:
    return ", ".join(tools) if tools else ""


def _extract_frontmatter_extras(frontmatter: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in frontmatter.items() if k not in KNOWN_FRONTMATTER_KEYS}


def _load_command_candidates(
    project_path: Path, name: str, scope_hint: Optional[str]
) -> List[CustomCommandDefinition]:
    normalized = normalize_command_name(name)
    if not normalized:
        return []
    result = load_custom_commands_by_scope(project_path=project_path)
    candidates = [cmd for cmd in result.commands if cmd.name == normalized]
    scope_key = _normalize_scope(scope_hint)
    if scope_key:
        candidates = [cmd for cmd in candidates if cmd.location.value == scope_key]
    return candidates


def _select_command(
    console: Any,
    project_path: Path,
    name: str,
    scope_hint: Optional[str],
) -> Optional[CustomCommandDefinition]:
    candidates = _load_command_candidates(project_path, name, scope_hint)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    console.print("\n[bold]Multiple commands matched; choose one:[/bold]")
    for idx, cmd in enumerate(candidates, start=1):
        location = getattr(cmd.location, "value", cmd.location)
        console.print(f"  [{idx}] /{escape(cmd.name)} ({escape(str(location))})")
        console.print(f"      {escape(str(cmd.path))}")

    while True:
        choice = console.input(f"Select [1-{len(candidates)}]: ").strip()
        if not choice:
            return None
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(candidates):
                return candidates[idx - 1]
        console.print("[red]Please choose a valid option.[/red]")


def _refresh_custom_command_cache(project_path: Path) -> None:
    try:
        from ripperdoc.cli import commands as commands_registry

        commands_registry.refresh_custom_commands(project_path)
    except Exception as exc:  # noqa: BLE001
        logger.debug("[commands_cmd] Failed to refresh command cache: %s", exc)


def _render_commands_plain(ui: Any, project_path: Path) -> bool:
    console = ui.console
    merged = load_all_custom_commands(project_path=project_path)
    active_by_name = {cmd.name: cmd.location for cmd in merged.commands}
    catalog = load_custom_commands_by_scope(project_path=project_path)

    console.print("\n[bold]Custom commands:[/bold]")
    _print_usage(console)
    if not catalog.commands:
        console.print("  • None configured")
        return True

    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("#", style="dim", width=3)
    table.add_column("Name")
    table.add_column("Scope", style="dim")
    table.add_column("Status", style="dim")
    table.add_column("Description")
    table.add_column("Path", style="dim")

    def _loc_sort(cmd: CustomCommandDefinition) -> int:
        return 0 if cmd.location == CommandLocation.PROJECT else 1

    for idx, cmd in enumerate(sorted(catalog.commands, key=lambda c: (c.name, _loc_sort(c))), start=1):
        location = getattr(cmd.location, "value", cmd.location)
        active_loc = active_by_name.get(cmd.name)
        if active_loc == cmd.location:
            status = "active"
        elif active_loc is None:
            status = "inactive"
        else:
            status = f"shadowed by {active_loc.value}"
        table.add_row(
            str(idx),
            f"/{cmd.name}",
            str(location),
            status,
            escape(cmd.description),
            escape(str(cmd.path)),
        )

    console.print(table)
    if catalog.errors:
        console.print("[yellow]Some command files could not be loaded:[/yellow]")
        for err in catalog.errors:
            console.print(f"  - {escape(str(err.path))}: {escape(err.reason)}")
    return True


def _handle_add(ui: Any, tokens: List[str], project_path: Path) -> bool:
    console = ui.console
    name = None
    scope_hint = None

    if tokens:
        if _normalize_scope(tokens[0]):
            scope_hint = tokens[0]
            if len(tokens) > 1:
                name = tokens[1]
        else:
            name = tokens[0]
            if len(tokens) > 1 and _normalize_scope(tokens[1]):
                scope_hint = tokens[1]

    if not name:
        name = console.input("Command name (e.g., sprint-summary): ").strip()

    error = validate_command_name(name)
    if error:
        console.print(f"[red]{escape(error)}[/red]")
        return True

    normalized = normalize_command_name(name)
    target = _select_target(console, project_path, scope_hint)
    if not target:
        return True

    path = command_name_to_path(normalized, target.path)
    if path.exists():
        confirmation = console.input("Command already exists. Overwrite? [y/N]: ").strip().lower()
        if confirmation not in ("y", "yes"):
            console.print("[yellow]Create cancelled.[/yellow]")
            return True

    description = console.input("Description (optional): ").strip()
    argument_hint = console.input("Argument hint (optional, e.g., <branch>): ").strip()
    allowed_tools_raw = console.input("Allowed tools (comma-separated, optional): ").strip()
    model = console.input("Model (optional): ").strip()
    thinking_mode = console.input(
        "Thinking mode (optional, e.g., think hard / ultrathink): "
    ).strip()
    content_raw = console.input(
        "Command content (single line, use \\n for newlines): "
    ).strip()
    content = content_raw.replace("\\n", "\n") if content_raw else ""

    frontmatter: Dict[str, Any] = {}
    if description:
        frontmatter["description"] = description
    if argument_hint:
        frontmatter["argument-hint"] = argument_hint
    allowed_tools = _parse_allowed_tools(allowed_tools_raw)
    if allowed_tools:
        frontmatter["allowed-tools"] = allowed_tools
    if model:
        frontmatter["model"] = model
    if thinking_mode:
        frontmatter["thinking-mode"] = thinking_mode

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_command_markdown(frontmatter, content), encoding="utf-8")
    except (OSError, IOError, PermissionError) as exc:
        console.print(f"[red]Failed to write command: {escape(str(exc))}[/red]")
        return True

    _refresh_custom_command_cache(project_path)
    console.print(f"[green]✓ Saved command /{normalized} in {escape(str(path))}[/green]")
    return True


def _handle_edit(ui: Any, tokens: List[str], project_path: Path) -> bool:
    console = ui.console
    if not tokens:
        console.print("[red]Usage: /commands edit <name> [scope][/red]")
        return True

    name = tokens[0]
    scope_hint = tokens[1] if len(tokens) > 1 else None
    cmd = _select_command(console, project_path, name, scope_hint)
    if not cmd:
        console.print(f"[yellow]No command named '{escape(name)}' found.[/yellow]")
        return True

    try:
        raw_text = cmd.path.read_text(encoding="utf-8")
    except (OSError, IOError, PermissionError) as exc:
        console.print(f"[red]Unable to read {escape(str(cmd.path))}: {escape(str(exc))}[/red]")
        return True

    frontmatter, body, parse_error = parse_command_markdown(raw_text)
    if parse_error:
        console.print(f"[yellow]{escape(parse_error)}[/yellow]")

    extras = _extract_frontmatter_extras(frontmatter)
    description = (frontmatter.get("description") or cmd.description or "").strip()
    argument_hint = (
        frontmatter.get("argument-hint")
        or frontmatter.get("argument_hint")
        or cmd.argument_hint
        or ""
    ).strip()
    allowed_tools = cmd.allowed_tools
    model = (frontmatter.get("model") or cmd.model or "").strip()
    thinking_mode = (
        frontmatter.get("thinking-mode") or frontmatter.get("thinking_mode") or cmd.thinking_mode or ""
    ).strip()

    console.print(f"\n[bold]Editing /{cmd.name} ({cmd.location.value})[/bold]")
    preview = body.strip()
    if len(preview) > 220:
        preview = preview[:220] + "..."
    if preview:
        console.print("[dim]Current content (preview):[/dim]")
        console.print(escape(preview), markup=False)

    new_description = console.input("Description [Enter to keep, '-' to clear]: ").strip()
    if new_description == "-":
        description = ""
    elif new_description:
        description = new_description

    new_hint = console.input("Argument hint [Enter to keep, '-' to clear]: ").strip()
    if new_hint == "-":
        argument_hint = ""
    elif new_hint:
        argument_hint = new_hint

    tools_input = console.input(
        f"Allowed tools [{_format_tools(allowed_tools)}] (Enter=keep, '-'=clear): "
    ).strip()
    if tools_input == "-":
        allowed_tools = []
    elif tools_input:
        allowed_tools = _parse_allowed_tools(tools_input)

    model_input = console.input(f"Model [{model or 'none'}] (Enter=keep, '-'=clear): ").strip()
    if model_input == "-":
        model = ""
    elif model_input:
        model = model_input

    think_input = console.input(
        f"Thinking mode [{thinking_mode or 'none'}] (Enter=keep, '-'=clear): "
    ).strip()
    if think_input == "-":
        thinking_mode = ""
    elif think_input:
        thinking_mode = think_input

    content_input = console.input(
        "Command content (single line, use \\n for newlines) [Enter to keep]: "
    ).strip()
    if content_input:
        body = content_input.replace("\\n", "\n")

    frontmatter_out: Dict[str, Any] = {}
    if description:
        frontmatter_out["description"] = description
    if argument_hint:
        frontmatter_out["argument-hint"] = argument_hint
    if allowed_tools:
        frontmatter_out["allowed-tools"] = allowed_tools
    if model:
        frontmatter_out["model"] = model
    if thinking_mode:
        frontmatter_out["thinking-mode"] = thinking_mode
    frontmatter_out.update(extras)

    try:
        cmd.path.write_text(render_command_markdown(frontmatter_out, body), encoding="utf-8")
    except (OSError, IOError, PermissionError) as exc:
        console.print(f"[red]Failed to write command: {escape(str(exc))}[/red]")
        return True

    _refresh_custom_command_cache(project_path)
    console.print(f"[green]✓ Updated /{cmd.name} ({escape(str(cmd.path))})[/green]")
    return True


def _handle_delete(ui: Any, tokens: List[str], project_path: Path) -> bool:
    console = ui.console
    if not tokens:
        console.print("[red]Usage: /commands delete <name> [scope][/red]")
        return True

    name = tokens[0]
    scope_hint = tokens[1] if len(tokens) > 1 else None
    cmd = _select_command(console, project_path, name, scope_hint)
    if not cmd:
        console.print(f"[yellow]No command named '{escape(name)}' found.[/yellow]")
        return True

    confirmation = console.input(f"Delete /{cmd.name} ({cmd.location.value})? [y/N]: ").strip().lower()
    if confirmation not in ("y", "yes"):
        console.print("[yellow]Delete cancelled.[/yellow]")
        return True

    try:
        cmd.path.unlink()
    except FileNotFoundError:
        console.print("[yellow]Command file already removed.[/yellow]")
        return True
    except (OSError, IOError, PermissionError) as exc:
        console.print(f"[red]Failed to delete {escape(str(cmd.path))}: {escape(str(exc))}[/red]")
        return True

    _refresh_custom_command_cache(project_path)
    console.print(f"[green]✓ Deleted /{cmd.name}[/green]")
    return True


def _handle_commands_tui(ui: Any) -> bool:
    project_path = getattr(ui, "project_path", Path.cwd())
    console = ui.console
    if not sys.stdin.isatty():
        console.print("[yellow]Interactive UI requires a TTY. Showing plain list instead.[/yellow]")
        return _render_commands_plain(ui, project_path)

    try:
        from ripperdoc.cli.ui.commands_tui import run_commands_tui
    except (ImportError, ModuleNotFoundError) as exc:
        console.print(
            f"[yellow]Textual UI not available ({escape(str(exc))}). Showing plain list.[/yellow]"
        )
        return _render_commands_plain(ui, project_path)

    try:
        return bool(run_commands_tui(project_path))
    except Exception as exc:  # noqa: BLE001 - fail safe in interactive UI
        console.print(f"[red]Textual UI failed: {escape(str(exc))}[/red]")
        return _render_commands_plain(ui, project_path)


def _handle(ui: Any, arg: str) -> bool:
    project_path = getattr(ui, "project_path", None) or Path.cwd()
    tokens = arg.split()
    subcmd = tokens[0].lower() if tokens else ""

    if subcmd in ("", "tui", "ui"):
        return _handle_commands_tui(ui)

    if subcmd in ("help", "-h", "--help"):
        _print_usage(ui.console)
        return True

    if subcmd in ("list", "ls"):
        return _render_commands_plain(ui, project_path)

    if subcmd in ("add", "create", "new"):
        return _handle_add(ui, tokens[1:], project_path)

    if subcmd in ("edit", "update"):
        return _handle_edit(ui, tokens[1:], project_path)

    if subcmd in ("delete", "del", "remove", "rm"):
        return _handle_delete(ui, tokens[1:], project_path)

    if subcmd:
        ui.console.print(f"[red]Unknown commands subcommand '{escape(subcmd)}'.[/red]")
        _print_usage(ui.console)
        return True

    return _render_commands_plain(ui, project_path)


command = SlashCommand(
    name="commands",
    description="Manage custom slash commands",
    handler=_handle,
)


__all__ = ["command"]
