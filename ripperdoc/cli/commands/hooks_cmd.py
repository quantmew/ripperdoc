"""Hooks command - view and edit hook configurations with guided prompts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from rich.markup import escape
from rich.table import Table

from .base import SlashCommand
from ripperdoc.core.hooks import (
    HookEvent,
    get_merged_hooks_config,
    get_global_hooks_path,
    get_project_hooks_path,
    get_project_local_hooks_path,
)
from ripperdoc.core.hooks.config import DEFAULT_HOOK_TIMEOUT, PROMPT_SUPPORTED_EVENTS
from ripperdoc.utils.log import get_logger

logger = get_logger()

MATCHER_EVENTS = {
    HookEvent.PRE_TOOL_USE.value,
    HookEvent.PERMISSION_REQUEST.value,
    HookEvent.POST_TOOL_USE.value,
}

EVENT_DESCRIPTIONS: Dict[str, str] = {
    HookEvent.PRE_TOOL_USE.value: "Before a tool runs (can block or edit input)",
    HookEvent.PERMISSION_REQUEST.value: "When a permission dialog is shown",
    HookEvent.POST_TOOL_USE.value: "After a tool finishes running",
    HookEvent.USER_PROMPT_SUBMIT.value: "When you submit a prompt",
    HookEvent.NOTIFICATION.value: "When Ripperdoc sends a notification",
    HookEvent.STOP.value: "When the agent stops responding",
    HookEvent.SUBAGENT_STOP.value: "When a Task/subagent stops",
    HookEvent.PRE_COMPACT.value: "Before compacting the conversation",
    HookEvent.SESSION_START.value: "When a session starts or resumes",
    HookEvent.SESSION_END.value: "When a session ends",
}


@dataclass(frozen=True)
class HookConfigTarget:
    """Target file for storing hooks."""

    key: str
    label: str
    path: Path
    description: str


def _print_usage(console: Any) -> None:
    """Display available subcommands."""
    console.print("[bold]/hooks[/bold] — show configured hooks")
    console.print("[bold]/hooks add [scope][/bold] — guided creator (scope: local|project|global)")
    console.print("[bold]/hooks edit [scope][/bold] — step-by-step edit of an existing hook")
    console.print("[bold]/hooks delete [scope][/bold] — remove a hook entry (alias: del)")
    console.print(
        "[dim]Scopes: local=.ripperdoc/hooks.local.json (git-ignored), "
        "project=.ripperdoc/hooks.json (shared), "
        "global=~/.ripperdoc/hooks.json (all projects)[/dim]"
    )


def _get_targets(project_path: Path) -> List[HookConfigTarget]:
    """Return available hook config destinations."""
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
            key="global",
            label="Global (~/.ripperdoc/hooks.json)",
            path=get_global_hooks_path(),
            description="Applies to all projects on this machine",
        ),
    ]


def _select_target(
    console: Any, project_path: Path, scope_hint: Optional[str]
) -> Optional[HookConfigTarget]:
    """Prompt user to choose a hooks config target."""
    targets = _get_targets(project_path)

    if scope_hint:
        match = next(
            (t for t in targets if t.key.startswith(scope_hint.lower())),
            None,
        )
        if match:
            return match
        console.print(
            f"[yellow]Unknown scope '{escape(scope_hint)}'. Choose from the options below.[/yellow]"
        )

    default_idx = 0
    console.print("\n[bold]Where should this hook live?[/bold]")
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


def _load_hooks_json(console: Any, path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load hooks config from disk, normalizing the structure."""
    if not path.exists():
        return {}

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        console.print(
            f"[red]Invalid JSON in {escape(str(path))}: {escape(str(exc))}. "
            "Starting from an empty config.[/red]"
        )
        logger.warning("[hooks_cmd] Invalid JSON in %s: %s", path, exc)
        return {}
    except (OSError, IOError, PermissionError) as exc:
        console.print(f"[red]Unable to read {escape(str(path))}: {escape(str(exc))}[/red]")
        logger.warning("[hooks_cmd] Failed to read %s: %s", path, exc)
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


def _save_hooks_json(console: Any, path: Path, hooks: Dict[str, List[Dict[str, Any]]]) -> bool:
    """Persist hooks to disk with indentation."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps({"hooks": hooks}, indent=2, ensure_ascii=False)
        path.write_text(serialized, encoding="utf-8")
        return True
    except (OSError, IOError, PermissionError) as exc:
        console.print(
            f"[red]Failed to write hooks to {escape(str(path))}: {escape(str(exc))}[/red]"
        )
        logger.warning("[hooks_cmd] Failed to write %s: %s", path, exc)
        return False


def _summarize_hook(hook: Dict[str, Any]) -> str:
    """Return a short human-friendly summary of a hook."""
    hook_type = hook.get("type", "command")
    if hook_type == "prompt":
        text = hook.get("prompt", "") or "(prompt missing)"
        label = "prompt"
    else:
        text = hook.get("command", "") or "(command missing)"
        label = "command"

    text = text.replace("\n", "\\n")
    if len(text) > 60:
        text = text[:57] + "..."

    timeout = hook.get("timeout", DEFAULT_HOOK_TIMEOUT)
    return f"{label}: {text} ({timeout}s)"


def _render_hooks_overview(ui: Any, project_path: Path) -> bool:
    """Display merged hooks with file locations."""
    config = get_merged_hooks_config(project_path)
    targets = _get_targets(project_path)

    ui.console.print()
    ui.console.print("[bold]Hook Configuration Files[/bold]")
    for target in targets:
        if target.path.exists():
            ui.console.print(f"  [green]✓[/green] {target.label}: {target.path}")
        else:
            ui.console.print(f"  [dim]○[/dim] {target.label}: {target.path} [dim](not found)[/dim]")

    ui.console.print()

    total_hooks = 0
    for matchers in config.hooks.values():
        for matcher in matchers:
            total_hooks += len(matcher.hooks)

    if not config.hooks:
        ui.console.print(
            "[yellow]No hooks configured.[/yellow]\n"
            "Use /hooks add to create one with a guided editor."
        )
        return True

    ui.console.print(f"[bold]Registered Hooks[/bold] ({total_hooks} total)\n")

    for event in HookEvent:
        event_name = event.value
        if event_name not in config.hooks:
            continue

        matchers = config.hooks[event_name]
        if not matchers:
            continue

        table = Table(
            title=f"[bold cyan]{event_name}[/bold cyan]",
            show_header=True,
            header_style="bold",
            expand=True,
            title_justify="left",
        )
        table.add_column("Matcher", style="yellow", width=20)
        table.add_column("Command", style="green")
        table.add_column("Timeout", style="dim", width=8, justify="right")

        for matcher in matchers:
            matcher_str = matcher.matcher or "*"
            for i, hook in enumerate(matcher.hooks):
                cmd_text = hook.command or hook.prompt or ""
                prefix = "[prompt] " if hook.prompt and not hook.command else ""
                if len(cmd_text) > 60:
                    cmd_text = cmd_text[:57] + "..."

                if i == 0:
                    table.add_row(
                        escape(matcher_str),
                        escape(prefix + cmd_text),
                        f"{hook.timeout}s",
                    )
                else:
                    table.add_row("", escape(prefix + cmd_text), f"{hook.timeout}s")

        ui.console.print(table)
        ui.console.print()

    ui.console.print("[dim]Tip: Hooks run in order. /hooks add launches a guided setup.[/dim]")
    return True


def _prompt_event_selection(
    console: Any,
    available_events: Sequence[str],
    default_event: Optional[str] = None,
) -> Optional[str]:
    """Prompt user to choose a hook event."""
    if not available_events:
        console.print("[red]No hook events available to choose from.[/red]")
        return None

    default_idx = 0
    if default_event and default_event in available_events:
        default_idx = available_events.index(default_event)

    console.print("\n[bold]Select hook event:[/bold]")
    for idx, event_name in enumerate(available_events, start=1):
        desc = EVENT_DESCRIPTIONS.get(event_name, "Custom event")
        console.print(f"  [{idx}] {event_name} — {desc}")

    while True:
        choice = console.input(
            f"Event [1-{len(available_events)}, default {default_idx + 1}]: "
        ).strip()
        if not choice:
            return available_events[default_idx]

        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(available_events):
                return available_events[idx]

        console.print("[red]Please enter a number from the list.[/red]")


def _prompt_matcher_selection(
    console: Any,
    event_name: str,
    matchers: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Prompt for matcher selection or creation."""
    if event_name not in MATCHER_EVENTS:
        if matchers:
            return matchers[0]
        default_matcher: Dict[str, Any] = {"matcher": None, "hooks": []}
        matchers.append(default_matcher)
        return default_matcher

    if not matchers:
        console.print("\nMatcher (tool name or regex). Leave empty to match all tools (*).")
        pattern = console.input("Matcher: ").strip() or "*"
        first_matcher: Dict[str, Any] = {"matcher": pattern, "hooks": []}
        matchers.append(first_matcher)
        return first_matcher

    console.print("\nSelect matcher:")
    for idx, matcher in enumerate(matchers, start=1):
        label = matcher.get("matcher") or "*"
        hook_count = len(matcher.get("hooks") or [])
        console.print(f"  [{idx}] {escape(str(label))} ({hook_count} hook(s))")
    new_idx = len(matchers) + 1
    console.print(f"  [{new_idx}] New matcher pattern")

    default_choice = 1
    while True:
        choice = console.input(f"Matcher [1-{new_idx}, default {default_choice}]: ").strip()
        if not choice:
            choice = str(default_choice)
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(matchers):
                return matchers[idx - 1]
            if idx == new_idx:
                pattern = console.input("New matcher (blank for '*'): ").strip() or "*"
                created_matcher: Dict[str, Any] = {"matcher": pattern, "hooks": []}
                matchers.append(created_matcher)
                return created_matcher
        console.print("[red]Choose a matcher number from the list.[/red]")


def _prompt_timeout(console: Any, default_timeout: int) -> int:
    """Prompt for timeout seconds with validation."""
    while True:
        raw = console.input(f"Timeout seconds [{default_timeout}]: ").strip()
        if not raw:
            return default_timeout
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
        console.print("[red]Please enter a positive integer timeout.[/red]")


def _prompt_hook_details(
    console: Any,
    event_name: str,
    existing_hook: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Collect hook details (type, command/prompt, timeout)."""
    allowed_types = ["command"]
    if event_name in PROMPT_SUPPORTED_EVENTS:
        allowed_types.append("prompt")

    default_type = (existing_hook or {}).get("type", allowed_types[0])
    if default_type not in allowed_types:
        default_type = allowed_types[0]

    type_label = "/".join(allowed_types)
    console.print(
        "[dim]Hook type: 'command' runs a shell command; "
        "'prompt' asks the model to evaluate (supported on selected events).[/dim]"
    )
    while True:
        hook_type = (
            console.input(f"Hook type ({type_label}) [default {default_type}]: ").strip()
            or default_type
        ).lower()
        if hook_type in allowed_types:
            break
        console.print("[red]Please choose a supported hook type.[/red]")

    timeout_default = (existing_hook or {}).get("timeout", DEFAULT_HOOK_TIMEOUT)

    if hook_type == "prompt":
        existing_prompt = (existing_hook or {}).get("prompt", "")
        if existing_prompt:
            console.print(f"[dim]Current prompt:[/dim] {escape(existing_prompt)}", markup=False)
        while True:
            prompt_text = (
                console.input("Prompt template (use $ARGUMENTS for JSON input): ").strip()
                or existing_prompt
            )
            if prompt_text:
                break
            console.print("[red]Prompt text is required for prompt hooks.[/red]")
        timeout = _prompt_timeout(console, timeout_default)
        return {"type": "prompt", "prompt": prompt_text, "timeout": timeout}

    # Command hook
    existing_command = (existing_hook or {}).get("command", "")
    while True:
        command = (
            console.input(
                f"Command to run{f' [{existing_command}]' if existing_command else ''}: "
            ).strip()
            or existing_command
        )
        if command:
            break
        console.print("[red]Command is required for command hooks.[/red]")

    timeout = _prompt_timeout(console, timeout_default)
    return {"type": "command", "command": command, "timeout": timeout}


def _select_hook(console: Any, matcher: Dict[str, Any]) -> Optional[int]:
    """Prompt the user to choose a specific hook index."""
    hooks = matcher.get("hooks", [])
    if not hooks:
        console.print("[yellow]No hooks found under this matcher.[/yellow]")
        return None

    console.print("\nSelect hook:")
    for idx, hook in enumerate(hooks, start=1):
        console.print(f"  [{idx}] {_summarize_hook(hook)}")

    while True:
        choice = console.input(f"Hook [1-{len(hooks)}]: ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(hooks):
                return idx
        console.print("[red]Choose a hook number from the list.[/red]")


def _handle_add(ui: Any, tokens: List[str], project_path: Path) -> bool:
    """Handle /hooks add."""
    console = ui.console
    target = _select_target(console, project_path, tokens[0] if tokens else None)
    if not target:
        return True

    hooks = _load_hooks_json(console, target.path)
    event_name = _prompt_event_selection(
        console, [e.value for e in HookEvent], HookEvent.PRE_TOOL_USE.value
    )
    if not event_name:
        return True

    matchers = hooks.setdefault(event_name, [])
    matcher = _prompt_matcher_selection(console, event_name, matchers)
    if matcher is None:
        return True

    hook_def = _prompt_hook_details(console, event_name)
    if not hook_def:
        return True

    matcher.setdefault("hooks", []).append(hook_def)
    if not _save_hooks_json(console, target.path, hooks):
        return True

    console.print(
        f"[green]✓ Added hook to {escape(str(target.path))} under {escape(event_name)}.[/green]"
    )
    return True


def _handle_edit(ui: Any, tokens: List[str], project_path: Path) -> bool:
    """Handle /hooks edit."""
    console = ui.console
    target = _select_target(console, project_path, tokens[0] if tokens else None)
    if not target:
        return True

    hooks = _load_hooks_json(console, target.path)
    if not hooks:
        console.print("[yellow]No hooks found in this file. Use /hooks add to create one.[/yellow]")
        return True

    event_options = list(hooks.keys())
    event_name = _prompt_event_selection(console, event_options, event_options[0])
    if not event_name:
        return True

    matchers = hooks.get(event_name, [])
    matcher = _prompt_matcher_selection(console, event_name, matchers)
    if matcher is None:
        return True

    hook_idx = _select_hook(console, matcher)
    if hook_idx is None:
        return True

    existing_hook = matcher.get("hooks", [])[hook_idx]
    updated_hook = _prompt_hook_details(console, event_name, existing_hook)
    if not updated_hook:
        return True

    matcher["hooks"][hook_idx] = updated_hook
    if not _save_hooks_json(console, target.path, hooks):
        return True

    console.print(
        f"[green]✓ Updated hook in {escape(str(target.path))} ({escape(event_name)}).[/green]"
    )
    return True


def _handle_delete(ui: Any, tokens: List[str], project_path: Path) -> bool:
    """Handle /hooks delete."""
    console = ui.console
    target = _select_target(console, project_path, tokens[0] if tokens else None)
    if not target:
        return True

    hooks = _load_hooks_json(console, target.path)
    if not hooks:
        console.print("[yellow]No hooks to delete.[/yellow]")
        return True

    event_options = list(hooks.keys())
    event_name = _prompt_event_selection(console, event_options, event_options[0])
    if not event_name:
        return True

    matchers = hooks.get(event_name, [])
    matcher = _prompt_matcher_selection(console, event_name, matchers)
    if matcher is None:
        return True

    hook_idx = _select_hook(console, matcher)
    if hook_idx is None:
        return True

    confirmation = console.input("Delete this hook? [Y/n]: ").strip().lower()
    if confirmation not in ("", "y", "yes"):
        console.print("[yellow]Delete cancelled.[/yellow]")
        return True

    matcher["hooks"].pop(hook_idx)
    if not matcher["hooks"]:
        matchers.remove(matcher)
    if not matchers:
        hooks.pop(event_name, None)

    if not _save_hooks_json(console, target.path, hooks):
        return True

    console.print(
        f"[green]✓ Deleted hook from {escape(str(target.path))} ({escape(event_name)}).[/green]"
    )
    return True


def _handle(ui: Any, arg: str) -> bool:
    """Entry point for /hooks command."""
    project_path = getattr(ui, "project_path", None) or Path.cwd()
    tokens = arg.split()
    subcmd = tokens[0].lower() if tokens else ""

    if subcmd in ("help", "-h", "--help"):
        _print_usage(ui.console)
        return True

    if subcmd in ("add", "create", "new"):
        return _handle_add(ui, tokens[1:], project_path)

    if subcmd in ("edit", "update"):
        return _handle_edit(ui, tokens[1:], project_path)

    if subcmd in ("delete", "del", "remove", "rm"):
        return _handle_delete(ui, tokens[1:], project_path)

    if subcmd:
        ui.console.print(f"[red]Unknown hooks subcommand '{escape(subcmd)}'.[/red]")
        _print_usage(ui.console)
        return True

    return _render_hooks_overview(ui, project_path)


command = SlashCommand(
    name="hooks",
    description="Show configured hooks and manage them",
    handler=_handle,
)


__all__ = ["command"]
