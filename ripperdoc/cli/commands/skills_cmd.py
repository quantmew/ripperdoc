from __future__ import annotations

import sys
from typing import Any

from rich.markup import escape

from ripperdoc.core.skills import (
    SkillDefinition,
    SkillLoadResult,
    SkillLocation,
    is_skill_definition_disabled,
    load_all_skills,
    set_skill_enabled,
    skill_directories,
)

from .base import SlashCommand


def _print_usage(console: Any) -> None:
    console.print("[bold]/skills[/bold] — open Enable/Disable Skills UI")
    console.print("[bold]/skills tui[/bold] — open Enable/Disable Skills UI")
    console.print("[bold]/skills list[/bold] — show skill details (plain)")
    console.print("[bold]/skills enable <name>[/bold] — enable one skill")
    console.print("[bold]/skills disable <name>[/bold] — disable one skill")


def _render_skills_plain(ui: Any) -> bool:
    console = ui.console
    project_path = getattr(ui, "project_path", None)
    result: SkillLoadResult = load_all_skills(project_path=project_path)

    if not result.skills:
        dirs = skill_directories(project_path=project_path)
        dir_paths = [f"'{d}'" for d, _ in dirs]
        console.print("[yellow]No skills found.[/yellow]")
        console.print(
            f"\n[bold]Create skills in:[/bold]\n"
            f"  • User: {dir_paths[0]}\n"
            f"  • Project: {dir_paths[1]}\n"
            f"\n[dim]Each skill needs a SKILL.md file with YAML frontmatter:\n"
            f"---\n"
            f"name: my-skill\n"
            f"description: A helpful skill\n"
            f"---\n\n"
            f"Skill content goes here...[/dim]"
        )
        if result.errors:
            console.print("\n[bold red]Errors encountered while loading skills:[/bold red]")
            for error in result.errors:
                console.print(f"  • {error.path}: {escape(error.reason)}", markup=False)
        return True

    console.print("\n[bold]Skills[/bold]")
    console.print("[dim]Use /skills to toggle enabled state.[/dim]")

    user_skills = [s for s in result.skills if s.location == SkillLocation.USER]
    project_skills = [s for s in result.skills if s.location == SkillLocation.PROJECT]
    other_skills = [s for s in result.skills if s.location == SkillLocation.OTHER]

    def print_skill(skill: SkillDefinition) -> None:
        location_tag = f"[dim]({skill.location.value})[/dim]" if skill.location else ""
        disabled = is_skill_definition_disabled(skill, project_path=project_path)
        status_tag = "[yellow]disabled[/yellow]" if disabled else "[green]enabled[/green]"
        console.print(f"\n[bold cyan]{escape(skill.name)}[/bold cyan] {location_tag} [dim]- {status_tag}[/dim]")
        console.print(f"  {escape(skill.description)}")

        if skill.allowed_tools:
            tools_str = ", ".join(escape(t) for t in skill.allowed_tools)
            console.print(f"  Tools: {tools_str}")

        if skill.model:
            console.print(f"  Model: {escape(skill.model)}")

        if skill.max_thinking_tokens:
            console.print(f"  Max thinking tokens: {skill.max_thinking_tokens}")

        if skill.skill_type != "prompt":
            console.print(f"  Type: {escape(skill.skill_type)}")

        if skill.disable_model_invocation:
            console.print("  [yellow]Model invocation disabled[/yellow]")

        console.print(f"  Path: {escape(str(skill.path))}", markup=False)

    if project_skills:
        console.print("\n[bold]Project skills:[/bold]")
        for skill in project_skills:
            print_skill(skill)

    if user_skills:
        console.print("\n[bold]User skills:[/bold]")
        for skill in user_skills:
            print_skill(skill)

    if other_skills:
        console.print("\n[bold]Other skills:[/bold]")
        for skill in other_skills:
            print_skill(skill)

    if result.errors:
        console.print("\n[bold red]Errors encountered while loading skills:[/bold red]")
        for error in result.errors:
            console.print(f"  • {error.path}: {escape(error.reason)}", markup=False)

    return True


def _set_skill_enabled(ui: Any, skill_name: str, enabled: bool) -> bool:
    normalized = skill_name.strip().lstrip("/")
    if not normalized:
        ui.console.print("[red]Usage: /skills enable|disable <skill-name>[/red]")
        return True

    project_path = getattr(ui, "project_path", None)
    all_skills = load_all_skills(project_path=project_path).skills
    skill = next((item for item in all_skills if item.name == normalized), None)
    if not skill:
        ui.console.print(f"[red]Unknown skill: {escape(normalized)}[/red]")
        return True

    if skill.location not in (SkillLocation.USER, SkillLocation.PROJECT):
        ui.console.print(
            f"[yellow]Skill '{escape(normalized)}' comes from '{escape(skill.location.value)}' and cannot be toggled persistently.[/yellow]"
        )
        return True

    changed = set_skill_enabled(skill, enabled=enabled, project_path=project_path)
    if not changed:
        current_state = "enabled" if enabled else "disabled"
        ui.console.print(f"[dim]Skill '{escape(normalized)}' is already {current_state}.[/dim]")
        return True
    state = "enabled" if enabled else "disabled"
    ui.console.print(f"[green]Skill '{escape(normalized)}' {state}.[/green]")
    return True


def _handle_skills_tui(ui: Any) -> bool:
    console = ui.console
    if not sys.stdin.isatty():
        console.print("[yellow]Interactive UI requires a TTY. Showing plain list instead.[/yellow]")
        _render_skills_plain(ui)
        return True

    try:
        from ripperdoc.cli.ui.skills_tui import run_skills_tui
    except (ImportError, ModuleNotFoundError) as exc:
        console.print(
            f"[yellow]Textual UI not available ({escape(str(exc))}). Showing plain list.[/yellow]"
        )
        _render_skills_plain(ui)
        return True

    try:
        project_path = getattr(ui, "project_path", None)
        return bool(run_skills_tui(project_path))
    except Exception as exc:  # noqa: BLE001 - fail safe in interactive UI
        console.print(f"[red]Textual UI failed: {escape(str(exc))}[/red]")
        _render_skills_plain(ui)
        return True


def _handle(ui: Any, trimmed_arg: str) -> bool:
    args = trimmed_arg.strip().split()
    subcmd = args[0].lower() if args else ""

    if subcmd in ("", "tui", "ui"):
        return _handle_skills_tui(ui)
    if subcmd in ("help", "-h", "--help"):
        _print_usage(ui.console)
        return True
    if subcmd in ("list", "ls"):
        return _render_skills_plain(ui)
    if subcmd in ("enable", "on"):
        if len(args) < 2:
            ui.console.print("[red]Usage: /skills enable <skill-name>[/red]")
            return True
        return _set_skill_enabled(ui, args[1], enabled=True)
    if subcmd in ("disable", "off"):
        if len(args) < 2:
            ui.console.print("[red]Usage: /skills disable <skill-name>[/red]")
            return True
        return _set_skill_enabled(ui, args[1], enabled=False)

    ui.console.print(f"[red]Unknown action: {escape(subcmd)}[/red]")
    _print_usage(ui.console)
    return True


command = SlashCommand(
    name="skills",
    description="Enable/disable skills or list skill details",
    handler=_handle,
)


__all__ = ["command"]
