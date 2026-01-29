from rich.markup import escape

from ripperdoc.core.skills import (
    SkillDefinition,
    SkillLoadResult,
    SkillLocation,
    load_all_skills,
    skill_directories,
)

from typing import Any
from .base import SlashCommand


def _handle(ui: Any, _: str) -> bool:
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

    # Group skills by location for better organization
    user_skills = [s for s in result.skills if s.location == SkillLocation.USER]
    project_skills = [s for s in result.skills if s.location == SkillLocation.PROJECT]
    other_skills = [s for s in result.skills if s.location == SkillLocation.OTHER]

    def print_skill(skill: SkillDefinition) -> None:
        location_tag = f"[dim]({skill.location.value})[/dim]" if skill.location else ""
        console.print(f"\n[bold cyan]{escape(skill.name)}[/bold cyan] {location_tag}")
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

    # Print project skills first (they have priority), then user skills
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


command = SlashCommand(
    name="skills",
    description="List available skills",
    handler=_handle,
)


__all__ = ["command"]
