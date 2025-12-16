"""Slash command to view and edit AGENTS memory files."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, List, Optional

from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from ripperdoc.utils.memory import (
    LOCAL_MEMORY_FILE_NAME,
    MEMORY_FILE_NAME,
    collect_all_memory_files,
)

from .base import SlashCommand


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


def _preferred_user_memory_path() -> Path:
    """Pick the user-level memory path, preferring ~/.ripperdoc/AGENTS.md."""
    home = Path.home()
    preferred_dir = home / ".ripperdoc"
    preferred_path = preferred_dir / MEMORY_FILE_NAME
    fallback_path = home / MEMORY_FILE_NAME

    if preferred_path.exists():
        return preferred_path
    if fallback_path.exists():
        return fallback_path

    preferred_dir.mkdir(parents=True, exist_ok=True)
    return preferred_path


def _ensure_file(path: Path) -> bool:
    """Ensure the target file exists. Returns True if created."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return False
    path.write_text("", encoding="utf-8")
    return True


def _ensure_gitignore_entry(project_path: Path, entry: str) -> bool:
    """Ensure an entry exists in .gitignore. Returns True if added."""
    gitignore_path = project_path / ".gitignore"
    try:
        text = ""
        if gitignore_path.exists():
            text = gitignore_path.read_text(encoding="utf-8", errors="ignore")
            existing_lines = text.splitlines()
            if entry in existing_lines:
                return False
        with gitignore_path.open("a", encoding="utf-8") as f:
            if text and not text.endswith("\n"):
                f.write("\n")
            f.write(f"{entry}\n")
        return True
    except (OSError, IOError):
        return False


def _determine_editor_command() -> Optional[List[str]]:
    """Resolve the editor command from environment or common defaults."""
    for env_var in ("VISUAL", "EDITOR"):
        value = os.environ.get(env_var)
        if value:
            return shlex.split(value)

    candidates = ["code", "nano", "vim", "vi"]
    if os.name == "nt":
        candidates.insert(0, "notepad")

    for candidate in candidates:
        if shutil.which(candidate):
            return [candidate]
    return None


def _open_in_editor(path: Path, console: Any) -> bool:
    """Open the file in a text editor; returns True if an editor was launched."""
    editor_cmd = _determine_editor_command()
    if not editor_cmd:
        console.print(
            f"[yellow]No editor configured. Set $EDITOR or $VISUAL, "
            f"or manually edit: {escape(str(path))}[/yellow]"
        )
        return False

    cmd = [*editor_cmd, str(path)]
    try:
        console.print(f"[dim]Opening with: {' '.join(editor_cmd)}[/dim]")
        subprocess.run(cmd, check=False)
        return True
    except FileNotFoundError:
        console.print(f"[red]Editor command not found: {escape(editor_cmd[0])}[/red]")
        return False
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - best-effort logging
        console.print(f"[red]Failed to launch editor: {escape(str(exc))}[/red]")
        return False


def _render_memory_table(console: Any, project_path: Path) -> None:
    files = collect_all_memory_files()
    table = Table(title="Memory files", show_header=True, header_style="bold cyan")
    table.add_column("Type", style="bold")
    table.add_column("Location")
    table.add_column("Nested", justify="center")

    for memory_file in files:
        display_path = _shorten_path(Path(memory_file.path), project_path)
        nested = "yes" if getattr(memory_file, "is_nested", False) else ""
        table.add_row(memory_file.type, escape(display_path), nested)

    if files:
        console.print(table)
    else:
        console.print("[yellow]No memory files found yet.[/yellow]")


def _handle(ui: Any, trimmed_arg: str) -> bool:
    project_path = getattr(ui, "project_path", Path.cwd())
    scope = trimmed_arg.strip().lower()

    if scope:
        scope_aliases = {
            "project": "project",
            "workspace": "project",
            "local": "local",
            "private": "local",
            "user": "user",
            "global": "user",
        }
        if scope not in scope_aliases:
            ui.console.print("[red]Unknown scope. Use one of: project, local, user.[/red]")
            return True

        resolved_scope = scope_aliases[scope]
        if resolved_scope == "project":
            target_path = project_path / MEMORY_FILE_NAME
            heading = "Project memory (checked in)"
        elif resolved_scope == "local":
            target_path = project_path / LOCAL_MEMORY_FILE_NAME
            heading = "Local memory (not checked in)"
        else:
            target_path = _preferred_user_memory_path()
            heading = "User memory (home directory)"

        created = _ensure_file(target_path)
        gitignore_added = False
        if resolved_scope == "local":
            gitignore_added = _ensure_gitignore_entry(project_path, LOCAL_MEMORY_FILE_NAME)

        _open_in_editor(target_path, ui.console)

        messages: List[str] = [f"{heading}: {escape(_shorten_path(target_path, project_path))}"]
        if created:
            messages.append("Created new memory file.")
        if gitignore_added:
            messages.append("Added AGENTS.local.md to .gitignore.")
        if not created:
            messages.append("Opened existing memory file.")

        ui.console.print(Panel("\n".join(messages), title="/memory"))
        return True

    _render_memory_table(ui.console, project_path)
    ui.console.print("[dim]Usage: /memory project | /memory local | /memory user[/dim]")
    ui.console.print("[dim]Project and user memories feed directly into the system prompt.[/dim]")
    return True


command = SlashCommand(
    name="memory",
    description="List and edit AGENTS memory files",
    handler=_handle,
    aliases=(),
)


__all__ = ["command"]
