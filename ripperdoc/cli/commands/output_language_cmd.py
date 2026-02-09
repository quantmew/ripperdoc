"""Output language command for controlling assistant response language."""

from __future__ import annotations

from typing import Any

from rich.markup import escape

from ripperdoc.core.config import get_project_local_config, save_project_local_config

from .base import SlashCommand


def _current_language(ui: Any) -> str:
    language = getattr(ui, "output_language", "auto")
    if isinstance(language, str) and language.strip():
        return language.strip()
    return "auto"


def _normalize_language(value: str) -> str:
    trimmed = value.strip()
    if not trimmed:
        return "auto"
    if trimmed.lower() in {"auto", "default", "reset", "follow-user", "follow_user"}:
        return "auto"
    return trimmed


def _apply_language(ui: Any, language: str) -> None:
    config = get_project_local_config(ui.project_path)
    config.output_language = language
    save_project_local_config(config, ui.project_path)

    setter = getattr(ui, "set_output_language", None)
    if callable(setter):
        setter(language)
    else:
        setattr(ui, "output_language", language)


def _show_current(ui: Any) -> bool:
    current = _current_language(ui)
    ui.console.print(f"[bold]Output language:[/bold] {escape(current)}")
    ui.console.print(
        "[dim]Use /output-language <language> to set, or /output-language auto to follow user language.[/dim]"
    )
    return True


def _handle(ui: Any, arg: str) -> bool:
    trimmed = arg.strip()
    if not trimmed or trimmed.lower() in {"show", "status", "current"}:
        return _show_current(ui)

    language = _normalize_language(trimmed)
    _apply_language(ui, language)
    if language == "auto":
        ui.console.print("[green]✓ Output language set to auto (follow user language).[/green]")
    else:
        ui.console.print(f"[green]✓ Output language set to {escape(language)}.[/green]")
    return True


command = SlashCommand(
    name="output-language",
    description="Set preferred response language for system prompt",
    handler=_handle,
    aliases=("language",),
)


__all__ = ["command"]
