from __future__ import annotations

from typing import Any

from rich.markup import escape

from ripperdoc.utils.session_index import set_session_index_title

from .base import SlashCommand

_MAX_AUTO_TITLE_LEN = 80


def _extract_text_from_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    parts: list[str] = []
    for block in content:
        block_type = getattr(block, "type", None)
        if block_type is None and isinstance(block, dict):
            block_type = block.get("type")
        if block_type != "text":
            continue
        text_val = getattr(block, "text", None)
        if text_val is None and isinstance(block, dict):
            text_val = block.get("text")
        if text_val:
            parts.append(str(text_val))
    return "\n".join(parts)


def _candidate_title_from_messages(ui: Any) -> str:
    messages = getattr(ui, "conversation_messages", None)
    if not isinstance(messages, list) or not messages:
        return ""

    for msg in reversed(messages):
        if getattr(msg, "type", None) != "user":
            continue
        message_obj = getattr(msg, "message", None)
        content = getattr(message_obj, "content", None) if message_obj is not None else None
        if content is None:
            content = getattr(msg, "content", None)

        extractor = getattr(ui, "_extract_visible_text", None)
        if callable(extractor):
            text = str(extractor(content) or "")
        else:
            text = _extract_text_from_content(content)
        normalized = " ".join(text.split()).strip()
        if not normalized:
            continue
        if len(normalized) > _MAX_AUTO_TITLE_LEN:
            normalized = normalized[: _MAX_AUTO_TITLE_LEN - 3].rstrip() + "..."
        return normalized
    return ""


def _handle(ui: Any, arg: str) -> bool:
    project_path = getattr(ui, "project_path", None)
    session_id = getattr(ui, "session_id", None)
    if project_path is None or not session_id:
        ui.console.print("[red]Rename is not available in this UI.[/red]")
        return True

    requested = (arg or "").strip()
    if requested:
        title = requested
    else:
        title = _candidate_title_from_messages(ui)
        if not title:
            ui.console.print(
                "[yellow]Could not generate a name: no conversation context yet. "
                "Usage: /rename[/yellow]"
            )
            return True

    set_session_index_title(project_path, session_id, title)
    ui.console.print(f"[green]⎿  Session renamed to: {escape(title)}[/green]")
    return True


command = SlashCommand(
    name="rename",
    description="Rename current session title",
    handler=_handle,
)


__all__ = ["command"]
