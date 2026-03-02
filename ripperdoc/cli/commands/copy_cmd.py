"""Copy command - copy last assistant response text to clipboard."""

from __future__ import annotations

from typing import Any

from rich.markup import escape

from .base import SlashCommand
from .export_cmd import _copy_to_clipboard


def _join_text_blocks(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    texts: list[str] = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text" and block.get("text"):
                texts.append(str(block.get("text")))
            continue
        if getattr(block, "type", None) == "text":
            text = getattr(block, "text", None)
            if text:
                texts.append(str(text))
    return "\n\n".join(texts)


def _find_last_assistant_message(messages: list[Any]) -> Any | None:
    for message in reversed(messages):
        if getattr(message, "type", None) == "assistant":
            return message
    return None


def _handle(ui: Any, arg: str) -> bool:
    _ = arg
    messages = list(getattr(ui, "conversation_messages", []) or [])

    message = _find_last_assistant_message(messages)
    if message is None:
        ui.console.print("[yellow]No assistant message to copy[/yellow]")
        return True

    content = getattr(getattr(message, "message", None), "content", None)
    if isinstance(content, list) and len(content) == 0:
        ui.console.print("[yellow]No content to copy[/yellow]")
        return True
    if content is None:
        ui.console.print("[yellow]No content to copy[/yellow]")
        return True

    copied_text = _join_text_blocks(content)
    if not copied_text:
        ui.console.print("[yellow]No text content to copy[/yellow]")
        return True

    ok, error = _copy_to_clipboard(copied_text)
    if ok:
        line_count = copied_text.count("\n") + 1
        ui.console.print(
            f"[green]Copied to clipboard ({len(copied_text)} characters, {line_count} lines)[/green]"
        )
        return True

    ui.console.print(f"[red]{escape(error)}[/red]")
    return True


command = SlashCommand(
    name="copy",
    description="Copy Claude's last response to clipboard as markdown",
    handler=_handle,
)


__all__ = ["command"]
