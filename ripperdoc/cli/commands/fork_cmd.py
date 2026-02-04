from typing import Any

from .base import SlashCommand


def _handle(ui: Any, _: str) -> bool:
    fork = getattr(ui, "_fork_session", None)
    if not callable(fork):
        ui.console.print("[red]Fork is not available in this UI.[/red]")
        return True
    old_session_id, _new_session_id = fork()
    ui.console.print(
        "[green]⎿  Forked conversation. You are now in the fork.[/green]\n"
        f"[dim]   To resume the original: ripperdoc -r {old_session_id}[/dim]"
    )
    return True


command = SlashCommand(
    name="fork",
    description="Fork the current conversation into a new session",
    handler=_handle,
)


__all__ = ["command"]
