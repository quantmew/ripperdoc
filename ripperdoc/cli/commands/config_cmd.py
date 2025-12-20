from rich.markup import escape

from ripperdoc.core.config import get_global_config
from ripperdoc.cli.ui.helpers import get_profile_for_pointer

from typing import Any
from .base import SlashCommand


def _handle(ui: Any, _: str) -> bool:
    config = get_global_config()
    profile = get_profile_for_pointer("main")
    main_pointer = getattr(config.model_pointers, "main", "default")
    model_label = profile.model if profile else "Not configured"

    ui.console.print(
        f"\n[bold]Model (main -> {escape(str(main_pointer))}):[/bold] {escape(str(model_label))}"
    )
    ui.console.print(f"[bold]Yolo Mode:[/bold] {escape(str(ui.yolo_mode))}")
    ui.console.print(f"[bold]Verbose:[/bold] {escape(str(ui.verbose))}")
    return True


command = SlashCommand(
    name="config",
    description="Show current configuration",
    handler=_handle,
)


__all__ = ["command"]
