from ripperdoc.core.config import get_global_config
from ripperdoc.cli.ui.helpers import get_profile_for_pointer

from .base import SlashCommand


def _handle(ui, _: str) -> bool:
    config = get_global_config()
    profile = get_profile_for_pointer("main")
    main_pointer = getattr(config.model_pointers, "main", "default")
    model_label = profile.model if profile else "Not configured"

    ui.console.print(f"\n[bold]Model (main -> {main_pointer}):[/bold] {model_label}")
    ui.console.print(f"[bold]Safe Mode:[/bold] {ui.safe_mode}")
    ui.console.print(f"[bold]Verbose:[/bold] {ui.verbose}")
    return True


command = SlashCommand(
    name="config",
    description="Show current configuration",
    handler=_handle,
)


__all__ = ["command"]
