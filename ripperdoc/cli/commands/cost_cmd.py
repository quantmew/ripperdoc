from ripperdoc.utils.session_usage import get_session_usage

from typing import Any
from .base import SlashCommand


def _fmt_tokens(value: int) -> str:
    """Format integers with thousand separators."""
    return f"{int(value):,}"


def _format_duration(duration_ms: float) -> str:
    """Render milliseconds into a compact human-readable duration."""
    seconds = int(duration_ms // 1000)
    if seconds < 60:
        return f"{duration_ms / 1000:.2f}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours, mins = divmod(minutes, 60)
    return f"{hours}h {mins}m {secs}s"


def _handle(ui: Any, _: str) -> bool:
    usage = get_session_usage()
    if not usage.models:
        ui.console.print("[yellow]No model usage recorded yet.[/yellow]")
        return True

    total_input = usage.total_input_tokens
    total_output = usage.total_output_tokens
    total_cache_read = usage.total_cache_read_tokens
    total_cache_creation = usage.total_cache_creation_tokens
    total_tokens = total_input + total_output + total_cache_read + total_cache_creation
    total_cost = usage.total_cost_usd

    ui.console.print("\n[bold]Session token usage[/bold]")
    ui.console.print(
        f" Total: {_fmt_tokens(total_tokens)} tokens "
        f"(input {_fmt_tokens(total_input)}, output {_fmt_tokens(total_output)})"
    )
    if total_cache_read or total_cache_creation:
        ui.console.print(
            f" Cache: {_fmt_tokens(total_cache_read)} read, "
            f"{_fmt_tokens(total_cache_creation)} write"
        )
    ui.console.print(f" Requests: {usage.total_requests}")
    if total_cost:
        ui.console.print(f" Cost: ${total_cost:.4f}")
    if usage.total_duration_ms:
        ui.console.print(f" API time: {_format_duration(usage.total_duration_ms)}")

    ui.console.print("\n[bold]By model:[/bold]")
    for model_name, stats in usage.models.items():
        line = (
            f"  {model_name}: "
            f"{_fmt_tokens(stats.input_tokens)} in, "
            f"{_fmt_tokens(stats.output_tokens)} out"
        )
        if stats.cache_read_input_tokens:
            line += f", {_fmt_tokens(stats.cache_read_input_tokens)} cache read"
        if stats.cache_creation_input_tokens:
            line += f", {_fmt_tokens(stats.cache_creation_input_tokens)} cache write"
        line += f" ({stats.requests} call{'' if stats.requests == 1 else 's'}"
        if stats.duration_ms:
            line += f", {_format_duration(stats.duration_ms)} total"
        line += ")"
        if stats.cost_usd:
            line += f", ${stats.cost_usd:.4f}"
        ui.console.print(line)

    return True


command = SlashCommand(
    name="cost",
    description="Show total tokens used in this session",
    handler=_handle,
)


__all__ = ["command"]
