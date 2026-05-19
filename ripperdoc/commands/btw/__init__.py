"""Btw command - ask a quick side question without interrupting the main conversation."""

from __future__ import annotations

from typing import Any, List

from rich.markdown import Markdown
from rich.markup import escape

from ripperdoc.commands.base import SlashCommand
from ripperdoc.utils.messaging.message_utils import resolve_model_profile
from ripperdoc.services.providers import get_provider_client

TOOL_MODE_NONE = "none"


async def _do_side_question(ui: Any, question: str) -> None:
    """Run a side question against the LLM and print the answer."""
    model_pointer = getattr(ui, "model", None) or "main"
    profile = resolve_model_profile(model_pointer)
    if profile is None:
        ui.console.print("[red]No active model profile found. Configure a model first.[/red]")
        return

    if profile.protocol is None:
        ui.console.print("[red]Model profile has no protocol configured.[/red]")
        return

    client = get_provider_client(profile.protocol)
    if client is None:
        ui.console.print(
            f"[red]No provider client available for protocol: {profile.protocol}[/red]"
        )
        return

    normalized_messages: List[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": question}]}
    ]

    system_prompt = (
        "You are a helpful, concise assistant. "
        "Answer the user's question directly and briefly. "
        "Use markdown formatting where appropriate."
    )

    ui.console.print("[dim]/btw answering...[/dim]")
    try:
        response = await client.call(
            model_profile=profile,
            system_prompt=system_prompt,
            normalized_messages=normalized_messages,
            tools=[],
            tool_mode=TOOL_MODE_NONE,
            stream=True,
            progress_callback=None,
            request_timeout=60.0,
            max_retries=1,
            max_thinking_tokens=0,
        )
    except Exception as exc:
        ui.console.print(f"[red]Error: {escape(str(exc))}[/red]")
        return

    if response.is_error:
        ui.console.print(
            f"[red]Error: {escape(response.error_message or 'Unknown error')}[/red]"
        )
        return

    text_parts: List[str] = []
    for block in response.content_blocks:
        if isinstance(block, dict):
            if block.get("type") == "text":
                text_parts.append(str(block.get("text", "")))
        elif hasattr(block, "type") and block.type == "text":
            text_parts.append(str(getattr(block, "text", "")))

    full_text = "\n".join(text_parts)
    if full_text:
        ui.console.print()
        ui.console.print(Markdown(full_text))
        ui.console.print()
        ui.console.print("[dim]— /btw answer (not saved to conversation)[/dim]")
    else:
        ui.console.print("[yellow]No response content received.[/yellow]")


def _handle(ui: Any, arg: str) -> bool:
    """Handle the /btw command."""
    question = arg.strip()
    if not question:
        ui.console.print("[yellow]Usage: /btw <your quick question>[/yellow]")
        return True

    ui.run_async(_do_side_question(ui, question))
    return True


command = SlashCommand(
    name="btw",
    description="Ask a quick side question without interrupting the main conversation",
    handler=_handle,
    aliases=(),
)


__all__ = ["command"]
