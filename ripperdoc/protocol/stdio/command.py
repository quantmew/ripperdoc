"""CLI entrypoints for stdio protocol support."""

from __future__ import annotations

import asyncio
from typing import Any

import click

from .handler import StdioProtocolHandler


@click.command(name="stdio")
@click.option(
    "--input-format",
    type=click.Choice(["stream-json", "auto"]),
    default="stream-json",
    help="Input format for messages.",
)
@click.option(
    "--output-format",
    type=click.Choice(["stream-json", "json"]),
    default="stream-json",
    help="Output format for messages.",
)
@click.option(
    "--model",
    type=str,
    default=None,
    help="Model profile for the current session.",
)
@click.option(
    "--permission-mode",
    type=click.Choice(["default", "acceptEdits", "plan", "dontAsk", "bypassPermissions"]),
    default="default",
    help="Permission mode for tool usage.",
)
@click.option(
    "--max-turns",
    type=int,
    default=None,
    help="Maximum number of conversation turns.",
)
@click.option(
    "--system-prompt",
    type=str,
    default=None,
    help="System prompt to use for the session.",
)
@click.option(
    "--print",
    "-p",
    is_flag=True,
    help="Print mode (for single prompt queries).",
)
@click.option(
    "--",
    "prompt",
    type=str,
    default=None,
    help="Direct prompt (for print mode).",
)
def stdio_cmd(
    input_format: str,
    output_format: str,
    model: str | None,
    permission_mode: str,
    max_turns: int | None,
    system_prompt: str | None,
    print: bool,
    prompt: str | None,
) -> None:
    """Stdio mode for SDK subprocess communication.

    This command enables Ripperdoc to communicate with SDKs via JSON Control
    Protocol over stdin/stdout. It's designed for subprocess architecture where
    the SDK manages the CLI process.

    The protocol supports:
    - control_request/control_response for protocol management
    - Message streaming for query results
    - Bidirectional communication for hooks and permissions

    Example:
        ripperdoc stdio --output-format stream-json
    """
    # Set up async event loop
    asyncio.run(
        run_stdio(
            input_format=input_format,
            output_format=output_format,
            model=model,
            permission_mode=permission_mode,
            max_turns=max_turns,
            system_prompt=system_prompt,
            print_mode=print,
            prompt=prompt,
        )
    )


async def run_stdio(
    input_format: str,
    output_format: str,
    model: str | None,
    permission_mode: str,
    max_turns: int | None,
    system_prompt: str | None,
    print_mode: bool,
    prompt: str | None,
    default_options: dict[str, Any] | None = None,
) -> None:
    """Async entry point for stdio command."""
    handler = StdioProtocolHandler(
        input_format=input_format,
        output_format=output_format,
        default_options=default_options,
    )

    # If print mode with prompt, handle as single query
    if print_mode and prompt:
        # This is a single-shot query mode
        # Initialize with defaults and run query
        request_options: dict[str, Any] = dict(default_options or {})
        if model is not None:
            request_options["model"] = model
        if permission_mode is not None:
            request_options["permission_mode"] = permission_mode
        if max_turns is not None:
            request_options["max_turns"] = max_turns
        if system_prompt is not None:
            request_options["system_prompt"] = system_prompt
        request_options["sdk_can_use_tool"] = False

        request = {
            "options": request_options,
            "prompt": prompt,
        }

        # Mock request_id for print mode
        request_id = "print_query"

        # Initialize
        await handler._handle_initialize(request, request_id)

        # Query
        query_request = {
            "prompt": prompt,
        }
        await handler._handle_query(query_request, f"{request_id}_query")
        await handler.flush_output()

        return

    # Otherwise, run the stdio protocol loop
    await handler.run()


__all__ = ["stdio_cmd", "run_stdio"]
