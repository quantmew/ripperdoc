from rich.markup import escape

from ripperdoc.utils.mcp import load_mcp_servers_async

from typing import Any
from .base import SlashCommand


def _run_in_ui(ui: Any, coro: Any) -> Any:
    runner = getattr(ui, "run_async", None)
    if callable(runner):
        return runner(coro)
    # Fallback for non-UI contexts.
    import asyncio

    return asyncio.run(coro)


def _handle(ui: Any, _: str) -> bool:
    async def _load() -> list:
        return await load_mcp_servers_async(ui.project_path)

    servers = _run_in_ui(ui, _load())
    if not servers:
        ui.console.print(
            "[yellow]No MCP servers configured. Add servers to ~/.ripperdoc/mcp.json, ~/.mcp.json, or a project .mcp.json file.[/yellow]"
        )
        return True

    ui.console.print("\n[bold]MCP servers[/bold]")
    for server in servers:
        status = server.status or "unknown"
        url_part = f" ({server.url})" if server.url else ""
        ui.console.print(f"- {server.name}{url_part} — {status}", markup=False)
        if server.command:
            cmd_line = " ".join([server.command, *server.args]) if server.args else server.command
            ui.console.print(f"    Command: {cmd_line}", markup=False)
        if server.description:
            ui.console.print(f"    {server.description}", markup=False)
        if server.error:
            ui.console.print(f"    [red]Error:[/red] {escape(str(server.error))}")
        if server.instructions:
            snippet = server.instructions.strip()
            if len(snippet) > 160:
                snippet = snippet[:157] + "..."
            ui.console.print(f"    Instructions: {snippet}", markup=False)
        if server.tools:
            ui.console.print("    Tools:")
            for tool in server.tools:
                desc = f" — {tool.description}" if tool.description else ""
                ui.console.print(f"      • {tool.name}{desc}", markup=False)
        else:
            ui.console.print("    Tools: none discovered")
        if server.resources:
            ui.console.print(
                "    Resources: " + ", ".join(res.uri for res in server.resources), markup=False
            )
        elif not server.tools:
            ui.console.print("    Resources: none")
    return True


command = SlashCommand(
    name="mcp",
    description="Show configured MCP servers and their tools",
    handler=_handle,
)


__all__ = ["command"]
