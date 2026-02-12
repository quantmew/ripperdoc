"""Slash command for MCP server inspection and MCP stderr log viewing."""

from __future__ import annotations

import asyncio
import os
import shlex
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Optional, cast

from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from ripperdoc.utils.mcp import (
    McpServerInfo,
    get_mcp_stderr_log_path,
    get_mcp_stderr_mode,
    load_mcp_servers_async,
)

from .base import SlashCommand

_DEFAULT_LOG_LINES = 80
_MAX_LOG_LINES = 2000
_FOLLOW_POLL_SECONDS = 0.25


def _run_in_ui(ui: Any, coro: Any) -> Any:
    runner = getattr(ui, "run_async", None)
    if callable(runner):
        return runner(coro)
    # Fallback for non-UI contexts.
    return asyncio.run(coro)


def _print_usage(ui: Any) -> None:
    ui.console.print("[dim]Usage:[/dim]")
    ui.console.print("[dim]  /mcp[/dim]")
    ui.console.print("[dim]  /mcp tui[/dim]")
    ui.console.print("[dim]  /mcp list[/dim]")
    ui.console.print("[dim]  /mcp logs[/dim]")
    ui.console.print("[dim]  /mcp logs <server> [-n|--lines <count>] [-f|--follow][/dim]")


def _load_servers(ui: Any) -> list[McpServerInfo]:
    async def _load() -> list[McpServerInfo]:
        return await load_mcp_servers_async(ui.project_path)

    return cast(list[McpServerInfo], _run_in_ui(ui, _load()))


def _show_server_overview(ui: Any, servers: list[McpServerInfo]) -> bool:
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
        if status == "connecting":
            ui.console.print("    Tools: discovering...")
        elif server.tools:
            ui.console.print("    Tools:")
            for tool in server.tools:
                desc = f" — {tool.description}" if tool.description else ""
                if tool.description and len(tool.description) > 80:
                    desc = f" — {tool.description[:77]}..."
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


def _handle_tui(ui: Any) -> bool:
    if not sys.stdin.isatty():
        ui.console.print("[yellow]Interactive UI requires a TTY. Showing plain overview.[/yellow]")
        servers = _load_servers(ui)
        return _show_server_overview(ui, servers)
    try:
        from ripperdoc.cli.ui.mcp_tui import run_mcp_tui
    except (ImportError, ModuleNotFoundError) as exc:
        ui.console.print(
            f"[yellow]Textual UI not available ({escape(str(exc))}). Showing plain overview.[/yellow]"
        )
        servers = _load_servers(ui)
        return _show_server_overview(ui, servers)

    try:
        return bool(run_mcp_tui(getattr(ui, "project_path", None)))
    except Exception as exc:  # noqa: BLE001 - fail safe in interactive UI
        ui.console.print(f"[red]Textual UI failed: {escape(str(exc))}[/red]")
        servers = _load_servers(ui)
        return _show_server_overview(ui, servers)


def _tail_text(path: Path, line_count: int) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            lines = deque(handle, maxlen=line_count)
    except OSError as exc:
        return f"(failed to read log: {exc})"
    return "".join(lines).rstrip("\n")


def _parse_line_count(raw: str) -> Optional[int]:
    try:
        value = int(raw)
    except ValueError:
        return None
    if value < 1:
        return None
    return min(value, _MAX_LOG_LINES)


def _parse_logs_args(raw_args: str) -> tuple[Optional[str], int, bool, Optional[str]]:
    try:
        tokens = shlex.split(raw_args)
    except ValueError as exc:
        return None, _DEFAULT_LOG_LINES, False, f"Invalid arguments: {exc}"

    server_name: Optional[str] = None
    follow = False
    line_count = _DEFAULT_LOG_LINES
    idx = 0

    while idx < len(tokens):
        token = tokens[idx]
        if token in {"-f", "--follow"}:
            follow = True
            idx += 1
            continue
        if token in {"-n", "--lines"}:
            if idx + 1 >= len(tokens):
                return None, line_count, follow, f"Missing value for {token}"
            parsed = _parse_line_count(tokens[idx + 1])
            if parsed is None:
                return None, line_count, follow, f"Invalid line count: {tokens[idx + 1]}"
            line_count = parsed
            idx += 2
            continue
        if token.startswith("--lines="):
            parsed = _parse_line_count(token.split("=", 1)[1])
            if parsed is None:
                return None, line_count, follow, f"Invalid line count: {token}"
            line_count = parsed
            idx += 1
            continue
        if token.startswith("-"):
            return None, line_count, follow, f"Unknown option: {token}"
        if server_name is not None:
            return None, line_count, follow, "Too many positional arguments"
        server_name = token
        idx += 1

    return server_name, line_count, follow, None


def _format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024.0:.1f} KB"
    return f"{size_bytes / (1024.0 * 1024.0):.1f} MB"


def _format_mtime(mtime: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))


def _iter_server_names(servers: Iterable[McpServerInfo]) -> list[str]:
    return [server.name for server in servers]


def _resolve_server_name(
    servers: list[McpServerInfo], requested: Optional[str]
) -> tuple[Optional[str], Optional[str]]:
    if requested:
        exact = next((s.name for s in servers if s.name == requested), None)
        if exact:
            return exact, None
        lowered = [s.name for s in servers if s.name.lower() == requested.lower()]
        if len(lowered) == 1:
            return lowered[0], None
        names = _iter_server_names(servers)
        if names:
            return None, f"Unknown MCP server '{requested}'. Available: {', '.join(names)}"
        return requested, None

    if len(servers) == 1:
        return servers[0].name, None
    if not servers:
        return None, "No MCP servers configured."
    return None, "Please specify a server. Use /mcp logs to list log targets."


def _show_log_targets(ui: Any, servers: list[McpServerInfo]) -> bool:
    mode = get_mcp_stderr_mode()
    ui.console.print("\n[bold]MCP stderr logs[/bold]")
    ui.console.print(f"[dim]stderr mode: {mode}[/dim]")
    if mode != "log":
        ui.console.print(
            "[yellow]MCP stderr is not in 'log' mode; log files may be missing. "
            "Set RIPPERDOC_MCP_STDERR_MODE=log to persist stderr.[/yellow]"
        )

    if not servers:
        ui.console.print("[yellow]No MCP servers configured.[/yellow]")
        _print_usage(ui)
        return True

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Server")
    table.add_column("Log status")
    table.add_column("Size", justify="right")
    table.add_column("Updated")
    table.add_column("Path")

    for server in servers:
        path = get_mcp_stderr_log_path(ui.project_path, server.name)
        if path.exists():
            try:
                stat = path.stat()
                size = _format_size(stat.st_size)
                updated = _format_mtime(stat.st_mtime)
                state = "ready" if stat.st_size > 0 else "empty"
            except OSError:
                size = "-"
                updated = "-"
                state = "unreadable"
        else:
            size = "-"
            updated = "-"
            state = "missing"

        table.add_row(server.name, state, size, updated, str(path))

    ui.console.print(table)
    ui.console.print(
        "[dim]Use /mcp logs <server> to view recent lines, or /mcp logs <server> -f to follow.[/dim]"
    )
    return True


def _show_log_snapshot(ui: Any, path: Path, server_name: str, line_count: int) -> bool:
    if not path.exists():
        ui.console.print(
            f"[yellow]No log file yet for MCP server '{escape(server_name)}'.[/yellow]"
        )
        ui.console.print(f"[dim]Expected path: {path}[/dim]")
        return True

    content = _tail_text(path, line_count)
    body = escape(content) if content else "[dim](log file is empty)[/dim]"
    ui.console.print(
        Panel(
            body,
            title=f"MCP stderr · {escape(server_name)} · last {line_count} lines",
            border_style="cyan",
        )
    )
    ui.console.print(f"[dim]Log file: {path}[/dim]")
    return True


def _follow_log(ui: Any, path: Path, server_name: str) -> bool:
    if not path.exists():
        ui.console.print(
            f"[yellow]No log file yet for MCP server '{escape(server_name)}'.[/yellow]"
        )
        ui.console.print(f"[dim]Expected path: {path}[/dim]")
        return True

    ui.console.print(
        f"[dim]Following MCP stderr for {escape(server_name)}. Press Ctrl+C to stop.[/dim]"
    )
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(0, os.SEEK_END)
            while True:
                line = handle.readline()
                if line:
                    ui.console.print(line.rstrip("\n"), markup=False)
                    continue

                time.sleep(_FOLLOW_POLL_SECONDS)
                try:
                    file_size = path.stat().st_size
                except OSError:
                    continue
                if file_size < handle.tell():
                    ui.console.print("[dim]Log file truncated; continuing from start.[/dim]")
                    handle.seek(0, os.SEEK_SET)
    except KeyboardInterrupt:
        ui.console.print("\n[dim]Stopped MCP log follow.[/dim]")
    return True


def _handle_logs(ui: Any, raw_args: str) -> bool:
    requested_server, line_count, follow, error = _parse_logs_args(raw_args)
    if error:
        ui.console.print(f"[red]{escape(error)}[/red]")
        _print_usage(ui)
        return True

    servers = _load_servers(ui)
    if requested_server is None and not follow:
        return _show_log_targets(ui, servers)

    resolved_name, resolve_error = _resolve_server_name(servers, requested_server)
    if resolve_error:
        ui.console.print(f"[red]{escape(resolve_error)}[/red]")
        if servers:
            _show_log_targets(ui, servers)
        else:
            _print_usage(ui)
        return True
    if resolved_name is None:
        return True

    log_path = get_mcp_stderr_log_path(ui.project_path, resolved_name)
    _show_log_snapshot(ui, log_path, resolved_name, line_count)
    if follow:
        return _follow_log(ui, log_path, resolved_name)
    return True


def _handle(ui: Any, arg: str) -> bool:
    trimmed = arg.strip()
    if not trimmed:
        return _handle_tui(ui)

    parts = trimmed.split(maxsplit=1)
    subcommand = parts[0].lower()
    sub_args = parts[1] if len(parts) > 1 else ""

    if subcommand in {"log", "logs"}:
        return _handle_logs(ui, sub_args)
    if subcommand in {"tui", "ui"}:
        return _handle_tui(ui)
    if subcommand in {"list", "ls"}:
        servers = _load_servers(ui)
        return _show_server_overview(ui, servers)

    if subcommand in {"help", "-h", "--help"}:
        _print_usage(ui)
        return True

    ui.console.print(
        "[red]Unknown subcommand. Use /mcp, /mcp tui, /mcp logs, or /mcp logs <server> -f.[/red]"
    )
    return True


command = SlashCommand(
    name="mcp",
    description="Show MCP servers and inspect MCP stderr logs",
    handler=_handle,
)


__all__ = ["command"]
