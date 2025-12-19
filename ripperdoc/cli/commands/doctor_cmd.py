"""Slash command to diagnose common setup issues."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from ripperdoc.core.config import (
    ProviderType,
    api_key_env_candidates,
    get_global_config,
    get_project_config,
)
from ripperdoc.cli.ui.helpers import get_profile_for_pointer
from ripperdoc.utils.log import get_logger
from ripperdoc.utils.mcp import load_mcp_servers_async
from ripperdoc.utils.sandbox_utils import is_sandbox_available

from .base import SlashCommand

logger = get_logger()


def _status_row(label: str, status: str, detail: str = "") -> Tuple[str, str, str]:
    """Build a (label, status, detail) tuple with icon."""
    icons = {
        "ok": "[green]✓[/green]",
        "warn": "[yellow]![/yellow]",
        "error": "[red]×[/red]",
    }
    icon = icons.get(status, "[yellow]?[/yellow]")
    return (label, icon, detail)


def _api_key_status(provider: ProviderType, profile_key: Optional[str]) -> Tuple[str, str]:
    """Check API key presence and source."""
    import os

    for env_var in api_key_env_candidates(provider):
        if os.environ.get(env_var):
            masked = os.environ[env_var]
            masked = masked[:4] + "…" if len(masked) > 4 else "set"
            return ("ok", f"Found in ${env_var} ({masked})")

    if profile_key:
        return ("ok", "Stored in config profile")

    return ("error", "Missing API key for active provider; set $ENV or edit config")


def _model_status(project_path: Path) -> List[Tuple[str, str, str]]:
    config = get_global_config()
    pointer = getattr(config.model_pointers, "main", "default")
    profile = get_profile_for_pointer("main")
    rows: List[Tuple[str, str, str]] = []

    if not profile:
        rows.append(
            _status_row("Model profile", "error", "No profile configured for pointer 'main'")
        )
        return rows

    if pointer not in config.model_profiles:
        rows.append(
            _status_row(
                "Model pointer",
                "warn",
                f"Pointer 'main' targets '{pointer}' which is missing; using fallback.",
            )
        )
    rows.append(
        _status_row(
            "Model",
            "ok",
            f"{profile.model} ({profile.provider.value})",
        )
    )

    key_status, key_detail = _api_key_status(profile.provider, profile.api_key)
    rows.append(_status_row("API key", key_status, key_detail))
    return rows


def _onboarding_status() -> Tuple[str, str, str]:
    config = get_global_config()
    if config.has_completed_onboarding:
        return _status_row(
            "Onboarding",
            "ok",
            f"Completed (version {str(config.last_onboarding_version or 'unknown')})",
        )
    return _status_row(
        "Onboarding",
        "warn",
        "Not completed; run the CLI without flags to configure provider/model.",
    )


def _sandbox_status() -> Tuple[str, str, str]:
    available = is_sandbox_available()
    if available:
        return _status_row("Sandbox", "ok", "'srt' runtime is available")
    return _status_row("Sandbox", "warn", "Sandbox runtime not detected; commands run normally")


def _mcp_status(
    project_path: Path, runner: Optional[Callable[[Any], Any]] = None
) -> Tuple[List[Tuple[str, str, str]], List[str]]:
    """Return MCP status rows and errors."""
    rows: List[Tuple[str, str, str]] = []
    errors: List[str] = []

    async def _load() -> List[Any]:
        return await load_mcp_servers_async(project_path)

    try:
        if runner is None:
            import asyncio

            servers = asyncio.run(_load())
        else:
            servers = runner(_load())
    except (
        OSError,
        RuntimeError,
        ConnectionError,
        ValueError,
        TypeError,
    ) as exc:  # pragma: no cover - defensive
        logger.warning(
            "[doctor] Failed to load MCP servers: %s: %s",
            type(exc).__name__,
            exc,
            exc_info=exc,
        )
        rows.append(_status_row("MCP", "error", f"Failed to load MCP config: {exc}"))
        return rows, errors

    if not servers:
        rows.append(_status_row("MCP", "warn", "No MCP servers configured (.mcp.json)"))
        return rows, errors

    failing = [s for s in servers if getattr(s, "error", None)]
    rows.append(
        _status_row(
            "MCP",
            "ok" if not failing else "warn",
            f"{len(servers)} configured; {len(failing)} with errors",
        )
    )
    for server in failing[:5]:
        errors.append(f"{server.name}: {server.error}")
    if len(failing) > 5:
        errors.append(f"... {len(failing) - 5} more")
    return rows, errors


def _project_status(project_path: Path) -> Tuple[str, str, str]:
    try:
        config = get_project_config(project_path)
        # Access a field to ensure model parsing does not throw.
        _ = len(config.allowed_tools)
        return _status_row(
            "Project config", "ok", f".ripperdoc/config.json loaded for {project_path}"
        )
    except (
        OSError,
        IOError,
        json.JSONDecodeError,
        ValueError,
        TypeError,
    ) as exc:  # pragma: no cover - defensive
        logger.warning(
            "[doctor] Failed to load project config: %s: %s",
            type(exc).__name__,
            exc,
            exc_info=exc,
        )
        return _status_row(
            "Project config", "warn", f"Could not read .ripperdoc/config.json: {exc}"
        )


def _render_table(console: Any, rows: List[Tuple[str, str, str]]) -> None:
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Check")
    table.add_column("")
    table.add_column("Details")
    for label, status, detail in rows:
        table.add_row(label, status, escape(detail) if detail else "")
    console.print(table)


def _handle(ui: Any, _: str) -> bool:
    project_path = getattr(ui, "project_path", Path.cwd())
    results: List[Tuple[str, str, str]] = []

    results.append(_onboarding_status())
    results.extend(_model_status(project_path))
    project_row = _project_status(project_path)
    results.append(project_row)

    runner = getattr(ui, "run_async", None)
    mcp_rows, mcp_errors = _mcp_status(project_path, runner=runner)
    results.extend(mcp_rows)
    results.append(_sandbox_status())

    ui.console.print(Panel("Environment diagnostics", title="/doctor", border_style="cyan"))
    _render_table(ui.console, results)

    if mcp_errors:
        ui.console.print("\n[bold]MCP issues:[/bold]")
        for err in mcp_errors:
            ui.console.print(f"  • {escape(err)}")

    ui.console.print(
        "\n[dim]If a check is failing, run `ripperdoc` without flags "
        "to rerun onboarding or update ~/.ripperdoc.json[/dim]"
    )
    return True


command = SlashCommand(
    name="doctor",
    description="Diagnose model config, API keys, MCP, and sandbox support",
    handler=_handle,
)


__all__ = ["command"]
