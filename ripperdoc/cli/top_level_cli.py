"""Top-level utility commands for `ripperdoc` CLI."""

from __future__ import annotations

import click
from rich.console import Console

from ripperdoc import __version__
from ripperdoc.core.config import get_global_config
from ripperdoc.utils.self_update import (
    get_install_metadata,
    get_latest_version,
    is_update_available,
    run_upgrade,
)

console = Console()


@click.command(name="config")
def config_cmd() -> None:
    """Show current configuration."""
    config = get_global_config()

    console.print("\n[bold]Global Configuration[/bold]\n")
    console.print(f"Version: {__version__}")
    console.print(f"Onboarding Complete: {config.has_completed_onboarding}")
    console.print(f"Theme: {config.theme}")
    console.print(f"Verbose: {config.verbose}")
    console.print(f"Yolo Mode: {config.yolo_mode}")
    console.print(f"Show Full Thinking: {config.show_full_thinking}\n")

    if config.model_profiles:
        console.print("[bold]Model Profiles:[/bold]")
        for name, profile in config.model_profiles.items():
            console.print(f"  {name}:")
            console.print(f"    Protocol: {profile.protocol}")
            console.print(f"    Model: {profile.model}")
            console.print(f"    API Key: {'***' if profile.api_key else 'Not set'}")
        console.print()


@click.command(name="update")
def update_cmd() -> None:
    """Check for updates and install if available."""
    metadata = get_install_metadata()
    latest_version = get_latest_version(metadata.method)
    if latest_version is None:
        raise click.ClickException(
            "Unable to check for updates. Check network access and retry."
        )

    if not is_update_available(__version__, latest_version):
        console.print(
            f"[green]You are already on the latest version: {__version__}[/green]"
        )
        console.print(
            f"[dim]Detected install method: {metadata.method_label} ({metadata.location})[/dim]"
        )
        return

    console.print(
        f"[yellow]Update available:[/yellow] "
        f"{__version__} -> {latest_version}"
    )
    success, exit_code, message = run_upgrade()
    if not success:
        raise click.ClickException(f"{message} (code: {exit_code})")

    console.print(
        "[green]Update completed.[/green] "
        f"{metadata.upgrade_hint}"
    )


@click.command(name="version")
def version_cmd() -> None:
    """Show version information."""
    console.print(f"Ripperdoc version {__version__}")


__all__ = ["config_cmd", "update_cmd", "version_cmd"]
