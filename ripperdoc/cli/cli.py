"""Main CLI entry point for Ripperdoc.

This module provides the command-line interface for the Ripperdoc agent.
"""

import asyncio
import click
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from ripperdoc import __version__
from ripperdoc.core.config import (
    get_global_config,
    save_global_config,
    get_project_config,
    ModelProfile,
    ProviderType,
)
from ripperdoc.core.default_tools import get_default_tools
from ripperdoc.core.query import query, QueryContext
from ripperdoc.core.system_prompt import build_system_prompt
from ripperdoc.utils.messages import create_user_message
from ripperdoc.utils.memory import build_memory_instructions
from ripperdoc.core.permissions import make_permission_checker
from ripperdoc.utils.mcp import (
    load_mcp_servers_async,
    format_mcp_instructions,
    shutdown_mcp_runtime,
)
from ripperdoc.tools.mcp_tools import load_dynamic_mcp_tools_async, merge_tools_with_dynamic
from ripperdoc.utils.log import enable_session_file_logging, get_logger
from ripperdoc.utils.prompt import prompt_secret

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.markup import escape

console = Console()
logger = get_logger()


async def run_query(
    prompt: str,
    tools: list,
    safe_mode: bool = False,
    verbose: bool = False,
    session_id: Optional[str] = None,
) -> None:
    """Run a single query and print the response."""

    logger.info(
        "[cli] Running single prompt session",
        extra={
            "safe_mode": safe_mode,
            "verbose": verbose,
            "session_id": session_id,
            "prompt_length": len(prompt),
        },
    )
    if prompt:
        logger.debug(
            "[cli] Prompt preview",
            extra={"session_id": session_id, "prompt_preview": prompt[:200]},
        )

    project_path = Path.cwd()
    can_use_tool = make_permission_checker(project_path, safe_mode) if safe_mode else None

    # Create initial user message
    from ripperdoc.utils.messages import UserMessage, AssistantMessage, ProgressMessage

    messages: List[UserMessage | AssistantMessage | ProgressMessage] = [create_user_message(prompt)]

    # Create query context
    query_context = QueryContext(tools=tools, safe_mode=safe_mode, verbose=verbose)

    try:
        context: Dict[str, Any] = {}
        # System prompt
        servers = await load_mcp_servers_async(Path.cwd())
        dynamic_tools = await load_dynamic_mcp_tools_async(Path.cwd())
        if dynamic_tools:
            tools = merge_tools_with_dynamic(tools, dynamic_tools)
            query_context.tools = tools
        mcp_instructions = format_mcp_instructions(servers)
        base_system_prompt = build_system_prompt(
            tools,
            prompt,
            context,
            mcp_instructions=mcp_instructions,
        )
        memory_instructions = build_memory_instructions()
        system_prompt = (
            f"{base_system_prompt}\n\n{memory_instructions}"
            if memory_instructions
            else base_system_prompt
        )

        # Run the query
        try:
            async for message in query(
                messages, system_prompt, context, query_context, can_use_tool
            ):
                if message.type == "assistant" and hasattr(message, "message"):
                    # Print assistant message
                    if isinstance(message.message.content, str):
                        console.print(
                            Panel(
                                Markdown(message.message.content),
                                title="Ripperdoc",
                                border_style="cyan",
                            )
                        )
                    else:
                        # Handle structured content
                        for block in message.message.content:
                            if isinstance(block, dict):
                                if block.get("type") == "text":
                                    console.print(
                                        Panel(
                                            Markdown(block["text"]),
                                            title="Ripperdoc",
                                            border_style="cyan",
                                        )
                                    )
                            else:
                                if hasattr(block, "type") and block.type == "text":
                                    console.print(
                                        Panel(
                                            Markdown(block.text or ""),
                                            title="Ripperdoc",
                                            border_style="cyan",
                                        )
                                    )

                elif message.type == "progress" and hasattr(message, "content"):
                    # Print progress
                    if verbose:
                        console.print(f"[dim]Progress: {escape(str(message.content))}[/dim]")

                # Add message to history
                messages.append(message)  # type: ignore[arg-type]

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user[/yellow]")
        except Exception as e:
            console.print(f"[red]Error: {escape(str(e))}[/red]")
            logger.exception(
                "[cli] Unhandled error while running prompt", extra={"session_id": session_id}
            )
            if verbose:
                import traceback

                console.print(traceback.format_exc(), markup=False)
        logger.info(
            "[cli] Prompt session completed",
            extra={"session_id": session_id, "message_count": len(messages)},
        )
    finally:
        await shutdown_mcp_runtime()
        logger.debug("[cli] Shutdown MCP runtime", extra={"session_id": session_id})


def check_onboarding() -> bool:
    """Check if onboarding is complete and run if needed."""
    config = get_global_config()

    if config.has_completed_onboarding:
        return True

    console.print("[bold cyan]Welcome to Ripperdoc![/bold cyan]\n")
    console.print("Let's set up your AI model configuration.\n")

    # Simple onboarding
    provider_choices = [
        *[p.value for p in ProviderType],
        "openai",
        "deepseek",
        "mistral",
        "kimi",
        "qwen",
        "glm",
        "custom",
    ]
    provider_choice = click.prompt(
        "Choose your model protocol",
        type=click.Choice(provider_choices),
        default=ProviderType.ANTHROPIC.value,
    )

    api_base = None
    if provider_choice == "custom":
        provider_choice = click.prompt(
            "Protocol family (for API compatibility)",
            type=click.Choice([p.value for p in ProviderType]),
            default=ProviderType.OPENAI_COMPATIBLE.value,
        )
        api_base = click.prompt("API Base URL")

    api_key = ""
    while not api_key:
        api_key = prompt_secret("Enter your API key").strip()
        if not api_key:
            console.print("[red]API key is required.[/red]")

    provider = ProviderType(provider_choice)

    # Get model name
    if provider == ProviderType.ANTHROPIC:
        model = click.prompt("Model name", default="claude-3-5-sonnet-20241022")
    elif provider == ProviderType.OPENAI_COMPATIBLE:
        default_model = "gpt-4o-mini"
        if provider_choice == "deepseek":
            default_model = "deepseek-chat"
            api_base = api_base or "https://api.deepseek.com"
        model = click.prompt("Model name", default=default_model)
        if api_base is None:
            api_base = (
                click.prompt("API base URL (optional)", default="", show_default=False) or None
            )
    elif provider == ProviderType.GEMINI:
        console.print(
            "[yellow]Gemini protocol support is not yet available; configuration is saved for "
            "future support.[/yellow]"
        )
        model = click.prompt("Model name", default="gemini-1.5-pro")
        if api_base is None:
            api_base = (
                click.prompt("API base URL (optional)", default="", show_default=False) or None
            )
    else:
        model = click.prompt("Model name")

    context_window_input = click.prompt(
        "Context window in tokens (optional, press Enter to skip)", default="", show_default=False
    )
    context_window = None
    if context_window_input.strip():
        try:
            context_window = int(context_window_input.strip())
        except ValueError:
            console.print("[yellow]Invalid context window, using auto-detected defaults.[/yellow]")

    # Create model profile
    config.model_profiles["default"] = ModelProfile(
        provider=provider,
        model=model,
        api_key=api_key,
        api_base=api_base,
        context_window=context_window,
    )

    config.has_completed_onboarding = True
    config.last_onboarding_version = __version__

    save_global_config(config)

    console.print("\n[green]✓ Configuration saved![/green]\n")

    return True


@click.group(invoke_without_command=True)
@click.version_option(version=__version__)
@click.option("--cwd", type=click.Path(exists=True), help="Working directory")
@click.option(
    "--unsafe",
    is_flag=True,
    help="Disable safe mode (skip permission prompts for tools)",
)
@click.option("--verbose", is_flag=True, help="Verbose output")
@click.option("-p", "--prompt", type=str, help="Direct prompt (non-interactive)")
@click.pass_context
def cli(
    ctx: click.Context, cwd: Optional[str], unsafe: bool, verbose: bool, prompt: Optional[str]
) -> None:
    """Ripperdoc - AI-powered coding agent"""
    session_id = str(uuid.uuid4())

    # Set working directory
    if cwd:
        import os

        os.chdir(cwd)
        logger.debug(
            "[cli] Changed working directory via --cwd",
            extra={"cwd": cwd, "session_id": session_id},
        )

    project_path = Path.cwd()
    log_file = enable_session_file_logging(project_path, session_id)
    logger.info(
        "[cli] Starting CLI invocation",
        extra={
            "session_id": session_id,
            "project_path": str(project_path),
            "log_file": str(log_file),
            "prompt_mode": bool(prompt),
        },
    )

    # Ensure onboarding is complete
    if not check_onboarding():
        logger.info(
            "[cli] Onboarding check failed or aborted; exiting.",
            extra={"session_id": session_id},
        )
        sys.exit(1)

    # Initialize project configuration for the current working directory
    get_project_config(project_path)

    safe_mode = not unsafe
    logger.debug(
        "[cli] Configuration initialized",
        extra={"session_id": session_id, "safe_mode": safe_mode, "verbose": verbose},
    )

    # If prompt is provided, run directly
    if prompt:
        tools = get_default_tools()
        asyncio.run(run_query(prompt, tools, safe_mode, verbose, session_id=session_id))
        return

    # If no command specified, start interactive REPL with Rich interface
    if ctx.invoked_subcommand is None:
        # Use Rich interface by default
        from ripperdoc.cli.ui.rich_ui import main_rich

        main_rich(
            safe_mode=safe_mode,
            verbose=verbose,
            session_id=session_id,
            log_file_path=log_file,
        )
        return


@cli.command(name="config")
def config_cmd() -> None:
    """Show current configuration"""
    config = get_global_config()

    console.print("\n[bold]Global Configuration[/bold]\n")
    console.print(f"Version: {__version__}")
    console.print(f"Onboarding Complete: {config.has_completed_onboarding}")
    console.print(f"Theme: {config.theme}")
    console.print(f"Verbose: {config.verbose}")
    console.print(f"Safe Mode: {config.safe_mode}\n")

    if config.model_profiles:
        console.print("[bold]Model Profiles:[/bold]")
        for name, profile in config.model_profiles.items():
            console.print(f"  {name}:")
            console.print(f"    Provider: {profile.provider}")
            console.print(f"    Model: {profile.model}")
            console.print(f"    API Key: {'***' if profile.api_key else 'Not set'}")
        console.print()


@cli.command(name="version")
def version_cmd() -> None:
    """Show version information"""
    console.print(f"Ripperdoc version {__version__}")


def main() -> None:
    """Main entry point."""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"[red]Fatal error: {escape(str(e))}[/red]")
        logger.exception("[cli] Fatal error in main CLI entrypoint")
        sys.exit(1)


if __name__ == "__main__":
    main()
