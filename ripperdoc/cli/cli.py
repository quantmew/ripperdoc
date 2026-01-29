"""Main CLI entry point for Ripperdoc.

This module provides the command-line interface for the Ripperdoc agent.
"""

import asyncio
import click
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from ripperdoc import __version__
from ripperdoc.core.config import (
    get_global_config,
    get_project_config,
)
from ripperdoc.cli.ui.wizard import check_onboarding
from ripperdoc.core.default_tools import get_default_tools, BUILTIN_TOOL_NAMES
from ripperdoc.core.query import query, QueryContext
from ripperdoc.core.system_prompt import build_system_prompt
from ripperdoc.core.skills import build_skill_summary, load_all_skills
from ripperdoc.core.hooks.manager import hook_manager
from ripperdoc.core.hooks.llm_callback import build_hook_llm_callback
from ripperdoc.utils.messages import create_user_message
from ripperdoc.utils.memory import build_memory_instructions
from ripperdoc.core.permissions import make_permission_checker
from ripperdoc.utils.session_history import (
    SessionHistory,
    list_session_summaries,
    load_session_messages,
)
from ripperdoc.utils.mcp import (
    load_mcp_servers_async,
    format_mcp_instructions,
    shutdown_mcp_runtime,
)
from ripperdoc.utils.lsp import shutdown_lsp_manager
from ripperdoc.tools.background_shell import shutdown_background_shell
from ripperdoc.tools.mcp_tools import load_dynamic_mcp_tools_async, merge_tools_with_dynamic
from ripperdoc.utils.log import enable_session_file_logging, get_logger


from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape

console = Console()
logger = get_logger()


def parse_tools_option(tools_arg: Optional[str]) -> Optional[List[str]]:
    """Parse the --tools argument.

    Args:
        tools_arg: The raw tools argument from CLI.

    Returns:
        None for default (all tools), empty list for "" (no tools),
        or a list of tool names for filtering.
    """
    if tools_arg is None:
        return None  # Use all default tools

    tools_arg = tools_arg.strip()

    if tools_arg == "":
        return []  # Disable all tools

    if tools_arg.lower() == "default":
        return None  # Use all default tools

    # Parse comma-separated list
    tool_names = [name.strip() for name in tools_arg.split(",") if name.strip()]

    # Validate tool names
    invalid_tools = [name for name in tool_names if name not in BUILTIN_TOOL_NAMES]
    if invalid_tools:
        logger.warning(
            "[cli] Unknown tools specified: %s. Available tools: %s",
            ", ".join(invalid_tools),
            ", ".join(BUILTIN_TOOL_NAMES),
        )

    return tool_names if tool_names else None


async def run_query(
    prompt: str,
    tools: list,
    yolo_mode: bool = False,
    verbose: bool = False,
    session_id: Optional[str] = None,
    custom_system_prompt: Optional[str] = None,
    append_system_prompt: Optional[str] = None,
    model: Optional[str] = None,
) -> None:
    """Run a single query and print the response."""
    logger.info(
        "[cli] Running single prompt session",
        extra={
            "yolo_mode": yolo_mode,
            "verbose": verbose,
            "session_id": session_id,
            "prompt_length": len(prompt),
            "model": model,
            "has_custom_system_prompt": custom_system_prompt is not None,
            "has_append_system_prompt": append_system_prompt is not None,
        },
    )
    if prompt:
        logger.debug(
            "[cli] Prompt preview",
            extra={"session_id": session_id, "prompt_preview": prompt[:200]},
        )

    project_path = Path.cwd()
    can_use_tool = None if yolo_mode else make_permission_checker(project_path, yolo_mode=False)

    # Initialize hook manager
    hook_manager.set_project_dir(project_path)
    hook_manager.set_session_id(session_id)
    hook_manager.set_llm_callback(build_hook_llm_callback())
    session_history = SessionHistory(project_path, session_id or str(uuid.uuid4()))
    hook_manager.set_transcript_path(str(session_history.path))

    def _collect_hook_contexts(result: Any) -> List[str]:
        contexts: List[str] = []
        system_message = getattr(result, "system_message", None)
        additional_context = getattr(result, "additional_context", None)
        if system_message:
            contexts.append(str(system_message))
        if additional_context:
            contexts.append(str(additional_context))
        return contexts

    # Create initial user message
    from ripperdoc.utils.messages import UserMessage, AssistantMessage, ProgressMessage

    messages: List[UserMessage | AssistantMessage | ProgressMessage] = [create_user_message(prompt)]
    session_history.append(messages[0])

    # Create query context
    query_context = QueryContext(
        tools=tools, yolo_mode=yolo_mode, verbose=verbose, model=model or "main"
    )

    session_start_time = time.time()
    try:
        context: Dict[str, Any] = {}
        # System prompt
        servers = await load_mcp_servers_async(Path.cwd())
        dynamic_tools = await load_dynamic_mcp_tools_async(Path.cwd())
        if dynamic_tools:
            tools = merge_tools_with_dynamic(tools, dynamic_tools)
            query_context.tools = tools
        mcp_instructions = format_mcp_instructions(servers)
        skill_result = load_all_skills(Path.cwd())
        for err in skill_result.errors:
            logger.warning(
                "[skills] Failed to load skill",
                extra={"path": str(err.path), "reason": err.reason},
            )
        skill_instructions = build_skill_summary(skill_result.skills)
        additional_instructions: List[str] = []
        if skill_instructions:
            additional_instructions.append(skill_instructions)
        memory_instructions = build_memory_instructions()
        if memory_instructions:
            additional_instructions.append(memory_instructions)

        session_start_result = await hook_manager.run_session_start_async("startup")
        session_hook_contexts = _collect_hook_contexts(session_start_result)
        if session_hook_contexts:
            additional_instructions.extend(session_hook_contexts)

        prompt_hook_result = await hook_manager.run_user_prompt_submit_async(prompt)
        if prompt_hook_result.should_block or not prompt_hook_result.should_continue:
            reason = (
                prompt_hook_result.block_reason
                or prompt_hook_result.stop_reason
                or "Prompt blocked by hook."
            )
            console.print(f"[red]{escape(str(reason))}[/red]")
            return
        prompt_hook_contexts = _collect_hook_contexts(prompt_hook_result)
        if prompt_hook_contexts:
            additional_instructions.extend(prompt_hook_contexts)

        # Build system prompt based on options:
        # - custom_system_prompt: replaces the default entirely
        # - append_system_prompt: appends to the default system prompt
        if custom_system_prompt:
            # Complete replacement
            system_prompt = custom_system_prompt
            # Still append if both are provided
            if append_system_prompt:
                system_prompt = f"{system_prompt}\n\n{append_system_prompt}"
        else:
            # Build default with optional append
            all_instructions = list(additional_instructions) if additional_instructions else []
            if append_system_prompt:
                all_instructions.append(append_system_prompt)
            system_prompt = build_system_prompt(
                tools,
                prompt,
                context,
                additional_instructions=all_instructions or None,
                mcp_instructions=mcp_instructions,
            )

        # Run the query - collect final response text
        final_response_parts: List[str] = []
        try:
            async for message in query(
                messages, system_prompt, context, query_context, can_use_tool
            ):
                if message.type == "assistant" and hasattr(message, "message"):
                    # Collect assistant message text for final output
                    if isinstance(message.message.content, str):
                        final_response_parts.append(message.message.content)
                    else:
                        # Handle structured content
                        for block in message.message.content:
                            if isinstance(block, dict):
                                if block.get("type") == "text":
                                    final_response_parts.append(block["text"])
                            else:
                                if hasattr(block, "type") and block.type == "text":
                                    final_response_parts.append(block.text or "")

                # Skip progress messages entirely for -p mode

                # Add message to history
                messages.append(message)  # type: ignore[arg-type]
                session_history.append(message)  # type: ignore[arg-type]

            # Print final response as clean markdown (no panel, no decoration)
            if final_response_parts:
                final_text = "\n".join(final_response_parts)
                console.print(Markdown(final_text))

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user[/yellow]")
        except asyncio.CancelledError:
            console.print("\n[yellow]Operation cancelled[/yellow]")
        except (RuntimeError, ValueError, TypeError, OSError, IOError, ConnectionError) as e:
            console.print(f"[red]Error: {escape(str(e))}[/red]")
            logger.warning(
                "[cli] Unhandled error while running prompt: %s: %s",
                type(e).__name__,
                e,
                extra={"session_id": session_id},
            )
            if verbose:
                import traceback

                console.print(traceback.format_exc(), markup=False)
        logger.info(
            "[cli] Prompt session completed",
            extra={"session_id": session_id, "message_count": len(messages)},
        )
    finally:
        duration = max(time.time() - session_start_time, 0.0)
        try:
            await hook_manager.run_session_end_async(
                "other", duration_seconds=duration, message_count=len(messages)
            )
        except (OSError, RuntimeError, ConnectionError, ValueError, TypeError) as exc:
            logger.warning(
                "[cli] SessionEnd hook failed: %s: %s",
                type(exc).__name__,
                exc,
                extra={"session_id": session_id},
            )
        await shutdown_mcp_runtime()
        await shutdown_lsp_manager()
        # Shutdown background shell manager
        try:
            shutdown_background_shell(force=True)
        except (OSError, RuntimeError) as exc:
            logger.debug(
                "[cli] Failed to shut down background shell: %s: %s",
                type(exc).__name__,
                exc,
            )
        logger.debug("[cli] Shutdown MCP runtime", extra={"session_id": session_id})


@click.group(invoke_without_command=True)
@click.version_option(version=__version__)
@click.option("--cwd", type=click.Path(exists=True), help="Working directory")
@click.option(
    "--yolo",
    is_flag=True,
    help="YOLO mode: skip all permission prompts for tools",
)
@click.option("--verbose", is_flag=True, help="Verbose output")
@click.option(
    "--show-full-thinking/--hide-full-thinking",
    default=None,
    help="Show full reasoning content instead of truncated preview",
)
@click.option("-p", "--prompt", type=str, help="Direct prompt (non-interactive)")
@click.option(
    "--tools",
    type=str,
    default=None,
    help=(
        'Specify the list of available tools. Use "" to disable all tools, '
        '"default" to use all tools, or specify tool names (e.g. "Bash,Edit,Read").'
    ),
)
@click.option(
    "--system-prompt",
    type=str,
    default=None,
    help="System prompt to use for the session (replaces default).",
)
@click.option(
    "--append-system-prompt",
    type=str,
    default=None,
    help="Additional instructions to append to the system prompt.",
)
@click.option(
    "--model",
    type=str,
    default=None,
    help="Model profile for the current session.",
)
@click.option(
    "-c",
    "--continue",
    "continue_session",
    is_flag=True,
    help="Continue the most recent conversation in the current directory.",
)
@click.pass_context
def cli(
    ctx: click.Context,
    cwd: Optional[str],
    yolo: bool,
    verbose: bool,
    show_full_thinking: Optional[bool],
    prompt: Optional[str],
    tools: Optional[str],
    system_prompt: Optional[str],
    append_system_prompt: Optional[str],
    model: Optional[str],
    continue_session: bool,
) -> None:
    """Ripperdoc - AI-powered coding agent"""
    session_id = str(uuid.uuid4())
    cwd_changed: Optional[str] = None

    # Set working directory
    if cwd:
        import os

        os.chdir(cwd)
        cwd_changed = cwd

    project_path = Path.cwd()

    # Handle --continue option: load the most recent session
    resume_messages = None
    most_recent = None
    if continue_session:
        summaries = list_session_summaries(project_path)
        if summaries:
            most_recent = summaries[0]
            session_id = most_recent.session_id
            resume_messages = load_session_messages(project_path, session_id)
            console.print(f"[dim]Continuing session: {most_recent.last_prompt}[/dim]")
        else:
            console.print("[yellow]No previous sessions found in this directory.[/yellow]")

    log_file = enable_session_file_logging(project_path, session_id)

    if cwd_changed:
        logger.debug(
            "[cli] Changed working directory via --cwd",
            extra={"cwd": cwd_changed, "session_id": session_id},
        )

    if most_recent:
        logger.info(
            "[cli] Continuing session",
            extra={
                "session_id": session_id,
                "message_count": len(resume_messages) if resume_messages else 0,
                "last_prompt": most_recent.last_prompt,
                "log_file": str(log_file),
            },
        )
    elif continue_session:
        logger.warning("[cli] No previous sessions found to continue")

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

    yolo_mode = yolo
    # Parse --tools option
    allowed_tools = parse_tools_option(tools)

    # Handle piped stdin input
    # - With -p flag: Not applicable (prompt comes from -p argument)
    # - Without -p: stdin becomes the initial query in interactive mode
    initial_query: Optional[str] = None
    if prompt is None and ctx.invoked_subcommand is None:
        stdin_stream = click.get_text_stream("stdin")
        try:
            stdin_is_tty = stdin_stream.isatty()
        except Exception:
            stdin_is_tty = True

        if not stdin_is_tty:
            try:
                stdin_data = stdin_stream.read()
            except (OSError, ValueError) as exc:
                logger.warning(
                    "[cli] Failed to read stdin for initial query: %s: %s",
                    type(exc).__name__,
                    exc,
                    extra={"session_id": session_id},
                )
            else:
                trimmed = stdin_data.rstrip("\n")
                if trimmed.strip():
                    initial_query = trimmed
                    logger.info(
                        "[cli] Received initial query from stdin",
                        extra={
                            "session_id": session_id,
                            "query_length": len(initial_query),
                            "query_preview": initial_query[:200],
                        },
                    )

    logger.debug(
        "[cli] Configuration initialized",
        extra={
            "session_id": session_id,
            "yolo_mode": yolo_mode,
            "verbose": verbose,
            "allowed_tools": allowed_tools,
            "model": model,
            "has_system_prompt": system_prompt is not None,
            "has_append_system_prompt": append_system_prompt is not None,
            "continue_session": continue_session,
        },
    )

    # If prompt is provided, run directly
    if prompt:
        tool_list = get_default_tools(allowed_tools=allowed_tools)
        asyncio.run(
            run_query(
                prompt,
                tool_list,
                yolo_mode,
                verbose,
                session_id=session_id,
                custom_system_prompt=system_prompt,
                append_system_prompt=append_system_prompt,
                model=model,
            )
        )
        return

    # If no command specified, start interactive REPL with Rich interface
    if ctx.invoked_subcommand is None:
        # Use Rich interface by default
        from ripperdoc.cli.ui.rich_ui import main_rich

        main_rich(
            yolo_mode=yolo_mode,
            verbose=verbose,
            show_full_thinking=show_full_thinking,
            session_id=session_id,
            log_file_path=log_file,
            allowed_tools=allowed_tools,
            custom_system_prompt=system_prompt,
            append_system_prompt=append_system_prompt,
            model=model,
            resume_messages=resume_messages,
            initial_query=initial_query,
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
    console.print(f"Yolo Mode: {config.yolo_mode}")
    console.print(f"Show Full Thinking: {config.show_full_thinking}\n")

    if config.model_profiles:
        console.print("[bold]Model Profiles:[/bold]")
        for name, profile in config.model_profiles.items():
            console.print(f"  {name}:")
            console.print(f"    Provider: {profile.provider}")
            console.print(f"    Model: {profile.model}")
            console.print(f"    API Key: {'***' if profile.api_key else 'Not set'}")
        console.print()


@cli.command(name="stdio")
@click.option(
    "--input-format",
    type=click.Choice(["stream-json", "auto"]),
    default="stream-json",
    help="Input format for messages.",
)
@click.option(
    "--output-format",
    type=click.Choice(["stream-json"]),
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
    type=click.Choice(["default", "acceptEdits", "plan", "bypassPermissions"]),
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
    Protocol over stdin/stdout.
    """
    from ripperdoc.protocol.stdio import _run_stdio
    import asyncio

    asyncio.run(
        _run_stdio(
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
    except SystemExit:
        raise
    except (
        RuntimeError,
        ValueError,
        TypeError,
        OSError,
        IOError,
        ConnectionError,
        click.ClickException,
    ) as e:
        console.print(f"[red]Fatal error: {escape(str(e))}[/red]")
        logger.warning(
            "[cli] Fatal error in main CLI entrypoint: %s: %s",
            type(e).__name__,
            e,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
