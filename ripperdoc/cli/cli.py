"""Main CLI entry point for Ripperdoc."""

from __future__ import annotations

import asyncio
import click
import sys
import uuid
from pathlib import Path
from typing import Optional, cast

from rich.console import Console
from rich.markup import escape

from ripperdoc import __version__
from ripperdoc.cli.agents_cli import agents_cmd
from ripperdoc.cli.bootstrap_cli import (
    _change_cwd_if_requested,
    _log_resume_state,
    _prepare_cli_runtime_inputs,
    _read_initial_query_from_stdin,
    _resolve_permission_mode,
    _resolve_resume_state,
    _resolve_root_extra_args,
    _resolve_setup_trigger,
    _run_setup_if_needed,
    _run_stdio_mode_if_requested,
    resolve_debug_filter_from_extra_args,
    parse_tools_option,
)
from ripperdoc.cli.mcp_cli import mcp_group
from ripperdoc.cli.runtime_cli import run_query
from ripperdoc.cli.top_level_cli import config_cmd, update_cmd, version_cmd
from ripperdoc.cli.ui.wizard import check_onboarding
from ripperdoc.core.config import get_effective_model_profile, get_project_config
from ripperdoc.core.plugins import set_runtime_plugin_dirs
from ripperdoc.core.tool_defaults import get_default_tools
from ripperdoc.utils.log import configure_debug_logging, enable_session_file_logging, get_logger
from ripperdoc.utils.tasks import set_runtime_task_scope

console = Console()
logger = get_logger()


def _resolve_model_pointer_with_fallback(
    model: Optional[str],
    fallback_model: Optional[str],
    *,
    session_id: Optional[str],
    route: str,
) -> str:
    """Resolve model pointer with optional fallback when primary profile is missing."""
    resolved_model = model or "main"
    resolved_profile = get_effective_model_profile(resolved_model)
    fallback_profile = get_effective_model_profile(fallback_model) if fallback_model else None
    if resolved_profile is None and fallback_profile is not None:
        logger.warning(
            "[cli] Falling back to secondary model",
            extra={
                "session_id": session_id,
                "route": route,
                "requested_model": resolved_model,
                "fallback_model": fallback_model,
            },
        )
        return cast(str, fallback_model)
    return resolved_model


def _merge_append_system_prompt(
    append_system_prompt: Optional[str],
    session_agent_prompt: Optional[str],
) -> Optional[str]:
    """Compose agent prompt and append-system-prompt with deterministic order."""
    if not session_agent_prompt:
        return append_system_prompt
    if not append_system_prompt:
        return session_agent_prompt
    return f"{session_agent_prompt}\n\n{append_system_prompt}"


@click.group(
    invoke_without_command=True,
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.version_option(
    __version__,
    "--version",
    "-v",
    message="%(version)s (Ripperdoc)",
)
@click.option("--cwd", type=click.Path(exists=True), help="Working directory")
@click.option(
    "-d",
    "--debug",
    "debug_mode",
    is_flag=True,
    help=(
        'Enable debug mode with optional category filtering '
        '(e.g., "api,hooks" or "!1p,!file").'
    ),
)
@click.option(
    "--debug-filter",
    type=str,
    default=None,
    help="Debug filter categories (internal helper for --debug [filter]).",
    hidden=True,
)
@click.option(
    "--debug-file",
    type=click.Path(),
    default=None,
    help="Write debug logs to a specific file path (implicitly enables debug mode).",
)
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
@click.option(
    "--input-format",
    type=click.Choice(["stream-json", "auto"]),
    default="stream-json",
    help="Input format for messages.",
)
@click.option("-p", "--prompt", type=str, help="Direct prompt (non-interactive)")
@click.option(
    "--output-format",
    type=click.Choice(["text", "json", "stream-json"]),
    default="text",
    help=(
        'Output format (SDK/--print): "text" (default), "json" '
        '(single result), or "stream-json" (realtime streaming)'
    ),
)
@click.option(
    "--print",
    "print_mode",
    is_flag=True,
    help="Print mode (for single prompt queries).",
)
@click.option(
    "--",
    "print_prompt",
    type=str,
    default=None,
    help="Direct prompt (for print mode).",
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
    "--allowedTools",
    "allowed_tools_csv",
    type=str,
    default=None,
    help="Allowed tool list (comma-separated, SDK only).",
    hidden=True,
)
@click.option(
    "--disallowedTools",
    "disallowed_tools_csv",
    type=str,
    default=None,
    help="Disallowed tool list (comma-separated, SDK only).",
    hidden=True,
)
@click.option(
    "--permission-prompt-tool",
    type=str,
    default=None,
    help="Permission prompt tool name (SDK only).",
    hidden=True,
)
@click.option(
    "--settings",
    type=str,
    default=None,
    help="Settings JSON or path (SDK only).",
    hidden=True,
)
@click.option(
    "--add-dir",
    "add_dirs",
    multiple=True,
    type=click.Path(),
    help="Additional working directory (repeatable).",
)
@click.option(
    "--mcp-config",
    type=str,
    default=None,
    help="MCP config JSON or path (SDK only).",
    hidden=True,
)
@click.option(
    "--include-partial-messages",
    is_flag=True,
    help="Include partial messages (SDK only).",
    hidden=True,
)
@click.option(
    "--fork-session",
    is_flag=True,
    help="Fork session on resume (SDK only).",
    hidden=True,
)
@click.option(
    "--agent",
    type=str,
    default=None,
    help="Agent for the current session. Overrides the 'agent' setting.",
)
@click.option(
    "--agents",
    type=str,
    default=None,
    help=(
        "JSON object defining custom agents "
        '(e.g. \'{"reviewer":{"description":"Reviews code","prompt":"You are a code reviewer"}}\').'
    ),
)
@click.option(
    "--setting-sources",
    type=str,
    default=None,
    help="Setting sources (SDK only).",
    hidden=True,
)
@click.option(
    "--disable-slash-commands",
    is_flag=True,
    help="Disable all skills.",
)
@click.option(
    "--plugin-dir",
    "plugin_dirs",
    multiple=True,
    type=click.Path(),
    help="Additional plugin directory (repeatable).",
)
@click.option(
    "--betas",
    type=str,
    default=None,
    help="SDK beta flags (SDK only).",
    hidden=True,
)
@click.option(
    "--fallback-model",
    type=str,
    default=None,
    help="Fallback model (SDK only).",
    hidden=True,
)
@click.option(
    "--max-budget-usd",
    type=float,
    default=None,
    help="Max budget (SDK only).",
    hidden=True,
)
@click.option(
    "--max-thinking-tokens",
    type=int,
    default=None,
    help="Max thinking tokens (SDK only).",
    hidden=True,
)
@click.option(
    "--json-schema",
    type=str,
    default=None,
    help="JSON schema for output (SDK only).",
    hidden=True,
)
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
@click.option(
    "-r",
    "--resume",
    "resume_session",
    type=str,
    default=None,
    help="Resume a specific session by id prefix.",
)
@click.option(
    "--init",
    "setup_init",
    is_flag=True,
    help="Run initialization hooks and start interactive mode.",
)
@click.option(
    "--init-only",
    "setup_init_only",
    is_flag=True,
    help="Run initialization hooks and exit (no interactive session).",
)
@click.option(
    "--maintenance",
    "setup_maintenance",
    is_flag=True,
    help="Run maintenance hooks and exit.",
)
@click.pass_context
def cli(
    ctx: click.Context,
    cwd: Optional[str],
    debug_mode: bool,
    debug_filter: Optional[str],
    debug_file: Optional[str],
    yolo: bool,
    verbose: bool,
    show_full_thinking: Optional[bool],
    input_format: str,
    prompt: Optional[str],
    output_format: str,
    print_mode: bool,
    print_prompt: Optional[str],
    permission_mode: str,
    max_turns: Optional[int],
    allowed_tools_csv: Optional[str],
    disallowed_tools_csv: Optional[str],
    permission_prompt_tool: Optional[str],
    settings: Optional[str],
    add_dirs: tuple[str, ...],
    mcp_config: Optional[str],
    include_partial_messages: bool,
    fork_session: bool,
    agent: Optional[str],
    agents: Optional[str],
    setting_sources: Optional[str],
    disable_slash_commands: bool,
    plugin_dirs: tuple[str, ...],
    betas: Optional[str],
    fallback_model: Optional[str],
    max_budget_usd: Optional[float],
    max_thinking_tokens: Optional[int],
    json_schema: Optional[str],
    tools: Optional[str],
    system_prompt: Optional[str],
    append_system_prompt: Optional[str],
    model: Optional[str],
    continue_session: bool,
    resume_session: Optional[str],
    setup_init: bool,
    setup_init_only: bool,
    setup_maintenance: bool,
) -> None:
    """Ripperdoc - AI-powered coding agent"""
    session_id = str(uuid.uuid4())
    debug_filter = resolve_debug_filter_from_extra_args(
        ctx=ctx,
        debug_enabled=debug_mode or bool(debug_file),
        explicit_debug_filter=debug_filter,
        print_mode=print_mode,
        print_prompt=print_prompt,
        prompt=prompt,
    )
    effective_permission_mode = _resolve_permission_mode(yolo, permission_mode)
    print_prompt = _resolve_root_extra_args(
        ctx=ctx, print_mode=print_mode, print_prompt=print_prompt
    )
    cwd_changed = _change_cwd_if_requested(cwd)
    debug_log_path = configure_debug_logging(
        enabled=debug_mode or bool(debug_file),
        filter_spec=debug_filter,
        debug_file=Path(debug_file) if debug_file else None,
    )

    (
        project_path,
        additional_working_dirs,
        sdk_default_options,
        session_agent_prompt,
    ) = _prepare_cli_runtime_inputs(
        ctx=ctx,
        output_format=output_format,
        print_mode=print_mode,
        settings=settings,
        add_dirs=add_dirs,
        agent=agent,
        agents=agents,
        disable_slash_commands=disable_slash_commands,
        allowed_tools_csv=allowed_tools_csv,
        disallowed_tools_csv=disallowed_tools_csv,
        permission_prompt_tool=permission_prompt_tool,
        mcp_config=mcp_config,
        include_partial_messages=include_partial_messages,
        fork_session=fork_session,
        setting_sources=setting_sources,
        plugin_dirs=plugin_dirs,
        betas=betas,
        fallback_model=fallback_model,
        max_budget_usd=max_budget_usd,
        max_thinking_tokens=max_thinking_tokens,
        json_schema=json_schema,
    )
    effective_append_system_prompt = _merge_append_system_prompt(
        append_system_prompt,
        session_agent_prompt,
    )
    set_runtime_plugin_dirs(plugin_dirs, base_dir=project_path)

    if _run_stdio_mode_if_requested(
        ctx=ctx,
        output_format=output_format,
        print_mode=print_mode,
        print_prompt=print_prompt,
        prompt=prompt,
        input_format=input_format,
        model=model,
        permission_mode=effective_permission_mode,
        max_turns=max_turns,
        system_prompt=system_prompt,
        verbose=verbose,
        allowed_tools_csv=allowed_tools_csv,
        disallowed_tools_csv=disallowed_tools_csv,
        tools=tools,
        project_path=project_path,
        additional_working_dirs=additional_working_dirs,
        sdk_default_options=sdk_default_options,
    ):
        return

    session_id, resume_messages, resumed_summary, most_recent = _resolve_resume_state(
        project_path=project_path,
        session_id=session_id,
        resume_session=resume_session,
        continue_session=continue_session,
    )
    set_runtime_task_scope(session_id=session_id, project_root=project_path)
    try:
        if debug_file and debug_log_path is not None:
            log_file = debug_log_path
        else:
            log_file = enable_session_file_logging(project_path, session_id)

        if cwd_changed:
            logger.debug(
                "[cli] Changed working directory via --cwd",
                extra={"cwd": cwd_changed, "session_id": session_id},
            )

        _log_resume_state(
            session_id=session_id,
            log_file=log_file,
            resume_messages=resume_messages,
            resumed_summary=resumed_summary,
            most_recent=most_recent,
            continue_session=continue_session,
        )

        logger.info(
            "[cli] Starting CLI invocation",
            extra={
                "session_id": session_id,
                "project_path": str(project_path),
                "log_file": str(log_file),
                "prompt_mode": bool(prompt),
            },
        )

        skip_onboarding = ctx.invoked_subcommand in ("update", "upgrade", "mcp", "agents")
        if not skip_onboarding:
            if not check_onboarding():
                logger.info(
                    "[cli] Onboarding check failed or aborted; exiting.",
                    extra={"session_id": session_id},
                )
                sys.exit(1)

        get_project_config(project_path)

        yolo_mode = effective_permission_mode == "bypassPermissions"
        allowed_tools = parse_tools_option(tools)

        setup_trigger = _resolve_setup_trigger(
            setup_init=setup_init,
            setup_init_only=setup_init_only,
            setup_maintenance=setup_maintenance,
        )
        if _run_setup_if_needed(
            setup_trigger=setup_trigger,
            project_path=project_path,
            session_id=session_id,
            setup_init_only=setup_init_only,
            setup_maintenance=setup_maintenance,
        ):
            return

        initial_query = _read_initial_query_from_stdin(
            prompt=prompt,
            invoked_subcommand=ctx.invoked_subcommand,
            session_id=session_id,
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
                "has_append_system_prompt": effective_append_system_prompt is not None,
                "disable_slash_commands": disable_slash_commands,
                "debug_mode": bool(debug_mode or debug_file),
                "debug_filter": debug_filter,
                "debug_file": debug_file,
                "continue_session": continue_session,
            },
        )

        if prompt:
            tool_list = get_default_tools(allowed_tools=allowed_tools)
            if disable_slash_commands:
                tool_list = [t for t in tool_list if getattr(t, "name", None) != "Skill"]
            asyncio.run(
                run_query(
                    prompt,
                    tool_list,
                    yolo_mode,
                    verbose,
                    session_id=session_id,
                    custom_system_prompt=system_prompt,
                    append_system_prompt=effective_append_system_prompt,
                    model=model,
                    fallback_model=fallback_model,
                    max_thinking_tokens=max_thinking_tokens,
                    max_turns=max_turns,
                    permission_mode=effective_permission_mode,
                    allowed_tools=allowed_tools,
                    additional_working_dirs=additional_working_dirs,
                    disable_skills=disable_slash_commands,
                )
            )
            return

        if ctx.invoked_subcommand is None:
            from ripperdoc.cli.ui.rich_ui import main_rich

            interactive_model = _resolve_model_pointer_with_fallback(
                model,
                fallback_model,
                session_id=session_id,
                route="interactive",
            )
            main_rich(
                yolo_mode=yolo_mode,
                verbose=verbose,
                show_full_thinking=show_full_thinking,
                session_id=session_id,
                log_file_path=log_file,
                allowed_tools=allowed_tools,
                custom_system_prompt=system_prompt,
                append_system_prompt=effective_append_system_prompt,
                model=interactive_model,
                max_thinking_tokens=max_thinking_tokens,
                max_turns=max_turns,
                permission_mode=effective_permission_mode,
                resume_messages=resume_messages,
                initial_query=initial_query,
                additional_working_dirs=additional_working_dirs,
                disable_slash_commands=disable_slash_commands,
            )
            return
    finally:
        set_runtime_task_scope(session_id=None)


cli.add_command(config_cmd, name="config")
cli.add_command(update_cmd, name="update")
cli.add_command(update_cmd, name="upgrade")
cli.add_command(mcp_group, name="mcp")
cli.add_command(agents_cmd, name="agents")
cli.add_command(version_cmd, name="version")


def main() -> None:
    """Main entry point."""
    try:
        argv = sys.argv[1:]
        top_level_commands = {"config", "update", "upgrade", "mcp", "agents", "version"}
        if "--print" in argv and "--" in argv:
            sep_index = argv.index("--")
            prompt = " ".join(argv[sep_index + 1 :]).strip()
            argv = argv[:sep_index] + ["--prompt", prompt]
        rewritten_argv: list[str] = []
        index = 0
        while index < len(argv):
            token = argv[index]
            rewritten_argv.append(token)
            if token in ("-d", "--debug"):
                next_index = index + 1
                if next_index < len(argv):
                    candidate = argv[next_index].strip()
                    if (
                        candidate
                        and not candidate.startswith("-")
                        and candidate not in top_level_commands
                        and "--debug-filter" not in argv
                    ):
                        rewritten_argv.extend(["--debug-filter", candidate])
                        index += 1
            index += 1
        argv = rewritten_argv
        cli.main(args=argv, prog_name="ripperdoc")
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
