"""Main CLI entry point for Ripperdoc."""

from __future__ import annotations

import asyncio
import click
import os
import shutil
import sys
import uuid
import subprocess
from pathlib import Path
from typing import Optional, cast

from rich.console import Console
from rich.markup import escape

from ripperdoc import __version__
from ripperdoc.cli import worktree_tmux
from ripperdoc.cli.agents_cli import agents_cmd
from ripperdoc.cli.bootstrap_cli import (
    _change_cwd_if_requested,
    _is_non_interactive_mode,
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
from ripperdoc.cli.remote_control_cli import remote_control_cmd
from ripperdoc.cli.runtime_cli import run_query
from ripperdoc.cli.top_level_cli import config_cmd, update_cmd, version_cmd
from ripperdoc.cli.ui.wizard import check_onboarding
from ripperdoc.core.config import get_effective_model_profile, get_project_config
from ripperdoc.core.plugins import set_runtime_plugin_dirs
from ripperdoc.core.tool_defaults import get_default_tools
from ripperdoc.utils.filesystem.git_utils import get_git_root
from ripperdoc.utils.log import configure_debug_logging, enable_session_file_logging, get_logger
from ripperdoc.utils.sessions.session_history import SessionHistory
from ripperdoc.utils.collaboration.tasks import set_runtime_task_scope
from ripperdoc.utils.collaboration.worktree import (
    WorktreeSession,
    create_task_worktree,
    generate_cli_worktree_name,
    register_session_worktree,
)

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


def _coerce_session_id(
    session_id: Optional[str],
    *,
    sdk_url: Optional[str],
) -> Optional[str]:
    """Validate and normalize explicit session IDs from CLI input."""
    if session_id is None:
        return None
    if not session_id.strip():
        return None
    if sdk_url:
        return str(session_id)
    try:
        return str(uuid.UUID(session_id))
    except (ValueError, TypeError, AttributeError) as exc:
        raise click.ClickException("Error: --session-id must be a valid UUID.") from exc


def _ensure_session_id_available(*, session_id: str, project_path: Path) -> None:
    """Fail fast when an explicit session ID is already persisted on disk."""
    session_history = SessionHistory(project_path, session_id, session_persistence=False)
    if session_history.path.exists():
        raise click.ClickException(f"Error: Session ID {session_id} is already in use.")


_PR_WORKTREE_URL_RE = worktree_tmux._PR_WORKTREE_URL_RE
_PR_WORKTREE_HASH_RE = worktree_tmux._PR_WORKTREE_HASH_RE
_TMUX_PREFIX_ENV = worktree_tmux._TMUX_PREFIX_ENV
_PRECREATED_WORKTREE_ENV = worktree_tmux._PRECREATED_WORKTREE_ENV
_PRECREATED_WORKTREE_REPO_ENV = worktree_tmux._PRECREATED_WORKTREE_REPO_ENV
_PRECREATED_WORKTREE_NAME_ENV = worktree_tmux._PRECREATED_WORKTREE_NAME_ENV
_PRECREATED_WORKTREE_BRANCH_ENV = worktree_tmux._PRECREATED_WORKTREE_BRANCH_ENV
_PRECREATED_WORKTREE_HEAD_ENV = worktree_tmux._PRECREATED_WORKTREE_HEAD_ENV
_PRECREATED_WORKTREE_HOOK_ENV = worktree_tmux._PRECREATED_WORKTREE_HOOK_ENV

_extract_pr_number_from_worktree = worktree_tmux._extract_pr_number_from_worktree
_resolve_worktree_name_and_pr = worktree_tmux._resolve_worktree_name_and_pr
_has_tmux_worktree_flags = worktree_tmux._has_tmux_worktree_flags
_extract_worktree_arg = worktree_tmux._extract_worktree_arg
_extract_cwd_arg = worktree_tmux._extract_cwd_arg
_strip_tmux_worktree_args = worktree_tmux._strip_tmux_worktree_args
_strip_cwd_args = worktree_tmux._strip_cwd_args
_hash_repository_path = worktree_tmux._hash_repository_path
_build_tmux_session_name = worktree_tmux._build_tmux_session_name


def _run_tmux_command(
    args: list[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[dict[str, str]] = None,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    return worktree_tmux._run_tmux_command(args, cwd=cwd, env=env, capture=capture)


def _exec_into_tmux_worktree(argv: list[str]) -> tuple[bool, Optional[str]]:
    return worktree_tmux._exec_into_tmux_worktree(
        argv,
        get_git_root_fn=get_git_root,
        create_task_worktree_fn=create_task_worktree,
        generate_cli_worktree_name_fn=generate_cli_worktree_name,
        run_tmux_command_fn=_run_tmux_command,
        which_fn=shutil.which,
        environ=os.environ,
        platform_name=sys.platform,
        python_executable=sys.executable,
    )


def _register_precreated_worktree_from_env() -> Optional[WorktreeSession]:
    return worktree_tmux._register_precreated_worktree_from_env(
        environ=os.environ,
        register_session_worktree_fn=register_session_worktree,
    )


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
    "--replay-user-messages",
    is_flag=True,
    help=(
        "Re-emit user messages from stdin back on stdout for acknowledgment "
        "(only works with --input-format=stream-json and --output-format=stream-json)."
    ),
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
    "--session-id",
    type=str,
    default=None,
    help="Use a specific session ID for the conversation (must be a valid UUID).",
)
@click.option(
    "--resume-session-at",
    type=str,
    default=None,
    help="When resuming, only messages up to and including the message id.",
    hidden=True,
)
@click.option(
    "--rewind-files",
    type=str,
    default=None,
    help="Restore files to state at a specific user message and exit (requires --resume).",
    hidden=True,
)
@click.option(
    "--no-session-persistence",
    is_flag=True,
    default=False,
    help="Disable session persistence - sessions will not be saved to disk and cannot be resumed (only works with --print mode).",
)
@click.option(
    "--sdk-url",
    type=str,
    default=None,
    help="Use remote WebSocket endpoint for SDK I/O streaming (only with -p and stream-json).",
    hidden=True,
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
@click.option(
    "-w",
    "--worktree",
    "use_worktree",
    is_flag=True,
    help="Create and enter a temporary worktree for this session (optionally provide a name).",
)
@click.option(
    "--worktree-name",
    type=str,
    default=None,
    hidden=True,
)
@click.option(
    "--tmux",
    "use_tmux",
    is_flag=True,
    help="Create a tmux session for the worktree (requires --worktree).",
)
@click.option(
    "--tmux-classic",
    is_flag=True,
    default=False,
    hidden=True,
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
    session_id: Optional[str],
    sdk_url: Optional[str],
    no_session_persistence: bool,
    permission_mode: str,
    max_turns: Optional[int],
    allowed_tools_csv: Optional[str],
    disallowed_tools_csv: Optional[str],
    permission_prompt_tool: Optional[str],
    settings: Optional[str],
    add_dirs: tuple[str, ...],
    mcp_config: Optional[str],
    include_partial_messages: bool,
    replay_user_messages: bool,
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
    resume_session_at: Optional[str],
    rewind_files: Optional[str],
    setup_init: bool,
    setup_init_only: bool,
    setup_maintenance: bool,
    use_worktree: bool,
    worktree_name: Optional[str],
    use_tmux: bool,
    tmux_classic: bool,  # noqa: ARG001
) -> None:
    """Ripperdoc - AI-powered coding agent"""
    explicit_session_id = _coerce_session_id(session_id, sdk_url=sdk_url)
    if explicit_session_id and (continue_session or resume_session) and not fork_session:
        raise click.ClickException(
            "Error: --session-id can only be used with --continue or --resume if --fork-session is also specified."
        )
    if no_session_persistence and not _is_non_interactive_mode(
        output_format=output_format,
        print_mode=print_mode,
        setup_init_only=setup_init_only,
        sdk_url=sdk_url,
    ):
        raise click.ClickException(
            "Error: --no-session-persistence can only be used with --print mode."
        )
    if resume_session_at and not resume_session:
        raise click.ClickException(
            "Error: --resume-session-at requires --resume."
        )
    if rewind_files and not resume_session:
        raise click.ClickException("Error: --rewind-files requires --resume")
    if rewind_files and (print_prompt or prompt):
        raise click.ClickException(
            "Error: --rewind-files is a standalone operation and cannot be used with a prompt"
        )

    session_persistence = not no_session_persistence
    session_id = explicit_session_id or str(uuid.uuid4())
    os.environ.setdefault("RIPPERDOC_SESSION_ID", session_id)
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
        session_persistence,
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
        replay_user_messages=replay_user_messages,
        fork_session=fork_session,
        setting_sources=setting_sources,
        plugin_dirs=plugin_dirs,
        betas=betas,
        fallback_model=fallback_model,
        max_budget_usd=max_budget_usd,
        max_thinking_tokens=max_thinking_tokens,
        json_schema=json_schema,
        sdk_url=sdk_url,
        session_persistence=session_persistence,
    )
    effective_append_system_prompt = _merge_append_system_prompt(
        append_system_prompt,
        session_agent_prompt,
    )
    precreated_worktree = _register_precreated_worktree_from_env()
    if precreated_worktree is not None:
        project_path = precreated_worktree.worktree_path.resolve()
        if Path.cwd().resolve() != project_path:
            os.chdir(project_path)
        console.print(f"[dim]Switched to worktree: {precreated_worktree.worktree_path}[/dim]")

    if explicit_session_id is not None and not sdk_url:
        _ensure_session_id_available(session_id=explicit_session_id, project_path=project_path)

    if use_tmux and not use_worktree:
        raise click.ClickException("Error: --tmux requires --worktree")
    if use_tmux and sys.platform == "win32":
        raise click.ClickException("Error: --tmux is not supported on Windows")

    if use_worktree:
        if get_git_root(project_path) is None:
            raise click.ClickException(
                f"Can only use --worktree in a git repository, but {project_path} is not a git repository"
            )
        resolved_worktree_name, pr_number = _resolve_worktree_name_and_pr(worktree_name)
        if resolved_worktree_name is None:
            resolved_worktree_name = generate_cli_worktree_name()
        try:
            worktree_session = create_task_worktree(
                task_id=f"session_{session_id}",
                base_path=project_path,
                requested_name=resolved_worktree_name,
                pr_number=pr_number,
            )
        except (ValueError, RuntimeError, OSError) as exc:
            raise click.ClickException(f"Failed to create --worktree session: {exc}") from exc
        os.chdir(worktree_session.worktree_path)
        project_path = Path.cwd()
        console.print(f"[dim]Switched to worktree: {worktree_session.worktree_path}[/dim]")

    set_runtime_plugin_dirs(plugin_dirs, base_dir=project_path)

    if _run_stdio_mode_if_requested(
        ctx=ctx,
        output_format=output_format,
        print_mode=print_mode,
        print_prompt=print_prompt,
        prompt=prompt,
        input_format=input_format,
        session_persistence=session_persistence,
        model=model,
        permission_mode=effective_permission_mode,
        max_turns=max_turns,
        system_prompt=system_prompt,
        verbose=verbose,
        allowed_tools_csv=allowed_tools_csv,
        disallowed_tools_csv=disallowed_tools_csv,
        continue_session=continue_session,
        resume_session=resume_session,
        resume_session_at=resume_session_at,
        rewind_files=rewind_files,
        fork_session=fork_session,
        replay_user_messages=replay_user_messages,
        session_id=session_id,
        tools=tools,
        project_path=project_path,
        additional_working_dirs=additional_working_dirs,
        sdk_default_options=sdk_default_options,
        sdk_url=sdk_url,
    ):
        return

    session_id, resume_messages, resumed_summary, most_recent = _resolve_resume_state(
        project_path=project_path,
        session_id=session_id,
        fork_session=fork_session,
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

        skip_onboarding = ctx.invoked_subcommand in (
            "update",
            "upgrade",
            "mcp",
            "agents",
            "remote-control",
        )
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
            session_persistence=session_persistence,
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
                    session_persistence=session_persistence,
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
cli.add_command(remote_control_cmd, name="remote-control")
cli.add_command(version_cmd, name="version")


def main() -> None:
    """Main entry point."""
    try:
        argv = sys.argv[1:]
        if _has_tmux_worktree_flags(argv):
            handled, error = _exec_into_tmux_worktree(argv)
            if handled:
                return
            if error:
                console.print(f"[red]{escape(error)}[/red]")
                sys.exit(1)
        top_level_commands = {
            "config",
            "update",
            "upgrade",
            "mcp",
            "agents",
            "remote-control",
            "version",
        }
        if "--print" in argv and "--" in argv:
            sep_index = argv.index("--")
            prompt = " ".join(argv[sep_index + 1 :]).strip()
            argv = argv[:sep_index] + ["--prompt", prompt]
        rewritten_argv: list[str] = []
        index = 0
        while index < len(argv):
            token = argv[index]
            if token.startswith("--worktree="):
                rewritten_argv.append("--worktree")
                value = token.split("=", 1)[1].strip()
                if value:
                    rewritten_argv.extend(["--worktree-name", value])
                index += 1
                continue
            if token == "--tmux=classic":
                rewritten_argv.extend(["--tmux", "--tmux-classic"])
                index += 1
                continue

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
            if token in ("-w", "--worktree"):
                next_index = index + 1
                if next_index < len(argv):
                    candidate = argv[next_index].strip()
                    if candidate and not candidate.startswith("-"):
                        rewritten_argv.extend(["--worktree-name", candidate])
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
