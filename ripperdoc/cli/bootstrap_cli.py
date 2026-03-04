"""Root CLI bootstrap helpers (options parsing, stdio routing, resume/setup)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, List, Optional

import click
from rich.console import Console

from ripperdoc.core.hooks.llm_callback import build_hook_llm_callback
from ripperdoc.core.hooks.manager import hook_manager
from ripperdoc.core.session_agents import (
    merge_session_agents,
    normalize_agent_name,
    parse_session_agents,
    resolve_session_agent_prompt,
)
from ripperdoc.core.tool_defaults import BUILTIN_TOOL_NAMES
from ripperdoc.utils.log import get_logger
from ripperdoc.utils.sessions.session_history import (
    SessionHistory,
    list_session_summaries,
    load_session_messages,
)
from ripperdoc.utils.working_directories import (
    extract_additional_directories,
    normalize_directory_inputs,
)

console = Console()
logger = get_logger()

_SDK_ONLY_OPTION_FLAGS: tuple[tuple[str, str], ...] = (
    ("allowed_tools_csv", "--allowedTools"),
    ("disallowed_tools_csv", "--disallowedTools"),
    ("permission_prompt_tool", "--permission-prompt-tool"),
    ("mcp_config", "--mcp-config"),
    ("include_partial_messages", "--include-partial-messages"),
    ("fork_session", "--fork-session"),
    ("setting_sources", "--setting-sources"),
    ("betas", "--betas"),
    ("max_budget_usd", "--max-budget-usd"),
    ("json_schema", "--json-schema"),
    ("sdk_url", "--sdk-url"),
)


def _resolve_permission_mode(yolo: bool, permission_mode: str) -> str:
    """Normalize effective permission mode, ensuring --yolo maps to bypass mode."""
    return "bypassPermissions" if yolo else permission_mode


def _is_stdio_mode_request(
    ctx: click.Context,
    output_format: str,
    print_mode: bool,
    sdk_url: Optional[str],
) -> bool:
    """Return True when invocation should run through stdio mode."""
    return ctx.invoked_subcommand is None and (
        output_format in ("json", "stream-json") or print_mode or bool(sdk_url)
    )


def _is_non_interactive_mode(
    output_format: str,
    print_mode: bool,
    setup_init_only: bool,
    sdk_url: Optional[str],
) -> bool:
    """Return True when invocation is non-interactive.
      - print mode
      - init-only mode
      - sdk-url mode
      - stdout is not a TTY
    """
    if print_mode or setup_init_only or bool(sdk_url):
        return True

    try:
        stdout_stream = click.get_text_stream("stdout")
        return not stdout_stream.isatty()
    except Exception:
        return False


def _collect_sdk_only_option_uses(
    *,
    allowed_tools_csv: Optional[str],
    disallowed_tools_csv: Optional[str],
    permission_prompt_tool: Optional[str],
    mcp_config: Optional[str],
    include_partial_messages: bool,
    fork_session: bool,
    setting_sources: Optional[str],
    plugin_dirs: tuple[str, ...],
    betas: Optional[str],
    max_budget_usd: Optional[float],
    json_schema: Optional[str],
    sdk_url: Optional[str],
) -> List[str]:
    """Collect SDK-only option flags that were explicitly provided."""
    values: dict[str, Any] = {
        "allowed_tools_csv": allowed_tools_csv,
        "disallowed_tools_csv": disallowed_tools_csv,
        "permission_prompt_tool": permission_prompt_tool,
        "mcp_config": mcp_config,
        "include_partial_messages": include_partial_messages,
        "fork_session": fork_session,
        "setting_sources": setting_sources,
        "plugin_dirs": plugin_dirs,
        "betas": betas,
        "max_budget_usd": max_budget_usd,
        "json_schema": json_schema,
        "sdk_url": sdk_url,
    }
    provided: List[str] = []
    for key, option_name in _SDK_ONLY_OPTION_FLAGS:
        value = values.get(key)
        if value is True or (value and not isinstance(value, bool)):
            provided.append(option_name)
    return provided


def _validate_sdk_only_options_usage(
    *,
    using_stdio_mode: bool,
    provided_options: List[str],
) -> None:
    """Fail fast when SDK-only flags are passed to non-stdio CLI routes."""
    if using_stdio_mode or not provided_options:
        return
    unique_options = ", ".join(sorted(set(provided_options)))
    raise click.ClickException(
        "The following options are SDK-only and require --output-format json/stream-json "
        f"or --print: {unique_options}"
    )


def parse_tools_option(tools_arg: Optional[str]) -> Optional[List[str]]:
    """Parse the --tools argument."""
    if tools_arg is None:
        return None

    tools_arg = tools_arg.strip()
    if tools_arg == "":
        return []
    if tools_arg.lower() == "default":
        return None

    tool_names = [name.strip() for name in tools_arg.split(",") if name.strip()]
    invalid_tools = [name for name in tool_names if name not in BUILTIN_TOOL_NAMES]
    if invalid_tools:
        logger.warning(
            "[cli] Unknown tools specified: %s. Available tools: %s",
            ", ".join(invalid_tools),
            ", ".join(BUILTIN_TOOL_NAMES),
        )

    return tool_names if tool_names else None


def parse_csv_option(raw_value: Optional[str]) -> Optional[List[str]]:
    """Parse a comma-separated CLI option into a list."""
    if raw_value is None:
        return None
    value = raw_value.strip()
    if value == "":
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_debug_filter_from_extra_args(
    *,
    ctx: click.Context,
    debug_enabled: bool,
    explicit_debug_filter: Optional[str],
    print_mode: bool,
    print_prompt: Optional[str],
    prompt: Optional[str],
) -> Optional[str]:
    """Resolve optional --debug filter from root extra args when safe."""
    if explicit_debug_filter is not None:
        cleaned = explicit_debug_filter.strip()
        return cleaned or None
    if not debug_enabled or ctx.invoked_subcommand is not None:
        return None
    if print_mode and print_prompt is None and prompt is None:
        return None
    if not ctx.args:
        return None
    candidate = str(ctx.args[0] or "").strip()
    if not candidate or candidate.startswith("-"):
        return None
    ctx.args = ctx.args[1:]
    return candidate


def _load_settings_payload(settings_value: Optional[str]) -> tuple[dict[str, Any], Path]:
    """Load settings JSON from an inline payload or a file path."""
    if not settings_value:
        return {}, Path.cwd()

    settings_text = settings_value
    settings_base_dir = Path.cwd()
    candidate = Path(settings_value).expanduser()
    if candidate.exists():
        try:
            settings_text = candidate.read_text(encoding="utf-8")
        except (OSError, IOError, UnicodeDecodeError) as exc:
            raise click.ClickException(
                f"Failed to read settings file '{candidate}': {type(exc).__name__}: {exc}"
            ) from exc
        settings_base_dir = candidate.resolve().parent

    try:
        parsed = json.loads(settings_text)
    except json.JSONDecodeError as exc:
        raise click.ClickException(
            f"--settings must be valid JSON or a valid JSON file path: {exc.msg}"
        ) from exc

    if not isinstance(parsed, dict):
        raise click.ClickException("--settings JSON must be an object.")
    return parsed, settings_base_dir


def _resolve_additional_working_dirs(
    settings: Optional[str],
    add_dirs: tuple[str, ...],
    project_path: Path,
) -> List[str]:
    settings_payload, settings_base_dir = _load_settings_payload(settings)
    settings_dirs_raw = extract_additional_directories(settings_payload)
    settings_dirs, settings_dir_errors = normalize_directory_inputs(
        settings_dirs_raw,
        base_dir=settings_base_dir,
        require_exists=True,
    )
    cli_dirs, cli_dir_errors = normalize_directory_inputs(
        add_dirs,
        base_dir=project_path,
        require_exists=True,
    )
    directory_errors = [*settings_dir_errors, *cli_dir_errors]
    if directory_errors:
        raise click.ClickException(
            "Invalid additional directory configuration:\n- " + "\n- ".join(directory_errors)
        )
    return list(dict.fromkeys([*settings_dirs, *cli_dirs]))


def _resolve_session_agent_options(
    *,
    settings: Optional[str],
    agent: Optional[str],
    agents: Optional[str],
) -> tuple[Optional[str], dict[str, dict[str, str]], Optional[str]]:
    settings_payload, _ = _load_settings_payload(settings)
    try:
        settings_agent = normalize_agent_name(settings_payload.get("agent"), source="settings.agent")
        settings_agents = parse_session_agents(
            settings_payload.get("agents"),
            source="settings.agents",
        )
        cli_agent = normalize_agent_name(agent, source="--agent")
        cli_agents = parse_session_agents(agents, source="--agents")
        effective_agents = merge_session_agents(settings_agents, cli_agents)
        effective_agent = cli_agent if cli_agent is not None else settings_agent
        effective_agent_prompt = resolve_session_agent_prompt(
            effective_agent,
            effective_agents,
            source="--agent",
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    return effective_agent, effective_agents, effective_agent_prompt


def _build_sdk_default_options(
    *,
    permission_prompt_tool: Optional[str],
    mcp_config: Optional[str],
    include_partial_messages: bool,
    fork_session: bool,
    agent: Optional[str],
    agents: Optional[dict[str, dict[str, str]]],
    disable_slash_commands: bool,
    setting_sources: Optional[str],
    plugin_dirs: tuple[str, ...],
    betas: Optional[str],
    fallback_model: Optional[str],
    max_budget_usd: Optional[float],
    max_thinking_tokens: Optional[int],
    json_schema: Optional[str],
    sdk_url: Optional[str],
) -> dict[str, Any]:
    options: dict[str, Any] = {}
    value_options: dict[str, Any] = {
        "permission_prompt_tool": permission_prompt_tool,
        "mcp_config": mcp_config,
        "agent": agent,
        "agents": agents,
        "setting_sources": setting_sources,
        "betas": betas,
        "fallback_model": fallback_model,
        "max_budget_usd": max_budget_usd,
        "max_thinking_tokens": max_thinking_tokens,
        "json_schema": json_schema,
    }
    for key, value in value_options.items():
        if value is not None:
            options[key] = value
    if include_partial_messages:
        options["include_partial_messages"] = True
    if fork_session:
        options["fork_session"] = True
    if disable_slash_commands:
        options["disable_slash_commands"] = True
    if plugin_dirs:
        options["plugin_dirs"] = list(plugin_dirs)
    if sdk_url:
        options["sdk_url"] = sdk_url
    return options


def _run_stdio_mode_if_requested(
    *,
    ctx: click.Context,
    output_format: str,
    print_mode: bool,
    print_prompt: Optional[str],
    prompt: Optional[str],
    input_format: str,
    model: Optional[str],
    session_id: str,
    session_persistence: bool,
    permission_mode: str,
    max_turns: Optional[int],
    system_prompt: Optional[str],
    verbose: bool,
    continue_session: bool,
    resume_session: Optional[str],
    resume_session_at: Optional[str],
    rewind_files: Optional[str],
    fork_session: bool,
    allowed_tools_csv: Optional[str],
    disallowed_tools_csv: Optional[str],
    tools: Optional[str],
    project_path: Path,
    additional_working_dirs: List[str],
    sdk_default_options: dict[str, Any],
    sdk_url: Optional[str],
) -> bool:
    if ctx.invoked_subcommand is not None:
        return False
    if output_format not in ("json", "stream-json") and not print_mode and not sdk_url:
        return False

    from ripperdoc.protocol.stdio import run_stdio

    effective_prompt = print_prompt or prompt
    effective_print_mode = bool(print_mode or sdk_url)
    if sdk_url:
        input_format = "stream-json"
        if output_format == "text":
            output_format = "stream-json"
        if output_format != "stream-json":
            raise click.ClickException(
                "Error: --sdk-url requires both --input-format=stream-json and --output-format=stream-json."
            )
        stdio_output_format = "stream-json"
        effective_verbose = True
    else:
        stdio_output_format = "stream-json" if output_format == "text" else output_format
        effective_verbose = verbose
    allowed_tools = parse_csv_option(allowed_tools_csv)
    disallowed_tools = parse_csv_option(disallowed_tools_csv)
    tools_list = parse_tools_option(tools)

    if effective_print_mode:
        stdin_prompt = _read_stdin_prompt_for_headless(
            base_prompt=effective_prompt,
            input_format=input_format,
            session_id=session_id,
        )
        if stdin_prompt is not None:
            effective_prompt = stdin_prompt

    default_options: dict[str, Any] = {
        "cwd": str(project_path),
        "model": model,
        "permission_mode": permission_mode,
        "max_turns": max_turns,
        "system_prompt": system_prompt,
        "verbose": effective_verbose,
    }
    if allowed_tools is not None:
        default_options["allowed_tools"] = allowed_tools
    if disallowed_tools is not None:
        default_options["disallowed_tools"] = disallowed_tools
    if tools_list is not None:
        default_options["tools"] = tools_list
    if additional_working_dirs:
        default_options["additional_directories"] = additional_working_dirs
    default_options["session_persistence"] = session_persistence
    default_options["session_id"] = session_id
    default_options.update(sdk_default_options)

    asyncio.run(
        run_stdio(
            input_format=input_format,
            output_format=stdio_output_format,
            model=model,
            permission_mode=permission_mode,
            max_turns=max_turns,
            system_prompt=system_prompt,
            session_id=session_id,
            resume_session=resume_session,
            continue_session=continue_session,
            resume_session_at=resume_session_at,
            rewind_files=rewind_files,
            fork_session=fork_session,
            project_path=str(project_path),
            print_mode=effective_print_mode,
            prompt=effective_prompt,
            default_options=default_options,
        )
    )
    return True


def _read_stdin_prompt_for_headless(
    *,
    base_prompt: Optional[str],
    input_format: str,
    session_id: str,
) -> Optional[str]:
    if input_format == "stream-json":
        return base_prompt

    stdin_stream = click.get_text_stream("stdin")
    try:
        stdin_is_tty = stdin_stream.isatty()
    except Exception:
        stdin_is_tty = True

    if stdin_is_tty:
        return base_prompt

    try:
        stdin_data = stdin_stream.read()
    except (OSError, ValueError) as exc:
        logger.warning(
            "[cli] Failed to read stdin for print query merge: %s: %s",
            type(exc).__name__,
            exc,
            extra={"session_id": session_id},
        )
        return base_prompt

    merged_input = stdin_data.rstrip("\n")
    if not merged_input and not base_prompt:
        return ""
    if not merged_input:
        return base_prompt
    if not base_prompt:
        return merged_input
    return f"{base_prompt}\n{merged_input}"


def _resolve_resume_state(
    *,
    project_path: Path,
    session_id: str,
    fork_session: bool,
    resume_session: Optional[str],
    continue_session: bool,
) -> tuple[str, Optional[List[Any]], Any, Any]:
    resume_messages: Optional[List[Any]] = None
    resumed_summary = None
    most_recent = None
    resumed_session_id = None

    if resume_session:
        summaries = list_session_summaries(project_path)
        if summaries:
            match = next((s for s in summaries if s.session_id.startswith(resume_session)), None)
            if match:
                resumed_summary = match
                resumed_session_id = match.session_id
                resume_messages = load_session_messages(project_path, resumed_session_id)
                console.print(f"[dim]Resuming session: {match.display_title}[/dim]")
            else:
                raise click.ClickException(f"No session found matching '{resume_session}'.")
        else:
            raise click.ClickException("No previous sessions found in this directory.")
    elif continue_session:
        summaries = list_session_summaries(project_path)
        if summaries:
            most_recent = summaries[0]
            resumed_session_id = most_recent.session_id
            resume_messages = load_session_messages(project_path, resumed_session_id)
            console.print(f"[dim]Continuing session: {most_recent.display_title}[/dim]")
        else:
            console.print("[yellow]No previous sessions found in this directory.[/yellow]")

    if resumed_session_id and not fork_session:
        session_id = resumed_session_id

    return session_id, resume_messages, resumed_summary, most_recent


def _log_resume_state(
    *,
    session_id: str,
    log_file: Path,
    resume_messages: Optional[List[Any]],
    resumed_summary: Any,
    most_recent: Any,
    continue_session: bool,
) -> None:
    if resumed_summary:
        logger.info(
            "[cli] Resuming session",
            extra={
                "session_id": session_id,
                "message_count": len(resume_messages) if resume_messages else 0,
                "title": resumed_summary.display_title,
                "last_prompt": resumed_summary.last_prompt,
                "log_file": str(log_file),
            },
        )
    elif most_recent:
        logger.info(
            "[cli] Continuing session",
            extra={
                "session_id": session_id,
                "message_count": len(resume_messages) if resume_messages else 0,
                "title": most_recent.display_title,
                "last_prompt": most_recent.last_prompt,
                "log_file": str(log_file),
            },
        )
    elif continue_session:
        logger.warning("[cli] No previous sessions found to continue")


def _resolve_setup_trigger(
    *,
    setup_init: bool,
    setup_init_only: bool,
    setup_maintenance: bool,
) -> Optional[str]:
    triggers = [(setup_init, "init"), (setup_init_only, "init"), (setup_maintenance, "maintenance")]
    active_triggers = [name for active, name in triggers if active]
    if len(active_triggers) > 1:
        raise click.ClickException("Use only one of --init, --init-only, or --maintenance.")
    return active_triggers[0] if active_triggers else None


def _run_setup_if_needed(
    *,
    setup_trigger: Optional[str],
    project_path: Path,
    session_persistence: bool,
    session_id: str,
    setup_init_only: bool,
    setup_maintenance: bool,
) -> bool:
    if not setup_trigger:
        return False
    hook_manager.set_project_dir(project_path)
    hook_manager.set_session_id(session_id)
    hook_manager.set_llm_callback(build_hook_llm_callback())
    session_history = SessionHistory(project_path, session_id, session_persistence=session_persistence)
    hook_manager.set_transcript_path(str(session_history.path))
    hook_manager.run_setup(setup_trigger)
    if setup_trigger == "init":
        hook_manager._setup_ran_for_project = project_path
    return bool(setup_init_only or setup_maintenance)


def _resolve_root_extra_args(
    *,
    ctx: click.Context,
    print_mode: bool,
    print_prompt: Optional[str],
) -> Optional[str]:
    """Normalize extra root args or raise a click error for invalid usage."""
    if ctx.invoked_subcommand is not None or not ctx.args:
        return print_prompt

    extra_args = list(ctx.args)
    if print_mode:
        return print_prompt or " ".join(extra_args).strip()

    first = extra_args[0]
    if first.startswith("-"):
        ctx.fail(f"No such option: {first}")
    ctx.fail(f"No such command '{first}'")


def _change_cwd_if_requested(cwd: Optional[str]) -> Optional[str]:
    """Change process cwd when --cwd is provided and return the applied value."""
    if not cwd:
        return None
    import os

    os.chdir(cwd)
    return cwd


def _prepare_cli_runtime_inputs(
    *,
    ctx: click.Context,
    output_format: str,
    print_mode: bool,
    settings: Optional[str],
    add_dirs: tuple[str, ...],
    agent: Optional[str],
    agents: Optional[str],
    disable_slash_commands: bool,
    allowed_tools_csv: Optional[str],
    disallowed_tools_csv: Optional[str],
    permission_prompt_tool: Optional[str],
    mcp_config: Optional[str],
    include_partial_messages: bool,
    fork_session: bool,
    setting_sources: Optional[str],
    plugin_dirs: tuple[str, ...],
    betas: Optional[str],
    fallback_model: Optional[str],
    max_budget_usd: Optional[float],
    max_thinking_tokens: Optional[int],
    json_schema: Optional[str],
    sdk_url: Optional[str],
    session_persistence: bool,
) -> tuple[Path, List[str], dict[str, Any], Optional[str], bool]:
    """Resolve working-directory inputs and SDK default options for CLI entrypoints."""
    project_path = Path.cwd()
    additional_working_dirs = _resolve_additional_working_dirs(settings, add_dirs, project_path)
    effective_agent, effective_agents, effective_agent_prompt = _resolve_session_agent_options(
        settings=settings,
        agent=agent,
        agents=agents,
    )
    stdio_mode_request = _is_stdio_mode_request(
        ctx=ctx,
        output_format=output_format,
        print_mode=print_mode,
        sdk_url=sdk_url,
    )
    provided_sdk_only_options = _collect_sdk_only_option_uses(
        allowed_tools_csv=allowed_tools_csv,
        disallowed_tools_csv=disallowed_tools_csv,
        permission_prompt_tool=permission_prompt_tool,
        mcp_config=mcp_config,
        include_partial_messages=include_partial_messages,
        fork_session=fork_session,
        setting_sources=setting_sources,
        plugin_dirs=plugin_dirs,
        betas=betas,
        max_budget_usd=max_budget_usd,
        json_schema=json_schema,
        sdk_url=sdk_url,
    )
    _validate_sdk_only_options_usage(
        using_stdio_mode=stdio_mode_request,
        provided_options=provided_sdk_only_options,
    )
    sdk_default_options = _build_sdk_default_options(
        permission_prompt_tool=permission_prompt_tool,
        mcp_config=mcp_config,
        include_partial_messages=include_partial_messages,
        fork_session=fork_session,
        agent=effective_agent,
        agents=effective_agents if effective_agents else None,
        disable_slash_commands=disable_slash_commands,
        setting_sources=setting_sources,
        plugin_dirs=plugin_dirs,
        betas=betas,
        fallback_model=fallback_model,
        max_budget_usd=max_budget_usd,
        max_thinking_tokens=max_thinking_tokens,
        json_schema=json_schema,
        sdk_url=sdk_url,
    )
    return (
        project_path,
        additional_working_dirs,
        sdk_default_options,
        effective_agent_prompt,
        session_persistence,
    )


def _read_initial_query_from_stdin(
    *,
    prompt: Optional[str],
    invoked_subcommand: Optional[str],
    session_id: str,
) -> Optional[str]:
    if prompt is not None or invoked_subcommand is not None:
        return None

    stdin_stream = click.get_text_stream("stdin")
    try:
        stdin_is_tty = stdin_stream.isatty()
    except Exception:
        stdin_is_tty = True

    if stdin_is_tty:
        return None

    try:
        stdin_data = stdin_stream.read()
    except (OSError, ValueError) as exc:
        logger.warning(
            "[cli] Failed to read stdin for initial query: %s: %s",
            type(exc).__name__,
            exc,
            extra={"session_id": session_id},
        )
        return None

    trimmed = stdin_data.rstrip("\n")
    if not trimmed.strip():
        return None

    logger.info(
        "[cli] Received initial query from stdin",
        extra={
            "session_id": session_id,
            "query_length": len(trimmed),
            "query_preview": trimmed[:200],
        },
    )
    return trimmed


__all__ = [
    "_change_cwd_if_requested",
    "_log_resume_state",
    "_prepare_cli_runtime_inputs",
    "_read_initial_query_from_stdin",
    "_resolve_permission_mode",
    "_resolve_resume_state",
    "_resolve_root_extra_args",
    "_resolve_setup_trigger",
    "_run_setup_if_needed",
    "_run_stdio_mode_if_requested",
    "parse_tools_option",
    "resolve_debug_filter_from_extra_args",
]
