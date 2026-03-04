"""Prompt/runtime helpers for the root CLI command."""

from __future__ import annotations

import asyncio
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape

from ripperdoc.core.config import get_effective_model_profile, get_project_local_config
from ripperdoc.core.hooks.llm_callback import build_hook_llm_callback
from ripperdoc.core.hooks.manager import hook_manager
from ripperdoc.core.hooks.state import bind_pending_message_queue
from ripperdoc.core.permission_engine import make_permission_checker
from ripperdoc.core.query import QueryContext, query
from ripperdoc.core.skills import build_skill_summary, filter_enabled_skills, load_all_skills
from ripperdoc.core.system_prompt import build_system_prompt
from ripperdoc.core.tool_defaults import filter_tools_by_names
from ripperdoc.cli.ui.choice import ChoiceOption, prompt_choice_async
from ripperdoc.tools.background_shell import shutdown_background_shell
from ripperdoc.tools.task_tool import list_running_agent_worktree_paths
from ripperdoc.tools.dynamic_mcp_tool import (
    load_dynamic_mcp_tools_async,
    merge_tools_with_dynamic,
)
from ripperdoc.utils.log import get_logger
from ripperdoc.utils.lsp import shutdown_lsp_manager
from ripperdoc.utils.mcp import (
    format_mcp_instructions,
    load_mcp_servers_async,
    shutdown_mcp_runtime,
)
from ripperdoc.utils.memory import build_memory_instructions
from ripperdoc.utils.messaging.messages import create_user_message
from ripperdoc.utils.sessions.session_history import SessionHistory
from ripperdoc.utils.collaboration.tasks import set_runtime_task_scope
from ripperdoc.utils.collaboration.worktree import (
    cleanup_worktree_sessions,
    consume_session_worktrees,
    list_session_worktrees,
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


def _collect_hook_contexts(result: Any) -> List[str]:
    contexts: List[str] = []
    additional_context = getattr(result, "additional_context", None)
    if additional_context:
        contexts.append(str(additional_context))
    return contexts


def _print_hook_system_message(result: Any, event: str) -> None:
    system_message = getattr(result, "system_message", None)
    if not system_message:
        return
    console.print(f"[yellow]Hook {event}: {escape(str(system_message))}[/yellow]")


async def _prepare_prompt_runtime_assets(
    *,
    project_path: Path,
    tools: list,
    allowed_tools: Optional[List[str]],
    query_context: QueryContext,
    disable_skills: bool,
) -> tuple[list, str, List[str]]:
    servers, dynamic_tools = await asyncio.gather(
        load_mcp_servers_async(project_path),
        load_dynamic_mcp_tools_async(project_path),
    )
    if dynamic_tools:
        tools = merge_tools_with_dynamic(tools, dynamic_tools)
        if allowed_tools is not None:
            tools = filter_tools_by_names(tools, allowed_tools)
        query_context.tools = tools
    mcp_instructions = format_mcp_instructions(servers)

    additional_instructions: List[str] = []
    if not disable_skills:
        skill_result = load_all_skills(project_path)
        for err in skill_result.errors:
            logger.warning(
                "[skills] Failed to load skill",
                extra={"path": str(err.path), "reason": err.reason},
            )
        enabled_skills = filter_enabled_skills(skill_result.skills, project_path=project_path)
        skill_instructions = build_skill_summary(enabled_skills)
        if skill_instructions:
            additional_instructions.append(skill_instructions)
    memory_instructions = build_memory_instructions()
    if memory_instructions:
        additional_instructions.append(memory_instructions)
    return tools, mcp_instructions, additional_instructions


async def _run_prompt_submission_hooks(
    *,
    prompt: str,
    query_context: QueryContext,
    additional_instructions: List[str],
) -> bool:
    with bind_pending_message_queue(query_context.pending_message_queue):
        session_start_result = await hook_manager.run_session_start_async("startup")
        _print_hook_system_message(session_start_result, "SessionStart")
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
            return False
        _print_hook_system_message(prompt_hook_result, "UserPromptSubmit")
        prompt_hook_contexts = _collect_hook_contexts(prompt_hook_result)
        if prompt_hook_contexts:
            additional_instructions.extend(prompt_hook_contexts)
    return True


def _build_effective_system_prompt(
    *,
    custom_system_prompt: Optional[str],
    append_system_prompt: Optional[str],
    additional_instructions: List[str],
    tools: list,
    prompt: str,
    context: Dict[str, Any],
    mcp_instructions: str,
    output_style: str,
    output_language: str,
    project_path: Path,
) -> str:
    if custom_system_prompt:
        system_prompt = custom_system_prompt
        if append_system_prompt:
            system_prompt = f"{system_prompt}\n\n{append_system_prompt}"
        return system_prompt

    all_instructions = list(additional_instructions) if additional_instructions else []
    if append_system_prompt:
        all_instructions.append(append_system_prompt)
    return build_system_prompt(
        tools,
        prompt,
        context,
        additional_instructions=all_instructions or None,
        mcp_instructions=mcp_instructions,
        output_style=output_style,
        output_language=output_language,
        project_path=project_path,
    )


def _print_hook_notice_from_progress(message: Any) -> None:
    content = getattr(message, "content", None)
    from ripperdoc.utils.messaging.messages import is_hook_notice_payload

    if not (is_hook_notice_payload(content) and isinstance(content, dict)):
        return
    event = content.get("hook_event", "Hook")
    tool_name = content.get("tool_name")
    label = f"{event}:{tool_name}" if tool_name else str(event)
    text = content.get("text", "")
    console.print(f"[yellow]Hook {escape(str(label))}[/yellow] {escape(str(text))}")


def _collect_assistant_text_blocks(message: Any) -> List[str]:
    if not (message.type == "assistant" and hasattr(message, "message")):
        return []
    if isinstance(message.message.content, str):
        return [message.message.content]

    parts: List[str] = []
    for block in message.message.content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(block["text"])
            continue
        if hasattr(block, "type") and block.type == "text":
            parts.append(block.text or "")
    return parts


async def _handle_session_worktree_cleanup(*, interactive: bool) -> None:
    sessions = list_session_worktrees()
    if not sessions:
        return
    running_paths = list_running_agent_worktree_paths()
    removable = [session for session in sessions if str(session.worktree_path) not in running_paths]
    pinned = [session for session in sessions if str(session.worktree_path) in running_paths]

    selection = "keep"
    if interactive:
        paths = "\n".join(f"- {session.worktree_path}" for session in sessions[:5])
        extra = ""
        if len(sessions) > 5:
            extra = f"\n- ... and {len(sessions) - 5} more"
        running_note = (
            f"\n\n<dim>{len(pinned)} worktree(s) are in use by running subagents and cannot be removed now.</dim>"
            if pinned
            else ""
        )
        prompt_message = (
            "<question>Worktree cleanup before exit?</question>\n\n"
            "The following Task worktrees were created in this session:\n"
            f"<value>{paths}{extra}</value>"
            f"{running_note}"
        )
        try:
            selection = await prompt_choice_async(
                message=prompt_message,
                options=[
                    ChoiceOption("keep", "<yes-option>Keep worktrees</yes-option>"),
                    ChoiceOption("remove", "<no-option>Remove worktrees</no-option>"),
                ],
                title="Worktree Cleanup",
                allow_esc=True,
                esc_value="keep",
            )
        except (EOFError, KeyboardInterrupt, RuntimeError, ValueError, OSError):
            selection = "keep"

    if selection != "remove":
        kept = consume_session_worktrees()
        console.print(
            f"[dim]Preserved {len(kept)} worktree(s).[/dim]"
        )
        return

    consume_session_worktrees()
    results = cleanup_worktree_sessions(removable, force=True)
    for session in pinned:
        register_session_worktree(session)

    removed_count = sum(1 for result in results if result.removed)
    failed_count = len(results) - removed_count
    console.print(
        f"[dim]Removed {removed_count}/{len(results)} removable worktree(s).[/dim]"
    )
    if pinned:
        console.print(
            f"[dim]Kept {len(pinned)} running worktree(s).[/dim]"
        )
    if failed_count:
        for result in results:
            if result.removed:
                continue
            console.print(
                "[yellow]"
                + f"Failed to remove worktree: {result.worktree_path} ({result.error or 'unknown error'})"
                + "[/yellow]"
            )


async def _stream_prompt_query_messages(
    *,
    messages: List[Any],
    system_prompt: str,
    context: Dict[str, Any],
    query_context: QueryContext,
    can_use_tool: Any,
    session_history: SessionHistory,
) -> List[str]:
    final_response_parts: List[str] = []
    async for message in query(messages, system_prompt, context, query_context, can_use_tool):
        if message.type == "progress":
            _print_hook_notice_from_progress(message)
        final_response_parts.extend(_collect_assistant_text_blocks(message))
        messages.append(message)
        session_history.append(message)
    return final_response_parts


async def run_query(
    prompt: str,
    tools: list,
    yolo_mode: bool = False,
    verbose: bool = False,
    session_id: Optional[str] = None,
    custom_system_prompt: Optional[str] = None,
    append_system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    fallback_model: Optional[str] = None,
    max_thinking_tokens: Optional[int] = None,
    max_turns: Optional[int] = None,
    permission_mode: str = "default",
    allowed_tools: Optional[List[str]] = None,
    additional_working_dirs: Optional[List[str]] = None,
    disable_skills: bool = False,
    session_persistence: bool = True,
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
            "fallback_model": fallback_model,
            "max_thinking_tokens": max(0, int(max_thinking_tokens or 0)),
            "max_turns": max_turns,
            "permission_mode": permission_mode,
            "has_custom_system_prompt": custom_system_prompt is not None,
            "has_append_system_prompt": append_system_prompt is not None,
            "disable_skills": disable_skills,
        },
    )
    if prompt:
        logger.debug(
            "[cli] Prompt preview",
            extra={"session_id": session_id, "prompt_preview": prompt[:200]},
        )

    project_path = Path.cwd()
    set_runtime_task_scope(session_id=session_id, project_root=project_path)
    project_local_config = get_project_local_config(project_path)
    output_style = getattr(project_local_config, "output_style", "default") or "default"
    output_language = getattr(project_local_config, "output_language", "auto") or "auto"
    can_use_tool = (
        None
        if yolo_mode
        else make_permission_checker(
            project_path,
            yolo_mode=False,
            permission_mode=permission_mode,
            session_additional_working_dirs=additional_working_dirs or [],
        )
    )

    hook_manager.set_project_dir(project_path)
    hook_manager.set_session_id(session_id)
    hook_manager.set_llm_callback(build_hook_llm_callback())
    session_history = SessionHistory(
        project_path,
        session_id or str(uuid.uuid4()),
        session_persistence=session_persistence,
    )
    hook_manager.set_transcript_path(str(session_history.path))
    messages: List[Any] = [create_user_message(prompt)]
    session_history.append(messages[0])

    resolved_model = _resolve_model_pointer_with_fallback(
        model,
        fallback_model,
        session_id=session_id,
        route="prompt",
    )
    hook_manager.set_permission_mode(permission_mode)

    query_context = QueryContext(
        tools=tools,
        yolo_mode=yolo_mode,
        verbose=verbose,
        model=resolved_model,
        max_thinking_tokens=max(0, int(max_thinking_tokens or 0)),
        max_turns=max_turns,
        permission_mode=permission_mode,
    )

    session_start_time = time.time()
    try:
        context: Dict[str, Any] = {}
        tools, mcp_instructions, additional_instructions = await _prepare_prompt_runtime_assets(
            project_path=project_path,
            tools=tools,
            allowed_tools=allowed_tools,
            query_context=query_context,
            disable_skills=disable_skills,
        )
        should_continue = await _run_prompt_submission_hooks(
            prompt=prompt,
            query_context=query_context,
            additional_instructions=additional_instructions,
        )
        if not should_continue:
            return

        system_prompt = _build_effective_system_prompt(
            custom_system_prompt=custom_system_prompt,
            append_system_prompt=append_system_prompt,
            additional_instructions=additional_instructions,
            tools=tools,
            prompt=prompt,
            context=context,
            mcp_instructions=mcp_instructions,
            output_style=output_style,
            output_language=output_language,
            project_path=project_path,
        )

        try:
            final_response_parts = await _stream_prompt_query_messages(
                messages=messages,
                system_prompt=system_prompt,
                context=context,
                query_context=query_context,
                can_use_tool=can_use_tool,
                session_history=session_history,
            )
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
        set_runtime_task_scope(session_id=None)
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
        await _handle_session_worktree_cleanup(
            interactive=bool(sys.stdin.isatty() and sys.stdout.isatty())
        )
        await shutdown_mcp_runtime()
        await shutdown_lsp_manager()
        try:
            shutdown_background_shell(force=True)
        except (OSError, RuntimeError) as exc:
            logger.debug(
                "[cli] Failed to shut down background shell: %s: %s",
                type(exc).__name__,
                exc,
            )
        logger.debug("[cli] Shutdown MCP runtime", extra={"session_id": session_id})


__all__ = ["run_query"]
