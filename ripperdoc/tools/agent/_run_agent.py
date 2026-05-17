"""Agent execution logic extracted from TaskTool for foreground/background runs."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from ripperdoc.core.query import QueryContext, query
from ripperdoc.core.tool import Tool, ToolProgress
from ripperdoc.utils.collaboration.teams import TeamMessageType, send_team_message
from ripperdoc.utils.collaboration.teammate_state import IdleReason, set_teammate_idle
from ripperdoc.utils.collaboration.worktree import (
    WorktreeSession,
    cleanup_worktree_session,
    has_worktree_changes,
    unregister_session_worktree,
)
from ripperdoc.utils.log import get_logger
from ripperdoc.utils.messaging.messages import (
    AssistantMessage,
    UserMessage,
    create_hook_notice_payload,
    create_user_message,
)
from ripperdoc.utils.messaging.pending_messages import PendingMessageQueue
from ripperdoc.utils.collaboration.task_notifications import enqueue_task_notification

from ripperdoc.tools.agent._store import (
    AgentRunRecord,
    _send_idle_notification_to_team_lead,
    _set_team_member_active_state,
    _write_task_output,
)
from ripperdoc.tools.agent._agent_utils import (
    extract_approved_shutdown_response,
    extract_text,
    summarize_tool_input,
)
from ripperdoc.tools.agent._constants import ONE_SHOT_BUILTIN_AGENT_TYPES

logger = get_logger()


def send_team_event(
    *,
    record: AgentRunRecord,
    message_type: TeamMessageType,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    if not record.team_name:
        return
    recipients = ["team-lead"]
    sender = record.teammate_name or record.agent_type
    try:
        send_team_message(
            team_name=record.team_name,
            sender=sender,
            recipients=recipients,
            message_type=message_type,
            content=content,
            metadata=metadata or {},
        )
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        logger.warning(
            "[task_tool] Failed to emit team event: %s: %s",
            type(exc).__name__,
            exc,
            extra={"team_name": record.team_name, "message_type": message_type},
        )


def enqueue_background_completion_notification(
    *,
    record: AgentRunRecord,
    queue: Optional[PendingMessageQueue],
    parent_tool_use_id: Optional[str],
) -> None:
    """Forward background subagent completion to the parent notification queue."""
    if queue is None or record.completion_notified:
        return
    record.completion_notified = True

    status = (record.status or "completed").strip() or "completed"
    summary_text = (record.result_text or record.error or "").strip()
    if not summary_text:
        summary_text = f"Subagent '{record.agent_type}' finished with status '{status}'."
    summary = summary_text if len(summary_text) <= 800 else summary_text[:797] + "..."

    enqueue_task_notification(
        queue,
        task_id=record.agent_id,
        status=status,
        summary=summary,
        tool_use_id=parent_tool_use_id,
        output_file=record.output_file,
        usage=record.usage,
        source="background_task",
        extra_metadata={
            "task_type": "local_agent",
            "agent_type": record.agent_type,
            "team_name": record.team_name,
            "teammate_name": record.teammate_name,
        },
    )


async def wait_for_running_record(record: AgentRunRecord) -> None:
    if not record.task or record.task.done():
        return
    try:
        await record.task
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        record.status = "failed"
        record.error = str(exc)


def handoff_foreground_run_to_background(
    *,
    record: AgentRunRecord,
    subagent_context: QueryContext,
    permission_checker: Any,
    notification_queue: Optional[PendingMessageQueue],
    parent_tool_use_id: Optional[str],
) -> bool:
    """Detach a foreground run and continue it as a background subagent task."""
    if record.task and not record.task.done():
        return True
    try:
        record.status = "running"
        record.error = None
        record.is_background = True
        record.completion_notified = False
        record.task = asyncio.create_task(
            run_subagent_background(
                record,
                subagent_context,
                permission_checker,
                notification_queue=notification_queue,
                parent_tool_use_id=parent_tool_use_id,
            )
        )
        return True
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "[task_tool] Failed to hand off foreground subagent to background: %s: %s",
            type(exc).__name__,
            exc,
            extra={"agent_id": record.agent_id, "team_name": record.team_name},
        )
        return False


def build_subagent_start_notices(
    hook_result: Any,
    *,
    agent_type: str,
) -> List[ToolProgress]:
    notices: List[ToolProgress] = []
    if hook_result.should_block or hook_result.should_ask or not hook_result.should_continue:
        reason = (
            hook_result.block_reason
            or hook_result.stop_reason
            or "SubagentStart hook requested to stop."
        )
        notices.append(
            ToolProgress(
                content=create_hook_notice_payload(
                    text=f"SubagentStart hook warning (ignored): {reason}",
                    hook_event="SubagentStart",
                    tool_name=agent_type,
                    level="warning",
                )
            )
        )
    if hook_result.system_message:
        notices.append(
            ToolProgress(
                content=create_hook_notice_payload(
                    text=str(hook_result.system_message),
                    hook_event="SubagentStart",
                    tool_name=agent_type,
                )
            )
        )
    return notices


def reset_record_for_resume_prompt(record: AgentRunRecord, prompt: str) -> None:
    record.history.append(create_user_message(prompt))
    record.task_prompt = prompt
    record.start_time = time.time()
    record.duration_ms = 0.0
    record.tool_use_count = 0
    record.total_tokens = 0
    record.usage = None
    record.status = "running"
    record.result_text = None
    record.error = None
    record.task = None
    _write_task_output(
        record.output_file,
        f"=== Resume {time.strftime('%Y-%m-%d %H:%M:%S')} ===\nPrompt: {prompt}",
        append=True,
    )


def build_subagent_query_context(
    *,
    tools: List[Tool[Any, Any]],
    yolo_mode: bool,
    verbose: bool,
    model: str,
    agent_type: str,
    team_name: Optional[str],
    teammate_name: Optional[str],
    agent_id: str,
    hook_scopes: List[Any],
    max_turns: Optional[int] = None,
    permission_mode: str = "default",
    working_directory: Optional[str] = None,
    task_notification_queue: Optional[PendingMessageQueue] = None,
) -> QueryContext:
    return QueryContext(
        tools=tools,
        yolo_mode=yolo_mode,
        verbose=verbose,
        model=model,
        stop_hook="subagent",
        subagent_type=agent_type,
        team_name=team_name,
        teammate_name=teammate_name,
        agent_id=agent_id,
        hook_scopes=hook_scopes,
        max_turns=max_turns,
        permission_mode=permission_mode,
        working_directory=working_directory,
        task_notification_queue=task_notification_queue,
    )


def maybe_autocleanup_worktree(record: AgentRunRecord) -> None:
    """Auto-clean worktree when there are no changes."""
    if record.isolation_mode != "worktree":
        return
    if not record.worktree_path:
        return

    worktree_path = Path(record.worktree_path)
    if not worktree_path.exists():
        unregister_session_worktree(worktree_path)
        return

    baseline_ref = record.worktree_head_commit or record.worktree_branch or None
    try:
        changed = has_worktree_changes(
            worktree_path=worktree_path,
            baseline_ref=baseline_ref,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        logger.debug(
            "[task_tool] Failed to evaluate worktree changes",
            extra={
                "agent_id": record.agent_id,
                "worktree_path": record.worktree_path,
                "error": str(exc),
            },
        )
        return

    if changed:
        return

    repo_root = (
        Path(record.worktree_repo_root).resolve()
        if record.worktree_repo_root
        else worktree_path.parent.parent.parent.resolve()
    )
    session = WorktreeSession(
        repo_root=repo_root,
        worktree_path=worktree_path.resolve(),
        branch=record.worktree_branch or "",
        name=record.worktree_name or worktree_path.name,
        head_commit=record.worktree_head_commit,
        hook_based=record.worktree_hook_based,
    )
    cleanup = cleanup_worktree_session(session, force=True)
    if cleanup.removed:
        unregister_session_worktree(worktree_path)
        record.worktree_path = None
    if cleanup.branch_deleted:
        record.worktree_branch = None
    if cleanup.removed and (cleanup.branch_deleted or not session.branch):
        if record.result_text:
            record.result_text = (
                f"{record.result_text} (No worktree changes detected; temporary worktree cleaned up automatically.)"
            )
        return

    if cleanup.error:
        record.error = (
            f"{record.error}; auto-cleanup failed: {cleanup.error}"
            if record.error
            else f"auto-cleanup failed: {cleanup.error}"
        )


def finalize_record_from_messages(
    record: AgentRunRecord,
    *,
    assistant_messages: List[AssistantMessage],
    tool_use_count: int,
    status: str = "completed",
    error: Optional[str] = None,
    result_text: Optional[str] = None,
) -> None:
    duration_ms = (time.time() - record.start_time) * 1000
    if result_text is None:
        result_text = (
            extract_text(assistant_messages[-1])
            if assistant_messages
            else (
                f"Subagent '{record.agent_type}' ended with status '{status}'."
                if status != "completed"
                else "Agent returned no response."
            )
        )
    record.duration_ms = duration_ms
    record.tool_use_count = tool_use_count
    record.result_text = result_text.strip()
    total_input_tokens = sum(max(getattr(msg, "input_tokens", 0), 0) for msg in assistant_messages)
    total_output_tokens = sum(max(getattr(msg, "output_tokens", 0), 0) for msg in assistant_messages)
    cache_read_tokens = sum(max(getattr(msg, "cache_read_tokens", 0), 0) for msg in assistant_messages)
    cache_creation_tokens = sum(
        max(getattr(msg, "cache_creation_tokens", 0), 0) for msg in assistant_messages
    )
    record.total_tokens = total_input_tokens + total_output_tokens
    record.usage = {
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "cache_creation_input_tokens": cache_creation_tokens or None,
        "cache_read_input_tokens": cache_read_tokens or None,
        "server_tool_use": None,
        "service_tier": None,
        "cache_creation": None,
    }
    record.status = status
    if status == "completed":
        record.error = None
    elif error is not None:
        record.error = error
    maybe_autocleanup_worktree(record)

    # Set idle state and send idle notification
    idle_reason = IdleReason.AVAILABLE
    if status == "failed":
        idle_reason = IdleReason.FAILED
    elif status == "cancelled":
        idle_reason = IdleReason.INTERRUPTED
    elif status == "shutdown":
        idle_reason = IdleReason.SHUTDOWN

    record.is_idle = True

    # Execute idle callbacks
    for callback in record.on_idle_callbacks:
        try:
            callback()
        except Exception as exc:
            logger.warning(
                "[task_tool] Idle callback failed for %s: %s: %s",
                record.agent_id,
                type(exc).__name__,
                exc,
            )
    record.on_idle_callbacks = []

    # Sync with teammate_state if available
    if record.teammate_state:
        set_teammate_idle(
            record.teammate_state.id,
            idle_reason=idle_reason,
            summary=result_text[:500] if result_text else None,
        )

    # Send idle notification to team lead
    if record.team_name:
        _send_idle_notification_to_team_lead(
            record,
            idle_reason=idle_reason,
            summary=result_text[:500] if result_text else None,
        )

    _set_team_member_active_state(
        record.team_name,
        record.teammate_name,
        False,
        default_agent_type=record.agent_type,
    )
    send_team_event(
        record=record,
        message_type="status",
        content=(
            f"Subagent '{record.agent_type}' {record.status}"
            + (f" for teammate '{record.teammate_name}'" if record.teammate_name else "")
            + "."
        ),
        metadata={
            "agent_id": record.agent_id,
            "status": record.status,
            "tool_use_count": record.tool_use_count,
        },
    )


def subagent_progress_label(record: AgentRunRecord) -> str:
    base = (record.teammate_name or record.agent_type or "subagent").strip() or "subagent"
    agent_id = (record.agent_id or "").strip()
    return f"{base}:{agent_id}" if agent_id else base


def subagent_progress_sender(record: AgentRunRecord) -> str:
    return f"Subagent({subagent_progress_label(record)})"


def track_subagent_message(
    record: AgentRunRecord,
    message: object,
    history: List[Any],
    assistant_messages: List[AssistantMessage],
    tool_use_count: int,
) -> Tuple[int, List[Tuple[str, str]]]:
    updates: List[Tuple[str, str]] = []
    msg_type = getattr(message, "type", "")
    if msg_type in ("assistant", "user"):
        history.append(message)  # type: ignore[arg-type]

    if msg_type == "assistant":
        if isinstance(message, AssistantMessage):
            tool_use_count += _count_tool_uses(message)
            text = extract_text(message).strip()
            if text:
                _write_task_output(record.output_file, text, append=True)
        updates = extract_progress_updates(message, record=record)
        assistant_messages.append(message)  # type: ignore[arg-type]

    return tool_use_count, updates


def extract_progress_updates(
    message: object, *, record: AgentRunRecord
) -> List[tuple[str, str]]:
    from ripperdoc.tools.agent._agent_utils import get_block_attr

    msg_content = getattr(message, "message", None)
    blocks = getattr(msg_content, "content", []) if msg_content else []
    if not isinstance(blocks, list):
        return []

    sender = subagent_progress_sender(record)
    updates: List[tuple[str, str]] = []
    for block in blocks:
        block_type = get_block_attr(block, "type") or ""
        if block_type == "tool_use":
            tool_name = get_block_attr(block, "name") or "unknown tool"
            block_input = get_block_attr(block, "input")
            summary = summarize_tool_input(block_input)
            label = f"requesting {tool_name}"
            if summary:
                label += f" — {summary}"
            updates.append((sender, label))
        elif block_type == "text":
            text_val = get_block_attr(block, "text") or ""
            if text_val:
                snippet = str(text_val).strip()
                if snippet:
                    short = snippet if len(snippet) <= 200 else snippet[:197] + "..."
                    updates.append((sender, short))
    return updates


async def run_subagent_foreground(
    *,
    record: AgentRunRecord,
    subagent_context: QueryContext,
    permission_checker: Any,
    parent_abort_signal: Optional[asyncio.Event],
    notification_queue: Optional[PendingMessageQueue],
    parent_tool_use_id: Optional[str],
) -> AsyncGenerator[ToolProgress, None]:
    assistant_messages: List[AssistantMessage] = []
    tool_use_count = 0
    finalize_status = "running"
    finalize_error: Optional[str] = None
    finalize_result_text: Optional[str] = None
    handed_off_to_background = False
    try:
        async for message in query(
            record.history,  # type: ignore[arg-type]
            record.system_prompt,
            {},
            subagent_context,
            permission_checker,
        ):
            if parent_abort_signal is not None and parent_abort_signal.is_set():
                handed_off_to_background = handoff_foreground_run_to_background(
                    record=record,
                    subagent_context=subagent_context,
                    permission_checker=permission_checker,
                    notification_queue=notification_queue,
                    parent_tool_use_id=parent_tool_use_id,
                )
                if handed_off_to_background:
                    finalize_result_text = "Subagent moved to background after interrupt."
                    break
            msg_type = getattr(message, "type", "")
            if msg_type == "progress":
                continue

            tool_use_count, updates = track_subagent_message(
                record,
                message,
                record.history,
                assistant_messages,
                tool_use_count,
            )
            if isinstance(message, UserMessage):
                shutdown_approval = extract_approved_shutdown_response(
                    record.history,
                    message,
                )
                if shutdown_approval is not None:
                    finalize_status = "shutdown"
                    finalize_error = (
                        shutdown_approval.get("content")
                        or "Approved shutdown_response sent to team lead."
                    ).strip()
                    finalize_result_text = (
                        "Subagent exited after approved shutdown_response"
                        + (
                            f" (request_id={shutdown_approval.get('request_id', '')})."
                            if shutdown_approval.get("request_id")
                            else "."
                        )
                    )
                    yield ToolProgress(
                        content=(
                            f"Shutdown approved for subagent '{record.agent_id}', exiting run."
                        ),
                        progress_sender=subagent_progress_sender(record),
                    )
                    break
            for sender, text in updates:
                yield ToolProgress(content=text, progress_sender=sender)

            if msg_type in ("assistant", "user"):
                message_with_parent = (
                    message.model_copy(update={"parent_tool_use_id": parent_tool_use_id})
                    if parent_tool_use_id
                    else message
                )
                yield ToolProgress(
                    content=message_with_parent,
                    is_subagent_message=True,
                    progress_sender=subagent_progress_sender(record),
                )
    except asyncio.CancelledError:
        if parent_abort_signal is not None and parent_abort_signal.is_set():
            handed_off_to_background = handoff_foreground_run_to_background(
                record=record,
                subagent_context=subagent_context,
                permission_checker=permission_checker,
                notification_queue=notification_queue,
                parent_tool_use_id=parent_tool_use_id,
            )
            if handed_off_to_background:
                finalize_result_text = "Subagent moved to background after interrupt."
                return
        finalize_status = "cancelled"
        finalize_error = "Subagent run was cancelled."
        raise
    except Exception as exc:
        finalize_status = "failed"
        finalize_error = str(exc)
        logger.warning(
            "[task_tool] Subagent foreground run failed: %s: %s",
            type(exc).__name__,
            exc,
            extra={"agent_id": record.agent_id, "team_name": record.team_name},
        )
    finally:
        if not handed_off_to_background:
            if finalize_status == "running":
                finalize_status = "completed"
            finalize_record_from_messages(
                record,
                assistant_messages=assistant_messages,
                tool_use_count=tool_use_count,
                status=finalize_status,
                error=finalize_error,
                result_text=finalize_result_text,
            )


async def run_subagent_background(
    record: AgentRunRecord,
    subagent_context: QueryContext,
    permission_checker: Any,
    *,
    notification_queue: Optional[PendingMessageQueue] = None,
    parent_tool_use_id: Optional[str] = None,
) -> None:
    assistant_messages: List[AssistantMessage] = []
    tool_use_count = 0
    finalize_status = "running"
    finalize_error: Optional[str] = None
    finalize_result_text: Optional[str] = None
    try:
        async for message in query(
            record.history,  # type: ignore[arg-type]
            record.system_prompt,
            {},
            subagent_context,
            permission_checker,
        ):
            if getattr(message, "type", "") == "progress":
                continue

            tool_use_count, _ = track_subagent_message(
                record,
                message,
                record.history,
                assistant_messages,
                tool_use_count,
            )
            if isinstance(message, UserMessage):
                shutdown_approval = extract_approved_shutdown_response(
                    record.history,
                    message,
                )
                if shutdown_approval is not None:
                    finalize_status = "shutdown"
                    finalize_error = (
                        shutdown_approval.get("content")
                        or "Approved shutdown_response sent to team lead."
                    ).strip()
                    finalize_result_text = (
                        "Subagent exited after approved shutdown_response"
                        + (
                            f" (request_id={shutdown_approval.get('request_id', '')})."
                            if shutdown_approval.get("request_id")
                            else "."
                        )
                    )
                    break
    except asyncio.CancelledError:
        finalize_status = "cancelled"
        finalize_error = "Subagent run was cancelled."
        raise
    except Exception as exc:
        finalize_status = "failed"
        finalize_error = str(exc)
        logger.warning(
            "[task_tool] Subagent background run failed: %s: %s",
            type(exc).__name__,
            exc,
            extra={"agent_id": record.agent_id, "team_name": record.team_name},
        )
    finally:
        if finalize_status == "running":
            finalize_status = "completed"
        finalize_record_from_messages(
            record,
            assistant_messages=assistant_messages,
            tool_use_count=tool_use_count,
            status=finalize_status,
            error=finalize_error,
            result_text=finalize_result_text,
        )
        record.task = None
        enqueue_background_completion_notification(
            record=record,
            queue=notification_queue,
            parent_tool_use_id=parent_tool_use_id,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _count_tool_uses(message: AssistantMessage) -> int:
    """Count tool_use blocks in an AssistantMessage."""
    content = message.message.content
    if not isinstance(content, list):
        return 0
    from ripperdoc.tools.agent._agent_utils import get_block_attr

    return sum(1 for block in content if get_block_attr(block, "type") == "tool_use")
