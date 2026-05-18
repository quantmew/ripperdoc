from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import uuid4

from ripperdoc.core.hooks.config import HooksConfig
from ripperdoc.core.tool import Tool
from ripperdoc.utils.collaboration.teams import (
    TeamMember,
    send_team_message,
    set_team_member_active,
    upsert_team_member,
)
from ripperdoc.utils.collaboration.teammate_state import (
    IdleReason,
    InProcessTeammateState,
    inject_user_message,
    set_teammate_active,
    set_teammate_idle,
)
from ripperdoc.utils.filesystem.config_paths import user_config_dir
from ripperdoc.utils.log import get_logger
from ripperdoc.utils.messaging.types import ConversationMessage

from ripperdoc.tools.agent._constants import DEFAULT_AGENT_RUN_TTL_SEC

logger = get_logger()
MessageType = ConversationMessage


@dataclass
class AgentRunRecord:
    """In-memory record for a subagent run (foreground or background).

    This record tracks both the run state and integrates with the teammate
    state management system for team coordination.
    """

    agent_id: str
    agent_type: str
    tools: List[Tool[Any, Any]]
    system_prompt: str
    history: List[MessageType]
    missing_tools: List[str]
    model_used: Optional[str]
    start_time: float
    duration_ms: float = 0.0
    tool_use_count: int = 0
    total_tokens: int = 0
    usage: Optional[Dict[str, Any]] = None
    status: str = "running"
    result_text: Optional[str] = None
    error: Optional[str] = None
    task_description: Optional[str] = None
    task_prompt: Optional[str] = None
    output_file: Optional[str] = None
    task: Optional[asyncio.Task] = None
    is_background: bool = False
    completion_notified: bool = False
    hook_scopes: List[HooksConfig] = field(default_factory=list)
    team_name: Optional[str] = None
    teammate_name: Optional[str] = None
    isolation_mode: Optional[str] = None
    worktree_path: Optional[str] = None
    worktree_branch: Optional[str] = None
    worktree_name: Optional[str] = None
    worktree_repo_root: Optional[str] = None
    worktree_head_commit: Optional[str] = None
    worktree_hook_based: bool = False

    # Idle state management
    is_idle: bool = False
    pending_user_messages: List[str] = field(default_factory=list)
    on_idle_callbacks: List[Callable[[], None]] = field(default_factory=list)

    # Permission mode (inherited from team lead)
    permission_mode: str = "default"  # "default", "plan", "dontAsk", "bypassPermissions", "acceptEdits"
    max_turns: Optional[int] = None

    # Plan approval workflow
    awaiting_plan_approval: bool = False

    # Shutdown protocol
    shutdown_requested: bool = False
    shutdown_request_id: Optional[str] = None

    # Linked teammate state (for advanced coordination)
    teammate_state: Optional[InProcessTeammateState] = None


_AGENT_RUNS: Dict[str, AgentRunRecord] = {}
_AGENT_RUNS_LOCK = threading.Lock()


def _new_agent_id() -> str:
    return f"agent_{uuid4().hex[:8]}"


def _task_output_root() -> Path:
    base = user_config_dir()
    root = base / "tasks" / "outputs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _task_output_path(agent_id: str) -> str:
    safe_agent_id = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in agent_id)
    return str(_task_output_root() / f"{safe_agent_id}.log")


def _write_task_output(path: Optional[str], text: str, *, append: bool = True) -> None:
    if not path:
        return
    try:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with output_path.open(mode, encoding="utf-8") as handle:
            handle.write(text)
            if not text.endswith("\n"):
                handle.write("\n")
    except OSError as exc:
        logger.debug("[task_tool] Failed writing task output file: %s", exc)


def _register_agent_run(record: AgentRunRecord) -> None:
    with _AGENT_RUNS_LOCK:
        _AGENT_RUNS[record.agent_id] = record
    prune_agent_runs()


def _get_agent_run(agent_id: str) -> Optional[AgentRunRecord]:
    with _AGENT_RUNS_LOCK:
        return _AGENT_RUNS.get(agent_id)


def _snapshot_agent_run(record: AgentRunRecord) -> dict:
    duration_ms = (
        record.duration_ms
        if record.duration_ms
        else max((time.time() - record.start_time) * 1000.0, 0.0)
    )
    return {
        "id": record.agent_id,
        "agent_type": record.agent_type,
        "status": record.status,
        "duration_ms": duration_ms,
        "tool_use_count": record.tool_use_count,
        "total_tokens": record.total_tokens,
        "usage": record.usage,
        "missing_tools": list(record.missing_tools),
        "model_used": record.model_used,
        "result_text": record.result_text,
        "error": record.error,
        "task_description": record.task_description,
        "task_prompt": record.task_prompt,
        "output_file": record.output_file,
        "is_background": record.is_background,
        "team_name": record.team_name,
        "teammate_name": record.teammate_name,
        "isolation_mode": record.isolation_mode,
        "worktree_path": record.worktree_path,
        "worktree_branch": record.worktree_branch,
        "worktree_name": record.worktree_name,
        "worktree_repo_root": record.worktree_repo_root,
        "worktree_head_commit": record.worktree_head_commit,
        "worktree_hook_based": record.worktree_hook_based,
        # Idle state fields
        "is_idle": record.is_idle,
        "pending_messages": len(record.pending_user_messages),
        "permission_mode": record.permission_mode,
        "max_turns": record.max_turns,
        "shutdown_requested": record.shutdown_requested,
    }


def inject_user_message_to_teammate(
    agent_id: str,
    message: str,
) -> bool:
    """Inject a user message into a running teammate's pending queue.

    This allows the team lead to send messages to teammates that will be
    processed on their next polling cycle. If the teammate is idle, it will
    be woken up to process the message.

    Args:
        agent_id: The agent ID to inject the message into.
        message: The message content to inject.

    Returns:
        True if the message was successfully injected, False otherwise.
    """
    record = _get_agent_run(agent_id)
    if not record:
        logger.debug(
            "[task_tool] inject_user_message: agent %s not found",
            agent_id,
        )
        return False

    if record.status != "running":
        logger.debug(
            "[task_tool] inject_user_message: agent %s is not running (status=%s)",
            agent_id,
            record.status,
        )
        return False

    record.pending_user_messages.append(message)
    record.is_idle = False

    # Also sync with teammate_state if available
    if record.teammate_state:
        inject_user_message(record.teammate_state.id, message)

    logger.debug(
        "[task_tool] Injected message into %s's queue (depth=%d)",
        agent_id,
        len(record.pending_user_messages),
    )
    return True


def pop_pending_user_message_from_teammate(agent_id: str) -> Optional[str]:
    """Pop the next pending user message from a teammate's queue.

    Args:
        agent_id: The agent ID to pop the message from.

    Returns:
        The next message in the queue, or None if the queue is empty.
    """
    record = _get_agent_run(agent_id)
    if not record or not record.pending_user_messages:
        return None

    return record.pending_user_messages.pop(0)


def set_agent_idle_state(
    agent_id: str,
    is_idle: bool,
    *,
    idle_reason: Optional[IdleReason] = None,
    summary: Optional[str] = None,
) -> bool:
    """Set the idle state of an agent run.

    When setting to idle, this will also trigger idle callbacks and send
    an idle notification to the team lead.

    Args:
        agent_id: The agent ID to update.
        is_idle: Whether the agent should be marked as idle.
        idle_reason: The reason for going idle (required when is_idle=True).
        summary: Optional summary of what the agent accomplished.

    Returns:
        True if the state was updated, False if the agent was not found.
    """
    record = _get_agent_run(agent_id)
    if not record:
        return False

    if is_idle and not record.is_idle:
        record.is_idle = True

        # Execute idle callbacks
        for callback in record.on_idle_callbacks:
            try:
                callback()
            except Exception as exc:
                logger.warning(
                    "[task_tool] Idle callback failed for %s: %s: %s",
                    agent_id,
                    type(exc).__name__,
                    exc,
                )
        record.on_idle_callbacks = []

        # Sync with teammate_state
        if record.teammate_state and idle_reason:
            set_teammate_idle(
                record.teammate_state.id,
                idle_reason=idle_reason,
                summary=summary,
            )

        # Send idle notification to team lead
        if record.team_name:
            _send_idle_notification_to_team_lead(
                record,
                idle_reason=idle_reason or IdleReason.AVAILABLE,
                summary=summary,
            )

        logger.debug(
            "[task_tool] Agent %s is now idle (reason=%s)",
            agent_id,
            idle_reason.value if idle_reason else "unknown",
        )
    elif not is_idle and record.is_idle:
        record.is_idle = False
        if record.teammate_state:
            set_teammate_active(record.teammate_state.id)
        logger.debug("[task_tool] Agent %s is now active", agent_id)

    return True


def _send_idle_notification_to_team_lead(
    record: AgentRunRecord,
    *,
    idle_reason: IdleReason,
    summary: Optional[str] = None,
) -> None:
    """Send an idle notification message to the team lead."""
    if not record.team_name:
        return

    import json
    from datetime import datetime, timezone

    notification = {
        "type": "idle_notification",
        "from": record.teammate_name or record.agent_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "idleReason": idle_reason.value,
        "summary": summary,
        "agentId": record.agent_id,
        "completedStatus": record.status if record.status != "running" else None,
    }

    try:
        send_team_message(
            team_name=record.team_name,
            sender=record.teammate_name or record.agent_type,
            recipients=["team-lead"],
            message_type="idle_notification",
            content=json.dumps(notification, ensure_ascii=False),
            metadata={
                "idle_notification": True,
                "idle_reason": idle_reason.value,
            },
        )
    except Exception as exc:
        logger.warning(
            "[task_tool] Failed to send idle notification: %s: %s",
            type(exc).__name__,
            exc,
        )


def list_agent_runs() -> List[str]:
    """Return known subagent run ids."""
    prune_agent_runs()
    with _AGENT_RUNS_LOCK:
        return list(_AGENT_RUNS.keys())


def list_running_team_members(team_name: Optional[str] = None) -> List[str]:
    """Return teammate names that currently have a running execution state."""
    target = (team_name or "").strip()
    with _AGENT_RUNS_LOCK:
        names: Set[str] = set()
        for record in _AGENT_RUNS.values():
            if record.status != "running":
                continue
            if not record.teammate_name:
                continue
            if target and (record.team_name or "").strip() != target:
                continue
            if record.task is not None and record.task.done():
                continue
            names.add(record.teammate_name)
        return sorted(names)


def list_running_agent_worktree_paths() -> Set[str]:
    """Return worktree paths currently used by running subagents."""
    with _AGENT_RUNS_LOCK:
        paths: Set[str] = set()
        for record in _AGENT_RUNS.values():
            if record.status != "running":
                continue
            if not record.worktree_path:
                continue
            if record.task is not None and record.task.done():
                continue
            paths.add(record.worktree_path)
        return paths


def get_agent_run_snapshot(agent_id: str) -> Optional[dict]:
    """Return a snapshot of a subagent run by id."""
    record = _get_agent_run(agent_id)
    if not record:
        return None
    return _snapshot_agent_run(record)


async def wait_for_agent_run_snapshot(
    agent_id: str,
    *,
    timeout_ms: int = 30_000,
) -> Optional[dict]:
    """Wait for a running agent record up to timeout and return its snapshot.

    Returns None when the record does not exist.
    """
    record = _get_agent_run(agent_id)
    if not record:
        return None

    if record.task and not record.task.done():
        timeout_s = max(timeout_ms, 0) / 1000.0
        try:
            await asyncio.wait_for(asyncio.shield(record.task), timeout=timeout_s)
        except asyncio.TimeoutError:
            pass
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            record.status = "failed"
            record.error = str(exc)

    return _snapshot_agent_run(record)


def prune_agent_runs(max_age_seconds: Optional[float] = None) -> int:
    """Remove finished subagent runs older than the TTL."""
    ttl = DEFAULT_AGENT_RUN_TTL_SEC if max_age_seconds is None else max_age_seconds
    if ttl is None or ttl <= 0:
        return 0
    now = time.time()
    removed = 0
    with _AGENT_RUNS_LOCK:
        for agent_id, record in list(_AGENT_RUNS.items()):
            if record.status == "running" or record.task:
                continue
            age = now - record.start_time
            if age > ttl:
                _AGENT_RUNS.pop(agent_id, None)
                removed += 1
    return removed


def _set_team_member_active_state(
    team_name: Optional[str],
    teammate_name: Optional[str],
    active: bool,
    *,
    default_agent_type: str = "general-purpose",
) -> None:
    if not team_name or not teammate_name:
        return
    try:
        set_team_member_active(
            team_name=team_name,
            member_name=teammate_name,
            active=active,
            default_agent_type=default_agent_type,
        )
    except (ValueError, OSError, RuntimeError, KeyError, TypeError):
        # Best-effort lifecycle tracking for team members.
        logger.debug(
            "[task_tool] Failed to update teammate active state",
            extra={"team_name": team_name, "teammate_name": teammate_name, "active": active},
        )


async def cancel_agent_run(agent_id: str) -> bool:
    """Cancel a running subagent, if possible."""
    record = _get_agent_run(agent_id)
    if not record or not record.task or record.task.done():
        return False
    record.task.cancel()
    try:
        await record.task
    except asyncio.CancelledError:
        pass
    record.status = "cancelled"
    _set_team_member_active_state(record.team_name, record.teammate_name, False)
    record.error = record.error or "Cancelled by user."
    record.duration_ms = (time.time() - record.start_time) * 1000
    record.task = None
    return True
