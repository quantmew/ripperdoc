"""In-process teammate state management.

This module provides state tracking for in-process teammates (subagents running
within the same process as part of a team), including idle state management,
message injection, shutdown protocol, and idle notifications.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from ripperdoc.utils.log import get_logger
from ripperdoc.utils.teams import send_team_message

logger = get_logger()


class IdleReason(str, Enum):
    """Reason why a teammate entered idle state."""

    AVAILABLE = "available"  # Normal completion, ready for more work
    INTERRUPTED = "interrupted"  # Work was interrupted (e.g., Escape pressed)
    FAILED = "failed"  # Work failed with an error
    SHUTDOWN = "shutdown"  # Shutdown was requested/approved


class TeammateStatus(str, Enum):
    """Status of an in-process teammate task."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    KILLED = "killed"
    SHUTDOWN = "shutdown"


@dataclass
class TeammateIdentity:
    """Identity information for an in-process teammate."""

    agent_id: str
    agent_name: str
    team_name: str
    color: str = "#888888"
    plan_mode_required: bool = False
    parent_session_id: Optional[str] = None


@dataclass
class IdleNotification:
    """Notification sent when a teammate goes idle."""

    type: str = "idle_notification"
    from_teammate: str = ""
    timestamp: str = ""
    idle_reason: Optional[str] = None
    summary: Optional[str] = None
    completed_task_id: Optional[str] = None
    completed_status: Optional[str] = None
    failure_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "from": self.from_teammate,
            "timestamp": self.timestamp,
            "idleReason": self.idle_reason,
            "summary": self.summary,
            "completedTaskId": self.completed_task_id,
            "completedStatus": self.completed_status,
            "failureReason": self.failure_reason,
        }


@dataclass
class InProcessTeammateState:
    """State tracking for an in-process teammate."""

    id: str
    identity: TeammateIdentity
    status: TeammateStatus = TeammateStatus.RUNNING
    prompt: str = ""
    model: Optional[str] = None

    # Idle state management
    is_idle: bool = False
    idle_since: Optional[float] = None
    on_idle_callbacks: List[Callable[[], None]] = field(default_factory=list)

    # Permission and mode
    permission_mode: str = "default"  # "default", "plan", "dontAsk", "bypassPermissions", "acceptEdits"
    awaiting_plan_approval: bool = False

    # Shutdown protocol
    shutdown_requested: bool = False
    shutdown_request_id: Optional[str] = None

    # Message queue for injecting user messages into running teammates
    pending_user_messages: List[str] = field(default_factory=list)

    # Progress tracking
    messages: List[Any] = field(default_factory=list)
    last_reported_tool_count: int = 0
    last_reported_token_count: int = 0
    in_progress_tool_use_ids: set = field(default_factory=set)

    # Timing
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    total_paused_ms: float = 0.0

    # Abort control
    abort_controller: Optional[asyncio.Event] = None

    # UI helpers
    spinner_verb: str = "working"
    past_tense_verb: str = "worked"

    # Result
    result_text: Optional[str] = None
    error: Optional[str] = None

    # Cleanup
    unregister_cleanup: Optional[Callable[[], None]] = None

    def to_snapshot(self) -> Dict[str, Any]:
        """Return a snapshot of the current state for logging/debugging."""
        return {
            "id": self.id,
            "identity": {
                "agent_id": self.identity.agent_id,
                "agent_name": self.identity.agent_name,
                "team_name": self.identity.team_name,
            },
            "status": self.status.value,
            "is_idle": self.is_idle,
            "shutdown_requested": self.shutdown_requested,
            "pending_messages_count": len(self.pending_user_messages),
            "tool_count": self.last_reported_tool_count,
            "token_count": self.last_reported_token_count,
            "permission_mode": self.permission_mode,
        }


# Global state storage for in-process teammates
_TEAMMATE_STATES: Dict[str, InProcessTeammateState] = {}
_TEAMMATE_STATES_LOCK = threading.Lock()


def _new_teammate_id() -> str:
    """Generate a new unique teammate task ID."""
    return f"teammate_{uuid4().hex[:10]}"


def _get_teammate_color(name: str) -> str:
    """Generate a deterministic color for a teammate based on name."""
    import hashlib

    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:6]
    return f"#{digest}"


def create_teammate_state(
    *,
    agent_name: str,
    team_name: str,
    prompt: str,
    model: Optional[str] = None,
    plan_mode_required: bool = False,
    permission_mode: str = "default",
    parent_session_id: Optional[str] = None,
) -> InProcessTeammateState:
    """Create a new in-process teammate state."""
    agent_id = f"{agent_name}@{team_name}"
    color = _get_teammate_color(agent_id)

    state = InProcessTeammateState(
        id=_new_teammate_id(),
        identity=TeammateIdentity(
            agent_id=agent_id,
            agent_name=agent_name,
            team_name=team_name,
            color=color,
            plan_mode_required=plan_mode_required,
            parent_session_id=parent_session_id,
        ),
        prompt=prompt,
        model=model,
        permission_mode="plan" if plan_mode_required else permission_mode,
        spinner_verb=_get_spinner_verb(agent_name),
        abort_controller=asyncio.Event(),
    )

    with _TEAMMATE_STATES_LOCK:
        _TEAMMATE_STATES[state.id] = state

    logger.debug(
        "[teammate_state] Created teammate state: %s",
        state.to_snapshot(),
    )
    return state


def get_teammate_state(teammate_id: str) -> Optional[InProcessTeammateState]:
    """Get a teammate state by ID."""
    with _TEAMMATE_STATES_LOCK:
        return _TEAMMATE_STATES.get(teammate_id)


def get_teammate_state_by_agent_id(agent_id: str) -> Optional[InProcessTeammateState]:
    """Find a teammate state by agent ID (e.g., 'researcher@my-team')."""
    with _TEAMMATE_STATES_LOCK:
        for state in _TEAMMATE_STATES.values():
            if state.identity.agent_id == agent_id and state.status == TeammateStatus.RUNNING:
                return state
        # Fallback to any matching state
        for state in _TEAMMATE_STATES.values():
            if state.identity.agent_id == agent_id:
                return state
    return None


def list_running_teammates(team_name: Optional[str] = None) -> List[InProcessTeammateState]:
    """List all running teammate states, optionally filtered by team."""
    with _TEAMMATE_STATES_LOCK:
        result = []
        for state in _TEAMMATE_STATES.values():
            if state.status != TeammateStatus.RUNNING:
                continue
            if team_name and state.identity.team_name != team_name:
                continue
            result.append(state)
        return result


def set_teammate_idle(
    teammate_id: str,
    *,
    idle_reason: IdleReason = IdleReason.AVAILABLE,
    summary: Optional[str] = None,
    completed_task_id: Optional[str] = None,
    failure_reason: Optional[str] = None,
) -> bool:
    """Set a teammate to idle state and trigger idle callbacks.

    Returns True if the state was updated, False if not found.
    """
    state = get_teammate_state(teammate_id)
    if not state:
        return False

    state.is_idle = True
    state.idle_since = time.time()

    # Execute idle callbacks
    for callback in state.on_idle_callbacks:
        try:
            callback()
        except Exception as exc:
            logger.warning(
                "[teammate_state] Idle callback failed for %s: %s: %s",
                teammate_id,
                type(exc).__name__,
                exc,
            )
    state.on_idle_callbacks = []

    # Send idle notification to team lead
    _send_idle_notification(
        state,
        idle_reason=idle_reason,
        summary=summary,
        completed_task_id=completed_task_id,
        failure_reason=failure_reason,
    )

    logger.debug(
        "[teammate_state] Teammate %s is now idle (reason=%s)",
        state.identity.agent_name,
        idle_reason.value,
    )
    return True


def set_teammate_active(teammate_id: str) -> bool:
    """Set a teammate back to active (not idle) state."""
    state = get_teammate_state(teammate_id)
    if not state:
        return False

    state.is_idle = False
    state.idle_since = None
    logger.debug(
        "[teammate_state] Teammate %s is now active",
        state.identity.agent_name,
    )
    return True


def inject_user_message(teammate_id: str, message: str) -> bool:
    """Inject a user message into a running teammate's pending queue.

    The teammate will pick up this message on its next poll cycle.
    """
    state = get_teammate_state(teammate_id)
    if not state:
        return False

    if state.status not in (TeammateStatus.RUNNING,):
        logger.debug(
            "[teammate_state] Dropping message for teammate %s: status is %s",
            teammate_id,
            state.status.value,
        )
        return False

    state.pending_user_messages.append(message)
    # Wake up the teammate if idle
    state.is_idle = False
    if state.abort_controller:
        state.abort_controller.set()

    logger.debug(
        "[teammate_state] Injected message into %s's queue (depth=%d)",
        state.identity.agent_name,
        len(state.pending_user_messages),
    )
    return True


def pop_pending_user_message(teammate_id: str) -> Optional[str]:
    """Pop the next pending user message from a teammate's queue."""
    state = get_teammate_state(teammate_id)
    if not state or not state.pending_user_messages:
        return None

    return state.pending_user_messages.pop(0)


def request_teammate_shutdown(
    teammate_id: str,
    *,
    request_id: Optional[str] = None,
    reason: str = "Shutdown requested.",
) -> bool:
    """Request a teammate to shut down gracefully.

    Sets the shutdown_requested flag so the teammate can exit on its next cycle.
    """
    state = get_teammate_state(teammate_id)
    if not state:
        return False

    if state.shutdown_requested:
        logger.debug(
            "[teammate_state] Shutdown already requested for %s",
            state.identity.agent_name,
        )
        return True

    state.shutdown_requested = True
    state.shutdown_request_id = request_id or f"shutdown-{teammate_id}-{int(time.time())}"

    logger.debug(
        "[teammate_state] Shutdown requested for %s (request_id=%s)",
        state.identity.agent_name,
        state.shutdown_request_id,
    )
    return True


def add_idle_callback(teammate_id: str, callback: Callable[[], None]) -> bool:
    """Add a callback to be executed when the teammate goes idle."""
    state = get_teammate_state(teammate_id)
    if not state:
        return False

    state.on_idle_callbacks.append(callback)
    return True


def complete_teammate(
    teammate_id: str,
    *,
    status: TeammateStatus = TeammateStatus.COMPLETED,
    result_text: Optional[str] = None,
    error: Optional[str] = None,
) -> bool:
    """Mark a teammate as completed and clean up."""
    state = get_teammate_state(teammate_id)
    if not state:
        return False

    state.status = status
    state.end_time = time.time()
    state.result_text = result_text
    state.error = error
    state.is_idle = True
    state.pending_user_messages = []

    # Execute any remaining idle callbacks
    for callback in state.on_idle_callbacks:
        try:
            callback()
        except Exception:
            pass
    state.on_idle_callbacks = []

    # Run cleanup
    if state.unregister_cleanup:
        try:
            state.unregister_cleanup()
        except Exception as exc:
            logger.warning(
                "[teammate_state] Cleanup failed for %s: %s: %s",
                teammate_id,
                type(exc).__name__,
                exc,
            )

    logger.debug(
        "[teammate_state] Teammate %s completed with status %s",
        state.identity.agent_name,
        status.value,
    )
    return True


def kill_teammate(teammate_id: str) -> bool:
    """Forcefully kill a teammate."""
    state = get_teammate_state(teammate_id)
    if not state:
        return False

    # Trigger abort
    if state.abort_controller:
        state.abort_controller.set()

    return complete_teammate(
        teammate_id,
        status=TeammateStatus.KILLED,
        error="Teammate was killed.",
    )


def cleanup_teammate_state(teammate_id: str) -> bool:
    """Remove a teammate state from memory."""
    with _TEAMMATE_STATES_LOCK:
        if teammate_id in _TEAMMATE_STATES:
            del _TEAMMATE_STATES[teammate_id]
            return True
    return False


def _send_idle_notification(
    state: InProcessTeammateState,
    *,
    idle_reason: IdleReason,
    summary: Optional[str] = None,
    completed_task_id: Optional[str] = None,
    failure_reason: Optional[str] = None,
) -> None:
    """Send an idle notification to the team lead."""
    notification = IdleNotification(
        from_teammate=state.identity.agent_name,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        idle_reason=idle_reason.value,
        summary=summary,
        completed_task_id=completed_task_id,
        completed_status=state.status.value if state.status != TeammateStatus.RUNNING else None,
        failure_reason=failure_reason,
    )

    try:
        import json

        send_team_message(
            team_name=state.identity.team_name,
            sender=state.identity.agent_name,
            recipients=["team-lead"],
            message_type="idle_notification",
            content=json.dumps(notification.to_dict(), ensure_ascii=False),
            metadata={
                "idle_notification": True,
                "idle_reason": idle_reason.value,
                "agent_id": state.identity.agent_id,
            },
        )
    except Exception as exc:
        logger.warning(
            "[teammate_state] Failed to send idle notification: %s: %s",
            type(exc).__name__,
            exc,
        )


def _get_spinner_verb(agent_name: str) -> str:
    """Get a spinner verb for the agent based on common patterns."""
    name_lower = agent_name.lower()

    if "test" in name_lower:
        return "testing"
    elif "review" in name_lower:
        return "reviewing"
    elif "research" in name_lower or "explore" in name_lower:
        return "researching"
    elif "plan" in name_lower:
        return "planning"
    elif "build" in name_lower or "implement" in name_lower:
        return "building"
    elif "fix" in name_lower or "debug" in name_lower:
        return "fixing"
    elif "doc" in name_lower:
        return "documenting"
    else:
        return "working"


def has_running_teammates(team_name: Optional[str] = None) -> bool:
    """Check if there are any running teammates (optionally in a specific team)."""
    with _TEAMMATE_STATES_LOCK:
        for state in _TEAMMATE_STATES.values():
            if state.status != TeammateStatus.RUNNING:
                continue
            if team_name and state.identity.team_name != team_name:
                continue
            return True
    return False


def has_active_teammates(team_name: Optional[str] = None) -> bool:
    """Check if there are any non-idle running teammates."""
    with _TEAMMATE_STATES_LOCK:
        for state in _TEAMMATE_STATES.values():
            if state.status != TeammateStatus.RUNNING:
                continue
            if team_name and state.identity.team_name != team_name:
                continue
            if not state.is_idle:
                return True
    return False


async def wait_for_teammates_idle(team_name: str, timeout_ms: float = 30000) -> bool:
    """Wait for all teammates in a team to become idle.

    Returns True if all teammates became idle, False if timeout was reached.
    """
    deadline = time.time() + (timeout_ms / 1000)

    while time.time() < deadline:
        if not has_active_teammates(team_name):
            return True
        await asyncio.sleep(0.1)

    return False


def format_teammate_message(
    from_name: str,
    message: str,
    color: Optional[str] = None,
    summary: Optional[str] = None,
) -> str:
    """Format a message for delivery to a teammate.

    This creates a structured message format that teammates can parse.
    """
    parts = [f'<teammate_message from="{from_name}"']

    if color:
        parts.append(f' color="{color}"')

    if summary:
        # Escape quotes in summary
        escaped_summary = summary.replace('"', '\\"')
        parts.append(f' summary="{escaped_summary}"')

    parts.append(">\n")
    parts.append(message)
    parts.append("\n</teammate_message>")

    return "".join(parts)


__all__ = [
    "IdleReason",
    "TeammateStatus",
    "TeammateIdentity",
    "IdleNotification",
    "InProcessTeammateState",
    "create_teammate_state",
    "get_teammate_state",
    "get_teammate_state_by_agent_id",
    "list_running_teammates",
    "set_teammate_idle",
    "set_teammate_active",
    "inject_user_message",
    "pop_pending_user_message",
    "request_teammate_shutdown",
    "add_idle_callback",
    "complete_teammate",
    "kill_teammate",
    "cleanup_teammate_state",
    "has_running_teammates",
    "has_active_teammates",
    "wait_for_teammates_idle",
    "format_teammate_message",
]
