"""Task reminder injection — nudges models to clean up stale tasks.

Every turn we check:
1. How many assistant turns since last TaskCreate/TaskUpdate use
2. How many assistant turns since last task reminder

When both exceed the configured thresholds we inject a gentle
<system-reminder> nudge listing all existing tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from ripperdoc.utils.messaging.messages import (
    AttachmentMessage,
    ConversationMessage,
)
from ripperdoc.utils.messaging.attachments import create_task_reminder_attachment_message
from ripperdoc.utils.collaboration.tasks import (
    is_task_system_enabled,
    list_tasks,
)
from ripperdoc.utils.log import get_logger

logger = get_logger()

# --- Constants ---

TURNS_SINCE_LAST_TASK_TOOL = 10
TURNS_BETWEEN_REMINDERS = 10
TASK_TOOL_NAMES = {"TaskCreate", "TaskUpdate"}


@dataclass(frozen=True)
class TaskReminderDecision:
    """Resolved task-reminder injection decision for one model turn."""

    should_inject: bool


def _assistant_turns_since_last_task_tool(
    messages: Sequence[ConversationMessage],
) -> Tuple[int, bool]:
    """Count assistant turns since the most recent TaskCreate/TaskUpdate usage.

    Returns (assistant_turns, found_task_tool_usage).
    """
    assistant_count = 0
    found_task_tool = False
    for message in reversed(messages):
        msg_type = getattr(message, "type", None)
        if msg_type == "assistant":
            if _assistant_used_task_tools(message):
                found_task_tool = True
                break
            assistant_count += 1
    return assistant_count, found_task_tool


def _assistant_used_task_tools(message: ConversationMessage) -> bool:
    """Check if an assistant message contains a TaskCreate or TaskUpdate tool use."""
    content = getattr(getattr(message, "message", None), "content", None)
    if not isinstance(content, list):
        return False
    for block in content:
        if getattr(block, "type", None) != "tool_use":
            continue
        tool_name = str(getattr(block, "name", "") or "")
        if tool_name in TASK_TOOL_NAMES:
            return True
    return False


def _turns_since_last_reminder(messages: Sequence[ConversationMessage]) -> int:
    """Count assistant turns since the most recent task_reminder attachment."""
    count = 0
    for message in reversed(messages):
        msg_type = getattr(message, "type", None)
        if msg_type == "attachment":
            _type = getattr(getattr(message, "attachment", None), "type", None)
            if _type == "task_reminder":
                return count
        elif msg_type == "assistant":
            count += 1
    return count


def resolve_task_reminder_decision(
    messages: Sequence[ConversationMessage],
) -> TaskReminderDecision:
    """Apply task-reminder cadence.

    Inject when:
    - At least ``TURNS_SINCE_LAST_TASK_TOOL`` assistant turns have passed since
      the last TaskCreate/TaskUpdate use, AND
    - At least ``TURNS_BETWEEN_REMINDERS`` assistant turns have passed since the
      last task_reminder attachment.
    """
    turns_since_task, found = _assistant_turns_since_last_task_tool(messages)
    if found and turns_since_task < TURNS_SINCE_LAST_TASK_TOOL:
        return TaskReminderDecision(should_inject=False)

    turns_since_reminder = _turns_since_last_reminder(messages)
    if turns_since_reminder < TURNS_BETWEEN_REMINDERS:
        return TaskReminderDecision(should_inject=False)

    return TaskReminderDecision(should_inject=True)


def build_task_reminder_messages(
    messages: Sequence[ConversationMessage],
    task_list_id: Optional[str] = None,
) -> List[AttachmentMessage]:
    """Build task-reminder attachment messages for the next model turn.

    Returns an empty list if the reminder should not be injected or the task
    system is disabled.
    """
    if not is_task_system_enabled():
        return []

    decision = resolve_task_reminder_decision(messages)
    if not decision.should_inject:
        return []

    try:
        tasks = list_tasks(task_list_id=task_list_id)
    except Exception:
        logger.debug("[task_reminder] Failed to list tasks, skipping reminder")
        return []

    if not tasks:
        return []

    content = [
        {"id": str(t.id), "status": t.status, "subject": t.subject}
        for t in tasks
    ]
    return [create_task_reminder_attachment_message(content)]
