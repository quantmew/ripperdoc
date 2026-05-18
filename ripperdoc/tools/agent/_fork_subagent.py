"""Fork subagent — inherits full parent conversation context.

When subagent_type is omitted, the agent forks the parent context instead of
starting fresh.

Forked agents:
- Inherit the parent's conversation history (all messages before the tool call)
- Use the parent's system prompt (not an agent-type-specific prompt)
- Get the same tools as the parent (minus the Task/Agent tool)
- Use FORK_PLACEHOLDER_RESULT for inherited tool outputs
"""

from __future__ import annotations

import os
from typing import Any, List, Optional, Sequence
from uuid import uuid4

from ripperdoc.utils.log import get_logger

logger = get_logger()

FORK_SUBAGENT_TYPE = "__fork__"
FORK_PLACEHOLDER_RESULT = (
    "[fork: result available in parent context — do not re-request]"
)


def is_fork_enabled() -> bool:
    """Check if fork subagent mode is enabled."""
    return os.getenv("RIPPERDOC_FORK_SUBAGENT", "").lower() in ("1", "true")


def build_fork_messages(
    directive: str,
    parent_messages: Sequence[Any],
) -> List[Any]:
    """Build messages for a forked subagent inheriting parent context.

    The fork child receives:
    1. A copy of the parent's assistant message (the one containing the Agent/Task tool call)
    2. A user message with placeholder tool_results + the fork directive

    This preserves the parent's cache prefix while giving the fork its own directive.
    """
    from ripperdoc.utils.messaging.messages import create_user_message
    from ripperdoc.utils.messaging.types import AssistantMessage

    # Find the last assistant message (the one with tool_use blocks)
    last_assistant = None
    for msg in reversed(list(parent_messages)):
        if getattr(msg, "type", None) == "assistant":
            last_assistant = msg
            break

    if last_assistant is None:
        # No assistant message to fork from — just return a simple directive
        return [
            create_user_message(
                content=[{"type": "text", "text": _build_child_directive(directive)}],
            ),
        ]

    # Clone the assistant message
    cloned_assistant = _clone_assistant_message(last_assistant)

    # Build placeholder tool_results for every tool_use in the assistant message
    tool_use_blocks = _extract_tool_use_blocks(cloned_assistant)
    tool_result_blocks = []
    for block in tool_use_blocks:
        block_id = _get_block_id(block)
        tool_result_blocks.append({
            "type": "tool_result",
            "tool_use_id": block_id,
            "content": [{"type": "text", "text": FORK_PLACEHOLDER_RESULT}],
        })

    # Append the directive after the tool results
    tool_result_blocks.append({
        "type": "text",
        "text": _build_child_directive(directive),
    })

    user_message = create_user_message(content=tool_result_blocks)
    return [cloned_assistant, user_message]


def _build_child_directive(directive: str) -> str:
    """Build the per-child directive appended to fork context."""
    return (
        f"[fork directive]\n"
        f"{directive}\n\n"
        f"You are a forked subagent. You inherited the parent's conversation context. "
        f"Execute the directive above autonomously."
    )


def _clone_assistant_message(msg: Any) -> Any:
    """Shallow-clone an assistant message with a new UUID."""
    from ripperdoc.utils.messaging.types import AssistantMessage

    if isinstance(msg, AssistantMessage):
        payload = getattr(msg, "message", None)
        if payload is not None:
            content = getattr(payload, "content", [])
            if isinstance(content, list):
                content = list(content)
        return AssistantMessage(
            uuid=str(uuid4()),
            message=payload,
        )
    # Fallback: return as-is
    return msg


def _extract_tool_use_blocks(msg: Any) -> List[Any]:
    """Extract tool_use blocks from an assistant message."""
    payload = getattr(msg, "message", None)
    content = getattr(payload, "content", None) if payload is not None else None
    if not isinstance(content, list):
        return []
    return [b for b in content if _get_block_type(b) == "tool_use"]


def _get_block_type(block: Any) -> str:
    return getattr(block, "type", "") or (block.get("type", "") if isinstance(block, dict) else "")


def _get_block_id(block: Any) -> str:
    bid = getattr(block, "id", None) or getattr(block, "tool_use_id", None)
    if bid is None and isinstance(block, dict):
        bid = block.get("id") or block.get("tool_use_id")
    return str(bid or "")
