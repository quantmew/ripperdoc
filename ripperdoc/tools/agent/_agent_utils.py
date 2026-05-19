"""Utility functions extracted from TaskTool for agent-related operations."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, cast

from ripperdoc.core.agents import AgentDefinition, resolve_agent_tools
from ripperdoc.core.tool import Tool
from ripperdoc.utils.log import get_logger
from ripperdoc.utils.messaging.messages import (
    AssistantMessage,
    ConversationMessage,
    UserMessage,
)

from ripperdoc.tools.agent._constants import (
    ALL_AGENT_DISALLOWED_TOOLS,
    CUSTOM_AGENT_DISALLOWED_TOOLS,
)

logger = get_logger()

MessageType = ConversationMessage


# ---------------------------------------------------------------------------
# Block / message attribute helpers
# ---------------------------------------------------------------------------


def get_block_attr(block: Any, attr_name: str, default: Any = None) -> Any:
    """Get attribute from block, supporting both object and dict access."""
    value = getattr(block, attr_name, None)
    if value is None and isinstance(block, dict):
        return block.get(attr_name, default)
    return value if value is not None else default


def normalize_tool_input(raw_input: Any) -> Dict[str, Any]:
    """Normalize tool input to a plain dictionary."""
    if hasattr(raw_input, "model_dump"):
        model_dump = raw_input.model_dump()
        if isinstance(model_dump, dict):
            return cast(Dict[str, Any], model_dump)
        return {}
    if hasattr(raw_input, "dict"):
        dict_method = getattr(raw_input, "dict")
        if callable(dict_method):
            as_dict = dict_method()
            if isinstance(as_dict, dict):
                return cast(Dict[str, Any], as_dict)
            return {}
    if isinstance(raw_input, dict):
        return dict(raw_input)
    return {}


def extract_tool_result_ids(
    message: UserMessage,
) -> List[str]:
    payload = getattr(message, "message", None)
    content = getattr(payload, "content", None) if payload is not None else None
    if not isinstance(content, list):
        return []

    tool_result_ids: List[str] = []
    for block in content:
        block_type = get_block_attr(block, "type") or ""
        if block_type != "tool_result":
            continue
        tool_use_id = get_block_attr(block, "tool_use_id") or ""
        if isinstance(tool_use_id, str) and tool_use_id.strip():
            tool_result_ids.append(tool_use_id.strip())
    return tool_result_ids


def lookup_tool_use_input_by_id(
    history: Sequence[MessageType],
    tool_use_id: str,
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    if not tool_use_id:
        return None, None

    for item in reversed(history):
        if getattr(item, "type", "") != "assistant":
            continue

        payload = getattr(item, "message", None)
        content = getattr(payload, "content", None) if payload is not None else None
        if not isinstance(content, list):
            continue

        for block in content:
            block_type = get_block_attr(block, "type") or ""
            if block_type != "tool_use":
                continue

            block_id = get_block_attr(block, "id") or get_block_attr(
                block, "tool_use_id"
            )
            if str(block_id or "").strip() != tool_use_id:
                continue

            tool_name = get_block_attr(block, "name")
            raw_input = get_block_attr(block, "input")
            parsed_input = normalize_tool_input(raw_input)
            return (str(tool_name) if tool_name else None), parsed_input

    return None, None


def extract_approved_shutdown_response(
    history: Sequence[MessageType],
    message: UserMessage,
) -> Optional[Dict[str, str]]:
    for tool_use_id in extract_tool_result_ids(message):
        tool_name, tool_input = lookup_tool_use_input_by_id(history, tool_use_id)
        if tool_name != "SendMessage":
            continue
        if tool_input is None:
            continue
        if str(tool_input.get("type") or "").strip() != "shutdown_response":
            continue
        if not bool(tool_input.get("approve")):
            continue
        request_id = str(tool_input.get("request_id") or "").strip()
        reason = str(tool_input.get("content") or "").strip()
        return {
            "request_id": request_id,
            "content": reason,
        }
    return None


def coerce_agent_tools(tools: List[object]) -> List[Tool[Any, Any]]:
    from ripperdoc.core.tool import Tool as ToolBase

    return [tool for tool in tools if isinstance(tool, ToolBase)]


def extract_text(message: AssistantMessage) -> str:
    content = message.message.content
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    parts = []
    for block in content:
        text = get_block_attr(block, "text")
        if text:
            parts.append(str(text))
    return "\n".join(parts)


def count_tool_uses(message: AssistantMessage) -> int:
    content = message.message.content
    if not isinstance(content, list):
        return 0
    return sum(1 for block in content if get_block_attr(block, "type") == "tool_use")


def summarize_tool_input(inp: Any) -> str:
    """Generate a short human-readable summary of a tool_use input."""
    if not inp or not isinstance(inp, (dict, Dict)):
        return ""

    pieces: List[str] = []
    # Prioritize common keys
    for key in ("command", "file_path", "path", "glob", "pattern", "description", "prompt"):
        if key in inp and inp[key]:
            val = str(inp[key])
            short = val if len(val) <= 80 else val[:77] + "..."
            pieces.append(f"{key}={short}")

    # Include range info if present
    start = inp.get("start_line") or inp.get("offset")
    end = inp.get("end_line") or inp.get("limit")
    if start is not None or end is not None:
        pieces.append(f"range={start or 0}-{end or '…'}")

    if not pieces:
        # Fallback to truncated dict representation
        try:
            serialized = json.dumps(inp, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            logger.warning(
                "[task_tool] Failed to serialize tool_use input: %s: %s",
                type(exc).__name__,
                exc,
                extra={"tool_use_input": str(inp)[:200]},
            )
            serialized = str(inp)
        return serialized if len(serialized) <= 120 else serialized[:117] + "..."

    return ", ".join(pieces)


def coerce_parent_history(messages: Optional[Sequence[object]]) -> List[MessageType]:
    if not messages:
        return []
    history: List[MessageType] = []
    for msg in messages:
        msg_type = getattr(msg, "type", None)
        if msg_type in ("user", "assistant"):
            history.append(msg)  # type: ignore[arg-type]
    return history


# ---------------------------------------------------------------------------
# Tool filtering
# ---------------------------------------------------------------------------


def filter_tools_for_agent(
    agent_def: AgentDefinition,
    available_tools: Sequence[object],
    task_tool_name: str,
) -> Tuple[List[Tool], List[str]]:
    """Resolve agent tools via *resolve_agent_tools* and additionally filter
    out any tools listed in ``disallowed_tools`` on the agent definition as
    well as the global disallowed-tool sets from constants.
    """
    resolved, missing = resolve_agent_tools(agent_def, available_tools, task_tool_name)

    # Build the full set of disallowed tool names
    disallowed: Set[str] = set(agent_def.disallowed_tools or [])
    disallowed.update(ALL_AGENT_DISALLOWED_TOOLS)
    disallowed.update(CUSTOM_AGENT_DISALLOWED_TOOLS)

    if disallowed:
        resolved = [t for t in resolved if getattr(t, "name", None) not in disallowed]

    # Ensure only Tool instances pass through
    coerced = coerce_agent_tools(resolved)
    return coerced, missing
