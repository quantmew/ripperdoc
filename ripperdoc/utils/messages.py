"""Message handling and formatting for Ripperdoc.

This module provides utilities for creating and normalizing messages
for communication with AI models.
"""

import json
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict
from uuid import uuid4
from enum import Enum
from ripperdoc.utils.log import get_logger

logger = get_logger()


class MessageRole(str, Enum):
    """Message roles in a conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class MessageContent(BaseModel):
    """Content of a message."""

    type: str
    text: Optional[str] = None
    # Some providers return tool_use IDs as "id", others as "tool_use_id"
    id: Optional[str] = None
    tool_use_id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[Dict[str, object]] = None
    is_error: Optional[bool] = None


def _content_block_to_api(block: MessageContent) -> Dict[str, Any]:
    """Convert a MessageContent block to API-ready dict for tool protocols."""
    block_type = getattr(block, "type", None)
    if block_type == "tool_use":
        return {
            "type": "tool_use",
            "id": getattr(block, "id", None) or getattr(block, "tool_use_id", "") or "",
            "name": getattr(block, "name", None) or "",
            "input": getattr(block, "input", None) or {},
        }
    if block_type == "tool_result":
        result: Dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": getattr(block, "tool_use_id", None) or getattr(block, "id", None) or "",
            "content": [
                {
                    "type": "text",
                    "text": getattr(block, "text", None) or getattr(block, "content", None) or "",
                }
            ],
        }
        if getattr(block, "is_error", None) is not None:
            result["is_error"] = block.is_error
        return result
    # Default to text block
    return {
        "type": "text",
        "text": getattr(block, "text", None) or getattr(block, "content", None) or str(block),
    }


def _content_block_to_openai(block: MessageContent) -> Dict[str, Any]:
    """Convert a MessageContent block to OpenAI chat-completions tool call format."""
    block_type = getattr(block, "type", None)
    if block_type == "tool_use":
        import json

        args = getattr(block, "input", None) or {}
        try:
            args_str = json.dumps(args)
        except Exception:
            logger.exception("[_content_block_to_openai] Failed to serialize tool arguments")
            args_str = "{}"
        tool_call_id = (
            getattr(block, "id", None) or getattr(block, "tool_use_id", "") or str(uuid4())
        )
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": getattr(block, "name", None) or "",
                        "arguments": args_str,
                    },
                }
            ],
        }
    if block_type == "tool_result":
        # OpenAI expects role=tool messages after a tool call
        tool_call_id = getattr(block, "tool_use_id", None) or getattr(block, "id", None) or ""
        if not tool_call_id:
            logger.debug("[_content_block_to_openai] Skipping tool_result without tool_call_id")
            return {}
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": getattr(block, "text", None) or getattr(block, "content", None) or "",
        }
    # Fallback text message
    return {
        "role": "assistant",
        "content": getattr(block, "text", None) or getattr(block, "content", None) or str(block),
    }


class Message(BaseModel):
    """A message in a conversation."""

    role: MessageRole
    content: Union[str, List[MessageContent]]
    uuid: str = ""

    def __init__(self, **data: object) -> None:
        if "uuid" not in data or not data["uuid"]:
            data["uuid"] = str(uuid4())
        super().__init__(**data)


class UserMessage(BaseModel):
    """User message with tool results."""

    type: str = "user"
    message: Message
    uuid: str = ""
    tool_use_result: Optional[object] = None

    def __init__(self, **data: object) -> None:
        if "uuid" not in data or not data["uuid"]:
            data["uuid"] = str(uuid4())
        super().__init__(**data)


class AssistantMessage(BaseModel):
    """Assistant message with metadata."""

    type: str = "assistant"
    message: Message
    uuid: str = ""
    cost_usd: float = 0.0
    duration_ms: float = 0.0
    is_api_error_message: bool = False

    def __init__(self, **data: object) -> None:
        if "uuid" not in data or not data["uuid"]:
            data["uuid"] = str(uuid4())
        super().__init__(**data)


class ProgressMessage(BaseModel):
    """Progress message during tool execution."""

    type: str = "progress"
    uuid: str = ""
    tool_use_id: str
    content: Any
    normalized_messages: List[Message] = []
    sibling_tool_use_ids: set[str] = set()
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data: object) -> None:
        if "uuid" not in data or not data["uuid"]:
            data["uuid"] = str(uuid4())
        super().__init__(**data)


def create_user_message(
    content: Union[str, List[Dict[str, Any]]], tool_use_result: Optional[object] = None
) -> UserMessage:
    """Create a user message."""
    if isinstance(content, str):
        message_content: Union[str, List[MessageContent]] = content
    else:
        message_content = [MessageContent(**item) for item in content]

    # Normalize tool_use_result to a dict if it's a Pydantic model
    if tool_use_result is not None:
        try:
            if hasattr(tool_use_result, "model_dump"):
                tool_use_result = tool_use_result.model_dump()
        except Exception:
            # Fallback: keep as-is if conversion fails
            logger.exception("[create_user_message] Failed to normalize tool_use_result")

    message = Message(role=MessageRole.USER, content=message_content)

    # Debug: record tool_result shaping
    if isinstance(message_content, list):
        tool_result_blocks = [
            blk for blk in message_content if getattr(blk, "type", None) == "tool_result"
        ]
        if tool_result_blocks:
            logger.debug(
                f"[create_user_message] tool_result blocks={len(tool_result_blocks)} "
                f"ids={[getattr(b, 'tool_use_id', None) for b in tool_result_blocks]}"
            )

    return UserMessage(message=message, tool_use_result=tool_use_result)


def create_assistant_message(
    content: Union[str, List[Dict[str, Any]]], cost_usd: float = 0.0, duration_ms: float = 0.0
) -> AssistantMessage:
    """Create an assistant message."""
    if isinstance(content, str):
        message_content: Union[str, List[MessageContent]] = content
    else:
        message_content = [MessageContent(**item) for item in content]

    message = Message(role=MessageRole.ASSISTANT, content=message_content)

    return AssistantMessage(message=message, cost_usd=cost_usd, duration_ms=duration_ms)


def create_progress_message(
    tool_use_id: str,
    sibling_tool_use_ids: set[str],
    content: Any,
    normalized_messages: Optional[List[Message]] = None,
) -> ProgressMessage:
    """Create a progress message."""
    return ProgressMessage(
        tool_use_id=tool_use_id,
        sibling_tool_use_ids=sibling_tool_use_ids,
        content=content,
        normalized_messages=normalized_messages or [],
    )


def normalize_messages_for_api(
    messages: List[Union[UserMessage, AssistantMessage, ProgressMessage]],
    protocol: str = "anthropic",
    tool_mode: str = "native",
) -> List[Dict[str, Any]]:
    """Normalize messages for API submission.

    Progress messages are filtered out as they are not sent to the API.
    """

    def _msg_type(msg: Any) -> Optional[str]:
        if hasattr(msg, "type"):
            return getattr(msg, "type", None)
        if isinstance(msg, dict):
            return msg.get("type")
        return None

    def _msg_content(msg: Any) -> Any:
        if hasattr(msg, "message"):
            return getattr(getattr(msg, "message", None), "content", None)
        if isinstance(msg, dict):
            message_payload = msg.get("message")
            if isinstance(message_payload, dict):
                return message_payload.get("content")
            if "content" in msg:
                return msg.get("content")
        return None

    def _block_type(block: Any) -> Optional[str]:
        if hasattr(block, "type"):
            return getattr(block, "type", None)
        if isinstance(block, dict):
            return block.get("type")
        return None

    def _block_attr(block: Any, attr: str, default: Any = None) -> Any:
        if hasattr(block, attr):
            return getattr(block, attr, default)
        if isinstance(block, dict):
            return block.get(attr, default)
        return default

    def _flatten_blocks_to_text(blocks: List[Any]) -> str:
        parts: List[str] = []
        for blk in blocks:
            btype = _block_type(blk)
            if btype == "text":
                text = _block_attr(blk, "text") or _block_attr(blk, "content") or ""
                if text:
                    parts.append(str(text))
            elif btype == "tool_result":
                text = _block_attr(blk, "text") or _block_attr(blk, "content") or ""
                tool_id = _block_attr(blk, "tool_use_id") or _block_attr(blk, "id")
                prefix = "Tool error" if _block_attr(blk, "is_error") else "Tool result"
                label = f"{prefix}{f' ({tool_id})' if tool_id else ''}"
                parts.append(f"{label}: {text}" if text else label)
            elif btype == "tool_use":
                name = _block_attr(blk, "name") or ""
                input_data = _block_attr(blk, "input")
                input_preview = ""
                if input_data not in (None, {}):
                    try:
                        input_preview = json.dumps(input_data)
                    except Exception:
                        input_preview = str(input_data)
                tool_id = _block_attr(blk, "tool_use_id") or _block_attr(blk, "id")
                desc = "Tool call"
                if name:
                    desc += f" {name}"
                if tool_id:
                    desc += f" ({tool_id})"
                if input_preview:
                    desc += f": {input_preview}"
                parts.append(desc)
            else:
                text = _block_attr(blk, "text") or _block_attr(blk, "content") or ""
                if text:
                    parts.append(str(text))
        return "\n".join(p for p in parts if p)

    effective_tool_mode = (tool_mode or "native").lower()
    if effective_tool_mode not in {"native", "text"}:
        effective_tool_mode = "native"

    normalized: List[Dict[str, Any]] = []
    tool_results_seen = 0
    tool_uses_seen = 0

    # Precompute tool_result positions so we can drop dangling tool_calls that
    # lack a following tool response (which OpenAI rejects).
    tool_result_positions: Dict[str, int] = {}
    skipped_tool_uses_no_result = 0
    skipped_tool_uses_no_id = 0
    if protocol == "openai":
        for idx, msg in enumerate(messages):
            if _msg_type(msg) != "user":
                continue
            content = _msg_content(msg)
            if not isinstance(content, list):
                continue
            for block in content:
                if getattr(block, "type", None) == "tool_result":
                    tool_id = getattr(block, "tool_use_id", None) or getattr(block, "id", None)
                    if tool_id and tool_id not in tool_result_positions:
                        tool_result_positions[tool_id] = idx

    for msg_index, msg in enumerate(messages):
        msg_type = _msg_type(msg)
        if msg_type == "progress":
            # Skip progress messages
            continue
        if msg_type is None:
            continue

        if msg_type == "user":
            user_content = _msg_content(msg)
            if isinstance(user_content, list):
                if protocol == "openai":
                    # Map each block to an OpenAI-style message
                    openai_msgs: List[Dict[str, Any]] = []
                    for block in user_content:
                        if getattr(block, "type", None) == "tool_result":
                            tool_results_seen += 1
                        mapped = _content_block_to_openai(block)
                        if mapped:
                            openai_msgs.append(mapped)
                    normalized.extend(openai_msgs)
                    continue
                api_blocks = []
                for block in user_content:
                    if getattr(block, "type", None) == "tool_result":
                        tool_results_seen += 1
                    api_blocks.append(_content_block_to_api(block))
                normalized.append({"role": "user", "content": api_blocks})
            else:
                normalized.append({"role": "user", "content": user_content})  # type: ignore
        elif msg_type == "assistant":
            asst_content = _msg_content(msg)
            if isinstance(asst_content, list):
                if protocol == "openai":
                    assistant_openai_msgs: List[Dict[str, Any]] = []
                    tool_calls: List[Dict[str, Any]] = []
                    text_parts: List[str] = []
                    for block in asst_content:
                        if getattr(block, "type", None) == "tool_use":
                            tool_uses_seen += 1
                            tool_id = getattr(block, "tool_use_id", None) or getattr(
                                block, "id", None
                            )
                            if not tool_id:
                                skipped_tool_uses_no_id += 1
                                continue
                            # Skip tool_use blocks that are not followed by a tool_result
                            result_pos = tool_result_positions.get(tool_id)
                            if result_pos is None:
                                skipped_tool_uses_no_result += 1
                                continue
                            if result_pos <= msg_index:
                                skipped_tool_uses_no_result += 1
                                continue
                            mapped = _content_block_to_openai(block)
                            if mapped.get("tool_calls"):
                                tool_calls.extend(mapped["tool_calls"])
                        elif getattr(block, "type", None) == "text":
                            text_parts.append(getattr(block, "text", "") or "")
                        else:
                            mapped = _content_block_to_openai(block)
                            if mapped:
                                assistant_openai_msgs.append(mapped)
                    if text_parts:
                        assistant_openai_msgs.append(
                            {"role": "assistant", "content": "\n".join(text_parts)}
                        )
                    if tool_calls:
                        assistant_openai_msgs.append(
                            {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": tool_calls,
                            }
                        )
                    normalized.extend(assistant_openai_msgs)
                    continue
                api_blocks = []
                for block in asst_content:
                    if getattr(block, "type", None) == "tool_use":
                        tool_uses_seen += 1
                    api_blocks.append(_content_block_to_api(block))
                normalized.append({"role": "assistant", "content": api_blocks})
            else:
                normalized.append({"role": "assistant", "content": asst_content})  # type: ignore

    logger.debug(
        f"[normalize_messages_for_api] protocol={protocol} tool_mode={effective_tool_mode} "
        f"input_msgs={len(messages)} normalized={len(normalized)} "
        f"tool_results_seen={tool_results_seen} tool_uses_seen={tool_uses_seen} "
        f"tool_result_positions={len(tool_result_positions)} "
        f"skipped_tool_uses_no_result={skipped_tool_uses_no_result} "
        f"skipped_tool_uses_no_id={skipped_tool_uses_no_id}"
    )
    return normalized


# Special interrupt messages
INTERRUPT_MESSAGE = "Request was interrupted by user."
INTERRUPT_MESSAGE_FOR_TOOL_USE = "Tool execution was interrupted by user."


def create_tool_result_stop_message(tool_use_id: str) -> Dict[str, Any]:
    """Create a tool result message for interruption."""
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "text": INTERRUPT_MESSAGE_FOR_TOOL_USE,
        "is_error": True,
    }
