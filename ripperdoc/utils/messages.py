"""Message handling and formatting for Ripperdoc.

This module provides utilities for creating and normalizing messages
for communication with AI models.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field
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
    thinking: Optional[str] = None
    signature: Optional[str] = None
    data: Optional[str] = None
    # Some providers return tool_use IDs as "id", others as "tool_use_id"
    id: Optional[str] = None
    tool_use_id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[Dict[str, object]] = None
    is_error: Optional[bool] = None


def _content_block_to_api(block: MessageContent) -> Dict[str, Any]:
    """Convert a MessageContent block to API-ready dict for tool protocols."""
    block_type = getattr(block, "type", None)
    if block_type == "thinking":
        return {
            "type": "thinking",
            "thinking": getattr(block, "thinking", None) or getattr(block, "text", None) or "",
            "signature": getattr(block, "signature", None),
        }
    if block_type == "redacted_thinking":
        return {
            "type": "redacted_thinking",
            "data": getattr(block, "data", None) or getattr(block, "text", None) or "",
            "signature": getattr(block, "signature", None),
        }
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
        except (TypeError, ValueError) as exc:
            logger.warning(
                "[_content_block_to_openai] Failed to serialize tool arguments: %s: %s",
                type(exc).__name__,
                exc,
            )
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
    reasoning: Optional[Any] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
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
        except (AttributeError, TypeError, ValueError) as exc:
            # Fallback: keep as-is if conversion fails
            logger.warning(
                "[create_user_message] Failed to normalize tool_use_result: %s: %s",
                type(exc).__name__,
                exc,
            )

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
    content: Union[str, List[Dict[str, Any]]],
    cost_usd: float = 0.0,
    duration_ms: float = 0.0,
    reasoning: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> AssistantMessage:
    """Create an assistant message."""
    if isinstance(content, str):
        message_content: Union[str, List[MessageContent]] = content
    else:
        message_content = [MessageContent(**item) for item in content]

    message = Message(
        role=MessageRole.ASSISTANT,
        content=message_content,
        reasoning=reasoning,
        metadata=metadata or {},
    )

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


def _apply_deepseek_reasoning_content(
    normalized: List[Dict[str, Any]],
    is_new_turn: bool = False,
) -> List[Dict[str, Any]]:
    """Apply DeepSeek reasoning_content handling to normalized messages.

    DeepSeek thinking mode requires special handling for tool calls:
    1. During a tool call loop (same turn), reasoning_content MUST be preserved
       in assistant messages that contain tool_calls
    2. When a new user turn starts, we can optionally clear previous reasoning_content
       to save bandwidth (the API will ignore them anyway)

    According to DeepSeek docs, an assistant message with tool_calls should look like:
    {
        'role': 'assistant',
        'content': response.choices[0].message.content,
        'reasoning_content': response.choices[0].message.reasoning_content,
        'tool_calls': response.choices[0].message.tool_calls,
    }

    Args:
        normalized: The normalized messages list
        is_new_turn: If True, clear reasoning_content from historical messages
                     to save network bandwidth

    Returns:
        The processed messages list
    """
    if not normalized:
        return normalized

    # Find the last user message index to determine the current turn boundary
    last_user_idx = -1
    for idx in range(len(normalized) - 1, -1, -1):
        if normalized[idx].get("role") == "user":
            last_user_idx = idx
            break

    if is_new_turn and last_user_idx > 0:
        # Clear reasoning_content from messages before the last user message
        # This is optional but recommended by DeepSeek to save bandwidth
        for idx in range(last_user_idx):
            msg = normalized[idx]
            if msg.get("role") == "assistant" and "reasoning_content" in msg:
                # Set to None instead of deleting to match DeepSeek's example
                msg["reasoning_content"] = None

    # Validate: ensure all assistant messages with tool_calls have reasoning_content
    # within the current turn (after last_user_idx)
    for idx in range(max(0, last_user_idx), len(normalized)):
        msg = normalized[idx]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            if "reasoning_content" not in msg:
                # This is a problem - DeepSeek requires reasoning_content for tool_calls
                logger.warning(
                    f"[deepseek] Assistant message at index {idx} has tool_calls "
                    f"but missing reasoning_content - this may cause API errors"
                )

    return normalized


def normalize_messages_for_api(
    messages: List[Union[UserMessage, AssistantMessage, ProgressMessage]],
    protocol: str = "anthropic",
    tool_mode: str = "native",
    thinking_mode: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Normalize messages for API submission.

    Progress messages are filtered out as they are not sent to the API.

    For DeepSeek thinking mode, this function ensures reasoning_content is properly
    included in assistant messages that contain tool_calls, as required by the API.
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

    def _msg_metadata(msg: Any) -> Dict[str, Any]:
        message_obj = getattr(msg, "message", None)
        if message_obj is not None and hasattr(message_obj, "metadata"):
            try:
                meta = getattr(message_obj, "metadata", {}) or {}
                meta_dict = dict(meta) if isinstance(meta, dict) else {}
            except (TypeError, ValueError):
                meta_dict = {}
            reasoning_val = getattr(message_obj, "reasoning", None)
            if reasoning_val is not None and "reasoning" not in meta_dict:
                meta_dict["reasoning"] = reasoning_val
            return meta_dict
        if isinstance(msg, dict):
            message_payload = msg.get("message")
            if isinstance(message_payload, dict):
                meta = message_payload.get("metadata") or {}
                meta_dict = dict(meta) if isinstance(meta, dict) else {}
                if "reasoning" not in meta_dict and "reasoning" in message_payload:
                    meta_dict["reasoning"] = message_payload.get("reasoning")
                return meta_dict
        return {}

    effective_tool_mode = (tool_mode or "native").lower()
    if effective_tool_mode not in {"native", "text"}:
        effective_tool_mode = "native"

    normalized: List[Dict[str, Any]] = []
    tool_results_seen = 0
    tool_uses_seen = 0

    # Precompute tool_result positions so we can drop dangling tool_calls that
    # lack a following tool response (which OpenAI rejects).
    tool_result_positions: Dict[str, int] = {}
    # Precompute tool_use positions so we can drop dangling tool_results that
    # lack a preceding tool_call (which OpenAI also rejects).
    tool_use_positions: Dict[str, int] = {}
    skipped_tool_uses_no_result = 0
    skipped_tool_uses_no_id = 0
    skipped_tool_results_no_call = 0
    if protocol == "openai":
        for idx, msg in enumerate(messages):
            msg_type = _msg_type(msg)
            content = _msg_content(msg)
            if not isinstance(content, list):
                continue
            if msg_type == "user":
                for block in content:
                    if getattr(block, "type", None) == "tool_result":
                        tool_id = getattr(block, "tool_use_id", None) or getattr(block, "id", None)
                        if tool_id and tool_id not in tool_result_positions:
                            tool_result_positions[tool_id] = idx
            elif msg_type == "assistant":
                for block in content:
                    if getattr(block, "type", None) == "tool_use":
                        tool_id = getattr(block, "id", None) or getattr(block, "tool_use_id", None)
                        if tool_id and tool_id not in tool_use_positions:
                            tool_use_positions[tool_id] = idx

    for msg_index, msg in enumerate(messages):
        msg_type = _msg_type(msg)
        if msg_type == "progress":
            # Skip progress messages
            continue
        if msg_type is None:
            continue

        if msg_type == "user":
            user_content = _msg_content(msg)
            meta = _msg_metadata(msg)
            if isinstance(user_content, list):
                if protocol == "openai":
                    # Map each block to an OpenAI-style message
                    openai_msgs: List[Dict[str, Any]] = []
                    for block in user_content:
                        block_type = getattr(block, "type", None)
                        if block_type == "tool_result":
                            tool_results_seen += 1
                            # Skip tool_result blocks that lack a preceding tool_use
                            tool_id = getattr(block, "tool_use_id", None) or getattr(
                                block, "id", None
                            )
                            if not tool_id:
                                skipped_tool_results_no_call += 1
                                continue
                            call_pos = tool_use_positions.get(tool_id)
                            if call_pos is None or call_pos >= msg_index:
                                skipped_tool_results_no_call += 1
                                continue
                        mapped = _content_block_to_openai(block)
                        if mapped:
                            openai_msgs.append(mapped)
                    if meta and openai_msgs:
                        for candidate in openai_msgs:
                            for key in ("reasoning_content", "reasoning_details", "reasoning"):
                                if key in meta and meta[key] is not None:
                                    candidate[key] = meta[key]
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
            meta = _msg_metadata(msg)
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
                    if tool_calls:
                        # For DeepSeek thinking mode, we must include reasoning_content
                        # in the assistant message that contains tool_calls
                        tool_call_msg: Dict[str, Any] = {
                            "role": "assistant",
                            "content": "\n".join(text_parts) if text_parts else None,
                            "tool_calls": tool_calls,
                        }
                        # Add reasoning_content if present (required for DeepSeek thinking mode)
                        reasoning_content = meta.get("reasoning_content") if meta else None
                        if reasoning_content is not None:
                            tool_call_msg["reasoning_content"] = reasoning_content
                            logger.debug(
                                f"[normalize_messages_for_api] Added reasoning_content to "
                                f"tool_call message (len={len(str(reasoning_content))})"
                            )
                        elif thinking_mode == "deepseek":
                            logger.warning(
                                f"[normalize_messages_for_api] DeepSeek mode: assistant "
                                f"message with tool_calls but no reasoning_content in metadata. "
                                f"meta_keys={list(meta.keys()) if meta else []}"
                            )
                        assistant_openai_msgs.append(tool_call_msg)
                    elif text_parts:
                        assistant_openai_msgs.append(
                            {"role": "assistant", "content": "\n".join(text_parts)}
                        )
                    # For non-tool-call messages, add reasoning metadata to the last message
                    if meta and assistant_openai_msgs and not tool_calls:
                        for key in ("reasoning_content", "reasoning_details", "reasoning"):
                            if key in meta and meta[key] is not None:
                                assistant_openai_msgs[-1][key] = meta[key]
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
        f"thinking_mode={thinking_mode} "
        f"input_msgs={len(messages)} normalized={len(normalized)} "
        f"tool_results_seen={tool_results_seen} tool_uses_seen={tool_uses_seen} "
        f"tool_result_positions={len(tool_result_positions)} "
        f"tool_use_positions={len(tool_use_positions)} "
        f"skipped_tool_uses_no_result={skipped_tool_uses_no_result} "
        f"skipped_tool_uses_no_id={skipped_tool_uses_no_id} "
        f"skipped_tool_results_no_call={skipped_tool_results_no_call}"
    )

    # Apply DeepSeek-specific reasoning_content handling
    if thinking_mode == "deepseek":
        normalized = _apply_deepseek_reasoning_content(normalized, is_new_turn=False)

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
