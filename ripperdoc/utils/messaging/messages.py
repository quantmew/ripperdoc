"""Message handling and formatting for Ripperdoc.

This module provides utilities for creating and normalizing messages
for communication with AI models.
"""

import re
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator
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
    tool_name: Optional[str] = None
    input: Optional[Dict[str, object]] = None
    content: Optional[Any] = None
    is_error: Optional[bool] = None
    # Image/vision content fields
    source_type: Optional[str] = None  # "base64", "url", "file"
    media_type: Optional[str] = None  # "image/jpeg", "image/png", etc.
    image_data: Optional[str] = None  # base64-encoded image data or URL
    model_config = ConfigDict(extra="allow")

    @field_validator("input", mode="before")
    @classmethod
    def validate_input(cls, v: Any) -> Any:
        """Ensure input is always a dict, never a Pydantic model."""
        if v is not None and not isinstance(v, dict):
            if hasattr(v, "model_dump"):
                v = v.model_dump()
            elif hasattr(v, "dict"):
                v = v.dict()
            else:
                v = {"value": str(v)}
        return v


def _content_block_to_api(block: MessageContent) -> Dict[str, Any]:
    """Convert a MessageContent block to API-ready dict for tool protocols."""
    def _to_plain_json(value: Any) -> Any:
        if value is None:
            return None
        if hasattr(value, "model_dump"):
            try:
                value = value.model_dump(mode="json")
            except (TypeError, ValueError):
                value = value.model_dump()
        elif hasattr(value, "dict"):
            value = value.dict()
        if isinstance(value, list):
            return [_to_plain_json(item) for item in value]
        if isinstance(value, tuple):
            return [_to_plain_json(item) for item in value]
        if isinstance(value, dict):
            return {str(key): _to_plain_json(item) for key, item in value.items()}
        return value

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
        input_value = getattr(block, "input", None) or {}
        # Ensure input is a dict, not a Pydantic model
        if hasattr(input_value, "model_dump"):
            input_value = input_value.model_dump()
        elif hasattr(input_value, "dict"):
            input_value = input_value.dict()
        elif not isinstance(input_value, dict):
            input_value = {"value": str(input_value)}
        return {
            "type": "tool_use",
            "id": getattr(block, "id", None) or getattr(block, "tool_use_id", "") or "",
            "name": getattr(block, "name", None) or "",
            "input": input_value,
        }
    if block_type == "server_tool_use":
        input_value = getattr(block, "input", None) or {}
        if hasattr(input_value, "model_dump"):
            input_value = input_value.model_dump()
        elif hasattr(input_value, "dict"):
            input_value = input_value.dict()
        elif not isinstance(input_value, dict):
            input_value = {"value": str(input_value)}
        return {
            "type": "server_tool_use",
            "id": getattr(block, "id", None) or getattr(block, "tool_use_id", "") or "",
            "name": getattr(block, "name", None) or "",
            "input": input_value,
        }
    if block_type == "tool_search_tool_result":
        payload = _to_plain_json(getattr(block, "content", None))
        if payload is None:
            payload = {}
        return {
            "type": "tool_search_tool_result",
            "tool_use_id": getattr(block, "tool_use_id", None) or getattr(block, "id", None) or "",
            "content": payload,
        }
    if block_type == "tool_reference":
        return {
            "type": "tool_reference",
            "tool_name": getattr(block, "tool_name", None) or getattr(block, "name", None) or "",
        }
    if block_type == "tool_result":
        content_value = _to_plain_json(getattr(block, "content", None))
        if content_value is None:
            content_value = [
                {
                    "type": "text",
                    "text": getattr(block, "text", None) or "",
                }
            ]
        elif isinstance(content_value, str):
            content_value = [{"type": "text", "text": content_value}]
        elif isinstance(content_value, dict):
            content_value = [content_value]
        result: Dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": getattr(block, "tool_use_id", None) or getattr(block, "id", None) or "",
            "content": content_value,
        }
        if getattr(block, "is_error", None) is not None:
            result["is_error"] = block.is_error
        return result
    if block_type == "image":
        return {
            "type": "image",
            "source": {
                "type": getattr(block, "source_type", None) or "base64",
                "media_type": getattr(block, "media_type", None) or "image/jpeg",
                "data": getattr(block, "image_data", None) or "",
            },
        }
    # Default to text block
    return {
        "type": "text",
        "text": getattr(block, "text", None) or getattr(block, "content", None) or str(block),
    }


def _content_block_to_openai(block: MessageContent) -> Dict[str, Any]:
    """Convert a MessageContent block to OpenAI chat-completions tool call format."""
    block_type = getattr(block, "type", None)
    if block_type in {"server_tool_use", "tool_search_tool_result", "tool_reference"}:
        # Anthropic-specific tool-search blocks are not valid OpenAI messages.
        return {}
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
    if block_type == "image":
        # OpenAI uses data URL format for images
        media_type = getattr(block, "media_type", None) or "image/jpeg"
        image_data = getattr(block, "image_data", None) or ""
        data_url = f"data:{media_type};base64,{image_data}"
        return {
            "type": "image_url",
            "image_url": {"url": data_url},
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
    parent_tool_use_id: Optional[str] = None
    tool_use_result: Optional[object] = None
    # is_meta: true indicates system-level messages (like hook contexts)
    # that should be treated specially during message processing
    is_meta: bool = False
    timestamp: Optional[str] = None

    def __init__(self, **data: object) -> None:
        if "uuid" not in data or not data["uuid"]:
            data["uuid"] = str(uuid4())
        if "timestamp" not in data or data["timestamp"] is None:
            from datetime import datetime
            data["timestamp"] = datetime.utcnow().isoformat()
        super().__init__(**data)


class AssistantMessage(BaseModel):
    """Assistant message with metadata."""

    type: str = "assistant"
    message: Message
    uuid: str = ""
    parent_tool_use_id: Optional[str] = None
    cost_usd: float = 0.0
    duration_ms: float = 0.0
    is_api_error_message: bool = False
    # Model and token usage information
    model: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    error: Optional[str] = None

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
    progress_sender: Optional[str] = None
    normalized_messages: List[Message] = []
    sibling_tool_use_ids: set[str] = set()
    is_subagent_message: bool = False  # Flag to indicate if content is a subagent message
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data: object) -> None:
        if "uuid" not in data or not data["uuid"]:
            data["uuid"] = str(uuid4())
        super().__init__(**data)


HOOK_NOTICE_TYPE = "hook_notice"


def create_hook_notice_payload(
    text: str,
    hook_event: str,
    tool_name: Optional[str] = None,
    level: str = "info",
) -> Dict[str, Any]:
    """Create a structured hook notice payload for user-facing messages."""
    payload: Dict[str, Any] = {
        "type": HOOK_NOTICE_TYPE,
        "text": text,
        "hook_event": hook_event,
        "level": level,
    }
    if tool_name:
        payload["tool_name"] = tool_name
    return payload


def is_hook_notice_payload(content: Any) -> bool:
    """Check whether a progress content payload is a hook notice."""
    return isinstance(content, dict) and content.get("type") == HOOK_NOTICE_TYPE


def create_hook_notice_message(
    text: str,
    hook_event: str,
    *,
    tool_name: Optional[str] = None,
    level: str = "info",
    tool_use_id: str = "hook_notice",
    sibling_tool_use_ids: Optional[set[str]] = None,
) -> ProgressMessage:
    """Create a progress message for hook notices that should be shown to the user."""
    payload = create_hook_notice_payload(
        text=text,
        hook_event=hook_event,
        tool_name=tool_name,
        level=level,
    )
    return create_progress_message(
        tool_use_id=tool_use_id,
        sibling_tool_use_ids=sibling_tool_use_ids or set(),
        content=payload,
    )


# Regex pattern for parsing system-reminder tags
# Matches: <system-reminder>\n?content\n?</system-reminder>
SYSTEM_REMINDER_PATTERN = re.compile(r"^<system-reminder>\n?([\s\S]*?)\n?</system-reminder>$")

# Regex pattern for stripping system-reminder tags from content
SYSTEM_REMINDER_STRIP_PATTERN = re.compile(r"<system-reminder>[\s\S]*?</system-reminder>")

def system_reminder_wrapper(content: str) -> str:
    return f"<system-reminder>\n{content}\n</system-reminder>"


def inline_system_reminder(message: str) -> str:
    """Create an inline system-reminder without newlines.

    Used for warnings in tool results (e.g., empty file, offset exceeded).
    Format: <system-reminder>Warning: ...</system-reminder>

    This matches the original implementation for Read tool warnings:
    "<system-reminder>Warning: the file exists but the contents are empty.</system-reminder>"
    `<system-reminder>Warning: the file exists but is shorter than the provided offset (${offset}). The file has ${totalLines} lines.</system-reminder>`
    """
    return f"<system-reminder>{message}</system-reminder>"


def format_empty_file_warning() -> str:
    """Format warning for empty file.

    Returns:
        Inline system-reminder warning for empty file.
    """
    return inline_system_reminder("Warning: the file exists but the contents are empty.")


def format_offset_exceeded_warning(offset: int, total_lines: int) -> str:
    """Format warning when offset exceeds file length.

    Args:
        offset: The requested line offset.
        total_lines: The total number of lines in the file.

    Returns:
        Inline system-reminder warning for offset exceeded.
    """
    return inline_system_reminder(
        f"Warning: the file exists but is shorter than the provided offset ({offset}). "
        f"The file has {total_lines} lines."
    )


def strip_system_reminders(content: str) -> str:
    """Remove all system-reminder tags and their contents from a string.

    Used when processing tool results that should not include system reminders
    in certain contexts (e.g., file content extraction for caching).
    """
    return SYSTEM_REMINDER_STRIP_PATTERN.sub("", content)


def render_system_messages(messages: List["UserMessage"]) -> List["UserMessage"]:
    """Wrap message content in system-reminder tags.

    This transforms messages so their content is wrapped in system-reminder tags,
    matching the original implementation:

    function renderSystemMessages(messages) {
      return messages.map((message) => {
        if (typeof message.message.content === "string")
          return {
            ...message,
            message: {
              ...message.message,
              content: systemReminderWrapper(message.message.content),
            },
          };
        else if (Array.isArray(message.message.content)) {
          let processedContent = message.message.content.map((contentItem) => {
            if (contentItem.type === "text")
              return {
                ...contentItem,
                text: systemReminderWrapper(contentItem.text),
              };
            return contentItem;
          });
          return {
            ...message,
            message: {
              ...message.message,
              content: processedContent,
            },
          };
        }
        return message;
      });
    }

    Args:
        messages: List of UserMessage objects to transform.

    Returns:
        New list of UserMessage objects with content wrapped in system-reminder tags.
    """
    result: List["UserMessage"] = []
    for msg in messages:
        content = msg.message.content

        if isinstance(content, str):
            # String content: wrap directly
            new_message = Message(
                role=msg.message.role,
                content=system_reminder_wrapper(content),
                metadata=getattr(msg.message, "metadata", None),
            )
            new_msg = UserMessage(
                message=new_message,
                uuid=msg.uuid,
                parent_tool_use_id=msg.parent_tool_use_id,
                tool_use_result=msg.tool_use_result,
                is_meta=msg.is_meta,  # Preserve is_meta field
                timestamp=msg.timestamp,  # Preserve timestamp field
            )
            result.append(new_msg)
        elif isinstance(content, list):
            # Array content: wrap text blocks
            processed_blocks: List[MessageContent] = []
            for block in content:
                block_type = getattr(block, "type", None)
                if block_type == "text":
                    text = getattr(block, "text", "")
                    # Create new block with wrapped text
                    new_block = MessageContent(
                        type="text",
                        text=system_reminder_wrapper(text) if text else "",
                    )
                    processed_blocks.append(new_block)
                else:
                    processed_blocks.append(block)

            new_message = Message(
                role=msg.message.role,
                content=processed_blocks,
                metadata=getattr(msg.message, "metadata", None),
            )
            new_msg = UserMessage(
                message=new_message,
                uuid=msg.uuid,
                parent_tool_use_id=msg.parent_tool_use_id,
                tool_use_result=msg.tool_use_result,
                is_meta=msg.is_meta,  # Preserve is_meta field
                timestamp=msg.timestamp,  # Preserve timestamp field
            )
            result.append(new_msg)
        else:
            # Unknown content type: pass through unchanged
            result.append(msg)

    return result


# Maximum characters for hook context before truncation
# Matches original implementation's _c8 constant (inferred from MAX_CHARS = 50000)
MAX_HOOK_CONTEXT_CHARS = 50000


def truncate_hook_context(content: str, max_chars: int = MAX_HOOK_CONTEXT_CHARS) -> str:
    """Truncate hook context if it exceeds max_chars.
    Args:
        content: The content string to potentially truncate
        max_chars: Maximum allowed characters (default 50000)

    Returns:
        Truncated content with suffix if exceeded, otherwise original content
    """
    if len(content) > max_chars:
        return f"{content[:max_chars]}… [output truncated - exceeded {max_chars} characters]"
    return content


def _format_hook_additional_context_text(hook_name: str, content: str) -> str:
    """Render hook additional context as a wrapped system reminder."""
    return system_reminder_wrapper(f"{hook_name} hook additional context: {content}")


def _normalize_hook_additional_context(content: Any) -> str:
    """Normalize hook additional context."""
    if isinstance(content, (list, tuple)):
        parts: List[str] = []
        for item in content:
            text = str(item).strip()
            if text:
                parts.append(text)
        return "\n".join(parts)
    return str(content).strip()


def create_hook_additional_context_message(
    content: Any,
    *,
    hook_name: str,
    hook_event: str,
    parent_tool_use_id: Optional[str] = None,
) -> Optional[UserMessage]:
    """Create a model-visible user message carrying hook additional context.
    Args:
        content: The additional context content (string or list of strings)
        hook_name: Name of the hook (e.g., "SessionStart", "UserPromptSubmit", "PreToolUse:Read")
        hook_event: Type of hook event (e.g., "SessionStart", "UserPromptSubmit", "PreToolUse")
        parent_tool_use_id: Optional tool use ID for tool-related hooks

    Returns:
        UserMessage with is_meta=True, or None if content is empty
    """
    # Normalize content (join array with newlines)
    content_text = _normalize_hook_additional_context(content)

    # Check for empty content - matches original: if (attachment.content.length === 0) return [];
    if not content_text:
        return None

    # Apply truncation for UserPromptSubmit events (matches original GVq behavior)
    # Original: content: hookEvent.additionalContexts.map(GVq)
    if hook_event == "UserPromptSubmit":
        content_text = truncate_hook_context(content_text)

    # Create message with system-reminder wrapper
    message = Message(
        role=MessageRole.USER,
        content=_format_hook_additional_context_text(hook_name, content_text),
        metadata={
            "hook_additional_context": True,
            "hook_event": hook_event,
            "hook_name": hook_name,
        },
    )

    # Return UserMessage with is_meta=True (matches original isMeta: !0)
    return UserMessage(
        message=message,
        parent_tool_use_id=parent_tool_use_id,
        is_meta=True,  # Matches original isMeta: !0
    )


def create_user_message(
    content: Union[str, List[Dict[str, Any]]],
    tool_use_result: Optional[object] = None,
    parent_tool_use_id: Optional[str] = None,
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
                tool_use_result = tool_use_result.model_dump(by_alias=True, mode="json")
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

    return UserMessage(
        message=message,
        tool_use_result=tool_use_result,
        parent_tool_use_id=parent_tool_use_id,
    )


def _normalize_content_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a content item to ensure all fields are JSON-serializable.

    This is needed because some API providers may return Pydantic models
    for tool input fields, which need to be converted to dicts for proper
    serialization and later processing.

    Args:
        item: The content item dict from API response

    Returns:
        Normalized content item with all fields JSON-serializable
    """
    normalized = dict(item)

    # If input is a Pydantic model, convert to dict
    if 'input' in normalized and normalized['input'] is not None:
        input_value = normalized['input']
        if hasattr(input_value, 'model_dump'):
            normalized['input'] = input_value.model_dump()
        elif hasattr(input_value, 'dict'):
            normalized['input'] = input_value.dict()
        elif not isinstance(input_value, dict):
            normalized['input'] = {'value': str(input_value)}

    # If content is a Pydantic model, convert to plain JSON-like data
    if 'content' in normalized and normalized['content'] is not None:
        content_value = normalized['content']
        if hasattr(content_value, 'model_dump'):
            normalized['content'] = content_value.model_dump(mode="json")
        elif hasattr(content_value, 'dict'):
            normalized['content'] = content_value.dict()

    return normalized


def create_assistant_message(
    content: Union[str, List[Dict[str, Any]]],
    cost_usd: float = 0.0,
    duration_ms: float = 0.0,
    reasoning: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
    model: Optional[str] = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
    parent_tool_use_id: Optional[str] = None,
    error: Optional[str] = None,
) -> AssistantMessage:
    """Create an assistant message."""
    if isinstance(content, str):
        message_content: Union[str, List[MessageContent]] = content
    else:
        # Normalize content items to ensure tool input is always a dict
        message_content = [MessageContent(**_normalize_content_item(item)) for item in content]

    message = Message(
        role=MessageRole.ASSISTANT,
        content=message_content,
        reasoning=reasoning,
        metadata=metadata or {},
    )

    return AssistantMessage(
        message=message,
        cost_usd=cost_usd,
        duration_ms=duration_ms,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_creation_tokens=cache_creation_tokens,
        parent_tool_use_id=parent_tool_use_id,
        error=error,
    )


def create_progress_message(
    tool_use_id: str,
    sibling_tool_use_ids: set[str],
    content: Any,
    progress_sender: Optional[str] = None,
    normalized_messages: Optional[List[Message]] = None,
    is_subagent_message: bool = False,
) -> ProgressMessage:
    """Create a progress message."""
    return ProgressMessage(
        tool_use_id=tool_use_id,
        sibling_tool_use_ids=sibling_tool_use_ids,
        content=content,
        progress_sender=progress_sender,
        normalized_messages=normalized_messages or [],
        is_subagent_message=is_subagent_message,
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

    Provider-specific behavior is delegated to strategy helpers in
    ``ripperdoc.utils.messaging.message_normalization`` to keep this module focused on
    message model definitions and block conversion primitives.
    """
    from ripperdoc.utils.messaging.message_normalization import normalize_messages_for_api_impl

    return normalize_messages_for_api_impl(
        messages,
        protocol=protocol,
        tool_mode=tool_mode,
        thinking_mode=thinking_mode,
        to_api=_content_block_to_api,
        to_openai=_content_block_to_openai,
        apply_deepseek_reasoning_content=_apply_deepseek_reasoning_content,
        logger=logger,
    )


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
