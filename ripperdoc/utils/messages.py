"""Message handling and formatting for Ripperdoc.

This module provides utilities for creating and normalizing messages
for communication with AI models.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict
from uuid import uuid4
from enum import Enum


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
    input: Optional[Dict[str, Any]] = None
    is_error: Optional[bool] = None


class Message(BaseModel):
    """A message in a conversation."""
    role: MessageRole
    content: Union[str, List[MessageContent]]
    uuid: str = ""

    def __init__(self, **data: Any) -> None:
        if "uuid" not in data or not data["uuid"]:
            data["uuid"] = str(uuid4())
        super().__init__(**data)


class UserMessage(BaseModel):
    """User message with tool results."""
    type: str = "user"
    message: Message
    uuid: str = ""
    tool_use_result: Optional[Any] = None

    def __init__(self, **data: Any) -> None:
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

    def __init__(self, **data: Any) -> None:
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

    def __init__(self, **data: Any) -> None:
        if "uuid" not in data or not data["uuid"]:
            data["uuid"] = str(uuid4())
        super().__init__(**data)


def create_user_message(
    content: Union[str, List[Dict[str, Any]]],
    tool_use_result: Optional[Any] = None
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
            pass

    message = Message(
        role=MessageRole.USER,
        content=message_content
    )

    return UserMessage(
        message=message,
        tool_use_result=tool_use_result
    )


def create_assistant_message(
    content: Union[str, List[Dict[str, Any]]],
    cost_usd: float = 0.0,
    duration_ms: float = 0.0
) -> AssistantMessage:
    """Create an assistant message."""
    if isinstance(content, str):
        message_content: Union[str, List[MessageContent]] = content
    else:
        message_content = [MessageContent(**item) for item in content]

    message = Message(
        role=MessageRole.ASSISTANT,
        content=message_content
    )

    return AssistantMessage(
        message=message,
        cost_usd=cost_usd,
        duration_ms=duration_ms
    )


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
        normalized_messages=normalized_messages or []
    )


def normalize_messages_for_api(
    messages: List[Union[UserMessage, AssistantMessage, ProgressMessage]]
) -> List[Dict[str, Any]]:
    """Normalize messages for API submission.

    Progress messages are filtered out as they are not sent to the API.
    """
    normalized = []

    for msg in messages:
        if msg.type == "progress":
            # Skip progress messages
            continue

        if msg.type == "user":
            user_msg = msg

            # Handle tool result messages specially
            if isinstance(user_msg.message.content, list):
                # Extract text content from tool results
                text_parts = []
                for block in user_msg.message.content:
                    if hasattr(block, 'type') and block.type == "tool_result" and block.text:
                        text_parts.append(block.text)

                if text_parts:
                    # Combine tool results into a single text message
                    normalized.append({
                        "role": "user",
                        "content": "\n\n".join(text_parts)
                    })
                else:
                    # Fallback: use original content
                    normalized.append({
                        "role": "user",
                        "content": user_msg.message.content  # type: ignore
                    })
            else:
                # Regular text message
                normalized.append({
                    "role": "user",
                    "content": user_msg.message.content  # type: ignore
                })
        elif msg.type == "assistant":
            asst_msg = msg

            # Handle assistant messages with tool use
            if isinstance(asst_msg.message.content, list):
                # For downstream APIs, only send text content; tool_use blocks stay local
                text_parts = []
                for block in asst_msg.message.content:
                    if hasattr(block, 'type') and block.type == "text" and getattr(block, "text", None):
                        text_parts.append(block.text)

                normalized.append({
                    "role": "assistant",
                    "content": "\n\n".join(text_parts)
                })
            else:
                # Regular text message
                normalized.append({
                    "role": "assistant",
                    "content": asst_msg.message.content  # type: ignore
                })

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
        "is_error": True
    }
