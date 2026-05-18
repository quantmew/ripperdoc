"""SDK stream message DTOs."""

from __future__ import annotations

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class MessageData(BaseModel):
    """Base message data."""

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )


class AssistantMessageData(MessageData):
    """Assistant message data."""

    role: str = "assistant"
    content: Union[list[dict[str, Any]], str]
    model: str = "main"


class UserMessageData(MessageData):
    """User message data."""

    role: str = "user"
    content: Union[list[dict[str, Any]], str] = ""


class AssistantStreamMessage(BaseModel):
    """An assistant message sent to SDK stream output."""

    type: str = Field(default="assistant")
    message: AssistantMessageData
    session_id: Optional[str] = None
    parent_tool_use_id: Optional[str] = None
    uuid: Optional[str] = None


class UserStreamMessage(BaseModel):
    """A user message sent to SDK stream output."""

    type: str = Field(default="user")
    message: UserMessageData
    uuid: Optional[str] = None
    session_id: Optional[str] = None
    parent_tool_use_id: Optional[str] = None
    tool_use_result: Any = None


class IncomingUserMessageData(BaseModel):
    """Validated incoming user message data from user-facing stream input."""

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )

    role: Literal["user"]
    content: Union[list[dict[str, Any]], str] = ""


class IncomingUserStreamMessage(BaseModel):
    """Validated incoming `type=user` message from stream input."""

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )

    type: Literal["user"]
    message: IncomingUserMessageData
    uuid: Optional[str] = None
    session_id: Optional[str] = None
    parent_tool_use_id: Optional[str] = None
    tool_use_result: Any = None


StreamMessage = Union[AssistantStreamMessage, UserStreamMessage]


__all__ = [
    "MessageData",
    "AssistantMessageData",
    "UserMessageData",
    "AssistantStreamMessage",
    "UserStreamMessage",
    "IncomingUserMessageData",
    "IncomingUserStreamMessage",
    "StreamMessage",
]
