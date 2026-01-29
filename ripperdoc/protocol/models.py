"""Pydantic models for stdio protocol messages.

This module defines type-safe models for all JSON messages exchanged
over the stdio protocol, replacing raw dictionary construction with
validated, self-documenting Pydantic models.
"""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field, ConfigDict


# ============================================================================
# Content Block Models
# ============================================================================


class ContentBlock(BaseModel):
    """Base class for content blocks in messages."""

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )


class TextContentBlock(ContentBlock):
    """A text content block."""

    type: Literal["text"] = "text"
    text: str


class ThinkingContentBlock(ContentBlock):
    """A thinking/reasoning content block."""

    type: str = Field(default="thinking")
    thinking: str = Field(alias="text")
    signature: str | None = None


class ToolUseContentBlock(ContentBlock):
    """A tool use content block."""

    type: str = Field(default="tool_use")
    id: str = Field(default="")
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class ToolResultContentBlock(ContentBlock):
    """A tool result content block."""

    type: str = Field(default="tool_result")
    tool_use_id: str = Field(default="")
    content: str = Field(default="")
    is_error: bool | None = None


class ImageSource(BaseModel):
    """Image source data."""

    type: str = Field(default="base64")
    media_type: str = Field(default="image/jpeg")
    data: str


class ImageContentBlock(ContentBlock):
    """An image content block."""

    type: str = Field(default="image")
    source: ImageSource


# Union type for all content blocks
ContentBlockType = (
    TextContentBlock
    | ThinkingContentBlock
    | ToolUseContentBlock
    | ToolResultContentBlock
    | ImageContentBlock
)


# ============================================================================
# Message Models
# ============================================================================


class MessageData(BaseModel):
    """Base message data."""

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )


class AssistantMessageData(MessageData):
    """Assistant message data."""

    content: list[dict[str, Any]] | str
    model: str = "main"


class UserMessageData(MessageData):
    """User message data."""

    content: str = ""


class AssistantStreamMessage(BaseModel):
    """An assistant message sent to the SDK."""

    type: str = Field(default="assistant")
    message: AssistantMessageData
    parent_tool_use_id: str | None = None


class UserStreamMessage(BaseModel):
    """A user message sent to the SDK."""

    type: str = Field(default="user")
    message: UserMessageData
    uuid: str | None = None
    parent_tool_use_id: str | None = None
    tool_use_result: Any = None


# Union type for stream messages
StreamMessage = AssistantStreamMessage | UserStreamMessage


# ============================================================================
# Control Protocol Models
# ============================================================================


class ControlResponseData(BaseModel):
    """Base class for control response data."""

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )


class ControlResponseSuccess(ControlResponseData):
    """A successful control response."""

    subtype: str = Field(default="success")
    request_id: str
    response: dict[str, Any] | None = None


class ControlResponseError(ControlResponseData):
    """An error control response."""

    subtype: str = Field(default="error")
    request_id: str
    error: str


class ControlResponseMessage(BaseModel):
    """A control response message wrapper."""

    type: str = Field(default="control_response")
    response: ControlResponseSuccess | ControlResponseError


# ============================================================================
# Result/Usage Models
# ============================================================================


class UsageInfo(BaseModel):
    """Token usage information."""

    input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    output_tokens: int = 0


class MCPServerInfo(BaseModel):
    """MCP server information."""

    name: str


class InitializeResponseData(BaseModel):
    """Response data for initialize request."""

    session_id: str
    system_prompt: str
    tools: list[str]
    mcp_servers: list[MCPServerInfo] = Field(default_factory=list)
    slash_commands: list[Any] = Field(default_factory=list)
    apiKeySource: str = "none"
    claude_code_version: str = "0.1.0"
    output_style: str = "default"
    agents: list[str] = Field(default_factory=list)
    skills: list[Any] = Field(default_factory=list)
    plugins: list[Any] = Field(default_factory=list)

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )


class ResultMessage(BaseModel):
    """A result message sent at the end of a query."""

    type: str = Field(default="result")
    subtype: str = Field(default="result")
    duration_ms: int
    duration_api_ms: int
    is_error: bool
    num_turns: int
    session_id: str
    total_cost_usd: float | None = None
    usage: UsageInfo | None = None
    result: str | None = None
    structured_output: Any = None


# ============================================================================
# Permission Response Models
# ============================================================================


class PermissionResponseAllow(BaseModel):
    """A permission allow response."""

    decision: str = Field(default="allow")
    updatedInput: dict[str, Any] | None = None


class PermissionResponseDeny(BaseModel):
    """A permission deny response."""

    decision: str = Field(default="deny")
    message: str = ""


# ============================================================================
# Helper Functions
# ============================================================================


def model_to_dict(model: BaseModel) -> dict[str, Any]:
    """Convert a Pydantic model to a JSON-serializable dictionary.

    This handles exclude_none=True and ensures proper serialization,
    while always including type/subtype fields for protocol messages.

    Args:
        model: The Pydantic model to convert.

    Returns:
        A JSON-serializable dictionary.
    """
    return model.model_dump(exclude_none=True, by_alias=True, mode="json")


__all__ = [
    # Content Blocks
    "ContentBlock",
    "TextContentBlock",
    "ThinkingContentBlock",
    "ToolUseContentBlock",
    "ToolResultContentBlock",
    "ImageContentBlock",
    "ImageSource",
    "ContentBlockType",
    # Messages
    "MessageData",
    "AssistantMessageData",
    "UserMessageData",
    "AssistantStreamMessage",
    "UserStreamMessage",
    "StreamMessage",
    # Control Protocol
    "ControlResponseData",
    "ControlResponseSuccess",
    "ControlResponseError",
    "ControlResponseMessage",
    # Result/Usage
    "UsageInfo",
    "MCPServerInfo",
    "InitializeResponseData",
    "ResultMessage",
    # Permissions
    "PermissionResponseAllow",
    "PermissionResponseDeny",
    # Helpers
    "model_to_dict",
]
