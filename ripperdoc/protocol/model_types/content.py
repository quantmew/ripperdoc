"""Protocol content block DTOs."""

from __future__ import annotations

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class ContentBlock(BaseModel):
    """Base class for message content blocks."""

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
    signature: Optional[str] = None


class ToolUseContentBlock(ContentBlock):
    """A tool call content block."""

    type: str = Field(default="tool_use")
    id: str = Field(default="")
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class ToolResultContentBlock(ContentBlock):
    """A tool result content block."""

    type: str = Field(default="tool_result")
    tool_use_id: str = Field(default="")
    content: Any = Field(default="")
    is_error: Optional[bool] = None


class ImageSource(BaseModel):
    """Image source data."""

    type: str = Field(default="base64")
    media_type: str = Field(default="image/jpeg")
    data: str


class ImageContentBlock(ContentBlock):
    """An image content block."""

    type: str = Field(default="image")
    source: ImageSource


ContentBlockType = Union[
    TextContentBlock,
    ThinkingContentBlock,
    ToolUseContentBlock,
    ToolResultContentBlock,
    ImageContentBlock,
]


__all__ = [
    "ContentBlock",
    "TextContentBlock",
    "ThinkingContentBlock",
    "ToolUseContentBlock",
    "ToolResultContentBlock",
    "ImageSource",
    "ImageContentBlock",
    "ContentBlockType",
]
