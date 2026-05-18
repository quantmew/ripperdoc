"""Content block type for messages."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, field_validator


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
