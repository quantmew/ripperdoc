"""Core message type definitions."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from .attachment import AttachmentPayloadModel, _coerce_attachment_payload
from .content import MessageContent


class MessageRole(str, Enum):
    """Message roles in a conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


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
            from datetime import datetime, timezone

            data["timestamp"] = datetime.now(timezone.utc).isoformat()
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
    sibling_tool_use_ids: Set[str] = set()
    is_subagent_message: bool = False  # Flag to indicate if content is a subagent message
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data: object) -> None:
        if "uuid" not in data or not data["uuid"]:
            data["uuid"] = str(uuid4())
        super().__init__(**data)


class AttachmentMessage(BaseModel):
    """Internal attachment item for transcript attachments."""

    type: str = "attachment"
    attachment: AttachmentPayloadModel
    uuid: str = ""
    timestamp: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data: object) -> None:
        if "attachment" not in data:
            legacy_type = data.pop("attachment_type", None)
            legacy_content = data.pop("content", None)
            legacy_metadata = data.pop("metadata", None)
            legacy_parent_tool_use_id = data.pop("parent_tool_use_id", None)
            attachment_payload: Dict[str, Any] = {"type": legacy_type or "unknown"}
            if legacy_content is not None:
                attachment_payload["content"] = legacy_content
            if isinstance(legacy_metadata, dict):
                attachment_payload["metadata"] = legacy_metadata
            if legacy_parent_tool_use_id is not None:
                attachment_payload["parent_tool_use_id"] = legacy_parent_tool_use_id
            data["attachment"] = attachment_payload
        data["attachment"] = _coerce_attachment_payload(data["attachment"])
        if "uuid" not in data or not data["uuid"]:
            data["uuid"] = str(uuid4())
        if "timestamp" not in data or data["timestamp"] is None:
            from datetime import datetime, timezone

            data["timestamp"] = datetime.now(timezone.utc).isoformat()
        super().__init__(**data)

    @property
    def attachment_type(self) -> str:
        return str(getattr(self.attachment, "type", "") or "")

    @property
    def content(self) -> Any:
        return getattr(self.attachment, "content", "")

    @property
    def metadata(self) -> Dict[str, Any]:
        metadata = getattr(self.attachment, "metadata", None)
        return dict(metadata) if isinstance(metadata, dict) else {}

    @property
    def parent_tool_use_id(self) -> Optional[str]:
        value = getattr(self.attachment, "parent_tool_use_id", None)
        return str(value) if isinstance(value, str) else None
