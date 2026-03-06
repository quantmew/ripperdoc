"""Message handling and formatting for Ripperdoc.

This module provides utilities for creating and normalizing messages
for communication with AI models.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Type, Union, cast
from pydantic import BaseModel, ConfigDict, Field, field_validator
from uuid import uuid4
from enum import Enum
from ripperdoc.utils.log import get_logger

logger = get_logger()
FILE_ATTACHMENT_TRUNCATION_LINE_LIMIT = int(os.getenv("RIPPERDOC_MAX_READ_LINES", "2000"))


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
    sibling_tool_use_ids: set[str] = set()
    is_subagent_message: bool = False  # Flag to indicate if content is a subagent message
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data: object) -> None:
        if "uuid" not in data or not data["uuid"]:
            data["uuid"] = str(uuid4())
        super().__init__(**data)


class AttachmentPayload(BaseModel):
    """Base attachment payload."""

    type: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    parent_tool_use_id: Optional[str] = None
    model_config = ConfigDict(extra="forbid")


class UnknownAttachmentPayload(AttachmentPayload):
    """Fallback payload for attachments without a dedicated model."""

    model_config = ConfigDict(extra="allow")


class HookAdditionalContextAttachmentPayload(AttachmentPayload):
    type: str = "hook_additional_context"
    hook_name: str
    hook_event: str
    content: Any


class PlanModeAttachmentPayload(AttachmentPayload):
    type: str
    content: str
    plan_file_path: str
    plan_exists: bool | None = None
    reminder_type: str


class DirectoryAttachmentPayload(AttachmentPayload):
    type: str = "directory"
    path: str
    content: str


class EditedTextFileAttachmentPayload(AttachmentPayload):
    type: str = "edited_text_file"
    filename: str
    snippet: str


class FileAttachmentContent(BaseModel):
    type: str
    model_config = ConfigDict(extra="allow")


class UnknownFileAttachmentContent(FileAttachmentContent):
    type: str = "unknown"


class FileTextAttachmentContent(FileAttachmentContent):
    type: str = "text"


class FileImageAttachmentContent(FileAttachmentContent):
    type: str = "image"


class FileNotebookAttachmentContent(FileAttachmentContent):
    type: str = "notebook"


class FilePdfAttachmentContent(FileAttachmentContent):
    type: str = "pdf"


FileAttachmentContentModel = Union[
    FileTextAttachmentContent,
    FileImageAttachmentContent,
    FileNotebookAttachmentContent,
    FilePdfAttachmentContent,
    UnknownFileAttachmentContent,
]


def _coerce_file_attachment_content(content: Any) -> FileAttachmentContentModel:
    if isinstance(content, FileAttachmentContent):
        return cast(FileAttachmentContentModel, content)
    if not isinstance(content, dict):
        return UnknownFileAttachmentContent(type="unknown", value=str(content))

    content_type = str(content.get("type") or "unknown")
    model_by_type: Dict[str, Type[FileAttachmentContent]] = {
        "text": FileTextAttachmentContent,
        "image": FileImageAttachmentContent,
        "notebook": FileNotebookAttachmentContent,
        "pdf": FilePdfAttachmentContent,
    }
    model = model_by_type.get(content_type, UnknownFileAttachmentContent)
    return cast(FileAttachmentContentModel, model(**content))


class FileAttachmentPayload(AttachmentPayload):
    type: str = "file"
    filename: str
    content: FileAttachmentContentModel
    truncated: bool = False

    @field_validator("content", mode="before")
    @classmethod
    def validate_content(cls, value: Any) -> FileAttachmentContentModel:
        return _coerce_file_attachment_content(value)


class CompactFileReferenceAttachmentPayload(AttachmentPayload):
    type: str = "compact_file_reference"
    filename: str


class PdfReferenceAttachmentPayload(AttachmentPayload):
    type: str = "pdf_reference"
    filename: str
    pageCount: int
    fileSize: int | float


class SelectedLinesInIdeAttachmentPayload(AttachmentPayload):
    type: str = "selected_lines_in_ide"
    filename: str
    lineStart: int
    lineEnd: int
    content: str


class OpenedFileInIdeAttachmentPayload(AttachmentPayload):
    type: str = "opened_file_in_ide"
    filename: str


class TodoAttachmentPayload(AttachmentPayload):
    type: str = "todo"
    itemCount: int = 0
    content: Any = Field(default_factory=list)


class PlanFileReferenceAttachmentPayload(AttachmentPayload):
    type: str = "plan_file_reference"
    planFilePath: str
    planContent: str


class InvokedSkillsAttachmentPayload(AttachmentPayload):
    type: str = "invoked_skills"
    skills: List[Dict[str, Any]] = Field(default_factory=list)


class TodoReminderAttachmentPayload(AttachmentPayload):
    type: str = "todo_reminder"
    content: List[Dict[str, Any]] = Field(default_factory=list)


class TaskReminderAttachmentPayload(AttachmentPayload):
    type: str = "task_reminder"
    content: List[Dict[str, Any]] = Field(default_factory=list)


class NestedMemoryAttachmentPayload(AttachmentPayload):
    type: str = "nested_memory"
    content: Dict[str, Any]


class RelevantMemoriesAttachmentPayload(AttachmentPayload):
    type: str = "relevant_memories"
    memories: List[Dict[str, Any]] = Field(default_factory=list)


class SkillListingAttachmentPayload(AttachmentPayload):
    type: str = "skill_listing"
    content: str


class QueuedCommandAttachmentPayload(AttachmentPayload):
    type: str = "queued_command"
    prompt: Any
    commandMode: Optional[str] = None


class UltramemoryAttachmentPayload(AttachmentPayload):
    type: str = "ultramemory"
    content: Any


class McpResourceAttachmentPayload(AttachmentPayload):
    type: str = "mcp_resource"
    server: str
    uri: str
    content: Any


class AgentMentionAttachmentPayload(AttachmentPayload):
    type: str = "agent_mention"
    agentType: str


class OutputStyleAttachmentPayload(AttachmentPayload):
    type: str = "output_style"
    style: str


class TaskStatusAttachmentPayload(AttachmentPayload):
    type: str = "task_status"
    status: str
    description: str
    taskId: str
    taskType: str = ""
    deltaSummary: str = ""


class TaskProgressAttachmentPayload(AttachmentPayload):
    type: str = "task_progress"
    message: str


class DiagnosticsAttachmentPayload(AttachmentPayload):
    type: str = "diagnostics"
    files: List[Dict[str, Any]]


class CriticalSystemReminderAttachmentPayload(AttachmentPayload):
    type: str = "critical_system_reminder"
    content: str


class DateChangeAttachmentPayload(AttachmentPayload):
    type: str = "date_change"
    newDate: str


class TokenUsageAttachmentPayload(AttachmentPayload):
    type: str = "token_usage"
    used: Any
    total: Any
    remaining: Any


class BudgetUsdAttachmentPayload(AttachmentPayload):
    type: str = "budget_usd"
    used: Any
    total: Any
    remaining: Any


class AsyncHookResponseAttachmentPayload(AttachmentPayload):
    type: str = "async_hook_response"
    response: Dict[str, Any]


class HookBlockingErrorAttachmentPayload(AttachmentPayload):
    type: str = "hook_blocking_error"
    hookName: str
    blockingError: Dict[str, Any]


class HookSuccessAttachmentPayload(AttachmentPayload):
    type: str = "hook_success"
    hookName: str
    hookEvent: str
    content: str


class HookStoppedContinuationAttachmentPayload(AttachmentPayload):
    type: str = "hook_stopped_continuation"
    hookName: str
    message: str


class CompactionReminderAttachmentPayload(AttachmentPayload):
    type: str = "compaction_reminder"


class VerifyPlanReminderAttachmentPayload(AttachmentPayload):
    type: str = "verify_plan_reminder"


class DynamicSkillAttachmentPayload(AttachmentPayload):
    type: str = "dynamic_skill"


class AlreadyReadFileAttachmentPayload(AttachmentPayload):
    type: str = "already_read_file"
    filename: Optional[str] = None


class CommandPermissionsAttachmentPayload(AttachmentPayload):
    type: str = "command_permissions"
    permissions: Any = None


class EditedImageFileAttachmentPayload(AttachmentPayload):
    type: str = "edited_image_file"
    filename: Optional[str] = None


class HookCancelledAttachmentPayload(AttachmentPayload):
    type: str = "hook_cancelled"
    hookName: Optional[str] = None
    hookEvent: Optional[str] = None
    reason: Optional[str] = None


class HookErrorDuringExecutionAttachmentPayload(AttachmentPayload):
    type: str = "hook_error_during_execution"
    hookName: Optional[str] = None
    hookEvent: Optional[str] = None
    error: Optional[str] = None


class HookNonBlockingErrorAttachmentPayload(AttachmentPayload):
    type: str = "hook_non_blocking_error"
    hookName: Optional[str] = None
    hookEvent: Optional[str] = None
    error: Optional[str] = None


class HookSystemMessageAttachmentPayload(AttachmentPayload):
    type: str = "hook_system_message"
    hookName: Optional[str] = None
    hookEvent: Optional[str] = None
    systemMessage: Optional[str] = None


class StructuredOutputAttachmentPayload(AttachmentPayload):
    type: str = "structured_output"
    content: Any = None


class HookPermissionDecisionAttachmentPayload(AttachmentPayload):
    type: str = "hook_permission_decision"
    hookName: Optional[str] = None
    hookEvent: Optional[str] = None
    decision: Any = None


class AutocheckpointingAttachmentPayload(AttachmentPayload):
    type: str = "autocheckpointing"
    content: Any = None


class BackgroundTaskStatusAttachmentPayload(AttachmentPayload):
    type: str = "background_task_status"
    taskId: Optional[str] = None
    status: Optional[str] = None
    content: Any = None


AttachmentPayloadModel = Union[
    HookAdditionalContextAttachmentPayload,
    PlanModeAttachmentPayload,
    DirectoryAttachmentPayload,
    EditedTextFileAttachmentPayload,
    FileAttachmentPayload,
    CompactFileReferenceAttachmentPayload,
    PdfReferenceAttachmentPayload,
    SelectedLinesInIdeAttachmentPayload,
    OpenedFileInIdeAttachmentPayload,
    TodoAttachmentPayload,
    PlanFileReferenceAttachmentPayload,
    InvokedSkillsAttachmentPayload,
    TodoReminderAttachmentPayload,
    TaskReminderAttachmentPayload,
    NestedMemoryAttachmentPayload,
    RelevantMemoriesAttachmentPayload,
    SkillListingAttachmentPayload,
    QueuedCommandAttachmentPayload,
    UltramemoryAttachmentPayload,
    McpResourceAttachmentPayload,
    AgentMentionAttachmentPayload,
    OutputStyleAttachmentPayload,
    TaskStatusAttachmentPayload,
    TaskProgressAttachmentPayload,
    DiagnosticsAttachmentPayload,
    CriticalSystemReminderAttachmentPayload,
    DateChangeAttachmentPayload,
    TokenUsageAttachmentPayload,
    BudgetUsdAttachmentPayload,
    AsyncHookResponseAttachmentPayload,
    HookBlockingErrorAttachmentPayload,
    HookSuccessAttachmentPayload,
    HookStoppedContinuationAttachmentPayload,
    CompactionReminderAttachmentPayload,
    VerifyPlanReminderAttachmentPayload,
    DynamicSkillAttachmentPayload,
    AlreadyReadFileAttachmentPayload,
    CommandPermissionsAttachmentPayload,
    EditedImageFileAttachmentPayload,
    HookCancelledAttachmentPayload,
    HookErrorDuringExecutionAttachmentPayload,
    HookNonBlockingErrorAttachmentPayload,
    HookSystemMessageAttachmentPayload,
    StructuredOutputAttachmentPayload,
    HookPermissionDecisionAttachmentPayload,
    AutocheckpointingAttachmentPayload,
    BackgroundTaskStatusAttachmentPayload,
    UnknownAttachmentPayload,
]

ATTACHMENT_PAYLOAD_MODEL_BY_TYPE: Dict[str, Type[AttachmentPayload]] = {
    "hook_additional_context": HookAdditionalContextAttachmentPayload,
    "plan_mode": PlanModeAttachmentPayload,
    "plan_mode_reentry": PlanModeAttachmentPayload,
    "plan_mode_exit": PlanModeAttachmentPayload,
    "directory": DirectoryAttachmentPayload,
    "edited_text_file": EditedTextFileAttachmentPayload,
    "file": FileAttachmentPayload,
    "compact_file_reference": CompactFileReferenceAttachmentPayload,
    "pdf_reference": PdfReferenceAttachmentPayload,
    "selected_lines_in_ide": SelectedLinesInIdeAttachmentPayload,
    "opened_file_in_ide": OpenedFileInIdeAttachmentPayload,
    "todo": TodoAttachmentPayload,
    "plan_file_reference": PlanFileReferenceAttachmentPayload,
    "invoked_skills": InvokedSkillsAttachmentPayload,
    "todo_reminder": TodoReminderAttachmentPayload,
    "task_reminder": TaskReminderAttachmentPayload,
    "nested_memory": NestedMemoryAttachmentPayload,
    "relevant_memories": RelevantMemoriesAttachmentPayload,
    "skill_listing": SkillListingAttachmentPayload,
    "queued_command": QueuedCommandAttachmentPayload,
    "ultramemory": UltramemoryAttachmentPayload,
    "mcp_resource": McpResourceAttachmentPayload,
    "agent_mention": AgentMentionAttachmentPayload,
    "output_style": OutputStyleAttachmentPayload,
    "task_status": TaskStatusAttachmentPayload,
    "task_progress": TaskProgressAttachmentPayload,
    "diagnostics": DiagnosticsAttachmentPayload,
    "critical_system_reminder": CriticalSystemReminderAttachmentPayload,
    "date_change": DateChangeAttachmentPayload,
    "token_usage": TokenUsageAttachmentPayload,
    "budget_usd": BudgetUsdAttachmentPayload,
    "async_hook_response": AsyncHookResponseAttachmentPayload,
    "hook_blocking_error": HookBlockingErrorAttachmentPayload,
    "hook_success": HookSuccessAttachmentPayload,
    "hook_stopped_continuation": HookStoppedContinuationAttachmentPayload,
    "compaction_reminder": CompactionReminderAttachmentPayload,
    "verify_plan_reminder": VerifyPlanReminderAttachmentPayload,
    "dynamic_skill": DynamicSkillAttachmentPayload,
    "already_read_file": AlreadyReadFileAttachmentPayload,
    "command_permissions": CommandPermissionsAttachmentPayload,
    "edited_image_file": EditedImageFileAttachmentPayload,
    "hook_cancelled": HookCancelledAttachmentPayload,
    "hook_error_during_execution": HookErrorDuringExecutionAttachmentPayload,
    "hook_non_blocking_error": HookNonBlockingErrorAttachmentPayload,
    "hook_system_message": HookSystemMessageAttachmentPayload,
    "structured_output": StructuredOutputAttachmentPayload,
    "hook_permission_decision": HookPermissionDecisionAttachmentPayload,
    "autocheckpointing": AutocheckpointingAttachmentPayload,
    "background_task_status": BackgroundTaskStatusAttachmentPayload,
}


def _coerce_attachment_payload(payload: Any) -> AttachmentPayloadModel:
    if isinstance(payload, AttachmentPayload):
        return cast(AttachmentPayloadModel, payload)
    if not isinstance(payload, dict):
        return UnknownAttachmentPayload(type="unknown", content=str(payload))
    attachment_type = str(payload.get("type") or "unknown")
    payload_model = ATTACHMENT_PAYLOAD_MODEL_BY_TYPE.get(attachment_type, UnknownAttachmentPayload)
    return cast(AttachmentPayloadModel, payload_model(**payload))


class AttachmentMessage(BaseModel):
    """Internal attachment item aligned with Claude Code style transcript attachments."""

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


# Unified conversation message type alias
ConversationMessage = Union[UserMessage, AssistantMessage, ProgressMessage, AttachmentMessage]


ATTACHMENT_IGNORED_TYPES: set[str] = {
    "dynamic_skill",
    "already_read_file",
    "command_permissions",
    "edited_image_file",
    "hook_cancelled",
    "hook_error_during_execution",
    "hook_non_blocking_error",
    "hook_system_message",
    "structured_output",
    "hook_permission_decision",
    "autocheckpointing",
    "background_task_status",
}

ATTACHMENT_EXPORT_HIDDEN_TYPES: set[str] = {
    "plan_mode",
    "plan_mode_reentry",
    "plan_mode_exit",
    "hook_additional_context",
    "critical_system_reminder",
    "todo_reminder",
    "task_reminder",
    "compaction_reminder",
    "token_usage",
    "budget_usd",
    "hook_blocking_error",
    "hook_success",
    "hook_stopped_continuation",
    "async_hook_response",
    "date_change",
    "verify_plan_reminder",
}

ATTACHMENT_SUMMARY_HIDDEN_TYPES: set[str] = {
    "plan_mode",
    "plan_mode_reentry",
    "plan_mode_exit",
    "hook_additional_context",
    "critical_system_reminder",
    "compaction_reminder",
    "token_usage",
    "budget_usd",
    "hook_success",
    "hook_stopped_continuation",
}


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


def _attachment_attr(attachment: AttachmentMessage, key: str, default: Any = None) -> Any:
    return getattr(attachment.attachment, key, default)


def _create_typed_attachment_message(payload: AttachmentPayloadModel) -> AttachmentMessage:
    return AttachmentMessage(attachment=payload)


def create_attachment_message(
    attachment_type: str,
    *,
    parent_tool_use_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    **payload: Any,
) -> AttachmentMessage:
    attachment_payload: Dict[str, Any] = {
        "type": attachment_type,
        **payload,
    }
    if metadata:
        attachment_payload["metadata"] = dict(metadata)
    if parent_tool_use_id is not None:
        attachment_payload["parent_tool_use_id"] = parent_tool_use_id
    return AttachmentMessage(attachment=attachment_payload)


def _json_serialize(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _humanize_bytes(size: Any) -> str:
    if not isinstance(size, (int, float)):
        return str(size)
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    idx = 0
    while value >= 1024 and idx < len(units) - 1:
        value /= 1024
        idx += 1
    if idx == 0:
        return f"{int(value)} {units[idx]}"
    return f"{value:.1f} {units[idx]}"


def create_directory_attachment_message(path: str, content: str) -> AttachmentMessage:
    return _create_typed_attachment_message(
        DirectoryAttachmentPayload(path=path, content=content)
    )


def create_file_attachment_message(
    filename: str,
    content: Any,
    *,
    truncated: bool = False,
) -> AttachmentMessage:
    return _create_typed_attachment_message(
        FileAttachmentPayload(filename=filename, content=content, truncated=truncated)
    )


def create_edited_text_file_attachment_message(filename: str, snippet: str) -> AttachmentMessage:
    return _create_typed_attachment_message(
        EditedTextFileAttachmentPayload(filename=filename, snippet=snippet)
    )


def create_compact_file_reference_attachment_message(filename: str) -> AttachmentMessage:
    return _create_typed_attachment_message(
        CompactFileReferenceAttachmentPayload(filename=filename)
    )


def create_pdf_reference_attachment_message(
    filename: str,
    *,
    page_count: int,
    file_size: int | float,
) -> AttachmentMessage:
    return _create_typed_attachment_message(
        PdfReferenceAttachmentPayload(filename=filename, pageCount=page_count, fileSize=file_size)
    )


def create_selected_lines_in_ide_attachment_message(
    filename: str,
    *,
    line_start: int,
    line_end: int,
    content: str,
) -> AttachmentMessage:
    return _create_typed_attachment_message(
        SelectedLinesInIdeAttachmentPayload(
            filename=filename,
            lineStart=line_start,
            lineEnd=line_end,
            content=content,
        )
    )


def create_opened_file_in_ide_attachment_message(filename: str) -> AttachmentMessage:
    return _create_typed_attachment_message(
        OpenedFileInIdeAttachmentPayload(filename=filename)
    )


def create_todo_attachment_message(
    *,
    item_count: int = 0,
    content: Optional[List[Dict[str, Any]]] = None,
) -> AttachmentMessage:
    return _create_typed_attachment_message(
        TodoAttachmentPayload(itemCount=item_count, content=content or [])
    )


def create_plan_file_reference_attachment_message(
    plan_file_path: str,
    plan_content: str,
) -> AttachmentMessage:
    return _create_typed_attachment_message(
        PlanFileReferenceAttachmentPayload(
            planFilePath=plan_file_path,
            planContent=plan_content,
        )
    )


def create_invoked_skills_attachment_message(skills: List[Dict[str, Any]]) -> AttachmentMessage:
    return _create_typed_attachment_message(
        InvokedSkillsAttachmentPayload(skills=skills)
    )


def create_todo_reminder_attachment_message(content: List[Dict[str, Any]]) -> AttachmentMessage:
    return _create_typed_attachment_message(
        TodoReminderAttachmentPayload(content=content)
    )


def create_task_reminder_attachment_message(content: List[Dict[str, Any]]) -> AttachmentMessage:
    return _create_typed_attachment_message(
        TaskReminderAttachmentPayload(content=content)
    )


def create_nested_memory_attachment_message(content: Dict[str, Any]) -> AttachmentMessage:
    return _create_typed_attachment_message(
        NestedMemoryAttachmentPayload(content=content)
    )


def create_relevant_memories_attachment_message(
    memories: List[Dict[str, Any]],
) -> AttachmentMessage:
    return _create_typed_attachment_message(
        RelevantMemoriesAttachmentPayload(memories=memories)
    )


def create_skill_listing_attachment_message(content: str) -> AttachmentMessage:
    return _create_typed_attachment_message(
        SkillListingAttachmentPayload(content=content)
    )


def create_queued_command_attachment_message(
    prompt: Any,
    *,
    command_mode: Optional[str] = None,
) -> AttachmentMessage:
    return _create_typed_attachment_message(
        QueuedCommandAttachmentPayload(prompt=prompt, commandMode=command_mode)
    )


def create_ultramemory_attachment_message(content: Any) -> AttachmentMessage:
    return _create_typed_attachment_message(
        UltramemoryAttachmentPayload(content=content)
    )


def create_mcp_resource_attachment_message(
    *,
    server: str,
    uri: str,
    content: Any,
) -> AttachmentMessage:
    return _create_typed_attachment_message(
        McpResourceAttachmentPayload(server=server, uri=uri, content=content)
    )


def create_agent_mention_attachment_message(agent_type: str) -> AttachmentMessage:
    return _create_typed_attachment_message(
        AgentMentionAttachmentPayload(agentType=agent_type)
    )


def create_output_style_attachment_message(style: str) -> AttachmentMessage:
    return _create_typed_attachment_message(
        OutputStyleAttachmentPayload(style=style)
    )


def create_task_status_attachment_message(
    *,
    status: str,
    description: str,
    task_id: str,
    task_type: str = "",
    delta_summary: str = "",
) -> AttachmentMessage:
    return _create_typed_attachment_message(
        TaskStatusAttachmentPayload(
            status=status,
            description=description,
            taskId=task_id,
            taskType=task_type,
            deltaSummary=delta_summary,
        )
    )


def create_task_progress_attachment_message(message: str) -> AttachmentMessage:
    return _create_typed_attachment_message(
        TaskProgressAttachmentPayload(message=message)
    )


def create_diagnostics_attachment_message(files: List[Dict[str, Any]]) -> AttachmentMessage:
    return _create_typed_attachment_message(
        DiagnosticsAttachmentPayload(files=files)
    )


def create_critical_system_reminder_attachment_message(content: str) -> AttachmentMessage:
    return _create_typed_attachment_message(
        CriticalSystemReminderAttachmentPayload(content=content)
    )


def create_date_change_attachment_message(new_date: str) -> AttachmentMessage:
    return _create_typed_attachment_message(
        DateChangeAttachmentPayload(newDate=new_date)
    )


def create_token_usage_attachment_message(*, used: Any, total: Any, remaining: Any) -> AttachmentMessage:
    return _create_typed_attachment_message(
        TokenUsageAttachmentPayload(used=used, total=total, remaining=remaining)
    )


def create_budget_usd_attachment_message(*, used: Any, total: Any, remaining: Any) -> AttachmentMessage:
    return _create_typed_attachment_message(
        BudgetUsdAttachmentPayload(used=used, total=total, remaining=remaining)
    )


def create_hook_additional_context_message(
    content: Any,
    *,
    hook_name: str,
    hook_event: str,
    parent_tool_use_id: Optional[str] = None,
) -> Optional[AttachmentMessage]:
    """Create a hook additional-context attachment."""
    content_text = _normalize_hook_additional_context(content)
    if not content_text:
        return None
    if hook_event == "UserPromptSubmit":
        content_text = truncate_hook_context(content_text)

    return _create_typed_attachment_message(
        HookAdditionalContextAttachmentPayload(
            hook_name=hook_name,
            hook_event=hook_event,
            content=content_text,
            metadata={
                "hook_additional_context": True,
                "hook_event": hook_event,
                "hook_name": hook_name,
            },
            parent_tool_use_id=parent_tool_use_id,
        )
    )


def create_plan_mode_attachment_message(
    content: str,
    *,
    plan_file_path: str,
    reminder_type: str,
    attachment_type: str = "plan_mode",
    plan_exists: bool | None = None,
) -> AttachmentMessage:
    """Create a plan-mode attachment entry."""
    return _create_typed_attachment_message(
        PlanModeAttachmentPayload(
            type=attachment_type,
            content=content,
            plan_file_path=plan_file_path,
            plan_exists=plan_exists,
            reminder_type=reminder_type,
            metadata={
                "plan_mode_attachment": True,
                "plan_mode_attachment_type": attachment_type,
                "plan_mode_reminder_type": reminder_type,
                "plan_file_path": plan_file_path,
                "plan_exists": plan_exists,
            },
        )
    )


def create_async_hook_response_attachment_message(response: Dict[str, Any]) -> AttachmentMessage:
    return _create_typed_attachment_message(
        AsyncHookResponseAttachmentPayload(response=response)
    )


def create_hook_blocking_error_attachment_message(
    *,
    hook_name: str,
    blocking_error: Dict[str, Any],
) -> AttachmentMessage:
    return _create_typed_attachment_message(
        HookBlockingErrorAttachmentPayload(
            hookName=hook_name,
            blockingError=blocking_error,
        )
    )


def create_hook_success_attachment_message(
    *,
    hook_name: str,
    hook_event: str,
    content: str,
) -> AttachmentMessage:
    return _create_typed_attachment_message(
        HookSuccessAttachmentPayload(
            hookName=hook_name,
            hookEvent=hook_event,
            content=content,
        )
    )


def create_hook_stopped_continuation_attachment_message(
    *,
    hook_name: str,
    message: str,
) -> AttachmentMessage:
    return _create_typed_attachment_message(
        HookStoppedContinuationAttachmentPayload(
            hookName=hook_name,
            message=message,
        )
    )


def create_compaction_reminder_attachment_message() -> AttachmentMessage:
    return _create_typed_attachment_message(CompactionReminderAttachmentPayload())


def create_verify_plan_reminder_attachment_message() -> AttachmentMessage:
    return _create_typed_attachment_message(VerifyPlanReminderAttachmentPayload())


def is_hidden_meta_message(message: Any) -> bool:
    """Return whether a message is a model-visible meta attachment hidden from the UI."""

    if getattr(message, "type", None) == "attachment":
        return True
    message_payload = getattr(message, "message", None)
    metadata = getattr(message_payload, "metadata", None) if message_payload is not None else None
    metadata = metadata if isinstance(metadata, dict) else {}
    return bool(metadata.get("hook_additional_context") or metadata.get("plan_mode_attachment"))


def _create_attachment_user_message(
    content: Union[str, List[Dict[str, Any]]],
    *,
    attachment: AttachmentMessage,
) -> UserMessage:
    rendered_content: Union[str, List[MessageContent]]
    if isinstance(content, str):
        rendered_content = content
    else:
        rendered_content = [
            MessageContent(**item) if isinstance(item, dict) else item
            for item in content
        ]

    rendered = Message(
        role=MessageRole.USER,
        content=rendered_content,
        metadata=dict(attachment.metadata or {}),
    )
    return UserMessage(
        message=rendered,
        uuid=attachment.uuid,
        parent_tool_use_id=attachment.parent_tool_use_id,
        is_meta=True,
        timestamp=attachment.timestamp,
    )


def _render_attachment_system_message(
    content: Union[str, List[Dict[str, Any]]],
    *,
    attachment: AttachmentMessage,
) -> List[UserMessage]:
    return render_system_messages([_create_attachment_user_message(content, attachment=attachment)])


def _render_attachment_pre_wrapped_message(
    content: Union[str, List[Dict[str, Any]]],
    *,
    attachment: AttachmentMessage,
) -> List[UserMessage]:
    return [_create_attachment_user_message(content, attachment=attachment)]


def _render_tool_call_and_result(
    *,
    attachment: AttachmentMessage,
    tool_name: str,
    tool_input: Dict[str, Any],
    tool_result: Any,
) -> List[UserMessage]:
    rendered_messages = [
        _create_attachment_user_message(
            f"Called the {tool_name} tool with the following input: {_json_serialize(tool_input)}",
            attachment=attachment,
        )
    ]

    if isinstance(tool_result, list):
        contains_image = any(
            isinstance(item, dict) and item.get("type") == "image" for item in tool_result
        )
        if contains_image:
            rendered_messages.append(
                _create_attachment_user_message(tool_result, attachment=attachment)
            )
            return render_system_messages(rendered_messages)

    rendered_messages.append(
        _create_attachment_user_message(
            f"Result of calling the {tool_name} tool: {_json_serialize(tool_result)}",
            attachment=attachment,
        )
    )
    return render_system_messages(rendered_messages)


def _render_multiple_system_messages(
    contents: Sequence[Union[str, List[Dict[str, Any]]]],
    *,
    attachment: AttachmentMessage,
) -> List[UserMessage]:
    messages = [_create_attachment_user_message(content, attachment=attachment) for content in contents]
    return render_system_messages(messages)


def _parse_hook_additional_context_attachment(attachment: AttachmentMessage) -> List[UserMessage]:
    hook_name = str(_attachment_attr(attachment, "hook_name", "") or "").strip()
    raw_content = _attachment_attr(attachment, "content", "")
    content = _normalize_hook_additional_context(raw_content)
    if not hook_name or not content:
        return []
    return _render_attachment_system_message(
        f"{hook_name} hook additional context: {content}",
        attachment=attachment,
    )


def _parse_plan_mode_attachment(attachment: AttachmentMessage) -> List[UserMessage]:
    content = str(_attachment_attr(attachment, "content", "") or "").strip()
    if not content:
        return []
    return _render_attachment_system_message(content, attachment=attachment)


def _parse_plan_mode_reentry_attachment(attachment: AttachmentMessage) -> List[UserMessage]:
    plan_file_path = str(_attachment_attr(attachment, "plan_file_path", "") or "").strip()
    if not plan_file_path:
        return []
    content = f"""## Re-entering Plan Mode

You are returning to plan mode after having previously exited it. A plan file exists at {plan_file_path} from your previous planning session.

**Before proceeding with any new planning, you should:**
1. Read the existing plan file to understand what was previously planned
2. Evaluate the user's current request against that plan
3. Decide how to proceed:
   - **Different task**: If the user's request is for a different task-even if it's similar or related-start fresh by overwriting the existing plan
   - **Same task, continuing**: If this is explicitly a continuation or refinement of the exact same task, modify the existing plan while cleaning up outdated or irrelevant sections
4. Continue on with the plan process and most importantly you should always edit the plan file one way or the other before calling ExitPlanMode

Treat this as a fresh planning session. Do not assume the existing plan is relevant without evaluating it first."""
    return _render_attachment_system_message(content, attachment=attachment)


def _parse_plan_mode_exit_attachment(attachment: AttachmentMessage) -> List[UserMessage]:
    plan_exists = bool(_attachment_attr(attachment, "plan_exists", None))
    plan_file_path = str(_attachment_attr(attachment, "plan_file_path", "") or "").strip()
    suffix = f" The plan file is located at {plan_file_path} if you need to reference it." if plan_exists and plan_file_path else ""
    content = f"## Exited Plan Mode\n\nYou have exited plan mode. You can now make edits, run tools, and take actions.{suffix}"
    return _render_attachment_system_message(content, attachment=attachment)


def _parse_critical_system_reminder_attachment(attachment: AttachmentMessage) -> List[UserMessage]:
    content = str(_attachment_attr(attachment, "content", "") or "").strip()
    if not content:
        return []
    return _render_attachment_system_message(content, attachment=attachment)


def _parse_directory_attachment(attachment: AttachmentMessage) -> List[UserMessage]:
    path = str(_attachment_attr(attachment, "path", "") or "").strip()
    content = _attachment_attr(attachment, "content", "")
    if not path:
        return []
    return _render_tool_call_and_result(
        attachment=attachment,
        tool_name="LS",
        tool_input={"path": path},
        tool_result=content,
    )


def _parse_edited_text_file_attachment(attachment: AttachmentMessage) -> List[UserMessage]:
    filename = str(_attachment_attr(attachment, "filename", "") or "").strip()
    snippet = str(_attachment_attr(attachment, "snippet", "") or "")
    if not filename:
        return []
    content = (
        f"Note: {filename} was modified, either by the user or by a linter. "
        "This change was intentional, so make sure to take it into account as you proceed "
        "(ie. don't revert it unless the user asks you to). Don't tell the user this, "
        "since they are already aware. Here are the relevant changes (shown with line numbers):\n"
        f"{snippet}"
    )
    return _render_attachment_system_message(content, attachment=attachment)


def _parse_file_attachment(attachment: AttachmentMessage) -> List[UserMessage]:
    filename = str(_attachment_attr(attachment, "filename", "") or "").strip()
    file_content = _attachment_attr(attachment, "content", None)
    truncated = bool(_attachment_attr(attachment, "truncated", False))
    if not filename or file_content is None:
        return []

    content_type = str(getattr(file_content, "type", "") or "")
    tool_result: Any = file_content
    if hasattr(file_content, "model_dump"):
        tool_result = file_content.model_dump(mode="json")

    result = _render_tool_call_and_result(
        attachment=attachment,
        tool_name="Read",
        tool_input={"file_path": filename},
        tool_result=tool_result,
    )

    if content_type == "text" and truncated:
        result.extend(
            _render_attachment_system_message(
                (
                    f"Note: The file {filename} was too large and has been truncated to the first "
                    f"{FILE_ATTACHMENT_TRUNCATION_LINE_LIMIT} lines. Don't tell the user about this "
                    "truncation. Use the Read tool to read more of the file if you need."
                ),
                attachment=attachment,
            )
        )
    return result


def _parse_compact_file_reference_attachment(attachment: AttachmentMessage) -> List[UserMessage]:
    filename = str(_attachment_attr(attachment, "filename", "") or "").strip()
    if not filename:
        return []
    return _render_attachment_system_message(
        (
            f"Note: {filename} was read before the last conversation was summarized, "
            "but the contents are too large to include. Use the Read tool if you need to access it."
        ),
        attachment=attachment,
    )


def _parse_pdf_reference_attachment(attachment: AttachmentMessage) -> List[UserMessage]:
    filename = str(_attachment_attr(attachment, "filename", "") or "").strip()
    page_count = _attachment_attr(attachment, "pageCount", 0)
    file_size = _humanize_bytes(_attachment_attr(attachment, "fileSize", 0))
    if not filename:
        return []
    return _render_attachment_system_message(
        (
            f"PDF file: {filename} ({page_count} pages, {file_size}). "
            "This PDF is too large to read all at once. You MUST use the Read tool with a page range "
            "to read specific page windows. Start by reading the first few pages to understand the structure, "
            "then read more as needed."
        ),
        attachment=attachment,
    )


def _parse_selected_lines_in_ide_attachment(attachment: AttachmentMessage) -> List[UserMessage]:
    filename = str(_attachment_attr(attachment, "filename", "") or "").strip()
    line_start = _attachment_attr(attachment, "lineStart", "")
    line_end = _attachment_attr(attachment, "lineEnd", "")
    raw_content = str(_attachment_attr(attachment, "content", "") or "")
    selected_content = raw_content[:2000] + ("\n... (truncated)" if len(raw_content) > 2000 else "")
    if not filename:
        return []
    content = (
        f"The user selected the lines {line_start} to {line_end} from {filename}:\n"
        f"{selected_content}\n\nThis may or may not be related to the current task."
    )
    return _render_attachment_system_message(content, attachment=attachment)


def _parse_opened_file_in_ide_attachment(attachment: AttachmentMessage) -> List[UserMessage]:
    filename = str(_attachment_attr(attachment, "filename", "") or "").strip()
    if not filename:
        return []
    return _render_attachment_system_message(
        f"The user opened the file {filename} in the IDE. This may or may not be related to the current task.",
        attachment=attachment,
    )


def _parse_todo_attachment(attachment: AttachmentMessage) -> List[UserMessage]:
    item_count = int(_attachment_attr(attachment, "itemCount", 0) or 0)
    content = _attachment_attr(attachment, "content", [])
    if item_count == 0:
        return _render_attachment_system_message(
            (
                "This is a reminder that your todo list is currently empty. DO NOT mention this to the user "
                "explicitly because they are already aware. If you are working on tasks that would benefit from "
                "a todo list please use the TodoWrite tool to create one. If not, please feel free to ignore."
            ),
            attachment=attachment,
        )
    return _render_attachment_system_message(
        (
            "Your todo list has changed. DO NOT mention this explicitly to the user. "
            f"Here are the latest contents of your todo list:\n\n{_json_serialize(content)}. "
            "Continue on with the tasks at hand if applicable."
        ),
        attachment=attachment,
    )


def _parse_plan_file_reference_attachment(attachment: AttachmentMessage) -> List[UserMessage]:
    plan_file_path = str(_attachment_attr(attachment, "planFilePath", "") or "").strip()
    plan_content = str(_attachment_attr(attachment, "planContent", "") or "")
    if not plan_file_path:
        return []
    return _render_attachment_system_message(
        (
            f"A plan file exists from plan mode at: {plan_file_path}\n\n"
            f"Plan contents:\n\n{plan_content}\n\n"
            "If this plan is relevant to the current work and not already complete, continue working on it."
        ),
        attachment=attachment,
    )


def _parse_invoked_skills_attachment(attachment: AttachmentMessage) -> List[UserMessage]:
    skills = _attachment_attr(attachment, "skills", []) or []
    if not skills:
        return []
    invoked_skills_text = "\n\n---\n\n".join(
        f"### Skill: {skill.get('name', '')}\nPath: {skill.get('path', '')}\n\n{skill.get('content', '')}"
        for skill in skills
    )
    return _render_attachment_system_message(
        f"The following skills were invoked in this session. Continue to follow these guidelines:\n\n{invoked_skills_text}",
        attachment=attachment,
    )


def _parse_todo_reminder_attachment(attachment: AttachmentMessage) -> List[UserMessage]:
    todos = _attachment_attr(attachment, "content", []) or []
    formatted = "\n".join(
        f"{index + 1}. [{item.get('status', '')}] {item.get('content', '')}"
        for index, item in enumerate(todos)
        if isinstance(item, dict)
    )
    message = (
        "The TodoWrite tool hasn't been used recently. If you're working on tasks that would benefit from "
        "tracking progress, consider using the TodoWrite tool to track progress. Also consider cleaning up the "
        "todo list if it has become stale and no longer matches what you are working on. Only use it if it's "
        "relevant to the current work. This is just a gentle reminder - ignore if not applicable. Make sure "
        "that you NEVER mention this reminder to the user"
    )
    if formatted:
        message += f"\n\nHere are the existing contents of your todo list:\n\n[{formatted}]"
    return _render_attachment_system_message(message, attachment=attachment)


def _parse_task_reminder_attachment(attachment: AttachmentMessage) -> List[UserMessage]:
    tasks = _attachment_attr(attachment, "content", []) or []
    task_list = "\n".join(
        f"#{task.get('id', '')}. [{task.get('status', '')}] {task.get('subject', '')}"
        for task in tasks
        if isinstance(task, dict)
    )
    message = (
        "The task tools haven't been used recently. If you're working on tasks that would benefit from "
        "tracking progress, consider using TaskCreate to add new tasks and TaskUpdate to update task status "
        "(set to in_progress when starting, completed when done). Also consider cleaning up the task list if "
        "it has become stale. Only use these if relevant to the current work. This is just a gentle reminder "
        "- ignore if not applicable. Make sure that you NEVER mention this reminder to the user"
    )
    if task_list:
        message += f"\n\nHere are the existing tasks:\n\n{task_list}"
    return _render_attachment_system_message(message, attachment=attachment)


def _parse_nested_memory_attachment(attachment: AttachmentMessage) -> List[UserMessage]:
    payload = _attachment_attr(attachment, "content", {}) or {}
    path = payload.get("path", "") if isinstance(payload, dict) else ""
    content = payload.get("content", "") if isinstance(payload, dict) else ""
    if not path:
        return []
    return _render_attachment_system_message(
        f"Contents of {path}:\n\n{content}",
        attachment=attachment,
    )


def _parse_relevant_memories_attachment(attachment: AttachmentMessage) -> List[UserMessage]:
    memories = _attachment_attr(attachment, "memories", []) or []
    contents = [
        f"Potentially relevant memory: {memory.get('path', '')}:\n\n{memory.get('content', '')}"
        for memory in memories
        if isinstance(memory, dict)
    ]
    if not contents:
        return []
    return _render_multiple_system_messages(contents, attachment=attachment)


def _parse_skill_listing_attachment(attachment: AttachmentMessage) -> List[UserMessage]:
    content = str(_attachment_attr(attachment, "content", "") or "")
    if not content:
        return []
    return _render_attachment_system_message(
        f"The following skills are available for use with the Skill tool:\n\n{content}",
        attachment=attachment,
    )


def _parse_queued_command_attachment(attachment: AttachmentMessage) -> List[UserMessage]:
    prompt = _attachment_attr(attachment, "prompt", "")
    if isinstance(prompt, list):
        text_content = "\n".join(
            str(item.get("text", "")).strip()
            for item in prompt
            if isinstance(item, dict) and item.get("type") == "text" and str(item.get("text", "")).strip()
        )
        image_elements = [
            item for item in prompt if isinstance(item, dict) and item.get("type") == "image"
        ]
        combined: List[Dict[str, Any]] = []
        if text_content:
            combined.append({"type": "text", "text": text_content})
        combined.extend(image_elements)
        if not combined:
            return []
        return _render_attachment_system_message(combined, attachment=attachment)
    if prompt is None:
        return []
    return _render_attachment_system_message(str(prompt), attachment=attachment)


def _parse_ultramemory_attachment(attachment: AttachmentMessage) -> List[UserMessage]:
    content = _attachment_attr(attachment, "content", "")
    if not content:
        return []
    return _render_attachment_system_message(str(content), attachment=attachment)


def _parse_output_style_attachment(attachment: AttachmentMessage) -> List[UserMessage]:
    style = str(_attachment_attr(attachment, "style", "") or "").strip()
    if not style:
        return []
    try:
        from ripperdoc.core.output_styles import resolve_output_style

        resolved_style, _ = resolve_output_style(style)
        style_name = resolved_style.name
    except Exception:
        style_name = style
    return _render_attachment_system_message(
        f"{style_name} output style is active. Remember to follow the specific guidelines for this style.",
        attachment=attachment,
    )


def _parse_diagnostics_attachment(attachment: AttachmentMessage) -> List[UserMessage]:
    files = _attachment_attr(attachment, "files", []) or []
    if not files:
        return []
    summary_lines: List[str] = []
    for file_entry in files:
        if not isinstance(file_entry, dict):
            summary_lines.append(str(file_entry))
            continue
        file_path = str(file_entry.get("file") or file_entry.get("path") or "(unknown)")
        issues = file_entry.get("diagnostics") or file_entry.get("issues") or []
        if isinstance(issues, list) and issues:
            summary_lines.append(f"{file_path}:")
            for issue in issues:
                if isinstance(issue, dict):
                    line = issue.get("line")
                    severity = issue.get("severity") or issue.get("level") or "issue"
                    message = issue.get("message") or issue.get("text") or str(issue)
                    prefix = f"  - line {line} [{severity}] " if line is not None else f"  - [{severity}] "
                    summary_lines.append(prefix + str(message))
                else:
                    summary_lines.append(f"  - {issue}")
        else:
            summary_lines.append(_json_serialize(file_entry))
    diagnostic_summary = "\n".join(summary_lines)
    return _render_attachment_system_message(
        f"<new-diagnostics>The following new diagnostic issues were detected:\n\n{diagnostic_summary}</new-diagnostics>",
        attachment=attachment,
    )


def _parse_mcp_resource_attachment(attachment: AttachmentMessage) -> List[UserMessage]:
    server = str(_attachment_attr(attachment, "server", "") or "")
    uri = str(_attachment_attr(attachment, "uri", "") or "")
    payload = _attachment_attr(attachment, "content", None)
    contents = payload.get("contents", []) if isinstance(payload, dict) else []
    if not contents:
        return _render_attachment_system_message(
            f'<mcp-resource server="{server}" uri="{uri}">(No content)</mcp-resource>',
            attachment=attachment,
        )
    message_blocks: List[Dict[str, Any]] = []
    for item in contents:
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("text"), str):
            message_blocks.extend(
                [
                    {"type": "text", "text": "Full contents of resource:"},
                    {"type": "text", "text": item["text"]},
                    {
                        "type": "text",
                        "text": "Do NOT read this resource again unless you think it may have changed, since you already have the full contents.",
                    },
                ]
            )
        elif "blob" in item:
            mime_type = str(item.get("mimeType") or "application/octet-stream")
            message_blocks.append({"type": "text", "text": f"[Binary content: {mime_type}]"})
    if message_blocks:
        return _render_attachment_system_message(message_blocks, attachment=attachment)
    return _render_attachment_system_message(
        f'<mcp-resource server="{server}" uri="{uri}">(No displayable content)</mcp-resource>',
        attachment=attachment,
    )


def _parse_agent_mention_attachment(attachment: AttachmentMessage) -> List[UserMessage]:
    agent_type = str(_attachment_attr(attachment, "agentType", "") or "").strip()
    if not agent_type:
        return []
    return _render_attachment_system_message(
        f'The user has expressed a desire to invoke the agent "{agent_type}". Please invoke the agent appropriately, passing in the required context to it.',
        attachment=attachment,
    )


def _parse_task_status_attachment(attachment: AttachmentMessage) -> List[UserMessage]:
    status = str(_attachment_attr(attachment, "status", "") or "")
    description = str(_attachment_attr(attachment, "description", "") or "")
    task_id = str(_attachment_attr(attachment, "taskId", "") or "")
    task_type = str(_attachment_attr(attachment, "taskType", "") or "")
    delta_summary = str(_attachment_attr(attachment, "deltaSummary", "") or "")
    if status == "killed":
        return _render_attachment_pre_wrapped_message(
            system_reminder_wrapper(f'Task "{description}" ({task_id}) was stopped by the user.'),
            attachment=attachment,
        )
    parts = [
        f"Task {task_id}",
        f"(type: {task_type})",
        f"(status: {status})",
        f"(description: {description})",
    ]
    if delta_summary:
        parts.append(f"Delta: {delta_summary}")
    parts.append("You can check its output using the TaskOutput tool.")
    return _render_attachment_pre_wrapped_message(
        system_reminder_wrapper(" ".join(parts)),
        attachment=attachment,
    )


def _parse_task_progress_attachment(attachment: AttachmentMessage) -> List[UserMessage]:
    message = str(_attachment_attr(attachment, "message", "") or "").strip()
    if not message:
        return []
    return _render_attachment_pre_wrapped_message(
        system_reminder_wrapper(message),
        attachment=attachment,
    )


def _parse_async_hook_response_attachment(attachment: AttachmentMessage) -> List[UserMessage]:
    response = _attachment_attr(attachment, "response", {}) or {}
    contents: List[str] = []
    if isinstance(response, dict):
        system_message = response.get("systemMessage")
        if system_message:
            contents.append(str(system_message))
        hook_specific_output = response.get("hookSpecificOutput")
        if isinstance(hook_specific_output, dict) and hook_specific_output.get("additionalContext"):
            contents.append(str(hook_specific_output["additionalContext"]))
    if not contents:
        return []
    return _render_multiple_system_messages(contents, attachment=attachment)


def _parse_token_usage_attachment(attachment: AttachmentMessage) -> List[UserMessage]:
    used = _attachment_attr(attachment, "used", "")
    total = _attachment_attr(attachment, "total", "")
    remaining = _attachment_attr(attachment, "remaining", "")
    return _render_attachment_pre_wrapped_message(
        system_reminder_wrapper(f"Token usage: {used}/{total}; {remaining} remaining"),
        attachment=attachment,
    )


def _parse_budget_usd_attachment(attachment: AttachmentMessage) -> List[UserMessage]:
    used = _attachment_attr(attachment, "used", "")
    total = _attachment_attr(attachment, "total", "")
    remaining = _attachment_attr(attachment, "remaining", "")
    return _render_attachment_pre_wrapped_message(
        system_reminder_wrapper(f"USD budget: ${used}/${total}; ${remaining} remaining"),
        attachment=attachment,
    )


def _parse_hook_blocking_error_attachment(attachment: AttachmentMessage) -> List[UserMessage]:
    hook_name = str(_attachment_attr(attachment, "hookName", "") or "")
    blocking_error = _attachment_attr(attachment, "blockingError", {}) or {}
    command = blocking_error.get("command", "") if isinstance(blocking_error, dict) else ""
    error = blocking_error.get("blockingError", "") if isinstance(blocking_error, dict) else ""
    return _render_attachment_pre_wrapped_message(
        system_reminder_wrapper(
            f'{hook_name} hook blocking error from command: "{command}": {error}'
        ),
        attachment=attachment,
    )


def _parse_hook_success_attachment(attachment: AttachmentMessage) -> List[UserMessage]:
    hook_event = str(_attachment_attr(attachment, "hookEvent", "") or "")
    if hook_event not in {"SessionStart", "UserPromptSubmit"}:
        return []
    content = str(_attachment_attr(attachment, "content", "") or "")
    hook_name = str(_attachment_attr(attachment, "hookName", "") or "")
    if not content:
        return []
    return _render_attachment_pre_wrapped_message(
        system_reminder_wrapper(f"{hook_name} hook success: {content}"),
        attachment=attachment,
    )


def _parse_hook_stopped_continuation_attachment(attachment: AttachmentMessage) -> List[UserMessage]:
    hook_name = str(_attachment_attr(attachment, "hookName", "") or "")
    message = str(_attachment_attr(attachment, "message", "") or "")
    if not message:
        return []
    return _render_attachment_pre_wrapped_message(
        system_reminder_wrapper(f"{hook_name} hook stopped continuation: {message}"),
        attachment=attachment,
    )


def _parse_compaction_reminder_attachment(attachment: AttachmentMessage) -> List[UserMessage]:
    return _render_attachment_system_message(
        "Auto-compact is enabled. When the context window is nearly full, older messages will be automatically summarized so you can continue working seamlessly. There is no need to stop or rush - you have unlimited context through automatic compaction.",
        attachment=attachment,
    )


def _parse_date_change_attachment(attachment: AttachmentMessage) -> List[UserMessage]:
    new_date = str(_attachment_attr(attachment, "newDate", "") or "").strip()
    if not new_date:
        return []
    return _render_attachment_system_message(
        f"The date has changed. Today's date is now {new_date}. DO NOT mention this to the user explicitly because they are already aware.",
        attachment=attachment,
    )


def _parse_verify_plan_reminder_attachment(attachment: AttachmentMessage) -> List[UserMessage]:
    return _render_attachment_system_message(
        'You have completed implementing the plan. Please call the "" tool directly to verify that all plan items were completed correctly.',
        attachment=attachment,
    )


def parse_attachment_message(attachment: AttachmentMessage) -> List[UserMessage]:
    """Central attachment dispatcher for attachment message parsing."""

    attachment_type = str(getattr(attachment.attachment, "type", "") or "")

    if attachment_type in ATTACHMENT_IGNORED_TYPES:
        return []
    if attachment_type == "directory":
        return _parse_directory_attachment(attachment)
    if attachment_type == "edited_text_file":
        return _parse_edited_text_file_attachment(attachment)
    if attachment_type == "file":
        return _parse_file_attachment(attachment)
    if attachment_type == "compact_file_reference":
        return _parse_compact_file_reference_attachment(attachment)
    if attachment_type == "pdf_reference":
        return _parse_pdf_reference_attachment(attachment)
    if attachment_type == "selected_lines_in_ide":
        return _parse_selected_lines_in_ide_attachment(attachment)
    if attachment_type == "opened_file_in_ide":
        return _parse_opened_file_in_ide_attachment(attachment)
    if attachment_type == "todo":
        return _parse_todo_attachment(attachment)
    if attachment_type == "plan_file_reference":
        return _parse_plan_file_reference_attachment(attachment)
    if attachment_type == "invoked_skills":
        return _parse_invoked_skills_attachment(attachment)
    if attachment_type == "todo_reminder":
        return _parse_todo_reminder_attachment(attachment)
    if attachment_type == "task_reminder":
        return _parse_task_reminder_attachment(attachment)
    if attachment_type == "nested_memory":
        return _parse_nested_memory_attachment(attachment)
    if attachment_type == "relevant_memories":
        return _parse_relevant_memories_attachment(attachment)
    if attachment_type == "skill_listing":
        return _parse_skill_listing_attachment(attachment)
    if attachment_type == "queued_command":
        return _parse_queued_command_attachment(attachment)
    if attachment_type == "ultramemory":
        return _parse_ultramemory_attachment(attachment)
    if attachment_type == "output_style":
        return _parse_output_style_attachment(attachment)
    if attachment_type == "diagnostics":
        return _parse_diagnostics_attachment(attachment)
    if attachment_type == "plan_mode":
        return _parse_plan_mode_attachment(attachment)
    if attachment_type == "plan_mode_reentry":
        return _parse_plan_mode_reentry_attachment(attachment)
    if attachment_type == "plan_mode_exit":
        return _parse_plan_mode_exit_attachment(attachment)
    if attachment_type == "critical_system_reminder":
        return _parse_critical_system_reminder_attachment(attachment)
    if attachment_type == "mcp_resource":
        return _parse_mcp_resource_attachment(attachment)
    if attachment_type == "agent_mention":
        return _parse_agent_mention_attachment(attachment)
    if attachment_type == "task_status":
        return _parse_task_status_attachment(attachment)
    if attachment_type == "task_progress":
        return _parse_task_progress_attachment(attachment)
    if attachment_type == "async_hook_response":
        return _parse_async_hook_response_attachment(attachment)
    if attachment_type == "token_usage":
        return _parse_token_usage_attachment(attachment)
    if attachment_type == "budget_usd":
        return _parse_budget_usd_attachment(attachment)
    if attachment_type == "hook_blocking_error":
        return _parse_hook_blocking_error_attachment(attachment)
    if attachment_type == "hook_success":
        return _parse_hook_success_attachment(attachment)
    if attachment_type == "hook_additional_context":
        return _parse_hook_additional_context_attachment(attachment)
    if attachment_type == "hook_stopped_continuation":
        return _parse_hook_stopped_continuation_attachment(attachment)
    if attachment_type == "compaction_reminder":
        return _parse_compaction_reminder_attachment(attachment)
    if attachment_type == "date_change":
        return _parse_date_change_attachment(attachment)
    if attachment_type == "verify_plan_reminder":
        return _parse_verify_plan_reminder_attachment(attachment)

    logger.warning("[messages] Unknown attachment type for API normalization: %s", attachment_type)
    return []


def render_attachment_message(message: AttachmentMessage) -> UserMessage:
    """Render a single attachment item into the model-visible user/meta message form."""

    rendered_messages = parse_attachment_message(message)
    if rendered_messages:
        return rendered_messages[0]
    fallback = _render_attachment_system_message(
        str(getattr(message.attachment, "content", "") or ""),
        attachment=message,
    )
    return cast(UserMessage, fallback[0])


def expand_attachment_messages(
    messages: Sequence[UserMessage | AssistantMessage | ProgressMessage | AttachmentMessage],
) -> List[UserMessage | AssistantMessage | ProgressMessage | AttachmentMessage]:
    """Expand attachment items into model-visible user/meta messages."""

    expanded: List[UserMessage | AssistantMessage | ProgressMessage | AttachmentMessage] = []
    for message in messages:
        if getattr(message, "type", None) == "attachment" and isinstance(message, AttachmentMessage):
            expanded.extend(parse_attachment_message(message))
            continue
        expanded.append(message)
    return expanded


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
    messages: List[Union[UserMessage, AssistantMessage, ProgressMessage, AttachmentMessage]],
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
        expand_attachment_messages(messages),
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
