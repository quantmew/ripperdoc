"""Attachment type definitions with discriminated unions.

All AttachmentPayload subclasses use Literal type fields as discriminators,
enabling Pydantic v2's native discriminated union validation for known types.
Unknown types fall back to UnknownAttachmentPayload with extra="allow".
"""

from __future__ import annotations

from typing import (
    Annotated,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Set,
    Type,
    Union,
    cast,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
)


class AttachmentPayload(BaseModel):
    """Base attachment payload."""

    type: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    parent_tool_use_id: Optional[str] = None
    model_config = ConfigDict(extra="forbid")


class UnknownAttachmentPayload(AttachmentPayload):
    """Fallback payload for attachments without a dedicated model."""

    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------------------
# Specific attachment payload types with Literal discriminators
# ---------------------------------------------------------------------------


class HookAdditionalContextAttachmentPayload(AttachmentPayload):
    type: Literal["hook_additional_context"] = "hook_additional_context"
    hook_name: str
    hook_event: str
    content: Any


class PlanModeAttachmentPayload(AttachmentPayload):
    type: Literal["plan_mode", "plan_mode_reentry", "plan_mode_exit"]
    content: str
    plan_file_path: str
    plan_exists: Optional[bool] = None
    reminder_type: str


class DirectoryAttachmentPayload(AttachmentPayload):
    type: Literal["directory"] = "directory"
    path: str
    content: str


class EditedTextFileAttachmentPayload(AttachmentPayload):
    type: Literal["edited_text_file"] = "edited_text_file"
    filename: str
    snippet: str


# ---------------------------------------------------------------------------
# FileAttachmentContent hierarchy
# ---------------------------------------------------------------------------


class FileAttachmentContent(BaseModel):
    type: str
    model_config = ConfigDict(extra="allow")


class UnknownFileAttachmentContent(FileAttachmentContent):
    type: Literal["unknown"] = "unknown"


class FileTextAttachmentContent(FileAttachmentContent):
    type: Literal["text"] = "text"


class FileImageAttachmentContent(FileAttachmentContent):
    type: Literal["image"] = "image"


class FileNotebookAttachmentContent(FileAttachmentContent):
    type: Literal["notebook"] = "notebook"


class FilePdfAttachmentContent(FileAttachmentContent):
    type: Literal["pdf"] = "pdf"


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
    type: Literal["file"] = "file"
    filename: str
    content: FileAttachmentContentModel
    truncated: bool = False

    @field_validator("content", mode="before")
    @classmethod
    def validate_content(cls, value: Any) -> FileAttachmentContentModel:
        return _coerce_file_attachment_content(value)


class CompactFileReferenceAttachmentPayload(AttachmentPayload):
    type: Literal["compact_file_reference"] = "compact_file_reference"
    filename: str


class PdfReferenceAttachmentPayload(AttachmentPayload):
    type: Literal["pdf_reference"] = "pdf_reference"
    filename: str
    pageCount: int
    fileSize: Union[int, float]


class SelectedLinesInIdeAttachmentPayload(AttachmentPayload):
    type: Literal["selected_lines_in_ide"] = "selected_lines_in_ide"
    filename: str
    lineStart: int
    lineEnd: int
    content: str


class OpenedFileInIdeAttachmentPayload(AttachmentPayload):
    type: Literal["opened_file_in_ide"] = "opened_file_in_ide"
    filename: str


class TodoAttachmentPayload(AttachmentPayload):
    type: Literal["todo"] = "todo"
    itemCount: int = 0
    content: Any = Field(default_factory=list)


class PlanFileReferenceAttachmentPayload(AttachmentPayload):
    type: Literal["plan_file_reference"] = "plan_file_reference"
    planFilePath: str
    planContent: str


class InvokedSkillsAttachmentPayload(AttachmentPayload):
    type: Literal["invoked_skills"] = "invoked_skills"
    skills: List[Dict[str, Any]] = Field(default_factory=list)


class TodoReminderAttachmentPayload(AttachmentPayload):
    type: Literal["todo_reminder"] = "todo_reminder"
    content: List[Dict[str, Any]] = Field(default_factory=list)


class TaskReminderAttachmentPayload(AttachmentPayload):
    type: Literal["task_reminder"] = "task_reminder"
    content: List[Dict[str, Any]] = Field(default_factory=list)


class NestedMemoryAttachmentPayload(AttachmentPayload):
    type: Literal["nested_memory"] = "nested_memory"
    content: Dict[str, Any]


class RelevantMemoriesAttachmentPayload(AttachmentPayload):
    type: Literal["relevant_memories"] = "relevant_memories"
    memories: List[Dict[str, Any]] = Field(default_factory=list)


class SkillListingAttachmentPayload(AttachmentPayload):
    type: Literal["skill_listing"] = "skill_listing"
    content: str


class QueuedCommandAttachmentPayload(AttachmentPayload):
    type: Literal["queued_command"] = "queued_command"
    prompt: Any
    commandMode: Optional[str] = None


class UltramemoryAttachmentPayload(AttachmentPayload):
    type: Literal["ultramemory"] = "ultramemory"
    content: Any


class McpResourceAttachmentPayload(AttachmentPayload):
    type: Literal["mcp_resource"] = "mcp_resource"
    server: str
    uri: str
    content: Any


class AgentMentionAttachmentPayload(AttachmentPayload):
    type: Literal["agent_mention"] = "agent_mention"
    agentType: str


class OutputStyleAttachmentPayload(AttachmentPayload):
    type: Literal["output_style"] = "output_style"
    style: str


class TaskStatusAttachmentPayload(AttachmentPayload):
    type: Literal["task_status"] = "task_status"
    status: str
    description: str
    taskId: str
    taskType: str = ""
    deltaSummary: str = ""


class TaskProgressAttachmentPayload(AttachmentPayload):
    type: Literal["task_progress"] = "task_progress"
    message: str


class DiagnosticsAttachmentPayload(AttachmentPayload):
    type: Literal["diagnostics"] = "diagnostics"
    files: List[Dict[str, Any]]


class CriticalSystemReminderAttachmentPayload(AttachmentPayload):
    type: Literal["critical_system_reminder"] = "critical_system_reminder"
    content: str


class DateChangeAttachmentPayload(AttachmentPayload):
    type: Literal["date_change"] = "date_change"
    newDate: str


class TokenUsageAttachmentPayload(AttachmentPayload):
    type: Literal["token_usage"] = "token_usage"
    used: Any
    total: Any
    remaining: Any


class BudgetUsdAttachmentPayload(AttachmentPayload):
    type: Literal["budget_usd"] = "budget_usd"
    used: Any
    total: Any
    remaining: Any


class AsyncHookResponseAttachmentPayload(AttachmentPayload):
    type: Literal["async_hook_response"] = "async_hook_response"
    response: Dict[str, Any]


class HookBlockingErrorAttachmentPayload(AttachmentPayload):
    type: Literal["hook_blocking_error"] = "hook_blocking_error"
    hookName: str
    blockingError: Dict[str, Any]


class HookSuccessAttachmentPayload(AttachmentPayload):
    type: Literal["hook_success"] = "hook_success"
    hookName: str
    hookEvent: str
    content: str


class HookStoppedContinuationAttachmentPayload(AttachmentPayload):
    type: Literal["hook_stopped_continuation"] = "hook_stopped_continuation"
    hookName: str
    message: str


class CompactionReminderAttachmentPayload(AttachmentPayload):
    type: Literal["compaction_reminder"] = "compaction_reminder"


class VerifyPlanReminderAttachmentPayload(AttachmentPayload):
    type: Literal["verify_plan_reminder"] = "verify_plan_reminder"


class DynamicSkillAttachmentPayload(AttachmentPayload):
    type: Literal["dynamic_skill"] = "dynamic_skill"


class AlreadyReadFileAttachmentPayload(AttachmentPayload):
    type: Literal["already_read_file"] = "already_read_file"
    filename: Optional[str] = None


class CommandPermissionsAttachmentPayload(AttachmentPayload):
    type: Literal["command_permissions"] = "command_permissions"
    permissions: Any = None


class EditedImageFileAttachmentPayload(AttachmentPayload):
    type: Literal["edited_image_file"] = "edited_image_file"
    filename: Optional[str] = None


class HookCancelledAttachmentPayload(AttachmentPayload):
    type: Literal["hook_cancelled"] = "hook_cancelled"
    hookName: Optional[str] = None
    hookEvent: Optional[str] = None
    reason: Optional[str] = None


class HookErrorDuringExecutionAttachmentPayload(AttachmentPayload):
    type: Literal["hook_error_during_execution"] = "hook_error_during_execution"
    hookName: Optional[str] = None
    hookEvent: Optional[str] = None
    error: Optional[str] = None


class HookNonBlockingErrorAttachmentPayload(AttachmentPayload):
    type: Literal["hook_non_blocking_error"] = "hook_non_blocking_error"
    hookName: Optional[str] = None
    hookEvent: Optional[str] = None
    error: Optional[str] = None


class HookSystemMessageAttachmentPayload(AttachmentPayload):
    type: Literal["hook_system_message"] = "hook_system_message"
    hookName: Optional[str] = None
    hookEvent: Optional[str] = None
    systemMessage: Optional[str] = None


class StructuredOutputAttachmentPayload(AttachmentPayload):
    type: Literal["structured_output"] = "structured_output"
    content: Any = None


class HookPermissionDecisionAttachmentPayload(AttachmentPayload):
    type: Literal["hook_permission_decision"] = "hook_permission_decision"
    hookName: Optional[str] = None
    hookEvent: Optional[str] = None
    decision: Any = None


class AutocheckpointingAttachmentPayload(AttachmentPayload):
    type: Literal["autocheckpointing"] = "autocheckpointing"
    content: Any = None


class BackgroundTaskStatusAttachmentPayload(AttachmentPayload):
    type: Literal["background_task_status"] = "background_task_status"
    taskId: Optional[str] = None
    status: Optional[str] = None
    content: Any = None


# ---------------------------------------------------------------------------
# Discriminated union for known types (validated via TypeAdapter)
# ---------------------------------------------------------------------------

_KnownAttachmentPayloadModel = Annotated[
    Union[
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
    ],
    Discriminator("type"),
]

_KNOWN_PAYLOAD_ADAPTER: TypeAdapter[Any] = TypeAdapter(_KnownAttachmentPayloadModel)

# Full union type for annotations (includes Unknown for type hints)
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
    try:
        return cast(AttachmentPayloadModel, _KNOWN_PAYLOAD_ADAPTER.validate_python(payload))
    except ValidationError:
        return cast(AttachmentPayloadModel, UnknownAttachmentPayload(**payload))


# ---------------------------------------------------------------------------
# Attachment type classification constants
# ---------------------------------------------------------------------------

ATTACHMENT_IGNORED_TYPES: Set[str] = {
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

ATTACHMENT_EXPORT_HIDDEN_TYPES: Set[str] = {
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

ATTACHMENT_SUMMARY_HIDDEN_TYPES: Set[str] = {
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
