"""Message handling and formatting for Ripperdoc.

This module keeps the legacy import surface while implementation details live in
focused messaging submodules.
"""

# ruff: noqa: F401

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Set, Union

from ripperdoc.utils.log import get_logger
from ripperdoc.utils.messaging.attachments import (  # noqa: F401 — legacy re-export surface
    create_agent_mention_attachment_message,
    create_attachment_message,
    create_budget_usd_attachment_message,
    create_compact_file_reference_attachment_message,
    create_compaction_reminder_attachment_message,
    create_critical_system_reminder_attachment_message,
    create_date_change_attachment_message,
    create_diagnostics_attachment_message,
    create_directory_attachment_message,
    create_edited_text_file_attachment_message,
    create_file_attachment_message,
    create_hook_additional_context_message,
    create_hook_blocking_error_attachment_message,
    create_hook_notice_message,
    create_hook_notice_payload,
    create_hook_stopped_continuation_attachment_message,
    create_hook_success_attachment_message,
    create_invoked_skills_attachment_message,
    create_mcp_resource_attachment_message,
    create_nested_memory_attachment_message,
    create_opened_file_in_ide_attachment_message,
    create_output_style_attachment_message,
    create_pdf_reference_attachment_message,
    create_plan_file_reference_attachment_message,
    create_plan_mode_attachment_message,
    create_queued_command_attachment_message,
    create_relevant_memories_attachment_message,
    create_selected_lines_in_ide_attachment_message,
    create_skill_listing_attachment_message,
    create_task_progress_attachment_message,
    create_task_reminder_attachment_message,
    create_task_status_attachment_message,
    create_todo_attachment_message,
    create_todo_reminder_attachment_message,
    create_token_usage_attachment_message,
    create_ultramemory_attachment_message,
    create_verify_plan_reminder_attachment_message,
    expand_attachment_messages,
    format_empty_file_warning,
    format_offset_exceeded_warning,
    is_hidden_meta_message,
    is_hook_notice_payload,
    parse_attachment_message,
    render_attachment_message,
)
from ripperdoc.utils.messaging.types import (  # noqa: F401 — legacy re-export surface
    AssistantMessage,
    AttachmentMessage,
    ConversationMessage,
    Message,
    MessageContent,
    MessageRole,
    ProgressMessage,
    UserMessage,
)
from ripperdoc.utils.messaging.types import (  # noqa: F401 — legacy re-export surface
    ATTACHMENT_EXPORT_HIDDEN_TYPES,
    ATTACHMENT_IGNORED_TYPES,
    ATTACHMENT_SUMMARY_HIDDEN_TYPES,
)
from ripperdoc.utils.messaging.types import (  # noqa: TID252 — legacy import surface
    AgentMentionAttachmentPayload,
    AlreadyReadFileAttachmentPayload,
    AsyncHookResponseAttachmentPayload,
    AttachmentPayload,
    AttachmentPayloadModel,
    AutocheckpointingAttachmentPayload,
    BackgroundTaskStatusAttachmentPayload,
    BudgetUsdAttachmentPayload,
    CommandPermissionsAttachmentPayload,
    CompactFileReferenceAttachmentPayload,
    CompactionReminderAttachmentPayload,
    CriticalSystemReminderAttachmentPayload,
    DateChangeAttachmentPayload,
    DiagnosticsAttachmentPayload,
    DirectoryAttachmentPayload,
    DynamicSkillAttachmentPayload,
    EditedImageFileAttachmentPayload,
    EditedTextFileAttachmentPayload,
    FileAttachmentContent,
    FileAttachmentPayload,
    FileImageAttachmentContent,
    FileNotebookAttachmentContent,
    FilePdfAttachmentContent,
    FileTextAttachmentContent,
    HookAdditionalContextAttachmentPayload,
    HookBlockingErrorAttachmentPayload,
    HookCancelledAttachmentPayload,
    HookErrorDuringExecutionAttachmentPayload,
    HookNonBlockingErrorAttachmentPayload,
    HookPermissionDecisionAttachmentPayload,
    HookStoppedContinuationAttachmentPayload,
    HookSuccessAttachmentPayload,
    HookSystemMessageAttachmentPayload,
    InvokedSkillsAttachmentPayload,
    McpResourceAttachmentPayload,
    NestedMemoryAttachmentPayload,
    OpenedFileInIdeAttachmentPayload,
    OutputStyleAttachmentPayload,
    PdfReferenceAttachmentPayload,
    PlanFileReferenceAttachmentPayload,
    PlanModeAttachmentPayload,
    QueuedCommandAttachmentPayload,
    RelevantMemoriesAttachmentPayload,
    SelectedLinesInIdeAttachmentPayload,
    SkillListingAttachmentPayload,
    StructuredOutputAttachmentPayload,
    TaskProgressAttachmentPayload,
    TaskReminderAttachmentPayload,
    TaskStatusAttachmentPayload,
    TodoAttachmentPayload,
    TodoReminderAttachmentPayload,
    TokenUsageAttachmentPayload,
    UltramemoryAttachmentPayload,
    UnknownAttachmentPayload,
    UnknownFileAttachmentContent,
    VerifyPlanReminderAttachmentPayload,
)
from ripperdoc.utils.messaging.mappers import _content_block_to_api, _content_block_to_openai

logger = get_logger()
FILE_ATTACHMENT_TRUNCATION_LINE_LIMIT = int(os.getenv("RIPPERDOC_MAX_READ_LINES", "2000"))


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
    sibling_tool_use_ids: Set[str],
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
