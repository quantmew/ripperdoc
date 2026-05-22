"""Attachment factories, parsing, rendering, and system reminder helpers."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Union, cast

from ripperdoc.utils.log import get_logger
from ripperdoc.utils.messaging.types import (
    AgentMentionAttachmentPayload,
    AssistantMessage,
    AsyncHookResponseAttachmentPayload,
    ATTACHMENT_IGNORED_TYPES,
    AttachmentMessage,
    AttachmentPayloadModel,
    BudgetUsdAttachmentPayload,
    CompactFileReferenceAttachmentPayload,
    CompactionReminderAttachmentPayload,
    CriticalSystemReminderAttachmentPayload,
    DateChangeAttachmentPayload,
    DiagnosticsAttachmentPayload,
    DirectoryAttachmentPayload,
    EditedTextFileAttachmentPayload,
    FileAttachmentPayload,
    HookAdditionalContextAttachmentPayload,
    HookBlockingErrorAttachmentPayload,
    HookStoppedContinuationAttachmentPayload,
    HookSuccessAttachmentPayload,
    InvokedSkillsAttachmentPayload,
    McpResourceAttachmentPayload,
    Message,
    MessageContent,
    MessageRole,
    NestedMemoryAttachmentPayload,
    OpenedFileInIdeAttachmentPayload,
    OutputStyleAttachmentPayload,
    PdfReferenceAttachmentPayload,
    PlanFileReferenceAttachmentPayload,
    PlanModeAttachmentPayload,
    ProgressMessage,
    QueuedCommandAttachmentPayload,
    RelevantMemoriesAttachmentPayload,
    SelectedLinesInIdeAttachmentPayload,
    SkillListingAttachmentPayload,
    TaskProgressAttachmentPayload,
    TaskReminderAttachmentPayload,
    TaskStatusAttachmentPayload,
    TodoAttachmentPayload,
    TodoReminderAttachmentPayload,
    TokenUsageAttachmentPayload,
    UltramemoryAttachmentPayload,
    UserMessage,
    VerifyPlanReminderAttachmentPayload,
)

logger = get_logger()
FILE_ATTACHMENT_TRUNCATION_LINE_LIMIT = int(os.getenv("RIPPERDOC_MAX_READ_LINES", "2000"))

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
    sibling_tool_use_ids: Optional[Set[str]] = None,
) -> ProgressMessage:
    """Create a progress message for hook notices that should be shown to the user."""
    from ripperdoc.utils.messaging.messages import create_progress_message

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
    file_size: Union[int, float],
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
    plan_exists: Optional[bool] = None,
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
            "PDF text extraction is not supported by the Read tool. Ask the user for extracted text "
            "or use an available document-processing workflow if one is provided."
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
        from ripperdoc.services.output_styles import resolve_output_style

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
    messages: Sequence[Union[UserMessage, AssistantMessage, ProgressMessage, AttachmentMessage]],
) -> List[Union[UserMessage, AssistantMessage, ProgressMessage, AttachmentMessage]]:
    """Expand attachment items into model-visible user/meta messages."""

    expanded: List[Union[UserMessage, AssistantMessage, ProgressMessage, AttachmentMessage]] = []
    for message in messages:
        if getattr(message, "type", None) == "attachment" and isinstance(message, AttachmentMessage):
            expanded.extend(parse_attachment_message(message))
            continue
        expanded.append(message)
    return expanded
