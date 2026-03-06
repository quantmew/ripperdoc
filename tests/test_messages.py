"""Test message utilities."""

from pydantic import BaseModel, Field

from ripperdoc.utils.messaging.messages import (
    AlreadyReadFileAttachmentPayload,
    AgentMentionAttachmentPayload,
    AttachmentMessage,
    AutocheckpointingAttachmentPayload,
    BackgroundTaskStatusAttachmentPayload,
    BudgetUsdAttachmentPayload,
    CommandPermissionsAttachmentPayload,
    CompactFileReferenceAttachmentPayload,
    CompactionReminderAttachmentPayload,
    DiagnosticsAttachmentPayload,
    DirectoryAttachmentPayload,
    DynamicSkillAttachmentPayload,
    EditedTextFileAttachmentPayload,
    EditedImageFileAttachmentPayload,
    FileImageAttachmentContent,
    FileTextAttachmentContent,
    InvokedSkillsAttachmentPayload,
    HookCancelledAttachmentPayload,
    HookBlockingErrorAttachmentPayload,
    HookErrorDuringExecutionAttachmentPayload,
    HookNonBlockingErrorAttachmentPayload,
    HookPermissionDecisionAttachmentPayload,
    HookStoppedContinuationAttachmentPayload,
    HookSuccessAttachmentPayload,
    HookSystemMessageAttachmentPayload,
    McpResourceAttachmentPayload,
    MessageRole,
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
    TaskStatusAttachmentPayload,
    TaskReminderAttachmentPayload,
    TodoAttachmentPayload,
    TodoReminderAttachmentPayload,
    TokenUsageAttachmentPayload,
    UltramemoryAttachmentPayload,
    VerifyPlanReminderAttachmentPayload,
    create_agent_mention_attachment_message,
    create_budget_usd_attachment_message,
    create_compact_file_reference_attachment_message,
    create_compaction_reminder_attachment_message,
    create_directory_attachment_message,
    create_diagnostics_attachment_message,
    create_edited_text_file_attachment_message,
    create_file_attachment_message,
    create_hook_additional_context_message,
    create_hook_blocking_error_attachment_message,
    create_hook_stopped_continuation_attachment_message,
    create_hook_success_attachment_message,
    create_mcp_resource_attachment_message,
    create_nested_memory_attachment_message,
    create_opened_file_in_ide_attachment_message,
    create_output_style_attachment_message,
    create_pdf_reference_attachment_message,
    create_plan_mode_attachment_message,
    create_plan_file_reference_attachment_message,
    create_queued_command_attachment_message,
    create_relevant_memories_attachment_message,
    create_selected_lines_in_ide_attachment_message,
    create_skill_listing_attachment_message,
    create_task_status_attachment_message,
    create_task_progress_attachment_message,
    create_task_reminder_attachment_message,
    create_token_usage_attachment_message,
    create_todo_attachment_message,
    create_todo_reminder_attachment_message,
    create_ultramemory_attachment_message,
    create_verify_plan_reminder_attachment_message,
    parse_attachment_message,
    render_attachment_message,
    create_user_message,
    create_assistant_message,
    create_progress_message,
    is_hidden_meta_message,
    normalize_messages_for_api,
)
from ripperdoc.core.providers.base import sanitize_tool_history


def test_create_user_message():
    """Test creating a user message."""
    msg = create_user_message("Hello, AI!")

    assert msg.type == "user"
    assert msg.message.role == MessageRole.USER
    assert msg.message.content == "Hello, AI!"
    assert msg.uuid != ""


def test_create_assistant_message():
    """Test creating an assistant message."""
    msg = create_assistant_message("Hello, user!", cost_usd=0.01, duration_ms=1000)

    assert msg.type == "assistant"
    assert msg.message.role == MessageRole.ASSISTANT
    assert msg.message.content == "Hello, user!"
    assert msg.cost_usd == 0.01
    assert msg.duration_ms == 1000


def test_create_progress_message():
    """Test creating a progress message."""
    msg = create_progress_message(
        tool_use_id="test_id", sibling_tool_use_ids={"id1", "id2"}, content="Working..."
    )

    assert msg.type == "progress"
    assert msg.tool_use_id == "test_id"
    assert msg.content == "Working..."
    assert msg.progress_sender is None


def test_create_progress_message_with_sender() -> None:
    msg = create_progress_message(
        tool_use_id="test_id",
        sibling_tool_use_ids=set(),
        content="Working...",
        progress_sender="Subagent(writer:agent_abcd1234)",
    )

    assert msg.progress_sender == "Subagent(writer:agent_abcd1234)"


def test_create_hook_additional_context_message() -> None:
    msg = create_hook_additional_context_message(
        "6666",
        hook_name="PreToolUse:Read",
        hook_event="PreToolUse",
        parent_tool_use_id="call_1",
    )
    assert msg is not None
    assert msg.type == "attachment"
    assert msg.parent_tool_use_id == "call_1"
    assert msg.attachment_type == "hook_additional_context"
    assert msg.attachment.type == "hook_additional_context"
    assert msg.attachment.hook_name == "PreToolUse:Read"
    assert msg.attachment.hook_event == "PreToolUse"
    assert msg.content == "6666"
    assert msg.metadata["hook_additional_context"] is True
    rendered = render_attachment_message(msg)
    assert rendered.message.role == MessageRole.USER
    assert (
        rendered.message.content
        == "<system-reminder>\nPreToolUse:Read hook additional context: 6666\n</system-reminder>"
    )


def test_create_hook_additional_context_message_joins_list_content() -> None:
    msg = create_hook_additional_context_message(
        ["6666", " ", "7777"],
        hook_name="PreToolUse:Read",
        hook_event="PreToolUse",
        parent_tool_use_id="call_1",
    )
    assert msg is not None
    assert msg.content == "6666\n7777"


def test_create_plan_mode_attachment_message() -> None:
    msg = create_plan_mode_attachment_message(
        "Plan mode still active.",
        plan_file_path="/tmp/plan.md",
        reminder_type="sparse",
    )

    assert msg.type == "attachment"
    assert msg.attachment_type == "plan_mode"
    assert msg.attachment.type == "plan_mode"
    assert msg.attachment.plan_file_path == "/tmp/plan.md"
    assert msg.attachment.reminder_type == "sparse"
    assert msg.content == "Plan mode still active."
    assert msg.metadata["plan_mode_attachment"] is True
    assert msg.metadata["plan_mode_attachment_type"] == "plan_mode"
    assert msg.metadata["plan_mode_reminder_type"] == "sparse"
    assert is_hidden_meta_message(msg) is True

    rendered = render_attachment_message(msg)
    assert rendered.message.role == MessageRole.USER
    assert rendered.message.content == "<system-reminder>\nPlan mode still active.\n</system-reminder>"


def test_parse_attachment_message_plan_mode_exit_uses_structured_payload() -> None:
    msg = create_plan_mode_attachment_message(
        "ignored",
        plan_file_path="/tmp/plan.md",
        reminder_type="exit",
        attachment_type="plan_mode_exit",
        plan_exists=True,
    )

    rendered = parse_attachment_message(msg)

    assert len(rendered) == 1
    assert rendered[0].message.content == (
        "<system-reminder>\n## Exited Plan Mode\n\n"
        "You have exited plan mode. You can now make edits, run tools, and take actions. "
        "The plan file is located at /tmp/plan.md if you need to reference it.\n"
        "</system-reminder>"
    )


def test_parse_attachment_message_directory_renders_tool_call_and_result() -> None:
    msg = AttachmentMessage(
        attachment={
            "type": "directory",
            "path": "/tmp/project",
            "content": "a.py\nb.py",
        }
    )

    rendered = parse_attachment_message(msg)

    assert len(rendered) == 2
    assert rendered[0].message.content == (
        '<system-reminder>\nCalled the LS tool with the following input: {"path": "/tmp/project"}\n</system-reminder>'
    )
    assert rendered[1].message.content == (
        '<system-reminder>\nResult of calling the LS tool: "a.py\\nb.py"\n</system-reminder>'
    )


def test_parse_attachment_message_file_text_truncated_adds_line_limit_note() -> None:
    msg = AttachmentMessage(
        attachment={
            "type": "file",
            "filename": "/tmp/a.py",
            "content": {"type": "text", "content": "print('x')"},
            "truncated": True,
        }
    )

    rendered = parse_attachment_message(msg)

    assert len(rendered) == 3
    assert isinstance(msg.attachment.content, FileTextAttachmentContent)
    assert rendered[2].message.content == (
        "<system-reminder>\n"
        "Note: The file /tmp/a.py was too large and has been truncated to the first 2000 lines. "
        "Don't tell the user about this truncation. Use the Read tool to read more of the file if you need.\n"
        "</system-reminder>"
    )


def test_parse_attachment_message_file_image_does_not_add_truncation_note() -> None:
    msg = AttachmentMessage(
        attachment={
            "type": "file",
            "filename": "/tmp/image.png",
            "content": {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "abc"}},
            "truncated": True,
        }
    )

    rendered = parse_attachment_message(msg)

    assert len(rendered) == 2
    assert isinstance(msg.attachment.content, FileImageAttachmentContent)


def test_parse_attachment_message_mcp_resource_renders_full_content_blocks() -> None:
    msg = AttachmentMessage(
        attachment={
            "type": "mcp_resource",
            "server": "calc",
            "uri": "mcp://calc/resource",
            "content": {
                "contents": [
                    {"text": "answer: 42"},
                ]
            },
        }
    )

    rendered = parse_attachment_message(msg)

    assert len(rendered) == 1
    assert isinstance(rendered[0].message.content, list)
    text_blocks = [block.text for block in rendered[0].message.content if getattr(block, "type", "") == "text"]
    assert text_blocks == [
        "<system-reminder>\nFull contents of resource:\n</system-reminder>",
        "<system-reminder>\nanswer: 42\n</system-reminder>",
        "<system-reminder>\nDo NOT read this resource again unless you think it may have changed, since you already have the full contents.\n</system-reminder>",
    ]


def test_parse_attachment_message_hook_success_only_for_supported_events() -> None:
    msg = AttachmentMessage(
        attachment={
            "type": "hook_success",
            "hookName": "PreToolUse:Read",
            "hookEvent": "PreToolUse",
            "content": "ok",
        }
    )

    assert parse_attachment_message(msg) == []


def test_parse_attachment_message_task_status_killed_is_prewrapped() -> None:
    msg = AttachmentMessage(
        attachment={
            "type": "task_status",
            "status": "killed",
            "description": "background lint",
            "taskId": "task_1",
        }
    )

    rendered = parse_attachment_message(msg)

    assert len(rendered) == 1
    assert rendered[0].message.content == (
        '<system-reminder>\nTask "background lint" (task_1) was stopped by the user.\n</system-reminder>'
    )


def test_parse_attachment_message_ignored_attachment_type_returns_empty() -> None:
    msg = AttachmentMessage(attachment={"type": "structured_output"})

    assert parse_attachment_message(msg) == []


def test_attachment_message_uses_typed_payload_models_for_ignored_attachment_types() -> None:
    typed_cases = [
        ({"type": "dynamic_skill"}, DynamicSkillAttachmentPayload),
        ({"type": "already_read_file", "filename": "a.py"}, AlreadyReadFileAttachmentPayload),
        ({"type": "command_permissions", "permissions": {"bash": "allow"}}, CommandPermissionsAttachmentPayload),
        ({"type": "edited_image_file", "filename": "image.png"}, EditedImageFileAttachmentPayload),
        ({"type": "hook_cancelled", "hookName": "h"}, HookCancelledAttachmentPayload),
        ({"type": "hook_error_during_execution", "error": "boom"}, HookErrorDuringExecutionAttachmentPayload),
        ({"type": "hook_non_blocking_error", "error": "warn"}, HookNonBlockingErrorAttachmentPayload),
        ({"type": "hook_system_message", "systemMessage": "msg"}, HookSystemMessageAttachmentPayload),
        ({"type": "structured_output", "content": {"ok": True}}, StructuredOutputAttachmentPayload),
        ({"type": "hook_permission_decision", "decision": {"allow": True}}, HookPermissionDecisionAttachmentPayload),
        ({"type": "autocheckpointing", "content": {"enabled": True}}, AutocheckpointingAttachmentPayload),
        ({"type": "background_task_status", "taskId": "task_1", "status": "running"}, BackgroundTaskStatusAttachmentPayload),
    ]

    for payload, expected_type in typed_cases:
        msg = AttachmentMessage(attachment=payload)
        assert isinstance(msg.attachment, expected_type)


def test_typed_attachment_factories_build_structured_payloads() -> None:
    directory = create_directory_attachment_message("/tmp/project", "a.py")
    file_attachment = create_file_attachment_message(
        "/tmp/a.py",
        {"type": "text", "content": "print('x')"},
        truncated=True,
    )
    edited_text_file = create_edited_text_file_attachment_message("a.py", "@@ -1 +1 @@")
    compact_file_ref = create_compact_file_reference_attachment_message("a.py")
    pdf_ref = create_pdf_reference_attachment_message("a.pdf", page_count=3, file_size=1024)
    selected_lines = create_selected_lines_in_ide_attachment_message(
        "a.py",
        line_start=1,
        line_end=2,
        content="print('x')",
    )
    opened_file = create_opened_file_in_ide_attachment_message("a.py")
    todo = create_todo_attachment_message(item_count=1, content=[{"content": "x", "status": "pending"}])
    plan_ref = create_plan_file_reference_attachment_message("/tmp/plan.md", "step1")
    nested_memory = create_nested_memory_attachment_message({"path": "a.md", "content": "memory"})
    relevant_memories = create_relevant_memories_attachment_message([{"path": "a.md", "content": "memory"}])
    skill_listing = create_skill_listing_attachment_message("skill-a")
    queued_command = create_queued_command_attachment_message("Run tests", command_mode="default")
    ultramemory = create_ultramemory_attachment_message("memory payload")
    mcp = create_mcp_resource_attachment_message(
        server="calc",
        uri="mcp://calc/resource",
        content={"contents": [{"text": "42"}]},
    )
    agent_mention = create_agent_mention_attachment_message("planner")
    output_style = create_output_style_attachment_message("learning")
    task_status = create_task_status_attachment_message(
        status="running",
        description="compile",
        task_id="task-1",
        task_type="background",
    )
    task_progress = create_task_progress_attachment_message("halfway")
    diagnostics = create_diagnostics_attachment_message([{"file": "a.py", "issues": []}])
    todo_reminder = create_todo_reminder_attachment_message([{"content": "x", "status": "pending"}])
    task_reminder = create_task_reminder_attachment_message([{"id": "1", "status": "pending", "subject": "x"}])
    token_usage = create_token_usage_attachment_message(used=1, total=10, remaining=9)
    budget = create_budget_usd_attachment_message(used=0.1, total=1.0, remaining=0.9)
    hook_success = create_hook_success_attachment_message(
        hook_name="SessionStart",
        hook_event="SessionStart",
        content="ok",
    )
    hook_blocking_error = create_hook_blocking_error_attachment_message(
        hook_name="PreToolUse:Bash",
        blocking_error={"command": "bash", "blockingError": "denied"},
    )
    hook_stopped = create_hook_stopped_continuation_attachment_message(
        hook_name="Stop",
        message="stop requested",
    )
    compaction_reminder = create_compaction_reminder_attachment_message()
    verify_plan_reminder = create_verify_plan_reminder_attachment_message()

    assert directory.attachment.type == "directory"
    assert file_attachment.attachment.type == "file"
    assert edited_text_file.attachment.type == "edited_text_file"
    assert compact_file_ref.attachment.type == "compact_file_reference"
    assert pdf_ref.attachment.type == "pdf_reference"
    assert selected_lines.attachment.type == "selected_lines_in_ide"
    assert opened_file.attachment.type == "opened_file_in_ide"
    assert todo.attachment.type == "todo"
    assert plan_ref.attachment.type == "plan_file_reference"
    assert nested_memory.attachment.type == "nested_memory"
    assert relevant_memories.attachment.type == "relevant_memories"
    assert skill_listing.attachment.type == "skill_listing"
    assert queued_command.attachment.type == "queued_command"
    assert ultramemory.attachment.type == "ultramemory"
    assert mcp.attachment.type == "mcp_resource"
    assert agent_mention.attachment.type == "agent_mention"
    assert output_style.attachment.type == "output_style"
    assert task_status.attachment.type == "task_status"
    assert task_progress.attachment.type == "task_progress"
    assert diagnostics.attachment.type == "diagnostics"
    assert todo_reminder.attachment.type == "todo_reminder"
    assert task_reminder.attachment.type == "task_reminder"
    assert token_usage.attachment.type == "token_usage"
    assert budget.attachment.type == "budget_usd"
    assert hook_success.attachment.type == "hook_success"
    assert hook_blocking_error.attachment.type == "hook_blocking_error"
    assert hook_stopped.attachment.type == "hook_stopped_continuation"
    assert compaction_reminder.attachment.type == "compaction_reminder"
    assert verify_plan_reminder.attachment.type == "verify_plan_reminder"
    assert isinstance(directory.attachment, DirectoryAttachmentPayload)
    assert isinstance(file_attachment.attachment.content, FileTextAttachmentContent)
    assert isinstance(edited_text_file.attachment, EditedTextFileAttachmentPayload)
    assert isinstance(compact_file_ref.attachment, CompactFileReferenceAttachmentPayload)
    assert isinstance(pdf_ref.attachment, PdfReferenceAttachmentPayload)
    assert isinstance(selected_lines.attachment, SelectedLinesInIdeAttachmentPayload)
    assert isinstance(opened_file.attachment, OpenedFileInIdeAttachmentPayload)
    assert isinstance(todo.attachment, TodoAttachmentPayload)
    assert isinstance(plan_ref.attachment, PlanFileReferenceAttachmentPayload)
    assert isinstance(nested_memory.attachment, NestedMemoryAttachmentPayload)
    assert isinstance(relevant_memories.attachment, RelevantMemoriesAttachmentPayload)
    assert isinstance(skill_listing.attachment, SkillListingAttachmentPayload)
    assert isinstance(queued_command.attachment, QueuedCommandAttachmentPayload)
    assert isinstance(ultramemory.attachment, UltramemoryAttachmentPayload)
    assert isinstance(mcp.attachment, McpResourceAttachmentPayload)
    assert isinstance(agent_mention.attachment, AgentMentionAttachmentPayload)
    assert isinstance(output_style.attachment, OutputStyleAttachmentPayload)
    assert isinstance(task_status.attachment, TaskStatusAttachmentPayload)
    assert isinstance(task_progress.attachment, TaskProgressAttachmentPayload)
    assert isinstance(diagnostics.attachment, DiagnosticsAttachmentPayload)
    assert isinstance(todo_reminder.attachment, TodoReminderAttachmentPayload)
    assert isinstance(task_reminder.attachment, TaskReminderAttachmentPayload)
    assert isinstance(token_usage.attachment, TokenUsageAttachmentPayload)
    assert isinstance(budget.attachment, BudgetUsdAttachmentPayload)
    assert isinstance(hook_success.attachment, HookSuccessAttachmentPayload)
    assert isinstance(hook_blocking_error.attachment, HookBlockingErrorAttachmentPayload)
    assert isinstance(hook_stopped.attachment, HookStoppedContinuationAttachmentPayload)
    assert isinstance(compaction_reminder.attachment, CompactionReminderAttachmentPayload)
    assert isinstance(verify_plan_reminder.attachment, VerifyPlanReminderAttachmentPayload)


def test_plan_mode_factory_uses_typed_payload_model() -> None:
    msg = create_plan_mode_attachment_message(
        "Plan mode still active.",
        plan_file_path="/tmp/plan.md",
        reminder_type="sparse",
    )

    assert isinstance(msg.attachment, PlanModeAttachmentPayload)


def test_attachment_message_coerces_more_known_types_to_specific_payload_models() -> None:
    payload_cases = [
        ("edited_text_file", {"filename": "a.py", "snippet": "@@\\n-old\\n+new"}, EditedTextFileAttachmentPayload),
        ("compact_file_reference", {"filename": "a.py"}, CompactFileReferenceAttachmentPayload),
        ("pdf_reference", {"filename": "a.pdf", "pageCount": 3, "fileSize": 1024}, PdfReferenceAttachmentPayload),
        (
            "selected_lines_in_ide",
            {"filename": "a.py", "lineStart": 1, "lineEnd": 2, "content": "print('x')"},
            SelectedLinesInIdeAttachmentPayload,
        ),
        ("opened_file_in_ide", {"filename": "a.py"}, OpenedFileInIdeAttachmentPayload),
        ("todo", {"itemCount": 1, "content": [{"content": "x", "status": "pending"}]}, TodoAttachmentPayload),
        ("invoked_skills", {"skills": [{"name": "skill", "path": "/tmp", "content": "rules"}]}, InvokedSkillsAttachmentPayload),
        ("todo_reminder", {"content": [{"content": "x", "status": "pending"}]}, TodoReminderAttachmentPayload),
        ("task_reminder", {"content": [{"id": "1", "status": "pending", "subject": "x"}]}, TaskReminderAttachmentPayload),
        ("nested_memory", {"content": {"path": "a.md", "content": "memory"}}, NestedMemoryAttachmentPayload),
        ("relevant_memories", {"memories": [{"path": "a.md", "content": "memory"}]}, RelevantMemoriesAttachmentPayload),
        ("skill_listing", {"content": "skill-a"}, SkillListingAttachmentPayload),
        ("queued_command", {"prompt": "Run tests"}, QueuedCommandAttachmentPayload),
        ("ultramemory", {"content": "remember this"}, UltramemoryAttachmentPayload),
        ("hook_blocking_error", {"hookName": "PreToolUse", "blockingError": {"command": "x", "blockingError": "boom"}}, HookBlockingErrorAttachmentPayload),
        ("hook_success", {"hookName": "SessionStart", "hookEvent": "SessionStart", "content": "ok"}, HookSuccessAttachmentPayload),
        ("hook_stopped_continuation", {"hookName": "Stop", "message": "paused"}, HookStoppedContinuationAttachmentPayload),
        ("compaction_reminder", {}, CompactionReminderAttachmentPayload),
        ("verify_plan_reminder", {}, VerifyPlanReminderAttachmentPayload),
    ]

    for attachment_type, payload, expected_cls in payload_cases:
        msg = AttachmentMessage(attachment={"type": attachment_type, **payload})
        assert isinstance(msg.attachment, expected_cls), attachment_type


def test_normalize_messages_for_api():
    """Test normalizing messages for API."""
    messages = [
        create_user_message("Hello"),
        create_assistant_message("Hi there"),
        create_progress_message("test_id", set(), "Progress"),
        create_user_message("How are you?"),
    ]

    normalized = normalize_messages_for_api(messages)

    # Progress messages should be filtered out
    assert len(normalized) == 3
    assert normalized[0]["role"] == "user"
    assert normalized[1]["role"] == "assistant"
    assert normalized[2]["role"] == "user"


def test_message_with_tool_result():
    """Test creating a message with tool result."""
    tool_result = {"type": "tool_result", "tool_use_id": "test_id", "content": "Result content"}

    msg = create_user_message([tool_result])

    assert msg.type == "user"
    assert isinstance(msg.message.content, list)
    assert len(msg.message.content) == 1


class _AliasStatusChange(BaseModel):
    from_status: str = Field(serialization_alias="from")
    to_status: str = Field(serialization_alias="to")


class _AliasToolResult(BaseModel):
    task_id: str = Field(serialization_alias="taskId")
    updated_fields: list[str] = Field(serialization_alias="updatedFields")
    status_change: _AliasStatusChange = Field(serialization_alias="statusChange")


def test_create_user_message_serializes_tool_result_with_aliases():
    payload = _AliasToolResult(
        task_id="1",
        updated_fields=["status"],
        status_change=_AliasStatusChange(from_status="pending", to_status="completed"),
    )

    msg = create_user_message("ok", tool_use_result=payload)

    assert isinstance(msg.tool_use_result, dict)
    assert msg.tool_use_result["taskId"] == "1"
    assert msg.tool_use_result["updatedFields"] == ["status"]
    assert msg.tool_use_result["statusChange"] == {"from": "pending", "to": "completed"}
    assert "task_id" not in msg.tool_use_result


def test_sanitize_tool_history_folds_multiple_tool_results_to_single_next_message():
    """Tool results for one assistant tool turn should be folded into one next user message."""
    normalized = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "call_1", "name": "foo", "input": {}},
                {"type": "tool_use", "id": "call_2", "name": "bar", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "call_1", "text": "ok"}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "call_2", "text": "ok"}],
        },
    ]

    sanitized = sanitize_tool_history(normalized)

    assert len(sanitized) == 2
    assert sanitized[0]["role"] == "assistant"
    assert sanitized[1]["role"] == "user"
    tool_use_ids = [
        block["id"] for block in sanitized[0]["content"] if block.get("type") == "tool_use"
    ]
    assert set(tool_use_ids) == {"call_1", "call_2"}
    tool_result_ids = [
        block["tool_use_id"]
        for block in sanitized[1]["content"]
        if block.get("type") == "tool_result"
    ]
    assert tool_result_ids == ["call_1", "call_2"]


def test_sanitize_tool_history_drops_unpaired_tool_use():
    """Unpaired tool_use blocks should still be removed."""
    normalized = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "call_1", "name": "foo", "input": {}},
                {"type": "tool_use", "id": "call_2", "name": "bar", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "call_1", "text": "ok"}],
        },
    ]

    sanitized = sanitize_tool_history(normalized)

    tool_use_ids = [
        block["id"] for block in sanitized[0]["content"] if block.get("type") == "tool_use"
    ]
    assert tool_use_ids == ["call_1"]


def test_sanitize_tool_history_keeps_non_tool_user_content_after_fold():
    """Non-tool user content should remain after folding paired tool results."""
    normalized = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "call_1", "name": "foo", "input": {}},
                {"type": "tool_use", "id": "call_2", "name": "bar", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "call_1", "text": "ok"}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "note"}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "call_2", "text": "ok"}],
        },
    ]

    sanitized = sanitize_tool_history(normalized)

    assert [msg.get("role") for msg in sanitized] == ["assistant", "user", "user"]
    folded_result_ids = [
        block["tool_use_id"]
        for block in sanitized[1]["content"]
        if block.get("type") == "tool_result"
    ]
    assert folded_result_ids == ["call_1", "call_2"]
    assert sanitized[2]["content"] == [{"type": "text", "text": "note"}]


def test_sanitize_tool_history_replays_real_session_parallel_git_tool_calls():
    """Regression: Anthropic requires all tool_results in the immediate next user message."""
    normalized = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "首先，让我并行运行git命令来分析当前状态："},
                {
                    "type": "tool_use",
                    "id": "call_00_vfVcN1GWWKIR4fnNEsbLMaWS",
                    "name": "Bash",
                    "input": {"command": "git status"},
                },
                {
                    "type": "tool_use",
                    "id": "call_01_MPKMZePGjQafIWuHMFXqbkoY",
                    "name": "Bash",
                    "input": {"command": "git diff --cached"},
                },
                {
                    "type": "tool_use",
                    "id": "call_02_eDQKzu84FV4zS4JtuDUutT2p",
                    "name": "Bash",
                    "input": {"command": "git diff"},
                },
                {
                    "type": "tool_use",
                    "id": "call_03_nBli3PtFi7ptFNSh12Iput45",
                    "name": "Bash",
                    "input": {"command": "git log --oneline -10"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_00_vfVcN1GWWKIR4fnNEsbLMaWS",
                    "text": "stdout: git status ...",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_01_MPKMZePGjQafIWuHMFXqbkoY",
                    "text": "stdout: git diff --cached ...",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_02_eDQKzu84FV4zS4JtuDUutT2p",
                    "text": "stdout: git diff ...",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_03_nBli3PtFi7ptFNSh12Iput45",
                    "text": "stdout: git log --oneline -10 ...",
                }
            ],
        },
    ]

    sanitized = sanitize_tool_history(normalized)

    assert len(sanitized) == 2
    assert sanitized[0]["role"] == "assistant"
    assert sanitized[1]["role"] == "user"

    folded_result_ids = [
        block["tool_use_id"]
        for block in sanitized[1]["content"]
        if block.get("type") == "tool_result"
    ]
    assert folded_result_ids == [
        "call_00_vfVcN1GWWKIR4fnNEsbLMaWS",
        "call_01_MPKMZePGjQafIWuHMFXqbkoY",
        "call_02_eDQKzu84FV4zS4JtuDUutT2p",
        "call_03_nBli3PtFi7ptFNSh12Iput45",
    ]


def test_normalize_messages_with_reasoning_metadata():
    """Ensure reasoning metadata is preserved for OpenAI-style messages."""
    assistant = create_assistant_message(
        [
            {
                "type": "tool_use",
                "tool_use_id": "call_reason",
                "name": "demo",
                "input": {},
            }
        ],
        metadata={"reasoning_content": "thinking..."},
    )
    user = create_user_message(
        [{"type": "tool_result", "tool_use_id": "call_reason", "text": "done"}]
    )

    normalized = normalize_messages_for_api([assistant, user], protocol="openai")
    asst_messages = [msg for msg in normalized if msg.get("role") == "assistant"]
    assert asst_messages, "assistant messages should be present"
    assert asst_messages[-1].get("reasoning_content") == "thinking..."


def test_normalize_messages_preserves_thinking_block():
    """Thinking/redacted thinking blocks should be passed through for Anthropic."""
    assistant = create_assistant_message(
        [
            {"type": "thinking", "thinking": "step 1", "signature": "sig"},
            {"type": "redacted_thinking", "data": "encrypted", "signature": "sig_redacted"},
            {"type": "text", "text": "answer"},
        ]
    )

    normalized = normalize_messages_for_api([assistant], protocol="anthropic")
    assert normalized and normalized[0].get("content")
    kinds = [blk.get("type") for blk in normalized[0]["content"]]
    assert kinds[:2] == ["thinking", "redacted_thinking"]


def test_normalize_messages_filters_unsigned_thinking_for_anthropic():
    """Anthropic replay should drop historical thinking blocks without signatures."""
    assistant = create_assistant_message(
        [
            {"type": "thinking", "thinking": "step 1", "signature": None},
            {"type": "redacted_thinking", "data": "encrypted", "signature": None},
            {"type": "text", "text": "answer"},
        ]
    )

    normalized = normalize_messages_for_api([assistant], protocol="anthropic")
    assert normalized and normalized[0].get("content")
    content = normalized[0]["content"]
    assert len(content) == 1
    assert content[0] == {"type": "text", "text": "answer"}


def test_normalize_messages_drops_empty_unsigned_thinking_message():
    """Anthropic replay should skip assistant turns that become empty after filtering."""
    assistant = create_assistant_message(
        [
            {"type": "thinking", "thinking": "step 1", "signature": None},
            {"type": "redacted_thinking", "data": "encrypted", "signature": None},
        ]
    )

    normalized = normalize_messages_for_api([assistant], protocol="anthropic")
    assert normalized == []


def test_normalize_messages_openai_skips_orphan_tool_results():
    """OpenAI protocol should skip tool_results without a preceding tool_use.

    This is critical for /compact functionality - after compaction, recent_tail
    may contain tool_result messages whose corresponding tool_use was in the
    compacted portion. OpenAI rejects these orphan tool_results with:
    "Messages with role 'tool' must be a response to a preceding message with 'tool_calls'"
    """
    # Simulate a post-compaction scenario: only a tool_result without the tool_use
    orphan_result = create_user_message(
        [{"type": "tool_result", "tool_use_id": "orphan_call", "text": "some result"}]
    )
    normalized = normalize_messages_for_api([orphan_result], protocol="openai")

    # The orphan tool_result should be filtered out, leaving no messages
    tool_messages = [msg for msg in normalized if msg.get("role") == "tool"]
    assert len(tool_messages) == 0, "Orphan tool_results should be skipped for OpenAI"


def test_normalize_messages_openai_keeps_paired_tool_results():
    """OpenAI protocol should keep tool_results that have a preceding tool_use."""
    assistant = create_assistant_message(
        [{"type": "tool_use", "id": "valid_call", "name": "demo", "input": {}}]
    )
    user = create_user_message(
        [{"type": "tool_result", "tool_use_id": "valid_call", "text": "result"}]
    )

    normalized = normalize_messages_for_api([assistant, user], protocol="openai")

    tool_messages = [msg for msg in normalized if msg.get("role") == "tool"]
    assert len(tool_messages) == 1, "Paired tool_results should be kept"
    assert tool_messages[0]["tool_call_id"] == "valid_call"


def test_normalize_messages_anthropic_preserves_tool_reference_blocks():
    assistant = create_assistant_message(
        [
            {
                "type": "server_tool_use",
                "id": "srvtoolu_01ABC123",
                "name": "tool_search_tool_regex",
                "input": {"query": "weather"},
            },
            {
                "type": "tool_search_tool_result",
                "tool_use_id": "srvtoolu_01ABC123",
                "content": {
                    "type": "tool_search_tool_search_result",
                    "tool_references": [
                        {"type": "tool_reference", "tool_name": "mcp__weather__get_weather"}
                    ],
                },
            },
        ]
    )

    normalized = normalize_messages_for_api([assistant], protocol="anthropic")
    assert len(normalized) == 1
    content = normalized[0]["content"]
    assert content[0]["type"] == "server_tool_use"
    assert content[0]["id"] == "srvtoolu_01ABC123"
    assert content[1]["type"] == "tool_search_tool_result"
    refs = content[1]["content"]["tool_references"]
    assert refs[0]["type"] == "tool_reference"
    assert refs[0]["tool_name"] == "mcp__weather__get_weather"


def test_normalize_messages_openai_skips_anthropic_tool_search_blocks():
    assistant = create_assistant_message(
        [
            {
                "type": "server_tool_use",
                "id": "srvtoolu_01ABC123",
                "name": "tool_search_tool_regex",
                "input": {"query": "weather"},
            },
            {
                "type": "tool_search_tool_result",
                "tool_use_id": "srvtoolu_01ABC123",
                "content": {"type": "tool_search_tool_search_result", "tool_references": []},
            },
        ]
    )

    normalized = normalize_messages_for_api([assistant], protocol="openai")
    # Anthropic-only blocks should not leak into OpenAI payload.
    assert normalized == []
