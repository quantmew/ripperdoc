"""Task graph tool suite (TaskCreate/TaskGet/TaskUpdate/TaskList)."""

from __future__ import annotations

from textwrap import dedent
from typing import Any, AsyncGenerator, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from ripperdoc.core.tool import (
    Tool,
    ToolOutput,
    ToolResult,
    ToolUseExample,
    ToolUseContext,
    ValidationResult,
)
from ripperdoc.utils.log import get_logger
from ripperdoc.utils.tasks import (
    TaskItem,
    TaskPatch,
    create_task,
    delete_task,
    get_task,
    list_tasks,
    resolve_task_list_id,
    unresolved_blockers,
    update_task,
)
from ripperdoc.utils.teams import find_team_by_task_list_id, get_active_team_name, get_team, send_team_message


logger = get_logger()

TaskStatusWithDelete = Literal["pending", "in_progress", "completed", "deleted"]

TASK_CREATE_PROMPT = dedent(
    """\
    Create a structured task in the shared task list.

    Use this when work is multi-step, non-trivial, or the user requests TODO/task tracking.

    Key rules:
    - `subject` should be imperative (for example: "Implement login endpoint")
    - `activeForm` should be present-progressive (for example: "Implementing login endpoint")
    - Newly created tasks start as `pending`
    - Add dependencies later with TaskUpdate (`addBlocks` / `addBlockedBy`)
    - Use TaskList first to avoid duplicate tasks
    - In team workflows, new tasks are unassigned by default; assign via TaskUpdate.owner
    """
).strip()

TASK_GET_PROMPT = dedent(
    """\
    Retrieve one task by ID.

    Use this before starting assigned work or when validating dependency context.
    Check `blockedBy` first; blocked tasks should not be started yet.
    """
).strip()

TASK_UPDATE_PROMPT = dedent(
    """\
    Update task status/details/owner/dependencies, or delete a task.

    Primary operations:
    - Status changes: `pending`, `in_progress`, `completed`
    - Deletion: `status: "deleted"`
    - Ownership assignment: `owner`
    - Dependency additions: `addBlocks`, `addBlockedBy`

    Completion discipline:
    - Mark `completed` only when work is fully finished and verified
    - Keep task `in_progress` when blocked or partially done

    Metadata behavior:
    - `metadata` merges into existing metadata
    - If a metadata value is `null`, that key is removed

    Recommended flow: TaskGet first, then TaskUpdate to avoid stale writes.
    """
).strip()

TASK_LIST_PROMPT = dedent(
    """\
    List all tasks as a summary board.

    Use this to:
    - Review progress and ownership
    - Find work that is ready to start
    - Inspect blockers and select next task

    Heuristic: prioritize lower task IDs first when possible.
    """
).strip()


class TaskCreateInput(BaseModel):
    """Input schema for TaskCreate."""

    subject: str = Field(description="Task title")
    description: str = Field(description="Detailed task description")
    active_form: Optional[str] = Field(
        default=None,
        validation_alias="activeForm",
        serialization_alias="activeForm",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class TaskGetInput(BaseModel):
    """Input schema for TaskGet."""

    task_id: str = Field(validation_alias="taskId", serialization_alias="taskId")
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class TaskUpdateInput(BaseModel):
    """Input schema for TaskUpdate."""

    task_id: str = Field(validation_alias="taskId", serialization_alias="taskId")
    subject: Optional[str] = None
    description: Optional[str] = None
    active_form: Optional[str] = Field(
        default=None,
        validation_alias="activeForm",
        serialization_alias="activeForm",
    )
    status: Optional[TaskStatusWithDelete] = None
    add_blocks: list[str] = Field(
        default_factory=list,
        validation_alias="addBlocks",
        serialization_alias="addBlocks",
    )
    add_blocked_by: list[str] = Field(
        default_factory=list,
        validation_alias="addBlockedBy",
        serialization_alias="addBlockedBy",
    )
    owner: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class TaskListInput(BaseModel):
    """Input schema for TaskList."""

    model_config = ConfigDict(extra="forbid")


class TaskCreateRef(BaseModel):
    id: str
    subject: str


class TaskCreateOutput(BaseModel):
    task: Optional[TaskCreateRef]


class TaskGetEntry(BaseModel):
    id: str
    subject: str
    description: str
    status: Literal["pending", "in_progress", "completed"]
    blocks: list[str]
    blocked_by: list[str] = Field(serialization_alias="blockedBy")
    model_config = ConfigDict(populate_by_name=True)


class TaskGetOutput(BaseModel):
    task: Optional[TaskGetEntry]


class TaskUpdateStatusChange(BaseModel):
    from_status: str = Field(serialization_alias="from")
    to_status: str = Field(serialization_alias="to")


class TaskUpdateOutput(BaseModel):
    success: bool
    task_id: str = Field(serialization_alias="taskId")
    updated_fields: list[str] = Field(default_factory=list, serialization_alias="updatedFields")
    error: Optional[str] = None
    status_change: Optional[TaskUpdateStatusChange] = Field(
        default=None,
        serialization_alias="statusChange",
    )


class TaskListEntry(BaseModel):
    id: str
    subject: str
    status: Literal["pending", "in_progress", "completed"]
    owner: Optional[str] = None
    blocked_by: list[str] = Field(default_factory=list, serialization_alias="blockedBy")
    model_config = ConfigDict(populate_by_name=True)


class TaskListOutput(BaseModel):
    tasks: list[TaskListEntry] = Field(default_factory=list)


class _BaseTaskGraphTool(Tool[BaseModel, BaseModel]):
    def needs_permissions(self, _input_data: Optional[BaseModel] = None) -> bool:
        return False


def _resolve_active_task_list_id() -> str:
    active_team_name = get_active_team_name()
    if active_team_name:
        team = get_team(active_team_name)
        if team is not None:
            return team.task_list_id
    return resolve_task_list_id()


def _task_id_sort_key(task_id: str) -> tuple[int, int | str]:
    if str(task_id).isdigit():
        return (0, int(task_id))
    return (1, str(task_id))


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        token = str(value or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        result.append(token)
    return result


class TaskCreateTool(_BaseTaskGraphTool):
    @property
    def name(self) -> str:
        return "TaskCreate"

    async def description(self) -> str:
        return (
            "Create a new structured task. Use for multi-step or complex work. "
            "Initial status is pending; assign owners/dependencies later via TaskUpdate."
        )

    @property
    def input_schema(self) -> type[TaskCreateInput]:
        return TaskCreateInput

    def input_examples(self) -> list[ToolUseExample]:
        return [
            ToolUseExample(
                description="Create a coding task with metadata",
                example={
                    "subject": "实现登录接口",
                    "description": "新增 /api/login，支持密码校验与错误码",
                    "activeForm": "实现登录接口中",
                    "metadata": {"module": "auth"},
                },
            ),
            ToolUseExample(
                description="Create a follow-up test task",
                example={
                    "subject": "补充登录测试",
                    "description": "覆盖成功登录、错误密码和锁定用户场景",
                    "activeForm": "补充登录测试中",
                },
            ),
        ]

    async def prompt(self, _yolo_mode: bool = False) -> str:
        return TASK_CREATE_PROMPT

    async def validate_input(
        self,
        input_data: TaskCreateInput,
        _context: Optional[ToolUseContext] = None,
    ) -> ValidationResult:
        if not input_data.subject.strip():
            return ValidationResult(result=False, message="subject is required")
        if not input_data.description.strip():
            return ValidationResult(result=False, message="description is required")
        return ValidationResult(result=True)

    def render_result_for_assistant(self, output: TaskCreateOutput) -> str:
        if output.task is None:
            return "TaskCreate failed."
        return f"Created task '{output.task.id}': {output.task.subject}"

    def render_tool_use_message(self, input_data: TaskCreateInput, _verbose: bool = False) -> str:
        return f"Creating task: {input_data.subject}"

    async def call(
        self,
        input_data: TaskCreateInput,
        _context: ToolUseContext,
    ) -> AsyncGenerator[ToolOutput, None]:
        task_list_id = _resolve_active_task_list_id()
        try:
            created = create_task(
                subject=input_data.subject,
                description=input_data.description,
                active_form=input_data.active_form,
                status="pending",
                metadata=input_data.metadata,
                task_list_id=task_list_id,
            )
            output = TaskCreateOutput(task=TaskCreateRef(id=created.id, subject=created.subject))
            yield ToolResult(data=output, result_for_assistant=self.render_result_for_assistant(output))
        except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
            logger.warning("[task_graph] TaskCreate failed: %s: %s", type(exc).__name__, exc)
            output = TaskCreateOutput(task=None)
            yield ToolResult(data=output, result_for_assistant=f"TaskCreate failed: {exc}")


class TaskGetTool(_BaseTaskGraphTool):
    @property
    def name(self) -> str:
        return "TaskGet"

    async def description(self) -> str:
        return "Get complete task details by task ID."

    @property
    def input_schema(self) -> type[TaskGetInput]:
        return TaskGetInput

    async def prompt(self, _yolo_mode: bool = False) -> str:
        return TASK_GET_PROMPT

    def is_read_only(self) -> bool:
        return True

    def is_concurrency_safe(self) -> bool:
        return True

    def render_result_for_assistant(self, output: TaskGetOutput) -> str:
        if output.task is None:
            return "Task not found."
        return (
            f"Task '{output.task.id}' ({output.task.status}): {output.task.subject} "
            f"blockedBy={output.task.blocked_by}"
        )

    def render_tool_use_message(self, input_data: TaskGetInput, _verbose: bool = False) -> str:
        return f"Reading task {input_data.task_id}"

    async def call(
        self,
        input_data: TaskGetInput,
        _context: ToolUseContext,
    ) -> AsyncGenerator[ToolOutput, None]:
        task_list_id = _resolve_active_task_list_id()
        task = get_task(input_data.task_id, task_list_id=task_list_id)
        if task is None:
            output = TaskGetOutput(task=None)
            yield ToolResult(data=output, result_for_assistant=self.render_result_for_assistant(output))
            return

        entry = TaskGetEntry(
            id=task.id,
            subject=task.subject,
            description=task.description,
            status=task.status,
            blocks=list(task.blocks),
            blocked_by=list(task.blocked_by),
        )
        output = TaskGetOutput(task=entry)
        yield ToolResult(data=output, result_for_assistant=self.render_result_for_assistant(output))


class TaskUpdateTool(_BaseTaskGraphTool):
    @property
    def name(self) -> str:
        return "TaskUpdate"

    async def description(self) -> str:
        return (
            "Update task state/details/owner/dependencies, or delete task with status=deleted. "
            "Supports addBlocks/addBlockedBy incremental dependency updates."
        )

    @property
    def input_schema(self) -> type[TaskUpdateInput]:
        return TaskUpdateInput

    def input_examples(self) -> list[ToolUseExample]:
        return [
            ToolUseExample(
                description="Start work on a task",
                example={"taskId": "1", "status": "in_progress"},
            ),
            ToolUseExample(
                description="Complete and assign task ownership",
                example={"taskId": "1", "status": "completed", "owner": "team-lead"},
            ),
            ToolUseExample(
                description="Add blockers to a task",
                example={"taskId": "2", "addBlockedBy": ["1"]},
            ),
        ]

    async def prompt(self, _yolo_mode: bool = False) -> str:
        return TASK_UPDATE_PROMPT

    async def validate_input(
        self,
        input_data: TaskUpdateInput,
        _context: Optional[ToolUseContext] = None,
    ) -> ValidationResult:
        if not input_data.task_id.strip():
            return ValidationResult(result=False, message="taskId is required")

        mutable_fields = [
            "subject",
            "description",
            "active_form",
            "status",
            "add_blocks",
            "add_blocked_by",
            "owner",
            "metadata",
        ]
        if not any(field in input_data.model_fields_set for field in mutable_fields):
            return ValidationResult(result=False, message="No update fields were provided")
        return ValidationResult(result=True)

    def render_result_for_assistant(self, output: TaskUpdateOutput) -> str:
        if not output.success:
            return f"TaskUpdate failed for '{output.task_id}': {output.error or 'unknown error'}"
        suffix = ""
        if output.status_change is not None:
            suffix = f" ({output.status_change.from_status} -> {output.status_change.to_status})"
        updated = ", ".join(output.updated_fields) if output.updated_fields else "none"
        return f"Updated task '{output.task_id}' fields: {updated}{suffix}"

    def render_tool_use_message(self, input_data: TaskUpdateInput, _verbose: bool = False) -> str:
        return f"Updating task {input_data.task_id}"

    async def call(
        self,
        input_data: TaskUpdateInput,
        context: ToolUseContext,
    ) -> AsyncGenerator[ToolOutput, None]:
        task_list_id = _resolve_active_task_list_id()
        previous_task = get_task(input_data.task_id, task_list_id=task_list_id)
        if previous_task is None:
            output = TaskUpdateOutput(
                success=False,
                task_id=input_data.task_id,
                updated_fields=[],
                error=f"Task '{input_data.task_id}' not found.",
            )
            yield ToolResult(data=output, result_for_assistant=self.render_result_for_assistant(output))
            return

        if input_data.status == "deleted":
            removed = delete_task(input_data.task_id, task_list_id=task_list_id)
            output = TaskUpdateOutput(
                success=removed,
                task_id=input_data.task_id,
                updated_fields=["status"] if removed else [],
                error=None if removed else f"Task '{input_data.task_id}' not found.",
                status_change=TaskUpdateStatusChange(
                    from_status=previous_task.status,
                    to_status="deleted",
                )
                if removed
                else None,
            )
            yield ToolResult(data=output, result_for_assistant=self.render_result_for_assistant(output))
            return

        merged_blocks = list(previous_task.blocks)
        if "add_blocks" in input_data.model_fields_set:
            merged_blocks = _dedupe([*merged_blocks, *input_data.add_blocks])

        merged_blocked_by = list(previous_task.blocked_by)
        if "add_blocked_by" in input_data.model_fields_set:
            merged_blocked_by = _dedupe([*merged_blocked_by, *input_data.add_blocked_by])

        if input_data.status == "completed":
            simulated = previous_task.model_copy(update={"blocked_by": merged_blocked_by})
            blockers = unresolved_blockers(simulated, list_tasks(task_list_id=task_list_id))
            if blockers:
                output = TaskUpdateOutput(
                    success=False,
                    task_id=input_data.task_id,
                    updated_fields=[],
                    error=(
                        "Cannot mark completed; unresolved blockers: "
                        + ", ".join(sorted(blockers, key=_task_id_sort_key))
                    ),
                )
                yield ToolResult(data=output, result_for_assistant=self.render_result_for_assistant(output))
                return

        owner_to_set = input_data.owner
        in_team_mode = find_team_by_task_list_id(task_list_id) is not None
        if (
            input_data.status == "in_progress"
            and "owner" not in input_data.model_fields_set
            and not previous_task.owner
            and in_team_mode
        ):
            owner_to_set = (context.agent_id or "team-lead").strip() or "team-lead"

        patch = TaskPatch(
            subject=input_data.subject,
            description=input_data.description,
            active_form=input_data.active_form,
            owner=owner_to_set if ("owner" in input_data.model_fields_set or owner_to_set) else None,
            status=input_data.status if input_data.status in {"pending", "in_progress", "completed"} else None,
            blocks=merged_blocks if "add_blocks" in input_data.model_fields_set else None,
            blocked_by=merged_blocked_by if "add_blocked_by" in input_data.model_fields_set else None,
            metadata=input_data.metadata,
        )

        try:
            updated = update_task(input_data.task_id, patch, task_list_id=task_list_id)

            updated_fields: list[str] = []
            if updated.subject != previous_task.subject:
                updated_fields.append("subject")
            if updated.description != previous_task.description:
                updated_fields.append("description")
            if updated.active_form != previous_task.active_form:
                updated_fields.append("activeForm")
            if updated.owner != previous_task.owner:
                updated_fields.append("owner")
            if updated.status != previous_task.status:
                updated_fields.append("status")
            if updated.blocks != previous_task.blocks:
                updated_fields.append("blocks")
            if updated.blocked_by != previous_task.blocked_by:
                updated_fields.append("blockedBy")
            if updated.metadata != previous_task.metadata:
                updated_fields.append("metadata")

            status_change = (
                TaskUpdateStatusChange(from_status=previous_task.status, to_status=updated.status)
                if updated.status != previous_task.status
                else None
            )

            if previous_task.owner != updated.owner and updated.owner:
                team = find_team_by_task_list_id(task_list_id)
                if team is not None:
                    try:
                        send_team_message(
                            team_name=team.name,
                            sender="system",
                            recipients=[updated.owner],
                            message_type="task_assignment",
                            content=(
                                f"Task '{updated.id}' assigned to '{updated.owner}'. "
                                f"Subject: {updated.subject}"
                            ),
                            metadata={"task_id": updated.id, "owner": updated.owner},
                        )
                    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
                        logger.warning(
                            "[task_graph] Failed task assignment message: %s: %s",
                            type(exc).__name__,
                            exc,
                            extra={"task_id": updated.id, "team": team.name},
                        )

            output = TaskUpdateOutput(
                success=True,
                task_id=input_data.task_id,
                updated_fields=updated_fields,
                status_change=status_change,
            )
            yield ToolResult(data=output, result_for_assistant=self.render_result_for_assistant(output))
        except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
            logger.warning("[task_graph] TaskUpdate failed: %s: %s", type(exc).__name__, exc)
            output = TaskUpdateOutput(
                success=False,
                task_id=input_data.task_id,
                updated_fields=[],
                error=str(exc),
            )
            yield ToolResult(data=output, result_for_assistant=self.render_result_for_assistant(output))


class TaskListTool(_BaseTaskGraphTool):
    @property
    def name(self) -> str:
        return "TaskList"

    async def description(self) -> str:
        return (
            "List task board summary with id/subject/status/owner/blockedBy. "
            "Filters internal tasks and strips completed blockers from blockedBy."
        )

    @property
    def input_schema(self) -> type[TaskListInput]:
        return TaskListInput

    def input_examples(self) -> list[ToolUseExample]:
        return [
            ToolUseExample(
                description="List current task board summary",
                example={},
            )
        ]

    async def prompt(self, _yolo_mode: bool = False) -> str:
        return TASK_LIST_PROMPT

    def is_read_only(self) -> bool:
        return True

    def is_concurrency_safe(self) -> bool:
        return True

    def render_result_for_assistant(self, output: TaskListOutput) -> str:
        if not output.tasks:
            return "No tasks stored yet."
        lines = ["Tasks:"]
        for task in output.tasks:
            owner = f" @{task.owner}" if task.owner else ""
            lines.append(
                f"- [{task.id}] {task.subject}{owner} ({task.status}) blockedBy={task.blocked_by}"
            )
        return "\n".join(lines)

    def render_tool_use_message(self, _input_data: TaskListInput, _verbose: bool = False) -> str:
        return "Listing tasks"

    async def call(
        self,
        input_data: TaskListInput,
        _context: ToolUseContext,
    ) -> AsyncGenerator[ToolOutput, None]:
        del input_data
        task_list_id = _resolve_active_task_list_id()
        tasks = list_tasks(task_list_id=task_list_id)

        by_id = {task.id: task for task in tasks}
        visible: list[TaskItem] = []
        for task in tasks:
            metadata = task.metadata if isinstance(task.metadata, dict) else {}
            if metadata.get("_internal"):
                continue
            visible.append(task)

        visible.sort(key=lambda item: _task_id_sort_key(item.id))

        entries: list[TaskListEntry] = []
        for task in visible:
            blockers = [
                dep
                for dep in task.blocked_by
                if (by_id.get(dep) is not None and by_id[dep].status != "completed")
            ]
            entries.append(
                TaskListEntry(
                    id=task.id,
                    subject=task.subject,
                    status=task.status,
                    owner=task.owner,
                    blocked_by=blockers,
                )
            )

        output = TaskListOutput(tasks=entries)
        yield ToolResult(data=output, result_for_assistant=self.render_result_for_assistant(output))


__all__ = [
    "TaskCreateTool",
    "TaskGetTool",
    "TaskUpdateTool",
    "TaskListTool",
    "TaskCreateInput",
    "TaskGetInput",
    "TaskUpdateInput",
    "TaskListInput",
]
