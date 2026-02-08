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
    Use this tool to create a structured task list for your current coding session. This helps you track progress, organize complex tasks, and demonstrate thoroughness to the user.
    It also helps the user understand the progress of the task and overall progress of their requests.

    ## When to Use This Tool

    Use this tool proactively in these scenarios:

    - Complex multi-step tasks - When a task requires 3 or more distinct steps or actions
    - Non-trivial and complex tasks - Tasks that require careful planning or multiple operations
    - Plan mode - When using plan mode, create a task list to track the work
    - User explicitly requests todo list - When the user directly asks you to use the todo list
    - User provides multiple tasks - When users provide a list of things to be done (numbered or comma-separated)
    - After receiving new instructions - Immediately capture user requirements as tasks
    - When you start working on a task - Mark it as in_progress BEFORE beginning work
    - After completing a task - Mark it as completed and add any new follow-up tasks discovered during implementation

    ## When NOT to Use This Tool

    Skip using this tool when:
    - There is only a single, straightforward task
    - The task is trivial and tracking it provides no organizational benefit
    - The task can be completed in less than 3 trivial steps
    - The task is purely conversational or informational

    NOTE that you should not use this tool if there is only one trivial task to do. In this case you are better off just doing the task directly.

    ## Task Fields

    - **subject**: A brief, actionable title in imperative form (e.g., "Fix authentication bug in login flow")
    - **description**: Detailed description of what needs to be done, including context and acceptance criteria
    - **activeForm**: Present continuous form shown in spinner when task is in_progress (e.g., "Fixing authentication bug"). This is displayed to the user while you work on the task.

    **IMPORTANT**: Always provide activeForm when creating tasks. The subject should be imperative ("Run tests") while activeForm should be present continuous ("Running tests"). All tasks are created with status `pending`.

    ## Tips

    - Create tasks with clear, specific subjects that describe the outcome
    - Include enough detail in the description for another agent to understand and complete the task
    - After creating tasks, use TaskUpdate to set up dependencies (blocks/blockedBy) if needed
    - Check TaskList first to avoid creating duplicate tasks
    """
).strip()

TASK_CREATE_PROMPT_TEAM = dedent(
    """\
    Use this tool to create a structured task list for your current coding session. This helps you track progress, organize complex tasks, and demonstrate thoroughness to the user.
    It also helps the user understand the progress of the task and overall progress of their requests.

    ## When to Use This Tool

    Use this tool proactively in these scenarios:

    - Complex multi-step tasks - When a task requires 3 or more distinct steps or actions
    - Non-trivial and complex tasks - Tasks that require careful planning or multiple operations and potentially assigned to teammates
    - Plan mode - When using plan mode, create a task list to track the work
    - User explicitly requests todo list - When the user directly asks you to use the todo list
    - User provides multiple tasks - When users provide a list of things to be done (numbered or comma-separated)
    - After receiving new instructions - Immediately capture user requirements as tasks
    - When you start working on a task - Mark it as in_progress BEFORE beginning work
    - After completing a task - Mark it as completed and add any new follow-up tasks discovered during implementation

    ## When NOT to Use This Tool

    Skip using this tool when:
    - There is only a single, straightforward task
    - The task is trivial and tracking it provides no organizational benefit
    - The task can be completed in less than 3 trivial steps
    - The task is purely conversational or informational

    NOTE that you should not use this tool if there is only one trivial task to do. In this case you are better off just doing the task directly.

    ## Task Fields

    - **subject**: A brief, actionable title in imperative form (e.g., "Fix authentication bug in login flow")
    - **description**: Detailed description of what needs to be done, including context and acceptance criteria
    - **activeForm**: Present continuous form shown in spinner when task is in_progress (e.g., "Fixing authentication bug"). This is displayed to the user while you work on the task.

    **IMPORTANT**: Always provide activeForm when creating tasks. The subject should be imperative ("Run tests") while activeForm should be present continuous ("Running tests"). All tasks are created with status `pending`.

    ## Tips

    - Create tasks with clear, specific subjects that describe the outcome
    - Include enough detail in the description for another agent to understand and complete the task
    - After creating tasks, use TaskUpdate to set up dependencies (blocks/blockedBy) if needed
    - New tasks are created with no owner - use TaskUpdate with the `owner` parameter to assign them
    - Check TaskList first to avoid creating duplicate tasks
    """
).strip()

TASK_GET_PROMPT = dedent(
    """\
    Use this tool to retrieve a task by its ID from the task list.

    ## When to Use This Tool

    - When you need the full description and context before starting work on a task
    - To understand task dependencies (what it blocks, what blocks it)
    - After being assigned a task, to get complete requirements

    ## Output

    Returns full task details:
    - **subject**: Task title
    - **description**: Detailed requirements and context
    - **status**: 'pending', 'in_progress', or 'completed'
    - **blocks**: Tasks waiting on this one to complete
    - **blockedBy**: Tasks that must complete before this one can start

    ## Tips

    - After fetching a task, verify its blockedBy list is empty before beginning work.
    - Use TaskList to see all tasks in summary form.
    """
).strip()

TASK_UPDATE_PROMPT = dedent(
    """\
    Use this tool to update a task in the task list.

    ## When to Use This Tool

    **Mark tasks as resolved:**
    - When you have completed the work described in a task
    - When a task is no longer needed or has been superseded
    - IMPORTANT: Always mark your assigned tasks as resolved when you finish them
    - After resolving, call TaskList to find your next task

    - ONLY mark a task as completed when you have FULLY accomplished it
    - If you encounter errors, blockers, or cannot finish, keep the task as in_progress
    - When blocked, create a new task describing what needs to be resolved
    - Never mark a task as completed if:
      - Tests are failing
      - Implementation is partial
      - You encountered unresolved errors
      - You couldn't find necessary files or dependencies

    **Delete tasks:**
    - When a task is no longer relevant or was created in error
    - Setting status to `deleted` permanently removes the task

    **Update task details:**
    - When requirements change or become clearer
    - When establishing dependencies between tasks

    ## Fields You Can Update

    - **status**: The task status (see Status Workflow below)
    - **subject**: Change the task title (imperative form, e.g., "Run tests")
    - **description**: Change the task description
    - **activeForm**: Present continuous form shown in spinner when in_progress (e.g., "Running tests")
    - **owner**: Change the task owner (agent name)
    - **metadata**: Merge metadata keys into the task (set a key to null to delete it)
    - **addBlocks**: Mark tasks that cannot start until this one completes
    - **addBlockedBy**: Mark tasks that must complete before this one can start

    ## Status Workflow

    Status progresses: `pending` → `in_progress` → `completed`

    Use `deleted` to permanently remove a task.

    ## Staleness

    Make sure to read a task's latest state using `TaskGet` before updating it.
    """
).strip()

TASK_LIST_PROMPT = dedent(
    """\
    Use this tool to list all tasks in the task list.

    ## When to Use This Tool

    - To see what tasks are available to work on (status: 'pending', no owner, not blocked)
    - To check overall progress on the project
    - To find tasks that are blocked and need dependencies resolved
    - After completing a task, to check for newly unblocked work or claim the next available task
    - **Prefer working on tasks in ID order** (lowest ID first) when multiple tasks are available, as earlier tasks often set up context for later ones

    ## Output

    Returns a summary of each task:
    - **id**: Task identifier (use with TaskGet, TaskUpdate)
    - **subject**: Brief description of the task
    - **status**: 'pending', 'in_progress', or 'completed'
    - **owner**: Agent ID if assigned, empty if available
    - **blockedBy**: List of open task IDs that must be resolved first (tasks with blockedBy cannot be claimed until dependencies resolve)

    Use TaskGet with a specific task ID to view full details including description and comments.
    """
).strip()

TASK_LIST_PROMPT_TEAM = dedent(
    """\
    Use this tool to list all tasks in the task list.

    ## When to Use This Tool

    - To see what tasks are available to work on (status: 'pending', no owner, not blocked)
    - To check overall progress on the project
    - To find tasks that are blocked and need dependencies resolved
    - Before assigning tasks to teammates, to see what's available
    - After completing a task, to check for newly unblocked work or claim the next available task
    - **Prefer working on tasks in ID order** (lowest ID first) when multiple tasks are available, as earlier tasks often set up context for later ones

    ## Output

    Returns a summary of each task:
    - **id**: Task identifier (use with TaskGet or TaskUpdate)
    - **subject**: Brief description of the task
    - **status**: 'pending', 'in_progress', or 'completed'
    - **owner**: Agent ID if assigned, empty if available
    - **blockedBy**: List of open task IDs that must be resolved first (tasks with blockedBy cannot be claimed until dependencies resolve)

    Use TaskGet with a specific task ID to view full details including description and comments.

    ## Teammate Workflow

    When working as a teammate:
    1. After completing your current task, call TaskList to find available work
    2. Look for tasks with status 'pending', no owner, and empty blockedBy
    3. **Prefer tasks in ID order** (lowest ID first) when multiple tasks are available, as earlier tasks often set up context for later ones
    4. Use TaskUpdate to claim an available task by setting `owner` to your teammate name
    5. If blocked, focus on unblocking tasks or notify the team lead
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


def _is_team_task_context() -> bool:
    team = find_team_by_task_list_id(_resolve_active_task_list_id())
    return team is not None


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
        return TASK_CREATE_PROMPT_TEAM if _is_team_task_context() else TASK_CREATE_PROMPT

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
        return TASK_LIST_PROMPT_TEAM if _is_team_task_context() else TASK_LIST_PROMPT

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
