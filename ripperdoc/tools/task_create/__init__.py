"""TaskCreate tool — creates structured tasks in the task list."""

from __future__ import annotations

from textwrap import dedent
from typing import Any, AsyncGenerator, Dict, Optional

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
from ripperdoc.utils.collaboration.tasks import create_task, resolve_task_list_id
from ripperdoc.utils.collaboration.teams import find_team_by_task_list_id, get_active_team_name, get_team


logger = get_logger()

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


def _resolve_active_task_list_id() -> str:
    active_team_name = get_active_team_name()
    if active_team_name:
        team = get_team(active_team_name)
        if team is not None:
            return team.task_list_id
    return resolve_task_list_id()


def _is_team_task_context() -> bool:
    from ripperdoc.utils.collaboration.teams import find_team_by_task_list_id
    team = find_team_by_task_list_id(_resolve_active_task_list_id())
    return team is not None


class TaskCreateInput(BaseModel):
    """Input schema for TaskCreate."""

    subject: str = Field(description="Task title")
    description: str = Field(description="Detailed task description")
    active_form: Optional[str] = Field(
        default=None,
        validation_alias="activeForm",
        serialization_alias="activeForm",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class TaskCreateRef(BaseModel):
    id: str
    subject: str


class TaskCreateOutput(BaseModel):
    task: Optional[TaskCreateRef]


class TaskCreateTool(Tool[TaskCreateInput, TaskCreateOutput]):
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

    async def prompt(self, yolo_mode: bool = False) -> str:
        from ripperdoc.tools.task_create._prompt import TASK_CREATE_PROMPT
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
            logger.warning("[task_create] TaskCreate failed: %s: %s", type(exc).__name__, exc)
            output = TaskCreateOutput(task=None)
            yield ToolResult(data=output, result_for_assistant=f"TaskCreate failed: {exc}")
