"""TaskList tool — lists all tasks in the task board."""

from __future__ import annotations

from textwrap import dedent
from typing import AsyncGenerator, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field

from ripperdoc.core.tool import (
    Tool,
    ToolOutput,
    ToolResult,
    ToolUseExample,
    ToolUseContext,
)
from ripperdoc.utils.log import get_logger
from ripperdoc.utils.collaboration.tasks import TaskItem, list_tasks, resolve_task_list_id
from ripperdoc.utils.collaboration.teams import find_team_by_task_list_id, get_active_team_name, get_team


logger = get_logger()

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


def _resolve_active_task_list_id() -> str:
    active_team_name = get_active_team_name()
    if active_team_name:
        team = get_team(active_team_name)
        if team is not None:
            return team.task_list_id
    return resolve_task_list_id()


def _task_id_sort_key(task_id: str) -> Tuple[int, Union[int, str]]:
    if str(task_id).isdigit():
        return (0, int(task_id))
    return (1, str(task_id))


def _is_team_task_context() -> bool:
    team = find_team_by_task_list_id(_resolve_active_task_list_id())
    return team is not None


class TaskListInput(BaseModel):
    """Input schema for TaskList."""

    model_config = ConfigDict(extra="forbid")


class TaskListEntry(BaseModel):
    id: str
    subject: str
    status: Literal["pending", "in_progress", "completed"]
    owner: Optional[str] = None
    blocked_by: List[str] = Field(default_factory=list, serialization_alias="blockedBy")
    model_config = ConfigDict(populate_by_name=True)


class TaskListOutput(BaseModel):
    tasks: List[TaskListEntry] = Field(default_factory=list)


class TaskListTool(Tool[TaskListInput, TaskListOutput]):
    @property
    def name(self) -> str:
        return "TaskList"

    async def description(self) -> str:
        return "List all tasks in the task list"

    @property
    def input_schema(self) -> type[TaskListInput]:
        return TaskListInput

    def input_examples(self) -> List[ToolUseExample]:
        return [
            ToolUseExample(
                description="List current task board summary",
                example={},
            )
        ]

    async def prompt(self, yolo_mode: bool = False) -> str:
        from ripperdoc.tools.task_list._prompt import TASK_LIST_PROMPT
        return TASK_LIST_PROMPT


    def is_read_only(self) -> bool:
        return True

    def is_concurrency_safe(self) -> bool:
        return True

    def render_result_for_assistant(self, output: TaskListOutput) -> str:
        if not output.tasks:
            return "No tasks found"
        lines: List[str] = []
        for task in output.tasks:
            owner = f" ({task.owner})" if task.owner else ""
            blocked = ""
            if task.blocked_by:
                blocked = f" [blocked by {', '.join(f'#{b}' for b in task.blocked_by)}]"
            lines.append(f"#{task.id} [{task.status}] {task.subject}{owner}{blocked}")
        return "\n".join(lines)

    def render_tool_use_message(self, _input_data: TaskListInput, _verbose: bool = False) -> str:
        return ""

    async def call(
        self,
        input_data: TaskListInput,
        _context: ToolUseContext,
    ) -> AsyncGenerator[ToolOutput, None]:
        del input_data
        task_list_id = _resolve_active_task_list_id()
        tasks = list_tasks(task_list_id=task_list_id)

        by_id = {task.id: task for task in tasks}
        visible: List[TaskItem] = []
        for task in tasks:
            metadata = task.metadata if isinstance(task.metadata, dict) else {}
            if metadata.get("_internal"):
                continue
            if task.status == "completed":
                continue
            visible.append(task)

        visible.sort(key=lambda item: _task_id_sort_key(item.id))

        entries: List[TaskListEntry] = []
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
