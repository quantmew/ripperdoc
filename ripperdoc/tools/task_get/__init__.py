"""TaskGet tool — retrieves a single task by ID."""

from __future__ import annotations

from textwrap import dedent
from typing import AsyncGenerator, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from ripperdoc.core.tool import (
    Tool,
    ToolOutput,
    ToolResult,
    ToolUseContext,
)
from ripperdoc.utils.log import get_logger
from ripperdoc.utils.collaboration.tasks import get_task, resolve_task_list_id
from ripperdoc.utils.collaboration.teams import get_active_team_name, get_team


logger = get_logger()

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


def _resolve_active_task_list_id() -> str:
    active_team_name = get_active_team_name()
    if active_team_name:
        team = get_team(active_team_name)
        if team is not None:
            return team.task_list_id
    return resolve_task_list_id()


class TaskGetInput(BaseModel):
    """Input schema for TaskGet."""

    task_id: str = Field(validation_alias="taskId", serialization_alias="taskId")
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class TaskGetEntry(BaseModel):
    id: str
    subject: str
    description: str
    status: Literal["pending", "in_progress", "completed"]
    blocks: List[str]
    blocked_by: List[str] = Field(serialization_alias="blockedBy")
    model_config = ConfigDict(populate_by_name=True)


class TaskGetOutput(BaseModel):
    task: Optional[TaskGetEntry]


class TaskGetTool(Tool[TaskGetInput, TaskGetOutput]):
    @property
    def name(self) -> str:
        return "TaskGet"

    async def description(self) -> str:
        return "Get a task by ID from the task list"

    @property
    def input_schema(self) -> type[TaskGetInput]:
        return TaskGetInput

    async def prompt(self, yolo_mode: bool = False) -> str:
        from ripperdoc.tools.task_get._prompt import TASK_GET_PROMPT
        return TASK_GET_PROMPT


    def is_read_only(self) -> bool:
        return True

    def is_concurrency_safe(self) -> bool:
        return True

    def render_result_for_assistant(self, output: TaskGetOutput) -> str:
        if output.task is None:
            return "Task not found"
        lines = [
            f"Task #{output.task.id}: {output.task.subject}",
            f"Status: {output.task.status}",
            f"Description: {output.task.description}",
        ]
        if output.task.blocked_by:
            lines.append(
                f"Blocked by: {', '.join(f'#{b}' for b in output.task.blocked_by)}"
            )
        if output.task.blocks:
            lines.append(
                f"Blocks: {', '.join(f'#{b}' for b in output.task.blocks)}"
            )
        return "\n".join(lines)

    def render_tool_use_message(self, input_data: TaskGetInput, _verbose: bool = False) -> str:
        return ""

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
