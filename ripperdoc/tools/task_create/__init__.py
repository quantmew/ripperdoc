"""TaskCreate tool — creates structured tasks in the task list."""

from __future__ import annotations

from typing import Any, AsyncGenerator, Dict, List, Optional

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


def _resolve_active_task_list_id() -> str:
    active_team_name = get_active_team_name()
    if active_team_name:
        team = get_team(active_team_name)
        if team is not None:
            return team.task_list_id
    return resolve_task_list_id()


def _is_team_task_context() -> bool:
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
        return "Create a new task in the task list"

    def needs_permissions(self, _input_data: Optional[TaskCreateInput] = None) -> bool:
        return False

    def is_concurrency_safe(self) -> bool:
        return True

    @property
    def input_schema(self) -> type[TaskCreateInput]:
        return TaskCreateInput

    def input_examples(self) -> List[ToolUseExample]:
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
