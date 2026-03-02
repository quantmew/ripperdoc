"""Unified TaskStop tool."""

from __future__ import annotations

from typing import AsyncGenerator, Optional

from pydantic import BaseModel, ConfigDict, Field

from ripperdoc.core.tool import Tool, ToolResult, ToolUseContext, ValidationResult
from ripperdoc.tools.background_shell import get_background_status, kill_background_task
from ripperdoc.tools.task_tool import cancel_agent_run, get_agent_run_snapshot


class TaskStopInput(BaseModel):
    """Input schema for TaskStop."""

    task_id: str = Field(description="ID of the running task to stop")
    model_config = ConfigDict(extra="forbid")

    def resolved_task_id(self) -> Optional[str]:
        return (self.task_id or "").strip() or None


class TaskStopOutput(BaseModel):
    """Result of stopping a task."""

    message: str
    task_id: str
    task_type: str
    command: Optional[str] = None


class TaskStopTool(Tool[TaskStopInput, TaskStopOutput]):
    """Stop a running background task by ID."""

    @property
    def name(self) -> str:
        return "TaskStop"

    async def description(self) -> str:
        return "Stop a running background task by ID."

    async def prompt(self, yolo_mode: bool = False) -> str:
        del yolo_mode
        return (
            "- Stops a running background task by task_id\n"
            "- Supports background Bash and Task subagent runs"
        )

    @property
    def input_schema(self) -> type[TaskStopInput]:
        return TaskStopInput

    def is_read_only(self) -> bool:
        return False

    def is_concurrency_safe(self) -> bool:
        return True

    async def validate_input(
        self, input_data: TaskStopInput, context: Optional[ToolUseContext] = None
    ) -> ValidationResult:
        del context
        task_id = input_data.resolved_task_id()
        if not task_id:
            return ValidationResult(result=False, message="Missing required parameter: task_id")

        background = self._safe_background_status(task_id)
        if background is not None:
            if background["status"] != "running":
                return ValidationResult(
                    result=False,
                    message=f"Task {task_id} is not running (status: {background['status']})",
                )
            return ValidationResult(result=True)

        snapshot = get_agent_run_snapshot(task_id)
        if snapshot is None:
            return ValidationResult(result=False, message=f"No task found with ID: {task_id}")
        if str(snapshot.get("status") or "") != "running":
            return ValidationResult(
                result=False,
                message=f"Task {task_id} is not running (status: {snapshot.get('status')})",
            )
        return ValidationResult(result=True)

    @staticmethod
    def _safe_background_status(task_id: str) -> Optional[dict]:
        try:
            return get_background_status(task_id, consume=False)
        except KeyError:
            return None

    def render_result_for_assistant(self, output: TaskStopOutput) -> str:
        return output.message

    def render_tool_use_message(self, input_data: TaskStopInput, verbose: bool = False) -> str:
        del verbose
        task_id = input_data.resolved_task_id() or "<missing>"
        return f"task-stop {task_id}"

    async def call(
        self, input_data: TaskStopInput, context: ToolUseContext
    ) -> AsyncGenerator[ToolResult, None]:
        del context
        task_id = input_data.resolved_task_id()
        if not task_id:
            raise ValueError("Missing required parameter: task_id")

        background = self._safe_background_status(task_id)
        if background is not None:
            killed = False
            if background["status"] == "running":
                killed = await kill_background_task(task_id)
            output = TaskStopOutput(
                message=(
                    f"Successfully stopped task: {task_id} ({background.get('command')})"
                    if killed
                    else f"Failed to stop task: {task_id}"
                ),
                task_id=task_id,
                task_type="local_bash",
                command=str(background.get("command") or ""),
            )
            yield ToolResult(data=output, result_for_assistant=self.render_result_for_assistant(output))
            return

        cancelled = await cancel_agent_run(task_id)
        snapshot = get_agent_run_snapshot(task_id)
        agent_label = str((snapshot or {}).get("agent_type") or "subagent")
        output = TaskStopOutput(
            message=(
                f"Successfully stopped task: {task_id} ({agent_label})"
                if cancelled
                else f"Failed to stop task: {task_id}"
            ),
            task_id=task_id,
            task_type="local_agent",
            command=agent_label,
        )
        yield ToolResult(data=output, result_for_assistant=self.render_result_for_assistant(output))
