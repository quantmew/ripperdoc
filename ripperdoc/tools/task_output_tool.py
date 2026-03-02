"""Unified TaskOutput tool."""

from __future__ import annotations

import asyncio
import time
from typing import AsyncGenerator, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from ripperdoc.core.tool import Tool, ToolResult, ToolUseContext, ValidationResult
from ripperdoc.tools.background_shell import get_background_status
from ripperdoc.tools.task_tool import get_agent_run_snapshot, wait_for_agent_run_snapshot


class TaskOutputInput(BaseModel):
    """Input schema for TaskOutput."""

    task_id: str = Field(description="Task ID from a background bash or Task run")
    block: bool = Field(default=True, description="Whether to wait for completion")
    timeout: int = Field(
        default=30_000,
        ge=0,
        le=600_000,
        description="Maximum wait time in milliseconds when block=true (does not stop the task)",
    )
    model_config = ConfigDict(extra="forbid")


class TaskOutputTask(BaseModel):
    """Normalized task details for output rendering."""

    task_id: str
    task_type: str
    status: str
    description: Optional[str] = None
    output: str = ""
    exit_code: Optional[int] = None
    prompt: Optional[str] = None
    output_file: Optional[str] = Field(default=None, serialization_alias="outputFile")
    error: Optional[str] = None
    isolation: Optional[str] = None
    worktree_path: Optional[str] = None
    worktree_branch: Optional[str] = None


class TaskOutputData(BaseModel):
    """TaskOutput response payload."""

    retrieval_status: Literal["success", "not_ready", "timeout"]
    task: Optional[TaskOutputTask] = None


class TaskOutputTool(Tool[TaskOutputInput, TaskOutputData]):
    """Retrieve output from a running or completed task."""

    @property
    def name(self) -> str:
        return "TaskOutput"

    async def description(self) -> str:
        return "Retrieves output from a running or completed task."

    async def prompt(self, yolo_mode: bool = False) -> str:
        del yolo_mode
        return (
            "- Retrieves output from a running or completed task (background shell or subagent)\n"
            "- Takes task_id, optional block (default true), and timeout (ms)\n"
            "- Returns retrieval_status with task details\n"
            "- timeout only limits waiting; it does not terminate the underlying task\n"
            "- Use block=false for non-blocking checks\n"
            "- Works with Bash background tasks and Task subagent runs"
        )

    @property
    def input_schema(self) -> type[TaskOutputInput]:
        return TaskOutputInput

    def is_read_only(self) -> bool:
        return True

    def is_concurrency_safe(self) -> bool:
        return True

    async def validate_input(
        self, input_data: TaskOutputInput, context: Optional[ToolUseContext] = None
    ) -> ValidationResult:
        del context
        if self._get_background_snapshot(input_data.task_id) is not None:
            return ValidationResult(result=True)
        if get_agent_run_snapshot(input_data.task_id) is not None:
            return ValidationResult(result=True)
        return ValidationResult(result=False, message=f"No task found with ID: {input_data.task_id}")

    def render_result_for_assistant(self, output: TaskOutputData) -> str:
        if not output.task:
            return f"retrieval_status={output.retrieval_status}"
        parts = [
            f"retrieval_status={output.retrieval_status}",
            f"task_type={output.task.task_type}",
            f"status={output.task.status}",
        ]
        if output.task.output:
            parts.append(f"output:\n{output.task.output}")
        if output.task.worktree_path:
            parts.append(f"worktree: {output.task.worktree_path}")
        if output.task.worktree_branch:
            parts.append(f"worktree_branch: {output.task.worktree_branch}")
        if output.task.error:
            parts.append(f"error: {output.task.error}")
        return "\n\n".join(parts)

    def render_tool_use_message(self, input_data: TaskOutputInput, verbose: bool = False) -> str:
        del verbose
        suffix = "" if input_data.block else " (non-blocking)"
        return f"task-output {input_data.task_id}{suffix}"

    @staticmethod
    def _get_background_snapshot(task_id: str) -> Optional[dict]:
        try:
            return get_background_status(task_id, consume=False)
        except KeyError:
            return None

    @staticmethod
    def _compose_stream_output(stdout: str, stderr: str) -> str:
        pieces = [part for part in (stdout, stderr) if part]
        return "\n".join(pieces).strip()

    async def _resolve_background_task(self, task_id: str, block: bool, timeout_ms: int) -> TaskOutputData:
        status = self._get_background_snapshot(task_id)
        if status is None:
            return TaskOutputData(retrieval_status="timeout", task=None)

        if block and status["status"] == "running":
            deadline = time.monotonic() + (max(timeout_ms, 0) / 1000.0)
            while time.monotonic() < deadline:
                await asyncio.sleep(0.1)
                status = self._get_background_snapshot(task_id)
                if status is None:
                    break
                if status["status"] != "running":
                    break

        if status is None:
            return TaskOutputData(retrieval_status="timeout", task=None)

        retrieval_status: Literal["success", "not_ready", "timeout"] = "success"
        if status["status"] == "running":
            retrieval_status = "timeout" if block else "not_ready"

        return TaskOutputData(
            retrieval_status=retrieval_status,
            task=TaskOutputTask(
                task_id=task_id,
                task_type="local_bash",
                status=status["status"],
                description=status.get("command"),
                output=self._compose_stream_output(status.get("stdout", ""), status.get("stderr", "")),
                exit_code=status.get("exit_code"),
                error=None,
            ),
        )

    async def _resolve_agent_task(self, task_id: str, block: bool, timeout_ms: int) -> TaskOutputData:
        snapshot = (
            await wait_for_agent_run_snapshot(task_id, timeout_ms=timeout_ms)
            if block
            else get_agent_run_snapshot(task_id)
        )
        if snapshot is None:
            return TaskOutputData(retrieval_status="timeout", task=None)

        status = str(snapshot.get("status") or "unknown")
        retrieval_status: Literal["success", "not_ready", "timeout"] = "success"
        if status == "running":
            retrieval_status = "timeout" if block else "not_ready"

        result_text = str(snapshot.get("result_text") or "").strip()
        error = str(snapshot.get("error") or "").strip() or None

        return TaskOutputData(
            retrieval_status=retrieval_status,
            task=TaskOutputTask(
                task_id=task_id,
                task_type="local_agent",
                status=status,
                description=str(snapshot.get("task_description") or f"Subagent {snapshot.get('agent_type') or 'unknown'}"),
                output=result_text or (error or ""),
                prompt=str(snapshot.get("task_prompt") or "") or None,
                output_file=str(snapshot.get("output_file") or "") or None,
                error=error,
                isolation=snapshot.get("isolation_mode"),
                worktree_path=snapshot.get("worktree_path"),
                worktree_branch=snapshot.get("worktree_branch"),
            ),
        )

    async def call(
        self, input_data: TaskOutputInput, context: ToolUseContext
    ) -> AsyncGenerator[ToolResult, None]:
        del context
        background_status = self._get_background_snapshot(input_data.task_id)
        if background_status is not None:
            output = await self._resolve_background_task(
                input_data.task_id, input_data.block, input_data.timeout
            )
            yield ToolResult(data=output, result_for_assistant=self.render_result_for_assistant(output))
            return

        output = await self._resolve_agent_task(input_data.task_id, input_data.block, input_data.timeout)
        yield ToolResult(data=output, result_for_assistant=self.render_result_for_assistant(output))
