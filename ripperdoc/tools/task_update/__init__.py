"""TaskUpdate tool — updates task state, details, owner, and dependencies."""

from __future__ import annotations

from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, Set, Tuple, Union

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
from ripperdoc.utils.collaboration.tasks import (
    TaskPatch,
    delete_task,
    get_task,
    list_tasks,
    resolve_task_list_id,
    unresolved_blockers,
    update_task,
)
from ripperdoc.utils.collaboration.teams import find_team_by_task_list_id, get_active_team_name, get_team, send_team_message


logger = get_logger()

TaskStatusWithDelete = Literal["pending", "in_progress", "completed", "deleted"]


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


def _dedupe(values: List[str]) -> List[str]:
    seen: Set[str] = set()
    result: List[str] = []
    for value in values:
        token = str(value or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        result.append(token)
    return result


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
    add_blocks: List[str] = Field(
        default_factory=list,
        validation_alias="addBlocks",
        serialization_alias="addBlocks",
    )
    add_blocked_by: List[str] = Field(
        default_factory=list,
        validation_alias="addBlockedBy",
        serialization_alias="addBlockedBy",
    )
    owner: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class TaskUpdateStatusChange(BaseModel):
    from_status: str = Field(serialization_alias="from")
    to_status: str = Field(serialization_alias="to")


class TaskUpdateOutput(BaseModel):
    success: bool
    task_id: str = Field(serialization_alias="taskId")
    updated_fields: List[str] = Field(default_factory=list, serialization_alias="updatedFields")
    error: Optional[str] = None
    status_change: Optional[TaskUpdateStatusChange] = Field(
        default=None,
        serialization_alias="statusChange",
    )


class TaskUpdateTool(Tool[TaskUpdateInput, TaskUpdateOutput]):
    @property
    def name(self) -> str:
        return "TaskUpdate"

    async def description(self) -> str:
        return (
            "Update task state/details/owner/dependencies, or delete task with status=deleted. "
            "Supports addBlocks/addBlockedBy incremental dependency updates."
        )

    def needs_permissions(self, _input_data: Optional[TaskUpdateInput] = None) -> bool:
        return False

    @property
    def input_schema(self) -> type[TaskUpdateInput]:
        return TaskUpdateInput

    def input_examples(self) -> List[ToolUseExample]:
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

    async def prompt(self, yolo_mode: bool = False) -> str:
        from ripperdoc.tools.task_update._prompt import TASK_UPDATE_PROMPT
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

            updated_fields: List[str] = []
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
                            "[task_update] Failed task assignment message: %s: %s",
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
            logger.warning("[task_update] TaskUpdate failed: %s: %s", type(exc).__name__, exc)
            output = TaskUpdateOutput(
                success=False,
                task_id=input_data.task_id,
                updated_fields=[],
                error=str(exc),
            )
            yield ToolResult(data=output, result_for_assistant=self.render_result_for_assistant(output))
