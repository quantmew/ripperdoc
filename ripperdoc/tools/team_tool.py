"""Team coordination tools for multi-agent collaboration domains."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
from textwrap import dedent
from typing import Any, AsyncGenerator, Literal, Optional
from uuid import uuid4

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
from ripperdoc.utils.tasks import task_list_dir
from ripperdoc.utils.teams import (
    TeamMessage,
    clear_active_team_name,
    create_team,
    delete_team,
    get_active_team_name,
    get_team,
    list_teams,
    send_team_message,
    set_active_team_name,
    team_config_path,
)


logger = get_logger()


TEAM_CREATE_PROMPT = dedent(
    """\
    # TeamCreate

    Create a collaboration team domain. Team and TaskList are 1:1.

    Use when user explicitly requests multi-agent/team/swarm collaboration,
    or work clearly benefits from parallel decomposition.

    Recommended flow:
    1. TeamCreate
    2. TaskCreate / TaskList to plan work
    3. Launch teammates with Task (team_name + teammate)
    4. Assign tasks with TaskUpdate.owner
    5. Coordinate using SendMessage (not plain chat)
    6. Ask teammates to shutdown, then TeamDelete
    """
).strip()

TEAM_DELETE_PROMPT = dedent(
    """\
    # TeamDelete

    Clean up current team resources when collaboration is complete.

    Precondition:
    - If active members still exist, TeamDelete should fail.
    - Request graceful shutdown first, then retry deletion.

    Cleanup:
    - Team config and message log
    - Shared team task list directory
    """
).strip()

SEND_MESSAGE_PROMPT = dedent(
    """\
    # SendMessage

    Structured intra-team communication protocol.

    Supported message types:
    - `message`: direct point-to-point message
    - `broadcast`: send to all teammates (use sparingly)
    - `shutdown_request`: request one teammate to shutdown
    - `shutdown_response`: approve/reject a shutdown request
    - `plan_approval_response`: approve/reject a plan request

    Rules:
    - Team communication must use this tool; plain text is not delivered to teammates
    - Prefer direct message over broadcast when possible
    - For `message`/`broadcast`, include a concise `summary` of 5-10 words
    - Keep content concise and actionable
    """
).strip()


_ACTIVE_TEAM_BY_AGENT: dict[str, str] = {}


def _context_key(context: ToolUseContext) -> str:
    return (context.agent_id or "__default__").strip() or "__default__"


def _remember_active_team(context: ToolUseContext, team_name: str) -> None:
    _ACTIVE_TEAM_BY_AGENT[_context_key(context)] = team_name


def _resolve_active_team_name(
    context: ToolUseContext, *, allow_single_team_fallback: bool = True
) -> Optional[str]:
    key = _context_key(context)
    if key in _ACTIVE_TEAM_BY_AGENT:
        return _ACTIVE_TEAM_BY_AGENT[key]

    env_team = os.getenv("RIPPERDOC_TEAM_NAME")
    if env_team and env_team.strip():
        return env_team.strip()

    disk_team = get_active_team_name()
    if disk_team:
        return disk_team

    if allow_single_team_fallback:
        teams = list_teams()
        if len(teams) == 1:
            return teams[0].name
    return None


def _sender_name(context: ToolUseContext) -> str:
    return (context.agent_id or "team-lead").strip() or "team-lead"


def _color_for(name: str) -> str:
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:6]
    return f"#{digest}"


_SUMMARY_WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[-_'][A-Za-z0-9]+)?|[\u4e00-\u9fff]+")


def _summary_word_count(text: str) -> int:
    return len(_SUMMARY_WORD_PATTERN.findall(text or ""))


class TeamCreateInput(BaseModel):
    team_name: str
    description: Optional[str] = None
    agent_type: Optional[str] = None
    model_config = ConfigDict(extra="forbid")


class TeamCreateOutput(BaseModel):
    team_name: str
    team_file_path: str
    lead_agent_id: str


class TeamDeleteInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TeamDeleteOutput(BaseModel):
    success: bool
    message: str
    team_name: Optional[str] = None


class SendMessageInput(BaseModel):
    type: Literal[
        "message",
        "broadcast",
        "shutdown_request",
        "shutdown_response",
        "plan_approval_response",
    ]
    recipient: Optional[str] = None
    content: Optional[str] = None
    summary: Optional[str] = None
    request_id: Optional[str] = None
    approve: Optional[bool] = None
    model_config = ConfigDict(extra="forbid")


class SendMessageRouting(BaseModel):
    sender: str
    sender_color: str = Field(serialization_alias="senderColor")
    target: str
    target_color: Optional[str] = Field(default=None, serialization_alias="targetColor")
    summary: Optional[str] = None
    content: Optional[str] = None


class SendMessageOutput(BaseModel):
    success: bool
    message: str
    recipients: Optional[list[str]] = None
    routing: Optional[SendMessageRouting] = None
    request_id: Optional[str] = None
    target: Optional[str] = None


class TeamCreateTool(Tool[TeamCreateInput, TeamCreateOutput]):
    @property
    def name(self) -> str:
        return "TeamCreate"

    async def description(self) -> str:
        return (
            "Create a team collaboration domain (Team = TaskList) for multi-agent work. "
            "Input: team_name, optional description and agent_type."
        )

    @property
    def input_schema(self) -> type[TeamCreateInput]:
        return TeamCreateInput

    def input_examples(self) -> list[ToolUseExample]:
        return [
            ToolUseExample(
                description="Create a team for auth refactor",
                example={
                    "team_name": "auth-refactor",
                    "description": "认证模块重构",
                    "agent_type": "general-purpose",
                },
            )
        ]

    async def prompt(self, _yolo_mode: bool = False) -> str:
        return TEAM_CREATE_PROMPT

    def needs_permissions(self, _input_data: Optional[TeamCreateInput] = None) -> bool:
        return False

    async def validate_input(
        self,
        input_data: TeamCreateInput,
        _context: Optional[ToolUseContext] = None,
    ) -> ValidationResult:
        if not input_data.team_name.strip():
            return ValidationResult(result=False, message="team_name is required")
        return ValidationResult(result=True)

    def render_result_for_assistant(self, output: TeamCreateOutput) -> str:
        return (
            f"Created team '{output.team_name}' (lead={output.lead_agent_id}) at "
            f"{output.team_file_path}"
        )

    def render_tool_use_message(self, input_data: TeamCreateInput, _verbose: bool = False) -> str:
        return f"Creating team {input_data.team_name}"

    async def call(
        self,
        input_data: TeamCreateInput,
        context: ToolUseContext,
    ) -> AsyncGenerator[ToolOutput, None]:
        existing_active_team = _resolve_active_team_name(context, allow_single_team_fallback=False)
        if existing_active_team and get_team(existing_active_team) is not None:
            raise ValueError(
                "Current session already has an active team context "
                f"('{existing_active_team}'). Delete it before creating another team."
            )

        metadata: dict[str, Any] = {}
        if input_data.description is not None:
            metadata["description"] = input_data.description
        if input_data.agent_type is not None:
            metadata["agent_type"] = input_data.agent_type

        try:
            team = create_team(name=input_data.team_name, metadata=metadata)
            set_active_team_name(team.name)
            _remember_active_team(context, team.name)
            output = TeamCreateOutput(
                team_name=team.name,
                team_file_path=str(team_config_path(team.name)),
                lead_agent_id=f"team-lead@{team.name}",
            )
            yield ToolResult(data=output, result_for_assistant=self.render_result_for_assistant(output))
        except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
            logger.warning("[team_tool] TeamCreate failed: %s: %s", type(exc).__name__, exc)
            raise ValueError(f"TeamCreate failed: {exc}") from exc


class TeamDeleteTool(Tool[TeamDeleteInput, TeamDeleteOutput]):
    @property
    def name(self) -> str:
        return "TeamDelete"

    async def description(self) -> str:
        return (
            "Delete current team context and clean up team/task resources. "
            "Fails when active members still exist."
        )

    @property
    def input_schema(self) -> type[TeamDeleteInput]:
        return TeamDeleteInput

    async def prompt(self, _yolo_mode: bool = False) -> str:
        return TEAM_DELETE_PROMPT

    def needs_permissions(self, _input_data: Optional[TeamDeleteInput] = None) -> bool:
        return False

    def render_result_for_assistant(self, output: TeamDeleteOutput) -> str:
        return output.message

    def render_tool_use_message(self, _input_data: TeamDeleteInput, _verbose: bool = False) -> str:
        return "Deleting current team"

    async def call(
        self,
        input_data: TeamDeleteInput,
        context: ToolUseContext,
    ) -> AsyncGenerator[ToolOutput, None]:
        del input_data
        team_name = _resolve_active_team_name(context)
        if not team_name:
            output = TeamDeleteOutput(
                success=False,
                message="No active team context found.",
                team_name=None,
            )
            yield ToolResult(data=output, result_for_assistant=self.render_result_for_assistant(output))
            return

        team = get_team(team_name)
        if team is None:
            clear_active_team_name(team_name)
            output = TeamDeleteOutput(
                success=False,
                message=f"Team '{team_name}' was not found.",
                team_name=team_name,
            )
            yield ToolResult(data=output, result_for_assistant=self.render_result_for_assistant(output))
            return

        active_members = [member.name for member in team.members if member.active]
        if active_members:
            output = TeamDeleteOutput(
                success=False,
                message=(
                    f"Cannot cleanup team with {len(active_members)} active member(s): "
                    + ", ".join(active_members)
                    + ". Use shutdown_request to gracefully terminate teammates first."
                ),
                team_name=team_name,
            )
            yield ToolResult(data=output, result_for_assistant=self.render_result_for_assistant(output))
            return

        removed = delete_team(team_name, purge_messages=True)
        task_dir = task_list_dir(task_list_id=team.task_list_id, ensure=False)
        try:
            shutil.rmtree(task_dir, ignore_errors=True)
        except OSError as exc:
            logger.warning("[team_tool] Failed deleting task dir: %s: %s", type(exc).__name__, exc)

        clear_active_team_name(team_name)
        _ACTIVE_TEAM_BY_AGENT.pop(_context_key(context), None)

        output = TeamDeleteOutput(
            success=removed,
            message=(
                f"Deleted team '{team_name}' and cleaned up resources."
                if removed
                else f"Team '{team_name}' was not removed."
            ),
            team_name=team_name,
        )
        yield ToolResult(data=output, result_for_assistant=self.render_result_for_assistant(output))


class SendMessageTool(Tool[SendMessageInput, SendMessageOutput]):
    @property
    def name(self) -> str:
        return "SendMessage"

    async def description(self) -> str:
        return (
            "Send structured intra-team protocol messages: "
            "message, broadcast, shutdown_request, shutdown_response, plan_approval_response."
        )

    @property
    def input_schema(self) -> type[SendMessageInput]:
        return SendMessageInput

    def input_examples(self) -> list[ToolUseExample]:
        return [
            ToolUseExample(
                description="Direct teammate message",
                example={
                    "type": "message",
                    "recipient": "researcher",
                    "content": "请先检查 auth 模块回归测试失败原因",
                    "summary": "Please investigate auth regression failures first",
                },
            ),
            ToolUseExample(
                description="Broadcast message",
                example={
                    "type": "broadcast",
                    "content": "暂停提交，主干构建故障，等待修复",
                    "summary": "Main branch broken pause all commits now",
                },
            ),
            ToolUseExample(
                description="Shutdown request",
                example={
                    "type": "shutdown_request",
                    "recipient": "tester",
                    "content": "任务已完成，请收尾并关闭",
                },
            ),
        ]

    async def prompt(self, _yolo_mode: bool = False) -> str:
        return SEND_MESSAGE_PROMPT

    def needs_permissions(self, _input_data: Optional[SendMessageInput] = None) -> bool:
        return False

    async def validate_input(
        self,
        input_data: SendMessageInput,
        _context: Optional[ToolUseContext] = None,
    ) -> ValidationResult:
        message_type = (input_data.type or "").strip()
        if message_type not in {
            "message",
            "broadcast",
            "shutdown_request",
            "shutdown_response",
            "plan_approval_response",
        }:
            return ValidationResult(result=False, message="Unsupported message type")

        if message_type == "message":
            if not (input_data.recipient or "").strip():
                return ValidationResult(result=False, message="recipient is required for type=message")
            if not (input_data.content or "").strip():
                return ValidationResult(result=False, message="content is required for type=message")
            if not (input_data.summary or "").strip():
                return ValidationResult(result=False, message="summary is required for type=message")
            summary_word_count = _summary_word_count(input_data.summary or "")
            if summary_word_count < 5 or summary_word_count > 10:
                return ValidationResult(
                    result=False,
                    message="summary must contain 5-10 words for type=message",
                )

        if message_type == "broadcast":
            if not (input_data.content or "").strip():
                return ValidationResult(result=False, message="content is required for type=broadcast")
            if not (input_data.summary or "").strip():
                return ValidationResult(result=False, message="summary is required for type=broadcast")
            summary_word_count = _summary_word_count(input_data.summary or "")
            if summary_word_count < 5 or summary_word_count > 10:
                return ValidationResult(
                    result=False,
                    message="summary must contain 5-10 words for type=broadcast",
                )

        if message_type == "shutdown_request":
            if not (input_data.recipient or "").strip():
                return ValidationResult(
                    result=False,
                    message="recipient is required for type=shutdown_request",
                )

        if message_type == "shutdown_response":
            if not (input_data.request_id or "").strip():
                return ValidationResult(
                    result=False,
                    message="request_id is required for type=shutdown_response",
                )
            if input_data.approve is None:
                return ValidationResult(
                    result=False,
                    message="approve is required for type=shutdown_response",
                )
            if input_data.approve is False and not (input_data.content or "").strip():
                return ValidationResult(
                    result=False,
                    message="content is required when approve=false",
                )

        if message_type == "plan_approval_response":
            if not (input_data.request_id or "").strip():
                return ValidationResult(
                    result=False,
                    message="request_id is required for type=plan_approval_response",
                )
            if input_data.approve is None:
                return ValidationResult(
                    result=False,
                    message="approve is required for type=plan_approval_response",
                )
            if not (input_data.recipient or "").strip():
                return ValidationResult(
                    result=False,
                    message="recipient is required for type=plan_approval_response",
                )

        return ValidationResult(result=True)

    def render_result_for_assistant(self, output: SendMessageOutput) -> str:
        return output.message

    def render_tool_use_message(self, input_data: SendMessageInput, _verbose: bool = False) -> str:
        return f"Sending {input_data.type} message"

    async def call(
        self,
        input_data: SendMessageInput,
        context: ToolUseContext,
    ) -> AsyncGenerator[ToolOutput, None]:
        team_name = _resolve_active_team_name(context)
        if not team_name:
            raise ValueError("No active team context found. Create/select a team first.")

        team = get_team(team_name)
        if team is None:
            raise ValueError(f"Team '{team_name}' not found.")

        sender = _sender_name(context)
        sender_color = _color_for(sender)

        message_type = input_data.type
        if message_type == "message":
            recipient = (input_data.recipient or "").strip()
            content = (input_data.content or "").strip()
            summary = (input_data.summary or "").strip()
            send_team_message(
                team_name=team_name,
                sender=sender,
                recipients=[recipient],
                message_type="message",
                content=content,
                metadata={"summary": summary},
            )
            output = SendMessageOutput(
                success=True,
                message=f"Message sent to {recipient}'s inbox",
                routing=SendMessageRouting(
                    sender=sender,
                    sender_color=sender_color,
                    target=f"@{recipient}",
                    target_color=_color_for(recipient),
                    summary=summary,
                    content=content,
                ),
            )
            yield ToolResult(data=output, result_for_assistant=self.render_result_for_assistant(output))
            return

        if message_type == "broadcast":
            content = (input_data.content or "").strip()
            summary = (input_data.summary or "").strip()
            recipients = [member.name for member in team.members if member.active]
            send_team_message(
                team_name=team_name,
                sender=sender,
                recipients=recipients or ["*"],
                message_type="broadcast",
                content=content,
                metadata={"summary": summary},
            )
            output = SendMessageOutput(
                success=True,
                message=(
                    f"Message broadcast to {len(recipients)} teammate(s): "
                    + (", ".join(recipients) if recipients else "*")
                ),
                recipients=recipients,
                routing=SendMessageRouting(
                    sender=sender,
                    sender_color=sender_color,
                    target="@team",
                    summary=summary,
                    content=content,
                ),
            )
            yield ToolResult(data=output, result_for_assistant=self.render_result_for_assistant(output))
            return

        if message_type == "shutdown_request":
            recipient = (input_data.recipient or "").strip()
            content = (input_data.content or "").strip()
            request_id = f"req_{uuid4().hex[:10]}"
            send_team_message(
                team_name=team_name,
                sender=sender,
                recipients=[recipient],
                message_type="shutdown_request",
                content=content or "Shutdown requested.",
                metadata={"request_id": request_id},
            )
            output = SendMessageOutput(
                success=True,
                message=f"Shutdown request sent to {recipient}. Request ID: {request_id}",
                request_id=request_id,
                target=recipient,
            )
            yield ToolResult(data=output, result_for_assistant=self.render_result_for_assistant(output))
            return

        if message_type == "shutdown_response":
            request_id = (input_data.request_id or "").strip()
            approve = bool(input_data.approve)
            content = (input_data.content or "").strip()
            send_team_message(
                team_name=team_name,
                sender=sender,
                recipients=["*"],
                message_type="shutdown_response",
                content=content or ("Approved" if approve else "Rejected"),
                metadata={"request_id": request_id, "approve": approve},
            )
            output = SendMessageOutput(
                success=True,
                message=(
                    f"Shutdown response recorded for request {request_id}: "
                    f"{'approved' if approve else 'rejected'}."
                ),
                request_id=request_id,
            )
            yield ToolResult(data=output, result_for_assistant=self.render_result_for_assistant(output))
            return

        request_id = (input_data.request_id or "").strip()
        approve = bool(input_data.approve)
        recipient = (input_data.recipient or "").strip()
        content = (input_data.content or "").strip()
        send_team_message(
            team_name=team_name,
            sender=sender,
            recipients=[recipient],
            message_type="plan_approval_response",
            content=content or ("Approved" if approve else "Rejected"),
            metadata={"request_id": request_id, "approve": approve},
        )
        output = SendMessageOutput(
            success=True,
            message=(
                f"Plan approval response sent to {recipient} for request {request_id}: "
                f"{'approved' if approve else 'rejected'}."
            ),
            request_id=request_id,
        )
        yield ToolResult(data=output, result_for_assistant=self.render_result_for_assistant(output))


__all__ = [
    "SendMessageTool",
    "TeamCreateTool",
    "TeamDeleteTool",
    "SendMessageInput",
    "TeamCreateInput",
    "TeamDeleteInput",
    "TeamMessage",
]
