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

    ## When to Use

    Use this tool proactively whenever:
    - The user explicitly asks to use a team, swarm, or group of agents
    - The user mentions wanting agents to work together, coordinate, or collaborate
    - A task is complex enough that it would benefit from parallel work by multiple agents (e.g., building a full-stack feature with frontend and backend work, refactoring a codebase while keeping tests passing, implementing a multi-step project with research, planning, and coding phases)

    When in doubt about whether a task warrants a team, prefer spawning a team.

    ## Choosing Agent Types for Teammates

    When spawning teammates via the Task tool, choose the `subagent_type` based on what tools the agent needs for its task. Each agent type has a different set of available tools — match the agent to the work:

    - **Read-only agents** (e.g., Explore, Plan) cannot edit or write files. Only assign them research, search, or planning tasks. Never assign them implementation work.
    - **Full-capability agents** (e.g., general-purpose) have access to all tools including file editing, writing, and bash. Use these for tasks that require making changes.
    - **Custom agents** defined in `.ripperdoc/agents/` may have their own tool restrictions. Check their descriptions to understand what they can and cannot do.

    Always review the agent type descriptions and their available tools listed in the Task tool prompt before selecting a `subagent_type` for a teammate.

    Create a new team to coordinate multiple agents working on a project. Teams have a 1:1 correspondence with task lists (Team = TaskList).

    ```json
    {
      "team_name": "my-project",
      "description": "Working on feature X"
    }
    ```

    This creates:
    - A team file at `~/.ripperdoc/teams/{team-name}/config.json`
    - A corresponding task list directory at `~/.ripperdoc/tasks/{team-name}/`

    ## Team Workflow

    1. **Create a team** with TeamCreate - this creates both the team and its task list
    2. **Create tasks** using the Task tools (TaskCreate, TaskList, etc.) - they automatically use the team's task list
    3. **Spawn teammates** using the Task tool with `team_name` and `teammate_name` parameters to create teammates that join the team
    4. **Assign tasks** using TaskUpdate with `owner` to give tasks to idle teammates
    5. **Teammates work on assigned tasks** and mark them completed via TaskUpdate
    6. **Teammates go idle between turns** - after each turn, teammates automatically go idle and send a notification. IMPORTANT: Be patient with idle teammates! Don't comment on their idleness until it actually impacts your work.
    7. **Shutdown your team** - when the task is completed, gracefully shut down your teammates via SendMessage with type: "shutdown_request".

    ## Task Ownership

    Tasks are assigned using TaskUpdate with the `owner` parameter. Any agent can set or change task ownership via TaskUpdate.

    ## Automatic Message Delivery

    **IMPORTANT**: Messages from teammates are automatically delivered to you. You do NOT need to manually check your inbox.

    When you spawn teammates:
    - They will send you messages when they complete tasks or need help
    - These messages appear automatically as new conversation turns (like user messages)
    - If you're busy (mid-turn), messages are queued and delivered when your turn ends
    - The UI shows a brief notification with the sender's name when messages are waiting

    Messages will be delivered automatically.

    When reporting on teammate messages, you do NOT need to quote the original message—it's already rendered to the user.

    ## Teammate Idle State

    Teammates go idle after every turn—this is completely normal and expected. A teammate going idle immediately after sending you a message does NOT mean they are done or unavailable. Idle simply means they are waiting for input.

    - **Idle teammates can receive messages.** Sending a message to an idle teammate wakes them up and they will process it normally.
    - **Idle notifications are automatic.** The system sends an idle notification whenever a teammate's turn ends. You do not need to react to idle notifications unless you want to assign new work or send a follow-up message.
    - **Do not treat idle as an error.** A teammate sending a message and then going idle is the normal flow—they sent their message and are now waiting for a response.
    - **Peer DM visibility.** When a teammate sends a DM to another teammate, a brief summary is included in their idle notification. This gives you visibility into peer collaboration without the full message content. You do not need to respond to these summaries — they are informational.

    ## Discovering Team Members

    Teammates can read the team config file to discover other team members:
    - **Team config location**: `~/.ripperdoc/teams/{team-name}/config.json`

    The config file contains a `members` array with each teammate's:
    - `name`: Human-readable name (**always use this** for messaging and task assignment)
    - `agentType`: Role/type of the agent

    **IMPORTANT**: Always refer to teammates by their NAME (e.g., "team-lead", "researcher", "tester"). Names are used for:
    - `recipient` when sending messages
    - Identifying task owners

    Example of reading team config:
    ```
    Use the Read tool to read ~/.ripperdoc/teams/{team-name}/config.json
    ```

    ## Task List Coordination

    Teams share a task list that all teammates can access at `~/.ripperdoc/tasks/{team-name}/`.

    Teammates should:
    1. Check TaskList periodically, **especially after completing each task**, to find available work or see newly unblocked tasks
    2. Claim unassigned, unblocked tasks with TaskUpdate (set `owner` to your name). **Prefer tasks in ID order** (lowest ID first) when multiple tasks are available, as earlier tasks often set up context for later ones
    3. Create new tasks with `TaskCreate` when identifying additional work
    4. Mark tasks as completed with `TaskUpdate` when done, then check TaskList for next work
    5. Coordinate with other teammates by reading the task list status
    6. If all available tasks are blocked, notify the team lead or help resolve blocking tasks

    **IMPORTANT notes for communication with your team**:
    - Do not use terminal tools to view your team's activity; always send a message to your teammates (and remember, refer to them by name).
    - Your team cannot hear you if you do not use the SendMessage tool. Always send a message to your teammates if you are responding to them.
    - Do NOT send structured JSON status messages like `{"type":"idle",...}` or `{"type":"task_completed",...}`. Just communicate in plain text when you need to message teammates.
    - Use TaskUpdate to mark tasks completed.
    - If you are an agent in the team, the system will automatically send idle notifications to the team lead when you stop.
    """
).strip()

TEAM_DELETE_PROMPT = dedent(
    """\
    # TeamDelete

    Remove team and task directories when the swarm work is complete.

    This operation:
    - Removes the team directory (`~/.ripperdoc/teams/{team-name}/`)
    - Removes the task directory (`~/.ripperdoc/tasks/{team-name}/`)
    - Clears team context from the current session

    **IMPORTANT**: TeamDelete will fail if the team still has active members. Gracefully terminate teammates first, then call TeamDelete after all teammates have shut down.

    Use this when all teammates have finished their work and you want to clean up the team resources. The team name is automatically determined from the current session's team context.
    """
).strip()

SEND_MESSAGE_PROMPT = dedent(
    """\
    # SendMessageTool

    Send messages to agent teammates and handle protocol requests/responses in a team.

    ## Message Types

    ### type: "message" - Send a Direct Message

    Send a message to a **single specific teammate**. You MUST specify the recipient.

    **IMPORTANT for teammates**: Your plain text output is NOT visible to the team lead or other teammates. To communicate with anyone on your team, you **MUST** use this tool. Just typing a response or acknowledgment in text is not enough.

    ```json
    {
      "type": "message",
      "recipient": "researcher",
      "content": "Your message here",
      "summary": "Brief status update on auth module"
    }
    ```

    - **recipient**: The name of the teammate to message (required)
    - **content**: The message text (required)
    - **summary**: A 5-10 word summary shown as preview in the UI (required)

    ### type: "broadcast" - Send Message to ALL Teammates (USE SPARINGLY)

    Send the **same message to everyone** on the team at once.

    **WARNING: Broadcasting is expensive.** Each broadcast sends a separate message to every teammate, which means:
    - N teammates = N separate message deliveries
    - Each delivery consumes API resources
    - Costs scale linearly with team size

    ```json
    {
      "type": "broadcast",
      "content": "Message to send to all teammates",
      "summary": "Critical blocking issue found"
    }
    ```

    - **content**: The message content to broadcast (required)
    - **summary**: A 5-10 word summary shown as preview in the UI (required)

    **CRITICAL: Use broadcast only when absolutely necessary.** Valid use cases:
    - Critical issues requiring immediate team-wide attention (e.g., "stop all work, blocking bug found")
    - Major announcements that genuinely affect every teammate equally

    **Default to "message" instead of "broadcast".** Use "message" for:
    - Responding to a single teammate
    - Normal back-and-forth communication
    - Following up on a task with one person
    - Sharing findings relevant to only some teammates
    - Any message that doesn't require everyone's attention

    ### type: "shutdown_request" - Request a Teammate to Shut Down

    Use this to ask a teammate to gracefully shut down:

    ```json
    {
      "type": "shutdown_request",
      "recipient": "researcher",
      "content": "Task complete, wrapping up the session"
    }
    ```

    The teammate will receive a shutdown request and can either approve (exit) or reject (continue working).

    ### type: "shutdown_response" - Respond to a Shutdown Request

    #### Approve Shutdown

    When you receive a shutdown request as a JSON message with `type: "shutdown_request"`, you **MUST** respond to approve or reject it. Do NOT just acknowledge the request in text - you must actually call this tool.

    ```json
    {
      "type": "shutdown_response",
      "request_id": "abc-123",
      "approve": true
    }
    ```

    **IMPORTANT**: Extract the `requestId` from the JSON message and pass it as `request_id` to the tool. Simply saying "I'll shut down" is not enough - you must call the tool.

    This will send confirmation to the leader and terminate your process.

    #### Reject Shutdown

    ```json
    {
      "type": "shutdown_response",
      "request_id": "abc-123",
      "approve": false,
      "content": "Still working on task #3, need 5 more minutes"
    }
    ```

    The leader will receive your rejection with the reason.

    ### type: "plan_approval_response" - Approve or Reject a Teammate's Plan

    #### Approve Plan

    When a teammate with `plan_mode_required` calls ExitPlanMode, they send you a plan approval request as a JSON message with `type: "plan_approval_request"`. Use this to approve their plan:

    ```json
    {
      "type": "plan_approval_response",
      "request_id": "abc-123",
      "recipient": "researcher",
      "approve": true
    }
    ```

    After approval, the teammate will automatically exit plan mode and can proceed with implementation.

    #### Reject Plan

    ```json
    {
      "type": "plan_approval_response",
      "request_id": "abc-123",
      "recipient": "researcher",
      "approve": false,
      "content": "Please add error handling for the API calls"
    }
    ```

    The teammate will receive the rejection with your feedback and can revise their plan.

    ## Important Notes

    - Messages from teammates are automatically delivered to you. You do NOT need to manually check your inbox.
    - When reporting on teammate messages, you do NOT need to quote the original message - it's already rendered to the user.
    - **IMPORTANT**: Always refer to teammates by their NAME (e.g., "team-lead", "researcher", "tester"), never by UUID.
    - Do NOT send structured JSON status messages. Use TaskUpdate to mark tasks completed and the system will automatically send idle notifications when you stop.
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
