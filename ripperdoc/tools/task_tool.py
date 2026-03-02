"""Task tool that delegates work to configured subagents."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    Iterable,
    Literal,
    List,
    Optional,
    Sequence,
    Union,
    cast,
)
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ripperdoc.core.agents import (
    AgentDefinition,
    AgentLoadResult,
    BASH_TOOL_NAME,
    FILE_EDIT_TOOL_NAME,
    GREP_TOOL_NAME,
    READ_TOOL_NAME,
    clear_agent_cache,
    load_agent_definitions,
    resolve_agent_tools,
    summarize_agent,
)
from ripperdoc.core.hooks.manager import HookResult, hook_manager
from ripperdoc.core.hooks.config import HooksConfig
from ripperdoc.core.query import QueryContext, query
from ripperdoc.core.system_prompt import build_environment_prompt
from ripperdoc.core.tool import (
    Tool,
    ToolOutput,
    ToolProgress,
    ToolResult,
    ToolUseContext,
    ValidationResult,
)
from ripperdoc.utils.messages import (
    AssistantMessage,
    UserMessage,
    create_hook_notice_payload,
    create_user_message,
)
from ripperdoc.utils.log import get_logger
from ripperdoc.utils.teams import (
    TeamMember,
    TeamMessageType,
    get_team,
    set_team_member_active,
    send_team_message,
    upsert_team_member,
)
from ripperdoc.utils.teammate_state import (
    IdleReason,
    InProcessTeammateState,
    set_teammate_idle,
    set_teammate_active,
    inject_user_message,
)
from ripperdoc.utils.worktree import (
    WorktreeSession,
    cleanup_worktree_session,
    create_task_worktree,
    has_worktree_changes,
    unregister_session_worktree,
)

logger = get_logger()


MessageType = Union[UserMessage, AssistantMessage]


@dataclass
class AgentRunRecord:
    """In-memory record for a subagent run (foreground or background).

    This record tracks both the run state and integrates with the teammate
    state management system for team coordination.
    """

    agent_id: str
    agent_type: str
    tools: List[Tool[Any, Any]]
    system_prompt: str
    history: List[MessageType]
    missing_tools: List[str]
    model_used: Optional[str]
    start_time: float
    duration_ms: float = 0.0
    tool_use_count: int = 0
    total_tokens: int = 0
    usage: Optional[Dict[str, Any]] = None
    status: str = "running"
    result_text: Optional[str] = None
    error: Optional[str] = None
    task_description: Optional[str] = None
    task_prompt: Optional[str] = None
    output_file: Optional[str] = None
    task: Optional[asyncio.Task] = None
    is_background: bool = False
    hook_scopes: List[HooksConfig] = field(default_factory=list)
    team_name: Optional[str] = None
    teammate_name: Optional[str] = None
    isolation_mode: Optional[str] = None
    worktree_path: Optional[str] = None
    worktree_branch: Optional[str] = None
    worktree_name: Optional[str] = None
    worktree_repo_root: Optional[str] = None
    worktree_head_commit: Optional[str] = None
    worktree_hook_based: bool = False

    # Idle state management
    is_idle: bool = False
    pending_user_messages: List[str] = field(default_factory=list)
    on_idle_callbacks: List[Callable[[], None]] = field(default_factory=list)

    # Permission mode (inherited from team lead)
    permission_mode: str = "default"  # "default", "plan", "dontAsk", "bypassPermissions", "acceptEdits"
    max_turns: Optional[int] = None

    # Plan approval workflow
    awaiting_plan_approval: bool = False

    # Shutdown protocol
    shutdown_requested: bool = False
    shutdown_request_id: Optional[str] = None

    # Linked teammate state (for advanced coordination)
    teammate_state: Optional[InProcessTeammateState] = None


_AGENT_RUNS: Dict[str, AgentRunRecord] = {}
_AGENT_RUNS_LOCK = threading.Lock()
DEFAULT_AGENT_RUN_TTL_SEC = float(os.getenv("RIPPERDOC_AGENT_RUN_TTL_SEC", "3600"))


def _new_agent_id() -> str:
    return f"agent_{uuid4().hex[:8]}"


def _task_output_root() -> Path:
    raw = os.getenv("RIPPERDOC_CONFIG_DIR")
    if raw and raw.strip():
        base = Path(raw).expanduser()
    else:
        base = Path.home() / ".ripperdoc"
    root = base / "tasks" / "outputs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _task_output_path(agent_id: str) -> str:
    safe_agent_id = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in agent_id)
    return str(_task_output_root() / f"{safe_agent_id}.log")


def _write_task_output(path: Optional[str], text: str, *, append: bool = True) -> None:
    if not path:
        return
    try:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with output_path.open(mode, encoding="utf-8") as handle:
            handle.write(text)
            if not text.endswith("\n"):
                handle.write("\n")
    except OSError as exc:
        logger.debug("[task_tool] Failed writing task output file: %s", exc)


def _register_agent_run(record: AgentRunRecord) -> None:
    with _AGENT_RUNS_LOCK:
        _AGENT_RUNS[record.agent_id] = record
    prune_agent_runs()


def _get_agent_run(agent_id: str) -> Optional[AgentRunRecord]:
    with _AGENT_RUNS_LOCK:
        return _AGENT_RUNS.get(agent_id)


def _snapshot_agent_run(record: AgentRunRecord) -> dict:
    duration_ms = (
        record.duration_ms
        if record.duration_ms
        else max((time.time() - record.start_time) * 1000.0, 0.0)
    )
    return {
        "id": record.agent_id,
        "agent_type": record.agent_type,
        "status": record.status,
        "duration_ms": duration_ms,
        "tool_use_count": record.tool_use_count,
        "total_tokens": record.total_tokens,
        "usage": record.usage,
        "missing_tools": list(record.missing_tools),
        "model_used": record.model_used,
        "result_text": record.result_text,
        "error": record.error,
        "task_description": record.task_description,
        "task_prompt": record.task_prompt,
        "output_file": record.output_file,
        "is_background": record.is_background,
        "team_name": record.team_name,
        "teammate_name": record.teammate_name,
        "isolation_mode": record.isolation_mode,
        "worktree_path": record.worktree_path,
        "worktree_branch": record.worktree_branch,
        "worktree_name": record.worktree_name,
        "worktree_repo_root": record.worktree_repo_root,
        "worktree_head_commit": record.worktree_head_commit,
        "worktree_hook_based": record.worktree_hook_based,
        # Idle state fields
        "is_idle": record.is_idle,
        "pending_messages": len(record.pending_user_messages),
        "permission_mode": record.permission_mode,
        "max_turns": record.max_turns,
        "shutdown_requested": record.shutdown_requested,
    }


def inject_user_message_to_teammate(
    agent_id: str,
    message: str,
) -> bool:
    """Inject a user message into a running teammate's pending queue.

    This allows the team lead to send messages to teammates that will be
    processed on their next polling cycle. If the teammate is idle, it will
    be woken up to process the message.

    Args:
        agent_id: The agent ID to inject the message into.
        message: The message content to inject.

    Returns:
        True if the message was successfully injected, False otherwise.
    """
    record = _get_agent_run(agent_id)
    if not record:
        logger.debug(
            "[task_tool] inject_user_message: agent %s not found",
            agent_id,
        )
        return False

    if record.status != "running":
        logger.debug(
            "[task_tool] inject_user_message: agent %s is not running (status=%s)",
            agent_id,
            record.status,
        )
        return False

    record.pending_user_messages.append(message)
    record.is_idle = False

    # Also sync with teammate_state if available
    if record.teammate_state:
        inject_user_message(record.teammate_state.id, message)

    logger.debug(
        "[task_tool] Injected message into %s's queue (depth=%d)",
        agent_id,
        len(record.pending_user_messages),
    )
    return True


def pop_pending_user_message_from_teammate(agent_id: str) -> Optional[str]:
    """Pop the next pending user message from a teammate's queue.

    Args:
        agent_id: The agent ID to pop the message from.

    Returns:
        The next message in the queue, or None if the queue is empty.
    """
    record = _get_agent_run(agent_id)
    if not record or not record.pending_user_messages:
        return None

    return record.pending_user_messages.pop(0)


def set_agent_idle_state(
    agent_id: str,
    is_idle: bool,
    *,
    idle_reason: Optional[IdleReason] = None,
    summary: Optional[str] = None,
) -> bool:
    """Set the idle state of an agent run.

    When setting to idle, this will also trigger idle callbacks and send
    an idle notification to the team lead.

    Args:
        agent_id: The agent ID to update.
        is_idle: Whether the agent should be marked as idle.
        idle_reason: The reason for going idle (required when is_idle=True).
        summary: Optional summary of what the agent accomplished.

    Returns:
        True if the state was updated, False if the agent was not found.
    """
    record = _get_agent_run(agent_id)
    if not record:
        return False

    if is_idle and not record.is_idle:
        record.is_idle = True

        # Execute idle callbacks
        for callback in record.on_idle_callbacks:
            try:
                callback()
            except Exception as exc:
                logger.warning(
                    "[task_tool] Idle callback failed for %s: %s: %s",
                    agent_id,
                    type(exc).__name__,
                    exc,
                )
        record.on_idle_callbacks = []

        # Sync with teammate_state
        if record.teammate_state and idle_reason:
            set_teammate_idle(
                record.teammate_state.id,
                idle_reason=idle_reason,
                summary=summary,
            )

        # Send idle notification to team lead
        if record.team_name:
            _send_idle_notification_to_team_lead(
                record,
                idle_reason=idle_reason or IdleReason.AVAILABLE,
                summary=summary,
            )

        logger.debug(
            "[task_tool] Agent %s is now idle (reason=%s)",
            agent_id,
            idle_reason.value if idle_reason else "unknown",
        )
    elif not is_idle and record.is_idle:
        record.is_idle = False
        if record.teammate_state:
            set_teammate_active(record.teammate_state.id)
        logger.debug("[task_tool] Agent %s is now active", agent_id)

    return True


def _send_idle_notification_to_team_lead(
    record: AgentRunRecord,
    *,
    idle_reason: IdleReason,
    summary: Optional[str] = None,
) -> None:
    """Send an idle notification message to the team lead."""
    if not record.team_name:
        return

    import json
    from datetime import datetime, timezone

    notification = {
        "type": "idle_notification",
        "from": record.teammate_name or record.agent_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "idleReason": idle_reason.value,
        "summary": summary,
        "agentId": record.agent_id,
        "completedStatus": record.status if record.status != "running" else None,
    }

    try:
        send_team_message(
            team_name=record.team_name,
            sender=record.teammate_name or record.agent_type,
            recipients=["team-lead"],
            message_type="idle_notification",
            content=json.dumps(notification, ensure_ascii=False),
            metadata={
                "idle_notification": True,
                "idle_reason": idle_reason.value,
            },
        )
    except Exception as exc:
        logger.warning(
            "[task_tool] Failed to send idle notification: %s: %s",
            type(exc).__name__,
            exc,
        )


def list_agent_runs() -> List[str]:
    """Return known subagent run ids."""
    prune_agent_runs()
    with _AGENT_RUNS_LOCK:
        return list(_AGENT_RUNS.keys())


def list_running_team_members(team_name: Optional[str] = None) -> list[str]:
    """Return teammate names that currently have a running execution state."""
    target = (team_name or "").strip()
    with _AGENT_RUNS_LOCK:
        names: set[str] = set()
        for record in _AGENT_RUNS.values():
            if record.status != "running":
                continue
            if not record.teammate_name:
                continue
            if target and (record.team_name or "").strip() != target:
                continue
            if record.task is not None and record.task.done():
                continue
            names.add(record.teammate_name)
        return sorted(names)


def list_running_agent_worktree_paths() -> set[str]:
    """Return worktree paths currently used by running subagents."""
    with _AGENT_RUNS_LOCK:
        paths: set[str] = set()
        for record in _AGENT_RUNS.values():
            if record.status != "running":
                continue
            if not record.worktree_path:
                continue
            if record.task is not None and record.task.done():
                continue
            paths.add(record.worktree_path)
        return paths


def get_agent_run_snapshot(agent_id: str) -> Optional[dict]:
    """Return a snapshot of a subagent run by id."""
    record = _get_agent_run(agent_id)
    if not record:
        return None
    return _snapshot_agent_run(record)


async def wait_for_agent_run_snapshot(
    agent_id: str,
    *,
    timeout_ms: int = 30_000,
) -> Optional[dict]:
    """Wait for a running agent record up to timeout and return its snapshot.

    Returns None when the record does not exist.
    """
    record = _get_agent_run(agent_id)
    if not record:
        return None

    if record.task and not record.task.done():
        timeout_s = max(timeout_ms, 0) / 1000.0
        try:
            await asyncio.wait_for(asyncio.shield(record.task), timeout=timeout_s)
        except asyncio.TimeoutError:
            pass
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            record.status = "failed"
            record.error = str(exc)

    return _snapshot_agent_run(record)


def prune_agent_runs(max_age_seconds: Optional[float] = None) -> int:
    """Remove finished subagent runs older than the TTL."""
    ttl = DEFAULT_AGENT_RUN_TTL_SEC if max_age_seconds is None else max_age_seconds
    if ttl is None or ttl <= 0:
        return 0
    now = time.time()
    removed = 0
    with _AGENT_RUNS_LOCK:
        for agent_id, record in list(_AGENT_RUNS.items()):
            if record.status == "running" or record.task:
                continue
            age = now - record.start_time
            if age > ttl:
                _AGENT_RUNS.pop(agent_id, None)
                removed += 1
    return removed


def _set_team_member_active_state(
    team_name: Optional[str],
    teammate_name: Optional[str],
    active: bool,
    *,
    default_agent_type: str = "general-purpose",
) -> None:
    if not team_name or not teammate_name:
        return
    try:
        set_team_member_active(
            team_name=team_name,
            member_name=teammate_name,
            active=active,
            default_agent_type=default_agent_type,
        )
    except (ValueError, OSError, RuntimeError, KeyError, TypeError):
        # Best-effort lifecycle tracking for team members.
        logger.debug(
            "[task_tool] Failed to update teammate active state",
            extra={"team_name": team_name, "teammate_name": teammate_name, "active": active},
        )


async def cancel_agent_run(agent_id: str) -> bool:
    """Cancel a running subagent, if possible."""
    record = _get_agent_run(agent_id)
    if not record or not record.task or record.task.done():
        return False
    record.task.cancel()
    try:
        await record.task
    except asyncio.CancelledError:
        pass
    record.status = "cancelled"
    _set_team_member_active_state(record.team_name, record.teammate_name, False)
    record.error = record.error or "Cancelled by user."
    record.duration_ms = (time.time() - record.start_time) * 1000
    record.task = None
    return True


class TaskToolInput(BaseModel):
    """Input schema for delegating to a subagent."""

    description: str = Field(
        description="A short (3-5 word) description of the task.",
    )
    prompt: Optional[str] = Field(
        default=None,
        description="Detailed task description for the subagent to perform.",
    )
    subagent_type: Optional[str] = Field(
        default=None,
        description="Agent type to run (matches agent frontmatter name). Required for new runs unless team_name+name are provided.",
    )
    team_name: Optional[str] = Field(
        default=None,
        description="Optional Team domain name. When provided with teammate_name, agent type is resolved from the team roster.",
    )
    teammate_name: Optional[str] = Field(
        default=None,
        description="Optional teammate identifier inside team_name. Resolves to that member's configured agent type.",
    )
    name: Optional[str] = Field(
        default=None,
        description="Alias of teammate_name when spawning into a team.",
    )
    run_in_background: bool = Field(
        default=False,
        description="If true, start the agent in the background and return immediately.",
    )
    mode: Optional[Literal["default", "plan", "acceptEdits", "bypassPermissions"]] = Field(
        default=None,
        description="Teammate permission mode hint. Currently informational.",
    )
    isolation: Optional[Literal["worktree"]] = Field(
        default=None,
        description='Isolation mode. "worktree" runs the subagent in an isolated git worktree.',
    )
    model: Optional[str] = Field(
        default=None,
        description=(
            "Optional model to use for this agent. If omitted, inherits from parent model. "
            "Prefer haiku for quick, straightforward tasks."
        ),
    )
    max_turns: Optional[int] = Field(
        default=None,
        ge=1,
        description="Optional turn limit hint.",
    )
    resume: Optional[str] = Field(
        default=None,
        description="Agent id to resume from a previous agent context.",
    )
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _normalize_teammate_aliases(self) -> "TaskToolInput":
        self.description = (self.description or "").strip()
        if self.name and self.teammate_name and self.name != self.teammate_name:
            raise ValueError("name and teammate_name must match when both are provided.")
        if self.name and not self.teammate_name:
            self.teammate_name = self.name
        elif self.teammate_name and not self.name:
            self.name = self.teammate_name
        return self


class TaskToolContentBlock(BaseModel):
    """Text content block returned by Task tool."""

    type: Literal["text"] = "text"
    text: str


class TaskToolOutput(BaseModel):
    """Task tool output payload."""

    status: str = "completed"
    agent_id: Optional[str] = Field(
        default=None,
        validation_alias="agentId",
        serialization_alias="agentId",
    )
    description: Optional[str] = None
    prompt: Optional[str] = None
    output_file: Optional[str] = Field(
        default=None,
        validation_alias="outputFile",
        serialization_alias="outputFile",
    )
    can_read_output_file: Optional[bool] = Field(
        default=None,
        validation_alias="canReadOutputFile",
        serialization_alias="canReadOutputFile",
    )
    content: List[TaskToolContentBlock] = Field(default_factory=list)
    total_tool_use_count: int = Field(
        default=0,
        validation_alias="totalToolUseCount",
        serialization_alias="totalToolUseCount",
    )
    total_duration_ms: float = Field(
        default=0.0,
        validation_alias="totalDurationMs",
        serialization_alias="totalDurationMs",
    )
    total_tokens: int = Field(
        default=0,
        validation_alias="totalTokens",
        serialization_alias="totalTokens",
    )
    usage: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    agent_type: Optional[str] = None
    missing_tools: List[str] = Field(default_factory=list)
    model_used: Optional[str] = None
    is_background: bool = False
    is_resumed: bool = False
    isolation: Optional[str] = None
    worktree_path: Optional[str] = None
    worktree_branch: Optional[str] = None
    result_text: Optional[str] = None
    duration_ms: Optional[float] = None
    tool_use_count: Optional[int] = None

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class TaskTool(Tool[TaskToolInput, TaskToolOutput]):
    """Launches a configured agent in a fresh context."""

    def __init__(self, available_tools_provider: Callable[[], Iterable[Tool[Any, Any]]]) -> None:
        super().__init__()
        self._available_tools_provider = available_tools_provider

    @property
    def name(self) -> str:
        return "Task"

    async def description(self) -> str:
        clear_agent_cache()
        agents = load_agent_definitions()
        agent_lines = "\n".join(summarize_agent(agent) for agent in agents.active_agents)
        return (
            "Launch a specialized subagent in its own context window to handle a task.\n"
            f"Available agents:\n{agent_lines or '- general-purpose (built-in)'}"
        )

    @property
    def input_schema(self) -> type[TaskToolInput]:
        return TaskToolInput

    async def prompt(self, yolo_mode: bool = False) -> str:
        del yolo_mode
        clear_agent_cache()
        agents: AgentLoadResult = load_agent_definitions()

        agent_lines: List[str] = []
        for agent in agents.active_agents:
            properties = (
                "Properties: access to current context; "
                if getattr(agent, "fork_context", False)
                else ""
            )
            tools_label = "All tools"
            if getattr(agent, "tools", None):
                tools_label = "All tools" if "*" in agent.tools else ", ".join(agent.tools)
            agent_lines.append(
                f"- {agent.agent_type}: {agent.when_to_use} ({properties}Tools: {tools_label})"
            )

        agent_block = "\n".join(agent_lines) or "- general-purpose (built-in)"

        task_tool_name = self.name
        file_read_tool_name = READ_TOOL_NAME
        search_tool_name = GREP_TOOL_NAME
        code_tool_name = FILE_EDIT_TOOL_NAME
        background_fetch_tool_name = "TaskOutput"

        return (
            f"Launch a new agent to handle complex, multi-step tasks autonomously. \n\n"
            f"The {task_tool_name} tool launches specialized agents (subprocesses) that autonomously handle complex tasks. Each agent type has specific capabilities and tools available to it.\n\n"
            f"Available agent types and the tools they have access to:\n"
            f"{agent_block}\n\n"
            f"When starting a new agent with the {task_tool_name} tool, you must specify a subagent_type parameter to select which agent type to use.\n\n"
            f"When NOT to use the {task_tool_name} tool:\n"
            f"- If you want to read a specific file path, use the {file_read_tool_name} or {search_tool_name} tool instead of the {task_tool_name} tool, to find the match more quickly\n"
            f'- If you are searching for a specific class definition like "class Foo", use the {search_tool_name} tool instead, to find the match more quickly\n'
            f"- If you are searching for code within a specific file or set of 2-3 files, use the {file_read_tool_name} tool instead of the {task_tool_name} tool, to find the match more quickly\n"
            "- Other tasks that are not related to the agent descriptions above\n"
            "\n"
            "\n"
            "Usage notes:\n"
            "- Launch multiple agents concurrently whenever possible, to maximize performance; to do that, use a single message with multiple tool uses\n"
            "- When the agent is done, it will return a single message back to you. The result returned by the agent is not visible to the user. To show the user the result, you should send a text message back to the user with a concise summary of the result.\n"
            f"- Use run_in_background=true to launch an agent asynchronously. The tool will return an agent_id immediately for later retrieval.\n"
            f"- Check background progress/output by calling {background_fetch_tool_name} with task_id=<agent_id>.\n"
            "- To continue a completed agent, call Task with resume=<agent_id> and a new prompt.\n"
            '- Use isolation="worktree" to run the task in a dedicated git worktree under .ripperdoc/worktrees/. If the subagent makes no changes, the worktree is auto-cleaned; if changes are made, worktree path/branch are returned.\n'
            "- Provide clear, detailed prompts so the agent can work autonomously and return exactly the information you need.\n"
            "- Agents can opt into parent context by setting fork_context: true in their frontmatter. When enabled, they receive the full conversation history before the tool call.\n"
            "- The agent's outputs should generally be trusted\n"
            "- Clearly tell the agent whether you expect it to write code or just to do research (search, file reads, web fetches, etc.), since it is not aware of the user's intent\n"
            "- If the agent description mentions that it should be used proactively, then you should try your best to use it without the user having to ask for it first. Use your judgement.\n"
            f'- If the user specifies that they want you to run agents "in parallel", you MUST send a single message with multiple {task_tool_name} tool use content blocks. For example, if you need to launch both a code-reviewer agent and a test-runner agent in parallel, send a single message with both tool calls.\n'
            "\n"
            "Example usage:\n"
            "\n"
            "<example_agent_descriptions>\n"
            '"code-reviewer": use this agent after you are done writing a significant piece of code\n'
            '"greeting-responder": use this agent when to respond to user greetings with a friendly joke\n'
            "</example_agent_description>\n"
            "\n"
            "<example>\n"
            'user: "Please write a function that checks if a number is prime"\n'
            "assistant: Sure let me write a function that checks if a number is prime\n"
            f"assistant: First let me use the {code_tool_name} tool to write a function that checks if a number is prime\n"
            f"assistant: I'm going to use the {code_tool_name} tool to write the following code:\n"
            "<code>\n"
            "function isPrime(n) {\n"
            "  if (n <= 1) return false\n"
            "  for (let i = 2; i * i <= n; i++) {\n"
            "    if (n % i === 0) return false\n"
            "  }\n"
            "  return true\n"
            "}\n"
            "</code>\n"
            "<commentary>\n"
            "Since a significant piece of code was written and the task was completed, now use the code-reviewer agent to review the code\n"
            "</commentary>\n"
            "assistant: Now let me use the code-reviewer agent to review the code\n"
            f"assistant: Uses the {task_tool_name} tool to launch the code-reviewer agent \n"
            "</example>\n"
            "\n"
            "<example>\n"
            'user: "Hello"\n'
            "<commentary>\n"
            "Since the user is greeting, use the greeting-responder agent to respond with a friendly joke\n"
            "</commentary>\n"
            f'assistant: "I\'m going to use the {task_tool_name} tool to launch the greeting-responder agent"\n'
            "</example>"
        )

    def is_read_only(self) -> bool:
        return True

    def is_concurrency_safe(self) -> bool:
        return True

    async def validate_input(
        self, input_data: TaskToolInput, context: Optional[ToolUseContext] = None
    ) -> ValidationResult:
        if not (input_data.description or "").strip():
            return ValidationResult(
                result=False,
                message="description is required (3-5 words).",
            )
        if input_data.resume and input_data.run_in_background:
            return ValidationResult(
                result=False,
                message="run_in_background cannot be used when resuming an agent.",
            )
        if input_data.resume and input_data.isolation is not None:
            return ValidationResult(
                result=False,
                message=(
                    "Cannot change isolation when resuming an existing agent. "
                    "The resumed agent keeps its original isolation/worktree."
                ),
            )
        if context and context.teammate_name:
            if input_data.team_name or input_data.teammate_name or input_data.name:
                return ValidationResult(
                    result=False,
                    message=(
                        "In-process teammates cannot set team_name/name or spawn teammates. "
                        "Only the team lead can spawn teammates."
                    ),
                )
            if input_data.isolation is not None:
                return ValidationResult(
                    result=False,
                    message="In-process teammates cannot spawn worktree-isolated subagents.",
                )
            if input_data.mode:
                return ValidationResult(
                    result=False,
                    message=(
                        "In-process teammates cannot override mode via Task. "
                        "Use a synchronous subagent with inherited permissions."
                    ),
                )
            if input_data.run_in_background:
                return ValidationResult(
                    result=False,
                    message=(
                        "In-process teammates cannot spawn background agents. "
                        "Use run_in_background=false for synchronous subagents."
                    ),
                )
        if input_data.teammate_name and not input_data.team_name:
            return ValidationResult(
                result=False,
                message="team_name is required when teammate_name is provided.",
            )
        if input_data.resume:
            if not input_data.prompt or not input_data.prompt.strip():
                return ValidationResult(
                    result=False,
                    message=(
                        "prompt is required when using resume. "
                        "Use TaskOutput to check background progress/output."
                    ),
                )
            resumed_record = _get_agent_run(input_data.resume)
            if resumed_record and resumed_record.status == "running":
                return ValidationResult(
                    result=False,
                    message=(
                        f"Cannot resume agent '{input_data.resume}' while it is still running. "
                        "Use TaskOutput to check progress."
                    ),
                )
            return ValidationResult(result=True)

        if not input_data.subagent_type and not (input_data.team_name and input_data.teammate_name):
            return ValidationResult(
                result=False,
                message=(
                    "subagent_type is required when starting a new agent "
                    "(unless team_name + name is provided)."
                ),
            )
        if not input_data.prompt or not input_data.prompt.strip():
            return ValidationResult(
                result=False,
                message="prompt is required when starting a new agent.",
            )
        return ValidationResult(result=True)

    def render_result_for_assistant(self, output: TaskToolOutput) -> str:
        if output.status == "async_launched":
            lines = [
                "Async agent launched successfully.",
                f"agentId: {output.agent_id}",
            ]
            if output.output_file:
                lines.append(f"outputFile: {output.output_file}")
            return "\n".join(lines)

        if output.status == "teammate_spawned":
            return (
                f"Spawned teammate successfully.\nagentId: {output.agent_id}\n"
                f"team: {output.description or 'team'}"
            )

        if output.status == "failed":
            return f"Task failed (agentId={output.agent_id}): {output.error or 'unknown error'}"

        details: List[str] = []
        if output.agent_id:
            details.append(f"id={output.agent_id}")
        if output.total_tool_use_count:
            details.append(f"{output.total_tool_use_count} tool uses")
        if output.total_duration_ms:
            details.append(f"{output.total_duration_ms / 1000:.1f}s")
        if output.worktree_path:
            details.append(f"worktree={output.worktree_path}")
        if output.worktree_branch:
            details.append(f"branch={output.worktree_branch}")
        if output.error:
            details.append(f"error: {output.error}")

        text = output.result_text or (
            output.content[0].text if output.content else "Agent returned no response."
        )
        suffix = f" ({'; '.join(details)})" if details else ""
        return f"[subagent:{output.agent_type or 'unknown'}] {text}{suffix}"

    def render_tool_use_message(self, input_data: TaskToolInput, verbose: bool = False) -> str:
        del verbose
        if input_data.resume:
            return f"Resume subagent {input_data.resume}"
        target = input_data.subagent_type or "team-resolved"
        if input_data.team_name and input_data.teammate_name:
            target = f"{input_data.team_name}/{input_data.teammate_name}"
        label = f"Task via {target}: {input_data.description}"
        if input_data.run_in_background or bool(input_data.team_name and input_data.teammate_name):
            label += " (background)"
        return label

    async def _run_subagent_start_hook(
        self,
        context: ToolUseContext,
        *,
        subagent_type: str,
        prompt: Optional[str],
        resume: Optional[str],
        run_in_background: bool,
    ) -> HookResult:
        result = await hook_manager.run_subagent_start_async(
            subagent_type=subagent_type,
            prompt=prompt,
            resume=resume,
            run_in_background=run_in_background,
            tool_use_id=context.message_id,
        )
        return result

    def _build_subagent_hook_context(self, result: HookResult) -> Dict[str, str]:
        context: Dict[str, str] = {}
        if result.additional_context:
            context["Hook:SubagentStart"] = result.additional_context
        return context

    def _render_tool_result(self, output: TaskToolOutput) -> ToolResult:
        return ToolResult(
            data=output, result_for_assistant=self.render_result_for_assistant(output)
        )

    def _resolve_team_agent_target(
        self,
        *,
        team_name: Optional[str],
        teammate_name: Optional[str],
        fallback_agent_type: Optional[str],
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """Resolve agent type from team roster when team context is provided."""
        if not team_name:
            return fallback_agent_type, None, None

        team = get_team(team_name)
        if team is None:
            raise ValueError(f"Team '{team_name}' not found.")

        if teammate_name:
            member = next((item for item in team.members if item.name == teammate_name), None)
            if member is None:
                resolved_type = (
                    (fallback_agent_type or "").strip()
                    or str(team.metadata.get("agent_type") or "").strip()
                    or "general-purpose"
                )
                try:
                    upsert_team_member(
                        team.name,
                        TeamMember(
                            name=teammate_name,
                            agent_id=f"{teammate_name}@{team.name}",
                            agent_type=resolved_type,
                            backend_type="in-process",
                            role="worker",
                            active=False,
                        ),
                    )
                except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
                    raise ValueError(
                        f"Teammate '{teammate_name}' not found in team '{team_name}', "
                        "and automatic teammate registration failed."
                    ) from exc
                return resolved_type, team.name, teammate_name
            return member.agent_type, team.name, member.name

        return fallback_agent_type, team.name, None

    @staticmethod
    def _extract_team_event_metadata(message: UserMessage) -> Dict[str, Any]:
        message_payload = getattr(message, "message", None)
        if not message_payload:
            return {}
        metadata = getattr(message_payload, "metadata", None)
        if isinstance(metadata, dict):
            return metadata
        return {}

    @staticmethod
    def _get_block_attr(block: Any, attr_name: str, default: Any = None) -> Any:
        """Get attribute from block, supporting both object and dict access."""
        value = getattr(block, attr_name, None)
        if value is None and isinstance(block, dict):
            return block.get(attr_name, default)
        return value if value is not None else default

    @staticmethod
    def _extract_tool_result_ids(
        message: UserMessage,
    ) -> List[str]:
        payload = getattr(message, "message", None)
        content = getattr(payload, "content", None) if payload is not None else None
        if not isinstance(content, list):
            return []

        tool_result_ids: List[str] = []
        for block in content:
            block_type = TaskTool._get_block_attr(block, "type") or ""
            if block_type != "tool_result":
                continue
            tool_use_id = TaskTool._get_block_attr(block, "tool_use_id") or ""
            if isinstance(tool_use_id, str) and tool_use_id.strip():
                tool_result_ids.append(tool_use_id.strip())
        return tool_result_ids

    @staticmethod
    def _normalize_tool_input(raw_input: Any) -> Dict[str, Any]:
        """Normalize tool input to a plain dictionary."""
        if hasattr(raw_input, "model_dump"):
            model_dump = raw_input.model_dump()
            if isinstance(model_dump, dict):
                return cast(Dict[str, Any], model_dump)
            return {}
        if hasattr(raw_input, "dict"):
            dict_method = getattr(raw_input, "dict")
            if callable(dict_method):
                as_dict = dict_method()
                if isinstance(as_dict, dict):
                    return cast(Dict[str, Any], as_dict)
                return {}
        if isinstance(raw_input, dict):
            return dict(raw_input)
        return {}

    @staticmethod
    def _lookup_tool_use_input_by_id(
        history: Sequence[MessageType],
        tool_use_id: str,
    ) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
        if not tool_use_id:
            return None, None

        for item in reversed(history):
            if getattr(item, "type", "") != "assistant":
                continue

            payload = getattr(item, "message", None)
            content = getattr(payload, "content", None) if payload is not None else None
            if not isinstance(content, list):
                continue

            for block in content:
                block_type = TaskTool._get_block_attr(block, "type") or ""
                if block_type != "tool_use":
                    continue

                block_id = TaskTool._get_block_attr(block, "id") or TaskTool._get_block_attr(
                    block, "tool_use_id"
                )
                if str(block_id or "").strip() != tool_use_id:
                    continue

                tool_name = TaskTool._get_block_attr(block, "name")
                raw_input = TaskTool._get_block_attr(block, "input")
                parsed_input = TaskTool._normalize_tool_input(raw_input)
                return (str(tool_name) if tool_name else None), parsed_input

        return None, None

    @staticmethod
    def _extract_approved_shutdown_response(
        history: Sequence[MessageType],
        message: UserMessage,
    ) -> Optional[Dict[str, str]]:
        for tool_use_id in TaskTool._extract_tool_result_ids(message):
            tool_name, tool_input = TaskTool._lookup_tool_use_input_by_id(history, tool_use_id)
            if tool_name != "SendMessage":
                continue
            if tool_input is None:
                continue
            if str(tool_input.get("type") or "").strip() != "shutdown_response":
                continue
            if not bool(tool_input.get("approve")):
                continue
            request_id = str(tool_input.get("request_id") or "").strip()
            reason = str(tool_input.get("content") or "").strip()
            return {
                "request_id": request_id,
                "content": reason,
            }
        return None

    def _send_team_event(
        self,
        *,
        record: AgentRunRecord,
        message_type: TeamMessageType,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not record.team_name:
            return
        recipients = ["team-lead"]
        sender = record.teammate_name or record.agent_type
        try:
            send_team_message(
                team_name=record.team_name,
                sender=sender,
                recipients=recipients,
                message_type=message_type,
                content=content,
                metadata=metadata or {},
            )
        except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
            logger.warning(
                "[task_tool] Failed to emit team event: %s: %s",
                type(exc).__name__,
                exc,
                extra={"team_name": record.team_name, "message_type": message_type},
            )

    async def _wait_for_running_record(self, record: AgentRunRecord) -> None:
        if not record.task or record.task.done():
            return
        try:
            await record.task
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            record.status = "failed"
            record.error = str(exc)

    def _build_subagent_start_notices(
        self,
        hook_result: HookResult,
        *,
        agent_type: str,
    ) -> List[ToolProgress]:
        notices: List[ToolProgress] = []
        if hook_result.should_block or hook_result.should_ask or not hook_result.should_continue:
            reason = (
                hook_result.block_reason
                or hook_result.stop_reason
                or "SubagentStart hook requested to stop."
            )
            notices.append(
                ToolProgress(
                    content=create_hook_notice_payload(
                        text=f"SubagentStart hook warning (ignored): {reason}",
                        hook_event="SubagentStart",
                        tool_name=agent_type,
                        level="warning",
                    )
                )
            )
        if hook_result.system_message:
            notices.append(
                ToolProgress(
                    content=create_hook_notice_payload(
                        text=str(hook_result.system_message),
                        hook_event="SubagentStart",
                        tool_name=agent_type,
                    )
                )
            )
        return notices

    def _reset_record_for_resume_prompt(self, record: AgentRunRecord, prompt: str) -> None:
        record.history.append(create_user_message(prompt))
        record.task_prompt = prompt
        record.start_time = time.time()
        record.duration_ms = 0.0
        record.tool_use_count = 0
        record.total_tokens = 0
        record.usage = None
        record.status = "running"
        record.result_text = None
        record.error = None
        record.task = None
        _write_task_output(
            record.output_file,
            f"=== Resume {time.strftime('%Y-%m-%d %H:%M:%S')} ===\nPrompt: {prompt}",
            append=True,
        )

    def _build_subagent_query_context(
        self,
        *,
        tools: List[Tool[Any, Any]],
        yolo_mode: bool,
        verbose: bool,
        model: str,
        agent_type: str,
        team_name: Optional[str],
        teammate_name: Optional[str],
        agent_id: str,
        hook_scopes: List[HooksConfig],
        max_turns: Optional[int] = None,
        permission_mode: str = "default",
        working_directory: Optional[str] = None,
    ) -> QueryContext:
        return QueryContext(
            tools=tools,
            yolo_mode=yolo_mode,
            verbose=verbose,
            model=model,
            stop_hook="subagent",
            subagent_type=agent_type,
            team_name=team_name,
            teammate_name=teammate_name,
            agent_id=agent_id,
            hook_scopes=hook_scopes,
            max_turns=max_turns,
            permission_mode=permission_mode,
            working_directory=working_directory,
        )

    def _finalize_record_from_messages(
        self,
        record: AgentRunRecord,
        *,
        assistant_messages: List[AssistantMessage],
        tool_use_count: int,
        status: str = "completed",
        error: Optional[str] = None,
        result_text: Optional[str] = None,
    ) -> None:
        duration_ms = (time.time() - record.start_time) * 1000
        if result_text is None:
            result_text = (
                self._extract_text(assistant_messages[-1])
                if assistant_messages
                else (
                    f"Subagent '{record.agent_type}' ended with status '{status}'."
                    if status != "completed"
                    else "Agent returned no response."
                )
            )
        record.duration_ms = duration_ms
        record.tool_use_count = tool_use_count
        record.result_text = result_text.strip()
        total_input_tokens = sum(max(getattr(msg, "input_tokens", 0), 0) for msg in assistant_messages)
        total_output_tokens = sum(max(getattr(msg, "output_tokens", 0), 0) for msg in assistant_messages)
        cache_read_tokens = sum(max(getattr(msg, "cache_read_tokens", 0), 0) for msg in assistant_messages)
        cache_creation_tokens = sum(
            max(getattr(msg, "cache_creation_tokens", 0), 0) for msg in assistant_messages
        )
        record.total_tokens = total_input_tokens + total_output_tokens
        record.usage = {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "cache_creation_input_tokens": cache_creation_tokens or None,
            "cache_read_input_tokens": cache_read_tokens or None,
            "server_tool_use": None,
            "service_tier": None,
            "cache_creation": None,
        }
        record.status = status
        if status == "completed":
            record.error = None
        elif error is not None:
            record.error = error
        self._maybe_autocleanup_worktree(record)

        # Set idle state and send idle notification
        idle_reason = IdleReason.AVAILABLE
        if status == "failed":
            idle_reason = IdleReason.FAILED
        elif status == "cancelled":
            idle_reason = IdleReason.INTERRUPTED
        elif status == "shutdown":
            idle_reason = IdleReason.SHUTDOWN

        record.is_idle = True

        # Execute idle callbacks
        for callback in record.on_idle_callbacks:
            try:
                callback()
            except Exception as exc:
                logger.warning(
                    "[task_tool] Idle callback failed for %s: %s: %s",
                    record.agent_id,
                    type(exc).__name__,
                    exc,
                )
        record.on_idle_callbacks = []

        # Sync with teammate_state if available
        if record.teammate_state:
            set_teammate_idle(
                record.teammate_state.id,
                idle_reason=idle_reason,
                summary=result_text[:500] if result_text else None,
            )

        # Send idle notification to team lead
        if record.team_name:
            _send_idle_notification_to_team_lead(
                record,
                idle_reason=idle_reason,
                summary=result_text[:500] if result_text else None,
            )

        _set_team_member_active_state(
            record.team_name,
            record.teammate_name,
            False,
            default_agent_type=record.agent_type,
        )
        self._send_team_event(
            record=record,
            message_type="status",
            content=(
                f"Subagent '{record.agent_type}' {record.status}"
                + (f" for teammate '{record.teammate_name}'" if record.teammate_name else "")
                + "."
            ),
            metadata={
                "agent_id": record.agent_id,
                "status": record.status,
                "tool_use_count": record.tool_use_count,
            },
        )

    def _maybe_autocleanup_worktree(self, record: AgentRunRecord) -> None:
        """Auto-clean worktree when there are no changes."""
        if record.isolation_mode != "worktree":
            return
        if not record.worktree_path:
            return

        worktree_path = Path(record.worktree_path)
        if not worktree_path.exists():
            unregister_session_worktree(worktree_path)
            return

        baseline_ref = record.worktree_head_commit or record.worktree_branch or None
        try:
            changed = has_worktree_changes(
                worktree_path=worktree_path,
                baseline_ref=baseline_ref,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            logger.debug(
                "[task_tool] Failed to evaluate worktree changes",
                extra={
                    "agent_id": record.agent_id,
                    "worktree_path": record.worktree_path,
                    "error": str(exc),
                },
            )
            return

        if changed:
            return

        repo_root = (
            Path(record.worktree_repo_root).resolve()
            if record.worktree_repo_root
            else worktree_path.parent.parent.parent.resolve()
        )
        session = WorktreeSession(
            repo_root=repo_root,
            worktree_path=worktree_path.resolve(),
            branch=record.worktree_branch or "",
            name=record.worktree_name or worktree_path.name,
            head_commit=record.worktree_head_commit,
            hook_based=record.worktree_hook_based,
        )
        cleanup = cleanup_worktree_session(session, force=True)
        if cleanup.removed:
            unregister_session_worktree(worktree_path)
            record.worktree_path = None
        if cleanup.branch_deleted:
            record.worktree_branch = None
        if cleanup.removed and (cleanup.branch_deleted or not session.branch):
            if record.result_text:
                record.result_text = (
                    f"{record.result_text} (No worktree changes detected; temporary worktree cleaned up automatically.)"
                )
            return

        if cleanup.error:
            record.error = (
                f"{record.error}; auto-cleanup failed: {cleanup.error}"
                if record.error
                else f"auto-cleanup failed: {cleanup.error}"
            )

    async def _run_subagent_foreground(
        self,
        *,
        record: AgentRunRecord,
        subagent_context: QueryContext,
        permission_checker: Any,
        hook_context: Dict[str, str],
        parent_tool_use_id: Optional[str],
    ) -> AsyncGenerator[ToolProgress, None]:
        assistant_messages: List[AssistantMessage] = []
        tool_use_count = 0
        finalize_status = "running"
        finalize_error: Optional[str] = None
        finalize_result_text: Optional[str] = None
        try:
            async for message in query(
                record.history,  # type: ignore[arg-type]
                record.system_prompt,
                hook_context,
                subagent_context,
                permission_checker,
            ):
                msg_type = getattr(message, "type", "")
                if msg_type == "progress":
                    continue

                tool_use_count, updates = self._track_subagent_message(
                    record,
                    message,
                    record.history,
                    assistant_messages,
                    tool_use_count,
                )
                if isinstance(message, UserMessage):
                    shutdown_approval = self._extract_approved_shutdown_response(
                        record.history,
                        message,
                    )
                    if shutdown_approval is not None:
                        finalize_status = "shutdown"
                        finalize_error = (
                            shutdown_approval.get("content")
                            or "Approved shutdown_response sent to team lead."
                        ).strip()
                        finalize_result_text = (
                            "Subagent exited after approved shutdown_response"
                            + (
                                f" (request_id={shutdown_approval.get('request_id', '')})."
                                if shutdown_approval.get("request_id")
                                else "."
                            )
                        )
                        yield ToolProgress(
                            content=(
                                f"Shutdown approved for subagent '{record.agent_id}', exiting run."
                            ),
                            progress_sender=self._subagent_progress_sender(record),
                        )
                        break
                for sender, text in updates:
                    yield ToolProgress(content=text, progress_sender=sender)

                if msg_type in ("assistant", "user"):
                    message_with_parent = (
                        message.model_copy(update={"parent_tool_use_id": parent_tool_use_id})
                        if parent_tool_use_id
                        else message
                    )
                    yield ToolProgress(
                        content=message_with_parent,
                        is_subagent_message=True,
                        progress_sender=self._subagent_progress_sender(record),
                    )
        except asyncio.CancelledError:
            finalize_status = "cancelled"
            finalize_error = "Subagent run was cancelled."
            raise
        except Exception as exc:
            finalize_status = "failed"
            finalize_error = str(exc)
            logger.warning(
                "[task_tool] Subagent foreground run failed: %s: %s",
                type(exc).__name__,
                exc,
                extra={"agent_id": record.agent_id, "team_name": record.team_name},
            )
        finally:
            if finalize_status == "running":
                finalize_status = "completed"
            self._finalize_record_from_messages(
                record,
                assistant_messages=assistant_messages,
                tool_use_count=tool_use_count,
                status=finalize_status,
                error=finalize_error,
                result_text=finalize_result_text,
            )

    def _coerce_agent_tools(self, tools: List[object]) -> List[Tool[Any, Any]]:
        from ripperdoc.core.tool import Tool as ToolBase

        return [tool for tool in tools if isinstance(tool, ToolBase)]

    async def _handle_resume_call(
        self,
        input_data: TaskToolInput,
        context: ToolUseContext,
    ) -> AsyncGenerator[ToolOutput, None]:
        if not input_data.resume:
            return

        record = _get_agent_run(input_data.resume)
        if not record:
            raise ValueError(
                f"Agent id '{input_data.resume}' not found. "
                "Start a new agent to obtain a valid agent_id."
            )
        should_activate = bool(record.team_name and record.teammate_name)

        if should_activate:
            _set_team_member_active_state(
                record.team_name,
                record.teammate_name,
                True,
                default_agent_type=record.agent_type,
            )

        if record.task and not record.task.done():
            raise ValueError(
                f"Cannot resume agent '{record.agent_id}' while it is still running. "
                "Use TaskOutput to check progress/output."
            )

        hook_result = await self._run_subagent_start_hook(
            context,
            subagent_type=record.agent_type,
            prompt=input_data.prompt,
            resume=input_data.resume,
            run_in_background=False,
        )
        for notice in self._build_subagent_start_notices(hook_result, agent_type=record.agent_type):
            yield notice
        hook_context = self._build_subagent_hook_context(hook_result)

        if input_data.model:
            record.model_used = input_data.model
        if input_data.mode:
            record.permission_mode = input_data.mode
        if input_data.max_turns:
            record.max_turns = input_data.max_turns
        record.task_description = input_data.description

        self._reset_record_for_resume_prompt(record, input_data.prompt)
        subagent_context = self._build_subagent_query_context(
            tools=record.tools,
            yolo_mode=context.yolo_mode,
            verbose=context.verbose,
            model=record.model_used or "main",
            agent_type=record.agent_type,
            team_name=record.team_name,
            teammate_name=record.teammate_name,
            agent_id=record.agent_id,
            hook_scopes=record.hook_scopes,
            max_turns=record.max_turns,
            permission_mode=record.permission_mode,
            working_directory=record.worktree_path,
        )
        yield ToolProgress(
            content=f"Resuming subagent '{record.agent_type}'",
            progress_sender=self._subagent_progress_sender(record),
        )

        async for progress in self._run_subagent_foreground(
            record=record,
            subagent_context=subagent_context,
            permission_checker=context.permission_checker,
            hook_context=hook_context,
            parent_tool_use_id=context.message_id,
        ):
            yield progress

        output = self._output_from_record(record, is_resumed=True)
        yield self._render_tool_result(output)

    async def _handle_new_call(
        self,
        input_data: TaskToolInput,
        context: ToolUseContext,
    ) -> AsyncGenerator[ToolOutput, None]:
        clear_agent_cache()
        agents = load_agent_definitions()

        resolved_agent_type, resolved_team_name, resolved_teammate_name = (
            self._resolve_team_agent_target(
                team_name=input_data.team_name,
                teammate_name=input_data.teammate_name,
                fallback_agent_type=input_data.subagent_type,
            )
        )
        if not resolved_agent_type:
            raise ValueError(
                "Unable to resolve target agent type. Provide subagent_type or "
                "team_name + name."
            )
        should_spawn_teammate = bool(resolved_team_name and resolved_teammate_name)
        should_run_async = bool(input_data.run_in_background or should_spawn_teammate)
        selected_permission_mode = input_data.mode or "default"

        target_agent = next(
            (agent for agent in agents.active_agents if agent.agent_type == resolved_agent_type),
            None,
        )
        if not target_agent:
            raise ValueError(
                f"Agent type '{resolved_agent_type}' not found. "
                f"Available agents: {', '.join(agent.agent_type for agent in agents.active_agents)}"
            )
        selected_model = input_data.model or target_agent.model or "main"

        available_tools = list(self._available_tools_provider())
        agent_tools, missing_tools = resolve_agent_tools(target_agent, available_tools, self.name)
        if not agent_tools:
            raise ValueError(
                f"Agent '{target_agent.agent_type}' has no usable tools. "
                f"Missing or unknown tools: {', '.join(missing_tools) if missing_tools else 'none'}"
            )

        hook_result = await self._run_subagent_start_hook(
            context,
            subagent_type=target_agent.agent_type,
            prompt=input_data.prompt,
            resume=None,
            run_in_background=should_run_async,
        )
        for notice in self._build_subagent_start_notices(
            hook_result, agent_type=target_agent.agent_type
        ):
            yield notice
        hook_context = self._build_subagent_hook_context(hook_result)

        agent_id = _new_agent_id()
        worktree_path: Optional[str] = None
        worktree_branch: Optional[str] = None
        worktree_name: Optional[str] = None
        worktree_repo_root: Optional[str] = None
        worktree_head_commit: Optional[str] = None
        worktree_hook_based: bool = False
        if input_data.isolation == "worktree":
            base_path = (
                Path(context.working_directory).resolve()
                if context.working_directory
                else Path.cwd().resolve()
            )
            try:
                worktree_session = create_task_worktree(
                    task_id=agent_id,
                    base_path=base_path,
                    requested_name=(input_data.name or resolved_teammate_name or None),
                )
            except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
                raise ValueError(f"Failed to create worktree isolation: {exc}") from exc
            worktree_path = str(worktree_session.worktree_path)
            worktree_branch = worktree_session.branch
            worktree_name = worktree_session.name
            worktree_repo_root = str(worktree_session.repo_root)
            worktree_head_commit = worktree_session.head_commit
            worktree_hook_based = bool(worktree_session.hook_based)

        typed_agent_tools = self._coerce_agent_tools(agent_tools)
        agent_system_prompt = self._build_agent_prompt(
            target_agent,
            typed_agent_tools,
            working_directory=worktree_path,
        )
        parent_history = (
            self._coerce_parent_history(getattr(context, "conversation_messages", None))
            if target_agent.fork_context
            else []
        )
        subagent_messages = [
            *parent_history,
            create_user_message(input_data.prompt or ""),
        ]

        agent_hook_scopes: List[HooksConfig] = (
            [target_agent.hooks] if target_agent.hooks and target_agent.hooks.hooks else []
        )
        record = AgentRunRecord(
            agent_id=agent_id,
            agent_type=target_agent.agent_type,
            tools=typed_agent_tools,
            system_prompt=agent_system_prompt,
            history=subagent_messages,
            missing_tools=missing_tools,
            model_used=selected_model,
            start_time=time.time(),
            task_description=input_data.description,
            task_prompt=input_data.prompt,
            output_file=_task_output_path(agent_id),
            is_background=should_run_async,
            hook_scopes=agent_hook_scopes,
            team_name=resolved_team_name,
            teammate_name=resolved_teammate_name,
            permission_mode=selected_permission_mode,
            max_turns=input_data.max_turns,
            isolation_mode=input_data.isolation,
            worktree_path=worktree_path,
            worktree_branch=worktree_branch,
            worktree_name=worktree_name,
            worktree_repo_root=worktree_repo_root,
            worktree_head_commit=worktree_head_commit,
            worktree_hook_based=worktree_hook_based,
        )
        _write_task_output(
            record.output_file,
            (
                f"=== Task started {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
                f"Description: {input_data.description}\n"
                f"Prompt: {input_data.prompt}\n"
                f"Agent: {target_agent.agent_type}\n"
            ),
            append=False,
        )
        _register_agent_run(record)

        if resolved_team_name:
            self._send_team_event(
                record=record,
                message_type="delegate",
                content=(
                    f"Delegated work to subagent '{target_agent.agent_type}'"
                    + (f" ({resolved_teammate_name})" if resolved_teammate_name else "")
                    + "."
                ),
                metadata={
                    "agent_id": record.agent_id,
                    "run_in_background": should_run_async,
                    "mode": record.permission_mode,
                    "isolation": record.isolation_mode,
                "worktree_path": record.worktree_path,
                "worktree_branch": record.worktree_branch,
            },
        )
        _write_task_output(
            record.output_file,
            (
                f"=== Final status: {record.status} ===\n"
                f"{record.result_text or ''}\n"
                + (f"Error: {record.error}\n" if record.error else "")
            ),
            append=True,
        )

        subagent_context = self._build_subagent_query_context(
            tools=typed_agent_tools,
            yolo_mode=context.yolo_mode,
            verbose=context.verbose,
            model=selected_model,
            agent_type=target_agent.agent_type,
            team_name=resolved_team_name,
            teammate_name=resolved_teammate_name,
            agent_id=record.agent_id,
            hook_scopes=agent_hook_scopes,
            max_turns=input_data.max_turns,
            permission_mode=selected_permission_mode,
            working_directory=record.worktree_path,
        )

        if resolved_team_name and resolved_teammate_name:
            _set_team_member_active_state(
                resolved_team_name,
                resolved_teammate_name,
                True,
                default_agent_type=target_agent.agent_type,
            )

        if should_run_async:
            try:
                record.task = asyncio.create_task(
                    self._run_subagent_background(
                        record,
                        subagent_context,
                        context.permission_checker,
                        hook_context,
                    )
                )
            except Exception as exc:
                _set_team_member_active_state(
                    resolved_team_name,
                    resolved_teammate_name,
                    False,
                    default_agent_type=target_agent.agent_type,
                )
                self._finalize_record_from_messages(
                    record,
                    assistant_messages=[],
                    tool_use_count=0,
                    status="failed",
                    error=str(exc),
                    result_text="Failed to start background subagent.",
                )
                raise
            output = self._output_from_record(
                record,
                status_override="teammate_spawned" if should_spawn_teammate else "async_launched",
                result_text_override=(
                    "Teammate spawned and running asynchronously."
                    if should_spawn_teammate
                    else "Agent started in the background."
                ),
                is_background=True,
                can_read_output_file=any(
                    getattr(tool, "name", "") in {READ_TOOL_NAME, BASH_TOOL_NAME}
                    for tool in self._available_tools_provider()
                ),
            )
            yield self._render_tool_result(output)
            return

        yield ToolProgress(
            content=f"Launching subagent '{target_agent.agent_type}'",
            progress_sender=self._subagent_progress_sender(record),
        )
        async for progress in self._run_subagent_foreground(
            record=record,
            subagent_context=subagent_context,
            permission_checker=context.permission_checker,
            hook_context=hook_context,
            parent_tool_use_id=context.message_id,
        ):
            yield progress

        output = self._output_from_record(record)
        yield self._render_tool_result(output)

    async def call(
        self,
        input_data: TaskToolInput,
        context: ToolUseContext,
    ) -> AsyncGenerator[ToolOutput, None]:
        if input_data.resume:
            async for output in self._handle_resume_call(input_data, context):
                yield output
            return

        async for output in self._handle_new_call(input_data, context):
            yield output

    def _output_from_record(
        self,
        record: AgentRunRecord,
        *,
        status_override: Optional[str] = None,
        result_text_override: Optional[str] = None,
        is_background: bool = False,
        is_resumed: bool = False,
        error_override: Optional[str] = None,
        can_read_output_file: Optional[bool] = None,
    ) -> TaskToolOutput:
        status = status_override or record.status
        duration_ms = (
            record.duration_ms
            if record.duration_ms
            else max((time.time() - record.start_time) * 1000, 0.0)
        )
        result_text = (
            result_text_override
            or record.result_text
            or ("Agent is still running." if status == "running" else "Agent returned no response.")
        )
        content = (
            [TaskToolContentBlock(text=result_text)]
            if result_text
            else []
        )
        return TaskToolOutput(
            status=status,
            agent_id=record.agent_id,
            description=record.task_description,
            prompt=record.task_prompt,
            output_file=record.output_file,
            can_read_output_file=can_read_output_file,
            content=content,
            total_tool_use_count=record.tool_use_count,
            total_duration_ms=duration_ms,
            total_tokens=record.total_tokens,
            usage=record.usage,
            error=error_override or record.error,
            agent_type=record.agent_type,
            missing_tools=record.missing_tools,
            model_used=record.model_used,
            is_background=is_background,
            is_resumed=is_resumed,
            isolation=record.isolation_mode,
            worktree_path=record.worktree_path,
            worktree_branch=record.worktree_branch,
            result_text=result_text,
            duration_ms=duration_ms,
            tool_use_count=record.tool_use_count,
        )

    def _coerce_parent_history(self, messages: Optional[Sequence[object]]) -> List[MessageType]:
        if not messages:
            return []
        history: List[MessageType] = []
        for msg in messages:
            msg_type = getattr(msg, "type", None)
            if msg_type in ("user", "assistant"):
                history.append(msg)  # type: ignore[arg-type]
        return history

    def _track_subagent_message(
        self,
        record: AgentRunRecord,
        message: object,
        history: List[MessageType],
        assistant_messages: List[AssistantMessage],
        tool_use_count: int,
    ) -> tuple[int, List[tuple[str, str]]]:
        updates: List[tuple[str, str]] = []
        msg_type = getattr(message, "type", "")
        if msg_type in ("assistant", "user"):
            history.append(message)  # type: ignore[arg-type]

        if msg_type == "assistant":
            if isinstance(message, AssistantMessage):
                tool_use_count += self._count_tool_uses(message)
                text = self._extract_text(message).strip()
                if text:
                    _write_task_output(record.output_file, text, append=True)
            updates = self._extract_progress_updates(message, record=record)
            assistant_messages.append(message)  # type: ignore[arg-type]

        return tool_use_count, updates

    @staticmethod
    def _subagent_progress_label(record: AgentRunRecord) -> str:
        base = (record.teammate_name or record.agent_type or "subagent").strip() or "subagent"
        agent_id = (record.agent_id or "").strip()
        return f"{base}:{agent_id}" if agent_id else base

    @classmethod
    def _subagent_progress_sender(cls, record: AgentRunRecord) -> str:
        return f"Subagent({cls._subagent_progress_label(record)})"

    def _extract_progress_updates(
        self, message: object, *, record: AgentRunRecord
    ) -> List[tuple[str, str]]:
        msg_content = getattr(message, "message", None)
        blocks = getattr(msg_content, "content", []) if msg_content else []
        if not isinstance(blocks, list):
            return []

        sender = self._subagent_progress_sender(record)
        updates: List[tuple[str, str]] = []
        for block in blocks:
            block_type = TaskTool._get_block_attr(block, "type") or ""
            if block_type == "tool_use":
                tool_name = TaskTool._get_block_attr(block, "name") or "unknown tool"
                block_input = TaskTool._get_block_attr(block, "input")
                summary = self._summarize_tool_input(block_input)
                label = f"requesting {tool_name}"
                if summary:
                    label += f" — {summary}"
                updates.append((sender, label))
            elif block_type == "text":
                text_val = TaskTool._get_block_attr(block, "text") or ""
                if text_val:
                    snippet = str(text_val).strip()
                    if snippet:
                        short = snippet if len(snippet) <= 200 else snippet[:197] + "..."
                        updates.append((sender, short))
        return updates

    async def _run_subagent_background(
        self,
        record: AgentRunRecord,
        subagent_context: QueryContext,
        permission_checker: Any,
        hook_context: Optional[Dict[str, str]] = None,
    ) -> None:
        assistant_messages: List[AssistantMessage] = []
        tool_use_count = 0
        finalize_status = "running"
        finalize_error: Optional[str] = None
        finalize_result_text: Optional[str] = None
        try:
            async for message in query(
                record.history,  # type: ignore[arg-type]
                record.system_prompt,
                hook_context or {},
                subagent_context,
                permission_checker,
            ):
                if getattr(message, "type", "") == "progress":
                    continue

                tool_use_count, _ = self._track_subagent_message(
                    record,
                    message,
                    record.history,
                    assistant_messages,
                    tool_use_count,
                )
                if isinstance(message, UserMessage):
                    shutdown_approval = self._extract_approved_shutdown_response(
                        record.history,
                        message,
                    )
                    if shutdown_approval is not None:
                        finalize_status = "shutdown"
                        finalize_error = (
                            shutdown_approval.get("content")
                            or "Approved shutdown_response sent to team lead."
                        ).strip()
                        finalize_result_text = (
                            "Subagent exited after approved shutdown_response"
                            + (
                                f" (request_id={shutdown_approval.get('request_id', '')})."
                                if shutdown_approval.get("request_id")
                                else "."
                            )
                        )
                        break
        except asyncio.CancelledError:
            finalize_status = "cancelled"
            finalize_error = "Subagent run was cancelled."
            raise
        except Exception as exc:
            finalize_status = "failed"
            finalize_error = str(exc)
            logger.warning(
                "[task_tool] Subagent background run failed: %s: %s",
                type(exc).__name__,
                exc,
                extra={"agent_id": record.agent_id, "team_name": record.team_name},
            )
        finally:
            if finalize_status == "running":
                finalize_status = "completed"
            self._finalize_record_from_messages(
                record,
                assistant_messages=assistant_messages,
                tool_use_count=tool_use_count,
                status=finalize_status,
                error=finalize_error,
                result_text=finalize_result_text,
            )
            record.task = None

    def _build_agent_prompt(
        self,
        agent: AgentDefinition,
        tools: List[Tool[Any, Any]],
        *,
        working_directory: Optional[str] = None,
    ) -> str:
        tool_names = ", ".join(tool.name for tool in tools if getattr(tool, "name", None))
        env_cwd = Path(working_directory) if working_directory else None
        guidance = (
            "You are a specialized Ripperdoc subagent working autonomously. "
            "Execute the task completely using the allowed tools. "
            "Return a single, concise summary for the parent agent that includes what you did, "
            "important findings, and any follow-ups. Do not ask the user questions."
        )
        sections = [
            guidance,
            f"Agent type: {agent.agent_type}",
            f"When to use: {agent.when_to_use}",
            f"Allowed tools: {tool_names}",
            "Agent system prompt:",
            agent.system_prompt or "(no additional prompt)",
            build_environment_prompt(cwd=env_cwd),
        ]
        return "\n\n".join(sections)

    def _extract_text(self, message: AssistantMessage) -> str:
        content = message.message.content
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""

        parts = []
        for block in content:
            text = TaskTool._get_block_attr(block, "text")
            if text:
                parts.append(str(text))
        return "\n".join(parts)

    def _count_tool_uses(self, message: AssistantMessage) -> int:
        content = message.message.content
        if not isinstance(content, list):
            return 0
        return sum(1 for block in content if TaskTool._get_block_attr(block, "type") == "tool_use")

    def _summarize_tool_input(self, inp: Any) -> str:
        """Generate a short human-readable summary of a tool_use input."""
        if not inp or not isinstance(inp, (dict, Dict)):
            return ""

        pieces: List[str] = []
        # Prioritize common keys
        for key in ("command", "file_path", "path", "glob", "pattern", "description", "prompt"):
            if key in inp and inp[key]:
                val = str(inp[key])
                short = val if len(val) <= 80 else val[:77] + "..."
                pieces.append(f"{key}={short}")

        # Include range info if present
        start = inp.get("start_line") or inp.get("offset")
        end = inp.get("end_line") or inp.get("limit")
        if start is not None or end is not None:
            pieces.append(f"range={start or 0}-{end or '…'}")

        if not pieces:
            # Fallback to truncated dict representation
            try:
                serialized = json.dumps(inp, ensure_ascii=False)
            except (TypeError, ValueError) as exc:
                logger.warning(
                    "[task_tool] Failed to serialize tool_use input: %s: %s",
                    type(exc).__name__,
                    exc,
                    extra={"tool_use_input": str(inp)[:200]},
                )
                serialized = str(inp)
            return serialized if len(serialized) <= 120 else serialized[:117] + "..."

        return ", ".join(pieces)
