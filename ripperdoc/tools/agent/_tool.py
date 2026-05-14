"""Agent tool class — schemas and core routing logic."""

from __future__ import annotations

import asyncio
import subprocess
import time
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
)

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ripperdoc.core.agents import (
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
from ripperdoc.core.tool import (
    Tool,
    ToolOutput,
    ToolProgress,
    ToolResult,
    ToolUseContext,
    ValidationResult,
)
from ripperdoc.utils.log import get_logger
from ripperdoc.utils.messaging.messages import (
    AttachmentMessage,
    UserMessage,
    create_hook_additional_context_message,
    create_user_message,
)

from ripperdoc.tools.agent._store import (
    AgentRunRecord,
    _get_agent_run,
    _new_agent_id,
    _register_agent_run,
    _set_team_member_active_state,
    _task_output_path,
    _write_task_output,
)

logger = get_logger()


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class AgentToolInput(BaseModel):
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
    def _normalize_teammate_aliases(self) -> "AgentToolInput":
        self.description = (self.description or "").strip()
        if self.name and self.teammate_name and self.name != self.teammate_name:
            raise ValueError("name and teammate_name must match when both are provided.")
        if self.name and not self.teammate_name:
            self.teammate_name = self.name
        elif self.teammate_name and not self.name:
            self.name = self.teammate_name
        return self


class AgentToolContentBlock(BaseModel):
    """Text content block returned by Agent tool."""

    type: Literal["text"] = "text"
    text: str


class AgentToolOutput(BaseModel):
    """Agent tool output payload."""

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
    content: List[AgentToolContentBlock] = Field(default_factory=list)
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


# ---------------------------------------------------------------------------
# TaskTool class
# ---------------------------------------------------------------------------


class AgentTool(Tool[AgentToolInput, AgentToolOutput]):
    """Launches a configured agent in a fresh context."""

    def __init__(self, available_tools_provider: Callable[[], Iterable[Tool[Any, Any]]]) -> None:
        super().__init__()
        self._available_tools_provider = available_tools_provider

    @property
    def name(self) -> str:
        return "Agent"

    async def description(self) -> str:
        from ripperdoc.tools.agent._built_in import _built_in_agents

        agents = _built_in_agents()
        agent_lines = "\n".join(summarize_agent(agent) for agent in agents)
        return (
            "Launch a specialized subagent in its own context window to handle a task.\n"
            f"Available agents:\n{agent_lines or '- general-purpose (built-in)'}"
        )

    @property
    def input_schema(self) -> type[AgentToolInput]:
        return AgentToolInput

    async def prompt(self, yolo_mode: bool = False) -> str:
        from ripperdoc.core.agents import clear_agent_cache, load_agent_definitions, READ_TOOL_NAME, GREP_TOOL_NAME, FILE_EDIT_TOOL_NAME
        from ripperdoc.tools.agent._prompt import build_task_tool_prompt, build_agent_listing

        del yolo_mode
        clear_agent_cache()
        agents = load_agent_definitions()
        agent_block = build_agent_listing(agents.active_agents)
        return build_task_tool_prompt(
            task_tool_name=self.name,
            file_read_tool_name=READ_TOOL_NAME,
            search_tool_name=GREP_TOOL_NAME,
            code_tool_name=FILE_EDIT_TOOL_NAME,
            background_fetch_tool_name="TaskOutput",
            agent_block=agent_block,
        )

    def is_read_only(self) -> bool:
        return True

    def is_concurrency_safe(self) -> bool:
        return True

    async def validate_input(
        self, input_data: AgentToolInput, context: Optional[ToolUseContext] = None
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

    def render_result_for_assistant(self, output: AgentToolOutput) -> str:
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

    def render_tool_use_message(self, input_data: AgentToolInput, verbose: bool = False) -> str:
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

    # ------------------------------------------------------------------
    # Hook helpers
    # ------------------------------------------------------------------

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

    def _build_subagent_hook_messages(
        self,
        result: HookResult,
        *,
        parent_tool_use_id: Optional[str] = None,
    ) -> List[UserMessage | AttachmentMessage]:
        messages: List[UserMessage | AttachmentMessage] = []
        if result.additional_context:
            additional_context_message = create_hook_additional_context_message(
                str(result.additional_context),
                hook_name="SubagentStart",
                hook_event="SubagentStart",
                parent_tool_use_id=parent_tool_use_id,
            )
            if additional_context_message:
                messages.append(additional_context_message)
        return messages

    # ------------------------------------------------------------------
    # Prompt builder delegation
    # ------------------------------------------------------------------

    def _build_agent_prompt(self, agent, tools, *, working_directory=None):
        from ripperdoc.tools.agent._prompt import build_agent_prompt

        return build_agent_prompt(agent.agent_type, tools, working_directory, agent.system_prompt)

    # ------------------------------------------------------------------
    # Tool coercion helpers
    # ------------------------------------------------------------------

    def _coerce_agent_tools(self, tools: List[object]) -> List[Tool[Any, Any]]:
        from ripperdoc.core.tool import Tool as ToolBase

        return [tool for tool in tools if isinstance(tool, ToolBase)]

    def _coerce_parent_history(self, messages: Optional[Sequence[object]]) -> List[Any]:
        from ripperdoc.tools.agent._agent_utils import coerce_parent_history

        return coerce_parent_history(messages)

    # ------------------------------------------------------------------
    # Team resolution
    # ------------------------------------------------------------------

    def _resolve_team_agent_target(
        self,
        *,
        team_name: Optional[str],
        teammate_name: Optional[str],
        fallback_agent_type: Optional[str],
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """Resolve agent type from team roster when team context is provided."""
        from ripperdoc.utils.collaboration.teams import get_team, upsert_team_member, TeamMember

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

    # ------------------------------------------------------------------
    # Render helpers
    # ------------------------------------------------------------------

    def _render_tool_result(self, output: AgentToolOutput) -> ToolResult:
        return ToolResult(
            data=output, result_for_assistant=self.render_result_for_assistant(output)
        )

    # ------------------------------------------------------------------
    # Progress sender
    # ------------------------------------------------------------------

    @staticmethod
    def _subagent_progress_label(record: AgentRunRecord) -> str:
        base = (record.teammate_name or record.agent_type or "subagent").strip() or "subagent"
        agent_id = (record.agent_id or "").strip()
        return f"{base}:{agent_id}" if agent_id else base

    @classmethod
    def _subagent_progress_sender(cls, record: AgentRunRecord) -> str:
        return f"Subagent({cls._subagent_progress_label(record)})"

    # ------------------------------------------------------------------
    # Output construction
    # ------------------------------------------------------------------

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
    ) -> AgentToolOutput:
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
            [AgentToolContentBlock(text=result_text)]
            if result_text
            else []
        )
        return AgentToolOutput(
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

    # ------------------------------------------------------------------
    # Main call router
    # ------------------------------------------------------------------

    async def call(
        self,
        input_data: AgentToolInput,
        context: ToolUseContext,
    ) -> AsyncGenerator[ToolOutput, None]:
        if input_data.resume:
            async for output in self._handle_resume_call(input_data, context):
                yield output
            return

        async for output in self._handle_new_call(input_data, context):
            yield output

    # ------------------------------------------------------------------
    # Resume call — delegates to _resume_agent module
    # ------------------------------------------------------------------

    async def _handle_resume_call(
        self,
        input_data: AgentToolInput,
        context: ToolUseContext,
    ) -> AsyncGenerator[ToolOutput, None]:
        from ripperdoc.tools.agent._resume_agent import handle_resume_call
        from ripperdoc.tools.agent._run_agent import (
            build_subagent_start_notices,
            build_subagent_query_context,
            run_subagent_foreground,
            reset_record_for_resume_prompt,
            subagent_progress_sender,
        )

        async for output in handle_resume_call(
            input_data,
            context,
            run_subagent_start_hook=self._run_subagent_start_hook,
            build_subagent_start_notices=build_subagent_start_notices,
            build_subagent_hook_messages=self._build_subagent_hook_messages,
            reset_record_for_resume_prompt=reset_record_for_resume_prompt,
            build_subagent_query_context=build_subagent_query_context,
            run_subagent_foreground=run_subagent_foreground,
            output_from_record=self._output_from_record,
            render_tool_result=self._render_tool_result,
            subagent_progress_sender=subagent_progress_sender,
            available_tools_provider=self._available_tools_provider,
        ):
            yield output

    # ------------------------------------------------------------------
    # New call — the big orchestrator
    # ------------------------------------------------------------------

    async def _handle_new_call(
        self,
        input_data: AgentToolInput,
        context: ToolUseContext,
    ) -> AsyncGenerator[ToolOutput, None]:
        from ripperdoc.tools.agent._run_agent import (
            build_subagent_start_notices,
            build_subagent_query_context,
            run_subagent_foreground,
            run_subagent_background,
            send_team_event,
            subagent_progress_sender,
        )
        from ripperdoc.tools.agent._agent_utils import coerce_agent_tools, coerce_parent_history
        from ripperdoc.utils.collaboration.worktree import create_task_worktree

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
        for notice in build_subagent_start_notices(
            hook_result, agent_type=target_agent.agent_type
        ):
            yield notice
        hook_messages = self._build_subagent_hook_messages(
            hook_result, parent_tool_use_id=context.message_id
        )

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

        typed_agent_tools = coerce_agent_tools(agent_tools)
        agent_system_prompt = self._build_agent_prompt(
            target_agent,
            typed_agent_tools,
            working_directory=worktree_path,
        )
        parent_history = (
            coerce_parent_history(getattr(context, "conversation_messages", None))
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
        if hook_messages:
            record.history.extend(hook_messages)
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
            send_team_event(
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

        subagent_context = build_subagent_query_context(
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
            task_notification_queue=context.task_notification_queue,
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
                    run_subagent_background(
                        record,
                        subagent_context,
                        context.permission_checker,
                        notification_queue=context.task_notification_queue,
                        parent_tool_use_id=context.message_id,
                    )
                )
            except Exception as exc:
                _set_team_member_active_state(
                    resolved_team_name,
                    resolved_teammate_name,
                    False,
                    default_agent_type=target_agent.agent_type,
                )
                from ripperdoc.tools.agent._run_agent import finalize_record_from_messages

                finalize_record_from_messages(
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
            progress_sender=subagent_progress_sender(record),
        )
        async for progress in run_subagent_foreground(
            record=record,
            subagent_context=subagent_context,
            permission_checker=context.permission_checker,
            parent_abort_signal=context.abort_signal,
            notification_queue=context.task_notification_queue,
            parent_tool_use_id=context.message_id,
        ):
            yield progress

        if record.task and not record.task.done():
            output = self._output_from_record(
                record,
                status_override="async_launched",
                result_text_override="Subagent moved to background after interrupt.",
                is_background=True,
                can_read_output_file=any(
                    getattr(tool, "name", "") in {READ_TOOL_NAME, BASH_TOOL_NAME}
                    for tool in self._available_tools_provider()
                ),
            )
            yield self._render_tool_result(output)
            return

        output = self._output_from_record(record)
        yield self._render_tool_result(output)


TaskToolOutput = AgentToolOutput
