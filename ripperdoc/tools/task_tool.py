"""Task tool that delegates work to configured subagents."""

from __future__ import annotations

import asyncio
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Callable, Dict, Iterable, List, Optional, Sequence
from uuid import uuid4

from pydantic import BaseModel, Field

from ripperdoc.core.agents import (
    AgentDefinition,
    AgentLoadResult,
    FILE_EDIT_TOOL_NAME,
    GREP_TOOL_NAME,
    VIEW_TOOL_NAME,
    clear_agent_cache,
    load_agent_definitions,
    resolve_agent_tools,
    summarize_agent,
)
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
from ripperdoc.utils.messages import AssistantMessage, UserMessage, create_user_message
from ripperdoc.utils.log import get_logger

logger = get_logger()


MessageType = UserMessage | AssistantMessage


@dataclass
class AgentRunRecord:
    """In-memory record for a subagent run (foreground or background)."""

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
    status: str = "running"
    result_text: Optional[str] = None
    error: Optional[str] = None
    task: Optional[asyncio.Task] = None
    is_background: bool = False


_AGENT_RUNS: Dict[str, AgentRunRecord] = {}
_AGENT_RUNS_LOCK = threading.Lock()
DEFAULT_AGENT_RUN_TTL_SEC = float(os.getenv("RIPPERDOC_AGENT_RUN_TTL_SEC", "3600"))


def _new_agent_id() -> str:
    return f"agent_{uuid4().hex[:8]}"


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
        "missing_tools": list(record.missing_tools),
        "model_used": record.model_used,
        "result_text": record.result_text,
        "error": record.error,
        "is_background": record.is_background,
    }


def list_agent_runs() -> List[str]:
    """Return known subagent run ids."""
    prune_agent_runs()
    with _AGENT_RUNS_LOCK:
        return list(_AGENT_RUNS.keys())


def get_agent_run_snapshot(agent_id: str) -> Optional[dict]:
    """Return a snapshot of a subagent run by id."""
    record = _get_agent_run(agent_id)
    if not record:
        return None
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
    record.error = record.error or "Cancelled by user."
    record.duration_ms = (time.time() - record.start_time) * 1000
    record.task = None
    return True

class TaskToolInput(BaseModel):
    """Input schema for delegating to a subagent."""

    prompt: Optional[str] = Field(
        default=None,
        description="Detailed task description for the subagent to perform.",
    )
    subagent_type: Optional[str] = Field(
        default=None,
        description="Agent type to run (matches agent frontmatter name). Required for new runs.",
    )
    run_in_background: bool = Field(
        default=False,
        description="If true, start the agent in the background and return immediately.",
    )
    resume: Optional[str] = Field(
        default=None,
        description="Agent id to resume or fetch results for a background run.",
    )
    wait: bool = Field(
        default=True,
        description="When resuming a background agent, wait for completion before returning.",
    )


class TaskToolOutput(BaseModel):
    """Summary of a completed subagent run."""

    agent_id: Optional[str] = None
    agent_type: str
    result_text: str
    duration_ms: float
    tool_use_count: int
    missing_tools: List[str] = Field(default_factory=list)
    model_used: Optional[str] = None
    status: str = "completed"
    is_background: bool = False
    is_resumed: bool = False
    error: Optional[str] = None


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
        file_read_tool_name = VIEW_TOOL_NAME
        search_tool_name = GREP_TOOL_NAME
        code_tool_name = FILE_EDIT_TOOL_NAME
        background_fetch_tool_name = task_tool_name

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
            f"- Fetch background results by calling {background_fetch_tool_name} with resume=<agent_id>. If the agent is still running, set wait=true to block or wait=false to get status only.\n"
            "- To continue a completed agent, call Task with resume=<agent_id> and a new prompt.\n"
            "- Provide clear, detailed prompts so the agent can work autonomously and return exactly the information you need.\n"
            '- Agents can opt into parent context by setting fork_context: true in their frontmatter. When enabled, they receive the full conversation history before the tool call.\n'
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
        del context
        if input_data.resume and input_data.run_in_background:
            return ValidationResult(
                result=False,
                message="run_in_background cannot be used when resuming an agent.",
            )
        if input_data.resume:
            if input_data.prompt is not None and not input_data.prompt.strip():
                return ValidationResult(
                    result=False,
                    message="prompt cannot be empty when resuming with new work.",
                )
            return ValidationResult(result=True)

        if not input_data.subagent_type:
            return ValidationResult(
                result=False,
                message="subagent_type is required when starting a new agent.",
            )
        if not input_data.prompt or not input_data.prompt.strip():
            return ValidationResult(
                result=False,
                message="prompt is required when starting a new agent.",
            )
        return ValidationResult(result=True)

    def render_result_for_assistant(self, output: TaskToolOutput) -> str:
        details: List[str] = []
        if output.agent_id:
            details.append(f"id={output.agent_id}")
        if output.status and output.status != "completed":
            details.append(output.status)
        if output.tool_use_count:
            details.append(f"{output.tool_use_count} tool uses")
        details.append(f"{output.duration_ms / 1000:.1f}s")
        if output.missing_tools:
            details.append(f"missing tools: {', '.join(output.missing_tools)}")
        if output.error:
            details.append(f"error: {output.error}")

        suffix = f" ({'; '.join(details)})" if details else ""
        return f"[subagent:{output.agent_type}] {output.result_text}{suffix}"

    def render_tool_use_message(self, input_data: TaskToolInput, verbose: bool = False) -> str:
        del verbose
        if input_data.resume:
            return f"Resume subagent {input_data.resume}"
        label = f"Task via {input_data.subagent_type}: {input_data.prompt}"
        if input_data.run_in_background:
            label += " (background)"
        return label

    async def call(
        self,
        input_data: TaskToolInput,
        context: ToolUseContext,
    ) -> AsyncGenerator[ToolOutput, None]:
        clear_agent_cache()
        agents = load_agent_definitions()

        if input_data.resume:
            record = _get_agent_run(input_data.resume)
            if not record:
                raise ValueError(
                    f"Agent id '{input_data.resume}' not found. "
                    "Start a new agent to obtain a valid agent_id."
                )

            if record.task and not record.task.done():
                if not input_data.wait:
                    output = self._output_from_record(
                        record,
                        status_override="running",
                        result_text_override="Agent is still running in the background.",
                        is_background=True,
                        is_resumed=True,
                    )
                    yield ToolResult(
                        data=output, result_for_assistant=self.render_result_for_assistant(output)
                    )
                    return

                yield ToolProgress(
                    content=f"Waiting for subagent '{record.agent_type}' ({record.agent_id})"
                )
                try:
                    await record.task
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    record.status = "failed"
                    record.error = str(exc)

            if not input_data.prompt:
                output = self._output_from_record(
                    record,
                    is_background=bool(record.task),
                    is_resumed=True,
                )
                yield ToolResult(
                    data=output, result_for_assistant=self.render_result_for_assistant(output)
                )
                return

            record.history.append(create_user_message(input_data.prompt))
            record.start_time = time.time()
            record.duration_ms = 0.0
            record.tool_use_count = 0
            record.status = "running"
            record.result_text = None
            record.error = None
            record.task = None

            subagent_context = QueryContext(
                tools=record.tools,
                yolo_mode=context.yolo_mode,
                verbose=context.verbose,
                model=record.model_used or "main",
                stop_hook="subagent",
            )

            yield ToolProgress(content=f"Resuming subagent '{record.agent_type}'")

            assistant_messages: List[AssistantMessage] = []
            tool_use_count = 0
            async for message in query(
                record.history,  # type: ignore[arg-type]
                record.system_prompt,
                {},
                subagent_context,
                context.permission_checker,
            ):
                if getattr(message, "type", "") == "progress":
                    continue
                tool_use_count, updates = self._track_subagent_message(
                    message,
                    record.history,
                    assistant_messages,
                    tool_use_count,
                )
                for update in updates:
                    yield ToolProgress(content=update)

            duration_ms = (time.time() - record.start_time) * 1000
            result_text = (
                self._extract_text(assistant_messages[-1])
                if assistant_messages
                else "Agent returned no response."
            )
            record.duration_ms = duration_ms
            record.tool_use_count = tool_use_count
            record.result_text = result_text.strip()
            record.status = "completed"

            output = self._output_from_record(
                record,
                result_text_override=result_text.strip(),
                is_resumed=True,
            )
            yield ToolResult(
                data=output, result_for_assistant=self.render_result_for_assistant(output)
            )
            return

        target_agent = next(
            (
                agent
                for agent in agents.active_agents
                if agent.agent_type == input_data.subagent_type
            ),
            None,
        )
        if not target_agent:
            raise ValueError(
                f"Agent type '{input_data.subagent_type}' not found. "
                f"Available agents: {', '.join(agent.agent_type for agent in agents.active_agents)}"
            )

        available_tools = list(self._available_tools_provider())
        agent_tools, missing_tools = resolve_agent_tools(target_agent, available_tools, self.name)
        if not agent_tools:
            raise ValueError(
                f"Agent '{target_agent.agent_type}' has no usable tools. "
                f"Missing or unknown tools: {', '.join(missing_tools) if missing_tools else 'none'}"
            )

        # Type conversion: List[object] -> List[Tool[Any, Any]]
        from ripperdoc.core.tool import Tool

        typed_agent_tools: List[Tool[Any, Any]] = [
            tool for tool in agent_tools if isinstance(tool, Tool)
        ]

        agent_system_prompt = self._build_agent_prompt(target_agent, typed_agent_tools)
        parent_history = (
            self._coerce_parent_history(getattr(context, "conversation_messages", None))
            if target_agent.fork_context
            else []
        )
        subagent_messages = [
            *parent_history,
            create_user_message(input_data.prompt or ""),
        ]

        agent_id = _new_agent_id()
        record = AgentRunRecord(
            agent_id=agent_id,
            agent_type=target_agent.agent_type,
            tools=typed_agent_tools,
            system_prompt=agent_system_prompt,
            history=subagent_messages,
            missing_tools=missing_tools,
            model_used=target_agent.model or "main",
            start_time=time.time(),
            is_background=bool(input_data.run_in_background),
        )
        _register_agent_run(record)

        subagent_context = QueryContext(
            tools=typed_agent_tools,
            yolo_mode=context.yolo_mode,
            verbose=context.verbose,
            model=target_agent.model or "main",
            stop_hook="subagent",
        )

        if input_data.run_in_background:
            record.task = asyncio.create_task(
                self._run_subagent_background(
                    record,
                    subagent_context,
                    context.permission_checker,
                )
            )
            output = self._output_from_record(
                record,
                status_override="running",
                result_text_override="Agent started in the background.",
                is_background=True,
            )
            yield ToolResult(
                data=output, result_for_assistant=self.render_result_for_assistant(output)
            )
            return

        yield ToolProgress(content=f"Launching subagent '{target_agent.agent_type}'")

        assistant_messages = []
        tool_use_count = 0
        async for message in query(
            record.history,  # type: ignore[arg-type]
            agent_system_prompt,
            {},
            subagent_context,
            context.permission_checker,
        ):
            if getattr(message, "type", "") == "progress":
                continue
            tool_use_count, updates = self._track_subagent_message(
                message,
                record.history,
                assistant_messages,
                tool_use_count,
            )
            for update in updates:
                yield ToolProgress(content=update)

        duration_ms = (time.time() - record.start_time) * 1000
        result_text = (
            self._extract_text(assistant_messages[-1])
            if assistant_messages
            else "Agent returned no response."
        )

        record.duration_ms = duration_ms
        record.tool_use_count = tool_use_count
        record.result_text = result_text.strip()
        record.status = "completed"

        output = self._output_from_record(record, result_text_override=result_text.strip())

        yield ToolResult(data=output, result_for_assistant=self.render_result_for_assistant(output))

    def _output_from_record(
        self,
        record: AgentRunRecord,
        *,
        status_override: Optional[str] = None,
        result_text_override: Optional[str] = None,
        is_background: bool = False,
        is_resumed: bool = False,
        error_override: Optional[str] = None,
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
        return TaskToolOutput(
            agent_id=record.agent_id,
            agent_type=record.agent_type,
            result_text=result_text,
            duration_ms=duration_ms,
            tool_use_count=record.tool_use_count,
            missing_tools=record.missing_tools,
            model_used=record.model_used,
            status=status,
            is_background=is_background,
            is_resumed=is_resumed,
            error=error_override or record.error,
        )

    def _coerce_parent_history(
        self, messages: Optional[Sequence[object]]
    ) -> List[MessageType]:
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
        message: object,
        history: List[MessageType],
        assistant_messages: List[AssistantMessage],
        tool_use_count: int,
    ) -> tuple[int, List[str]]:
        updates: List[str] = []
        msg_type = getattr(message, "type", "")
        if msg_type in ("assistant", "user"):
            history.append(message)  # type: ignore[arg-type]

        if msg_type == "assistant":
            if isinstance(message, AssistantMessage):
                tool_use_count += self._count_tool_uses(message)
            updates = self._extract_progress_updates(message)
            assistant_messages.append(message)  # type: ignore[arg-type]

        return tool_use_count, updates

    def _extract_progress_updates(self, message: object) -> List[str]:
        msg_content = getattr(message, "message", None)
        blocks = getattr(msg_content, "content", []) if msg_content else []
        if not isinstance(blocks, list):
            return []

        updates: List[str] = []
        for block in blocks:
            block_type = getattr(block, "type", None) or (
                block.get("type") if isinstance(block, Dict) else None
            )
            if block_type == "tool_use":
                tool_name = getattr(block, "name", None) or (
                    block.get("name") if isinstance(block, Dict) else "unknown tool"
                )
                block_input = (
                    getattr(block, "input", None)
                    if hasattr(block, "input")
                    else (block.get("input") if isinstance(block, Dict) else None)
                )
                summary = self._summarize_tool_input(block_input)
                label = f"Subagent requesting {tool_name}"
                if summary:
                    label += f" — {summary}"
                updates.append(label)
            if block_type == "text":
                text_val = getattr(block, "text", None) or (
                    block.get("text") if isinstance(block, Dict) else ""
                )
                if text_val:
                    snippet = str(text_val).strip()
                    if snippet:
                        short = snippet if len(snippet) <= 200 else snippet[:197] + "..."
                        updates.append(f"Subagent: {short}")
        return updates

    async def _run_subagent_background(
        self,
        record: AgentRunRecord,
        subagent_context: QueryContext,
        permission_checker: Any,
    ) -> None:
        assistant_messages: List[AssistantMessage] = []
        tool_use_count = 0
        try:
            async for message in query(
                record.history,  # type: ignore[arg-type]
                record.system_prompt,
                {},
                subagent_context,
                permission_checker,
            ):
                if getattr(message, "type", "") == "progress":
                    continue
                tool_use_count, _ = self._track_subagent_message(
                    message,
                    record.history,
                    assistant_messages,
                    tool_use_count,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            record.status = "failed"
            record.error = str(exc)
        finally:
            record.duration_ms = (time.time() - record.start_time) * 1000
            record.tool_use_count = tool_use_count
            if record.status != "failed":
                result_text = (
                    self._extract_text(assistant_messages[-1])
                    if assistant_messages
                    else "Agent returned no response."
                )
                record.result_text = result_text.strip()
                record.status = "completed"
            record.task = None

    def _build_agent_prompt(self, agent: AgentDefinition, tools: List[Tool[Any, Any]]) -> str:
        tool_names = ", ".join(tool.name for tool in tools if getattr(tool, "name", None))
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
            build_environment_prompt(),
        ]
        return "\n\n".join(sections)

    def _extract_text(self, message: AssistantMessage) -> str:
        content = message.message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for block in content:
                text = getattr(block, "text", None) or (
                    block.get("text") if isinstance(block, Dict) else None
                )
                if text:
                    parts.append(str(text))
            return "\n".join(parts)
        return ""

    def _count_tool_uses(self, message: AssistantMessage) -> int:
        content = message.message.content
        if not isinstance(content, list):
            return 0
        count = 0
        for block in content:
            block_type = getattr(block, "type", None) or (
                block.get("type") if isinstance(block, Dict) else None
            )
            if block_type == "tool_use":
                count += 1
        return count

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
            import json

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
