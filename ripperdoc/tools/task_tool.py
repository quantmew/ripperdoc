"""Task tool that delegates work to configured subagents."""

from __future__ import annotations

import time
from typing import Any, AsyncGenerator, Callable, Dict, Iterable, List, Optional

from pydantic import BaseModel, Field

from ripperdoc.core.agents import (
    AgentDefinition,
    AgentLoadResult,
    clear_agent_cache,
    load_agent_definitions,
    resolve_agent_tools,
    summarize_agent,
)
from ripperdoc.core.query import QueryContext, query
from ripperdoc.core.system_prompt import build_environment_prompt
from ripperdoc.core.tool import Tool, ToolOutput, ToolProgress, ToolResult, ToolUseContext
from ripperdoc.utils.messages import AssistantMessage, create_user_message
from ripperdoc.utils.log import get_logger

logger = get_logger()


class TaskToolInput(BaseModel):
    """Input schema for delegating to a subagent."""

    prompt: str = Field(description="Detailed task description for the subagent to perform")
    subagent_type: str = Field(description="Agent type to run (matches agent frontmatter name)")


class TaskToolOutput(BaseModel):
    """Summary of a completed subagent run."""

    agent_type: str
    result_text: str
    duration_ms: float
    tool_use_count: int
    missing_tools: List[str] = Field(default_factory=list)
    model_used: Optional[str] = None


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

    async def prompt(self, safe_mode: bool = False) -> str:
        del safe_mode
        clear_agent_cache()
        agents: AgentLoadResult = load_agent_definitions()
        agent_lines = "\n".join(summarize_agent(agent) for agent in agents.active_agents)
        return (
            "Use this tool to delegate a well-scoped task to a subagent. "
            "Always set subagent_type to one of the available agent types below. "
            "Provide a detailed prompt so the agent can work autonomously and return a single, concise report.\n\n"
            f"Available agents:\n{agent_lines or '- general-purpose (built-in)'}"
        )

    def is_read_only(self) -> bool:
        return True

    def is_concurrency_safe(self) -> bool:
        return True

    def render_result_for_assistant(self, output: TaskToolOutput) -> str:
        details: List[str] = []
        if output.tool_use_count:
            details.append(f"{output.tool_use_count} tool uses")
        details.append(f"{output.duration_ms / 1000:.1f}s")
        if output.missing_tools:
            details.append(f"missing tools: {', '.join(output.missing_tools)}")

        suffix = f" ({'; '.join(details)})" if details else ""
        return f"[subagent:{output.agent_type}] {output.result_text}{suffix}"

    def render_tool_use_message(self, input_data: TaskToolInput, verbose: bool = False) -> str:
        del verbose
        return f"Task via {input_data.subagent_type}: {input_data.prompt}"

    async def call(
        self,
        input_data: TaskToolInput,
        context: ToolUseContext,
    ) -> AsyncGenerator[ToolOutput, None]:
        clear_agent_cache()
        agents = load_agent_definitions()
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
        subagent_messages = [create_user_message(input_data.prompt)]

        subagent_context = QueryContext(
            tools=typed_agent_tools,
            safe_mode=context.safe_mode,
            verbose=context.verbose,
            model=target_agent.model or "task",
        )

        start = time.time()
        assistant_messages: List[AssistantMessage] = []
        tool_use_count = 0

        yield ToolProgress(content=f"Launching subagent '{target_agent.agent_type}'")

        async for message in query(
            subagent_messages,  # type: ignore[arg-type]
            agent_system_prompt,
            {},
            subagent_context,
            context.permission_checker,
        ):
            if getattr(message, "type", "") == "assistant":
                # Surface subagent tool requests as progress so the user sees activity.
                msg_content = getattr(message, "message", None)
                blocks = getattr(msg_content, "content", []) if msg_content else []
                if isinstance(blocks, list):
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
                            yield ToolProgress(content=label)
                        if block_type == "text":
                            text_val = getattr(block, "text", None) or (
                                block.get("text") if isinstance(block, Dict) else ""
                            )
                            if text_val:
                                snippet = str(text_val).strip()
                                if snippet:
                                    short = (
                                        snippet if len(snippet) <= 200 else snippet[:197] + "..."
                                    )
                                    yield ToolProgress(content=f"Subagent: {short}")
                assistant_messages.append(message)  # type: ignore[arg-type]
                if isinstance(message, AssistantMessage):
                    tool_use_count += self._count_tool_uses(message)

        duration_ms = (time.time() - start) * 1000
        result_text = (
            self._extract_text(assistant_messages[-1])
            if assistant_messages
            else "Agent returned no response."
        )

        output = TaskToolOutput(
            agent_type=target_agent.agent_type,
            result_text=result_text.strip(),
            duration_ms=duration_ms,
            tool_use_count=tool_use_count,
            missing_tools=missing_tools,
            model_used=target_agent.model or "task",
        )

        yield ToolResult(data=output, result_for_assistant=self.render_result_for_assistant(output))

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
            except Exception:
                logger.exception(
                    "[task_tool] Failed to serialize tool_use input",
                    extra={"tool_use_input": str(inp)},
                )
                serialized = str(inp)
            return serialized if len(serialized) <= 120 else serialized[:117] + "..."

        return ", ".join(pieces)
