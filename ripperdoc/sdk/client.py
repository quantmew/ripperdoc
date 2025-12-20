"""Headless Python SDK for Ripperdoc.

`query` helper for simple calls and a `RipperdocClient` for long-lived
sessions that keep conversation history.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from ripperdoc.core.default_tools import get_default_tools
from ripperdoc.core.query import QueryContext, query as _core_query
from ripperdoc.core.permissions import PermissionResult
from ripperdoc.core.system_prompt import build_system_prompt
from ripperdoc.core.skills import build_skill_summary, load_all_skills
from ripperdoc.core.tool import Tool
from ripperdoc.tools.task_tool import TaskTool
from ripperdoc.tools.mcp_tools import load_dynamic_mcp_tools_async, merge_tools_with_dynamic
from ripperdoc.utils.memory import build_memory_instructions
from ripperdoc.utils.messages import (
    AssistantMessage,
    ProgressMessage,
    UserMessage,
    create_user_message,
)
from ripperdoc.utils.mcp import (
    format_mcp_instructions,
    load_mcp_servers_async,
    shutdown_mcp_runtime,
)
from ripperdoc.utils.log import get_logger

MessageType = Union[UserMessage, AssistantMessage, ProgressMessage]
PermissionChecker = Callable[
    [Tool[Any, Any], Any],
    Union[
        PermissionResult,
        Dict[str, Any],
        Tuple[bool, Optional[str]],
        bool,
        Awaitable[Union[PermissionResult, Dict[str, Any], Tuple[bool, Optional[str]], bool]],
    ],
]
QueryRunner = Callable[
    [
        List[MessageType],
        str,
        Dict[str, str],
        QueryContext,
        Optional[PermissionChecker],
    ],
    AsyncIterator[MessageType],
]

_END_OF_STREAM = object()

logger = get_logger()


def _coerce_to_path(path: Union[str, Path]) -> Path:
    return path if isinstance(path, Path) else Path(path)


@dataclass
class RipperdocOptions:
    """Configuration for SDK usage."""

    tools: Optional[Sequence[Tool[Any, Any]]] = None
    allowed_tools: Optional[Sequence[str]] = None
    disallowed_tools: Optional[Sequence[str]] = None
    yolo_mode: bool = False
    verbose: bool = False
    model: str = "main"
    max_thinking_tokens: int = 0
    context: Dict[str, str] = field(default_factory=dict)
    system_prompt: Optional[str] = None
    additional_instructions: Optional[Union[str, Sequence[str]]] = None
    permission_checker: Optional[PermissionChecker] = None
    cwd: Optional[Union[str, Path]] = None

    def build_tools(self) -> List[Tool[Any, Any]]:
        """Create the tool set with allow/deny filters applied."""
        base_tools = list(self.tools) if self.tools is not None else get_default_tools()
        allowed = set(self.allowed_tools) if self.allowed_tools is not None else None
        disallowed = set(self.disallowed_tools or [])

        filtered: List[Tool[Any, Any]] = []
        for tool in base_tools:
            name = getattr(tool, "name", tool.__class__.__name__)
            if allowed is not None and name not in allowed:
                continue
            if name in disallowed:
                continue
            filtered.append(tool)

        if allowed is not None and not filtered:
            raise ValueError("No tools remain after applying allowed_tools/disallowed_tools.")

        # The default Task tool captures the original base tools. If filters are
        # applied, recreate it so the subagent only sees the filtered set.
        if (self.allowed_tools or self.disallowed_tools) and self.tools is None:
            has_task = any(getattr(tool, "name", None) == "Task" for tool in filtered)
            if has_task:
                filtered_base = [tool for tool in filtered if getattr(tool, "name", None) != "Task"]

                def _filtered_base_provider() -> List[Tool[Any, Any]]:
                    return filtered_base

                filtered = [
                    (
                        TaskTool(_filtered_base_provider)
                        if getattr(tool, "name", None) == "Task"
                        else tool
                    )
                    for tool in filtered
                ]

        return filtered

    def extra_instructions(self) -> List[str]:
        """Normalize additional instructions to a list."""
        if self.additional_instructions is None:
            return []
        if isinstance(self.additional_instructions, str):
            return [self.additional_instructions]
        return [text for text in self.additional_instructions if text]


class RipperdocClient:
    """Persistent Ripperdoc session with conversation history."""

    def __init__(
        self,
        options: Optional[RipperdocOptions] = None,
        query_runner: Optional[QueryRunner] = None,
    ) -> None:
        self.options = options or RipperdocOptions()
        self._tools = self.options.build_tools()
        self._query_runner = query_runner or _core_query

        self._history: List[MessageType] = []
        self._queue: asyncio.Queue = asyncio.Queue()
        self._current_task: Optional[asyncio.Task] = None
        self._current_context: Optional[QueryContext] = None
        self._connected = False
        self._previous_cwd: Optional[Path] = None

    @property
    def tools(self) -> List[Tool[Any, Any]]:
        return self._tools

    @property
    def history(self) -> List[MessageType]:
        return list(self._history)

    async def __aenter__(self) -> "RipperdocClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:  # type: ignore[override]
        await self.disconnect()

    async def connect(self, prompt: Optional[str] = None) -> None:
        """Prepare the session and optionally send an initial prompt."""
        if not self._connected:
            if self.options.cwd is not None:
                self._previous_cwd = Path.cwd()
                os.chdir(_coerce_to_path(self.options.cwd))
            self._connected = True

        if prompt:
            await self.query(prompt)

    async def disconnect(self) -> None:
        """Tear down the session and restore the working directory."""
        if self._current_context:
            self._current_context.abort_controller.set()

        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            try:
                await self._current_task
            except asyncio.CancelledError:
                pass

        if self._previous_cwd:
            os.chdir(self._previous_cwd)
            self._previous_cwd = None

        self._connected = False
        await shutdown_mcp_runtime()

    async def query(self, prompt: str) -> None:
        """Send a prompt and start streaming the response."""
        if self._current_task and not self._current_task.done():
            raise RuntimeError(
                "A query is already in progress; wait for it to finish or interrupt it."
            )

        if not self._connected:
            await self.connect()

        self._queue = asyncio.Queue()

        user_message = create_user_message(prompt)
        history = list(self._history) + [user_message]
        self._history.append(user_message)

        system_prompt = await self._build_system_prompt(prompt)
        context = dict(self.options.context)

        query_context = QueryContext(
            tools=self._tools,
            max_thinking_tokens=self.options.max_thinking_tokens,
            yolo_mode=self.options.yolo_mode,
            model=self.options.model,
            verbose=self.options.verbose,
        )
        self._current_context = query_context

        async def _runner() -> None:
            try:
                async for message in self._query_runner(
                    history,
                    system_prompt,
                    context,
                    query_context,
                    self.options.permission_checker,
                ):
                    if getattr(message, "type", None) in ("user", "assistant"):
                        self._history.append(message)  # type: ignore[arg-type]
                    await self._queue.put(message)
            finally:
                await self._queue.put(_END_OF_STREAM)

        self._current_task = asyncio.create_task(_runner())

    async def receive_messages(self) -> AsyncIterator[MessageType]:
        """Yield messages for the active query."""
        if self._current_task is None:
            raise RuntimeError("No active query to receive messages from.")

        while True:
            message = await self._queue.get()
            if message is _END_OF_STREAM:
                break
            yield message  # type: ignore[misc]

    async def receive_response(self) -> AsyncIterator[MessageType]:
        """Alias for receive_messages."""
        async for message in self.receive_messages():
            yield message

    async def interrupt(self) -> None:
        """Request cancellation of the active query."""
        if self._current_context:
            self._current_context.abort_controller.set()

        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            try:
                await self._current_task
            except asyncio.CancelledError:
                pass

        await self._queue.put(_END_OF_STREAM)

    async def _build_system_prompt(self, user_prompt: str) -> str:
        if self.options.system_prompt:
            return self.options.system_prompt

        instructions: List[str] = []
        project_path = _coerce_to_path(self.options.cwd or Path.cwd())
        skill_result = load_all_skills(project_path)
        for err in skill_result.errors:
            logger.warning(
                "[skills] Failed to load skill",
                extra={"path": str(err.path), "reason": err.reason},
            )
        skill_instructions = build_skill_summary(skill_result.skills)
        if skill_instructions:
            instructions.append(skill_instructions)
        instructions.extend(self.options.extra_instructions())
        memory = build_memory_instructions()
        if memory:
            instructions.append(memory)

        dynamic_tools = await load_dynamic_mcp_tools_async(project_path)
        if dynamic_tools:
            self._tools = merge_tools_with_dynamic(self._tools, dynamic_tools)

        servers = await load_mcp_servers_async(project_path)
        mcp_instructions = format_mcp_instructions(servers)

        return build_system_prompt(
            self._tools,
            user_prompt,
            dict(self.options.context),
            instructions or None,
            mcp_instructions=mcp_instructions,
        )


async def query(
    prompt: str,
    options: Optional[RipperdocOptions] = None,
    query_runner: Optional[QueryRunner] = None,
) -> AsyncIterator[MessageType]:
    """One-shot helper: run a prompt in a fresh session."""
    client = RipperdocClient(options=options, query_runner=query_runner)
    await client.connect()
    await client.query(prompt)

    try:
        async for message in client.receive_messages():
            yield message
    finally:
        await client.disconnect()
