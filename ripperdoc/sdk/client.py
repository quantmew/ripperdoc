"""Headless Python SDK for Ripperdoc.

`query` helper for simple calls and a `RipperdocClient` for long-lived
sessions that keep conversation history.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
import warnings
from dataclasses import dataclass, field
from enum import Enum
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
from ripperdoc.core.hooks.llm_callback import build_hook_llm_callback
from ripperdoc.core.hooks.manager import hook_manager
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
    create_assistant_message,
    create_user_message,
)
from ripperdoc.utils.mcp import (
    McpServerInfo,
    format_mcp_instructions,
    load_mcp_servers_async,
    shutdown_mcp_runtime,
)
from ripperdoc.utils.session_history import (
    list_session_summaries,
    load_session_messages,
)
from ripperdoc.utils.log import get_logger


class PermissionMode(str, Enum):
    """Permission mode for SDK operations.

    - DEFAULT: Standard permission behavior, prompts for dangerous operations
    - ACCEPT_EDITS: Auto-accept file edits without prompting
    - BYPASS_PERMISSIONS: Bypass all permission checks (equivalent to yolo_mode)
    - PLAN: Planning mode - no execution, only planning
    """

    DEFAULT = "default"
    ACCEPT_EDITS = "acceptEdits"
    BYPASS_PERMISSIONS = "bypassPermissions"
    PLAN = "plan"


@dataclass
class McpServerConfig:
    """Configuration for an MCP server.

    Supports stdio, SSE, and HTTP server types.
    """

    # Server type: 'stdio', 'sse', or 'http'
    type: str = "stdio"
    # Command for stdio servers
    command: Optional[str] = None
    # Arguments for stdio servers
    args: Optional[List[str]] = None
    # URL for SSE/HTTP servers
    url: Optional[str] = None
    # Environment variables for stdio servers
    env: Optional[Dict[str, str]] = None
    # Headers for SSE/HTTP servers
    headers: Optional[Dict[str, str]] = None
    # Optional server description
    description: Optional[str] = None
    # Optional instructions for the server
    instructions: Optional[str] = None

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
    """Configuration for SDK usage.

    Attributes:
        tools: Custom tools to use instead of defaults.
        allowed_tools: List of tool names to allow (whitelist).
        disallowed_tools: List of tool names to disallow (blacklist).
        permission_mode: Permission mode for operations. Defaults to DEFAULT.
        verbose: Enable verbose output.
        model: Model pointer to use. Defaults to "main".
        max_thinking_tokens: Maximum tokens for thinking (0 = disabled).
        max_turns: Maximum conversation turns before stopping. None = unlimited.
        context: Additional context dictionary.
        system_prompt: Custom system prompt (overrides default).
        additional_instructions: Extra instructions to append to system prompt.
        permission_checker: Custom function to check tool permissions.
        cwd: Working directory for the session.
        resume: Session ID to resume from.
        continue_conversation: Continue the most recent conversation.
        mcp_servers: Programmatic MCP server configurations.
    """

    tools: Optional[Sequence[Tool[Any, Any]]] = None
    allowed_tools: Optional[Sequence[str]] = None
    disallowed_tools: Optional[Sequence[str]] = None
    permission_mode: PermissionMode = PermissionMode.DEFAULT
    verbose: bool = False
    model: str = "main"
    max_thinking_tokens: int = 0
    max_turns: Optional[int] = None
    context: Dict[str, str] = field(default_factory=dict)
    system_prompt: Optional[str] = None
    additional_instructions: Optional[Union[str, Sequence[str]]] = None
    permission_checker: Optional[PermissionChecker] = None
    cwd: Optional[Union[str, Path]] = None
    # Session management
    resume: Optional[str] = None
    continue_conversation: bool = False
    # MCP configuration
    mcp_servers: Optional[Dict[str, McpServerConfig]] = None
    # Deprecated: use permission_mode instead (kept for backward compatibility)
    yolo_mode: bool = False

    def __post_init__(self) -> None:
        """Handle deprecated yolo_mode parameter."""
        # If yolo_mode is explicitly set to True, apply permission_mode
        if self.yolo_mode:
            warnings.warn(
                "yolo_mode is deprecated, use permission_mode=PermissionMode.BYPASS_PERMISSIONS instead",
                DeprecationWarning,
                stacklevel=3,
            )
            self.permission_mode = PermissionMode.BYPASS_PERMISSIONS
        # If permission_mode is set to BYPASS_PERMISSIONS, sync yolo_mode
        elif self.permission_mode == PermissionMode.BYPASS_PERMISSIONS:
            object.__setattr__(self, "yolo_mode", True)

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
        self._session_hook_contexts: List[str] = []
        self._session_id: Optional[str] = None
        self._session_start_time: Optional[float] = None
        self._session_end_sent: bool = False
        self._turn_count: int = 0

    @property
    def tools(self) -> List[Tool[Any, Any]]:
        return self._tools

    @property
    def history(self) -> List[MessageType]:
        return list(self._history)

    @property
    def session_id(self) -> Optional[str]:
        """Return the current session ID."""
        return self._session_id

    @property
    def turn_count(self) -> int:
        """Return the number of turns in the current session."""
        return self._turn_count

    async def __aenter__(self) -> "RipperdocClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:  # type: ignore[override]
        await self.disconnect()

    def _load_resumed_session(self, project_path: Path) -> None:
        """Load history from a resumed or continued session."""
        session_id_to_load: Optional[str] = None

        if self.options.resume:
            session_id_to_load = self.options.resume
            logger.info(
                "[sdk] Resuming session",
                extra={"session_id": session_id_to_load},
            )
        elif self.options.continue_conversation:
            summaries = list_session_summaries(project_path)
            if summaries:
                session_id_to_load = summaries[0].session_id
                logger.info(
                    "[sdk] Continuing most recent session",
                    extra={
                        "session_id": session_id_to_load,
                        "last_prompt": summaries[0].last_prompt,
                    },
                )
            else:
                logger.debug("[sdk] No previous session found to continue")

        if session_id_to_load:
            self._session_id = session_id_to_load
            messages = load_session_messages(project_path, session_id_to_load)
            if messages:
                self._history = list(messages)
                # Count turns (each user message is a turn)
                self._turn_count = sum(
                    1 for m in messages if getattr(m, "type", None) == "user"
                )
                logger.info(
                    "[sdk] Loaded session history",
                    extra={
                        "session_id": session_id_to_load,
                        "message_count": len(messages),
                        "turn_count": self._turn_count,
                    },
                )

    async def connect(self, prompt: Optional[str] = None) -> None:
        """Prepare the session and optionally send an initial prompt."""
        if not self._connected:
            if self.options.cwd is not None:
                self._previous_cwd = Path.cwd()
                os.chdir(_coerce_to_path(self.options.cwd))
            self._connected = True
            project_path = _coerce_to_path(self.options.cwd or Path.cwd())
            hook_manager.set_project_dir(project_path)

            # Handle resume/continue_conversation
            self._load_resumed_session(project_path)

            self._session_id = self._session_id or str(uuid.uuid4())
            hook_manager.set_session_id(self._session_id)
            hook_manager.set_llm_callback(build_hook_llm_callback())

            # Load programmatic MCP servers if configured
            if self.options.mcp_servers:
                await self._load_programmatic_mcp_servers(project_path)

            try:
                source = "resume" if (self.options.resume or self.options.continue_conversation) else "startup"
                result = await hook_manager.run_session_start_async(source)
                self._session_hook_contexts = self._collect_hook_contexts(result)
                self._session_start_time = time.time()
                self._session_end_sent = False
            except (OSError, RuntimeError, ConnectionError, ValueError, TypeError) as exc:
                logger.warning(
                    "[sdk] SessionStart hook failed: %s: %s",
                    type(exc).__name__,
                    exc,
                )

        if prompt:
            await self.query(prompt)

    async def _load_programmatic_mcp_servers(self, project_path: Path) -> None:
        """Load MCP servers from programmatic configuration."""
        from ripperdoc.utils.mcp import McpRuntime, McpServerInfo, _runtime_var

        if not self.options.mcp_servers:
            return

        # Convert McpServerConfig to McpServerInfo
        configs: Dict[str, McpServerInfo] = {}
        for name, config in self.options.mcp_servers.items():
            configs[name] = McpServerInfo(
                name=name,
                type=config.type,
                url=config.url,
                description=config.description or "",
                command=config.command,
                args=config.args or [],
                env=config.env or {},
                headers=config.headers or {},
                instructions=config.instructions,
            )

        # Create and connect MCP runtime
        runtime = McpRuntime(project_path)
        await runtime.connect(configs)
        _runtime_var.set(runtime)
        logger.info(
            "[sdk] Loaded programmatic MCP servers",
            extra={
                "server_count": len(configs),
                "servers": list(configs.keys()),
            },
        )

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
        if not self._session_end_sent:
            duration = (
                max(time.time() - self._session_start_time, 0.0)
                if self._session_start_time is not None
                else None
            )
            try:
                await hook_manager.run_session_end_async(
                    "other",
                    duration_seconds=duration,
                    message_count=len(self._history),
                )
            except (OSError, RuntimeError, ConnectionError, ValueError, TypeError) as exc:
                logger.warning(
                    "[sdk] SessionEnd hook failed: %s: %s",
                    type(exc).__name__,
                    exc,
                )
            self._session_end_sent = True
        await shutdown_mcp_runtime()

    async def query(self, prompt: str) -> None:
        """Send a prompt and start streaming the response."""
        if self._current_task and not self._current_task.done():
            raise RuntimeError(
                "A query is already in progress; wait for it to finish or interrupt it."
            )

        if not self._connected:
            await self.connect()

        # Check max_turns limit
        if self.options.max_turns is not None and self._turn_count >= self.options.max_turns:
            error_message = create_assistant_message(
                f"Maximum turns ({self.options.max_turns}) reached. "
                "Create a new session to continue."
            )
            self._queue = asyncio.Queue()
            await self._queue.put(error_message)
            await self._queue.put(_END_OF_STREAM)
            self._current_task = asyncio.create_task(asyncio.sleep(0))
            return

        self._queue = asyncio.Queue()

        hook_result = await hook_manager.run_user_prompt_submit_async(prompt)
        if hook_result.should_block or not hook_result.should_continue:
            reason = (
                hook_result.block_reason
                or hook_result.stop_reason
                or "Prompt blocked by hook."
            )
            blocked_message = create_assistant_message(str(reason))
            self._history.append(blocked_message)
            await self._queue.put(blocked_message)
            await self._queue.put(_END_OF_STREAM)
            self._current_task = asyncio.create_task(asyncio.sleep(0))
            return
        hook_instructions = self._collect_hook_contexts(hook_result)

        user_message = create_user_message(prompt)
        history = list(self._history) + [user_message]
        self._history.append(user_message)
        self._turn_count += 1

        system_prompt = await self._build_system_prompt(prompt, hook_instructions)
        context = dict(self.options.context)

        # Determine yolo_mode from permission_mode
        yolo_mode = self.options.permission_mode in (
            PermissionMode.BYPASS_PERMISSIONS,
            PermissionMode.ACCEPT_EDITS,
        )

        query_context = QueryContext(
            tools=self._tools,
            max_thinking_tokens=self.options.max_thinking_tokens,
            yolo_mode=yolo_mode,
            model=self.options.model,
            verbose=self.options.verbose,
            max_turns=self.options.max_turns,
            permission_mode=self.options.permission_mode.value,
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

    async def _build_system_prompt(
        self, user_prompt: str, hook_instructions: Optional[List[str]] = None
    ) -> str:
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
        if self._session_hook_contexts:
            instructions.extend(self._session_hook_contexts)
        if hook_instructions:
            instructions.extend([text for text in hook_instructions if text])

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

    def _collect_hook_contexts(self, hook_result: Any) -> List[str]:
        contexts: List[str] = []
        system_message = getattr(hook_result, "system_message", None)
        additional_context = getattr(hook_result, "additional_context", None)
        if system_message:
            contexts.append(str(system_message))
        if additional_context:
            contexts.append(str(additional_context))
        return contexts


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
