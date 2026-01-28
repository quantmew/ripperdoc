"""Headless Python SDK for Ripperdoc.
"""

from __future__ import annotations

import asyncio
import os
import warnings
from collections.abc import AsyncIterable
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
from ripperdoc.core.tool import Tool
from ripperdoc.tools.task_tool import TaskTool
from ripperdoc.utils.messages import (
    AssistantMessage as RipperdocAssistantMessage,
    ProgressMessage as RipperdocProgressMessage,
    UserMessage as RipperdocUserMessage,
    create_assistant_message,
)
from ripperdoc.utils.log import get_logger

from .types import (
    Message,
    UserMessage,
    AssistantMessage,
    SystemMessage,
    ResultMessage,
    ContentBlock,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    ToolResultBlock,
    PermissionMode as TypedPermissionMode,
    SettingSource as TypedSettingSource,
    McpServerConfig as TypedMcpServerConfig,
    AgentDefinition as TypedAgentDefinition,
    HookMatcher as TypedHookMatcher,
    CanUseTool,
    ToolPermissionContext,
    PermissionResult,
    PermissionResultAllow,
    PermissionResultDeny,
    PermissionUpdate,
    SdkBeta,
    SandboxSettings,
    SystemPromptPreset,
    ToolsPreset,
)
from .adapter import (
    MessageAdapter,
    AsyncMessageAdapter,
    ResultMessageFactory,
)
# Subprocess mode imports (lazy loaded to avoid circular imports)
_subprocess_transport = None
_query_class = None


# PermissionMode: Use Literal type
# This is a type alias, not a class
PermissionMode = TypedPermissionMode

# Helper class for backward compatibility with code that uses PermissionMode.DEFAULT
class _PermissionModeCompat:
    """Compatibility class for PermissionMode enum-like access.

    Deprecated: Use string literals directly instead.
    Example: Use "default" instead of PermissionMode.DEFAULT
    """
    DEFAULT = "default"
    ACCEPT_EDITS = "acceptEdits"
    BYPASS_PERMISSIONS = "bypassPermissions"
    PLAN = "plan"

# For backward compatibility, create an enum-like object
PermissionModeCompat = _PermissionModeCompat()


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

    def to_typed_dict(self) -> TypedMcpServerConfig:
        """Convert to Claude SDK compatible TypedDict format."""
        if self.type == "stdio":
            return {
                "type": "stdio",
                "command": self.command or "",
                **({"args": self.args} if self.args else {}),
                **({"env": self.env} if self.env else {}),
            }
        elif self.type == "sse":
            return {
                "type": "sse",
                "url": self.url or "",
                **({"headers": self.headers} if self.headers else {}),
            }
        elif self.type == "http":
            return {
                "type": "http",
                "url": self.url or "",
                **({"headers": self.headers} if self.headers else {}),
            }
        else:
            # Default to stdio
            return {
                "type": "stdio",
                "command": self.command or "",
            }

    @classmethod
    def from_typed_dict(cls, config: TypedMcpServerConfig) -> "McpServerConfig":
        """Create from Claude SDK compatible TypedDict format."""
        server_type = config.get("type", "stdio")
        return cls(
            type=server_type,
            command=config.get("command"),
            args=config.get("args"),
            url=config.get("url"),
            env=config.get("env"),
            headers=config.get("headers"),
            description=config.get("description"),
            instructions=config.get("instructions"),
        )


@dataclass
class AgentConfig:
    """Programmatic configuration for a subagent.

    Allows defining custom subagents without using markdown files.
    """

    # Description of when to use this agent (shown in Task tool)
    description: str
    # System prompt for the agent
    prompt: str
    # Tools available to this agent. Use ["*"] for all tools.
    tools: Optional[List[str]] = None
    # Model to use: 'sonnet', 'opus', 'haiku', or None to inherit
    model: Optional[str] = None
    # Display color for the agent
    color: Optional[str] = None
    # Whether to fork context for this agent
    fork_context: bool = False

    def to_agent_definition(self) -> TypedAgentDefinition:
        """Convert to Claude SDK compatible AgentDefinition."""
        return TypedAgentDefinition(
            description=self.description,
            prompt=self.prompt,
            tools=self.tools,
            model=self._map_model_value(self.model),
        )

    @staticmethod
    def _map_model_value(model: Optional[str]) -> Optional[Literal["sonnet", "opus", "haiku", "inherit"]]:
        """Map model string to Claude SDK compatible literal."""
        if model is None:
            return None
        model_lower = model.lower()
        if "sonnet" in model_lower:
            return "sonnet"
        elif "opus" in model_lower:
            return "opus"
        elif "haiku" in model_lower:
            return "haiku"
        elif model_lower in ("inherit", "main"):
            return "inherit"
        return None

    @classmethod
    def from_agent_definition(cls, definition: TypedAgentDefinition) -> "AgentConfig":
        """Create from Claude SDK compatible AgentDefinition."""
        return cls(
            description=definition.description,
            prompt=definition.prompt,
            tools=definition.tools,
            model=definition.model,
        )


# Type alias for hook callback functions
# Hook callbacks receive event type, input data, and return a decision dict
HookCallback = Callable[
    [str, Dict[str, Any]],
    Union[
        Dict[str, Any],  # Sync return
        Awaitable[Dict[str, Any]],  # Async return
    ],
]


@dataclass
class HookMatcher:
    """Matcher configuration for a programmatic hook.

    Defines when a hook should be triggered based on tool names or patterns.
    """
    # Callback function to execute
    callback: HookCallback
    # Tool name pattern to match (for PreToolUse/PostToolUse hooks)
    tool_pattern: Optional[str] = None

    def to_typed_matcher(self) -> TypedHookMatcher:
        """Convert to Claude SDK compatible HookMatcher."""
        # Wrap single callback in a list for compatibility
        return TypedHookMatcher(
            matcher=self.tool_pattern,
            hooks=[self._wrap_callback(self.callback)],
        )

    @staticmethod
    def _wrap_callback(
        callback: HookCallback,
    ) -> Callable[
        [Any, str | None, Any],  # HookInput, tool_use_id, HookContext
        Awaitable[Dict[str, Any]],
    ]:
        """Wrap Ripperdoc-style hook callback to Claude SDK format."""
        async def wrapped(
            input_data: Any,
            tool_use_id: str | None,
            context: Any,
        ) -> Dict[str, Any]:
            # Extract event type from input_data if available
            event_type = getattr(input_data, "hook_event_name", "Unknown")
            # Convert input to dict format expected by Ripperdoc callback
            input_dict = input_data if isinstance(input_data, dict) else {
                "event": event_type,
                "data": input_data,
            }
            result = callback(event_type, input_dict)
            if asyncio.iscoroutine(result):
                return await result
            return result  # type: ignore

        return wrapped


# Type alias for stderr callback
StderrCallback = Callable[[str], None]


class SettingSource(str, Enum):
    """Sources for loading settings configuration.

    Controls which settings files are loaded during session initialization.

    Deprecated: Use string literals directly.
    Example: Use "user" instead of SettingSource.USER
    """

    USER = "user"        # ~/.ripperdoc/settings.json
    PROJECT = "project"  # .ripperdoc/settings.json in project
    LOCAL = "local"      # .ripperdoc.local/settings.json
    ENV = "env"          # Environment variables


# Use TypedSettingSource for Claude SDK compatibility
SettingSourceType = TypedSettingSource


MessageType = Union[RipperdocUserMessage, RipperdocAssistantMessage, RipperdocProgressMessage]
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

_END_OF_STREAM = object()

logger = get_logger()

# Module-level registries for programmatic agents and hooks
# These allow TaskTool and HookManager to access SDK-defined configurations
_programmatic_agents: Dict[str, Any] = {}  # agent_type -> AgentDefinition
_programmatic_hooks: Dict[str, List[HookMatcher]] = {}  # event_name -> List[HookMatcher]


def get_programmatic_agents() -> Dict[str, Any]:
    """Get programmatically registered agents."""
    return _programmatic_agents


def get_programmatic_hooks() -> Dict[str, List[HookMatcher]]:
    """Get programmatically registered hooks."""
    return _programmatic_hooks


def clear_programmatic_registries() -> None:
    """Clear all programmatic registries."""
    _programmatic_agents.clear()
    _programmatic_hooks.clear()


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
        agents: Programmatic subagent definitions (keyed by agent type name).
        hooks: Programmatic hook callbacks (keyed by event name).
        env: Environment variables to pass to subprocesses.
        additional_directories: Extra directories the agent can access.
        include_partial_messages: Include partial message events during streaming.
        stderr: Callback for stderr output from subprocesses.
        fork_session: Create a new session branch when resuming.
        setting_sources: Which settings sources to load (user, project, local, env).
        user: User identifier for the session.
        permission_prompt_tool_name: MCP tool name for permission prompts.
        settings: Path to custom settings file.
        extra_args: Additional arguments to pass through.
        max_buffer_size: Maximum buffer size for streaming responses.
    """

    tools: Optional[Sequence[Tool[Any, Any]]] = None
    allowed_tools: Optional[Sequence[str]] = None
    disallowed_tools: Optional[Sequence[str]] = None
    permission_mode: PermissionMode = "default"  # Use string literal for Claude SDK compatibility
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
    fork_session: bool = False
    # MCP configuration
    mcp_servers: Optional[Dict[str, McpServerConfig]] = None
    # Programmatic agents (key = agent type name)
    agents: Optional[Dict[str, AgentConfig]] = None
    # Programmatic hooks (key = event name like "PreToolUse", "PostToolUse", etc.)
    hooks: Optional[Dict[str, List[HookMatcher]]] = None
    # Environment variables for subprocesses
    env: Optional[Dict[str, str]] = None
    # Additional directories the agent can access
    additional_directories: Optional[List[str]] = None
    # Include partial messages during streaming
    include_partial_messages: bool = False
    # Stderr callback for subprocess output
    stderr: Optional[StderrCallback] = None
    # Low priority options
    setting_sources: Optional[List[SettingSource]] = None
    user: Optional[str] = None
    permission_prompt_tool_name: Optional[str] = None
    settings: Optional[Union[str, Path]] = None
    extra_args: Optional[Dict[str, Optional[str]]] = None
    max_buffer_size: Optional[int] = None
    # Deprecated: use permission_mode instead (kept for backward compatibility)
    yolo_mode: bool = False
    # Claude SDK specific fields (accepted but may not be fully supported)
    max_budget_usd: Optional[float] = None
    fallback_model: Optional[str] = None
    betas: List[str] = field(default_factory=list)
    sandbox: Optional[Dict[str, Any]] = None
    enable_file_checkpointing: bool = False
    output_format: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Handle deprecated yolo_mode parameter."""
        # If yolo_mode is explicitly set to True, apply permission_mode
        if self.yolo_mode:
            warnings.warn(
                "yolo_mode is deprecated, use permission_mode='bypassPermissions' instead",
                DeprecationWarning,
                stacklevel=3,
            )
            self.permission_mode = "bypassPermissions"
        # If permission_mode is set to BYPASS_PERMISSIONS, sync yolo_mode
        elif self.permission_mode == "bypassPermissions":
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


# For Claude SDK compatibility, we create an alias
ClaudeAgentOptions = RipperdocOptions


class RipperdocSDKClient:
    """Persistent session with conversation history.

    This class provides Claude Agent SDK compatible interface using
    subprocess architecture. The SDK communicates with a Ripperdoc CLI
    subprocess via JSON Control Protocol over stdio.

    Subprocess Architecture:
    - Client spawns a CLI subprocess
    - Communication via JSON Control Protocol over stdin/stdout
    - Enables multi-language SDK support and process isolation
    """

    def __init__(
        self,
        options: Optional[ClaudeAgentOptions] = None,
    ) -> None:
        self.options = options or ClaudeAgentOptions()

        self._history: List[Union[RipperdocUserMessage, RipperdocAssistantMessage, RipperdocProgressMessage]] = []
        self._queue: asyncio.Queue = asyncio.Queue()
        self._current_task: Optional[asyncio.Task] = None
        self._connected = False
        self._previous_cwd: Optional[Path] = None
        self._session_hook_contexts: List[str] = []
        self._session_id: Optional[str] = None
        self._session_start_time: Optional[float] = None
        self._session_end_sent: bool = False
        self._turn_count: int = 0
        # Track current model for Claude SDK compatibility
        self._current_model: str = self.options.model or "unknown"

        # Subprocess components
        self._transport: Optional[Any] = None  # SubprocessCLITransport
        self._query: Optional[Any] = None  # Query class

        # Initialize subprocess components
        self._init_subprocess_components()

    def _init_subprocess_components(self) -> None:
        """Initialize subprocess mode components.

        Imports are done here to avoid circular import issues.
        """
        global _subprocess_transport, _query_class

        if _subprocess_transport is None:
            from ripperdoc.sdk._internal.transport.stdio_cli import SubprocessCLITransport
            _subprocess_transport = SubprocessCLITransport

        if _query_class is None:
            from ripperdoc.sdk._internal.query import Query
            _query_class = Query

        # Convert options to typed dict format for transport
        self._transport_options = self._build_transport_options()

    def _build_transport_options(self) -> Any:
        """Build options dict for SubprocessCLITransport."""
        # Build the options dict
        options_dict = {
            "model": self.options.model or "main",
            "permission_mode": self.options.permission_mode,
            "max_turns": self.options.max_turns,
            "system_prompt": self.options.system_prompt,
            "cwd": str(self.options.cwd) if self.options.cwd else None,
            "allowed_tools": list(self.options.allowed_tools) if self.options.allowed_tools else None,
            "disallowed_tools": list(self.options.disallowed_tools) if self.options.disallowed_tools else None,
            "env": self.options.env or {},
            "stderr": self.options.stderr,
            "max_buffer_size": self.options.max_buffer_size,
            # Claude SDK compatibility fields (passed but may be ignored)
            "max_budget_usd": self.options.max_budget_usd,
            "fallback_model": self.options.fallback_model,
            "betas": self.options.betas,
            "sandbox": self.options.sandbox,
            "enable_file_checkpointing": self.options.enable_file_checkpointing,
            "output_format": self.options.output_format,
        }

        # Remove None values
        options_dict = {k: v for k, v in options_dict.items() if v is not None}

        return options_dict

    @property
    def history(self) -> List[Union[RipperdocUserMessage, RipperdocAssistantMessage, RipperdocProgressMessage]]:
        return list(self._history)

    @property
    def session_id(self) -> Optional[str]:
        """Return the current session ID."""
        return self._session_id

    @property
    def turn_count(self) -> int:
        """Return the number of turns in the current session."""
        return self._turn_count

    @property
    def user(self) -> Optional[str]:
        """Return the user identifier for this session."""
        return self.options.user

    async def __aenter__(self) -> "ClaudeSDKClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:  # type: ignore[override]
        await self.disconnect()

    async def connect(self, prompt: Optional[str] = None) -> None:
        """Connect to the CLI subprocess and initialize the session."""
        if not self._connected:
            # Change working directory if specified
            if self.options.cwd is not None:
                self._previous_cwd = Path.cwd()
                os.chdir(_coerce_to_path(self.options.cwd))

            # Initialize subprocess connection
            await self._connect_subprocess()

            self._connected = True

        if prompt:
            await self.query(prompt)

    async def _connect_subprocess(self) -> None:
        """Initialize subprocess mode connection.

        Creates the SubprocessCLITransport and Query instances,
        connects to the CLI subprocess, and sends initialize request.
        """
        if not _subprocess_transport or not _query_class:
            raise RuntimeError("Subprocess components not initialized")

        logger.info("[sdk] Connecting in subprocess mode")

        # Build options dict for transport (avoids circular imports)
        options_dict = self._build_transport_options()

        # Create the transport with dict options
        self._transport = _subprocess_transport(
            prompt="",  # Empty prompt for streaming mode
            options=options_dict,  # Pass dict directly, transport accepts both dict and object
        )

        # Connect the transport (starts the subprocess)
        await self._transport.connect()

        # Create the Query handler
        self._query = _query_class(
            transport=self._transport,
            is_streaming_mode=True,
            can_use_tool=self._build_permission_checker(),
            hooks=self._build_hooks_dict(),
            sdk_mcp_servers=self.options.mcp_servers,
        )

        # Start the query's message reading task
        await self._query.start()

        # Send initialize request
        try:
            init_response = await self._query._send_control_request(
                {
                    "subtype": "initialize",
                    "options": options_dict,
                },
                timeout=60.0,
            )
            self._session_id = init_response.get("session_id")
            logger.info(
                "[sdk] Subprocess initialized",
                extra={"session_id": self._session_id},
            )
        except Exception as e:
            logger.error(f"[sdk] Failed to initialize subprocess: {e}")
            await self._transport.close()
            raise

    def _build_permission_checker(self) -> Optional[Callable]:
        """Build permission checker for subprocess mode."""
        if self.options.permission_mode == "bypassPermissions":
            return None

        # Create a wrapper that converts between SDK and CLI formats
        async def checker(
            tool_name: str,
            tool_input: dict[str, Any],
            context: Any,
        ) -> Any:
            from ripperdoc.sdk.types import PermissionResultAllow, PermissionResultDeny

            # Call the original permission checker
            if self.options.permission_checker:
                result = self.options.permission_checker(
                    tool_name,
                    tool_input,
                    context,
                )
                if asyncio.iscoroutine(result):
                    result = await result

                # Convert result to SDK format
                if isinstance(result, bool):
                    return PermissionResultAllow() if result else PermissionResultDeny()
                elif isinstance(result, tuple):
                    allowed, message = result
                    return PermissionResultAllow() if allowed else PermissionResultDeny(message=message)
                elif isinstance(result, dict):
                    if result.get("decision") == "allow":
                        return PermissionResultAllow(updated_input=result.get("updated_input"))
                    else:
                        return PermissionResultDeny(message=result.get("message"), interrupt=result.get("interrupt", False))
                elif isinstance(result, PermissionResult):
                    return result
                else:
                    return PermissionResultAllow()

            # Default: allow
            return PermissionResultAllow()

        return checker

    def _build_hooks_dict(self) -> dict[str, list[dict[str, Any]]]:
        """Build hooks dict for subprocess mode."""
        hooks_dict: dict[str, list[dict[str, Any]]] = {}

        if not self.options.hooks:
            return hooks_dict

        # Convert HookMatcher objects to dict format
        for event_name, matchers in self.options.hooks.items():
            hooks_dict[event_name] = []
            for matcher in matchers:
                hook_dict = {
                    "matcher": matcher.matcher if hasattr(matcher, "matcher") else "*",
                    "callback_id": f"hook_{id(matcher)}",  # Use id as unique identifier
                }
                hooks_dict[event_name].append(hook_dict)

        return hooks_dict

    async def disconnect(self) -> None:
        """Close the subprocess connection and clean up resources."""
        # Close query and transport
        if self._query:
            await self._query.close()
            self._query = None
        if self._transport:
            await self._transport.close()
            self._transport = None

        # Cancel current task if running
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            try:
                await self._current_task
            except asyncio.CancelledError:
                pass

        # Restore working directory
        if self._previous_cwd:
            os.chdir(self._previous_cwd)
            self._previous_cwd = None

        self._connected = False

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

        # Send query via subprocess
        if not self._query:
            raise RuntimeError("Query not initialized in subprocess mode")

        async def _runner() -> None:
            try:
                # Send query request via control protocol
                await self._query._send_control_request(
                    {
                        "subtype": "query",
                        "prompt": prompt,
                    }
                )

                # Receive messages from the query's message stream
                async for message in self._query.receive_messages():
                    if message.type in ("user", "assistant", "progress"):
                        self._history.append(message)  # type: ignore[arg-type]
                    await self._queue.put(message)
            finally:
                await self._queue.put(_END_OF_STREAM)

        self._current_task = asyncio.create_task(_runner())

    async def receive_messages(self) -> AsyncIterator[Message]:
        """Yield messages for the active query in Claude SDK compatible format.

        This method returns messages in Claude Agent SDK compatible format
        (UserMessage, AssistantMessage, SystemMessage, ResultMessage).
        """
        if self._current_task is None:
            raise RuntimeError("No active query to receive messages from.")

        while True:
            message = await self._queue.get()
            if message is _END_OF_STREAM:
                break

            # Messages are already in Claude SDK format from subprocess
            yield message  # type: ignore

    async def receive_response(self) -> AsyncIterator[Message]:
        """Yield messages until and including a ResultMessage.

        This async iterator yields all messages in sequence and automatically terminates
        after yielding a ResultMessage (which indicates the response is complete).

        Yields:
            Message: Each message received (UserMessage, AssistantMessage, SystemMessage, ResultMessage)
        """
        async for message in self.receive_messages():
            yield message
            if isinstance(message, ResultMessage):
                return

    async def interrupt(self) -> None:
        """Request cancellation of the active query."""
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            try:
                await self._current_task
            except asyncio.CancelledError:
                pass

        await self._queue.put(_END_OF_STREAM)

    async def set_permission_mode(self, mode: PermissionMode) -> None:
        """Change permission mode during conversation.

        Args:
            mode: The permission mode to set. Valid options:
                - 'default': Prompts for dangerous tools
                - 'acceptEdits': Auto-accept file edits
                - 'bypassPermissions': Allow all tools (use with caution)
                - 'plan': Planning mode - no execution

        Example:
            ```python
            async with ClaudeSDKClient() as client:
                await client.query("Help me analyze this code")
                await client.set_permission_mode('acceptEdits')
                await client.query("Now implement the fix")
            ```
        """
        if mode not in ("default", "acceptEdits", "bypassPermissions", "plan"):
            raise ValueError(f"Invalid permission mode: {mode}")

        self.options.permission_mode = mode
        logger.info(f"[sdk] Permission mode changed to {mode}")

    async def set_model(self, model: Optional[str] = None) -> None:
        """Change the AI model during conversation.

        Args:
            model: The model to use, or None to use default.

        Example:
            ```python
            async with ClaudeSDKClient() as client:
                await client.query("Help me understand this problem")
                await client.set_model('claude-sonnet-4-5')
                await client.query("Now implement the solution")
            ```
        """
        self.options.model = model
        self._current_model = model or "unknown"
        logger.info(f"[sdk] Model changed to {model}")

    async def rewind_files(self, user_message_id: str) -> None:
        """Rewind tracked files to their state at a specific user message.

        Note: This is a placeholder for Claude SDK compatibility.
        File checkpointing is not currently implemented in Ripperdoc.

        Args:
            user_message_id: UUID of the user message to rewind to.

        Raises:
            NotImplementedError: File checkpointing is not supported yet.
        """
        raise NotImplementedError(
            "File checkpointing and rewind_files() are not currently supported "
            "in Ripperdoc. This method exists for Claude SDK API compatibility."
        )

    async def get_server_info(self) -> Dict[str, Any] | None:
        """Get server initialization info.

        Returns basic information about the current session.

        Returns:
            Dictionary with session info, or None if not connected.

        Example:
            ```python
            async with ClaudeSDKClient() as client:
                info = await client.get_server_info()
                if info:
                    print(f"Session ID: {info.get('session_id')}")
            ```
        """
        if not self._connected:
            return None

        return {
            "session_id": self._session_id,
            "turn_count": self._turn_count,
            "model": self._current_model,
            "cwd": str(self.options.cwd) if self.options.cwd else None,
            "permission_mode": self.options.permission_mode,
        }


async def query(
    *,
    prompt: Union[str, AsyncIterable[dict[str, Any]]],
    options: Optional[ClaudeAgentOptions] = None,
    transport: Any = None,  # Ignored, for Claude SDK compatibility
) -> AsyncIterator[Message]:
    """Query for one-shot or unidirectional streaming interactions.

    This function is compatible with Claude Agent SDK's query() function.
    It provides a simple, stateless interface for queries where you don't need
    bidirectional communication or conversation management.

    Args:
        prompt: The prompt to send. Can be a string for single-shot queries
                or an AsyncIterable[dict] for streaming mode.
        options: Optional configuration (defaults to ClaudeAgentOptions() if None).
        transport: Ignored parameter for Claude SDK compatibility.

    Yields:
        Messages from the conversation in Claude SDK compatible format.

    Example:
        ```python
        async for message in query(
            prompt="What is the capital of France?",
            options=ClaudeAgentOptions(allowed_tools=["Bash"])
        ):
            print(message)
        ```
    """
    # Handle streaming mode (AsyncIterable prompt)
    if isinstance(prompt, AsyncIterable):
        # For streaming mode, we use the client directly
        client = RipperdocSDKClient(options=options)
        await client.connect()
        try:
            async for msg in client.receive_messages():
                yield msg
        finally:
            await client.disconnect()
        return

    # For simple string prompts, use the original flow
    internal_options = options or ClaudeAgentOptions()
    client = RipperdocSDKClient(options=internal_options)
    await client.connect()
    await client.query(str(prompt))

    try:
        async for message in client.receive_messages():
            # receive_messages() already converts to Claude SDK compatible format
            yield message
    finally:
        await client.disconnect()


# =============================================================================
# Backward Compatibility Aliases
# =============================================================================

# For backward compatibility, provide aliases using old names
RipperdocClient = RipperdocSDKClient
# For Claude SDK compatibility, provide alias
ClaudeSDKClient = RipperdocSDKClient
# Note: RipperdocOptions is already an alias for ClaudeAgentOptions (defined above)

# Re-export internal message types for compatibility
MessageType = Union[RipperdocUserMessage, RipperdocAssistantMessage, RipperdocProgressMessage]
ProgressMessage = RipperdocProgressMessage
