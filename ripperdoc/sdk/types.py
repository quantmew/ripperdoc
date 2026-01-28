"""SDK compatible type definitions for Ripperdoc.
"""

from __future__ import annotations

from collections.abc import AsyncIterable, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Literal,
    NotRequired,
    TypedDict,
    TypeVar,
    Union,
    Generic,
)

# =============================================================================
# ContentBlock Types
# =============================================================================


@dataclass
class TextBlock:
    """Text content block.

    Represents plain text content in a message.
    """

    text: str


@dataclass
class ThinkingBlock:
    """Thinking/reasoning content block.

    Contains extended thinking output from models with reasoning capabilities.
    """

    thinking: str
    signature: str


@dataclass
class ToolUseBlock:
    """Tool use content block.

    Represents a tool invocation with its parameters.
    """

    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ToolResultBlock:
    """Tool result content block.

    Contains the result of a tool execution.
    """

    tool_use_id: str
    content: str | list[dict[str, Any]] | None = None
    is_error: bool | None = None


# Union type for all content block types
ContentBlock = TextBlock | ThinkingBlock | ToolUseBlock | ToolResultBlock


# =============================================================================
# Message Types
# =============================================================================


@dataclass
class UserMessage:
    """User message with optional tool results.

    Represents a message sent by the user, which may include tool execution results.
    """

    content: str | list[ContentBlock]
    uuid: str | None = None
    parent_tool_use_id: str | None = None
    tool_use_result: dict[str, Any] | None = None


@dataclass
class AssistantMessage:
    """Assistant message with content blocks.

    Represents a response from the AI assistant, containing text, thinking,
    and/or tool use blocks.
    """

    content: list[ContentBlock]
    model: str
    parent_tool_use_id: str | None = None
    error: str | None = None


@dataclass
class SystemMessage:
    """System message with metadata.

    Contains system-level information and events.
    """

    subtype: str
    data: dict[str, Any]


@dataclass
class ResultMessage:
    """Result message with cost and usage information.

    Indicates the completion of a request with timing, cost, and usage statistics.
    """

    subtype: str
    duration_ms: int
    duration_api_ms: int
    is_error: bool
    num_turns: int
    session_id: str
    total_cost_usd: float | None = None
    usage: dict[str, Any] | None = None
    result: str | None = None
    structured_output: Any = None


@dataclass
class StreamEvent:
    """Stream event for partial message updates during streaming.

    Contains raw stream events from the underlying API for advanced use cases.
    """

    uuid: str
    session_id: str
    event: dict[str, Any]
    parent_tool_use_id: str | None = None


# Union type for all message types
Message = UserMessage | AssistantMessage | SystemMessage | ResultMessage | StreamEvent


# =============================================================================
# Permission System Types
# =============================================================================

# Permission mode literal type
PermissionMode = Literal["default", "acceptEdits", "plan", "bypassPermissions"]


# Permission update destination
PermissionUpdateDestination = Literal[
    "userSettings", "projectSettings", "localSettings", "session"
]

# Permission behavior
PermissionBehavior = Literal["allow", "deny", "ask"]


@dataclass
class PermissionRuleValue:
    """Permission rule value."""

    tool_name: str
    rule_content: str | None = None


@dataclass
class PermissionUpdate:
    """Permission update configuration.

    Defines how permissions should be modified during a session.
    """

    type: Literal[
        "addRules",
        "replaceRules",
        "removeRules",
        "setMode",
        "addDirectories",
        "removeDirectories",
    ]
    rules: list[PermissionRuleValue] | None = None
    behavior: PermissionBehavior | None = None
    mode: PermissionMode | None = None
    directories: list[str] | None = None
    destination: PermissionUpdateDestination | None = None


@dataclass
class ToolPermissionContext:
    """Context information for tool permission callbacks.

    Provides additional context when making permission decisions.
    """

    signal: Any | None = None
    suggestions: list[PermissionUpdate] = field(default_factory=list)


@dataclass
class PermissionResultAllow:
    """Allow permission result.

    Indicates that a tool operation is allowed, optionally with modifications.
    """

    behavior: Literal["allow"] = "allow"
    updated_input: dict[str, Any] | None = None
    updated_permissions: list[PermissionUpdate] | None = None


@dataclass
class PermissionResultDeny:
    """Deny permission result.

    Indicates that a tool operation is denied.
    """

    behavior: Literal["deny"] = "deny"
    message: str = ""
    interrupt: bool = False


# Union type for permission results
PermissionResult = PermissionResultAllow | PermissionResultDeny

# Tool permission callback type
CanUseTool = Callable[
    [str, dict[str, Any], ToolPermissionContext],
    Awaitable[PermissionResult],
]


# =============================================================================
# Hook System Types
# =============================================================================

# Hook event types
HookEvent = (
    Literal["PreToolUse"]
    | Literal["PostToolUse"]
    | Literal["UserPromptSubmit"]
    | Literal["Stop"]
    | Literal["SubagentStop"]
    | Literal["PreCompact"]
)


# Base hook input fields
class BaseHookInput(TypedDict, total=False):
    """Base hook input fields present across many hook events."""

    session_id: str
    transcript_path: str
    cwd: str
    permission_mode: str


# Specific hook input types
class PreToolUseHookInput(BaseHookInput):
    """Input data for PreToolUse hook events."""

    hook_event_name: Literal["PreToolUse"]
    tool_name: str
    tool_input: dict[str, Any]


class PostToolUseHookInput(BaseHookInput):
    """Input data for PostToolUse hook events."""

    hook_event_name: Literal["PostToolUse"]
    tool_name: str
    tool_input: dict[str, Any]
    tool_response: Any


class UserPromptSubmitHookInput(BaseHookInput):
    """Input data for UserPromptSubmit hook events."""

    hook_event_name: Literal["UserPromptSubmit"]
    prompt: str


class StopHookInput(BaseHookInput):
    """Input data for Stop hook events."""

    hook_event_name: Literal["Stop"]
    stop_hook_active: bool


class SubagentStopHookInput(BaseHookInput):
    """Input data for SubagentStop hook events."""

    hook_event_name: Literal["SubagentStop"]
    stop_hook_active: bool


class PreCompactHookInput(BaseHookInput):
    """Input data for PreCompact hook events."""

    hook_event_name: Literal["PreCompact"]
    trigger: Literal["manual", "auto"]
    custom_instructions: str | None


# Union type for all hook inputs
HookInput = (
    PreToolUseHookInput
    | PostToolUseHookInput
    | UserPromptSubmitHookInput
    | StopHookInput
    | SubagentStopHookInput
    | PreCompactHookInput
)


# Hook-specific output types
class PreToolUseHookSpecificOutput(TypedDict):
    """Hook-specific output for PreToolUse events."""

    hookEventName: Literal["PreToolUse"]
    permissionDecision: NotRequired[Literal["allow", "deny", "ask"]]
    permissionDecisionReason: NotRequired[str]
    updatedInput: NotRequired[dict[str, Any]]


class PostToolUseHookSpecificOutput(TypedDict):
    """Hook-specific output for PostToolUse events."""

    hookEventName: Literal["PostToolUse"]
    additionalContext: NotRequired[str]


class UserPromptSubmitHookSpecificOutput(TypedDict):
    """Hook-specific output for UserPromptSubmit events."""

    hookEventName: Literal["UserPromptSubmit"]
    additionalContext: NotRequired[str]


HookSpecificOutput = (
    PreToolUseHookSpecificOutput
    | PostToolUseHookSpecificOutput
    | UserPromptSubmitHookSpecificOutput
)


# Hook JSON output types
class AsyncHookJSONOutput(TypedDict):
    """Async hook output that defers hook execution.

    Note: async_ is used instead of async to avoid Python keyword conflict.
    """

    async_: Literal[True]
    asyncTimeout: NotRequired[int]


class SyncHookJSONOutput(TypedDict):
    """Synchronous hook output with control and decision fields.

    Note: continue_ is used instead of continue to avoid Python keyword conflict.
    """

    continue_: NotRequired[bool]
    suppressOutput: NotRequired[bool]
    stopReason: NotRequired[str]
    decision: NotRequired[Literal["block"]]
    systemMessage: NotRequired[str]
    reason: NotRequired[str]
    hookSpecificOutput: NotRequired[HookSpecificOutput]


HookJSONOutput = AsyncHookJSONOutput | SyncHookJSONOutput


class HookContext(TypedDict):
    """Context information for hook callbacks."""

    signal: Any | None


# Hook callback type
HookCallback = Callable[
    [HookInput, str | None, HookContext],
    Awaitable[HookJSONOutput],
]


@dataclass
class HookMatcher:
    """Hook matcher configuration.

    Defines when a hook should be triggered based on tool names or patterns.
    """

    matcher: str | None = None
    hooks: list[HookCallback] = field(default_factory=list)
    timeout: float | None = None


# =============================================================================
# Agent Definition Types
# =============================================================================

@dataclass
class AgentDefinition:
    """Agent definition configuration.

    Defines a subagent with its own system prompt and available tools.
    """

    description: str
    prompt: str
    tools: list[str] | None = None
    model: Literal["sonnet", "opus", "haiku", "inherit"] | None = None


# Setting source literal type
SettingSource = Literal["user", "project", "local"]


# =============================================================================
# MCP Server Types
# =============================================================================

class McpStdioServerConfig(TypedDict):
    """MCP stdio server configuration."""

    type: NotRequired[Literal["stdio"]]
    command: str
    args: NotRequired[list[str]]
    env: NotRequired[dict[str, str]]


class McpSSEServerConfig(TypedDict):
    """MCP SSE server configuration."""

    type: Literal["sse"]
    url: str
    headers: NotRequired[dict[str, str]]


class McpHttpServerConfig(TypedDict):
    """MCP HTTP server configuration."""

    type: Literal["http"]
    url: str
    headers: NotRequired[dict[str, str]]


class McpSdkServerConfig(TypedDict):
    """SDK MCP server configuration (for in-process servers).

    Note: Ripperdoc doesn't support this type yet, but the type is
    defined for API compatibility.
    """

    type: Literal["sdk"]
    name: str
    instance: Any  # MCP Server instance


# Union type for all MCP server configs
McpServerConfig = (
    McpStdioServerConfig | McpSSEServerConfig | McpHttpServerConfig | McpSdkServerConfig
)


# =============================================================================
# Plugin and Beta Types
# =============================================================================

class SdkPluginConfig(TypedDict):
    """SDK plugin configuration."""

    type: Literal["local"]
    path: str


# Beta features - see https://docs.anthropic.com/en/api/beta-headers
SdkBeta = Literal["context-1m-2025-08-07"]


# =============================================================================
# Sandbox Types
# =============================================================================

class SandboxNetworkConfig(TypedDict, total=False):
    """Network configuration for sandbox."""

    allowUnixSockets: list[str]
    allowAllUnixSockets: bool
    allowLocalBinding: bool
    httpProxyPort: int
    socksProxyPort: int


class SandboxIgnoreViolations(TypedDict, total=False):
    """Violations to ignore in sandbox."""

    file: list[str]
    network: list[str]


class SandboxSettings(TypedDict, total=False):
    """Sandbox settings configuration.

    Controls how bash commands are sandboxed for isolation.
    Note: Ripperdoc doesn't fully support sandboxing yet,
    but the type is defined for API compatibility.
    """

    enabled: bool
    autoAllowBashIfSandboxed: bool
    excludedCommands: list[str]
    allowUnsandboxedCommands: bool
    network: SandboxNetworkConfig
    ignoreViolations: SandboxIgnoreViolations
    enableWeakerNestedSandbox: bool


# =============================================================================
# System Prompt Types
# =============================================================================

class SystemPromptPreset(TypedDict):
    """System prompt preset configuration."""

    type: Literal["preset"]
    preset: Literal["claude_code"]
    append: NotRequired[str]


class ToolsPreset(TypedDict):
    """Tools preset configuration."""

    type: Literal["preset"]
    preset: Literal["claude_code"]


# =============================================================================
# Error Types
# =============================================================================


class ClaudeSDKError(Exception):
    """Base exception for all SDK errors."""


class CLIConnectionError(ClaudeSDKError):
    """Raised when unable to connect to the service."""


class CLINotFoundError(CLIConnectionError):
    """Raised when the CLI is not found or not installed."""


class ProcessError(ClaudeSDKError):
    """Raised when the process fails."""


class CLIJSONDecodeError(ClaudeSDKError):
    """Raised when unable to decode JSON from output."""


class MessageParseError(ClaudeSDKError):
    """Raised when unable to parse a message from output."""


# =============================================================================
# Assistant Message Error Types
# =============================================================================

AssistantMessageError = Literal[
    "authentication_failed",
    "billing_error",
    "rate_limit",
    "invalid_request",
    "server_error",
    "unknown",
]


# =============================================================================
# Transport Abstract Base Class
# =============================================================================

from abc import ABC, abstractmethod


class Transport(ABC):
    """Abstract base class for transport implementations.

    Transport defines how the SDK communicates with the underlying service.
    Ripperdoc doesn't use subprocess transport like Claude SDK, but
    this class is provided for API compatibility.
    """

    @abstractmethod
    async def connect(self) -> None:
        """Establish the connection."""

    @abstractmethod
    async def write(self, data: str) -> None:
        """Write data to the transport."""

    @abstractmethod
    def read_messages(self) -> AsyncIterable[dict[str, Any]]:
        """Read messages from the transport."""

    @abstractmethod
    async def close(self) -> None:
        """Close the connection."""

    @abstractmethod
    def is_ready(self) -> bool:
        """Check if the transport is ready."""

    @abstractmethod
    async def end_input(self) -> None:
        """Signal that no more input will be sent."""

