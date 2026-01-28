"""Claude Agent SDK compatible Python SDK for Ripperdoc.

This SDK provides Claude Agent SDK compatible interfaces while using
Ripperdoc's internal implementation.
"""

from ripperdoc.sdk.client import (
    # Core client and options
    query,
    RipperdocSDKClient,
    ClaudeAgentOptions,
    # Compatibility aliases
    ClaudeSDKClient,
    RipperdocClient,
    RipperdocOptions,
    # Types from client module
    AgentConfig,
    HookCallback,
    HookMatcher,
    McpServerConfig,
    PermissionMode,
    SettingSource,
    StderrCallback,
    clear_programmatic_registries,
    get_programmatic_agents,
    get_programmatic_hooks,
)

from ripperdoc.sdk.types import (
    # Main exports
    Message as _Message,
    # Message types
    UserMessage as _UserMessage,
    AssistantMessage as _AssistantMessage,
    SystemMessage as _SystemMessage,
    ResultMessage as _ResultMessage,
    StreamEvent as _StreamEvent,
    # Content blocks
    ContentBlock as _ContentBlock,
    TextBlock as _TextBlock,
    ThinkingBlock as _ThinkingBlock,
    ToolUseBlock as _ToolUseBlock,
    ToolResultBlock as _ToolResultBlock,
    # Permission types (already imported from client, skip)
    PermissionUpdate,
    PermissionResult,
    PermissionResultAllow,
    PermissionResultDeny,
    # Tool callback types
    CanUseTool,
    ToolPermissionContext,
    # Hook support
    HookContext,
    HookInput,
    BaseHookInput,
    PreToolUseHookInput,
    PostToolUseHookInput,
    UserPromptSubmitHookInput,
    StopHookInput,
    SubagentStopHookInput,
    PreCompactHookInput,
    HookJSONOutput,
    HookEvent,
    # Agent support
    AgentDefinition,
    # MCP Server Support (already imported from client, skip)
    # Beta support
    SdkBeta,
    # Sandbox support
    SandboxSettings,
    SandboxNetworkConfig,
    SandboxIgnoreViolations,
    # Plugin support
    SdkPluginConfig,
    # System prompt types
    SystemPromptPreset,
    ToolsPreset,
    # Error types
    ClaudeSDKError,
    CLIConnectionError,
    CLINotFoundError,
    ProcessError,
    CLIJSONDecodeError,
    MessageParseError,
    # Transport
    Transport,
    # Assistant message error types
    AssistantMessageError,
)

__version__ = "0.2.10"

# Re-export types with their original names
Message = _Message
UserMessage = _UserMessage
AssistantMessage = _AssistantMessage
SystemMessage = _SystemMessage
ResultMessage = _ResultMessage
StreamEvent = _StreamEvent
ContentBlock = _ContentBlock
TextBlock = _TextBlock
ThinkingBlock = _ThinkingBlock
ToolUseBlock = _ToolUseBlock
ToolResultBlock = _ToolResultBlock

__all__ = [
    # Main exports
    "query",
    "__version__",
    # Transport
    "Transport",
    # Client and Options
    "RipperdocSDKClient",
    "ClaudeAgentOptions",
    # Compatibility aliases
    "ClaudeSDKClient",
    "RipperdocClient",
    "RipperdocOptions",
    # Types - Core
    "Message",
    "UserMessage",
    "AssistantMessage",
    "SystemMessage",
    "ResultMessage",
    "StreamEvent",
    # Types - Content Blocks
    "ContentBlock",
    "TextBlock",
    "ThinkingBlock",
    "ToolUseBlock",
    "ToolResultBlock",
    # Types - Permissions
    "PermissionMode",
    "McpServerConfig",
    "PermissionUpdate",
    "PermissionResult",
    "PermissionResultAllow",
    "PermissionResultDeny",
    # Types - Tool Callbacks
    "CanUseTool",
    "ToolPermissionContext",
    # Types - Hooks
    "HookCallback",
    "HookContext",
    "HookInput",
    "BaseHookInput",
    "PreToolUseHookInput",
    "PostToolUseHookInput",
    "UserPromptSubmitHookInput",
    "StopHookInput",
    "SubagentStopHookInput",
    "PreCompactHookInput",
    "HookJSONOutput",
    "HookMatcher",
    "HookEvent",
    # Types - Agents
    "AgentDefinition",
    "SettingSource",
    # Types - Beta and Plugins
    "SdkBeta",
    "SdkPluginConfig",
    # Types - Sandbox
    "SandboxSettings",
    "SandboxNetworkConfig",
    "SandboxIgnoreViolations",
    # Types - System Prompt
    "SystemPromptPreset",
    "ToolsPreset",
    # Types - Errors
    "ClaudeSDKError",
    "CLIConnectionError",
    "CLINotFoundError",
    "ProcessError",
    "CLIJSONDecodeError",
    "MessageParseError",
    # Types - Assistant Message Errors
    "AssistantMessageError",
    # Legacy types (for backward compatibility)
    "AgentConfig",
    "StderrCallback",
    # Registry access
    "get_programmatic_agents",
    "get_programmatic_hooks",
    "clear_programmatic_registries",
]
