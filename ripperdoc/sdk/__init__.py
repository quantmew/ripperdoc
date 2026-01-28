"""Claude Agent SDK compatible Python SDK for Ripperdoc.

This SDK provides Claude Agent SDK compatible interfaces while using
Ripperdoc's internal implementation.

The SDK supports two communication modes:
1. In-process mode (default): Direct Python implementation
2. Subprocess mode: Communication via JSON Control Protocol over stdio

To use subprocess mode, set `use_subprocess=True` in ClaudeAgentOptions:

    ```python
    from ripperdoc.sdk import query, ClaudeAgentOptions

    async for message in query(
        prompt="Hello!",
        options=ClaudeAgentOptions(use_subprocess=True)
    ):
        print(message)
    ```

Subprocess mode enables:
- Multi-language SDK support (future)
- Process isolation
- Better resource management
- Consistent protocol across all SDK implementations
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

# Transport layer
from ripperdoc.sdk.transport import (
    Transport,
    StdioTransport,
    StdioTransportConfig,
    InProcessTransport,
)

# Control protocol types (for subprocess communication)
from ripperdoc.sdk.control_protocol import (
    SDKControlRequest,
    SDKControlResponse,
    StreamMessage,
    PermissionUpdate as ControlPermissionUpdate,
    ServerInfo,
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
    "StdioTransport",
    "StdioTransportConfig",
    "InProcessTransport",
    # Control Protocol
    "SDKControlRequest",
    "SDKControlResponse",
    "StreamMessage",
    "PermissionUpdate",
    "ServerInfo",
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
