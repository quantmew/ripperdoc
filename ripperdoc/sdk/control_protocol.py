"""JSON Control Protocol types for subprocess communication.

This module defines the message types used for communication between
the SDK and the CLI subprocess over stdio.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, NotRequired, TypedDict, Union


# =============================================================================
# Control Request Types (SDK → CLI)
# =============================================================================

class SDKControlInitializeRequest(TypedDict):
    """Initialize request - sent when SDK connects to CLI."""
    subtype: Literal["initialize"]
    options: dict[str, Any]  # ClaudeAgentOptions as dict
    hooks: dict[str, list[dict[str, Any]]] | None


class SDKControlQueryRequest(TypedDict):
    """Query request - send a query to the CLI."""
    subtype: Literal["query"]
    prompt: str
    session_id: str


class SDKControlPermissionRequest(TypedDict):
    """Permission request - ask if a tool can be used."""
    subtype: Literal["can_use_tool"]
    tool_name: str
    input: dict[str, Any]
    permission_suggestions: list[dict[str, Any]] | None
    blocked_path: str | None


class SDKControlInterruptRequest(TypedDict):
    """Interrupt request - interrupt the current query."""
    subtype: Literal["interrupt"]


class SDKControlSetPermissionModeRequest(TypedDict):
    """Set permission mode request."""
    subtype: Literal["set_permission_mode"]
    mode: str  # PermissionMode


class SDKControlSetModelRequest(TypedDict):
    """Set model request."""
    subtype: Literal["set_model"]
    model: str | None


class SDKControlRewindFilesRequest(TypedDict):
    """Rewind files request."""
    subtype: Literal["rewind_files"]
    user_message_id: str


class SDKHookCallbackRequest(TypedDict):
    """Hook callback request - execute a hook callback."""
    subtype: Literal["hook_callback"]
    callback_id: str
    input: dict[str, Any]
    tool_use_id: str | None


class SDKControlMcpMessageRequest(TypedDict):
    """MCP message request - send message to MCP server."""
    subtype: Literal["mcp_message"]
    server_name: str
    message: dict[str, Any]


# Union of all control request types
SDKControlRequest = TypedDict(
    "SDKControlRequest",
    {
        "type": Literal["control_request"],
        "request_id": str,
        "request": Union[
            SDKControlInitializeRequest,
            SDKControlQueryRequest,
            SDKControlPermissionRequest,
            SDKControlInterruptRequest,
            SDKControlSetPermissionModeRequest,
            SDKControlSetModelRequest,
            SDKControlRewindFilesRequest,
            SDKHookCallbackRequest,
            SDKControlMcpMessageRequest,
        ],
    },
)


# =============================================================================
# Control Response Types (CLI → SDK)
# =============================================================================

class ControlResponseSuccess(TypedDict):
    """Successful control response."""
    subtype: Literal["success"]
    request_id: str
    response: dict[str, Any] | None


class ControlResponseError(TypedDict):
    """Error control response."""
    subtype: Literal["error"]
    request_id: str
    error: str


ControlResponse = Union[ControlResponseSuccess, ControlResponseError]


class SDKControlResponse(TypedDict):
    """Control response wrapper."""
    type: Literal["control_response"]
    response: ControlResponse


# =============================================================================
# Stream Message Types (CLI → SDK)
# =============================================================================

class StreamUserMessage(TypedDict):
    """User message from CLI."""
    type: Literal["user"]
    message: dict[str, Any]
    uuid: NotRequired[str]
    parent_tool_use_id: NotRequired[str]
    tool_use_result: NotRequired[dict[str, Any]]
    session_id: str


class StreamAssistantMessage(TypedDict):
    """Assistant message from CLI."""
    type: Literal["assistant"]
    message: dict[str, Any]
    model: str
    parent_tool_use_id: NotRequired[str]
    error: NotRequired[str]
    session_id: str


class StreamProgressMessage(TypedDict):
    """Progress message from CLI."""
    type: Literal["progress"]
    tool_use_id: str
    content: Any
    session_id: str


class StreamResultMessage(TypedDict):
    """Result message from CLI."""
    type: Literal["result"]
    subtype: Literal["result"]
    duration_ms: int
    duration_api_ms: int
    is_error: bool
    num_turns: int
    session_id: str
    total_cost_usd: NotRequired[float]
    usage: NotRequired[dict[str, Any]]
    result: NotRequired[str]


class StreamSystemMessage(TypedDict):
    """System message from CLI."""
    type: Literal["system"]
    subtype: str
    data: dict[str, Any]


class StreamErrorMessage(TypedDict):
    """Error message from CLI."""
    type: Literal["error"]
    error: str


# Union of all stream message types
StreamMessage = Union[
    StreamUserMessage,
    StreamAssistantMessage,
    StreamProgressMessage,
    StreamResultMessage,
    StreamSystemMessage,
    StreamErrorMessage,
]


# =============================================================================
# Permission Update Types (used in control protocol)
# =============================================================================

PermissionUpdateDestination = Literal[
    "userSettings", "projectSettings", "localSettings", "session"
]

PermissionBehavior = Literal["allow", "deny", "ask"]


@dataclass
class PermissionRuleValue:
    """Permission rule value."""
    tool_name: str
    rule_content: str | None = None


@dataclass
class PermissionUpdate:
    """Permission update configuration."""
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
    mode: str | None = None  # PermissionMode
    directories: list[str] | None = None
    destination: PermissionUpdateDestination | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for JSON serialization."""
        result: dict[str, Any] = {"type": self.type}

        if self.destination is not None:
            result["destination"] = self.destination

        if self.type in ["addRules", "replaceRules", "removeRules"]:
            if self.rules is not None:
                result["rules"] = [
                    {
                        "toolName": rule.tool_name,
                        "ruleContent": rule.rule_content,
                    }
                    for rule in self.rules
                ]
            if self.behavior is not None:
                result["behavior"] = self.behavior

        elif self.type == "setMode":
            if self.mode is not None:
                result["mode"] = self.mode

        elif self.type in ["addDirectories", "removeDirectories"]:
            if self.directories is not None:
                result["directories"] = self.directories

        return result


# =============================================================================
# Hook Types (used in control protocol)
# =============================================================================

HookEvent = (
    Literal["PreToolUse"]
    | Literal["PostToolUse"]
    | Literal["UserPromptSubmit"]
    | Literal["Stop"]
    | Literal["SubagentStop"]
    | Literal["PreCompact"]
)


@dataclass
class HookMatcher:
    """Hook matcher configuration for control protocol."""
    matcher: str | None = None
    hooks: list[Any] = field(default_factory=list)  # HookCallback functions
    timeout: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for JSON serialization."""
        result: dict[str, Any] = {"matcher": self.matcher}
        if self.timeout is not None:
            result["timeout"] = self.timeout
        return result


# =============================================================================
# Server Info Types
# =============================================================================

@dataclass
class ServerInfo:
    """Server initialization information."""
    commands: list[dict[str, Any]] = field(default_factory=list)
    output_style: str = "default"
    version: str = "unknown"
    features: list[str] = field(default_factory=list)


__all__ = [
    # Control Requests
    "SDKControlRequest",
    "SDKControlInitializeRequest",
    "SDKControlQueryRequest",
    "SDKControlPermissionRequest",
    "SDKControlInterruptRequest",
    "SDKControlSetPermissionModeRequest",
    "SDKControlSetModelRequest",
    "SDKControlRewindFilesRequest",
    "SDKHookCallbackRequest",
    "SDKControlMcpMessageRequest",
    # Control Responses
    "SDKControlResponse",
    "ControlResponse",
    "ControlResponseSuccess",
    "ControlResponseError",
    # Stream Messages
    "StreamMessage",
    "StreamUserMessage",
    "StreamAssistantMessage",
    "StreamProgressMessage",
    "StreamResultMessage",
    "StreamSystemMessage",
    "StreamErrorMessage",
    # Permission
    "PermissionUpdate",
    "PermissionRuleValue",
    "PermissionUpdateDestination",
    "PermissionBehavior",
    # Hooks
    "HookEvent",
    "HookMatcher",
    # Server Info
    "ServerInfo",
]
