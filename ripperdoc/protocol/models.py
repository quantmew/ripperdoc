"""Pydantic models for stdio protocol messages.

The protocol is now expressed as JSON-RPC 2.0 request/response envelopes with
`initialize` and `sampling/createMessage` flows aligned to MCP-style clients.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

from ripperdoc import __version__

DEFAULT_PROTOCOL_VERSION = "2025-11-25"


class JsonRpcErrorCodes(IntEnum):
    """Subset of JSON-RPC error codes used by the protocol."""

    ConnectionClosed = -32000
    RequestTimeout = -32001
    ParseError = -32700
    InvalidRequest = -32600
    MethodNotFound = -32601
    InvalidParams = -32602
    InternalError = -32603
    UrlElicitationRequired = -32042


# ==========================================================================
# JSON-RPC Transport Models
# ==========================================================================


class JsonRpcError(BaseModel):
    """JSON-RPC error envelope payload."""

    code: int
    message: str
    data: Any | None = None


class JsonRpcResponse(BaseModel):
    """JSON-RPC success/error response for an in-flight request."""

    jsonrpc: str = "2.0"
    id: str | int
    result: Any | None = None
    error: JsonRpcError | None = None

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )


class JsonRpcResponseError(Exception):
    """Typed exception for raising JSON-RPC style errors from awaited calls."""

    def __init__(
        self,
        code: int,
        message: str,
        data: Any | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


# ============================================================================
# Content Block / Stream Message Models
# ============================================================================


class ContentBlock(BaseModel):
    """Base class for message content blocks."""

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )


class TextContentBlock(ContentBlock):
    """A text content block."""

    type: Literal["text"] = "text"
    text: str


class ThinkingContentBlock(ContentBlock):
    """A thinking/reasoning content block."""

    type: str = Field(default="thinking")
    thinking: str = Field(alias="text")
    signature: str | None = None


class ToolUseContentBlock(ContentBlock):
    """A tool call content block."""

    type: str = Field(default="tool_use")
    id: str = Field(default="")
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class ToolResultContentBlock(ContentBlock):
    """A tool result content block."""

    type: str = Field(default="tool_result")
    tool_use_id: str = Field(default="")
    content: Any = Field(default="")
    is_error: bool | None = None


class ImageSource(BaseModel):
    """Image source data."""

    type: str = Field(default="base64")
    media_type: str = Field(default="image/jpeg")
    data: str


class ImageContentBlock(ContentBlock):
    """An image content block."""

    type: str = Field(default="image")
    source: ImageSource


# Union type for all content blocks
ContentBlockType = (
    TextContentBlock
    | ThinkingContentBlock
    | ToolUseContentBlock
    | ToolResultContentBlock
    | ImageContentBlock
)


class MessageData(BaseModel):
    """Base message data."""

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )


class AssistantMessageData(MessageData):
    """Assistant message data."""

    role: str = "assistant"
    content: list[dict[str, Any]] | str
    model: str = "main"


class UserMessageData(MessageData):
    """User message data."""

    role: str = "user"
    content: list[dict[str, Any]] | str = ""


class AssistantStreamMessage(BaseModel):
    """An assistant message sent to SDK stream output."""

    type: str = Field(default="assistant")
    message: AssistantMessageData
    session_id: str | None = None
    parent_tool_use_id: str | None = None
    uuid: str | None = None


class UserStreamMessage(BaseModel):
    """A user message sent to SDK stream output."""

    type: str = Field(default="user")
    message: UserMessageData
    uuid: str | None = None
    session_id: str | None = None
    parent_tool_use_id: str | None = None
    tool_use_result: Any = None


class IncomingUserMessageData(BaseModel):
    """Validated incoming user message data from user-facing stream input."""

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )

    role: Literal["user"]
    content: list[dict[str, Any]] | str = ""


class IncomingUserStreamMessage(BaseModel):
    """Validated incoming `type=user` message from stream input."""

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )

    type: Literal["user"]
    message: IncomingUserMessageData
    uuid: str | None = None
    session_id: str | None = None
    parent_tool_use_id: str | None = None
    tool_use_result: Any = None


# Union type for stream messages
StreamMessage = AssistantStreamMessage | UserStreamMessage


class MCPServerInfo(BaseModel):
    """MCP server information."""

    name: str


class MCPServerStatusInfo(BaseModel):
    """MCP server status information."""

    name: str
    status: str


class ProtocolCapabilities(BaseModel):
    """Server capability set returned in `initialize` result."""

    experimental: dict[str, Any] | None = None
    sampling: dict[str, Any] | None = None
    tools: dict[str, Any] | None = Field(
        default_factory=lambda: {"listChanged": False}
    )
    tasks: dict[str, Any] | None = None
    logging: bool | dict[str, Any] | None = None
    completions: bool | dict[str, Any] | None = None
    prompts: dict[str, Any] | None = None
    resources: dict[str, Any] | None = None

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )


class InitializeClientIcon(BaseModel):
    """Client info metadata icon descriptor."""

    src: str
    mimeType: str | None = None
    sizes: list[str] | None = None
    theme: Literal["light", "dark"] | None = None

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )


class InitializeClientInfo(BaseModel):
    """Client metadata from `initialize` request."""

    name: str
    title: str | None = None
    version: str
    websiteUrl: str | None = None
    description: str | None = None
    icons: list[InitializeClientIcon] | None = None

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )


class InitializeClientCapabilitiesSampling(BaseModel):
    """Client sampling capability descriptor."""

    context: Any | None = None
    tools: Any | None = None

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )


class InitializeClientCapabilitiesElicitation(BaseModel):
    """Client elicitation capability descriptor."""

    form: Any | None = None
    url: Any | None = None

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )


class InitializeClientCapabilitiesTasksSampling(BaseModel):
    """Client task/sampling capability descriptor."""

    createMessage: Any | None = None

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )


class InitializeClientCapabilitiesTasksRequests(BaseModel):
    """Client task request capability descriptors."""

    sampling: InitializeClientCapabilitiesTasksSampling | None = None
    elicitation: dict[str, Any] | None = None

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )


class InitializeClientCapabilitiesTasks(BaseModel):
    """Client task capability descriptor."""

    list: Any | None = None
    cancel: Any | None = None
    requests: InitializeClientCapabilitiesTasksRequests | None = None

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )


class InitializeClientCapabilitiesRoots(BaseModel):
    """Client roots capability descriptor."""

    listChanged: bool | None = None

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )


class InitializeClientCapabilities(BaseModel):
    """Client capability shape expected by `initialize`."""

    experimental: dict[str, Any] | None = None
    sampling: InitializeClientCapabilitiesSampling | None = None
    elicitation: InitializeClientCapabilitiesElicitation | None = None
    roots: InitializeClientCapabilitiesRoots | None = None
    tasks: InitializeClientCapabilitiesTasks | None = None

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )


class InitializeServerInfo(BaseModel):
    """Server metadata returned from `initialize` response."""

    name: str = "ripperdoc"
    title: str = "Ripperdoc"
    version: str = __version__
    websiteUrl: str | None = None
    description: str | None = None

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )


class InitializeResult(BaseModel):
    """Result shape for JSON-RPC `initialize`."""

    protocolVersion: str = DEFAULT_PROTOCOL_VERSION
    capabilities: ProtocolCapabilities
    serverInfo: InitializeServerInfo
    instructions: str | None = None


class InitializeParams(BaseModel):
    """Expected parameters for JSON-RPC `initialize`."""

    protocolVersion: str
    capabilities: InitializeClientCapabilities
    clientInfo: InitializeClientInfo
    meta: dict[str, Any] | None = Field(default=None, alias="_meta")

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        protected_namespaces=(),
    )


class UsageInfo(BaseModel):
    """Token usage information."""

    input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    output_tokens: int = 0

    # SDK-compatible optional fields
    server_tool_use: dict[str, int] = Field(
        default_factory=lambda: {}
    )
    service_tier: str = "standard"
    cache_creation: dict[str, int] = Field(
        default_factory=lambda: {
            "ephemeral_1h_input_tokens": 0,
            "ephemeral_5m_input_tokens": 0,
        }
    )


class SamplingRequestMessage(BaseModel):
    """Single message in a sampling/createMessage request."""

    role: Literal["user", "assistant"]
    content: list[dict[str, Any]] | str
    meta: dict[str, Any] | None = Field(default=None, alias="_meta")

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        protected_namespaces=(),
    )


class SamplingRequest(BaseModel):
    """Request body for `sampling/createMessage`."""

    messages: list[SamplingRequestMessage]
    modelPreferences: dict[str, Any] | None = None
    systemPrompt: str | None = None
    appendSystemPrompt: str | None = None
    includeContext: Literal["none", "thisServer", "allServers"] | None = None
    temperature: float | None = None
    maxTokens: int
    stopSequences: list[str] | None = None
    metadata: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    toolChoice: dict[str, Any] | None = None
    meta: dict[str, Any] | None = Field(default=None, alias="_meta")

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        protected_namespaces=(),
    )


class SamplingStopReason(str):
    """Stop reason enum values used by MCP sampling result."""


class SamplingResult(BaseModel):
    """Result payload for `sampling/createMessage`."""

    model: str
    stopReason: Literal[
        "endTurn",
        "stopSequence",
        "maxTokens",
        "toolUse",
    ] | str = "endTurn"
    role: Literal["assistant"] = "assistant"
    content: list[dict[str, Any]] | dict[str, Any] | str
    usage: UsageInfo | None = None
    meta: dict[str, Any] | None = Field(default=None, alias="_meta")

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        protected_namespaces=(),
    )


class ToolCallRequest(BaseModel):
    """Request payload for MCP-style tool invocations."""

    name: str
    arguments: dict[str, Any] | None = None
    meta: dict[str, Any] | None = Field(default=None, alias="_meta")

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        protected_namespaces=(),
    )


# ============================================================================
# Permission Response Models (internal compatibility only)
# ============================================================================


class PermissionResponseAllow(BaseModel):
    """A permission allow response."""

    behavior: str = Field(default="allow")
    updatedInput: dict[str, Any] | None = None
    toolUseID: str | None = None
    decisionReason: dict[str, Any] | None = None
    updatedPermissions: list[dict[str, Any]] | None = None


class PermissionResponseDeny(BaseModel):
    """A permission deny response."""

    behavior: str = Field(default="deny")
    message: str = ""
    toolUseID: str | None = None
    decisionReason: dict[str, Any] | None = None


class PermissionRequestPayload(BaseModel):
    """Payload for SDK can_use_tool permission requests.

    Aligned with SDK protocol conventions for permission suggestions,
    blocked path, and decision reason tracking.
    """

    subtype: str = Field(default="can_use_tool")
    tool_name: str
    input: dict[str, Any] | None = None
    tool_use_id: str | None = None
    agent_id: str | None = None
    permission_suggestions: list[dict[str, Any]] | None = None
    blocked_path: str | None = None
    decision_reason: dict[str, Any] | None = None
    force_prompt: bool = False


# ============================================================================
# SDK Session Metadata Models
# ============================================================================


class FastModeState(str):
    """Fast mode state values."""

    OFF = "off"
    COOLDOWN = "cooldown"
    ON = "on"


class ThinkingConfig(BaseModel):
    """Thinking/reasoning configuration."""

    mode: Literal["adaptive", "enabled", "disabled"] = "adaptive"
    budget_tokens: int | None = Field(default=None, alias="budgetTokens")

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )


class ModelInfo(BaseModel):
    """Model metadata for SDK initialize response."""

    value: str
    display_name: str | None = Field(default=None, alias="displayName")
    description: str | None = None
    supports_effort: bool = Field(default=False, alias="supportsEffort")
    supported_effort_levels: list[str] | None = Field(
        default=None, alias="supportedEffortLevels"
    )
    supports_adaptive_thinking: bool = Field(
        default=False, alias="supportsAdaptiveThinking"
    )
    supports_fast_mode: bool = Field(default=False, alias="supportsFastMode")
    supports_auto_mode: bool = Field(default=False, alias="supportsAutoMode")
    max_tokens: int | None = Field(default=None, alias="maxTokens")
    max_thinking_tokens: int | None = Field(
        default=None, alias="maxThinkingTokens"
    )

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )


class AccountInfo(BaseModel):
    """Account metadata for SDK initialize response."""

    email: str | None = None
    organization: str | None = None
    subscription_type: str | None = Field(default=None, alias="subscriptionType")
    token_source: str | None = Field(default=None, alias="tokenSource")
    api_key_source: str | None = Field(default=None, alias="apiKeySource")
    api_provider: str | None = Field(default=None, alias="apiProvider")

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )


class SlashCommandInfo(BaseModel):
    """Slash command metadata for SDK responses."""

    name: str
    description: str = ""
    argument_hint: str = Field(default="", alias="argumentHint")

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )


class SdkBeta(str):
    """SDK beta capability identifiers."""

    CONTEXT_1M = "context-1m-2025-08-07"


# ============================================================================
# Permission Update Models
# ============================================================================


class PermissionRuleValue(BaseModel):
    """A single permission rule with tool name and content."""

    tool_name: str = Field(alias="toolName")
    rule_content: str | None = Field(default=None, alias="ruleContent")

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )


class PermissionUpdateDestination(str):
    """Where to apply a permission update."""

    USER_SETTINGS = "userSettings"
    PROJECT_SETTINGS = "projectSettings"
    LOCAL_SETTINGS = "localSettings"
    SESSION = "session"
    CLI_ARG = "cliArg"

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.str_schema()


class PermissionDecisionClassification(str):
    """Classification of a permission decision."""

    USER_TEMPORARY = "user_temporary"
    USER_PERMANENT = "user_permanent"
    USER_REJECT = "user_reject"


class PermissionUpdateAddRules(BaseModel):
    """Permission update: add rules."""

    type: Literal["addRules"] = "addRules"
    rules: list[PermissionRuleValue]
    behavior: Literal["allow", "deny", "ask"] = "allow"
    destination: PermissionUpdateDestination | None = None

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )


class PermissionUpdateReplaceRules(BaseModel):
    """Permission update: replace all rules."""

    type: Literal["replaceRules"] = "replaceRules"
    rules: list[PermissionRuleValue]
    behavior: Literal["allow", "deny", "ask"] = "allow"
    destination: PermissionUpdateDestination | None = None

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )


class PermissionUpdateRemoveRules(BaseModel):
    """Permission update: remove rules."""

    type: Literal["removeRules"] = "removeRules"
    rules: list[PermissionRuleValue]
    behavior: Literal["allow", "deny", "ask"] = "allow"
    destination: PermissionUpdateDestination | None = None

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )


class PermissionUpdateSetMode(BaseModel):
    """Permission update: set permission mode."""

    type: Literal["setMode"] = "setMode"
    mode: str
    destination: PermissionUpdateDestination | None = None

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )


class PermissionUpdateAddDirectories(BaseModel):
    """Permission update: add working directories."""

    type: Literal["addDirectories"] = "addDirectories"
    directories: list[str]
    destination: PermissionUpdateDestination | None = None

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )


class PermissionUpdateRemoveDirectories(BaseModel):
    """Permission update: remove working directories."""

    type: Literal["removeDirectories"] = "removeDirectories"
    directories: list[str]
    destination: PermissionUpdateDestination | None = None

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )


PermissionUpdate = (
    PermissionUpdateAddRules
    | PermissionUpdateReplaceRules
    | PermissionUpdateRemoveRules
    | PermissionUpdateSetMode
    | PermissionUpdateAddDirectories
    | PermissionUpdateRemoveDirectories
)


# ============================================================================
# Enhanced MCP Status Models
# ============================================================================


class McpToolAnnotation(BaseModel):
    """MCP tool annotation metadata."""

    title: str | None = None
    read_only_hint: bool | None = Field(default=None, alias="readOnlyHint")
    destructive_hint: bool | None = Field(default=None, alias="destructiveHint")
    idempotent_hint: bool | None = Field(default=None, alias="idempotentHint")
    open_world_hint: bool | None = Field(default=None, alias="openWorldHint")

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )


class McpToolInfo(BaseModel):
    """MCP tool with annotations."""

    name: str
    description: str | None = None
    annotations: McpToolAnnotation | None = None

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )


class McpServerCapabilities(BaseModel):
    """MCP server capability set."""

    tools: dict[str, Any] | None = None
    resources: dict[str, Any] | None = None
    prompts: dict[str, Any] | None = None
    experimental: dict[str, Any] | None = None

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )


class McpServerStatusDetail(BaseModel):
    """Enhanced MCP server status with full metadata."""

    name: str
    status: str
    type: str | None = None
    error: str | None = None
    server_info: dict[str, Any] | None = Field(default=None, alias="serverInfo")
    config: dict[str, Any] | None = None
    tools: list[McpToolInfo] = Field(default_factory=list)
    resources: int = 0
    capabilities: McpServerCapabilities | None = None
    scope: str | None = None

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )


# ============================================================================
# Context Usage Models
# ============================================================================


class ContextCategory(BaseModel):
    """A category of context window usage."""

    name: str
    tokens: int
    color: str | None = None
    is_deferred: bool = Field(default=False, alias="isDeferred")

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )


class ContextGridSquare(BaseModel):
    """A single square in the context usage grid."""

    color: str | None = None
    is_filled: bool = Field(default=False, alias="isFilled")
    category_name: str | None = Field(default=None, alias="categoryName")
    tokens: int = 0
    percentage: float = 0.0
    square_fullness: float = Field(default=0.0, alias="squareFullness")

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )


class ContextUsageResult(BaseModel):
    """Full context usage breakdown returned by get_context_usage."""

    max_tokens: int = Field(alias="maxTokens")
    used_tokens: int = Field(alias="usedTokens")
    free_tokens: int = Field(alias="freeTokens")
    percent_used: float = Field(alias="percentUsed")
    categories: list[ContextCategory] = Field(default_factory=list)
    grid: list[ContextGridSquare] = Field(default_factory=list)
    auto_compact_enabled: bool = Field(default=False, alias="autoCompactEnabled")
    message_count: int = Field(default=0, alias="messageCount")

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )


# ============================================================================
# SDK Stream Message Types
# ============================================================================


class SDKResultMessage(BaseModel):
    """End-of-turn result with cost/usage summary."""

    type: Literal["result"] = "result"
    subtype: str = "success"
    duration_ms: int = Field(default=0, alias="durationMs")
    duration_api_ms: int = Field(default=0, alias="durationApiMs")
    is_error: bool = Field(default=False, alias="isError")
    result: str = ""
    num_turns: int = Field(default=0, alias="numTurns")
    session_id: str | None = Field(default=None, alias="sessionId")
    stop_reason: str | None = Field(default="endTurn", alias="stopReason")
    total_cost_usd: float = Field(default=0.0, alias="totalCostUsd")
    usage: UsageInfo | None = None
    model_usage: dict[str, Any] | None = Field(default=None, alias="modelUsage")
    permission_denials: list[dict[str, Any]] | None = Field(
        default=None, alias="permissionDenials"
    )
    structured_output: Any | None = Field(default=None, alias="structuredOutput")
    uuid: str | None = None

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )


class SDKStatusMessage(BaseModel):
    """Status update message (tool progress, etc.)."""

    type: Literal["status"] = "status"
    subtype: str = ""
    message: str = ""
    tool_use_id: str | None = Field(default=None, alias="toolUseID")
    progress: float | None = None
    session_id: str | None = Field(default=None, alias="sessionId")

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )


class SDKToolProgressMessage(BaseModel):
    """Tool execution progress update."""

    type: Literal["tool_progress"] = "tool_progress"
    tool_use_id: str = Field(alias="toolUseID")
    progress: float | None = None
    message: str | None = None
    session_id: str | None = Field(default=None, alias="sessionId")

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )


class SDKAuthStatusMessage(BaseModel):
    """Auth status change notification."""

    type: Literal["auth_status"] = "auth_status"
    authenticated: bool = False
    provider: str | None = None
    message: str | None = None
    session_id: str | None = Field(default=None, alias="sessionId")

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )


class SDKPromptSuggestionMessage(BaseModel):
    """Prompt suggestion from the server."""

    type: Literal["prompt_suggestion"] = "prompt_suggestion"
    suggestions: list[str] = Field(default_factory=list)
    session_id: str | None = Field(default=None, alias="sessionId")

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )


# ============================================================================
# Helpers
# ============================================================================


def model_to_dict(model: BaseModel) -> dict[str, Any]:
    """Convert a pydantic model to JSON-serializable dict."""

    return model.model_dump(exclude_none=True, by_alias=True, mode="json")


__all__ = [
    # Protocol
    "JsonRpcErrorCodes",
    "JsonRpcError",
    "JsonRpcResponse",
    "JsonRpcResponseError",
    # Content Blocks
    "ContentBlock",
    "TextContentBlock",
    "ThinkingContentBlock",
    "ToolUseContentBlock",
    "ToolResultContentBlock",
    "ImageSource",
    "ImageContentBlock",
    "ContentBlockType",
    # Stream Messages
    "MessageData",
    "AssistantMessageData",
    "UserMessageData",
    "AssistantStreamMessage",
    "UserStreamMessage",
    "StreamMessage",
    "IncomingUserMessageData",
    "IncomingUserStreamMessage",
    # MCP
    "MCPServerInfo",
    "MCPServerStatusInfo",
    "McpToolAnnotation",
    "McpToolInfo",
    "McpServerCapabilities",
    "McpServerStatusDetail",
    # Initialize
    "ProtocolCapabilities",
    "InitializeClientIcon",
    "InitializeClientInfo",
    "InitializeClientCapabilities",
    "InitializeServerInfo",
    "InitializeResult",
    "InitializeParams",
    # Sampling
    "SamplingRequestMessage",
    "SamplingRequest",
    "SamplingResult",
    # Usage / Permissions
    "UsageInfo",
    "PermissionResponseAllow",
    "PermissionResponseDeny",
    "PermissionRequestPayload",
    "ToolCallRequest",
    "PermissionRuleValue",
    "PermissionUpdateDestination",
    "PermissionDecisionClassification",
    "PermissionUpdateAddRules",
    "PermissionUpdateReplaceRules",
    "PermissionUpdateRemoveRules",
    "PermissionUpdateSetMode",
    "PermissionUpdateAddDirectories",
    "PermissionUpdateRemoveDirectories",
    "PermissionUpdate",
    # SDK Session Metadata
    "FastModeState",
    "ThinkingConfig",
    "ModelInfo",
    "AccountInfo",
    "SlashCommandInfo",
    "SdkBeta",
    # Context Usage
    "ContextCategory",
    "ContextGridSquare",
    "ContextUsageResult",
    # SDK Stream Messages
    "SDKResultMessage",
    "SDKStatusMessage",
    "SDKToolProgressMessage",
    "SDKAuthStatusMessage",
    "SDKPromptSuggestionMessage",
    # Helpers
    "model_to_dict",
]
