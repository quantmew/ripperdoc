"""Pydantic models for stdio protocol messages.

The protocol is now expressed as JSON-RPC 2.0 request/response envelopes with
`initialize` and `sampling/createMessage` flows aligned to Claude Code MCP.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

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

    # Claude-compatible optional fields
    server_tool_use: dict[str, int] = Field(
        default_factory=lambda: {"web_search_requests": 0, "web_fetch_requests": 0}
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

    Aligned with Claude Code SDK protocol for permission suggestions,
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
    # Helpers
    "model_to_dict",
]
