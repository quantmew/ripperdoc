"""Initialize request and response DTOs."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from ripperdoc import __version__
from ripperdoc.protocol.model_types.jsonrpc import DEFAULT_PROTOCOL_VERSION


class ProtocolCapabilities(BaseModel):
    """Server capability set returned in `initialize` result."""

    experimental: Optional[dict[str, Any]] = None
    sampling: Optional[dict[str, Any]] = None
    tools: Optional[dict[str, Any]] = Field(default_factory=lambda: {"listChanged": False})
    tasks: Optional[dict[str, Any]] = None
    logging: bool | Optional[dict[str, Any]] = None
    completions: bool | Optional[dict[str, Any]] = None
    prompts: Optional[dict[str, Any]] = None
    resources: Optional[dict[str, Any]] = None

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )


class InitializeClientIcon(BaseModel):
    """Client info metadata icon descriptor."""

    src: str
    mimeType: Optional[str] = None
    sizes: Optional[list[str]] = None
    theme: Optional[Literal["light", "dark"]] = None

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )


class InitializeClientInfo(BaseModel):
    """Client metadata from `initialize` request."""

    name: str
    title: Optional[str] = None
    version: str
    websiteUrl: Optional[str] = None
    description: Optional[str] = None
    icons: Optional[list[InitializeClientIcon]] = None

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )


class InitializeClientCapabilitiesSampling(BaseModel):
    """Client sampling capability descriptor."""

    context: Optional[Any] = None
    tools: Optional[Any] = None

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )


class InitializeClientCapabilitiesElicitation(BaseModel):
    """Client elicitation capability descriptor."""

    form: Optional[Any] = None
    url: Optional[Any] = None

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )


class InitializeClientCapabilitiesTasksSampling(BaseModel):
    """Client task/sampling capability descriptor."""

    createMessage: Optional[Any] = None

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )


class InitializeClientCapabilitiesTasksRequests(BaseModel):
    """Client task request capability descriptors."""

    sampling: Optional[InitializeClientCapabilitiesTasksSampling] = None
    elicitation: Optional[dict[str, Any]] = None

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )


class InitializeClientCapabilitiesTasks(BaseModel):
    """Client task capability descriptor."""

    list: Optional[Any] = None
    cancel: Optional[Any] = None
    requests: Optional[InitializeClientCapabilitiesTasksRequests] = None

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )


class InitializeClientCapabilitiesRoots(BaseModel):
    """Client roots capability descriptor."""

    listChanged: Optional[bool] = None

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )


class InitializeClientCapabilities(BaseModel):
    """Client capability shape expected by `initialize`."""

    experimental: Optional[dict[str, Any]] = None
    sampling: Optional[InitializeClientCapabilitiesSampling] = None
    elicitation: Optional[InitializeClientCapabilitiesElicitation] = None
    roots: Optional[InitializeClientCapabilitiesRoots] = None
    tasks: Optional[InitializeClientCapabilitiesTasks] = None

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )


class InitializeServerInfo(BaseModel):
    """Server metadata returned from `initialize` response."""

    name: str = "ripperdoc"
    title: str = "Ripperdoc"
    version: str = __version__
    websiteUrl: Optional[str] = None
    description: Optional[str] = None

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )


class InitializeResult(BaseModel):
    """Result shape for JSON-RPC `initialize`."""

    protocolVersion: str = DEFAULT_PROTOCOL_VERSION
    capabilities: ProtocolCapabilities
    serverInfo: InitializeServerInfo
    instructions: Optional[str] = None


class InitializeParams(BaseModel):
    """Expected parameters for JSON-RPC `initialize`."""

    protocolVersion: str
    capabilities: InitializeClientCapabilities
    clientInfo: InitializeClientInfo
    meta: Optional[dict[str, Any]] = Field(default=None, alias="_meta")

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        protected_namespaces=(),
    )


__all__ = [
    "ProtocolCapabilities",
    "InitializeClientIcon",
    "InitializeClientInfo",
    "InitializeClientCapabilitiesSampling",
    "InitializeClientCapabilitiesElicitation",
    "InitializeClientCapabilitiesTasksSampling",
    "InitializeClientCapabilitiesTasksRequests",
    "InitializeClientCapabilitiesTasks",
    "InitializeClientCapabilitiesRoots",
    "InitializeClientCapabilities",
    "InitializeServerInfo",
    "InitializeResult",
    "InitializeParams",
]
