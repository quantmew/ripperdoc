"""
MCP type definitions matching the reference services/mcp/types.ts.

Configuration schemas and types for MCP server management.
"""

from __future__ import annotations

from typing import Union

# MCP SDK availability
MCP_AVAILABLE = False

from dataclasses import dataclass, field  # noqa: E402
from enum import Enum  # noqa: E402
from typing import Any, Dict, List, Optional  # noqa: E402


class ConfigScope(str, Enum):
    """Scope of an MCP server configuration."""

    LOCAL = "local"
    USER = "user"
    PROJECT = "project"
    DYNAMIC = "dynamic"
    ENTERPRISE = "enterprise"
    CLAUDEAI = "claudeai"
    MANAGED = "managed"


class TransportType(str, Enum):
    """Transport protocol for MCP server connections."""

    STDIO = "stdio"
    SSE = "sse"
    SSE_IDE = "sse-ide"
    HTTP = "http"
    WS = "ws"
    SDK = "sdk"
    STREAMABLE_HTTP = "streamable-http"


@dataclass
class McpToolInfo:
    """Information about an MCP tool exposed by a server."""

    name: str
    description: str = ""
    input_schema: Optional[Dict[str, Any]] = None
    annotations: Dict[str, Any] = field(default_factory=dict)


@dataclass
class McpResourceInfo:
    """Information about an MCP resource exposed by a server."""

    uri: str
    name: Optional[str] = None
    description: str = ""
    mime_type: Optional[str] = None
    size: Optional[int] = None
    text: Optional[str] = None


@dataclass
class McpServerConfig:
    """Base MCP server configuration."""

    name: str
    type: TransportType = TransportType.STDIO
    scope: ConfigScope = ConfigScope.USER
    description: str = ""


@dataclass
class StdioMcpServerConfig(McpServerConfig):
    """Configuration for a stdio-based MCP server."""

    type: TransportType = TransportType.STDIO
    command: Optional[str] = None
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    stderr_mode: Optional[str] = None


@dataclass
class SSEMcpServerConfig(McpServerConfig):
    """Configuration for an SSE-based MCP server."""

    type: TransportType = TransportType.SSE
    url: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    headers_helper: Optional[str] = None


@dataclass
class HTTPMcpServerConfig(McpServerConfig):
    """Configuration for an HTTP/streamable-http MCP server."""

    type: TransportType = TransportType.HTTP
    url: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    headers_helper: Optional[str] = None


# Union type for any MCP server configuration
AnyMcpServerConfig = Union[StdioMcpServerConfig, SSEMcpServerConfig, HTTPMcpServerConfig]


@dataclass
class McpServerInfo:
    """Runtime information about an MCP server (connected state + config)."""

    name: str
    type: str = "stdio"
    scope: str = "user"
    url: Optional[str] = None
    description: str = ""
    command: Optional[str] = None
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    headers_helper: Optional[str] = None
    tools: List[McpToolInfo] = field(default_factory=list)
    resources: List[McpResourceInfo] = field(default_factory=list)
    status: str = "configured"
    error: Optional[str] = None
    instructions: Optional[str] = None
    server_version: Optional[str] = None
    capabilities: Dict[str, Any] = field(default_factory=dict)
    tools_discovered: bool = False


ScopedMcpServerConfig = Union[StdioMcpServerConfig, SSEMcpServerConfig, HTTPMcpServerConfig]
