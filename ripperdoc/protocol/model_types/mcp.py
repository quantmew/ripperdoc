"""MCP protocol status DTOs."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class MCPServerInfo(BaseModel):
    """MCP server information."""

    name: str


class MCPServerStatusInfo(BaseModel):
    """MCP server status information."""

    name: str
    status: str


class McpToolAnnotation(BaseModel):
    """MCP tool annotation metadata."""

    title: Optional[str] = None
    read_only_hint: Optional[bool] = Field(default=None, alias="readOnlyHint")
    destructive_hint: Optional[bool] = Field(default=None, alias="destructiveHint")
    idempotent_hint: Optional[bool] = Field(default=None, alias="idempotentHint")
    open_world_hint: Optional[bool] = Field(default=None, alias="openWorldHint")

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )


class McpToolInfo(BaseModel):
    """MCP tool with annotations."""

    name: str
    description: Optional[str] = None
    annotations: Optional[McpToolAnnotation] = None

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )


class McpServerCapabilities(BaseModel):
    """MCP server capability set."""

    tools: Optional[dict[str, Any]] = None
    resources: Optional[dict[str, Any]] = None
    prompts: Optional[dict[str, Any]] = None
    experimental: Optional[dict[str, Any]] = None

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )


class McpServerStatusDetail(BaseModel):
    """Enhanced MCP server status with full metadata."""

    name: str
    status: str
    type: Optional[str] = None
    error: Optional[str] = None
    server_info: Optional[dict[str, Any]] = Field(default=None, alias="serverInfo")
    config: Optional[dict[str, Any]] = None
    tools: list[McpToolInfo] = Field(default_factory=list)
    resources: int = 0
    capabilities: Optional[McpServerCapabilities] = None
    scope: Optional[str] = None

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )


__all__ = [
    "MCPServerInfo",
    "MCPServerStatusInfo",
    "McpToolAnnotation",
    "McpToolInfo",
    "McpServerCapabilities",
    "McpServerStatusDetail",
]
