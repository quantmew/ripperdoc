"""Sampling/createMessage protocol DTOs."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class UsageInfo(BaseModel):
    """Token usage information."""

    input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    output_tokens: int = 0

    server_tool_use: dict[str, int] = Field(default_factory=lambda: {})
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
    content: Union[list[dict[str, Any]], str]
    meta: Optional[dict[str, Any]] = Field(default=None, alias="_meta")

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        protected_namespaces=(),
    )


class SamplingRequest(BaseModel):
    """Request body for `sampling/createMessage`."""

    messages: list[SamplingRequestMessage]
    modelPreferences: Optional[dict[str, Any]] = None
    systemPrompt: Optional[str] = None
    appendSystemPrompt: Optional[str] = None
    includeContext: Optional[Literal["none", "thisServer", "allServers"]] = None
    temperature: Optional[float] = None
    maxTokens: int
    stopSequences: Optional[list[str]] = None
    metadata: Optional[dict[str, Any]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    toolChoice: Optional[dict[str, Any]] = None
    meta: Optional[dict[str, Any]] = Field(default=None, alias="_meta")

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
    content: Union[list[dict[str, Any]], dict[str, Any], str]
    usage: Optional[UsageInfo] = None
    meta: Optional[dict[str, Any]] = Field(default=None, alias="_meta")

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        protected_namespaces=(),
    )


__all__ = [
    "UsageInfo",
    "SamplingRequestMessage",
    "SamplingRequest",
    "SamplingStopReason",
    "SamplingResult",
]
