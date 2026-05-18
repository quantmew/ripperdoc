"""SDK-facing protocol message DTOs."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from ripperdoc.protocol.model_types.sampling import UsageInfo


class FastModeState(str):
    """Fast mode state values."""

    OFF = "off"
    COOLDOWN = "cooldown"
    ON = "on"


class ThinkingConfig(BaseModel):
    """Thinking/reasoning configuration."""

    mode: Literal["adaptive", "enabled", "disabled"] = "adaptive"
    budget_tokens: Optional[int] = Field(default=None, alias="budgetTokens")

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )


class ModelInfo(BaseModel):
    """Model metadata for SDK initialize response."""

    value: str
    display_name: Optional[str] = Field(default=None, alias="displayName")
    description: Optional[str] = None
    supports_effort: bool = Field(default=False, alias="supportsEffort")
    supported_effort_levels: Optional[list[str]] = Field(
        default=None, alias="supportedEffortLevels"
    )
    supports_adaptive_thinking: bool = Field(
        default=False, alias="supportsAdaptiveThinking"
    )
    supports_fast_mode: bool = Field(default=False, alias="supportsFastMode")
    supports_auto_mode: bool = Field(default=False, alias="supportsAutoMode")
    max_tokens: Optional[int] = Field(default=None, alias="maxTokens")
    max_thinking_tokens: Optional[int] = Field(default=None, alias="maxThinkingTokens")

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )


class AccountInfo(BaseModel):
    """Account metadata for SDK initialize response."""

    email: Optional[str] = None
    organization: Optional[str] = None
    subscription_type: Optional[str] = Field(default=None, alias="subscriptionType")
    token_source: Optional[str] = Field(default=None, alias="tokenSource")
    api_key_source: Optional[str] = Field(default=None, alias="apiKeySource")
    api_provider: Optional[str] = Field(default=None, alias="apiProvider")

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


class SDKResultMessage(BaseModel):
    """End-of-turn result with cost/usage summary."""

    type: Literal["result"] = "result"
    subtype: str = "success"
    duration_ms: int = Field(default=0, alias="durationMs")
    duration_api_ms: int = Field(default=0, alias="durationApiMs")
    is_error: bool = Field(default=False, alias="isError")
    result: str = ""
    num_turns: int = Field(default=0, alias="numTurns")
    session_id: Optional[str] = Field(default=None, alias="sessionId")
    stop_reason: Optional[str] = Field(default="endTurn", alias="stopReason")
    total_cost_usd: float = Field(default=0.0, alias="totalCostUsd")
    usage: Optional[UsageInfo] = None
    model_usage: Optional[dict[str, Any]] = Field(default=None, alias="modelUsage")
    permission_denials: Optional[List[Dict[str, Any]]] = Field(
        default=None, alias="permissionDenials"
    )
    structured_output: Optional[Any] = Field(default=None, alias="structuredOutput")
    uuid: Optional[str] = None

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )


class SDKStatusMessage(BaseModel):
    """Status update message (tool progress, etc.)."""

    type: Literal["status"] = "status"
    subtype: str = ""
    message: str = ""
    tool_use_id: Optional[str] = Field(default=None, alias="toolUseID")
    progress: Optional[float] = None
    session_id: Optional[str] = Field(default=None, alias="sessionId")

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )


class SDKToolProgressMessage(BaseModel):
    """Tool execution progress update."""

    type: Literal["tool_progress"] = "tool_progress"
    tool_use_id: str = Field(alias="toolUseID")
    progress: Optional[float] = None
    message: Optional[str] = None
    session_id: Optional[str] = Field(default=None, alias="sessionId")

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )


class SDKAuthStatusMessage(BaseModel):
    """Auth status change notification."""

    type: Literal["auth_status"] = "auth_status"
    authenticated: bool = False
    provider: Optional[str] = None
    message: Optional[str] = None
    session_id: Optional[str] = Field(default=None, alias="sessionId")

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )


class SDKPromptSuggestionMessage(BaseModel):
    """Prompt suggestion from the server."""

    type: Literal["prompt_suggestion"] = "prompt_suggestion"
    suggestions: list[str] = Field(default_factory=list)
    session_id: Optional[str] = Field(default=None, alias="sessionId")

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )


__all__ = [
    "FastModeState",
    "ThinkingConfig",
    "ModelInfo",
    "AccountInfo",
    "SlashCommandInfo",
    "SdkBeta",
    "SDKResultMessage",
    "SDKStatusMessage",
    "SDKToolProgressMessage",
    "SDKAuthStatusMessage",
    "SDKPromptSuggestionMessage",
]
