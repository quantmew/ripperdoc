"""Context usage protocol DTOs."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class ContextCategory(BaseModel):
    """A category of context window usage."""

    name: str
    tokens: int
    color: Optional[str] = None
    is_deferred: bool = Field(default=False, alias="isDeferred")

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )


class ContextGridSquare(BaseModel):
    """A single square in the context usage grid."""

    color: Optional[str] = None
    is_filled: bool = Field(default=False, alias="isFilled")
    category_name: Optional[str] = Field(default=None, alias="categoryName")
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


__all__ = [
    "ContextCategory",
    "ContextGridSquare",
    "ContextUsageResult",
]
