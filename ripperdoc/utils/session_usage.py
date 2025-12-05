"""Session-level usage tracking for model calls."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ModelUsage:
    """Aggregate token and duration stats for a single model."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    requests: int = 0
    duration_ms: float = 0.0
    cost_usd: float = 0.0


@dataclass
class SessionUsage:
    """In-memory snapshot of usage for the current session."""

    models: Dict[str, ModelUsage] = field(default_factory=dict)

    @property
    def total_input_tokens(self) -> int:
        return sum(usage.input_tokens for usage in self.models.values())

    @property
    def total_output_tokens(self) -> int:
        return sum(usage.output_tokens for usage in self.models.values())

    @property
    def total_cache_read_tokens(self) -> int:
        return sum(usage.cache_read_input_tokens for usage in self.models.values())

    @property
    def total_cache_creation_tokens(self) -> int:
        return sum(usage.cache_creation_input_tokens for usage in self.models.values())

    @property
    def total_requests(self) -> int:
        return sum(usage.requests for usage in self.models.values())

    @property
    def total_duration_ms(self) -> float:
        return sum(usage.duration_ms for usage in self.models.values())

    @property
    def total_cost_usd(self) -> float:
        return sum(usage.cost_usd for usage in self.models.values())


_SESSION_USAGE = SessionUsage()


def _as_int(value: Any) -> int:
    """Best-effort integer conversion."""
    try:
        if value is None:
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def _model_key(model: str) -> str:
    """Normalize model names for use as dictionary keys."""
    return model or "unknown"


def record_usage(
    model: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
    duration_ms: float = 0.0,
    cost_usd: float = 0.0,
) -> None:
    """Record a single model invocation."""
    global _SESSION_USAGE
    key = _model_key(model)
    usage = _SESSION_USAGE.models.setdefault(key, ModelUsage())

    usage.input_tokens += _as_int(input_tokens)
    usage.output_tokens += _as_int(output_tokens)
    usage.cache_read_input_tokens += _as_int(cache_read_input_tokens)
    usage.cache_creation_input_tokens += _as_int(cache_creation_input_tokens)
    usage.duration_ms += float(duration_ms) if duration_ms and duration_ms > 0 else 0.0
    usage.requests += 1
    usage.cost_usd += float(cost_usd) if cost_usd and cost_usd > 0 else 0.0


def get_session_usage() -> SessionUsage:
    """Return a copy of the current session usage."""
    return deepcopy(_SESSION_USAGE)


def reset_session_usage() -> None:
    """Clear all recorded usage (primarily for tests)."""
    global _SESSION_USAGE
    _SESSION_USAGE = SessionUsage()


__all__ = [
    "ModelUsage",
    "SessionUsage",
    "get_session_usage",
    "record_usage",
    "reset_session_usage",
]
