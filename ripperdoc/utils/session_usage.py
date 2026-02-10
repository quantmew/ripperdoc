"""Session-level usage tracking for model calls."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable


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


def rebuild_session_usage(messages: Iterable[Any]) -> SessionUsage:
    """Rebuild in-memory usage counters from persisted conversation messages.

    This is used when resuming a session: usage is tracked in-memory at runtime,
    so a fresh process must reconstruct counters from stored assistant messages.
    """
    reset_session_usage()
    for msg in messages:
        if getattr(msg, "type", None) != "assistant":
            continue

        model = getattr(msg, "model", None) or "unknown"
        input_tokens = _as_int(getattr(msg, "input_tokens", 0))
        output_tokens = _as_int(getattr(msg, "output_tokens", 0))
        cache_read_tokens = _as_int(getattr(msg, "cache_read_tokens", 0))
        cache_creation_tokens = _as_int(getattr(msg, "cache_creation_tokens", 0))
        duration_ms = float(getattr(msg, "duration_ms", 0.0) or 0.0)
        cost_usd = float(getattr(msg, "cost_usd", 0.0) or 0.0)

        has_usage = any(
            [
                input_tokens,
                output_tokens,
                cache_read_tokens,
                cache_creation_tokens,
                duration_ms > 0,
                cost_usd > 0,
                bool(getattr(msg, "model", None)),
            ]
        )
        if not has_usage:
            continue

        record_usage(
            str(model),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read_tokens,
            cache_creation_input_tokens=cache_creation_tokens,
            duration_ms=duration_ms,
            cost_usd=cost_usd,
        )
    return get_session_usage()


__all__ = [
    "ModelUsage",
    "SessionUsage",
    "get_session_usage",
    "rebuild_session_usage",
    "record_usage",
    "reset_session_usage",
]
