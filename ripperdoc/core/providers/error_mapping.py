"""Shared helpers for provider exception mapping."""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Optional

from ripperdoc.core.providers.errors import (
    ProviderApiError,
    ProviderBadRequestError,
    ProviderConnectionError,
    ProviderContentPolicyViolationError,
    ProviderContextLengthExceededError,
    ProviderInsufficientBalanceError,
    ProviderMappedError,
    ProviderPermissionDeniedError,
    ProviderTimeoutError,
)

_TIMEOUT_HINTS = ("timed out", "timeout")
_RETRYABLE_CONNECTION_HINTS = (
    "incomplete chunked read",
    "peer closed connection",
    "connection reset",
    "connection aborted",
    "broken pipe",
    "server disconnected",
)
_CONTEXT_HINTS = (
    "context",
    "token",
    "prompt is too long",
    "input is too long",
    "exceeds the model's maximum context length",
)


def is_timeout_message(message: str) -> bool:
    """Return True when an error message describes timeout-like behavior."""
    lowered = message.lower()
    return any(hint in lowered for hint in _TIMEOUT_HINTS)


def map_connection_error(message: str, *, retryable: Optional[bool] = None) -> ProviderMappedError:
    """Map a provider connection error message to a normalized error."""
    if is_timeout_message(message):
        return ProviderTimeoutError(f"Request timed out: {message}")
    if retryable is None:
        lowered = message.lower()
        retryable = any(hint in lowered for hint in _RETRYABLE_CONNECTION_HINTS)
    return ProviderConnectionError(f"Connection error: {message}", retryable=retryable)


def map_permission_denied_error(message: str) -> ProviderMappedError:
    """Map permission-denied messages with balance-aware specialization."""
    lowered = message.lower()
    if "balance" in lowered or "insufficient" in lowered:
        return ProviderInsufficientBalanceError(f"Insufficient balance: {message}")
    return ProviderPermissionDeniedError(f"Permission denied: {message}")


def map_bad_request_error(message: str) -> ProviderMappedError:
    """Map invalid request messages including context/content policy variants."""
    lowered = message.lower()
    if any(hint in lowered for hint in _CONTEXT_HINTS):
        return ProviderContextLengthExceededError(f"Context length exceeded: {message}")
    if "content" in lowered and "policy" in lowered:
        return ProviderContentPolicyViolationError(
            f"Content policy violation: {message}"
        )
    return ProviderBadRequestError(f"Invalid request: {message}")


def map_api_status_error(message: str, status: Any) -> ProviderMappedError:
    """Map API status errors with context-length detection from payload text."""
    lowered = message.lower()
    if any(hint in lowered for hint in _CONTEXT_HINTS):
        return ProviderContextLengthExceededError(f"Context length exceeded: {message}")
    return ProviderApiError(f"API error ({status}): {message}")


def classify_mapped_error(exc: Exception) -> Optional[tuple[str, str]]:
    """Return normalized (code, message) if exception is already mapped."""
    if isinstance(exc, ProviderMappedError):
        return exc.error_code, str(exc)
    return None


async def run_with_exception_mapper(
    request_fn: Callable[[], Awaitable[Any]],
    mapper: Callable[[Exception], Exception],
) -> Any:
    """Execute request and transform provider exceptions via mapper."""
    try:
        return await request_fn()
    except Exception as exc:
        mapped_exc = mapper(exc)
        if mapped_exc is exc:
            raise
        raise mapped_exc from exc
