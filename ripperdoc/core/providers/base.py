"""Shared abstractions for provider clients."""

from __future__ import annotations

import asyncio
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
)

from ripperdoc.core.config import ModelProfile
from ripperdoc.core.tool import Tool
from ripperdoc.utils.log import get_logger

logger = get_logger()

ProgressCallback = Callable[[str], Awaitable[None]]


@dataclass
class ProviderResponse:
    """Normalized provider response payload."""

    content_blocks: List[Dict[str, Any]]
    usage_tokens: Dict[str, int]
    cost_usd: float
    duration_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Error handling fields
    is_error: bool = False
    error_code: Optional[str] = None  # e.g., "permission_denied", "context_length_exceeded"
    error_message: Optional[str] = None

    @classmethod
    def create_error(
        cls,
        error_code: str,
        error_message: str,
        duration_ms: float = 0.0,
    ) -> "ProviderResponse":
        """Create an error response with a text block containing the error message."""
        return cls(
            content_blocks=[{"type": "text", "text": f"[API Error] {error_message}"}],
            usage_tokens={},
            cost_usd=0.0,
            duration_ms=duration_ms,
            is_error=True,
            error_code=error_code,
            error_message=error_message,
        )


class ProviderClient(ABC):
    """Abstract base for model provider clients."""

    @abstractmethod
    async def call(
        self,
        *,
        model_profile: ModelProfile,
        system_prompt: str,
        normalized_messages: List[Dict[str, Any]],
        tools: List[Tool[Any, Any]],
        tool_mode: str,
        stream: bool,
        progress_callback: Optional[ProgressCallback],
        request_timeout: Optional[float],
        max_retries: int,
        max_thinking_tokens: int,
    ) -> ProviderResponse:
        """Execute a model call and return a normalized response."""


def sanitize_tool_history(normalized_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize tool-call history for strict provider sequencing requirements.

    This function enforces two invariants for assistant `tool_use` turns:
    1. Drop unpaired `tool_use` blocks that have no later `tool_result`.
    2. Collapse matching `tool_result` blocks into a single immediate next user
       message after the assistant turn (Anthropic-compatible requirement).
    """

    def _part_type(part: Any) -> Any:
        if isinstance(part, dict):
            return part.get("type")
        return getattr(part, "type", None)

    def _part_tool_id(part: Any) -> str:
        if isinstance(part, dict):
            return str(part.get("tool_use_id") or part.get("id") or "")
        return str(getattr(part, "tool_use_id", None) or getattr(part, "id", None) or "")

    def _tool_result_ids(msg: Dict[str, Any]) -> set[str]:
        ids: set[str] = set()
        content = msg.get("content")
        if not isinstance(content, list):
            return ids
        for part in content:
            if _part_type(part) != "tool_result":
                continue
            tid = _part_tool_id(part)
            if tid:
                ids.add(tid)
        return ids

    # Build lookahead map to identify paired tool_use IDs.
    tool_results_after: List[set[str]] = []
    if normalized_messages:
        tool_results_after = [set() for _ in normalized_messages]
        future_ids: set[str] = set()
        for idx in range(len(normalized_messages) - 1, -1, -1):
            tool_results_after[idx] = set(future_ids)
            future_ids.update(_tool_result_ids(normalized_messages[idx]))

    sanitized: List[Dict[str, Any]] = []
    i = 0
    while i < len(normalized_messages):
        message = normalized_messages[i]
        if message.get("role") != "assistant":
            sanitized.append(message)
            i += 1
            continue

        content = message.get("content")
        if not isinstance(content, list):
            sanitized.append(message)
            i += 1
            continue

        tool_use_ids = [
            _part_tool_id(part)
            for part in content
            if _part_type(part) == "tool_use" and _part_tool_id(part)
        ]
        if not tool_use_ids:
            sanitized.append(message)
            i += 1
            continue

        future_results = tool_results_after[i] if tool_results_after else set()
        paired_ids = {tool_id for tool_id in tool_use_ids if tool_id in future_results}
        unpaired_ids = {tool_id for tool_id in tool_use_ids if tool_id not in future_results}

        filtered_content = [
            part
            for part in content
            if not (_part_type(part) == "tool_use" and _part_tool_id(part) in unpaired_ids)
        ]
        if not filtered_content:
            logger.debug(
                "[provider_clients] Dropped assistant message with unpaired tool_use blocks",
                extra={"unpaired_ids": list(unpaired_ids)},
            )
            i += 1
            continue

        sanitized.append({**message, "content": filtered_content})
        if unpaired_ids:
            logger.debug(
                "[provider_clients] Sanitized message to remove unpaired tool_use blocks",
                extra={"unpaired_ids": list(unpaired_ids)},
            )

        # No paired IDs left: nothing to fold.
        if not paired_ids:
            i += 1
            continue

        # Collect paired tool_result blocks from following user messages until we hit
        # the next assistant turn, then insert a single immediate user message.
        collected_results: List[Any] = []
        consumed_user_messages: List[Dict[str, Any]] = []
        seen_result_ids: set[str] = set()
        j = i + 1
        while j < len(normalized_messages):
            next_message = normalized_messages[j]
            if next_message.get("role") == "assistant":
                break

            if next_message.get("role") != "user":
                consumed_user_messages.append(next_message)
                j += 1
                continue

            next_content = next_message.get("content")
            if not isinstance(next_content, list):
                consumed_user_messages.append(next_message)
                j += 1
                continue

            remaining_parts: List[Any] = []
            for part in next_content:
                if _part_type(part) != "tool_result":
                    remaining_parts.append(part)
                    continue
                result_id = _part_tool_id(part)
                if result_id in paired_ids and result_id not in seen_result_ids:
                    collected_results.append(part)
                    seen_result_ids.add(result_id)
                else:
                    remaining_parts.append(part)

            if remaining_parts:
                consumed_user_messages.append({**next_message, "content": remaining_parts})
            if paired_ids.issubset(seen_result_ids):
                j += 1
                break
            j += 1

        if collected_results:
            sanitized.append({"role": "user", "content": collected_results})
        sanitized.extend(consumed_user_messages)
        i = j

    return sanitized


def _retry_delay_seconds(attempt: int, base_delay: float = 0.5, max_delay: float = 32.0) -> float:
    """Calculate exponential backoff with jitter."""
    capped_base: float = float(min(base_delay * (2 ** max(0, attempt - 1)), max_delay))
    jitter: float = float(random.random() * 0.25 * capped_base)
    return float(capped_base + jitter)


async def iter_with_timeout(
    stream: Iterable[Any] | AsyncIterable[Any], timeout: Optional[float]
) -> AsyncIterator[Any]:
    """Yield items from an async or sync iterable, enforcing per-item timeout if provided."""
    if timeout is None or timeout <= 0:
        if hasattr(stream, "__aiter__"):
            async for item in stream:  # type: ignore[async-for]
                yield item
        else:
            for item in stream:
                yield item
        return

    if hasattr(stream, "__aiter__"):
        aiter = stream.__aiter__()  # type: ignore[attr-defined]
        while True:
            try:
                yield await asyncio.wait_for(aiter.__anext__(), timeout=timeout)  # type: ignore[attr-defined]
            except StopAsyncIteration:
                break
    else:
        iterator = iter(stream)
        while True:
            try:
                next_item = await asyncio.wait_for(
                    asyncio.to_thread(next, iterator), timeout=timeout
                )
            except StopIteration:
                break
            yield next_item


async def call_with_timeout_and_retries(
    coro_factory: Callable[[], Awaitable[Any]],
    request_timeout: Optional[float],
    max_retries: int,
) -> Any:
    """Run a coroutine with timeout and limited retries (exponential backoff)."""
    attempts = max(0, int(max_retries)) + 1
    last_error: Optional[Exception] = None

    for attempt in range(1, attempts + 1):
        try:
            if request_timeout and request_timeout > 0:
                return await asyncio.wait_for(coro_factory(), timeout=request_timeout)
            return await coro_factory()
        except asyncio.TimeoutError as exc:
            last_error = exc
            if attempt == attempts:
                break
            delay_seconds = _retry_delay_seconds(attempt)
            logger.warning(
                "[provider_clients] Request timed out; retrying",
                extra={
                    "attempt": attempt,
                    "max_retries": attempts - 1,
                    "delay_seconds": round(delay_seconds, 3),
                },
            )
            await asyncio.sleep(delay_seconds)
        except asyncio.CancelledError:
            raise  # Don't suppress task cancellation
        except (RuntimeError, ValueError, TypeError, OSError, ConnectionError) as exc:
            # Non-timeout errors are not retried; surface immediately.
            raise exc
    if last_error:
        raise RuntimeError(f"Request timed out after {attempts} attempts") from last_error
    raise RuntimeError("Unexpected error executing request with retries")
