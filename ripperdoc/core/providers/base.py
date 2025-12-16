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
    """Strip tool_use blocks that lack a following tool_result to satisfy provider constraints."""

    def _tool_result_ids(msg: Dict[str, Any]) -> set[str]:
        ids: set[str] = set()
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                part_type = getattr(
                    part, "get", lambda k, default=None: part.__dict__.get(k, default)
                )("type", None)
                if part_type == "tool_result":
                    tid = (
                        getattr(part, "tool_use_id", None)
                        or getattr(part, "id", None)
                        or part.get("tool_use_id")
                        or part.get("id")
                    )
                    if tid:
                        ids.add(str(tid))
        return ids

    # Build a lookahead map so we can pair tool_use blocks with tool_results that may
    # appear in any later message (not just the immediate next one).
    tool_results_after: List[set[str]] = []
    if normalized_messages:
        tool_results_after = [set() for _ in normalized_messages]
        future_ids: set[str] = set()
        for idx in range(len(normalized_messages) - 1, -1, -1):
            tool_results_after[idx] = set(future_ids)
            future_ids.update(_tool_result_ids(normalized_messages[idx]))

    sanitized: List[Dict[str, Any]] = []
    for idx, message in enumerate(normalized_messages):
        if message.get("role") != "assistant":
            sanitized.append(message)
            continue

        content = message.get("content")
        if not isinstance(content, list):
            sanitized.append(message)
            continue

        tool_use_blocks = [
            part
            for part in content
            if (
                getattr(part, "type", None)
                or (part.get("type") if isinstance(part, dict) else None)
            )
            == "tool_use"
        ]
        if not tool_use_blocks:
            sanitized.append(message)
            continue

        future_results = tool_results_after[idx] if tool_results_after else set()

        # Identify unpaired tool_use IDs
        unpaired_ids: set[str] = set()
        for block in tool_use_blocks:
            block_id = (
                getattr(block, "tool_use_id", None)
                or getattr(block, "id", None)
                or (block.get("tool_use_id") if isinstance(block, dict) else None)
                or (block.get("id") if isinstance(block, dict) else None)
            )
            if block_id and str(block_id) not in future_results:
                unpaired_ids.add(str(block_id))

        if not unpaired_ids:
            sanitized.append(message)
            continue

        # Drop unpaired tool_use blocks
        filtered_content = []
        for part in content:
            part_type = getattr(part, "type", None) or (
                part.get("type") if isinstance(part, dict) else None
            )
            if part_type == "tool_use":
                block_id = (
                    getattr(part, "tool_use_id", None)
                    or getattr(part, "id", None)
                    or (part.get("tool_use_id") if isinstance(part, dict) else None)
                    or (part.get("id") if isinstance(part, dict) else None)
                )
                if block_id and str(block_id) in unpaired_ids:
                    continue
            filtered_content.append(part)

        if not filtered_content:
            logger.debug(
                "[provider_clients] Dropped assistant message with unpaired tool_use blocks",
                extra={"unpaired_ids": list(unpaired_ids)},
            )
            continue

        sanitized.append({**message, "content": filtered_content})
        logger.debug(
            "[provider_clients] Sanitized message to remove unpaired tool_use blocks",
            extra={"unpaired_ids": list(unpaired_ids)},
        )

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
