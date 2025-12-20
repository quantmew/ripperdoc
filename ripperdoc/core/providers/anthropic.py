"""Anthropic provider client."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional
from uuid import uuid4

import anthropic
import httpx
from anthropic import AsyncAnthropic

from ripperdoc.core.config import ModelProfile
from ripperdoc.core.providers.base import (
    ProgressCallback,
    ProviderClient,
    ProviderResponse,
    call_with_timeout_and_retries,
    sanitize_tool_history,
)
from ripperdoc.core.query_utils import (
    anthropic_usage_tokens,
    build_anthropic_tool_schemas,
    estimate_cost_usd,
)
from ripperdoc.core.tool import Tool
from ripperdoc.utils.log import get_logger
from ripperdoc.utils.session_usage import record_usage

logger = get_logger()


def _classify_anthropic_error(exc: Exception) -> tuple[str, str]:
    """Classify an Anthropic exception into error code and user-friendly message."""
    exc_type = type(exc).__name__
    exc_msg = str(exc)

    if isinstance(exc, anthropic.AuthenticationError):
        return "authentication_error", f"Authentication failed: {exc_msg}"
    if isinstance(exc, anthropic.PermissionDeniedError):
        if "balance" in exc_msg.lower() or "insufficient" in exc_msg.lower():
            return "insufficient_balance", f"Insufficient balance: {exc_msg}"
        return "permission_denied", f"Permission denied: {exc_msg}"
    if isinstance(exc, anthropic.NotFoundError):
        return "model_not_found", f"Model not found: {exc_msg}"
    if isinstance(exc, anthropic.BadRequestError):
        if "context" in exc_msg.lower() or "token" in exc_msg.lower():
            return "context_length_exceeded", f"Context length exceeded: {exc_msg}"
        if "content" in exc_msg.lower() and "policy" in exc_msg.lower():
            return "content_policy_violation", f"Content policy violation: {exc_msg}"
        return "bad_request", f"Invalid request: {exc_msg}"
    if isinstance(exc, anthropic.RateLimitError):
        return "rate_limit", f"Rate limit exceeded: {exc_msg}"
    if isinstance(exc, anthropic.APIConnectionError):
        return "connection_error", f"Connection error: {exc_msg}"
    if isinstance(exc, anthropic.APIStatusError):
        status = getattr(exc, "status_code", "unknown")
        return "api_error", f"API error ({status}): {exc_msg}"
    if isinstance(exc, asyncio.TimeoutError):
        return "timeout", f"Request timed out: {exc_msg}"

    return "unknown_error", f"Unexpected error ({exc_type}): {exc_msg}"


def _content_blocks_from_stream_state(
    collected_text: List[str],
    collected_thinking: List[str],
    collected_tool_calls: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build content blocks from accumulated stream state."""
    blocks: List[Dict[str, Any]] = []

    # Add thinking block if present
    if collected_thinking:
        blocks.append(
            {
                "type": "thinking",
                "thinking": "".join(collected_thinking),
            }
        )

    # Add text block if present
    if collected_text:
        blocks.append(
            {
                "type": "text",
                "text": "".join(collected_text),
            }
        )

    # Add tool_use blocks
    for idx in sorted(collected_tool_calls.keys()):
        call = collected_tool_calls[idx]
        name = call.get("name")
        if not name:
            continue
        tool_use_id = call.get("id") or str(uuid4())
        blocks.append(
            {
                "type": "tool_use",
                "tool_use_id": tool_use_id,
                "name": name,
                "input": call.get("input", {}),
            }
        )

    return blocks


def _content_blocks_from_response(response: Any) -> List[Dict[str, Any]]:
    """Normalize Anthropic response content to our internal block format."""
    blocks: List[Dict[str, Any]] = []
    for block in getattr(response, "content", []) or []:
        btype = getattr(block, "type", None)
        if btype == "text":
            blocks.append({"type": "text", "text": getattr(block, "text", "")})
        elif btype == "thinking":
            blocks.append(
                {
                    "type": "thinking",
                    "thinking": getattr(block, "thinking", None) or "",
                    "signature": getattr(block, "signature", None),
                }
            )
        elif btype == "redacted_thinking":
            blocks.append(
                {
                    "type": "redacted_thinking",
                    "data": getattr(block, "data", None),
                    "signature": getattr(block, "signature", None),
                }
            )
        elif btype == "tool_use":
            raw_input = getattr(block, "input", {}) or {}
            blocks.append(
                {
                    "type": "tool_use",
                    "tool_use_id": getattr(block, "id", None) or str(uuid4()),
                    "name": getattr(block, "name", None),
                    "input": raw_input if isinstance(raw_input, dict) else {},
                }
            )
    return blocks


class AnthropicClient(ProviderClient):
    """Anthropic client with streaming and non-streaming support.

    Streaming mode (default):
    - Uses event-based streaming to capture both thinking and text tokens
    - Timeout applies per-token (chunk), not to the entire request
    - Thinking tokens are streamed in real-time via progress_callback

    Non-streaming mode:
    - Makes a single blocking request
    - Timeout applies to the entire request
    """

    def __init__(self, client_factory: Optional[Callable[[], Awaitable[AsyncAnthropic]]] = None):
        self._client_factory = client_factory

    async def _client(self, kwargs: Dict[str, Any]) -> AsyncAnthropic:
        if self._client_factory:
            return await self._client_factory()
        return AsyncAnthropic(**kwargs)

    async def call(
        self,
        *,
        model_profile: ModelProfile,
        system_prompt: str,
        normalized_messages: Any,
        tools: List[Tool[Any, Any]],
        tool_mode: str,
        stream: bool,
        progress_callback: Optional[ProgressCallback],
        request_timeout: Optional[float],
        max_retries: int,
        max_thinking_tokens: int,
    ) -> ProviderResponse:
        start_time = time.time()

        try:
            return await self._call_impl(
                model_profile=model_profile,
                system_prompt=system_prompt,
                normalized_messages=normalized_messages,
                tools=tools,
                tool_mode=tool_mode,
                stream=stream,
                progress_callback=progress_callback,
                request_timeout=request_timeout,
                max_retries=max_retries,
                max_thinking_tokens=max_thinking_tokens,
                start_time=start_time,
            )
        except asyncio.CancelledError:
            raise  # Don't suppress task cancellation
        except Exception as exc:
            duration_ms = (time.time() - start_time) * 1000
            error_code, error_message = _classify_anthropic_error(exc)
            logger.debug(
                "[anthropic_client] Exception details",
                extra={
                    "model": model_profile.model,
                    "exception_type": type(exc).__name__,
                    "exception_str": str(exc),
                    "error_code": error_code,
                },
            )
            logger.error(
                "[anthropic_client] API call failed",
                extra={
                    "model": model_profile.model,
                    "error_code": error_code,
                    "error_message": error_message,
                    "duration_ms": round(duration_ms, 2),
                },
            )
            return ProviderResponse.create_error(
                error_code=error_code,
                error_message=error_message,
                duration_ms=duration_ms,
            )

    async def _call_impl(
        self,
        *,
        model_profile: ModelProfile,
        system_prompt: str,
        normalized_messages: Any,
        tools: List[Tool[Any, Any]],
        tool_mode: str,
        stream: bool,
        progress_callback: Optional[ProgressCallback],
        request_timeout: Optional[float],
        max_retries: int,
        max_thinking_tokens: int,
        start_time: float,
    ) -> ProviderResponse:
        """Internal implementation of call, may raise exceptions."""
        tool_schemas = await build_anthropic_tool_schemas(tools)
        response_metadata: Dict[str, Any] = {}

        logger.debug(
            "[anthropic_client] Preparing request",
            extra={
                "model": model_profile.model,
                "tool_mode": tool_mode,
                "stream": stream,
                "max_thinking_tokens": max_thinking_tokens,
                "num_tools": len(tool_schemas),
            },
        )

        anthropic_kwargs: Dict[str, Any] = {}
        if model_profile.api_base:
            anthropic_kwargs["base_url"] = model_profile.api_base
        if model_profile.api_key:
            anthropic_kwargs["api_key"] = model_profile.api_key
        auth_token = getattr(model_profile, "auth_token", None)
        if auth_token:
            anthropic_kwargs["auth_token"] = auth_token

        # Set timeout for the Anthropic SDK client
        # For streaming, we want a long timeout since models may take time to start responding
        # httpx.Timeout: (connect, read, write, pool)
        if stream:
            # For streaming: long read timeout, reasonable connect timeout
            # The read timeout applies to waiting for each chunk from the server
            timeout_config = httpx.Timeout(
                connect=60.0,  # 60 seconds to establish connection
                read=600.0,  # 10 minutes to wait for each chunk (model may be thinking)
                write=60.0,  # 60 seconds to send request
                pool=60.0,  # 60 seconds to get connection from pool
            )
            anthropic_kwargs["timeout"] = timeout_config
        elif request_timeout and request_timeout > 0:
            # For non-streaming: use the provided timeout
            anthropic_kwargs["timeout"] = request_timeout

        normalized_messages = sanitize_tool_history(list(normalized_messages))

        thinking_payload: Optional[Dict[str, Any]] = None
        if max_thinking_tokens > 0:
            thinking_payload = {"type": "enabled", "budget_tokens": max_thinking_tokens}

        # Build common request kwargs
        request_kwargs: Dict[str, Any] = {
            "model": model_profile.model,
            "max_tokens": model_profile.max_tokens,
            "system": system_prompt,
            "messages": normalized_messages,
            "temperature": model_profile.temperature,
        }
        if tool_schemas:
            request_kwargs["tools"] = tool_schemas
        if thinking_payload:
            request_kwargs["thinking"] = thinking_payload

        logger.debug(
            "[anthropic_client] Request parameters",
            extra={
                "model": model_profile.model,
                "request_kwargs": json.dumps(
                    {k: v for k, v in request_kwargs.items() if k != "messages"},
                    ensure_ascii=False,
                    default=str,
                )[:1000],
                "thinking_payload": json.dumps(thinking_payload, ensure_ascii=False)
                if thinking_payload
                else None,
            },
        )

        async with await self._client(anthropic_kwargs) as client:
            if stream:
                # Streaming mode: use event-based streaming with per-token timeout
                content_blocks, usage_tokens = await self._stream_request(
                    client=client,
                    request_kwargs=request_kwargs,
                    progress_callback=progress_callback,
                    request_timeout=request_timeout,
                    max_retries=max_retries,
                    response_metadata=response_metadata,
                )
            else:
                # Non-streaming mode: single request with overall timeout
                content_blocks, usage_tokens = await self._non_stream_request(
                    client=client,
                    request_kwargs=request_kwargs,
                    request_timeout=request_timeout,
                    max_retries=max_retries,
                    response_metadata=response_metadata,
                )

        duration_ms = (time.time() - start_time) * 1000
        cost_usd = estimate_cost_usd(model_profile, usage_tokens)
        record_usage(
            model_profile.model, duration_ms=duration_ms, cost_usd=cost_usd, **usage_tokens
        )

        logger.debug(
            "[anthropic_client] Response content blocks",
            extra={
                "model": model_profile.model,
                "content_blocks": json.dumps(content_blocks, ensure_ascii=False)[:1000],
                "usage_tokens": json.dumps(usage_tokens, ensure_ascii=False),
                "metadata": json.dumps(response_metadata, ensure_ascii=False)[:500],
            },
        )

        logger.info(
            "[anthropic_client] Response received",
            extra={
                "model": model_profile.model,
                "duration_ms": round(duration_ms, 2),
                "tool_mode": tool_mode,
                "tool_schemas": len(tool_schemas),
                "stream": stream,
                "content_blocks": len(content_blocks),
            },
        )

        return ProviderResponse(
            content_blocks=content_blocks,
            usage_tokens=usage_tokens,
            cost_usd=cost_usd,
            duration_ms=duration_ms,
            metadata=response_metadata,
        )

    async def _stream_request(
        self,
        *,
        client: AsyncAnthropic,
        request_kwargs: Dict[str, Any],
        progress_callback: Optional[ProgressCallback],
        request_timeout: Optional[float],
        max_retries: int,
        response_metadata: Dict[str, Any],
    ) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
        """Execute a streaming request with per-token timeout.

        Uses Anthropic's event-based streaming API to capture:
        - thinking tokens (streamed in real-time)
        - text tokens (streamed in real-time)
        - tool_use blocks

        In streaming mode:
        - Connection timeout uses request_timeout
        - Per-event timeout is disabled (None) because the model may take
          a long time to generate the first token (especially during thinking)
        - Once streaming starts, events should flow continuously
        """
        collected_text: List[str] = []
        collected_thinking: List[str] = []
        collected_tool_calls: Dict[int, Dict[str, Any]] = {}
        usage_tokens: Dict[str, int] = {}

        # Use mutable containers to track state across event handling
        current_block_index_ref: List[int] = [-1]
        current_block_type_ref: List[Optional[str]] = [None]

        event_count = 0
        message_stop_received = False

        async def _do_stream() -> None:
            nonlocal event_count, message_stop_received
            event_count = 0
            message_stop_received = False

            logger.debug(
                "[anthropic_client] Initiating stream request",
                extra={
                    "model": request_kwargs.get("model"),
                },
            )

            # Create the stream - this initiates the connection
            stream = client.messages.stream(**request_kwargs)

            # Enter the stream context
            stream_manager = await stream.__aenter__()

            try:
                # Iterate over events
                # Some API proxies don't properly close the stream after message_stop,
                # so we break out of the loop when we receive message_stop
                async for event in stream_manager:
                    event_count += 1
                    event_type = getattr(event, "type", "unknown")

                    await self._handle_stream_event(
                        event=event,
                        collected_text=collected_text,
                        collected_thinking=collected_thinking,
                        collected_tool_calls=collected_tool_calls,
                        usage_tokens=usage_tokens,
                        progress_callback=progress_callback,
                        current_block_index_ref=current_block_index_ref,
                        current_block_type_ref=current_block_type_ref,
                    )

                    # Check if we received message_stop - break out of loop
                    # Some API proxies don't properly close the SSE stream
                    if event_type == "message_stop":
                        message_stop_received = True
                        break

            except Exception:
                raise
            finally:
                try:
                    # Use timeout for __aexit__ in case the stream doesn't close properly
                    await asyncio.wait_for(stream.__aexit__(None, None, None), timeout=5.0)
                except asyncio.TimeoutError:
                    pass  # Stream didn't close properly, continue anyway
                except Exception:
                    pass  # Ignore __aexit__ errors

        # For streaming, we don't use call_with_timeout_and_retries on the whole operation
        # Instead, timeout is applied per-event inside _iter_events_with_timeout
        # But we still want retries for connection failures
        attempts = max(0, int(max_retries)) + 1
        last_error: Optional[Exception] = None

        for attempt in range(1, attempts + 1):
            try:
                # Reset state for retry
                collected_text.clear()
                collected_thinking.clear()
                collected_tool_calls.clear()
                usage_tokens.clear()
                current_block_index_ref[0] = -1
                current_block_type_ref[0] = None

                await _do_stream()
                break  # Success
            except asyncio.TimeoutError as exc:
                last_error = exc
                if attempt == attempts:
                    break
                delay = 0.5 * (2 ** (attempt - 1))  # Exponential backoff
                logger.warning(
                    "[anthropic_client] Stream timed out; retrying",
                    extra={
                        "attempt": attempt,
                        "max_retries": max_retries,
                        "delay_seconds": delay,
                    },
                )
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                raise
            except (RuntimeError, ValueError, TypeError, OSError, ConnectionError) as exc:
                # Non-timeout errors: retry for connection errors only
                if isinstance(exc, (OSError, ConnectionError)):
                    last_error = exc
                    if attempt == attempts:
                        raise
                    delay = 0.5 * (2 ** (attempt - 1))
                    logger.warning(
                        "[anthropic_client] Connection error; retrying",
                        extra={
                            "attempt": attempt,
                            "error": str(exc),
                        },
                    )
                    await asyncio.sleep(delay)
                else:
                    raise

        if (
            last_error
            and not collected_text
            and not collected_thinking
            and not collected_tool_calls
        ):
            raise RuntimeError(f"Stream failed after {attempts} attempts") from last_error

        # Store reasoning content in metadata
        if collected_thinking:
            response_metadata["reasoning_content"] = "".join(collected_thinking)

        content_blocks = _content_blocks_from_stream_state(
            collected_text, collected_thinking, collected_tool_calls
        )

        return content_blocks, usage_tokens

    async def _handle_stream_event(
        self,
        *,
        event: Any,
        collected_text: List[str],
        collected_thinking: List[str],
        collected_tool_calls: Dict[int, Dict[str, Any]],
        usage_tokens: Dict[str, int],
        progress_callback: Optional[ProgressCallback],
        current_block_index_ref: List[int],
        current_block_type_ref: List[Optional[str]],
    ) -> None:
        """Handle a single stream event.

        Supports both standard Anthropic API events and non-standard formats
        from API proxies like aiping.cn.

        Standard Anthropic events:
        - message_start, content_block_start, content_block_delta, content_block_stop
        - message_delta, message_stop

        Non-standard events (aiping.cn style):
        - thinking (direct thinking content)
        - text (direct text content)
        - signature (thinking signature)
        """
        event_type = getattr(event, "type", None)

        if event_type == "message_start":
            # Extract initial usage info if available
            message = getattr(event, "message", None)
            if message:
                usage = getattr(message, "usage", None)
                if usage:
                    usage_tokens.update(anthropic_usage_tokens(usage))

        elif event_type == "content_block_start":
            # New content block starting
            index = getattr(event, "index", 0)
            content_block = getattr(event, "content_block", None)
            if content_block:
                block_type = getattr(content_block, "type", None)
                current_block_index_ref[0] = index
                current_block_type_ref[0] = block_type

                if block_type == "tool_use":
                    # Initialize tool call state
                    collected_tool_calls[index] = {
                        "id": getattr(content_block, "id", None),
                        "name": getattr(content_block, "name", None),
                        "input_json": "",
                        "input": {},
                    }
                    # Announce tool start
                    if progress_callback:
                        tool_name = getattr(content_block, "name", "unknown")
                        try:
                            await progress_callback(f"[tool:{tool_name}]")
                        except (RuntimeError, ValueError, TypeError, OSError):
                            pass

        elif event_type == "content_block_delta":
            # Content delta within a block
            index = getattr(event, "index", current_block_index_ref[0])
            delta = getattr(event, "delta", None)
            if not delta:
                return

            delta_type = getattr(delta, "type", None)

            if delta_type == "thinking_delta":
                # Thinking content delta
                thinking_text = getattr(delta, "thinking", "")
                if thinking_text:
                    collected_thinking.append(thinking_text)
                    if progress_callback:
                        try:
                            await progress_callback(thinking_text)
                        except (RuntimeError, ValueError, TypeError, OSError) as cb_exc:
                            logger.warning(
                                "[anthropic_client] Progress callback failed: %s: %s",
                                type(cb_exc).__name__,
                                cb_exc,
                            )

            elif delta_type == "text_delta":
                # Text content delta
                text = getattr(delta, "text", "")
                if text:
                    collected_text.append(text)
                    if progress_callback:
                        try:
                            await progress_callback(text)
                        except (RuntimeError, ValueError, TypeError, OSError) as cb_exc:
                            logger.warning(
                                "[anthropic_client] Progress callback failed: %s: %s",
                                type(cb_exc).__name__,
                                cb_exc,
                            )

            elif delta_type == "input_json_delta":
                # Tool input JSON delta
                partial_json = getattr(delta, "partial_json", "")
                if partial_json and index in collected_tool_calls:
                    collected_tool_calls[index]["input_json"] += partial_json
                    if progress_callback:
                        try:
                            await progress_callback(partial_json)
                        except (RuntimeError, ValueError, TypeError, OSError):
                            pass

        # ===== Non-standard events from aiping.cn and similar proxies =====
        # NOTE: aiping.cn sends BOTH standard (content_block_delta) and non-standard
        # (text, thinking) events. We only process the non-standard events if we
        # haven't already collected content from standard events in this block.
        # This is controlled by checking if the standard delta was processed.

        elif event_type == "thinking":
            # Direct thinking content (non-standard, aiping.cn style)
            # Skip - already handled via content_block_delta (aiping.cn sends both)
            pass

        elif event_type == "text":
            # Direct text content (non-standard, aiping.cn style)
            # Skip - already handled via content_block_delta (aiping.cn sends both)
            pass

        elif event_type == "signature":
            # Thinking signature (non-standard, aiping.cn style)
            pass

        # ===== Standard events continued =====

        elif event_type == "content_block_stop":
            # Content block finished
            index = getattr(event, "index", current_block_index_ref[0])

            # Parse accumulated JSON for tool calls
            if index in collected_tool_calls:
                import json

                json_str = collected_tool_calls[index].get("input_json", "")
                if json_str:
                    try:
                        collected_tool_calls[index]["input"] = json.loads(json_str)
                    except json.JSONDecodeError:
                        logger.warning(
                            "[anthropic_client] Failed to parse tool input JSON",
                            extra={"json": json_str[:200]},
                        )
                        collected_tool_calls[index]["input"] = {}

        elif event_type == "message_delta":
            # Message-level delta (usually contains usage info at the end)
            usage = getattr(event, "usage", None)
            if usage:
                # Update with final usage - output_tokens comes here
                usage_tokens["output_tokens"] = getattr(usage, "output_tokens", 0)

        elif event_type == "message_stop":
            # Message complete
            pass

        # Unknown event types are silently ignored

    async def _non_stream_request(
        self,
        *,
        client: AsyncAnthropic,
        request_kwargs: Dict[str, Any],
        request_timeout: Optional[float],
        max_retries: int,
        response_metadata: Dict[str, Any],
    ) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
        """Execute a non-streaming request with overall timeout."""

        async def _do_request() -> Any:
            return await client.messages.create(**request_kwargs)

        response = await call_with_timeout_and_retries(
            _do_request,
            request_timeout,
            max_retries,
        )

        usage_tokens = anthropic_usage_tokens(getattr(response, "usage", None))
        content_blocks = _content_blocks_from_response(response)

        # Extract reasoning content for metadata
        for block in content_blocks:
            if block.get("type") == "thinking":
                thinking_text = block.get("thinking") or ""
                if thinking_text:
                    response_metadata["reasoning_content"] = thinking_text
                    break

        return content_blocks, usage_tokens
