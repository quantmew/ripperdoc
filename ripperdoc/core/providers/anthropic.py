"""Anthropic provider client."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional

import anthropic
from anthropic import AsyncAnthropic

from ripperdoc.core.config import ModelProfile
from ripperdoc.core.providers.base import (
    ProgressCallback,
    ProviderClient,
    ProviderResponse,
    call_with_timeout_and_retries,
    iter_with_timeout,
    sanitize_tool_history,
)
from ripperdoc.core.query_utils import (
    anthropic_usage_tokens,
    build_anthropic_tool_schemas,
    content_blocks_from_anthropic_response,
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


class AnthropicClient(ProviderClient):
    """Anthropic client with streaming and non-streaming support."""

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
        collected_text: List[str] = []
        reasoning_parts: List[str] = []
        response_metadata: Dict[str, Any] = {}

        anthropic_kwargs = {"base_url": model_profile.api_base}
        if model_profile.api_key:
            anthropic_kwargs["api_key"] = model_profile.api_key
        auth_token = getattr(model_profile, "auth_token", None)
        if auth_token:
            anthropic_kwargs["auth_token"] = auth_token

        normalized_messages = sanitize_tool_history(list(normalized_messages))

        thinking_payload: Optional[Dict[str, Any]] = None
        if max_thinking_tokens > 0:
            thinking_payload = {"type": "enabled", "budget_tokens": max_thinking_tokens}

        async with await self._client(anthropic_kwargs) as client:

            async def _stream_request() -> Any:
                stream_cm = client.messages.stream(
                    model=model_profile.model,
                    max_tokens=model_profile.max_tokens,
                    system=system_prompt,
                    messages=normalized_messages,  # type: ignore[arg-type]
                    tools=tool_schemas if tool_schemas else None,  # type: ignore
                    temperature=model_profile.temperature,
                    thinking=thinking_payload,  # type: ignore[arg-type]
                )
                stream_resp = (
                    await asyncio.wait_for(stream_cm.__aenter__(), timeout=request_timeout)
                    if request_timeout and request_timeout > 0
                    else await stream_cm.__aenter__()
                )
                try:
                    async for text in iter_with_timeout(stream_resp.text_stream, request_timeout):
                        if text:
                            collected_text.append(text)
                            if progress_callback:
                                try:
                                    await progress_callback(text)
                                except (RuntimeError, ValueError, TypeError, OSError) as cb_exc:
                                    logger.warning(
                                        "[anthropic_client] Stream callback failed: %s: %s",
                                        type(cb_exc).__name__, cb_exc,
                                    )
                    getter = getattr(stream_resp, "get_final_response", None) or getattr(
                        stream_resp, "get_final_message", None
                    )
                    if getter:
                        return await getter()
                    return None
                finally:
                    await stream_cm.__aexit__(None, None, None)

            async def _non_stream_request() -> Any:
                return await client.messages.create(
                    model=model_profile.model,
                    max_tokens=model_profile.max_tokens,
                    system=system_prompt,
                    messages=normalized_messages,  # type: ignore[arg-type]
                    tools=tool_schemas if tool_schemas else None,  # type: ignore
                    temperature=model_profile.temperature,
                    thinking=thinking_payload,  # type: ignore[arg-type]
                )

            timeout_for_call = None if stream else request_timeout
            response = await call_with_timeout_and_retries(
                _stream_request if stream else _non_stream_request,
                timeout_for_call,
                max_retries,
            )

        duration_ms = (time.time() - start_time) * 1000
        usage_tokens = anthropic_usage_tokens(getattr(response, "usage", None))
        cost_usd = estimate_cost_usd(model_profile, usage_tokens)
        record_usage(
            model_profile.model, duration_ms=duration_ms, cost_usd=cost_usd, **usage_tokens
        )

        content_blocks = content_blocks_from_anthropic_response(response, tool_mode)
        for blk in content_blocks:
            if blk.get("type") == "thinking":
                thinking_text = blk.get("thinking") or blk.get("text") or ""
                if thinking_text:
                    reasoning_parts.append(str(thinking_text))
        if reasoning_parts:
            response_metadata["reasoning_content"] = "\n".join(reasoning_parts)
        # Streaming progress is handled via text_stream; final content retains thinking blocks.

        logger.info(
            "[anthropic_client] Response received",
            extra={
                "model": model_profile.model,
                "duration_ms": round(duration_ms, 2),
                "tool_mode": tool_mode,
                "tool_schemas": len(tool_schemas),
            },
        )

        return ProviderResponse(
            content_blocks=content_blocks,
            usage_tokens=usage_tokens,
            cost_usd=cost_usd,
            duration_ms=duration_ms,
            metadata=response_metadata,
        )
