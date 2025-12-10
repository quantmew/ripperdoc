"""Anthropic provider client."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional

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
    ) -> ProviderResponse:
        start_time = time.time()
        tool_schemas = await build_anthropic_tool_schemas(tools)
        collected_text: List[str] = []

        anthropic_kwargs = {"base_url": model_profile.api_base}
        if model_profile.api_key:
            anthropic_kwargs["api_key"] = model_profile.api_key
        auth_token = getattr(model_profile, "auth_token", None)
        if auth_token:
            anthropic_kwargs["auth_token"] = auth_token

        normalized_messages = sanitize_tool_history(list(normalized_messages))

        async with await self._client(anthropic_kwargs) as client:

            async def _stream_request() -> Any:
                stream_cm = client.messages.stream(
                    model=model_profile.model,
                    max_tokens=model_profile.max_tokens,
                    system=system_prompt,
                    messages=normalized_messages,  # type: ignore[arg-type]
                    tools=tool_schemas if tool_schemas else None,  # type: ignore
                    temperature=model_profile.temperature,
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
                                except Exception:
                                    logger.exception("[anthropic_client] Stream callback failed")
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
        if stream and collected_text and tool_mode == "text":
            content_blocks = [{"type": "text", "text": "".join(collected_text)}]

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
        )
