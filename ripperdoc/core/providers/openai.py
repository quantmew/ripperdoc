"""OpenAI-compatible provider client."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, cast

from openai import AsyncOpenAI

from ripperdoc.core.config import ModelProfile
from ripperdoc.core.providers.base import (
    ProgressCallback,
    ProviderClient,
    ProviderResponse,
    call_with_timeout_and_retries,
    sanitize_tool_history,
)
from ripperdoc.core.query_utils import (
    build_openai_tool_schemas,
    content_blocks_from_openai_choice,
    estimate_cost_usd,
    openai_usage_tokens,
)
from ripperdoc.core.tool import Tool
from ripperdoc.utils.log import get_logger
from ripperdoc.utils.session_usage import record_usage

logger = get_logger()


class OpenAIClient(ProviderClient):
    """OpenAI-compatible client with streaming and non-streaming support."""

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
    ) -> ProviderResponse:
        start_time = time.time()
        openai_tools = await build_openai_tool_schemas(tools)
        openai_messages: List[Dict[str, object]] = [
            {"role": "system", "content": system_prompt}
        ] + sanitize_tool_history(list(normalized_messages))
        collected_text: List[str] = []

        can_stream = stream and tool_mode == "text" and not openai_tools

        async with AsyncOpenAI(
            api_key=model_profile.api_key, base_url=model_profile.api_base
        ) as client:

            async def _stream_request() -> Dict[str, Dict[str, int]]:
                stream_resp = await client.chat.completions.create(  # type: ignore[call-overload]
                    model=model_profile.model,
                    messages=cast(Any, openai_messages),
                    tools=None,
                    temperature=model_profile.temperature,
                    max_tokens=model_profile.max_tokens,
                    stream=True,
                )
                usage_tokens: Dict[str, int] = {}
                async for chunk in stream_resp:
                    delta = getattr(chunk.choices[0], "delta", None)
                    delta_content = getattr(delta, "content", None) if delta else None
                    text_delta = ""
                    if delta_content:
                        if isinstance(delta_content, list):
                            for part in delta_content:
                                text_val = getattr(part, "text", None) or getattr(
                                    part, "content", None
                                )
                                if isinstance(text_val, str):
                                    text_delta += text_val
                        elif isinstance(delta_content, str):
                            text_delta += delta_content
                    if text_delta:
                        collected_text.append(text_delta)
                        if progress_callback:
                            try:
                                await progress_callback(text_delta)
                            except Exception:
                                logger.exception("[openai_client] Stream callback failed")
                    if getattr(chunk, "usage", None):
                        usage_tokens = openai_usage_tokens(chunk.usage)
                return {"usage": usage_tokens}

            async def _non_stream_request() -> Any:
                return await client.chat.completions.create(  # type: ignore[call-overload]
                    model=model_profile.model,
                    messages=cast(Any, openai_messages),
                    tools=openai_tools if openai_tools else None,  # type: ignore[arg-type]
                    temperature=model_profile.temperature,
                    max_tokens=model_profile.max_tokens,
                )

            openai_response: Any = await call_with_timeout_and_retries(
                _stream_request if can_stream else _non_stream_request,
                request_timeout,
                max_retries,
            )

        duration_ms = (time.time() - start_time) * 1000
        usage_tokens = openai_usage_tokens(getattr(openai_response, "usage", None))
        cost_usd = estimate_cost_usd(model_profile, usage_tokens)
        record_usage(
            model_profile.model, duration_ms=duration_ms, cost_usd=cost_usd, **usage_tokens
        )

        finish_reason: Optional[str]
        if can_stream:
            content_blocks = [{"type": "text", "text": "".join(collected_text)}]
            finish_reason = "stream"
        else:
            choice = openai_response.choices[0]
            content_blocks = content_blocks_from_openai_choice(choice, tool_mode)
            finish_reason = cast(Optional[str], getattr(choice, "finish_reason", None))

        logger.info(
            "[openai_client] Response received",
            extra={
                "model": model_profile.model,
                "duration_ms": round(duration_ms, 2),
                "tool_mode": tool_mode,
                "tool_count": len(openai_tools),
                "finish_reason": finish_reason,
            },
        )

        return ProviderResponse(
            content_blocks=content_blocks,
            usage_tokens=usage_tokens,
            cost_usd=cost_usd,
            duration_ms=duration_ms,
        )
