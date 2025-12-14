"""OpenAI-compatible provider client."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional, cast
from uuid import uuid4

from openai import AsyncOpenAI

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
    build_openai_tool_schemas,
    content_blocks_from_openai_choice,
    estimate_cost_usd,
    _normalize_tool_args,
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
        streamed_tool_calls: Dict[int, Dict[str, Optional[str]]] = {}
        streamed_tool_text: List[str] = []
        streamed_usage: Dict[str, int] = {}

        can_stream_text = stream and tool_mode == "text" and not openai_tools
        can_stream_tools = stream and tool_mode != "text" and bool(openai_tools)
        can_stream = can_stream_text or can_stream_tools

        async with AsyncOpenAI(
            api_key=model_profile.api_key, base_url=model_profile.api_base
        ) as client:

            async def _stream_request() -> Dict[str, Dict[str, int]]:
                announced_tool_indexes: set[int] = set()
                stream_coro = client.chat.completions.create(  # type: ignore[call-overload]
                    model=model_profile.model,
                    messages=cast(Any, openai_messages),
                    tools=openai_tools if can_stream_tools else None,
                    temperature=model_profile.temperature,
                    max_tokens=model_profile.max_tokens,
                    stream=True,
                    stream_options={"include_usage": True},
                )
                stream_resp = (
                    await asyncio.wait_for(stream_coro, timeout=request_timeout)
                    if request_timeout and request_timeout > 0
                    else await stream_coro
                )
                async for chunk in iter_with_timeout(stream_resp, request_timeout):
                    if getattr(chunk, "usage", None):
                        streamed_usage.update(openai_usage_tokens(chunk.usage))

                    if not getattr(chunk, "choices", None):
                        continue
                    delta = getattr(chunk.choices[0], "delta", None)
                    if not delta:
                        continue

                    # Text deltas (rare in native tool mode but supported)
                    delta_content = getattr(delta, "content", None)
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
                        target_collector = streamed_tool_text if can_stream_tools else collected_text
                        target_collector.append(text_delta)
                        if progress_callback:
                            try:
                                await progress_callback(text_delta)
                            except Exception:
                                logger.exception("[openai_client] Stream callback failed")

                    # Tool call deltas for native tool mode
                    if not can_stream_tools:
                        continue

                    for tool_delta in getattr(delta, "tool_calls", []) or []:
                        idx = getattr(tool_delta, "index", 0) or 0
                        state = streamed_tool_calls.get(idx, {"id": None, "name": None, "arguments": ""})

                        if getattr(tool_delta, "id", None):
                            state["id"] = tool_delta.id

                        function_delta = getattr(tool_delta, "function", None)
                        if function_delta:
                            fn_name = getattr(function_delta, "name", None)
                            if fn_name:
                                state["name"] = fn_name
                            args_delta = getattr(function_delta, "arguments", None)
                            if args_delta:
                                state["arguments"] = (state.get("arguments") or "") + args_delta
                                if progress_callback:
                                    try:
                                        await progress_callback(args_delta)
                                    except Exception:
                                        logger.exception("[openai_client] Stream callback failed")

                        if idx not in announced_tool_indexes and state.get("name"):
                            announced_tool_indexes.add(idx)
                            if progress_callback:
                                try:
                                    await progress_callback(f"[tool:{state['name']}]")
                                except Exception:
                                    logger.exception("[openai_client] Stream callback failed")

                        streamed_tool_calls[idx] = state

                return {"usage": streamed_usage}

            async def _non_stream_request() -> Any:
                return await client.chat.completions.create(  # type: ignore[call-overload]
                    model=model_profile.model,
                    messages=cast(Any, openai_messages),
                    tools=openai_tools if openai_tools else None,  # type: ignore[arg-type]
                    temperature=model_profile.temperature,
                    max_tokens=model_profile.max_tokens,
                )

            timeout_for_call = None if can_stream else request_timeout
            openai_response: Any = await call_with_timeout_and_retries(
                _stream_request if can_stream else _non_stream_request,
                timeout_for_call,
                max_retries,
            )

            if (
                can_stream_text
                and not collected_text
                and not streamed_tool_calls
                and not streamed_tool_text
            ):
                logger.debug(
                    "[openai_client] Streaming returned no content; retrying without stream",
                    extra={"model": model_profile.model},
                )
                can_stream = False
                can_stream_text = False
                can_stream_tools = False
                openai_response = await call_with_timeout_and_retries(
                    _non_stream_request, request_timeout, max_retries
                )

        duration_ms = (time.time() - start_time) * 1000
        usage_tokens = streamed_usage if can_stream else openai_usage_tokens(
            getattr(openai_response, "usage", None)
        )
        cost_usd = estimate_cost_usd(model_profile, usage_tokens)
        record_usage(
            model_profile.model, duration_ms=duration_ms, cost_usd=cost_usd, **usage_tokens
        )

        if not can_stream and (not openai_response or not getattr(openai_response, "choices", None)):
            logger.warning(
                "[openai_client] No choices returned from OpenAI response",
                extra={"model": model_profile.model},
            )
            empty_text = "Model returned no content."
            return ProviderResponse(
                content_blocks=[{"type": "text", "text": empty_text}],
                usage_tokens=usage_tokens,
                cost_usd=cost_usd,
                duration_ms=duration_ms,
            )

        content_blocks: List[Dict[str, Any]] = []
        finish_reason: Optional[str] = None
        if can_stream_text:
            content_blocks = [{"type": "text", "text": "".join(collected_text)}]
            finish_reason = "stream"
        elif can_stream_tools:
            if streamed_tool_text:
                content_blocks.append({"type": "text", "text": "".join(streamed_tool_text)})
            for idx in sorted(streamed_tool_calls.keys()):
                call = streamed_tool_calls[idx]
                name = call.get("name")
                if not name:
                    continue
                tool_use_id = call.get("id") or str(uuid4())
                content_blocks.append(
                    {
                        "type": "tool_use",
                        "tool_use_id": tool_use_id,
                        "name": name,
                        "input": _normalize_tool_args(call.get("arguments")),
                    }
                )
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
