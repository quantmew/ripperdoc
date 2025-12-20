"""OpenAI-compatible provider client."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, cast
from uuid import uuid4

import openai
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


def _classify_openai_error(exc: Exception) -> tuple[str, str]:
    """Classify an OpenAI exception into error code and user-friendly message."""
    exc_type = type(exc).__name__
    exc_msg = str(exc)

    if isinstance(exc, openai.AuthenticationError):
        return "authentication_error", f"Authentication failed: {exc_msg}"
    if isinstance(exc, openai.PermissionDeniedError):
        # Check for common permission denied reasons
        if "balance" in exc_msg.lower() or "insufficient" in exc_msg.lower():
            return "insufficient_balance", f"Insufficient balance: {exc_msg}"
        return "permission_denied", f"Permission denied: {exc_msg}"
    if isinstance(exc, openai.NotFoundError):
        return "model_not_found", f"Model not found: {exc_msg}"
    if isinstance(exc, openai.BadRequestError):
        # Check for context length errors
        if "context" in exc_msg.lower() or "token" in exc_msg.lower():
            return "context_length_exceeded", f"Context length exceeded: {exc_msg}"
        if "content" in exc_msg.lower() and "policy" in exc_msg.lower():
            return "content_policy_violation", f"Content policy violation: {exc_msg}"
        return "bad_request", f"Invalid request: {exc_msg}"
    if isinstance(exc, openai.RateLimitError):
        return "rate_limit", f"Rate limit exceeded: {exc_msg}"
    if isinstance(exc, openai.APIConnectionError):
        return "connection_error", f"Connection error: {exc_msg}"
    if isinstance(exc, openai.APIStatusError):
        return "api_error", f"API error ({exc.status_code}): {exc_msg}"
    if isinstance(exc, asyncio.TimeoutError):
        return "timeout", f"Request timed out: {exc_msg}"

    # Generic fallback
    return "unknown_error", f"Unexpected error ({exc_type}): {exc_msg}"


def _effort_from_tokens(max_thinking_tokens: int) -> Optional[str]:
    """Map a thinking token budget to a coarse effort label."""
    if max_thinking_tokens <= 0:
        return None
    if max_thinking_tokens <= 1024:
        return "low"
    if max_thinking_tokens <= 8192:
        return "medium"
    return "high"


def _detect_openai_vendor(model_profile: ModelProfile) -> str:
    """Best-effort vendor hint for OpenAI-compatible endpoints."""
    override = getattr(model_profile, "thinking_mode", None)
    if isinstance(override, str) and override.strip():
        return override.strip().lower()
    base = (model_profile.api_base or "").lower()
    name = (model_profile.model or "").lower()
    if "openrouter.ai" in base:
        return "openrouter"
    if "deepseek" in base or name.startswith("deepseek"):
        return "deepseek"
    if "dashscope" in base or "qwen" in name:
        return "qwen"
    if "generativelanguage.googleapis.com" in base or name.startswith("gemini"):
        return "gemini_openai"
    if "gpt-5" in name:
        return "openai"
    return "openai"


def _build_thinking_kwargs(
    model_profile: ModelProfile, max_thinking_tokens: int
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Return (extra_body, top_level_kwargs) for thinking-enabled calls."""
    extra_body: Dict[str, Any] = {}
    top_level: Dict[str, Any] = {}
    vendor = _detect_openai_vendor(model_profile)
    effort = _effort_from_tokens(max_thinking_tokens)

    if vendor == "deepseek":
        if max_thinking_tokens != 0:
            extra_body["thinking"] = {"type": "enabled"}
    elif vendor == "qwen":
        if max_thinking_tokens > 0:
            extra_body["enable_thinking"] = True
        elif max_thinking_tokens == 0:
            extra_body["enable_thinking"] = False
    elif vendor == "openrouter":
        if max_thinking_tokens > 0:
            extra_body["reasoning"] = {"max_tokens": max_thinking_tokens}
        elif max_thinking_tokens == 0:
            extra_body["reasoning"] = {"effort": "none"}
    elif vendor == "gemini_openai":
        google_cfg: Dict[str, Any] = {}
        if max_thinking_tokens > 0:
            google_cfg["thinking_budget"] = max_thinking_tokens
            google_cfg["include_thoughts"] = True
        if google_cfg:
            extra_body["google"] = {"thinking_config": google_cfg}
        if effort:
            top_level["reasoning_effort"] = effort
            extra_body.setdefault("reasoning", {"effort": effort})
    elif vendor == "openai":
        if effort:
            extra_body["reasoning"] = {"effort": effort}
    else:
        if effort:
            extra_body["reasoning"] = {"effort": effort}

    return extra_body, top_level


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
            error_code, error_message = _classify_openai_error(exc)
            logger.debug(
                "[openai_client] Exception details",
                extra={
                    "model": model_profile.model,
                    "exception_type": type(exc).__name__,
                    "exception_str": str(exc),
                    "error_code": error_code,
                },
            )
            logger.error(
                "[openai_client] API call failed",
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
        normalized_messages: List[Dict[str, Any]],
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
        openai_tools = await build_openai_tool_schemas(tools)
        openai_messages: List[Dict[str, object]] = [
            {"role": "system", "content": system_prompt}
        ] + sanitize_tool_history(list(normalized_messages))

        logger.debug(
            "[openai_client] Preparing request",
            extra={
                "model": model_profile.model,
                "tool_mode": tool_mode,
                "stream": stream,
                "max_thinking_tokens": max_thinking_tokens,
                "num_tools": len(openai_tools),
                "num_messages": len(openai_messages),
            },
        )
        collected_text: List[str] = []
        streamed_tool_calls: Dict[int, Dict[str, Optional[str]]] = {}
        streamed_tool_text: List[str] = []
        streamed_usage: Dict[str, int] = {}
        stream_reasoning_text: List[str] = []
        stream_reasoning_details: List[Any] = []
        response_metadata: Dict[str, Any] = {}

        can_stream_text = stream and tool_mode == "text" and not openai_tools
        can_stream_tools = stream and tool_mode != "text" and bool(openai_tools)
        can_stream = can_stream_text or can_stream_tools
        thinking_extra_body, thinking_top_level = _build_thinking_kwargs(
            model_profile, max_thinking_tokens
        )

        logger.debug(
            "[openai_client] Request parameters",
            extra={
                "model": model_profile.model,
                "thinking_extra_body": json.dumps(thinking_extra_body, ensure_ascii=False),
                "thinking_top_level": json.dumps(thinking_top_level, ensure_ascii=False),
                "messages_preview": json.dumps(openai_messages[:2], ensure_ascii=False)[:500],
            },
        )

        async with AsyncOpenAI(
            api_key=model_profile.api_key, base_url=model_profile.api_base
        ) as client:

            async def _stream_request() -> Dict[str, Dict[str, int]]:
                announced_tool_indexes: set[int] = set()
                stream_kwargs: Dict[str, Any] = {
                    "model": model_profile.model,
                    "messages": cast(Any, openai_messages),
                    "tools": openai_tools if openai_tools else None,
                    "temperature": model_profile.temperature,
                    "max_tokens": model_profile.max_tokens,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    **thinking_top_level,
                }
                if thinking_extra_body:
                    stream_kwargs["extra_body"] = thinking_extra_body
                logger.debug(
                    "[openai_client] Initiating stream request",
                    extra={
                        "model": model_profile.model,
                        "stream_kwargs": json.dumps(
                            {k: v for k, v in stream_kwargs.items() if k != "messages"},
                            ensure_ascii=False,
                        ),
                    },
                )
                stream_coro = client.chat.completions.create(  # type: ignore[call-overload]
                    **stream_kwargs
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
                    delta_reasoning = getattr(delta, "reasoning_content", None) or getattr(
                        delta, "reasoning", None
                    )
                    if isinstance(delta_reasoning, str):
                        stream_reasoning_text.append(delta_reasoning)
                    elif isinstance(delta_reasoning, list):
                        for item in delta_reasoning:
                            if isinstance(item, str):
                                stream_reasoning_text.append(item)
                    delta_reasoning_details = getattr(delta, "reasoning_details", None)
                    if delta_reasoning_details:
                        if isinstance(delta_reasoning_details, list):
                            stream_reasoning_details.extend(delta_reasoning_details)
                        else:
                            stream_reasoning_details.append(delta_reasoning_details)
                    if text_delta:
                        target_collector = (
                            streamed_tool_text if can_stream_tools else collected_text
                        )
                        target_collector.append(text_delta)
                        if progress_callback:
                            try:
                                await progress_callback(text_delta)
                            except (RuntimeError, ValueError, TypeError, OSError) as cb_exc:
                                logger.warning(
                                    "[openai_client] Stream callback failed: %s: %s",
                                    type(cb_exc).__name__,
                                    cb_exc,
                                )

                    # Tool call deltas for native tool mode
                    if not can_stream_tools:
                        continue

                    for tool_delta in getattr(delta, "tool_calls", []) or []:
                        idx = getattr(tool_delta, "index", 0) or 0
                        state = streamed_tool_calls.get(
                            idx, {"id": None, "name": None, "arguments": ""}
                        )

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
                                    except (RuntimeError, ValueError, TypeError, OSError) as cb_exc:
                                        logger.warning(
                                            "[openai_client] Stream callback failed: %s: %s",
                                            type(cb_exc).__name__,
                                            cb_exc,
                                        )

                        if idx not in announced_tool_indexes and state.get("name"):
                            announced_tool_indexes.add(idx)
                            if progress_callback:
                                try:
                                    await progress_callback(f"[tool:{state['name']}]")
                                except (RuntimeError, ValueError, TypeError, OSError) as cb_exc:
                                    logger.warning(
                                        "[openai_client] Stream callback failed: %s: %s",
                                        type(cb_exc).__name__,
                                        cb_exc,
                                    )

                        streamed_tool_calls[idx] = state

                return {"usage": streamed_usage}

            async def _non_stream_request() -> Any:
                kwargs: Dict[str, Any] = {
                    "model": model_profile.model,
                    "messages": cast(Any, openai_messages),
                    "tools": openai_tools if openai_tools else None,  # type: ignore[arg-type]
                    "temperature": model_profile.temperature,
                    "max_tokens": model_profile.max_tokens,
                    **thinking_top_level,
                }
                if thinking_extra_body:
                    kwargs["extra_body"] = thinking_extra_body
                return await client.chat.completions.create(  # type: ignore[call-overload]
                    **kwargs
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
        usage_tokens = (
            streamed_usage
            if can_stream
            else openai_usage_tokens(getattr(openai_response, "usage", None))
        )
        cost_usd = estimate_cost_usd(model_profile, usage_tokens)
        record_usage(
            model_profile.model, duration_ms=duration_ms, cost_usd=cost_usd, **usage_tokens
        )

        if not can_stream and (
            not openai_response or not getattr(openai_response, "choices", None)
        ):
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
                metadata=response_metadata,
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
            message_obj = getattr(choice, "message", None) or choice
            reasoning_content = getattr(message_obj, "reasoning_content", None)
            if reasoning_content:
                response_metadata["reasoning_content"] = reasoning_content
            reasoning_field = getattr(message_obj, "reasoning", None)
            if reasoning_field:
                response_metadata["reasoning"] = reasoning_field
                if "reasoning_content" not in response_metadata and isinstance(
                    reasoning_field, str
                ):
                    response_metadata["reasoning_content"] = reasoning_field
            reasoning_details = getattr(message_obj, "reasoning_details", None)
            if reasoning_details:
                response_metadata["reasoning_details"] = reasoning_details

        if can_stream:
            if stream_reasoning_text:
                joined = "".join(stream_reasoning_text)
                response_metadata["reasoning_content"] = joined
                response_metadata.setdefault("reasoning", joined)
            if stream_reasoning_details:
                response_metadata["reasoning_details"] = stream_reasoning_details

        logger.debug(
            "[openai_client] Response content blocks",
            extra={
                "model": model_profile.model,
                "content_blocks": json.dumps(content_blocks, ensure_ascii=False)[:1000],
                "usage_tokens": json.dumps(usage_tokens, ensure_ascii=False),
                "metadata": json.dumps(response_metadata, ensure_ascii=False)[:500],
            },
        )

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
            metadata=response_metadata,
        )
