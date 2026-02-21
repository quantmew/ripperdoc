"""OpenAI-compatible provider client."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, cast
from uuid import uuid4

import httpx
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
from ripperdoc.core.providers.errors import (
    ProviderAuthenticationError,
    ProviderModelNotFoundError,
    ProviderRateLimitError,
)
from ripperdoc.core.providers.error_mapping import (
    classify_mapped_error,
    map_api_status_error,
    map_bad_request_error,
    map_connection_error,
    map_permission_denied_error,
    run_with_exception_mapper,
)
from ripperdoc.core.message_utils import (
    build_openai_tool_schemas,
    content_blocks_from_openai_choice,
    estimate_cost_usd,
    normalize_tool_args,
    openai_usage_tokens,
)
from ripperdoc.core.tool import Tool
from ripperdoc.utils.log import get_logger
from ripperdoc.utils.session_usage import record_usage

logger = get_logger()


def _map_openai_exception(exc: Exception) -> Exception:
    """Normalize OpenAI SDK exceptions into shared provider error types."""
    exc_msg = str(exc)

    if isinstance(exc, httpx.TimeoutException):
        return map_connection_error(exc_msg)
    if isinstance(exc, httpx.RemoteProtocolError):
        return map_connection_error(exc_msg, retryable=True)
    if isinstance(exc, httpx.TransportError):
        return map_connection_error(exc_msg)

    if isinstance(exc, openai.APIConnectionError):
        return map_connection_error(exc_msg)
    if isinstance(exc, openai.RateLimitError):
        return ProviderRateLimitError(f"Rate limit exceeded: {exc_msg}")
    if isinstance(exc, openai.AuthenticationError):
        return ProviderAuthenticationError(f"Authentication failed: {exc_msg}")
    if isinstance(exc, openai.NotFoundError):
        return ProviderModelNotFoundError(f"Model not found: {exc_msg}")
    if isinstance(exc, openai.BadRequestError):
        return map_bad_request_error(exc_msg)
    if isinstance(exc, openai.PermissionDeniedError):
        return map_permission_denied_error(exc_msg)
    if isinstance(exc, openai.APIStatusError):
        status = getattr(exc, "status_code", "unknown")
        return map_api_status_error(exc_msg, status)

    return exc


async def _run_with_provider_error_mapping(request_fn: Callable[[], Awaitable[Any]]) -> Any:
    """Map OpenAI exceptions to shared provider errors before outer handling/retries."""
    return await run_with_exception_mapper(request_fn, _map_openai_exception)


def _classify_openai_error(exc: Exception) -> tuple[str, str]:
    """Classify an OpenAI exception into error code and user-friendly message."""
    mapped = classify_mapped_error(exc)
    if mapped is not None:
        return mapped

    exc_type = type(exc).__name__
    exc_msg = str(exc)

    if isinstance(exc, openai.PermissionDeniedError):
        lowered = exc_msg.lower()
        if "balance" in lowered or "insufficient" in lowered:
            return "insufficient_balance", f"Insufficient balance: {exc_msg}"
        return "permission_denied", f"Permission denied: {exc_msg}"

    if isinstance(exc, openai.BadRequestError):
        lowered = exc_msg.lower()
        if "context" in lowered or "token" in lowered:
            return "context_length_exceeded", f"Context length exceeded: {exc_msg}"
        if "content" in lowered and "policy" in lowered:
            return "content_policy_violation", f"Content policy violation: {exc_msg}"
        return "bad_request", f"Invalid request: {exc_msg}"

    static_mappings: List[tuple[type[Exception], str, str]] = [
        (openai.AuthenticationError, "authentication_error", "Authentication failed"),
        (openai.NotFoundError, "model_not_found", "Model not found"),
        (openai.RateLimitError, "rate_limit", "Rate limit exceeded"),
        (openai.APIConnectionError, "connection_error", "Connection error"),
        (httpx.TransportError, "connection_error", "Connection error"),
        (asyncio.TimeoutError, "timeout", "Request timed out"),
    ]
    for error_type, code, label in static_mappings:
        if isinstance(exc, error_type):
            return code, f"{label}: {exc_msg}"

    if isinstance(exc, openai.APIStatusError):
        return "api_error", f"API error ({exc.status_code}): {exc_msg}"

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
    """Best-effort vendor hint for OpenAI-compatible endpoints.

    If thinking_mode is explicitly set to "none" or "disabled", returns "none"
    to skip all thinking protocol handling.
    """
    override = getattr(model_profile, "thinking_mode", None)
    if isinstance(override, str) and override.strip():
        mode = override.strip().lower()
        # Allow explicit disable of thinking protocol
        if mode in ("disabled", "off"):
            return "none"
        return mode
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


def _apply_deepseek_thinking(
    extra_body: Dict[str, Any],
    _top_level: Dict[str, Any],
    max_thinking_tokens: int,
    _effort: Optional[str],
) -> None:
    if max_thinking_tokens != 0:
        extra_body["thinking"] = {"type": "enabled"}


def _apply_qwen_thinking(
    extra_body: Dict[str, Any],
    _top_level: Dict[str, Any],
    max_thinking_tokens: int,
    _effort: Optional[str],
) -> None:
    # Some qwen-compatible APIs do not support this parameter.
    if max_thinking_tokens > 0:
        extra_body["enable_thinking"] = True


def _apply_openrouter_thinking(
    extra_body: Dict[str, Any],
    _top_level: Dict[str, Any],
    max_thinking_tokens: int,
    _effort: Optional[str],
) -> None:
    if max_thinking_tokens > 0:
        extra_body["reasoning"] = {"max_tokens": max_thinking_tokens}


def _apply_gemini_openai_thinking(
    extra_body: Dict[str, Any],
    top_level: Dict[str, Any],
    max_thinking_tokens: int,
    effort: Optional[str],
) -> None:
    google_cfg: Dict[str, Any] = {}
    if max_thinking_tokens > 0:
        google_cfg["thinking_budget"] = max_thinking_tokens
        google_cfg["include_thoughts"] = True
    if google_cfg:
        extra_body["google"] = {"thinking_config": google_cfg}
    if effort:
        top_level["reasoning_effort"] = effort
        extra_body.setdefault("reasoning", {"effort": effort})


def _apply_default_reasoning_thinking(
    extra_body: Dict[str, Any],
    _top_level: Dict[str, Any],
    _max_thinking_tokens: int,
    effort: Optional[str],
) -> None:
    if effort:
        extra_body["reasoning"] = {"effort": effort}


def _build_thinking_kwargs(
    model_profile: ModelProfile, max_thinking_tokens: int
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Return (extra_body, top_level_kwargs) for thinking-enabled calls."""
    extra_body: Dict[str, Any] = {}
    top_level: Dict[str, Any] = {}
    vendor = _detect_openai_vendor(model_profile)

    # Skip thinking protocol if explicitly disabled
    if vendor == "none":
        return extra_body, top_level

    effort = _effort_from_tokens(max_thinking_tokens)

    handlers: Dict[str, Any] = {
        "deepseek": _apply_deepseek_thinking,
        "qwen": _apply_qwen_thinking,
        "openrouter": _apply_openrouter_thinking,
        "gemini_openai": _apply_gemini_openai_thinking,
    }
    handler = handlers.get(vendor, _apply_default_reasoning_thinking)
    handler(extra_body, top_level, max_thinking_tokens, effort)

    return extra_body, top_level


@dataclass
class _StreamAccumulator:
    """Accumulate state while consuming a streaming OpenAI response."""

    collected_text: List[str] = field(default_factory=list)
    streamed_tool_calls: Dict[int, Dict[str, Optional[str]]] = field(default_factory=dict)
    streamed_tool_text: List[str] = field(default_factory=list)
    usage_tokens: Dict[str, int] = field(default_factory=dict)
    reasoning_text: List[str] = field(default_factory=list)
    reasoning_details: List[Any] = field(default_factory=list)
    announced_tool_indexes: set[int] = field(default_factory=set)


async def _safe_emit_progress(
    progress_callback: Optional[ProgressCallback],
    chunk: str,
) -> None:
    """Best-effort streaming callback dispatch with guarded error handling."""
    if not progress_callback or not chunk:
        return
    try:
        await progress_callback(chunk)
    except (RuntimeError, ValueError, TypeError, OSError) as cb_exc:
        logger.warning(
            "[openai_client] Stream callback failed: %s: %s",
            type(cb_exc).__name__,
            cb_exc,
        )


def _build_openai_request_kwargs(
    *,
    model_profile: ModelProfile,
    openai_messages: List[Dict[str, object]],
    openai_tools: List[Dict[str, Any]],
    thinking_top_level: Dict[str, Any],
    thinking_extra_body: Dict[str, Any],
    stream: bool,
) -> Dict[str, Any]:
    """Build OpenAI chat completion kwargs shared by stream and non-stream calls."""
    kwargs: Dict[str, Any] = {
        "model": model_profile.model,
        "messages": cast(Any, openai_messages),
        "tools": openai_tools if openai_tools else None,
        "temperature": model_profile.temperature,
        "max_tokens": model_profile.max_tokens,
        **thinking_top_level,
    }
    if stream:
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}
    if thinking_extra_body:
        kwargs["extra_body"] = thinking_extra_body
    return kwargs


def _extract_delta_text(delta_content: Any) -> str:
    """Extract text delta from OpenAI streaming content payload."""
    text_delta = ""
    if not delta_content:
        return text_delta
    if isinstance(delta_content, list):
        for part in delta_content:
            text_val = getattr(part, "text", None) or getattr(part, "content", None)
            if isinstance(text_val, str):
                text_delta += text_val
        return text_delta
    if isinstance(delta_content, str):
        return delta_content
    return text_delta


def _collect_reasoning_delta(delta: Any, stream_state: _StreamAccumulator) -> None:
    """Capture reasoning deltas from a stream chunk."""
    delta_reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
    if isinstance(delta_reasoning, str):
        stream_state.reasoning_text.append(delta_reasoning)
    elif isinstance(delta_reasoning, list):
        for item in delta_reasoning:
            if isinstance(item, str):
                stream_state.reasoning_text.append(item)

    delta_reasoning_details = getattr(delta, "reasoning_details", None)
    if not delta_reasoning_details:
        return
    if isinstance(delta_reasoning_details, list):
        stream_state.reasoning_details.extend(delta_reasoning_details)
    else:
        stream_state.reasoning_details.append(delta_reasoning_details)


async def _collect_tool_deltas(
    delta: Any,
    stream_state: _StreamAccumulator,
    progress_callback: Optional[ProgressCallback],
) -> None:
    """Capture streamed tool call ids/names/arguments from delta chunks."""
    for tool_delta in getattr(delta, "tool_calls", []) or []:
        idx = getattr(tool_delta, "index", 0) or 0
        state = stream_state.streamed_tool_calls.get(idx, {"id": None, "name": None, "arguments": ""})

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
                await _safe_emit_progress(progress_callback, args_delta)

        if idx not in stream_state.announced_tool_indexes and state.get("name"):
            stream_state.announced_tool_indexes.add(idx)
            await _safe_emit_progress(progress_callback, f"[tool:{state['name']}]")

        stream_state.streamed_tool_calls[idx] = state


async def _consume_stream_chunk(
    chunk: Any,
    *,
    stream_state: _StreamAccumulator,
    can_stream_tools: bool,
    progress_callback: Optional[ProgressCallback],
) -> None:
    """Parse one stream chunk and update accumulators."""
    if getattr(chunk, "usage", None):
        stream_state.usage_tokens.update(openai_usage_tokens(chunk.usage))

    choices = getattr(chunk, "choices", None)
    if not choices or len(choices) == 0:
        return
    delta = getattr(choices[0], "delta", None)
    if not delta:
        return

    text_delta = _extract_delta_text(getattr(delta, "content", None))
    _collect_reasoning_delta(delta, stream_state)

    if text_delta:
        target_collector = stream_state.streamed_tool_text if can_stream_tools else stream_state.collected_text
        target_collector.append(text_delta)
        await _safe_emit_progress(progress_callback, text_delta)

    if can_stream_tools:
        await _collect_tool_deltas(delta, stream_state, progress_callback)


def _stream_has_output(stream_state: _StreamAccumulator) -> bool:
    """Return True when stream accumulators contain any meaningful output."""
    return bool(
        stream_state.collected_text
        or stream_state.streamed_tool_calls
        or stream_state.streamed_tool_text
        or stream_state.reasoning_text
    )


def _extract_choice_reasoning_metadata(choice: Any, response_metadata: Dict[str, Any]) -> None:
    """Populate reasoning-related response metadata from a non-stream choice."""
    message_obj = getattr(choice, "message", None) or choice
    reasoning_content = getattr(message_obj, "reasoning_content", None)
    if reasoning_content:
        response_metadata["reasoning_content"] = reasoning_content
    reasoning_field = getattr(message_obj, "reasoning", None)
    if reasoning_field:
        response_metadata["reasoning"] = reasoning_field
        if "reasoning_content" not in response_metadata and isinstance(reasoning_field, str):
            response_metadata["reasoning_content"] = reasoning_field
    reasoning_details = getattr(message_obj, "reasoning_details", None)
    if reasoning_details:
        response_metadata["reasoning_details"] = reasoning_details


def _build_stream_content_blocks(
    stream_state: _StreamAccumulator,
    *,
    can_stream_text: bool,
    can_stream_tools: bool,
) -> tuple[List[Dict[str, Any]], str]:
    """Build normalized content blocks for a successful stream response."""
    if can_stream_text:
        return [{"type": "text", "text": "".join(stream_state.collected_text)}], "stream"

    content_blocks: List[Dict[str, Any]] = []
    if can_stream_tools:
        if stream_state.streamed_tool_text:
            content_blocks.append({"type": "text", "text": "".join(stream_state.streamed_tool_text)})
        for idx in sorted(stream_state.streamed_tool_calls.keys()):
            call = stream_state.streamed_tool_calls[idx]
            name = call.get("name")
            if not name:
                continue
            content_blocks.append(
                {
                    "type": "tool_use",
                    "tool_use_id": call.get("id") or str(uuid4()),
                    "name": name,
                    "input": normalize_tool_args(call.get("arguments")),
                }
            )
    return content_blocks, "stream"


def _build_non_stream_content_blocks(
    openai_response: Any,
    *,
    model: Optional[str],
    tool_mode: str,
    response_metadata: Dict[str, Any],
) -> tuple[List[Dict[str, Any]], Optional[str]]:
    """Build normalized content blocks for a non-stream response."""
    response_choices = getattr(openai_response, "choices", None)
    if not response_choices or len(response_choices) == 0:
        logger.warning(
            "[openai_client] Empty choices in response",
            extra={"model": model},
        )
        return [{"type": "text", "text": ""}], "error"

    choice = response_choices[0]
    content_blocks = content_blocks_from_openai_choice(choice, tool_mode)
    finish_reason = cast(Optional[str], getattr(choice, "finish_reason", None))
    _extract_choice_reasoning_metadata(choice, response_metadata)
    return content_blocks, finish_reason


def _apply_stream_reasoning_metadata(
    stream_state: _StreamAccumulator,
    response_metadata: Dict[str, Any],
) -> None:
    """Attach collected stream reasoning fields to response metadata."""
    if stream_state.reasoning_text:
        joined = "".join(stream_state.reasoning_text)
        response_metadata["reasoning_content"] = joined
        response_metadata.setdefault("reasoning", joined)
    if stream_state.reasoning_details:
        response_metadata["reasoning_details"] = stream_state.reasoning_details


def _build_non_stream_empty_or_error_response(
    *,
    openai_response: Any,
    model: Optional[str],
    duration_ms: float,
    usage_tokens: Dict[str, int],
    cost_usd: float,
    response_metadata: Dict[str, Any],
) -> Optional[ProviderResponse]:
    """Return an error/empty response when non-stream response has no choices."""
    if openai_response and getattr(openai_response, "choices", None):
        return None

    error_msg = (
        getattr(openai_response, "msg", None)
        or getattr(openai_response, "message", None)
        or getattr(openai_response, "error", None)
    )
    error_status = getattr(openai_response, "status", None)
    if error_msg or error_status:
        error_text = f"API Error: {error_msg or 'Unknown error'}"
        if error_status:
            error_text = f"API Error ({error_status}): {error_msg or 'Unknown error'}"
        logger.error(
            "[openai_client] Non-standard error response from API",
            extra={
                "model": model,
                "error_status": error_status,
                "error_msg": error_msg,
            },
        )
        return ProviderResponse.create_error(
            error_code="api_error",
            error_message=error_text,
            duration_ms=duration_ms,
        )

    logger.warning(
        "[openai_client] No choices returned from OpenAI response",
        extra={"model": model},
    )
    return ProviderResponse(
        content_blocks=[{"type": "text", "text": "Model returned no content."}],
        usage_tokens=usage_tokens,
        cost_usd=cost_usd,
        duration_ms=duration_ms,
        metadata=response_metadata,
    )


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
        stream_state = _StreamAccumulator()
        response_metadata: Dict[str, Any] = {}

        can_stream_text = stream and tool_mode == "text" and not openai_tools
        can_stream_tools = stream and tool_mode != "text" and bool(openai_tools)
        can_stream = can_stream_text or can_stream_tools
        thinking_extra_body, thinking_top_level = _build_thinking_kwargs(
            model_profile, max_thinking_tokens
        )

        logger.debug(
            "[openai_client] Starting API request",
            extra={
                "model": model_profile.model,
                "api_base": model_profile.api_base,
                "request_timeout": request_timeout,
            },
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
                stream_kwargs = _build_openai_request_kwargs(
                    model_profile=model_profile,
                    openai_messages=openai_messages,
                    openai_tools=openai_tools,
                    thinking_top_level=thinking_top_level,
                    thinking_extra_body=thinking_extra_body,
                    stream=True,
                )
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
                    await _consume_stream_chunk(
                        chunk,
                        stream_state=stream_state,
                        can_stream_tools=can_stream_tools,
                        progress_callback=progress_callback,
                    )
                return {"usage": stream_state.usage_tokens}

            async def _non_stream_request() -> Any:
                kwargs = _build_openai_request_kwargs(
                    model_profile=model_profile,
                    openai_messages=openai_messages,
                    openai_tools=openai_tools,
                    thinking_top_level=thinking_top_level,
                    thinking_extra_body=thinking_extra_body,
                    stream=False,
                )
                return await client.chat.completions.create(  # type: ignore[call-overload]
                    **kwargs
                )

            timeout_for_call = None if can_stream else request_timeout
            openai_response: Any = await call_with_timeout_and_retries(
                lambda: _run_with_provider_error_mapping(
                    _stream_request if can_stream else _non_stream_request
                ),
                timeout_for_call,
                max_retries,
            )

            if (
                can_stream
                and not _stream_has_output(stream_state)
            ):
                logger.warning(
                    "[openai_client] Streaming returned no content; retrying without stream",
                    extra={"model": model_profile.model},
                )
                can_stream = False
                can_stream_text = False
                can_stream_tools = False
                openai_response = await call_with_timeout_and_retries(
                    lambda: _run_with_provider_error_mapping(_non_stream_request),
                    request_timeout,
                    max_retries,
                )

        duration_ms = (time.time() - start_time) * 1000
        usage_tokens = (
            stream_state.usage_tokens
            if can_stream
            else openai_usage_tokens(getattr(openai_response, "usage", None))
        )
        cost_usd = estimate_cost_usd(model_profile, usage_tokens)
        record_usage(
            model_profile.model, duration_ms=duration_ms, cost_usd=cost_usd, **usage_tokens
        )

        if not can_stream:
            empty_or_error = _build_non_stream_empty_or_error_response(
                openai_response=openai_response,
                model=model_profile.model,
                duration_ms=duration_ms,
                usage_tokens=usage_tokens,
                cost_usd=cost_usd,
                response_metadata=response_metadata,
            )
            if empty_or_error is not None:
                return empty_or_error

        content_blocks: List[Dict[str, Any]] = []
        finish_reason: Optional[str] = None
        if can_stream:
            content_blocks, finish_reason = _build_stream_content_blocks(
                stream_state,
                can_stream_text=can_stream_text,
                can_stream_tools=can_stream_tools,
            )
        else:
            content_blocks, finish_reason = _build_non_stream_content_blocks(
                openai_response,
                model=model_profile.model,
                tool_mode=tool_mode,
                response_metadata=response_metadata,
            )

        if can_stream:
            _apply_stream_reasoning_metadata(stream_state, response_metadata)

        logger.debug(
            "[openai_client] Response content blocks",
            extra={
                "model": model_profile.model,
                "content_blocks": json.dumps(content_blocks, ensure_ascii=False)[:1000],
                "usage_tokens": json.dumps(usage_tokens, ensure_ascii=False),
                "metadata": json.dumps(response_metadata, ensure_ascii=False)[:500],
            },
        )

        logger.debug(
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
