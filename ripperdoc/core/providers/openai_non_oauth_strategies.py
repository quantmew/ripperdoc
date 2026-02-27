"""Strategy implementations for non-OAuth OpenAI-compatible calls."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol, cast
from uuid import uuid4

from openai import AsyncOpenAI

from ripperdoc.core.config import ModelProfile
from ripperdoc.core.message_utils import (
    build_openai_tool_schemas,
    content_blocks_from_openai_choice,
    estimate_cost_usd,
    normalize_tool_args,
    openai_usage_tokens,
)
from ripperdoc.core.providers.base import (
    ProgressCallback,
    ProviderResponse,
    call_with_timeout_and_retries,
    iter_with_timeout,
    sanitize_tool_history,
)
from ripperdoc.core.providers.openai_responses import (
    build_input_from_normalized_messages,
    convert_chat_function_tools_to_responses_tools,
    extract_content_blocks_from_output,
    extract_text_usage_from_sse_events,
)
from ripperdoc.core.tool import Tool
from ripperdoc.utils.log import get_logger
from ripperdoc.utils.session_usage import record_usage
from ripperdoc.utils.user_agent import build_user_agent

logger = get_logger()

ThinkingKwargsBuilder = Callable[[ModelProfile, int], tuple[Dict[str, Any], Dict[str, Any]]]
ProviderErrorMapper = Callable[[Callable[[], Awaitable[Any]]], Awaitable[Any]]
ProgressEmitter = Callable[[Optional[ProgressCallback], str], Awaitable[None]]


class NonOAuthOpenAIStrategy(Protocol):
    """Unified call interface for non-OAuth OpenAI-compatible protocol strategies."""

    name: str

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
        start_time: float,
        default_headers: Optional[Dict[str, str]] = None,
    ) -> ProviderResponse: ...


@dataclass
class _StreamAccumulator:
    """Accumulate state while consuming a streaming OpenAI chat response."""

    collected_text: List[str] = field(default_factory=list)
    streamed_tool_calls: Dict[int, Dict[str, Optional[str]]] = field(default_factory=dict)
    streamed_tool_text: List[str] = field(default_factory=list)
    usage_tokens: Dict[str, int] = field(default_factory=dict)
    reasoning_text: List[str] = field(default_factory=list)
    reasoning_details: List[Any] = field(default_factory=list)
    announced_tool_indexes: set[int] = field(default_factory=set)


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
    safe_emit_progress: Optional[ProgressEmitter] = None,
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
                if safe_emit_progress is not None:
                    await safe_emit_progress(progress_callback, args_delta)
                elif progress_callback:
                    await progress_callback(args_delta)

        if idx not in stream_state.announced_tool_indexes and state.get("name"):
            stream_state.announced_tool_indexes.add(idx)
            if safe_emit_progress is not None:
                await safe_emit_progress(progress_callback, f"[tool:{state['name']}]")
            elif progress_callback:
                await progress_callback(f"[tool:{state['name']}]")

        stream_state.streamed_tool_calls[idx] = state


async def _consume_stream_chunk(
    chunk: Any,
    *,
    stream_state: _StreamAccumulator,
    can_stream_tools: bool,
    progress_callback: Optional[ProgressCallback],
    safe_emit_progress: Optional[ProgressEmitter] = None,
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
        if safe_emit_progress is not None:
            await safe_emit_progress(progress_callback, text_delta)
        elif progress_callback:
            await progress_callback(text_delta)

    if can_stream_tools:
        await _collect_tool_deltas(
            delta,
            stream_state,
            progress_callback,
            safe_emit_progress,
        )


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
    """Build normalized content blocks for a successful chat stream response."""
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
    """Build normalized content blocks for a non-stream chat response."""
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
    """Return an error/empty response when non-stream chat response has no choices."""
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


def _build_responses_request_kwargs(
    *,
    model_profile: ModelProfile,
    instructions: str,
    response_input: List[Dict[str, Any]],
    response_tools: List[Dict[str, Any]],
    thinking_top_level: Dict[str, Any],
    thinking_extra_body: Dict[str, Any],
    stream: bool,
) -> Dict[str, Any]:
    """Build OpenAI Responses API kwargs shared by stream and non-stream calls."""
    kwargs: Dict[str, Any] = {
        "model": model_profile.model,
        "instructions": instructions,
        "input": response_input,
        "tools": response_tools if response_tools else None,
        "temperature": model_profile.temperature,
        "max_output_tokens": model_profile.max_tokens,
        "store": False,
    }

    reasoning_obj: Optional[Dict[str, Any]] = None
    raw_reasoning = thinking_extra_body.get("reasoning")
    if isinstance(raw_reasoning, dict) and raw_reasoning:
        reasoning_obj = cast(Dict[str, Any], raw_reasoning)
    if reasoning_obj is None:
        effort = thinking_top_level.get("reasoning_effort")
        if isinstance(effort, str) and effort:
            reasoning_obj = {"effort": effort}
    if reasoning_obj:
        kwargs["reasoning"] = reasoning_obj

    if stream:
        kwargs["stream"] = True

    return kwargs


def _response_obj_to_dict(response_obj: Any) -> Dict[str, Any]:
    """Best-effort conversion of Responses API object to dict."""
    if isinstance(response_obj, dict):
        return response_obj
    for method_name in ("model_dump", "to_dict"):
        method = getattr(response_obj, method_name, None)
        if callable(method):
            try:
                maybe = method(exclude_none=True)
            except TypeError:
                maybe = method()
            if isinstance(maybe, dict):
                return maybe
    if hasattr(response_obj, "__dict__") and isinstance(response_obj.__dict__, dict):
        return {k: v for k, v in response_obj.__dict__.items() if not k.startswith("_")}
    return {}


def _response_event_to_dict(event: Any) -> Dict[str, Any]:
    """Best-effort conversion of one Responses stream event object to dict."""
    if isinstance(event, dict):
        return event
    payload = _response_obj_to_dict(event)
    if payload:
        return payload
    event_type = getattr(event, "type", None)
    if isinstance(event_type, str) and event_type:
        fallback: Dict[str, Any] = {"type": event_type}
        for attr in ("delta", "response"):
            value = getattr(event, attr, None)
            if value is None:
                continue
            if isinstance(value, dict):
                fallback[attr] = value
            else:
                maybe = _response_obj_to_dict(value)
                fallback[attr] = maybe if maybe else value
        return fallback
    return {}


def _responses_error_message(payload: Dict[str, Any]) -> Optional[str]:
    """Extract a user-facing error message from Responses payload."""
    error_obj = payload.get("error")
    if isinstance(error_obj, dict):
        message = error_obj.get("message")
        if isinstance(message, str) and message:
            return message
    return None


def _resolve_strategy_name(model_profile: ModelProfile) -> str:
    """Pick chat/responses strategy from model metadata."""
    explicit_mode = (getattr(model_profile, "openai_mode", None) or "").strip().lower()
    if explicit_mode == "responses":
        return "responses"
    if explicit_mode == "legacy":
        return "chat"

    mode = (model_profile.mode or "").strip().lower()
    if mode == "responses":
        return "responses"
    return "chat"


class _BaseNonOAuthStrategy:
    """Shared dependency holder for concrete non-OAuth strategies."""

    def __init__(
        self,
        *,
        build_thinking_kwargs: ThinkingKwargsBuilder,
        run_with_provider_error_mapping: ProviderErrorMapper,
        safe_emit_progress: ProgressEmitter,
    ) -> None:
        self._build_thinking_kwargs = build_thinking_kwargs
        self._run_with_provider_error_mapping = run_with_provider_error_mapping
        self._safe_emit_progress = safe_emit_progress


class OpenAIChatStrategy(_BaseNonOAuthStrategy):
    """Classic chat.completions API strategy."""

    name = "chat"

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
        start_time: float,
        default_headers: Optional[Dict[str, str]] = None,
    ) -> ProviderResponse:
        openai_tools = await build_openai_tool_schemas(tools)
        openai_messages: List[Dict[str, object]] = [
            {"role": "system", "content": system_prompt}
        ] + sanitize_tool_history(list(normalized_messages))

        logger.debug(
            "[openai_client] Preparing request",
            extra={
                "model": model_profile.model,
                "strategy": self.name,
                "tool_mode": tool_mode,
                "stream": stream,
                "max_thinking_tokens": max_thinking_tokens,
                "num_tools": len(openai_tools),
                "num_messages": len(openai_messages),
            },
        )
        stream_state = _StreamAccumulator()
        response_metadata: Dict[str, Any] = {"strategy": self.name}

        can_stream_text = stream and tool_mode == "text" and not openai_tools
        can_stream_tools = stream and tool_mode != "text" and bool(openai_tools)
        can_stream = can_stream_text or can_stream_tools
        thinking_extra_body, thinking_top_level = self._build_thinking_kwargs(
            model_profile, max_thinking_tokens
        )

        logger.debug(
            "[openai_client] Starting API request",
            extra={
                "model": model_profile.model,
                "strategy": self.name,
                "api_base": model_profile.api_base,
                "request_timeout": request_timeout,
            },
        )

        logger.debug(
            "[openai_client] Request parameters",
            extra={
                "model": model_profile.model,
                "strategy": self.name,
                "thinking_extra_body": json.dumps(thinking_extra_body, ensure_ascii=False),
                "thinking_top_level": json.dumps(thinking_top_level, ensure_ascii=False),
                "messages_preview": json.dumps(openai_messages[:2], ensure_ascii=False)[:500],
            },
        )

        headers = {"User-Agent": build_user_agent()}
        if default_headers:
            headers.update(default_headers)

        async with AsyncOpenAI(
            api_key=model_profile.api_key,
            base_url=model_profile.api_base,
            default_headers=headers,
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
                        "strategy": self.name,
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
                        safe_emit_progress=self._safe_emit_progress,
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
                lambda: self._run_with_provider_error_mapping(
                    _stream_request if can_stream else _non_stream_request
                ),
                timeout_for_call,
                max_retries,
            )

            if can_stream and not _stream_has_output(stream_state):
                logger.warning(
                    "[openai_client] Streaming returned no content; retrying without stream",
                    extra={"model": model_profile.model, "strategy": self.name},
                )
                can_stream = False
                can_stream_text = False
                can_stream_tools = False
                openai_response = await call_with_timeout_and_retries(
                    lambda: self._run_with_provider_error_mapping(_non_stream_request),
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
            "[openai_client] Response received",
            extra={
                "model": model_profile.model,
                "strategy": self.name,
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


class OpenAIResponsesStrategy(_BaseNonOAuthStrategy):
    """Responses API strategy for models cataloged as mode='responses'."""

    name = "responses"

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
        start_time: float,
        default_headers: Optional[Dict[str, str]] = None,
    ) -> ProviderResponse:
        openai_tools = await build_openai_tool_schemas(tools)
        response_tools = convert_chat_function_tools_to_responses_tools(openai_tools)
        sanitized_messages = sanitize_tool_history(list(normalized_messages))
        response_input = build_input_from_normalized_messages(
            cast(List[Dict[str, Any]], sanitized_messages),
            assistant_text_type="output_text",
            include_phase=True,
        )
        if not response_input:
            response_input = [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Continue."}],
                }
            ]

        instructions = (system_prompt or "").strip() or "You are a helpful assistant."
        thinking_extra_body, thinking_top_level = self._build_thinking_kwargs(
            model_profile, max_thinking_tokens
        )

        logger.debug(
            "[openai_client] Preparing request",
            extra={
                "model": model_profile.model,
                "strategy": self.name,
                "tool_mode": tool_mode,
                "stream": stream,
                "max_thinking_tokens": max_thinking_tokens,
                "num_tools": len(response_tools),
                "num_messages": len(response_input),
            },
        )

        headers = {"User-Agent": build_user_agent()}
        if default_headers:
            headers.update(default_headers)

        async with AsyncOpenAI(
            api_key=model_profile.api_key,
            base_url=model_profile.api_base,
            default_headers=headers,
        ) as client:

            async def _stream_request() -> Dict[str, Any]:
                kwargs = _build_responses_request_kwargs(
                    model_profile=model_profile,
                    instructions=instructions,
                    response_input=response_input,
                    response_tools=response_tools,
                    thinking_top_level=thinking_top_level,
                    thinking_extra_body=thinking_extra_body,
                    stream=True,
                )
                stream_coro = client.responses.create(**kwargs)
                stream_resp = (
                    await asyncio.wait_for(stream_coro, timeout=request_timeout)
                    if request_timeout and request_timeout > 0
                    else await stream_coro
                )

                events: List[Dict[str, Any]] = []
                text_parts: List[str] = []
                async for event in iter_with_timeout(stream_resp, request_timeout):
                    event_payload = _response_event_to_dict(event)
                    if event_payload:
                        events.append(event_payload)
                    event_type = str(event_payload.get("type") or "")
                    if event_type == "response.output_text.delta":
                        delta = event_payload.get("delta")
                        if isinstance(delta, str) and delta:
                            text_parts.append(delta)
                            await self._safe_emit_progress(progress_callback, delta)

                return {
                    "events": events,
                    "streamed_text": "".join(text_parts),
                }

            async def _non_stream_request() -> Any:
                kwargs = _build_responses_request_kwargs(
                    model_profile=model_profile,
                    instructions=instructions,
                    response_input=response_input,
                    response_tools=response_tools,
                    thinking_top_level=thinking_top_level,
                    thinking_extra_body=thinking_extra_body,
                    stream=False,
                )
                return await client.responses.create(**kwargs)

            timeout_for_call = None if stream else request_timeout
            raw_response: Any = await call_with_timeout_and_retries(
                lambda: self._run_with_provider_error_mapping(
                    _stream_request if stream else _non_stream_request
                ),
                timeout_for_call,
                max_retries,
            )

        duration_ms = (time.time() - start_time) * 1000
        usage_tokens: Dict[str, int] = {"input_tokens": 0, "output_tokens": 0}
        content_blocks: List[Dict[str, Any]] = []
        finish_reason: Optional[str] = None
        response_metadata: Dict[str, Any] = {"strategy": self.name}

        if stream:
            stream_payload = cast(Dict[str, Any], raw_response)
            events = cast(List[Dict[str, Any]], stream_payload.get("events") or [])
            streamed_text = cast(str, stream_payload.get("streamed_text") or "")

            sse_text, sse_usage_tokens, final_response = extract_text_usage_from_sse_events(events)
            if any(int(v or 0) > 0 for v in sse_usage_tokens.values()):
                usage_tokens = sse_usage_tokens

            final_payload: Dict[str, Any] = {}
            if isinstance(final_response, dict):
                final_payload = {
                    "output": final_response.get("output"),
                    "usage": final_response.get("usage"),
                    "error": final_response.get("error"),
                    "incomplete_details": final_response.get("incomplete_details"),
                    "status": final_response.get("status"),
                }
                if not any(int(v or 0) > 0 for v in usage_tokens.values()):
                    usage_tokens = openai_usage_tokens(final_response.get("usage"))
                if isinstance(final_response.get("incomplete_details"), dict):
                    response_metadata["incomplete_details"] = final_response.get("incomplete_details")
                if isinstance(final_response.get("status"), str):
                    response_metadata["status"] = final_response.get("status")

            content_blocks = extract_content_blocks_from_output(final_payload)
            stream_text = sse_text or streamed_text
            if not content_blocks and stream_text:
                content_blocks = [{"type": "text", "text": stream_text}]
            finish_reason = "stream"
            stream_error_message = _responses_error_message(final_payload)
            if stream_error_message:
                return ProviderResponse.create_error(
                    error_code="api_error",
                    error_message=stream_error_message,
                    duration_ms=duration_ms,
                )
        else:
            response_obj = raw_response
            response_payload = _response_obj_to_dict(response_obj)
            usage_tokens = openai_usage_tokens(
                getattr(response_obj, "usage", None) or response_payload.get("usage")
            )
            non_stream_error_message = _responses_error_message(response_payload)
            if non_stream_error_message:
                return ProviderResponse.create_error(
                    error_code="api_error",
                    error_message=non_stream_error_message,
                    duration_ms=duration_ms,
                )
            content_blocks = extract_content_blocks_from_output(response_payload)
            output_text = getattr(response_obj, "output_text", None)
            if isinstance(output_text, str) and output_text and not content_blocks:
                content_blocks = [{"type": "text", "text": output_text}]
            finish_reason = cast(Optional[str], response_payload.get("status"))
            if isinstance(finish_reason, str):
                response_metadata["status"] = finish_reason
            incomplete_details = response_payload.get("incomplete_details")
            if isinstance(incomplete_details, dict):
                response_metadata["incomplete_details"] = incomplete_details

        if not content_blocks:
            content_blocks = [{"type": "text", "text": "Model returned no content."}]

        cost_usd = estimate_cost_usd(model_profile, usage_tokens)
        record_usage(
            model_profile.model, duration_ms=duration_ms, cost_usd=cost_usd, **usage_tokens
        )

        logger.debug(
            "[openai_client] Response received",
            extra={
                "model": model_profile.model,
                "strategy": self.name,
                "duration_ms": round(duration_ms, 2),
                "tool_mode": tool_mode,
                "tool_count": len(response_tools),
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


def build_non_oauth_openai_strategy(
    *,
    model_profile: ModelProfile,
    build_thinking_kwargs: ThinkingKwargsBuilder,
    run_with_provider_error_mapping: ProviderErrorMapper,
    safe_emit_progress: ProgressEmitter,
) -> NonOAuthOpenAIStrategy:
    """Construct the appropriate non-OAuth strategy for a model profile."""
    name = _resolve_strategy_name(model_profile)
    base_kwargs = {
        "build_thinking_kwargs": build_thinking_kwargs,
        "run_with_provider_error_mapping": run_with_provider_error_mapping,
        "safe_emit_progress": safe_emit_progress,
    }
    if name == "responses":
        return OpenAIResponsesStrategy(**base_kwargs)
    return OpenAIChatStrategy(**base_kwargs)


__all__ = [
    "NonOAuthOpenAIStrategy",
    "OpenAIChatStrategy",
    "OpenAIResponsesStrategy",
    "build_non_oauth_openai_strategy",
    "_StreamAccumulator",
    "_apply_stream_reasoning_metadata",
    "_build_non_stream_content_blocks",
    "_build_non_stream_empty_or_error_response",
    "_build_openai_request_kwargs",
    "_build_stream_content_blocks",
    "_consume_stream_chunk",
]
