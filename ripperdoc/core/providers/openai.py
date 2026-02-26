"""OpenAI-compatible provider client."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional

import httpx
import openai

from ripperdoc.core.config import ModelProfile, ProtocolType
from ripperdoc.core.providers.base import ProgressCallback, ProviderClient, ProviderResponse
from ripperdoc.core.providers.error_mapping import (
    classify_mapped_error,
    map_api_status_error,
    map_bad_request_error,
    map_connection_error,
    map_permission_denied_error,
    run_with_exception_mapper,
)
from ripperdoc.core.providers.errors import (
    ProviderAuthenticationError,
    ProviderModelNotFoundError,
    ProviderRateLimitError,
)
from ripperdoc.core.providers.openai_non_oauth_strategies import (
    _StreamAccumulator,
    _apply_stream_reasoning_metadata,
    _build_non_stream_content_blocks,
    _build_non_stream_empty_or_error_response,
    _build_openai_request_kwargs,
    _build_stream_content_blocks,
    _consume_stream_chunk,
    build_non_oauth_openai_strategy,
)
from ripperdoc.core.providers.openai_oauth_codex import call_oauth_codex
from ripperdoc.core.providers.openai_responses import (
    build_input_from_normalized_messages as _build_codex_responses_input,
    convert_chat_function_tools_to_responses_tools as _build_codex_oauth_tools,
    extract_content_blocks_from_output as _extract_content_blocks_from_responses_payload,
    extract_text_usage_from_sse_events as _extract_from_codex_sse_events,
    parse_sse_json_events as _parse_sse_json_events,
)
from ripperdoc.core.tool import Tool
from ripperdoc.utils.log import get_logger

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


class OpenAIClient(ProviderClient):
    """OpenAI-compatible client with protocol strategy dispatch."""

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
        if model_profile.protocol == ProtocolType.OAUTH:
            return await self._call_oauth_codex(
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

        strategy = build_non_oauth_openai_strategy(
            model_profile=model_profile,
            build_thinking_kwargs=_build_thinking_kwargs,
            run_with_provider_error_mapping=_run_with_provider_error_mapping,
            safe_emit_progress=_safe_emit_progress,
        )
        logger.debug(
            "[openai_client] Selected non-oauth strategy",
            extra={"model": model_profile.model, "strategy": strategy.name},
        )

        return await strategy.call(
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

    async def _call_oauth_codex(
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
        return await call_oauth_codex(
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
            build_thinking_kwargs=_build_thinking_kwargs,
            run_with_provider_error_mapping=_run_with_provider_error_mapping,
            safe_emit_progress=_safe_emit_progress,
        )
