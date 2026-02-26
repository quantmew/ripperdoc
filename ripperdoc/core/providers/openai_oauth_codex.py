"""Codex OAuth request path for OpenAI provider."""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Dict, List, Optional, cast

import httpx

from ripperdoc.core.config import ModelProfile
from ripperdoc.core.message_utils import (
    build_openai_tool_schemas,
    content_blocks_from_openai_choice,
    estimate_cost_usd,
    openai_usage_tokens,
)
from ripperdoc.core.oauth import OAuthTokenType, add_oauth_token, get_oauth_token
from ripperdoc.core.oauth_codex import CodexOAuthError, refresh_codex_access_token
from ripperdoc.core.providers.base import (
    ProgressCallback,
    ProviderResponse,
    call_with_timeout_and_retries,
    sanitize_tool_history,
)
from ripperdoc.core.providers.error_mapping import (
    classify_mapped_error,
    map_api_status_error,
)
from ripperdoc.core.providers.openai_responses import (
    build_input_from_normalized_messages,
    convert_chat_function_tools_to_responses_tools,
    extract_content_blocks_from_output,
    extract_text_usage_from_sse_events,
    extract_unsupported_parameter_name as extract_unsupported_parameter_name_from_responses,
    parse_sse_json_events,
)
from ripperdoc.core.tool import Tool
from ripperdoc.utils.log import get_logger
from ripperdoc.utils.session_usage import record_usage
from ripperdoc.utils.user_agent import build_user_agent

logger = get_logger()
_CODEX_OAUTH_ENDPOINT = "https://chatgpt.com/backend-api/codex/responses"

ThinkingKwargsBuilder = Callable[[ModelProfile, int], tuple[Dict[str, Any], Dict[str, Any]]]
ProviderErrorMapper = Callable[[Callable[[], Awaitable[Any]]], Awaitable[Any]]
ProgressEmitter = Callable[[Optional[ProgressCallback], str], Awaitable[None]]

# Backward-compatible helper aliases used by tests/importers.
_build_codex_oauth_tools = convert_chat_function_tools_to_responses_tools
_build_codex_responses_input = build_input_from_normalized_messages
_extract_content_blocks_from_responses_payload = extract_content_blocks_from_output
_parse_sse_json_events = parse_sse_json_events
_extract_from_codex_sse_events = extract_text_usage_from_sse_events
_extract_unsupported_parameter_name = extract_unsupported_parameter_name_from_responses


def _to_namespace(value: Any) -> Any:
    """Convert nested dict/list structures into attribute-access objects."""
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value


async def call_oauth_codex(
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
    build_thinking_kwargs: ThinkingKwargsBuilder,
    run_with_provider_error_mapping: ProviderErrorMapper,
    safe_emit_progress: ProgressEmitter,
) -> ProviderResponse:
    """OAuth-backed Codex request path."""
    token_name = (model_profile.oauth_token_name or "").strip()
    if not token_name:
        return ProviderResponse.create_error(
            error_code="authentication_error",
            error_message="OAuth token is not configured for this model profile.",
            duration_ms=(time.time() - start_time) * 1000,
        )

    oauth_token = get_oauth_token(token_name)
    if oauth_token is None:
        return ProviderResponse.create_error(
            error_code="authentication_error",
            error_message=(
                f"OAuth token '{token_name}' was not found. "
                "Configure it with /oauth add <name>."
            ),
            duration_ms=(time.time() - start_time) * 1000,
        )

    now_ms = int(time.time() * 1000)
    refresh_deadline_ms = now_ms + 30_000
    needs_refresh = bool(
        oauth_token.refresh_token
        and (
            not oauth_token.access_token
            or (oauth_token.expires_at is not None and oauth_token.expires_at <= refresh_deadline_ms)
        )
    )
    if needs_refresh:
        try:
            refreshed = refresh_codex_access_token(oauth_token)
            add_oauth_token(token_name, refreshed, overwrite=True)
            oauth_token = refreshed
        except CodexOAuthError as exc:
            logger.warning(
                "[openai_client] OAuth token refresh failed; using existing token",
                extra={"token_name": token_name, "error": str(exc)},
            )

    if not oauth_token.access_token:
        return ProviderResponse.create_error(
            error_code="authentication_error",
            error_message=(
                f"OAuth token '{token_name}' does not have an access token."
            ),
            duration_ms=(time.time() - start_time) * 1000,
        )

    if oauth_token.type != OAuthTokenType.CODEX:
        return ProviderResponse.create_error(
            error_code="authentication_error",
            error_message=(
                f"OAuth token '{token_name}' has unsupported type '{oauth_token.type.value}'."
            ),
            duration_ms=(time.time() - start_time) * 1000,
        )

    if stream:
        logger.debug(
            "[openai_client] OAuth Codex path does not support streaming; using non-stream request",
            extra={"model": model_profile.model},
        )

    openai_tools = await build_openai_tool_schemas(tools)
    sanitized_messages = sanitize_tool_history(list(normalized_messages))
    response_input = _build_codex_responses_input(
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

    instructions = (system_prompt or "").strip() or "You are a helpful coding assistant."
    thinking_extra_body, thinking_top_level = build_thinking_kwargs(
        model_profile, max_thinking_tokens
    )
    payload: Dict[str, Any] = {
        "model": model_profile.model,
        "instructions": instructions,
        "input": response_input,
        "store": False,
        "stream": True,
    }
    codex_tools = _build_codex_oauth_tools(openai_tools)
    if codex_tools:
        payload["tools"] = codex_tools
    if model_profile.temperature is not None:
        payload["temperature"] = model_profile.temperature

    reasoning_obj: Optional[Dict[str, Any]] = None
    raw_reasoning = thinking_extra_body.get("reasoning")
    if isinstance(raw_reasoning, dict) and raw_reasoning:
        reasoning_obj = cast(Dict[str, Any], raw_reasoning)
    if reasoning_obj is None:
        effort = thinking_top_level.get("reasoning_effort")
        if isinstance(effort, str) and effort:
            reasoning_obj = {"effort": effort}
    if reasoning_obj:
        payload["reasoning"] = reasoning_obj

    headers = {
        "Authorization": f"Bearer {oauth_token.access_token}",
        "User-Agent": build_user_agent(),
        "originator": "ripperdoc",
    }
    if oauth_token.account_id:
        headers["ChatGPT-Account-Id"] = oauth_token.account_id

    timeout = request_timeout if request_timeout and request_timeout > 0 else 120.0

    async with httpx.AsyncClient(timeout=timeout) as client:
        payload_for_request: Dict[str, Any] = dict(payload)
        unsupported_pruned: set[str] = set()

        async def _request() -> httpx.Response:
            return await client.post(
                _CODEX_OAUTH_ENDPOINT,
                json=payload_for_request,
                headers=headers,
            )

        while True:
            response = await call_with_timeout_and_retries(
                lambda: run_with_provider_error_mapping(_request),
                request_timeout,
                max_retries,
            )
            if response.status_code < 400:
                break

            response_text = response.text or ""
            response_payload: Dict[str, Any] = {}
            if response_text:
                try:
                    maybe = response.json()
                    if isinstance(maybe, dict):
                        response_payload = maybe
                except ValueError:
                    response_payload = {}

            error_message = (
                response_payload.get("error", {}).get("message")
                if isinstance(response_payload.get("error"), dict)
                else None
            )
            if not isinstance(error_message, str) or not error_message:
                detail = response_payload.get("detail")
                if isinstance(detail, str) and detail:
                    error_message = detail
            if not isinstance(error_message, str) or not error_message:
                error_message = response_text

            unsupported_key = _extract_unsupported_parameter_name(error_message or "")
            if (
                response.status_code == 400
                and unsupported_key
                and unsupported_key in payload_for_request
                and unsupported_key not in unsupported_pruned
            ):
                unsupported_pruned.add(unsupported_key)
                payload_for_request.pop(unsupported_key, None)
                logger.debug(
                    "[openai_client] Removing unsupported Codex OAuth parameter and retrying",
                    extra={
                        "model": model_profile.model,
                        "parameter": unsupported_key,
                    },
                )
                continue
            break

    duration_ms = (time.time() - start_time) * 1000
    response_metadata: Dict[str, Any] = {
        "oauth_token_name": token_name,
        "oauth_token_type": oauth_token.type.value,
        "http_status": response.status_code,
    }

    raw_text = response.text or ""
    payload_json: Dict[str, Any] = {}
    if raw_text:
        try:
            maybe_json = response.json()
            if isinstance(maybe_json, dict):
                payload_json = maybe_json
        except ValueError:
            payload_json = {}

    sse_text = ""
    sse_usage_tokens: Dict[str, int] = {"input_tokens": 0, "output_tokens": 0}
    if not payload_json and raw_text:
        sse_events = _parse_sse_json_events(raw_text)
        if sse_events:
            sse_text, sse_usage_tokens, sse_response = _extract_from_codex_sse_events(
                sse_events
            )
            if isinstance(sse_response, dict):
                payload_json = {
                    "output": sse_response.get("output"),
                    "usage": sse_response.get("usage"),
                }

    if response.status_code >= 400:
        message = (
            payload_json.get("error", {}).get("message")
            if isinstance(payload_json.get("error"), dict)
            else None
        )
        if not isinstance(message, str) or not message:
            detail = payload_json.get("detail")
            if isinstance(detail, str) and detail:
                message = detail
        if not isinstance(message, str) or not message:
            message = raw_text or f"HTTP {response.status_code}"
        mapped = map_api_status_error(str(message), response.status_code)
        code, msg = classify_mapped_error(mapped) or ("api_error", str(mapped))
        return ProviderResponse.create_error(
            error_code=code,
            error_message=msg,
            duration_ms=duration_ms,
        )

    usage_tokens = openai_usage_tokens(payload_json.get("usage"))
    if any(int(v or 0) > 0 for v in sse_usage_tokens.values()):
        usage_tokens = sse_usage_tokens
    cost_usd = estimate_cost_usd(model_profile, usage_tokens)
    record_usage(
        model_profile.model,
        duration_ms=duration_ms,
        cost_usd=cost_usd,
        **usage_tokens,
    )

    content_blocks: List[Dict[str, Any]] = []
    response_choices = payload_json.get("choices")
    if isinstance(response_choices, list) and response_choices:
        try:
            choice_obj = _to_namespace(response_choices[0])
            content_blocks = content_blocks_from_openai_choice(choice_obj, tool_mode)
        except Exception:
            content_blocks = []
    if not content_blocks:
        content_blocks = _extract_content_blocks_from_responses_payload(payload_json)
    if not content_blocks and sse_text:
        content_blocks = [{"type": "text", "text": sse_text}]
    if not content_blocks and raw_text:
        content_blocks = [{"type": "text", "text": raw_text}]
    if not content_blocks:
        content_blocks = [{"type": "text", "text": "Model returned no content."}]

    if progress_callback:
        for block in content_blocks:
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                await safe_emit_progress(progress_callback, cast(str, block["text"]))

    return ProviderResponse(
        content_blocks=content_blocks,
        usage_tokens=usage_tokens,
        cost_usd=cost_usd,
        duration_ms=duration_ms,
        metadata=response_metadata,
    )
