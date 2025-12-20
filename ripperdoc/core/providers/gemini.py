"""Gemini provider client with function/tool calling support."""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import os
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, cast
from uuid import uuid4

from ripperdoc.core.config import ModelProfile
from ripperdoc.core.providers.base import (
    ProgressCallback,
    ProviderClient,
    ProviderResponse,
    call_with_timeout_and_retries,
    iter_with_timeout,
)
from ripperdoc.core.query_utils import _normalize_tool_args, build_tool_description
from ripperdoc.core.tool import Tool
from ripperdoc.utils.log import get_logger
from ripperdoc.utils.session_usage import record_usage
from ripperdoc.core.query_utils import estimate_cost_usd

logger = get_logger()

# Constants
GEMINI_SDK_IMPORT_ERROR = (
    "Gemini client requires the 'google-genai' package. Install it with: pip install google-genai"
)
GEMINI_MODELS_ENDPOINT_ERROR = "Gemini client is missing 'models' endpoint"
GEMINI_GENERATE_CONTENT_ERROR = "Gemini client is missing generate_content() method"


def _classify_gemini_error(exc: Exception) -> tuple[str, str]:
    """Classify a Gemini exception into error code and user-friendly message."""
    exc_type = type(exc).__name__
    exc_msg = str(exc)

    # Try to import Google's exception types for more specific handling
    try:
        from google.api_core import exceptions as google_exceptions  # type: ignore

        if isinstance(exc, google_exceptions.Unauthenticated):
            return "authentication_error", f"Authentication failed: {exc_msg}"
        if isinstance(exc, google_exceptions.PermissionDenied):
            return "permission_denied", f"Permission denied: {exc_msg}"
        if isinstance(exc, google_exceptions.NotFound):
            return "model_not_found", f"Model not found: {exc_msg}"
        if isinstance(exc, google_exceptions.InvalidArgument):
            if "context" in exc_msg.lower() or "token" in exc_msg.lower():
                return "context_length_exceeded", f"Context length exceeded: {exc_msg}"
            return "bad_request", f"Invalid request: {exc_msg}"
        if isinstance(exc, google_exceptions.ResourceExhausted):
            return "rate_limit", f"Rate limit exceeded: {exc_msg}"
        if isinstance(exc, google_exceptions.ServiceUnavailable):
            return "service_unavailable", f"Service unavailable: {exc_msg}"
        if isinstance(exc, google_exceptions.GoogleAPICallError):
            return "api_error", f"API error: {exc_msg}"
    except ImportError:
        pass

    # Fallback for generic exceptions
    if isinstance(exc, asyncio.TimeoutError):
        return "timeout", f"Request timed out: {exc_msg}"
    if isinstance(exc, ConnectionError):
        return "connection_error", f"Connection error: {exc_msg}"
    if "quota" in exc_msg.lower() or "limit" in exc_msg.lower():
        return "rate_limit", f"Rate limit exceeded: {exc_msg}"
    if "auth" in exc_msg.lower() or "key" in exc_msg.lower():
        return "authentication_error", f"Authentication error: {exc_msg}"
    if "not found" in exc_msg.lower():
        return "model_not_found", f"Model not found: {exc_msg}"

    return "unknown_error", f"Unexpected error ({exc_type}): {exc_msg}"


def _extract_usage_metadata(payload: Any) -> Dict[str, int]:
    """Best-effort token extraction from Gemini responses."""
    usage = getattr(payload, "usage_metadata", None) or getattr(payload, "usageMetadata", None)
    if not usage:
        usage = getattr(payload, "usage", None)
    if not usage and getattr(payload, "candidates", None):
        usage = getattr(payload.candidates[0], "usage_metadata", None)

    def safe_get_int(key: str) -> int:
        """Safely extract integer value from usage metadata."""
        if not usage:
            return 0
        value = getattr(usage, key, 0)
        return int(value) if value else 0

    thought_tokens = safe_get_int("thoughts_token_count")
    candidate_tokens = safe_get_int("candidates_token_count")

    return {
        "input_tokens": safe_get_int("prompt_token_count")
        + safe_get_int("cached_content_token_count"),
        "output_tokens": candidate_tokens + thought_tokens,
        "cache_read_input_tokens": safe_get_int("cached_content_token_count"),
        "cache_creation_input_tokens": 0,
    }


def _collect_parts(candidate: Any) -> List[Any]:
    """Return a list of parts from a candidate regardless of SDK shape."""
    content = getattr(candidate, "content", None)
    if content is None:
        return []
    if hasattr(content, "parts"):
        return list(getattr(content, "parts", []) or [])
    if isinstance(content, list):
        return content
    return []


def _collect_text_from_parts(parts: List[Any]) -> str:
    texts: List[str] = []
    for part in parts:
        text_val = (
            getattr(part, "text", None)
            or getattr(part, "content", None)
            or getattr(part, "raw_text", None)
        )
        if isinstance(text_val, str):
            texts.append(text_val)
    return "".join(texts)


def _extract_function_calls(parts: List[Any]) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    for part in parts:
        fn_call = getattr(part, "function_call", None) or getattr(part, "functionCall", None)
        if not fn_call:
            continue
        name = getattr(fn_call, "name", None) or getattr(fn_call, "function_name", None)
        args = getattr(fn_call, "args", None) or getattr(fn_call, "arguments", None) or {}
        call_id = getattr(fn_call, "id", None) or getattr(fn_call, "call_id", None)
        calls.append({"name": name, "args": _normalize_tool_args(args), "id": call_id})
    return calls


def _flatten_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Inline $ref entries and drop $defs/$ref for Gemini Schema compatibility.

    Gemini API doesn't support JSON Schema references, so this function
    resolves all $ref pointers by inlining the referenced definitions.
    """
    definitions = copy.deepcopy(schema.get("$defs") or schema.get("definitions") or {})

    def _resolve(node: Any) -> Any:
        """Recursively resolve $ref pointers and remove unsupported fields."""
        if isinstance(node, dict):
            # Handle $ref resolution
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/"):
                ref_key = ref.split("/")[-1]
                if ref_key in definitions:
                    return _resolve(copy.deepcopy(definitions[ref_key]))

            # Process remaining fields, excluding schema metadata
            resolved: Dict[str, Any] = {}
            for key, value in node.items():
                if key in {"$ref", "$defs", "definitions"}:
                    continue
                resolved[key] = _resolve(value)
            return resolved

        if isinstance(node, list):
            return [_resolve(item) for item in node]

        return node

    return cast(Dict[str, Any], _resolve(copy.deepcopy(schema)))


def _supports_stream_arg(fn: Any) -> bool:
    """Return True if the callable appears to accept a 'stream' kwarg."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        # If we cannot inspect, avoid passing stream to prevent TypeErrors.
        return False

    for param in sig.parameters.values():
        if param.kind == param.VAR_KEYWORD:
            return True
        if param.name == "stream":
            return True
    return False


def _build_thinking_config(max_thinking_tokens: int, model_name: str) -> Dict[str, Any]:
    """Map max_thinking_tokens to Gemini thinking_config settings."""
    if max_thinking_tokens <= 0:
        return {}
    name = (model_name or "").lower()
    config: Dict[str, Any] = {"include_thoughts": True}
    if "gemini-3" in name:
        config["thinking_level"] = "low" if max_thinking_tokens <= 2048 else "high"
    else:
        config["thinking_budget"] = max_thinking_tokens
    return config


def _collect_thoughts_from_parts(parts: List[Any]) -> List[str]:
    """Extract thought summaries from parts flagged as thoughts."""
    snippets: List[str] = []
    for part in parts:
        is_thought = getattr(part, "thought", None)
        if is_thought is None and isinstance(part, dict):
            is_thought = part.get("thought")
        if not is_thought:
            continue
        text_val = (
            getattr(part, "text", None)
            or getattr(part, "content", None)
            or getattr(part, "raw_text", None)
        )
        if isinstance(text_val, str):
            snippets.append(text_val)
    return snippets


async def _async_build_tool_declarations(tools: List[Tool[Any, Any]]) -> List[Dict[str, Any]]:
    declarations: List[Dict[str, Any]] = []
    try:
        from google.genai import types as genai_types  # type: ignore
    except (ImportError, ModuleNotFoundError):  # pragma: no cover - fallback when SDK not installed
        genai_types = None  # type: ignore[assignment]

    for tool in tools:
        description = await build_tool_description(tool, include_examples=True, max_examples=2)
        parameters_schema = _flatten_schema(tool.input_schema.model_json_schema())
        if genai_types:
            func_decl = genai_types.FunctionDeclaration(
                name=tool.name,
                description=description,
                parameters_json_schema=parameters_schema,
            )
            declarations.append(func_decl.model_dump(mode="json", exclude_none=True))
        else:
            declarations.append(
                {
                    "name": tool.name,
                    "description": description,
                    "parameters_json_schema": parameters_schema,
                }
            )
    return declarations


def _convert_messages_to_genai_contents(
    normalized_messages: List[Dict[str, Any]],
) -> Tuple[List[Any], Dict[str, str]]:
    """Map normalized OpenAI-style messages to Gemini content payloads.

    Returns:
        contents: List of Content-like dicts/objects
        tool_name_by_id: Map of tool_call_id -> function name (for pairing responses)
    """
    tool_name_by_id: Dict[str, str] = {}
    contents: List[Any] = []

    # Lazy import to avoid hard dependency in tests.
    try:
        from google.genai import types as genai_types  # type: ignore
    except (ImportError, ModuleNotFoundError):  # pragma: no cover - fallback when SDK not installed
        genai_types = None  # type: ignore[assignment]

    def _mk_part_from_text(text: str) -> Any:
        if genai_types:
            return genai_types.Part(text=text)
        return {"text": text}

    def _mk_part_from_function_call(name: str, args: Dict[str, Any], call_id: Optional[str]) -> Any:
        # Store mapping using actual call_id if available, otherwise generate one
        actual_id = call_id or str(uuid4())
        tool_name_by_id[actual_id] = name
        if genai_types:
            return genai_types.Part(function_call=genai_types.FunctionCall(name=name, args=args))
        return {"function_call": {"name": name, "args": args, "id": actual_id}}

    def _mk_part_from_function_response(
        name: str, response: Dict[str, Any], call_id: Optional[str]
    ) -> Any:
        if call_id:
            response = {**response, "call_id": call_id}
        if genai_types:
            return genai_types.Part.from_function_response(name=name, response=response)
        payload = {"function_response": {"name": name, "response": response}}
        if call_id:
            payload["function_response"]["id"] = call_id
        return payload

    def _mk_content(role: str, parts: List[Any]) -> Any:
        if genai_types:
            return genai_types.Content(role=role, parts=parts)
        return {"role": role, "parts": parts}

    for message in normalized_messages:
        role = message.get("role") or ""
        msg_parts: List[Any] = []

        # Assistant tool calls
        for tool_call in message.get("tool_calls") or []:
            func = tool_call.get("function") or {}
            name = func.get("name") or ""
            args = _normalize_tool_args(func.get("arguments") or {})
            call_id = tool_call.get("id")
            msg_parts.append(_mk_part_from_function_call(name, args, call_id))

        content_value = message.get("content")
        if isinstance(content_value, str) and content_value:
            msg_parts.append(_mk_part_from_text(content_value))

        if role == "tool":
            call_id = message.get("tool_call_id") or ""
            name = tool_name_by_id.get(call_id, call_id or "tool_response")
            response = {"result": content_value}
            msg_parts.append(_mk_part_from_function_response(name, response, call_id))
            role = "user"  # Tool responses are treated as user-provided context

        if not msg_parts:
            continue

        mapped_role = "user" if role == "user" else "model"
        contents.append(_mk_content(mapped_role, msg_parts))

    return contents, tool_name_by_id


class GeminiClient(ProviderClient):
    """Gemini client with streaming and function calling support."""

    def __init__(self, client_factory: Optional[Any] = None) -> None:
        self._client_factory = client_factory

    async def _client(self, model_profile: ModelProfile) -> Any:
        if self._client_factory is not None:
            client = self._client_factory
            if inspect.iscoroutinefunction(client):
                return await client()
            if inspect.isawaitable(client):
                return await client  # type: ignore[return-value]
            if callable(client):
                result = client()
                return await result if inspect.isawaitable(result) else result
            return client

        try:
            from google import genai  # type: ignore
        except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - import guard
            raise RuntimeError(GEMINI_SDK_IMPORT_ERROR) from exc

        client_kwargs: Dict[str, Any] = {}
        api_key = (
            model_profile.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        )
        if api_key:
            client_kwargs["api_key"] = api_key
        if model_profile.api_base:
            from google.genai import types as genai_types  # type: ignore

            client_kwargs["http_options"] = genai_types.HttpOptions(base_url=model_profile.api_base)
        return genai.Client(**client_kwargs)

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

        logger.debug(
            "[gemini_client] Preparing request",
            extra={
                "model": model_profile.model,
                "tool_mode": tool_mode,
                "stream": stream,
                "max_thinking_tokens": max_thinking_tokens,
                "num_tools": len(tools),
            },
        )

        try:
            client = await self._client(model_profile)
        except asyncio.CancelledError:
            raise  # Don't suppress task cancellation
        except Exception as exc:
            duration_ms = (time.time() - start_time) * 1000
            error_code, error_message = _classify_gemini_error(exc)
            logger.debug(
                "[gemini_client] Exception details during init",
                extra={
                    "model": model_profile.model,
                    "exception_type": type(exc).__name__,
                    "exception_str": str(exc),
                    "error_code": error_code,
                },
            )
            logger.error(
                "[gemini_client] Initialization failed",
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

        declarations: List[Dict[str, Any]] = []
        if tools and tool_mode != "text":
            declarations = await _async_build_tool_declarations(tools)

        contents, _ = _convert_messages_to_genai_contents(normalized_messages)

        config: Dict[str, Any] = {"system_instruction": system_prompt}
        if model_profile.max_tokens:
            config["max_output_tokens"] = model_profile.max_tokens
        thinking_config = _build_thinking_config(max_thinking_tokens, model_profile.model)
        if thinking_config:
            try:
                from google.genai import types as genai_types  # type: ignore

                config["thinking_config"] = genai_types.ThinkingConfig(**thinking_config)
            except (
                ImportError,
                ModuleNotFoundError,
                TypeError,
                ValueError,
            ):  # pragma: no cover - fallback when SDK not installed
                config["thinking_config"] = thinking_config
        if declarations:
            config["tools"] = [{"function_declarations": declarations}]

        generate_kwargs: Dict[str, Any] = {
            "model": model_profile.model,
            "contents": contents,
            "config": config,
        }

        logger.debug(
            "[gemini_client] Request parameters",
            extra={
                "model": model_profile.model,
                "config": json.dumps(
                    {k: v for k, v in config.items() if k != "system_instruction"},
                    ensure_ascii=False,
                    default=str,
                )[:1000],
                "num_declarations": len(declarations),
                "thinking_config": json.dumps(thinking_config, ensure_ascii=False)
                if thinking_config
                else None,
            },
        )

        usage_tokens: Dict[str, int] = {}
        collected_text: List[str] = []
        function_calls: List[Dict[str, Any]] = []
        reasoning_parts: List[str] = []
        response_metadata: Dict[str, Any] = {}

        async def _call_generate(streaming: bool) -> Any:
            models_api = getattr(client, "models", None) or getattr(
                getattr(client, "aio", None), "models", None
            )
            if models_api is None:
                raise RuntimeError(GEMINI_MODELS_ENDPOINT_ERROR)

            generate_fn = getattr(models_api, "generate_content", None)
            stream_fn = getattr(models_api, "generate_content_stream", None) or getattr(
                models_api, "stream_generate_content", None
            )

            if streaming:
                if stream_fn:
                    result = stream_fn(**generate_kwargs)
                    if inspect.isawaitable(result):
                        return await result
                    return result

                if generate_fn is None:
                    raise RuntimeError(GEMINI_GENERATE_CONTENT_ERROR)

                if _supports_stream_arg(generate_fn):
                    gen_kwargs: Dict[str, Any] = dict(generate_kwargs)
                    gen_kwargs["stream"] = True
                    result = generate_fn(**gen_kwargs)
                    if inspect.isawaitable(result):
                        return await result
                    return result

                # Fallback: non-streaming generate; wrap to keep downstream iterator usage
                result = generate_fn(**generate_kwargs)
                if inspect.isawaitable(result):
                    result = await result

                async def _single_chunk_stream() -> AsyncIterator[Any]:
                    yield result

                return _single_chunk_stream()

            if generate_fn is None:
                raise RuntimeError(GEMINI_GENERATE_CONTENT_ERROR)

        try:
            if stream:
                logger.debug(
                    "[gemini_client] Initiating stream request",
                    extra={"model": model_profile.model},
                )
                stream_resp = await _call_generate(streaming=True)

                # Normalize streams into an async iterator to avoid StopIteration surfacing through
                # asyncio executors and to handle sync iterables.
                def _to_async_iter(obj: Any) -> AsyncIterator[Any]:
                    """Convert various iterable types to async generator."""
                    if inspect.isasyncgen(obj) or hasattr(obj, "__aiter__"):

                        async def _wrap_async() -> AsyncIterator[Any]:
                            async for item in obj:
                                yield item

                        return _wrap_async()
                    if hasattr(obj, "__iter__"):

                        async def _wrap_sync() -> AsyncIterator[Any]:
                            for item in obj:
                                yield item

                        return _wrap_sync()

                    async def _single() -> AsyncIterator[Any]:
                        yield obj

                    return _single()

                stream_iter = _to_async_iter(stream_resp)

                async for chunk in iter_with_timeout(stream_iter, request_timeout):
                    candidates = getattr(chunk, "candidates", None) or []
                    for candidate in candidates:
                        parts = _collect_parts(candidate)
                        text_chunk = _collect_text_from_parts(parts)
                        if progress_callback:
                            if text_chunk:
                                try:
                                    await progress_callback(text_chunk)
                                except (RuntimeError, ValueError, TypeError, OSError) as cb_exc:
                                    logger.warning(
                                        "[gemini_client] Stream callback failed: %s: %s",
                                        type(cb_exc).__name__,
                                        cb_exc,
                                    )
                        if text_chunk:
                            collected_text.append(text_chunk)
                        reasoning_parts.extend(_collect_thoughts_from_parts(parts))
                        function_calls.extend(_extract_function_calls(parts))
                    usage_tokens = _extract_usage_metadata(chunk) or usage_tokens
            else:
                # Use retry logic for non-streaming calls
                response = await call_with_timeout_and_retries(
                    lambda: _call_generate(streaming=False),
                    request_timeout,
                    max_retries,
                )
                candidates = getattr(response, "candidates", None) or []
                if candidates:
                    parts = _collect_parts(candidates[0])
                    collected_text.append(_collect_text_from_parts(parts))
                    reasoning_parts.extend(_collect_thoughts_from_parts(parts))
                    function_calls.extend(_extract_function_calls(parts))
                else:
                    # Fallback: try to read text directly
                    collected_text.append(getattr(response, "text", "") or "")
                usage_tokens = _extract_usage_metadata(response)
        except asyncio.CancelledError:
            raise  # Don't suppress task cancellation
        except Exception as exc:
            duration_ms = (time.time() - start_time) * 1000
            error_code, error_message = _classify_gemini_error(exc)
            logger.debug(
                "[gemini_client] Exception details",
                extra={
                    "model": model_profile.model,
                    "exception_type": type(exc).__name__,
                    "exception_str": str(exc),
                    "error_code": error_code,
                },
            )
            logger.error(
                "[gemini_client] API call failed",
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

        content_blocks: List[Dict[str, Any]] = []
        combined_text = "".join(collected_text).strip()
        if combined_text:
            content_blocks.append({"type": "text", "text": combined_text})
        if reasoning_parts:
            response_metadata["reasoning_content"] = "".join(reasoning_parts)

        for call in function_calls:
            if not call.get("name"):
                continue
            content_blocks.append(
                {
                    "type": "tool_use",
                    "tool_use_id": call.get("id") or str(uuid4()),
                    "name": call["name"],
                    "input": call.get("args") or {},
                }
            )

        duration_ms = (time.time() - start_time) * 1000
        cost_usd = estimate_cost_usd(model_profile, usage_tokens) if usage_tokens else 0.0
        record_usage(
            model_profile.model,
            duration_ms=duration_ms,
            cost_usd=cost_usd,
            **(usage_tokens or {}),
        )

        logger.debug(
            "[gemini_client] Response content blocks",
            extra={
                "model": model_profile.model,
                "content_blocks": json.dumps(content_blocks, ensure_ascii=False)[:1000],
                "usage_tokens": json.dumps(usage_tokens, ensure_ascii=False),
                "metadata": json.dumps(response_metadata, ensure_ascii=False)[:500],
            },
        )

        logger.info(
            "[gemini_client] Response received",
            extra={
                "model": model_profile.model,
                "duration_ms": round(duration_ms, 2),
                "tool_mode": tool_mode,
                "stream": stream,
                "function_call_count": len(function_calls),
            },
        )

        return ProviderResponse(
            content_blocks=content_blocks or [{"type": "text", "text": ""}],
            usage_tokens=usage_tokens,
            cost_usd=cost_usd,
            duration_ms=duration_ms,
            metadata=response_metadata,
        )
