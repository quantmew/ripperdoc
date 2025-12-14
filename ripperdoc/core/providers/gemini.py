"""Gemini provider client with function/tool calling support."""

from __future__ import annotations

import copy
import inspect
import os
import time
from typing import Any, AsyncIterable, AsyncIterator, Dict, List, Optional, Tuple, cast
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
    "Gemini client requires the 'google-genai' package. "
    "Install it with: pip install google-genai"
)
GEMINI_MODELS_ENDPOINT_ERROR = "Gemini client is missing 'models' endpoint"
GEMINI_GENERATE_CONTENT_ERROR = "Gemini client is missing generate_content() method"


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

    return {
        "input_tokens": safe_get_int("prompt_token_count") + safe_get_int("cached_content_token_count"),
        "output_tokens": safe_get_int("candidates_token_count"),
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
        text_val = getattr(part, "text", None) or getattr(part, "content", None) or getattr(
            part, "raw_text", None
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


async def _async_build_tool_declarations(tools: List[Tool[Any, Any]]) -> List[Dict[str, Any]]:
    declarations: List[Dict[str, Any]] = []
    try:
        from google.genai import types as genai_types  # type: ignore
    except Exception:  # pragma: no cover - fallback when SDK not installed
        genai_types = None

    for tool in tools:
        description = await build_tool_description(tool, include_examples=True, max_examples=2)
        parameters_schema = _flatten_schema(tool.input_schema.model_json_schema())
        if genai_types:
            declarations.append(
                genai_types.FunctionDeclaration(
                    name=tool.name,
                    description=description,
                    parameters=genai_types.Schema(**parameters_schema),
                )
            )
        else:
            declarations.append(
                {"name": tool.name, "description": description, "parameters": parameters_schema}
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
    except Exception:  # pragma: no cover - fallback when SDK not installed
        genai_types = None

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
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError(GEMINI_SDK_IMPORT_ERROR) from exc

        client_kwargs: Dict[str, Any] = {}
        api_key = model_profile.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if api_key:
            client_kwargs["api_key"] = api_key
        if model_profile.api_base:
            from google.genai import types as genai_types  # type: ignore

            client_kwargs["http_options"] = genai_types.HttpOptions(
                base_url=model_profile.api_base
            )
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
    ) -> ProviderResponse:
        start_time = time.time()

        try:
            client = await self._client(model_profile)
        except Exception as exc:
            msg = str(exc)
            logger.warning("[gemini_client] Initialization failed", extra={"error": msg})
            return ProviderResponse(
                content_blocks=[{"type": "text", "text": msg}],
                usage_tokens={},
                cost_usd=0.0,
                duration_ms=(time.time() - start_time) * 1000,
            )

        declarations: List[Dict[str, Any]] = []
        if tools and tool_mode != "text":
            declarations = await _async_build_tool_declarations(tools)

        contents, _ = _convert_messages_to_genai_contents(normalized_messages)

        config: Dict[str, Any] = {"system_instruction": system_prompt}
        if model_profile.max_tokens:
            config["max_output_tokens"] = model_profile.max_tokens
        if declarations:
            try:
                from google.genai import types as genai_types  # type: ignore

                config["tools"] = [genai_types.Tool(function_declarations=declarations)]
            except Exception:  # pragma: no cover - fallback when SDK not installed
                config["tools"] = [{"function_declarations": declarations}]

        generate_kwargs: Dict[str, Any] = {
            "model": model_profile.model,
            "contents": contents,
            "config": config,
        }
        usage_tokens: Dict[str, int] = {}
        collected_text: List[str] = []
        function_calls: List[Dict[str, Any]] = []

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

                result = generate_fn(**generate_kwargs)
                if inspect.isawaitable(result):
                    return await result
                return result

        try:
            if stream:
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
                        if progress_callback:
                            text_delta = _collect_text_from_parts(parts)
                            if text_delta:
                                try:
                                    await progress_callback(text_delta)
                                except Exception:
                                    logger.exception("[gemini_client] Stream callback failed")
                        collected_text.append(_collect_text_from_parts(parts))
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
                    function_calls.extend(_extract_function_calls(parts))
                else:
                    # Fallback: try to read text directly
                    collected_text.append(getattr(response, "text", "") or "")
                usage_tokens = _extract_usage_metadata(response)
        except Exception as exc:
            logger.exception("[gemini_client] Error during call", extra={"error": str(exc)})
            return ProviderResponse(
                content_blocks=[{"type": "text", "text": f"Gemini call failed: {exc}"}],
                usage_tokens={},
                cost_usd=0.0,
                duration_ms=(time.time() - start_time) * 1000,
            )

        content_blocks: List[Dict[str, Any]] = []
        combined_text = "".join(collected_text).strip()
        if combined_text:
            content_blocks.append({"type": "text", "text": combined_text})

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
        )
