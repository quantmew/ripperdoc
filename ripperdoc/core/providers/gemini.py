"""Gemini provider client."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from ripperdoc.core.config import ModelProfile
from ripperdoc.core.providers.base import (
    ProgressCallback,
    ProviderClient,
    ProviderResponse,
    call_with_timeout_and_retries,
)
from ripperdoc.core.tool import Tool
from ripperdoc.utils.log import get_logger

logger = get_logger()


def _extract_usage_metadata(payload: Any) -> Dict[str, int]:
    """Best-effort token extraction from Gemini responses."""
    usage = getattr(payload, "usage_metadata", None) or getattr(payload, "usageMetadata", None)
    if not usage:
        usage = getattr(payload, "usage", None)
    get = lambda key: int(getattr(usage, key, 0) or 0) if usage else 0  # noqa: E731
    return {
        "input_tokens": get("prompt_token_count") + get("cached_content_token_count"),
        "output_tokens": get("candidates_token_count"),
        "cache_read_input_tokens": get("cached_content_token_count"),
        "cache_creation_input_tokens": 0,
    }


def _collect_text_parts(candidate: Any) -> str:
    parts = getattr(candidate, "content", None)
    if not parts:
        return ""
    if isinstance(parts, list):
        texts = []
        for part in parts:
            text_val = getattr(part, "text", None) or getattr(part, "content", None)
            if isinstance(text_val, str):
                texts.append(text_val)
        return "".join(texts)
    return str(parts)


class GeminiClient(ProviderClient):
    """Gemini client with streaming and basic text support."""

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
        try:
            import google.generativeai as genai  # type: ignore
        except Exception as exc:  # pragma: no cover - import guard
            msg = (
                "Gemini client requires the 'google-generativeai' package. "
                "Install it to enable Gemini support."
            )
            logger.warning(msg, extra={"error": str(exc)})
            return ProviderResponse(
                content_blocks=[{"type": "text", "text": msg}],
                usage_tokens={},
                cost_usd=0.0,
                duration_ms=0.0,
            )

        if tools and tool_mode != "text":
            msg = (
                "Gemini client currently supports text-only responses; "
                "tool/function calling is not yet implemented."
            )
            return ProviderResponse(
                content_blocks=[{"type": "text", "text": msg}],
                usage_tokens={},
                cost_usd=0.0,
                duration_ms=0.0,
            )

        api_key = model_profile.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=api_key, client_options={"api_endpoint": model_profile.api_base})

        # Flatten normalized messages into a single text prompt (Gemini supports multi-turn, but keep it simple).
        prompt_parts: List[str] = [system_prompt]
        for msg in normalized_messages:
            role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "")
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
            if isinstance(content, list):
                for item in content:
                    text_val = getattr(item, "text", None) or item.get("text", "") if isinstance(item, dict) else ""
                    if text_val:
                        prompt_parts.append(f"{role}: {text_val}")
            elif isinstance(content, str):
                prompt_parts.append(f"{role}: {content}")
        full_prompt = "\n".join(part for part in prompt_parts if part)

        model = genai.GenerativeModel(model_profile.model)
        collected_text: List[str] = []
        start_time = time.time()

        async def _stream_request() -> Any:
            stream_resp = model.generate_content(full_prompt, stream=True)
            usage_tokens: Dict[str, int] = {}
            for chunk in stream_resp:
                text_delta = _collect_text_parts(chunk)
                if text_delta:
                    collected_text.append(text_delta)
                    if progress_callback:
                        try:
                            await progress_callback(text_delta)
                        except Exception:
                            logger.exception("[gemini_client] Stream callback failed")
                usage_tokens = _extract_usage_metadata(chunk) or usage_tokens
            return {"usage": usage_tokens}

        async def _non_stream_request() -> Any:
            return model.generate_content(full_prompt)

        response: Any = await call_with_timeout_and_retries(
            _stream_request if stream and progress_callback else _non_stream_request,
            request_timeout,
            max_retries,
        )

        duration_ms = (time.time() - start_time) * 1000
        usage_tokens = _extract_usage_metadata(response)
        cost_usd = 0.0  # Pricing unknown; leave as 0

        content_blocks = [{"type": "text", "text": "".join(collected_text)}] if collected_text else [
            {"type": "text", "text": _collect_text_parts(response)}
        ]

        logger.info(
            "[gemini_client] Response received",
            extra={
                "model": model_profile.model,
                "duration_ms": round(duration_ms, 2),
                "tool_mode": tool_mode,
                "stream": stream,
            },
        )

        return ProviderResponse(
            content_blocks=content_blocks,
            usage_tokens=usage_tokens,
            cost_usd=cost_usd,
            duration_ms=duration_ms,
        )
