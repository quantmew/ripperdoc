"""Detection helpers for context-window overflow errors across providers.

Observed provider responses when the request is too large:
- OpenAI/OpenRouter style (400 BadRequestError): error.code/context_length_exceeded with
  a message like "This model's maximum context length is 128000 tokens. However, you
  requested 130000 tokens (... in the messages, ... in the completion)."
- Anthropic (400 BadRequestError): invalid_request_error with a message such as
  "prompt is too long for model claude-3-5-sonnet. max tokens: 200000 prompt tokens: 240000".
- Gemini / google-genai (FAILED_PRECONDITION or INVALID_ARGUMENT): APIError message like
  "The input to the model was too long. The requested input has X tokens, which exceeds
  the maximum of Y tokens for models/gemini-...".

These helpers allow callers to detect the condition and trigger auto-compaction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Set

ContextLengthErrorCode = Optional[str]


@dataclass
class ContextLengthErrorInfo:
    """Normalized metadata about a context-length error."""

    provider: Optional[str]
    message: str
    error_code: ContextLengthErrorCode = None
    status_code: Optional[int] = None


_CONTEXT_PATTERNS = [
    "context_length_exceeded",
    "maximum context length",
    "max context length",
    "maximum context window",
    "max context window",
    "context length is",
    "context length was exceeded",
    "context window of",
    "token limit exceeded",
    "token length exceeded",
    "prompt is too long",
    "input is too long",
    "request is too large",
    "exceeds the maximum context",
    "exceeds the model's context",
    "requested input has",
    "too many tokens",
    "reduce the length of the messages",
]


def detect_context_length_error(error: Any) -> Optional[ContextLengthErrorInfo]:
    """Return normalized context-length error info if the exception matches."""
    if error is None:
        return None

    provider = _guess_provider(error)
    status_code = _extract_status_code(error)
    codes = _extract_codes(error)
    messages = _collect_strings(error)

    # Check explicit error codes first.
    for code in codes:
        normalized = code.lower()
        if any(
            keyword in normalized
            for keyword in (
                "context_length",
                "max_tokens",
                "token_length",
                "prompt_too_long",
                "input_too_large",
                "token_limit",
            )
        ):
            message = messages[0] if messages else code
            return ContextLengthErrorInfo(
                provider=provider,
                message=message,
                error_code=code,
                status_code=status_code,
            )

    # Fall back to message-based detection.
    for text in messages:
        if _looks_like_context_length_message(text):
            return ContextLengthErrorInfo(
                provider=provider,
                message=text,
                error_code=codes[0] if codes else None,
                status_code=status_code,
            )

    return None


def _looks_like_context_length_message(text: str) -> bool:
    lower = text.lower()
    if any(pattern in lower for pattern in _CONTEXT_PATTERNS):
        return True
    if "too long" in lower and (
        "prompt" in lower or "input" in lower or "context" in lower or "token" in lower
    ):
        return True
    if "exceed" in lower and ("token" in lower or "context" in lower):
        return True
    if "max" in lower and "token" in lower and ("context" in lower or "limit" in lower):
        return True
    return False


def _guess_provider(error: Any) -> Optional[str]:
    module = getattr(getattr(error, "__class__", None), "__module__", "") or ""
    name = getattr(getattr(error, "__class__", None), "__name__", "").lower()
    if "openai" in module or "openai" in name:
        return "openai"
    if "anthropic" in module or "claude" in module:
        return "anthropic"
    if "google.genai" in module or "vertexai" in module:
        return "gemini"
    return None


def _extract_status_code(error: Any) -> Optional[int]:
    for attr in ("status_code", "http_status", "code"):
        value = getattr(error, attr, None)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)

    for payload in (
        _safe_getattr(error, "body"),
        _safe_getattr(error, "details"),
        _safe_getattr(error, "error"),
    ):
        if isinstance(payload, dict):
            for key in ("status_code", "code"):
                value = payload.get(key)
                if isinstance(value, int):
                    return value
                if isinstance(value, str) and value.isdigit():
                    return int(value)

    return None


def _extract_codes(error: Any) -> List[str]:
    codes: List[str] = []
    seen: Set[str] = set()

    def _add(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, int):
            value = str(value)
        if not isinstance(value, str):
            return
        normalized = value.strip()
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        codes.append(normalized)

    for attr in ("code", "error_code", "type", "status"):
        _add(_safe_getattr(error, attr))

    for payload in (
        _safe_getattr(error, "body"),
        _safe_getattr(error, "details"),
        _safe_getattr(error, "error"),
    ):
        if isinstance(payload, dict):
            for key in ("code", "type", "status"):
                _add(payload.get(key))
            nested = payload.get("error")
            if isinstance(nested, dict):
                for key in ("code", "type", "status"):
                    _add(nested.get(key))

    if isinstance(error, dict):
        for key in ("code", "type", "status"):
            _add(error.get(key))

    return codes


def _collect_strings(error: Any) -> List[str]:
    """Collect human-readable strings from an exception/payload."""
    texts: List[str] = []
    seen_texts: Set[str] = set()
    seen_objs: Set[int] = set()

    def _add_text(value: Any) -> None:
        if not isinstance(value, str):
            return
        normalized = value.strip()
        if not normalized or normalized in seen_texts:
            return
        seen_texts.add(normalized)
        texts.append(normalized)

    def _walk(obj: Any) -> None:
        if obj is None:
            return
        obj_id = id(obj)
        if obj_id in seen_objs:
            return
        seen_objs.add(obj_id)

        if isinstance(obj, str):
            _add_text(obj)
            return

        if isinstance(obj, BaseException):
            _add_text(_safe_getattr(obj, "message"))
            for arg in getattr(obj, "args", ()):
                _walk(arg)
            for attr in ("body", "error", "details"):
                _walk(_safe_getattr(obj, attr))
            return

        if isinstance(obj, dict):
            for val in obj.values():
                _walk(val)
            return

        if isinstance(obj, (list, tuple, set)):
            for item in obj:
                _walk(item)
            return

        _add_text(_safe_getattr(obj, "message"))

    _walk(error)
    try:
        _add_text(str(error))
    except (TypeError, ValueError):
        pass

    return texts


def _safe_getattr(obj: Any, attr: str) -> Any:
    try:
        return getattr(obj, attr, None)
    except (TypeError, AttributeError):
        return None
