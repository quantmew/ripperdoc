import httpx
from anthropic import BadRequestError as AnthropicBadRequestError
from google.genai import errors as genai_errors
from openai import BadRequestError as OpenAIBadRequestError

from ripperdoc.utils.context_length_errors import detect_context_length_error


def _fake_response(status_code: int = 400) -> httpx.Response:
    request = httpx.Request("POST", "https://example.com")
    return httpx.Response(status_code=status_code, request=request)


def test_detects_openai_context_length_error():
    body = {
        "code": "context_length_exceeded",
        "message": (
            "This model's maximum context length is 128000 tokens. However, you requested 130000 "
            "tokens (129000 in the messages, 1000 in the completion)."
        ),
        "type": "invalid_request_error",
    }
    error = OpenAIBadRequestError(body["message"], response=_fake_response(), body=body)

    info = detect_context_length_error(error)

    assert info is not None
    assert info.provider == "openai"
    assert info.error_code == "context_length_exceeded"
    assert "maximum context length" in info.message.lower()


def test_detects_anthropic_context_length_error():
    body = {
        "error": {
            "type": "invalid_request_error",
            "message": (
                "prompt is too long for model claude-3-5-sonnet. "
                "max tokens: 200000 prompt tokens: 240000"
            ),
        }
    }
    error = AnthropicBadRequestError(body["error"]["message"], response=_fake_response(), body=body)

    info = detect_context_length_error(error)

    assert info is not None
    assert info.provider == "anthropic"
    assert "prompt is too long" in info.message.lower()


def test_detects_gemini_context_length_error():
    error = genai_errors.APIError(
        400,
        {
            "error": {
                "status": "FAILED_PRECONDITION",
                "message": (
                    "The input to the model was too long. The requested input has 450000 tokens, "
                    "which exceeds the maximum of 320000 tokens for models/gemini-1.5-pro-002."
                ),
            }
        },
    )

    info = detect_context_length_error(error)

    assert info is not None
    assert info.provider == "gemini"
    assert "too long" in info.message.lower()


def test_non_context_error_returns_none():
    assert detect_context_length_error(ValueError("some other failure")) is None
