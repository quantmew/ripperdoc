"""Focused regression tests for Gemini provider error mapping helpers."""

from __future__ import annotations

import asyncio

import pytest

from ripperdoc.core.providers.base import call_with_timeout_and_retries
from ripperdoc.core.providers.errors import ProviderMappedError, ProviderTimeoutError
from ripperdoc.core.providers.gemini import (
    _classify_gemini_error,
    _map_gemini_exception,
    _run_with_provider_error_mapping,
)


def test_map_gemini_exception_converts_timeout_to_provider_timeout() -> None:
    mapped = _map_gemini_exception(asyncio.TimeoutError("request timed out"))
    assert isinstance(mapped, ProviderTimeoutError)


def test_classify_gemini_error_accepts_provider_mapped_error() -> None:
    code, message = _classify_gemini_error(
        ProviderMappedError("authentication_error", "Authentication failed: bad key")
    )
    assert code == "authentication_error"
    assert "bad key" in message


@pytest.mark.asyncio
async def test_gemini_timeout_mapping_reuses_shared_retry_logic(monkeypatch) -> None:
    attempts = {"count": 0}

    async def _no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr("ripperdoc.core.providers.base.asyncio.sleep", _no_sleep)

    async def flaky_request() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise ConnectionError("operation timed out")
        return "ok"

    result = await call_with_timeout_and_retries(
        lambda: _run_with_provider_error_mapping(flaky_request),
        request_timeout=None,
        max_retries=5,
    )

    assert result == "ok"
    assert attempts["count"] == 3
