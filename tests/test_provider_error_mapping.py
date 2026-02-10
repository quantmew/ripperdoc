"""Tests for shared provider error-mapping helpers."""

from __future__ import annotations

import pytest

from ripperdoc.core.providers.error_mapping import (
    classify_mapped_error,
    map_bad_request_error,
    map_connection_error,
    map_permission_denied_error,
    run_with_exception_mapper,
)
from ripperdoc.core.providers.errors import (
    ProviderConnectionError,
    ProviderContentPolicyViolationError,
    ProviderContextLengthExceededError,
    ProviderInsufficientBalanceError,
    ProviderMappedError,
    ProviderPermissionDeniedError,
    ProviderTimeoutError,
)


def test_map_connection_error_distinguishes_timeout() -> None:
    timeout_err = map_connection_error("Request timed out")
    conn_err = map_connection_error("TLS handshake failed")

    assert isinstance(timeout_err, ProviderTimeoutError)
    assert isinstance(conn_err, ProviderConnectionError)
    assert conn_err.error_code == "connection_error"


def test_map_bad_request_error_variants() -> None:
    context_err = map_bad_request_error("context window exceeded")
    policy_err = map_bad_request_error("content policy violation")
    bad_req_err = map_bad_request_error("invalid json schema")

    assert isinstance(context_err, ProviderContextLengthExceededError)
    assert isinstance(policy_err, ProviderContentPolicyViolationError)
    assert context_err.error_code == "context_length_exceeded"
    assert policy_err.error_code == "content_policy_violation"
    assert bad_req_err.error_code == "bad_request"


def test_map_permission_denied_specializes_insufficient_balance() -> None:
    balance_err = map_permission_denied_error("insufficient balance")
    denied_err = map_permission_denied_error("forbidden")

    assert isinstance(balance_err, ProviderInsufficientBalanceError)
    assert isinstance(denied_err, ProviderPermissionDeniedError)


@pytest.mark.asyncio
async def test_run_with_exception_mapper_maps_and_classifies() -> None:
    async def _raise_raw() -> str:
        raise ValueError("boom")

    def _mapper(exc: Exception) -> Exception:
        if isinstance(exc, ValueError):
            return ProviderMappedError("bad_request", f"Invalid request: {exc}")
        return exc

    with pytest.raises(ProviderMappedError) as exc_info:
        await run_with_exception_mapper(_raise_raw, _mapper)

    assert classify_mapped_error(exc_info.value) == ("bad_request", str(exc_info.value))
