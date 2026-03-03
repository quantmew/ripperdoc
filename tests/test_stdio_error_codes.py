"""Error-code mapping tests for stdio protocol helpers."""

from __future__ import annotations

from pydantic import BaseModel, ValidationError

from ripperdoc.protocol.models import JsonRpcErrorCodes
from ripperdoc.protocol.stdio.error_codes import resolve_protocol_request_error_code


class _SimplePayload(BaseModel):
    value: int


def test_resolve_error_code_prefers_explicit_code() -> None:
    exc = ValueError("bad input")
    setattr(exc, "code", JsonRpcErrorCodes.InternalError)

    assert resolve_protocol_request_error_code(
        exc, default=JsonRpcErrorCodes.InvalidParams
    ) == JsonRpcErrorCodes.InternalError


def test_resolve_error_code_maps_validation_error_to_invalid_params() -> None:
    try:
        _SimplePayload.model_validate({"value": "not-int"})
    except ValidationError as exc:
        code = resolve_protocol_request_error_code(
            exc,
            default=JsonRpcErrorCodes.InternalError,
            invalid_request=JsonRpcErrorCodes.InvalidParams,
        )
    else:  # pragma: no cover
        raise AssertionError("Expected validation error")

    assert code == int(JsonRpcErrorCodes.InvalidParams)
