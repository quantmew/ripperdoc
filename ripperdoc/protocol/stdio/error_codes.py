"""Shared JSON-RPC error code helpers for stdio handlers."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from ripperdoc.protocol.models import JsonRpcErrorCodes


_MCP_VALIDATION_EXCEPTIONS = (ValidationError, TypeError, ValueError, KeyError, IndexError)
_DEFAULT_INVALID_PARAMS = JsonRpcErrorCodes.InvalidParams


def resolve_protocol_error_code(
    error: Any,
    *,
    default: int = JsonRpcErrorCodes.InternalError,
    invalid_request: int = _DEFAULT_INVALID_PARAMS,
) -> int:
    """Resolve an error to a protocol-facing JSON-RPC code.

    Resolution order:
    1. Use explicit ``code`` attribute if present and integer.
    2. Map validation-like failures to ``InvalidParams`` by default.
    3. Fall back to the provided default (usually ``InternalError``).
    """
    explicit_code = getattr(error, "code", None)
    if isinstance(explicit_code, int):
        return int(explicit_code)

    if isinstance(error, _MCP_VALIDATION_EXCEPTIONS):
        return int(invalid_request)

    if isinstance(default, int):
        return int(default)

    return int(JsonRpcErrorCodes.InternalError)


def resolve_protocol_request_error_code(
    error: Any,
    *,
    default: int = JsonRpcErrorCodes.InvalidParams,
    invalid_request: int = JsonRpcErrorCodes.InvalidParams,
) -> int:
    """Backward-compatible wrapper for protocol request/validation failures."""
    return resolve_protocol_error_code(
        error,
        default=default,
        invalid_request=invalid_request,
    )


def resolve_query_initialize_error_code(
    error: Any,
    *,
    default: int = JsonRpcErrorCodes.InvalidParams,
    invalid_request: int = JsonRpcErrorCodes.InvalidParams,
) -> int:
    """Backward-compatible wrapper for query/initialize validation mapping."""
    return resolve_protocol_request_error_code(
        error,
        default=default,
        invalid_request=invalid_request,
    )


def resolve_control_error_code(
    error: Any,
    *,
    default: int = JsonRpcErrorCodes.InvalidParams,
    invalid_request: int = JsonRpcErrorCodes.InvalidParams,
) -> int:
    """Resolve a unified error code for control channel failures."""
    return resolve_protocol_error_code(
        error,
        default=default,
        invalid_request=invalid_request,
    )


def resolve_control_request_error_code(
    error: Any,
    *,
    default: int = JsonRpcErrorCodes.InvalidParams,
    invalid_request: int = JsonRpcErrorCodes.InvalidParams,
) -> int:
    """Resolve error code for control request handling failures."""
    return resolve_control_error_code(
        error,
        default=default,
        invalid_request=invalid_request,
    )


def resolve_mcp_error_code(
    error: Any,
    *,
    default: int = JsonRpcErrorCodes.InternalError,
    invalid_request: int = JsonRpcErrorCodes.InvalidParams,
) -> int:
    """Resolve a JSON-RPC error code using shared MCP conventions."""
    return resolve_protocol_error_code(
        error,
        default=default,
        invalid_request=invalid_request,
    )


def resolve_jsonrpc_error_code(
    error: Any,
    *,
    default: int = JsonRpcErrorCodes.InternalError,
    invalid_request: int = JsonRpcErrorCodes.InvalidParams,
) -> int:
    """Resolve a JSON-RPC style error code using shared MCP rules."""
    return resolve_protocol_error_code(
        error,
        default=default,
        invalid_request=invalid_request,
    )


def resolve_query_control_error_code(
    error: Any,
    *,
    default: int = JsonRpcErrorCodes.InternalError,
    invalid_request: int = JsonRpcErrorCodes.InvalidParams,
) -> int:
    """Resolve error code for query/control dispatch failures."""
    return resolve_protocol_error_code(
        error,
        default=default,
        invalid_request=invalid_request,
    )
