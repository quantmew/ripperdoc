"""JSON-RPC protocol DTOs."""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict

DEFAULT_PROTOCOL_VERSION = "2025-11-25"


class JsonRpcErrorCodes(IntEnum):
    """Subset of JSON-RPC error codes used by the protocol."""

    ConnectionClosed = -32000
    RequestTimeout = -32001
    ParseError = -32700
    InvalidRequest = -32600
    MethodNotFound = -32601
    InvalidParams = -32602
    InternalError = -32603
    UrlElicitationRequired = -32042


class JsonRpcError(BaseModel):
    """JSON-RPC error envelope payload."""

    code: int
    message: str
    data: Optional[Any] = None


class JsonRpcResponse(BaseModel):
    """JSON-RPC success/error response for an in-flight request."""

    jsonrpc: str = "2.0"
    id: Union[str, int]
    result: Optional[Any] = None
    error: Optional[JsonRpcError] = None

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )


class JsonRpcResponseError(Exception):
    """Typed exception for raising JSON-RPC style errors from awaited calls."""

    def __init__(
        self,
        code: int,
        message: str,
        data: Optional[Any] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


__all__ = [
    "DEFAULT_PROTOCOL_VERSION",
    "JsonRpcErrorCodes",
    "JsonRpcError",
    "JsonRpcResponse",
    "JsonRpcResponseError",
]
