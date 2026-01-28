"""Error types for Ripperdoc SDK.

These exceptions follow the same structure as Claude Agent SDK errors
for API compatibility.
"""

from typing import Any


class RipperdocSDKError(Exception):
    """Base exception for all SDK errors."""

    def __str__(self) -> str:
        return self.args[0] if self.args else "An error occurred in the Ripperdoc SDK"


class CLIConnectionError(RipperdocSDKError):
    """Raised when unable to connect to the CLI service."""


class CLINotFoundError(CLIConnectionError):
    """Raised when the CLI is not found or not installed."""


class ProcessError(RipperdocSDKError):
    """Raised when the CLI process fails."""

    def __init__(self, message: str, exit_code: int | None = None, **kwargs):
        super().__init__(message)
        self.exit_code = exit_code
        self.extra = kwargs


class CLIJSONDecodeError(RipperdocSDKError):
    """Raised when unable to decode JSON from CLI output."""


class MessageParseError(RipperdocSDKError):
    """Raised when unable to parse a message from CLI output."""

    def __init__(self, message: str, data: Any = None, **kwargs):
        super().__init__(message)
        self.data = data
        self.extra = kwargs


# For API compatibility with Claude SDK, also export these aliases
SDKError = RipperdocSDKError
JSONDecodeError = CLIJSONDecodeError


__all__ = [
    "RipperdocSDKError",
    "CLIConnectionError",
    "CLINotFoundError",
    "ProcessError",
    "CLIJSONDecodeError",
    "MessageParseError",
    # Compatibility aliases
    "SDKError",
    "JSONDecodeError",
]
