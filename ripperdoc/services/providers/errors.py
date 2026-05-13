"""Shared provider error types for cross-protocol normalization."""

from __future__ import annotations


class ProviderMappedError(Exception):
    """Normalized provider exception with a stable error code."""

    def __init__(self, error_code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.retryable = retryable


class ProviderTimeoutError(ProviderMappedError):
    """Timeout error normalized across providers."""

    def __init__(self, message: str) -> None:
        super().__init__("timeout", message, retryable=True)


class ProviderConnectionError(ProviderMappedError):
    """Connection-level transport error."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__("connection_error", message, retryable=retryable)


class ProviderRateLimitError(ProviderMappedError):
    """Rate limit exceeded."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__("rate_limit", message, retryable=retryable)


class ProviderAuthenticationError(ProviderMappedError):
    """Authentication failure."""

    def __init__(self, message: str) -> None:
        super().__init__("authentication_error", message)


class ProviderPermissionDeniedError(ProviderMappedError):
    """Permission denied."""

    def __init__(self, message: str) -> None:
        super().__init__("permission_denied", message)


class ProviderInsufficientBalanceError(ProviderMappedError):
    """Insufficient account balance/credits."""

    def __init__(self, message: str) -> None:
        super().__init__("insufficient_balance", message)


class ProviderModelNotFoundError(ProviderMappedError):
    """Requested model not found."""

    def __init__(self, message: str) -> None:
        super().__init__("model_not_found", message)


class ProviderBadRequestError(ProviderMappedError):
    """Malformed/invalid request."""

    def __init__(self, message: str) -> None:
        super().__init__("bad_request", message)


class ProviderContextLengthExceededError(ProviderMappedError):
    """Context length/token limit exceeded."""

    def __init__(self, message: str) -> None:
        super().__init__("context_length_exceeded", message)


class ProviderContentPolicyViolationError(ProviderMappedError):
    """Content policy violation."""

    def __init__(self, message: str) -> None:
        super().__init__("content_policy_violation", message)


class ProviderServiceUnavailableError(ProviderMappedError):
    """Service unavailable/transient provider outage."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__("service_unavailable", message, retryable=retryable)


class ProviderApiError(ProviderMappedError):
    """Generic upstream API error."""

    def __init__(self, message: str) -> None:
        super().__init__("api_error", message)
