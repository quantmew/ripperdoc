"""Error types for remote-control bridge runtime."""

from __future__ import annotations


class BridgeFatalError(RuntimeError):
    """Fatal control-plane error that should terminate the bridge loop."""

    def __init__(self, message: str, *, status: int, error_type: str | None = None):
        super().__init__(message)
        self.status = status
        self.error_type = error_type
