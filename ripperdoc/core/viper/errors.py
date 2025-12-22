"""Error types for the Viper interpreter."""

from __future__ import annotations


class ViperError(Exception):
    """Base class for Viper-related errors."""

    def __init__(self, message: str, line: int | None = None, column: int | None = None) -> None:
        self.message = message
        self.line = line
        self.column = column
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        if self.line is None:
            return self.message
        if self.column is None:
            return f"{self.message} (line {self.line})"
        return f"{self.message} (line {self.line}, column {self.column})"


class ViperSyntaxError(ViperError):
    """Raised when parsing or tokenizing fails."""


class ViperRuntimeError(ViperError):
    """Raised when execution fails."""
