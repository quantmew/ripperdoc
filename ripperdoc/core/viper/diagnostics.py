"""Diagnostic rendering for Viper errors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ripperdoc.core.viper.errors import ViperError


@dataclass
class ViperDiagnostic:
    """Structured diagnostic for Viper errors."""

    message: str
    line: Optional[int]
    column: Optional[int]
    file_path: Optional[Path]
    source_line: Optional[str]
    hint: Optional[str] = None

    def format(self) -> str:
        lines = [f"Viper error: {self.message}"]
        if self.line is not None and self.column is not None:
            location = (
                f"{self.file_path}:{self.line}:{self.column}"
                if self.file_path is not None
                else f"{self.line}:{self.column}"
            )
            lines.append(f"  --> {location}")
            if self.source_line is not None:
                line_label = str(self.line)
                lines.append(f"  {line_label} | {self.source_line}")
                caret_offset = max(self.column - 1, 0)
                caret_pad = " " * (len(line_label) + 3 + caret_offset)
                lines.append(f"  {caret_pad}^")
        if self.hint:
            lines.append(f"  help: {self.hint}")
        return "\n".join(lines)


def _extract_source_line(source: str, line: Optional[int]) -> Optional[str]:
    if not source or line is None:
        return None
    lines = source.splitlines()
    if 1 <= line <= len(lines):
        return lines[line - 1]
    return None


def _hint_for_error(message: str, source_line: Optional[str]) -> Optional[str]:
    if "Unsupported statement 'with'" in message:
        return "The Viper language does not support 'with' statements yet."
    if source_line and source_line.lstrip().startswith("with "):
        return "The Viper language does not support 'with' statements yet."
    return None


def build_diagnostic(
    error: ViperError, source: str, file_path: Optional[Path] = None
) -> ViperDiagnostic:
    message = getattr(error, "message", str(error))
    line = getattr(error, "line", None)
    column = getattr(error, "column", None)
    source_line = _extract_source_line(source, line)
    hint = _hint_for_error(message, source_line)
    return ViperDiagnostic(
        message=message,
        line=line,
        column=column,
        file_path=file_path,
        source_line=source_line,
        hint=hint,
    )


def format_viper_diagnostic(
    error: ViperError, source: str, file_path: Optional[Path] = None
) -> str:
    return build_diagnostic(error, source, file_path).format()
