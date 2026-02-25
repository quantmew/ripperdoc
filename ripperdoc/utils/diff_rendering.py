"""Utilities for parsing and formatting unified diff lines."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Optional, Sequence

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")

DiffLineKind = Literal["hunk", "add", "del", "context"]


@dataclass(frozen=True)
class NumberedDiffLine:
    """Unified diff line with optional old/new line numbers."""

    kind: DiffLineKind
    raw: str
    content: str
    old_line: Optional[int] = None
    new_line: Optional[int] = None


@dataclass(frozen=True)
class NumberedDiffLayout:
    """Parsed diff lines and computed line-number column widths."""

    lines: list[NumberedDiffLine]
    old_width: int
    new_width: int


def build_numbered_diff_layout(diff_lines: Sequence[str]) -> NumberedDiffLayout:
    """Parse unified diff lines and compute aligned old/new number widths."""
    old_cursor: Optional[int] = None
    new_cursor: Optional[int] = None
    max_old: Optional[int] = None
    max_new: Optional[int] = None
    numbered: list[NumberedDiffLine] = []

    def _track_old(value: Optional[int]) -> None:
        nonlocal max_old
        if value is None:
            return
        if max_old is None or value > max_old:
            max_old = value

    def _track_new(value: Optional[int]) -> None:
        nonlocal max_new
        if value is None:
            return
        if max_new is None or value > max_new:
            max_new = value

    for raw_line in diff_lines:
        line = str(raw_line).rstrip("\r\n")
        if line.startswith("@@"):
            match = _HUNK_RE.search(line)
            if match:
                old_cursor = int(match.group(1))
                new_cursor = int(match.group(2))
                _track_old(old_cursor)
                _track_new(new_cursor)
            numbered.append(NumberedDiffLine(kind="hunk", raw=line, content=line))
            continue

        if line.startswith("+") and not line.startswith("+++"):
            _track_new(new_cursor)
            numbered.append(
                NumberedDiffLine(
                    kind="add",
                    raw=line,
                    content=line[1:],
                    old_line=None,
                    new_line=new_cursor,
                )
            )
            if new_cursor is not None:
                new_cursor += 1
            continue

        if line.startswith("-") and not line.startswith("---"):
            _track_old(old_cursor)
            numbered.append(
                NumberedDiffLine(
                    kind="del",
                    raw=line,
                    content=line[1:],
                    old_line=old_cursor,
                    new_line=None,
                )
            )
            if old_cursor is not None:
                old_cursor += 1
            continue

        _track_old(old_cursor)
        _track_new(new_cursor)
        content = line[1:] if line.startswith(" ") else line
        numbered.append(
            NumberedDiffLine(
                kind="context",
                raw=line,
                content=content,
                old_line=old_cursor,
                new_line=new_cursor,
            )
        )
        if old_cursor is not None:
            old_cursor += 1
        if new_cursor is not None:
            new_cursor += 1

    old_width = len(str(max_old)) if max_old is not None else 1
    new_width = len(str(max_new)) if max_new is not None else 1
    return NumberedDiffLayout(lines=numbered, old_width=old_width, new_width=new_width)


def format_number(value: Optional[int], width: int) -> str:
    """Render a right-aligned number or a blank column."""
    if value is None:
        return " " * width
    return f"{value:>{width}d}"


def format_numbered_diff_text(
    diff_line: NumberedDiffLine,
    *,
    old_width: int,
    new_width: int,
) -> str:
    """Format one parsed diff line into aligned two-column text."""
    if diff_line.kind == "hunk":
        return diff_line.raw

    left = format_number(diff_line.old_line, old_width)
    right = format_number(diff_line.new_line, new_width)
    if diff_line.kind == "context":
        return f"{left}   {right}   {diff_line.content}"

    marker = "+" if diff_line.kind == "add" else "-"
    return f"{left}   {right} {marker} {diff_line.content}"
