"""Parser for Viper formatted strings (f-strings and t-strings)."""

from __future__ import annotations

from typing import List, Optional, Tuple

from ripperdoc.core.viper.ast_nodes import FormatField, FormattedString, StringText, FStringPart
from ripperdoc.core.viper.errors import ViperSyntaxError
from ripperdoc.core.viper.parser import parse_expression_source


def parse_formatted_string(
    raw: str, *, is_template: bool, is_bytes: bool, line: int, column: int
) -> FormattedString:
    parts: List[FStringPart] = []
    text_buffer: List[str] = []
    idx = 0

    def flush_text() -> None:
        if text_buffer:
            text = "".join(text_buffer)
            parts.append(StringText(value=text, line=line, column=column))
            text_buffer.clear()

    while idx < len(raw):
        ch = raw[idx]
        if ch == "{":
            if _peek(raw, idx) == "{":
                text_buffer.append("{")
                idx += 2
                continue
            flush_text()
            field, idx = _parse_replacement(raw, idx + 1, line, column)
            parts.append(field)
            continue
        if ch == "}":
            if _peek(raw, idx) == "}":
                text_buffer.append("}")
                idx += 2
                continue
            raise ViperSyntaxError("Single '}' in formatted string", line, column + idx)
        text_buffer.append(ch)
        idx += 1

    flush_text()
    return FormattedString(
        parts=parts,
        is_template=is_template,
        is_bytes=is_bytes,
        line=line,
        column=column,
    )


def _parse_replacement(
    raw: str, start: int, line: int, column: int
) -> Tuple[FormatField, int]:
    idx = start
    expr_start = idx
    paren_depth = 0
    bracket_depth = 0
    brace_depth = 0
    quote: Optional[str] = None
    debug = False

    while idx < len(raw):
        ch = raw[idx]
        if quote is not None:
            if ch == "\\":
                idx += 2
                continue
            if ch == quote:
                quote = None
            idx += 1
            continue

        if ch in ("'", '"'):
            quote = ch
            idx += 1
            continue

        if ch == "(":
            paren_depth += 1
            idx += 1
            continue
        if ch == ")":
            paren_depth = max(paren_depth - 1, 0)
            idx += 1
            continue
        if ch == "[":
            bracket_depth += 1
            idx += 1
            continue
        if ch == "]":
            bracket_depth = max(bracket_depth - 1, 0)
            idx += 1
            continue
        if ch == "{":
            brace_depth += 1
            idx += 1
            continue
        if ch == "}":
            if brace_depth > 0:
                brace_depth -= 1
                idx += 1
                continue
            expr_text = raw[expr_start:idx].strip()
            if not expr_text:
                raise ViperSyntaxError("Empty expression in formatted string", line, column + start)
            expr = parse_expression_source(expr_text)
            return (
                FormatField(
                    expression=expr,
                    conversion=None,
                    format_spec=None,
                    debug=False,
                    expr_text=None,
                    line=line,
                    column=column + start,
                ),
                idx + 1,
            )

        if paren_depth == 0 and bracket_depth == 0 and brace_depth == 0:
            if ch == "=" and _is_debug_separator(raw, idx):
                debug = True
                expr_text = raw[expr_start:idx].strip()
                if not expr_text:
                    raise ViperSyntaxError(
                        "Empty expression in formatted string", line, column + start
                    )
                idx += 1
                while idx < len(raw) and raw[idx].isspace():
                    idx += 1
                return _parse_after_expr(raw, idx, expr_text, debug, line, column + start)
            if ch in ("!", ":"):
                expr_text = raw[expr_start:idx].strip()
                if not expr_text:
                    raise ViperSyntaxError(
                        "Empty expression in formatted string", line, column + start
                    )
                return _parse_after_expr(raw, idx, expr_text, debug, line, column + start)

        idx += 1

    raise ViperSyntaxError("Unterminated format field", line, column + start)


def _parse_after_expr(
    raw: str,
    idx: int,
    expr_text: str,
    debug: bool,
    line: int,
    column: int,
) -> Tuple[FormatField, int]:
    conversion = None
    format_spec: Optional[List[FStringPart]] = None

    if idx < len(raw) and raw[idx] == "!":
        idx += 1
        name_start = idx
        while idx < len(raw) and (raw[idx].isalnum() or raw[idx] == "_"):
            idx += 1
        if name_start == idx:
            raise ViperSyntaxError("Expected conversion name", line, column)
        conversion = raw[name_start:idx]

    if idx < len(raw) and raw[idx] == ":":
        format_spec, idx = _parse_format_spec(raw, idx + 1, line, column)

    if idx >= len(raw) or raw[idx] != "}":
        raise ViperSyntaxError("Unterminated format field", line, column)

    expr = parse_expression_source(expr_text)
    return (
        FormatField(
            expression=expr,
            conversion=conversion,
            format_spec=format_spec,
            debug=debug,
            expr_text=expr_text,
            line=line,
            column=column,
        ),
        idx + 1,
    )


def _parse_format_spec(
    raw: str, start: int, line: int, column: int
) -> Tuple[List[FStringPart], int]:
    parts: List[FStringPart] = []
    text_buffer: List[str] = []
    idx = start

    def flush_text() -> None:
        if text_buffer:
            text = "".join(text_buffer)
            parts.append(StringText(value=text, line=line, column=column))
            text_buffer.clear()

    while idx < len(raw):
        ch = raw[idx]
        if ch == "{":
            if _peek(raw, idx) == "{":
                text_buffer.append("{")
                idx += 2
                continue
            flush_text()
            field, idx = _parse_replacement(raw, idx + 1, line, column)
            parts.append(field)
            continue
        if ch == "}":
            if _peek(raw, idx) == "}":
                text_buffer.append("}")
                idx += 2
                continue
            flush_text()
            return parts, idx
        text_buffer.append(ch)
        idx += 1

    raise ViperSyntaxError("Unterminated format spec", line, column)


def _peek(raw: str, idx: int) -> Optional[str]:
    next_idx = idx + 1
    if next_idx >= len(raw):
        return None
    return raw[next_idx]


def _is_debug_separator(raw: str, idx: int) -> bool:
    next_ch = raw[idx + 1] if idx + 1 < len(raw) else ""
    prev_ch = raw[idx - 1] if idx - 1 >= 0 else ""
    if next_ch == "=":
        return False
    if prev_ch in ("<", ">", "=", "!"):
        return False
    return True
