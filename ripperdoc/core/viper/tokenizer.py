"""Tokenizer for the Viper language."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ripperdoc.core.viper.errors import ViperSyntaxError


KEYWORDS = {
    "if",
    "elif",
    "else",
    "while",
    "for",
    "in",
    "match",
    "case",
    "try",
    "except",
    "finally",
    "def",
    "class",
    "async",
    "return",
    "raise",
    "assert",
    "yield",
    "pass",
    "break",
    "continue",
    "import",
    "global",
    "nonlocal",
    "del",
    "type",
    "and",
    "or",
    "not",
    "is",
    "as",
    "from",
    "with",
    "await",
    "lambda",
    "True",
    "False",
    "None",
}

MULTI_CHAR_OPERATORS = {
    ":=",
    "+=",
    "-=",
    "*=",
    "/=",
    "//=",
    "%=",
    "**=",
    "@=",
    "|=",
    "&=",
    "^=",
    "<<=",
    ">>=",
    "<<",
    ">>",
    "==",
    "!=",
    "<=",
    ">=",
    "//",
    "**",
}

SINGLE_CHAR_OPERATORS = {
    "+",
    "-",
    "*",
    "/",
    "%",
    "@",
    "|",
    "&",
    "^",
    "~",
    "<",
    ">",
    "=",
    "(",
    ")",
    "[",
    "]",
    "{",
    "}",
    ",",
    ":",
    ".",
}

_PREFIX_CHARS = {"f", "t", "r", "b", "u"}


@dataclass(frozen=True)
class Token:
    kind: str
    value: str | bytes
    line: int
    column: int


class Tokenizer:
    """Convert Viper source text into a flat token stream."""

    def __init__(self, source: str) -> None:
        self.source = source.replace("\r\n", "\n").replace("\r", "\n")

    def tokenize(self) -> List[Token]:
        tokens: List[Token] = []
        indent_stack = [0]
        paren_level = 0
        i = 0
        line = 1
        col = 1
        at_line_start = True
        src = self.source
        length = len(src)

        if length == 0:
            return [Token("EOF", "", 1, 1)]

        while i < length:
            ch = src[i]

            if at_line_start and paren_level == 0:
                indent_width = 0
                while i < length and src[i] in " \t":
                    indent_width += 4 if src[i] == "\t" else 1
                    i += 1
                    col += 1
                if i >= length:
                    break
                if src[i] == "\n":
                    i += 1
                    line += 1
                    col = 1
                    at_line_start = True
                    continue
                if src[i] == "#":
                    while i < length and src[i] != "\n":
                        i += 1
                        col += 1
                    if i < length and src[i] == "\n":
                        i += 1
                        line += 1
                        col = 1
                    at_line_start = True
                    continue

                if indent_width > indent_stack[-1]:
                    indent_stack.append(indent_width)
                    tokens.append(Token("INDENT", "", line, 1))
                else:
                    while indent_width < indent_stack[-1]:
                        indent_stack.pop()
                        tokens.append(Token("DEDENT", "", line, 1))
                    if indent_width != indent_stack[-1]:
                        raise ViperSyntaxError("Inconsistent indentation", line, 1)
                at_line_start = False
                continue

            if ch in " \t":
                i += 1
                col += 1
                continue

            if ch == "#":
                comment_col = col
                start = i
                while i < length and src[i] != "\n":
                    i += 1
                    col += 1
                if not at_line_start:
                    comment = src[start + 1 : i].lstrip()
                    if comment.startswith("type:"):
                        value = comment[len("type:") :].strip()
                        tokens.append(Token("TYPE_COMMENT", value, line, comment_col))
                continue
            if ch == "\n":
                if paren_level == 0:
                    tokens.append(Token("NEWLINE", "", line, col))
                i += 1
                line += 1
                col = 1
                at_line_start = True
                continue
            if ch == ";":
                if paren_level != 0:
                    raise ViperSyntaxError("Unexpected ';'", line, col)
                tokens.append(Token("NEWLINE", "", line, col))
                i += 1
                col += 1
                at_line_start = False
                continue

            prefix = self._read_prefix(src, i)
            if prefix is not None:
                prefix_text, quote_idx, quote_char, triple = prefix
                kind, raw, is_bytes = self._classify_prefix(prefix_text, line, col)
                if is_bytes and kind in {"FSTRING", "TSTRING"}:
                    raise ViperSyntaxError("Bytes prefix not allowed for formatted strings", line, col)
                start_line = line
                start_col = col
                i = quote_idx
                col += len(prefix_text)
                value, i, line, col = self._read_string(
                    src,
                    i,
                    line,
                    col,
                    quote_char,
                    triple,
                    raw,
                    kind == "BYTES",
                )
                tokens.append(Token(kind, value, start_line, start_col))
                at_line_start = False
                continue

            if ch.isdigit():
                value, i = self._read_number(src, i, line)
                tokens.append(Token("NUMBER", value, line, col))
                col += len(value)
                at_line_start = False
                continue

            if ch.isalpha() or ch == "_":
                value, i = self._read_identifier(src, i)
                kind = "KEYWORD" if value in KEYWORDS else "NAME"
                tokens.append(Token(kind, value, line, col))
                col += len(value)
                at_line_start = False
                continue

            op = self._read_operator(src, i)
            if op is None:
                raise ViperSyntaxError(f"Unexpected character: {ch}", line, col)
            tokens.append(Token("OP", op, line, col))
            i += len(op)
            col += len(op)
            if op in "([{":
                paren_level += 1
            elif op in ")]}":
                paren_level -= 1
                if paren_level < 0:
                    raise ViperSyntaxError("Unmatched closing bracket", line, col)
            at_line_start = False

        if paren_level != 0:
            raise ViperSyntaxError("Unclosed bracket", line, col)

        if not at_line_start:
            tokens.append(Token("NEWLINE", "", line, col))

        while len(indent_stack) > 1:
            indent_stack.pop()
            tokens.append(Token("DEDENT", "", line, 1))

        tokens.append(Token("EOF", "", line, col))
        return tokens

    def _read_prefix(
        self, source: str, start: int
    ) -> Optional[tuple[str, int, str, bool]]:
        idx = start
        prefix = ""
        while idx < len(source) and source[idx].lower() in _PREFIX_CHARS:
            prefix += source[idx].lower()
            idx += 1
        if prefix:
            if idx < len(source) and source[idx] in "\"'":
                quote = source[idx]
                triple = source.startswith(quote * 3, idx)
                return prefix, idx, quote, triple
            return None
        if source[start] in "\"'":
            quote = source[start]
            triple = source.startswith(quote * 3, start)
            return "", start, quote, triple
        return None

    def _classify_prefix(self, prefix: str, line: int, column: int) -> tuple[str, bool, bool]:
        if len(prefix) != len(set(prefix)):
            raise ViperSyntaxError("Invalid string prefix", line, column)
        is_f = "f" in prefix
        is_t = "t" in prefix
        if is_f and is_t:
            raise ViperSyntaxError("Mixed f/t string prefix", line, column)
        raw = "r" in prefix
        is_bytes = "b" in prefix
        kind = "STRING"
        if is_f:
            kind = "BFSTRING" if is_bytes else "FSTRING"
        elif is_t:
            kind = "BTSTRING" if is_bytes else "TSTRING"
        elif is_bytes:
            kind = "BYTES"
        return kind, raw, is_bytes

    def _read_string(
        self,
        source: str,
        start: int,
        line: int,
        column: int,
        quote: str,
        triple: bool,
        raw: bool,
        as_bytes: bool,
    ) -> tuple[str | bytes, int, int, int]:
        idx = start
        if triple:
            idx += 3
            column += 3
        else:
            idx += 1
            column += 1
        if as_bytes:
            value_bytes = bytearray()
        else:
            value_chars: List[str] = []
        while idx < len(source):
            ch = source[idx]
            if triple and source.startswith(quote * 3, idx):
                idx += 3
                column += 3
                if as_bytes:
                    return bytes(value_bytes), idx, line, column
                return "".join(value_chars), idx, line, column
            if not triple and ch == quote:
                idx += 1
                column += 1
                if as_bytes:
                    return bytes(value_bytes), idx, line, column
                return "".join(value_chars), idx, line, column
            if ch == "\n":
                if not triple:
                    raise ViperSyntaxError("Unterminated string literal", line, column)
                if as_bytes:
                    value_bytes.extend(b"\n")
                else:
                    value_chars.append("\n")
                idx += 1
                line += 1
                column = 1
                continue
            if ch == "\\" and not raw:
                idx += 1
                column += 1
                if idx >= len(source):
                    break
                esc = source[idx]
                if esc == "\n":
                    idx += 1
                    line += 1
                    column = 1
                    continue
                if as_bytes:
                    chunk, idx, line, column = self._read_escape_bytes(source, idx, line, column)
                    value_bytes.extend(chunk)
                else:
                    chunk, idx, line, column = self._read_escape_text(source, idx, line, column)
                    value_chars.append(chunk)
                continue
            if as_bytes and ord(ch) > 0x7F:
                raise ViperSyntaxError("Bytes literals may only contain ASCII", line, column)
            if as_bytes:
                value_bytes.extend(ch.encode("utf-8"))
            else:
                value_chars.append(ch)
            idx += 1
            column += 1
        raise ViperSyntaxError("Unterminated string literal", line, column)

    def _escape_char(self, esc: str) -> str:
        mapping = {
            "n": "\n",
            "t": "\t",
            "r": "\r",
            "\\": "\\",
            "\"": "\"",
            "'": "'",
        }
        return mapping.get(esc, esc)

    def _read_escape_text(
        self, source: str, idx: int, line: int, column: int
    ) -> tuple[str, int, int, int]:
        esc = source[idx]
        if esc in "ntr\\\"'":
            return self._escape_char(esc), idx + 1, line, column + 1
        if esc == "a":
            return "\a", idx + 1, line, column + 1
        if esc == "b":
            return "\b", idx + 1, line, column + 1
        if esc == "f":
            return "\f", idx + 1, line, column + 1
        if esc == "v":
            return "\v", idx + 1, line, column + 1
        if esc == "x":
            value, idx, column = self._read_hex_escape(source, idx + 1, 2, line, column + 1)
            return chr(value), idx, line, column
        if esc == "u":
            value, idx, column = self._read_hex_escape(source, idx + 1, 4, line, column + 1)
            return chr(value), idx, line, column
        if esc == "U":
            value, idx, column = self._read_hex_escape(source, idx + 1, 8, line, column + 1)
            return chr(value), idx, line, column
        if esc == "N":
            return self._read_named_unicode(source, idx + 1, line, column + 1)
        if esc in "01234567":
            value, idx, column = self._read_octal_escape(source, idx, line, column)
            return chr(value), idx, line, column
        return esc, idx + 1, line, column + 1

    def _read_escape_bytes(
        self, source: str, idx: int, line: int, column: int
    ) -> tuple[bytes, int, int, int]:
        esc = source[idx]
        if esc in "ntr\\\"'":
            return self._escape_char(esc).encode("utf-8"), idx + 1, line, column + 1
        if esc == "a":
            return b"\a", idx + 1, line, column + 1
        if esc == "b":
            return b"\b", idx + 1, line, column + 1
        if esc == "f":
            return b"\f", idx + 1, line, column + 1
        if esc == "v":
            return b"\v", idx + 1, line, column + 1
        if esc == "x":
            value, idx, column = self._read_hex_escape(source, idx + 1, 2, line, column + 1)
            return bytes([value]), idx, line, column
        if esc in {"u", "U", "N"}:
            raise ViperSyntaxError("Bytes escapes must be ASCII", line, column)
        if esc in "01234567":
            value, idx, column = self._read_octal_escape(source, idx, line, column)
            if value > 0xFF:
                raise ViperSyntaxError("Octal escape out of range", line, column)
            return bytes([value]), idx, line, column
        return esc.encode("utf-8"), idx + 1, line, column + 1

    def _read_hex_escape(
        self, source: str, idx: int, length: int, line: int, column: int
    ) -> tuple[int, int, int]:
        if idx + length > len(source):
            raise ViperSyntaxError("Invalid hex escape", line, column)
        chunk = source[idx : idx + length]
        if any(ch not in "0123456789abcdefABCDEF" for ch in chunk):
            raise ViperSyntaxError("Invalid hex escape", line, column)
        value = int(chunk, 16)
        return value, idx + length, column + length

    def _read_octal_escape(
        self, source: str, idx: int, line: int, column: int
    ) -> tuple[int, int, int]:
        start = idx
        while idx < len(source) and source[idx] in "01234567" and idx - start < 3:
            idx += 1
        value = int(source[start:idx], 8)
        return value, idx, column + (idx - start)

    def _read_named_unicode(
        self, source: str, idx: int, line: int, column: int
    ) -> tuple[str, int, int, int]:
        if idx >= len(source) or source[idx] != "{":
            raise ViperSyntaxError("Invalid named unicode escape", line, column)
        end = source.find("}", idx + 1)
        if end == -1:
            raise ViperSyntaxError("Invalid named unicode escape", line, column)
        name = source[idx + 1 : end]
        if not name:
            raise ViperSyntaxError("Invalid named unicode escape", line, column)
        try:
            import unicodedata

            value = unicodedata.lookup(name)
        except Exception as exc:  # noqa: BLE001
            raise ViperSyntaxError("Unknown unicode name", line, column) from exc
        return value, end + 1, line, column + (end + 1 - idx)

    def _read_number(self, source: str, start: int, line: int) -> tuple[str, int]:
        idx = start
        while idx < len(source) and source[idx].isdigit():
            idx += 1
        if idx < len(source) and source[idx] == "." and idx + 1 < len(source) and source[idx + 1].isdigit():
            idx += 1
            while idx < len(source) and source[idx].isdigit():
                idx += 1
        if idx < len(source) and source[idx] in "eE":
            idx += 1
            if idx < len(source) and source[idx] in "+-":
                idx += 1
            if idx >= len(source) or not source[idx].isdigit():
                raise ViperSyntaxError("Invalid exponent in number", line, start + 1)
            while idx < len(source) and source[idx].isdigit():
                idx += 1
        return source[start:idx], idx

    def _read_identifier(self, source: str, start: int) -> tuple[str, int]:
        idx = start
        while idx < len(source) and (source[idx].isalnum() or source[idx] == "_"):
            idx += 1
        return source[start:idx], idx

    def _read_operator(self, source: str, start: int) -> str | None:
        for op in sorted(MULTI_CHAR_OPERATORS, key=len, reverse=True):
            if source.startswith(op, start):
                return op
        if source[start] in SINGLE_CHAR_OPERATORS:
            return source[start]
        return None


def tokenize(source: str) -> List[Token]:
    """Tokenize Viper source text into a list of tokens."""
    return Tokenizer(source).tokenize()
