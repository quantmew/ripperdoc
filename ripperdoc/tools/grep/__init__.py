"""Grep tool for searching code."""

from ripperdoc.tools.grep._tool import (
    GrepMatch,
    GrepTool,
    GrepToolInput,
    GrepToolOutput,
)
from ripperdoc.tools.grep._utils import (
    MAX_GREP_OUTPUT_CHARS,
    _grep_supports_pcre,
    _normalize_glob_for_grep,
    _parse_content_line,
    _parse_count_line,
    _split_globs,
    apply_head_limit,
    truncate_with_ellipsis,
)

__all__ = [
    "GrepMatch",
    "GrepTool",
    "GrepToolInput",
    "GrepToolOutput",
    "MAX_GREP_OUTPUT_CHARS",
    "_grep_supports_pcre",
    "_normalize_glob_for_grep",
    "_parse_content_line",
    "_parse_count_line",
    "_split_globs",
    "apply_head_limit",
    "truncate_with_ellipsis",
]
