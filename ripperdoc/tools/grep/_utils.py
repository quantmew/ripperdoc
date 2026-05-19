"""Utility functions for grep tool."""

from __future__ import annotations

import re
import shutil
import subprocess
from typing import List, Optional, Tuple


MAX_GREP_OUTPUT_CHARS = 20000


def truncate_with_ellipsis(
    text: str, max_chars: int = MAX_GREP_OUTPUT_CHARS
) -> Tuple[str, bool, int]:
    """Trim long output and note how many lines were removed."""
    if len(text) <= max_chars:
        return text, False, 0

    remaining = text[max_chars:]
    truncated_lines = remaining.count("\n") + (1 if remaining else 0)
    truncated_text = f"{text[:max_chars]}\n\n... [{truncated_lines} lines truncated] ..."
    return truncated_text, True, truncated_lines


def apply_head_limit(lines: List[str], head_limit: Optional[int]) -> Tuple[List[str], int]:
    """Limit the number of lines returned, recording how many were omitted."""
    if head_limit is None or head_limit <= 0:
        return lines, 0
    if len(lines) <= head_limit:
        return lines, 0
    return lines[:head_limit], len(lines) - head_limit


def _split_globs(glob_value: str) -> List[str]:
    """Split a glob string by whitespace and commas."""
    if not glob_value:
        return []
    globs: List[str] = []
    for token in re.split(r"\s+", glob_value.strip()):
        if not token:
            continue
        globs.extend([part for part in token.split(",") if part])
    return globs


def _normalize_glob_for_grep(glob_pattern: str) -> str:
    """grep --include matches basenames; drop path components."""
    return glob_pattern.split("/")[-1] or glob_pattern


_GREP_SUPPORTS_PCRE: Optional[bool] = None


def _grep_supports_pcre() -> bool:
    """Detect if the system grep supports -P (Perl regex), caching the result."""
    global _GREP_SUPPORTS_PCRE
    if _GREP_SUPPORTS_PCRE is not None:
        return _GREP_SUPPORTS_PCRE

    if shutil.which("grep") is None:
        _GREP_SUPPORTS_PCRE = False
        return _GREP_SUPPORTS_PCRE

    try:
        proc = subprocess.run(
            ["grep", "-P", ""],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
            timeout=15,
        )
        _GREP_SUPPORTS_PCRE = proc.returncode in (0, 1)
    except (OSError, ValueError, subprocess.SubprocessError, subprocess.TimeoutExpired):
        _GREP_SUPPORTS_PCRE = False

    return _GREP_SUPPORTS_PCRE


def _parse_count_line(
    line: str, default_file: str = ""
) -> Optional[Tuple[str, int]]:
    """Parse a count output line (file:count or just count for single file)."""
    match = re.match(r"^(?P<file>.*?):(?P<count>\d+)$", line)
    if match:
        file = match.group("file") or default_file
        return file, int(match.group("count"))
    if line.strip().isdigit():
        return default_file, int(line.strip())
    return None


def _parse_content_line(
    line: str, default_file: str = ""
) -> Optional[Tuple[str, int, str]]:
    """Parse a content output line (file:line:content or just line:content)."""
    match = re.match(r"^(?:(?P<file>.*?):)?(?P<line>\d+):(?P<content>.*)$", line)
    if match:
        file = match.group("file") or default_file
        return file, int(match.group("line")), match.group("content")
    return None


_TYPE_GLOB_MAP = {
    "py": "*.py", "js": "*.js", "ts": "*.ts", "tsx": "*.tsx",
    "jsx": "*.jsx", "rust": "*.rs", "go": "*.go", "java": "*.java",
    "c": "*.c", "cpp": "*.cpp *.cc *.cxx *.hpp",
    "rb": "*.rb", "rs": "*.rs", "swift": "*.swift",
    "html": "*.html *.htm", "css": "*.css",
    "sh": "*.sh *.bash", "sql": "*.sql",
}
