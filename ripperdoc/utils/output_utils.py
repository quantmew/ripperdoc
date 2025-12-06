"""Utilities for processing and truncating command output."""

import re
from typing import Any


# Maximum output length to prevent token overflow
MAX_OUTPUT_CHARS = 30000

# Threshold for considering output "large"
LARGE_OUTPUT_THRESHOLD = 5000

# When truncating, keep this many chars from start and end
TRUNCATE_KEEP_START = 15000
TRUNCATE_KEEP_END = 10000


def trim_blank_lines(text: str) -> str:
    """Remove leading and trailing blank lines while preserving internal spacing.

    Args:
        text: Input text

    Returns:
        Text with leading/trailing blank lines removed
    """
    lines = text.split("\n")

    # Remove leading blank lines
    start = 0
    while start < len(lines) and not lines[start].strip():
        start += 1

    # Remove trailing blank lines
    end = len(lines)
    while end > start and not lines[end - 1].strip():
        end -= 1

    return "\n".join(lines[start:end])


def is_image_data(text: str) -> bool:
    """Check if text appears to be base64 encoded image data.

    Args:
        text: Text to check

    Returns:
        True if text looks like image data
    """
    if not text:
        return False

    stripped = text.strip()

    # Check for data URI scheme (most reliable indicator)
    if stripped.startswith("data:image/"):
        return True

    # Don't treat arbitrary long text as base64 unless it has image indicators
    # Base64 images are typically very long AND have specific characteristics
    if len(stripped) < 1000:
        return False

    # Check for common image base64 patterns
    # Real base64 images usually have variety of characters and padding
    base64_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=")
    text_chars = set(stripped)

    # If text only uses a small subset of base64 chars, it's probably not base64
    # Real base64 uses a variety of characters
    if len(text_chars) < 10:
        return False

    # Must be valid base64 characters
    if not text_chars.issubset(base64_chars):
        return False

    # Must end with proper base64 padding or no padding
    if not (
        stripped.endswith("==")
        or stripped.endswith("=")
        or stripped[-1] in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    ):
        return False

    # If all checks pass and it's very long, might be base64 image
    return len(stripped) > 10000


def truncate_output(text: str, max_chars: int = MAX_OUTPUT_CHARS) -> dict[str, Any]:
    """Truncate output if it exceeds max length.

    Keeps both the beginning and end of output to preserve context.

    Args:
        text: Output text to truncate
        max_chars: Maximum character limit

    Returns:
        Dict with:
            - truncated_content: Potentially truncated text
            - is_truncated: Whether truncation occurred
            - original_length: Original text length
            - is_image: Whether content appears to be image data
    """
    if not text:
        return {
            "truncated_content": text,
            "is_truncated": False,
            "original_length": 0,
            "is_image": False,
        }

    # Check if it's image data
    if is_image_data(text):
        return {
            "truncated_content": text,
            "is_truncated": False,
            "original_length": len(text),
            "is_image": True,
        }

    original_length = len(text)

    if original_length <= max_chars:
        return {
            "truncated_content": text,
            "is_truncated": False,
            "original_length": original_length,
            "is_image": False,
        }

    marker_template = "\n\n... [Output truncated: {omitted} characters omitted] ...\n\n"
    short_marker = "... [truncated] ..."

    def _choose_marker(omitted: int, budget: int) -> str:
        """Pick the most informative marker that fits within the budget."""
        full_marker = marker_template.format(omitted=omitted)
        if len(full_marker) <= budget:
            return full_marker
        if len(short_marker) <= budget:
            return short_marker
        # Last resort: squeeze an ellipsis into the budget (may be empty for tiny budgets)
        return "..."[: max(budget, 0)]

    # Iteratively balance how much of the start/end to keep while ensuring we never exceed max_chars.
    marker = _choose_marker(original_length - max_chars, max_chars)
    keep_start = keep_end = 0
    for _ in range(2):
        available = max(0, max_chars - len(marker))
        keep_start = min(TRUNCATE_KEEP_START, available // 2)
        keep_end = min(TRUNCATE_KEEP_END, available - keep_start)
        marker = _choose_marker(max(0, original_length - (keep_start + keep_end)), max_chars)

    available = max(0, max_chars - len(marker))
    # Ensure kept sections fit the final budget; trim end first, then start if needed.
    if keep_start + keep_end > available:
        overflow = keep_start + keep_end - available
        trim_end = min(overflow, keep_end)
        keep_end -= trim_end
        overflow -= trim_end
        keep_start = max(0, keep_start - overflow)

    truncated = text[:keep_start] + marker + (text[-keep_end:] if keep_end else "")
    if len(truncated) > max_chars:
        truncated = truncated[:max_chars]

    return {
        "truncated_content": truncated,
        "is_truncated": True,
        "original_length": original_length,
        "is_image": False,
    }


def format_duration(duration_ms: float) -> str:
    """Format duration in milliseconds to human-readable string.

    Args:
        duration_ms: Duration in milliseconds

    Returns:
        Formatted duration string (e.g., "1.23s", "45.6ms")
    """
    if duration_ms < 1000:
        return f"{duration_ms:.0f}ms"
    else:
        return f"{duration_ms / 1000:.2f}s"


def is_output_large(text: str) -> bool:
    """Check if output is considered large.

    Args:
        text: Output text

    Returns:
        True if output exceeds large threshold
    """
    return len(text) > LARGE_OUTPUT_THRESHOLD


def count_lines(text: str) -> int:
    """Count number of lines in text.

    Args:
        text: Text to count

    Returns:
        Number of lines
    """
    if not text:
        return 0
    return text.count("\n") + 1


def get_last_n_lines(text: str, n: int) -> str:
    """Get the last N lines from text.

    Args:
        text: Input text
        n: Number of lines to keep

    Returns:
        Last N lines
    """
    if not text:
        return text

    lines = text.split("\n")
    if len(lines) <= n:
        return text

    return "\n".join(lines[-n:])


def sanitize_output(text: str) -> str:
    """Sanitize output by removing control/escape sequences and ensuring UTF-8."""
    # ANSI/VT escape patterns, including charset selection (e.g., ESC(B) and OSC)
    ansi_escape = re.compile(
        r"""
        \x1B
        (?:
            [@-Z\\-_]                      # 7-bit C1 control
          | \[ [0-?]* [ -/]* [@-~]         # CSI (colors, cursor moves, etc.)
          | [()][0-9A-Za-z]                # Charset selection like ESC(B
          | \] (?: [^\x07\x1B]* \x07 | [^\x1B]* \x1B\\ )  # OSC to BEL or ST
        )
        """,
        re.VERBOSE,
    )
    text = ansi_escape.sub("", text)

    # Remove remaining control characters except newline, tab, carriage return
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)

    return text
