"""Utility functions for file edit tool."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# Unicode curly/smart quote to ASCII mapping
_QUOTE_MAP = str.maketrans({
    "\u2018": "'",  # left single
    "\u2019": "'",  # right single
    "\u201c": '"',  # left double
    "\u201d": '"',  # right double
    "\u201a": "'",  # single low-9
    "\u201e": '"',  # double low-9
    "\u2032": "'",  # prime
    "\u2033": '"',  # double prime
})


def _normalize_quotes(text: str) -> str:
    """Normalize Unicode curly/smart quotes to ASCII equivalents."""
    return text.translate(_QUOTE_MAP)


_MAX_FILE_SIZE_BYTES = 1_000_000_000  # 1GB


def determine_edit_encoding(file_path: str, new_content: str) -> str:
    """Determine encoding for editing a file."""
    from ripperdoc.tools.file_read import detect_file_encoding

    detected_encoding, _ = detect_file_encoding(file_path)
    encoding = detected_encoding or "utf-8"

    try:
        new_content.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        encoding = "utf-8"

    return encoding


def validate_file_size(file_path: str) -> Optional[str]:
    """Validate that the file is within size limits.

    Returns an error message if too large, None otherwise.
    """
    try:
        file_size = os.path.getsize(file_path)
    except OSError:
        return None

    if file_size > _MAX_FILE_SIZE_BYTES:
        size_gb = file_size / (1024 ** 3)
        return f"File too large to edit: {size_gb:.1f}GB exceeds limit of 1GB"

    return None
