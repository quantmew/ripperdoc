"""File reading encoding detection utilities."""

from __future__ import annotations

import itertools
import os
from typing import List, Optional, Tuple

from charset_normalizer import from_bytes

from ripperdoc.utils.log import get_logger

logger = get_logger()


def detect_file_encoding(file_path: str, max_bytes: Optional[int] = None) -> Tuple[Optional[str], float]:
    """Detect file encoding using charset-normalizer.

    Returns:
        Tuple of (encoding, confidence). encoding is None if detection failed.
    """
    try:
        with open(file_path, "rb") as f:
            raw_data = f.read() if max_bytes is None else f.read(max_bytes)
        results = from_bytes(raw_data)

        if not results:
            return None, 0.0

        best = results.best()
        if not best:
            return None, 0.0

        # For Chinese content, prefer GB encodings over Big5/others
        if best.language == "Chinese":
            gb_encodings = {"gb18030", "gbk", "gb2312"}
            for result in results:
                if result.encoding.lower() in gb_encodings:
                    return result.encoding, 0.9

        return best.encoding, 0.9
    except (OSError, IOError) as e:
        logger.warning("Failed to detect encoding for %s: %s", file_path, e)
        return None, 0.0


def read_file_with_encoding(file_path: str) -> Tuple[Optional[List[str]], str, Optional[str]]:
    """Read file with proper encoding detection.

    Returns:
        Tuple of (lines, encoding_used, error_message).
        If successful: (lines, encoding, None)
        If failed: (None, "", error_message)
    """
    # First, try UTF-8 (most common)
    try:
        with open(file_path, "r", encoding="utf-8", errors="strict") as f:
            lines = f.readlines()
        return lines, "utf-8", None
    except UnicodeDecodeError:
        pass

    # UTF-8 failed, use charset-normalizer to detect encoding
    detected_encoding, confidence = detect_file_encoding(file_path)

    if detected_encoding:
        try:
            with open(file_path, "r", encoding=detected_encoding, errors="strict") as f:
                lines = f.readlines()
            logger.info(
                "File %s decoded using detected encoding %s",
                file_path,
                detected_encoding,
            )
            return lines, detected_encoding, None
        except (UnicodeDecodeError, LookupError) as e:
            logger.warning(
                "Failed to read %s with detected encoding %s: %s",
                file_path,
                detected_encoding,
                e,
            )

    # Detection failed - try latin-1 as last resort
    try:
        with open(file_path, "r", encoding="latin-1", errors="strict") as f:
            lines = f.readlines()
        logger.warning(
            "File %s: encoding detection failed, using latin-1 fallback",
            file_path,
        )
        return lines, "latin-1", None
    except (UnicodeDecodeError, LookupError):
        pass

    # All attempts failed - return error
    error_msg = (
        f"Unable to determine file encoding. "
        f"Detected: {detected_encoding or 'unknown'} (confidence: {confidence * 100:.0f}%). "
        f"Tried fallback encodings: utf-8, latin-1. "
        f"Please convert the file to UTF-8."
    )
    return None, "", error_msg


def read_file_slice_with_encoding(
    file_path: str, offset: int, limit: Optional[int], sample_bytes: int = 65536
) -> Tuple[Optional[List[str]], str, Optional[str]]:
    """Read a slice of a file with encoding detection.

    Returns:
        Tuple of (lines, encoding_used, error_message).
        If successful: (lines, encoding, None)
        If failed: (None, "", error_message)
    """

    def _read_slice(encoding: str) -> List[str]:
        start = max(offset, 0)
        if limit is None:
            end = None
        elif limit <= 0:
            return []
        else:
            end = start + limit
        with open(file_path, "r", encoding=encoding, errors="strict") as f:
            return list(itertools.islice(f, start, end))

    # First, try UTF-8 (most common)
    try:
        lines = _read_slice("utf-8")
        return lines, "utf-8", None
    except UnicodeDecodeError:
        pass

    # UTF-8 failed, use charset-normalizer to detect encoding (sampled)
    detected_encoding, confidence = detect_file_encoding(file_path, max_bytes=sample_bytes)

    if detected_encoding:
        try:
            lines = _read_slice(detected_encoding)
            logger.info(
                "File %s decoded using detected encoding %s",
                file_path,
                detected_encoding,
            )
            return lines, detected_encoding, None
        except (UnicodeDecodeError, LookupError) as e:
            logger.warning(
                "Failed to read %s with detected encoding %s: %s",
                file_path,
                detected_encoding,
                e,
            )

    # Detection failed - try latin-1 as last resort
    try:
        lines = _read_slice("latin-1")
        logger.warning(
            "File %s: encoding detection failed, using latin-1 fallback",
            file_path,
        )
        return lines, "latin-1", None
    except (UnicodeDecodeError, LookupError):
        pass

    # All attempts failed - return error
    error_msg = (
        f"Unable to determine file encoding. "
        f"Detected: {detected_encoding or 'unknown'} (confidence: {confidence * 100:.0f}%). "
        f"Tried fallback encodings: utf-8, latin-1. "
        f"Please convert the file to UTF-8."
    )
    return None, "", error_msg
