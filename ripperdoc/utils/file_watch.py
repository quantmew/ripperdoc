"""File-change detection for notifying the model about user edits.

Cache data structures live in ``fileStateCache.py``.  This module handles
change detection (mtime polling + diff) and file-read listener callbacks.
"""

from __future__ import annotations

import difflib
import itertools
import os
from dataclasses import dataclass
from typing import Callable, List, Optional

from ripperdoc.utils.fileStateCache import FileCacheType, record_snapshot
from ripperdoc.utils.log import get_logger

logger = get_logger()


# ---------------------------------------------------------------------------
# File read listeners
# ---------------------------------------------------------------------------

_file_read_listeners: List[Callable[[str], None]] = []


def register_file_read_listener(callback: Callable[[str], None]) -> None:
    _file_read_listeners.append(callback)


def notify_file_read_listeners(file_path: str) -> None:
    for callback in _file_read_listeners:
        try:
            callback(file_path)
        except Exception as exc:
            logger.debug(
                "[file_watch] File read listener error: %s: %s",
                type(exc).__name__,
                exc,
                extra={"file_path": file_path},
            )


# ---------------------------------------------------------------------------
# Change detection
# ---------------------------------------------------------------------------


@dataclass
class ChangedFileNotice:
    """Information about a file that changed after it was read."""

    file_path: str
    summary: str


def _read_portion(
    file_path: str, offset: int, limit: Optional[int], encoding: str = "utf-8"
) -> str:
    """Read a slice of a file by line without loading the entire file."""
    start = max(offset, 0)
    with open(file_path, "r", encoding=encoding, errors="replace") as handle:
        if limit is None:
            return "".join(itertools.islice(handle, start, None))
        end = start + limit
        return "".join(itertools.islice(handle, start, end))


def _build_diff_summary(old_content: str, new_content: str, file_path: str, max_lines: int) -> str:
    diff = list(
        difflib.unified_diff(
            old_content.splitlines(),
            new_content.splitlines(),
            fromfile=file_path,
            tofile=file_path,
            lineterm="",
        )
    )
    if not diff:
        return "File was modified but contents appear unchanged."
    if len(diff) > max_lines:
        diff = diff[:max_lines] + ["... (diff truncated)"]
    return "\n".join(diff)


def detect_changed_files(
    cache: FileCacheType, *, max_diff_lines: int = 80
) -> List[ChangedFileNotice]:
    """Return notices for files whose mtime increased since they were read."""
    notices: List[ChangedFileNotice] = []

    for file_path, snapshot in list(cache.items()):
        try:
            current_mtime = os.path.getmtime(file_path)
        except OSError:
            notices.append(
                ChangedFileNotice(
                    file_path=file_path, summary="File was deleted or is no longer accessible."
                )
            )
            if hasattr(cache, "pop"):
                cache.pop(file_path, None)
            continue

        if current_mtime <= snapshot.timestamp:
            continue

        try:
            new_content = _read_portion(
                file_path, snapshot.offset or 0, snapshot.limit, "utf-8"
            )
        except (
            OSError,
            IOError,
            UnicodeDecodeError,
            ValueError,
        ) as exc:
            logger.warning(
                "[file_watch] Failed reading changed file: %s: %s",
                type(exc).__name__,
                exc,
                extra={"file_path": file_path},
            )
            notices.append(
                ChangedFileNotice(
                    file_path=file_path,
                    summary=f"File changed but could not be read: {exc}",
                )
            )
            snapshot.timestamp = current_mtime
            cache[file_path] = snapshot
            continue

        diff_summary = _build_diff_summary(
            snapshot.content, new_content, file_path, max_lines=max_diff_lines
        )
        notices.append(ChangedFileNotice(file_path=file_path, summary=diff_summary))
        record_snapshot(
            file_path,
            new_content,
            cache,
            offset=snapshot.offset or 0,
            limit=snapshot.limit,
        )

    return notices
