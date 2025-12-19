"""Lightweight file-change tracking for notifying the model about user edits."""

from __future__ import annotations

import difflib
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from ripperdoc.utils.log import get_logger

logger = get_logger()


@dataclass
class FileSnapshot:
    """Snapshot of a file read by the agent."""

    content: str
    timestamp: float
    offset: int = 0
    limit: Optional[int] = None


@dataclass
class ChangedFileNotice:
    """Information about a file that changed after it was read."""

    file_path: str
    summary: str


def record_snapshot(
    file_path: str,
    content: str,
    cache: Dict[str, FileSnapshot],
    *,
    offset: int = 0,
    limit: Optional[int] = None,
) -> None:
    """Store the current contents and mtime for a file."""
    try:
        timestamp = os.path.getmtime(file_path)
    except OSError:
        timestamp = 0.0
    cache[file_path] = FileSnapshot(
        content=content, timestamp=timestamp, offset=offset, limit=limit
    )


def _read_portion(file_path: str, offset: int, limit: Optional[int]) -> str:
    with open(file_path, "r", encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()
    start = max(offset, 0)
    if limit is None:
        selected = lines[start:]
    else:
        selected = lines[start : start + limit]
    return "".join(selected)


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

    # Keep the diff short to avoid flooding the model.
    if len(diff) > max_lines:
        diff = diff[:max_lines] + ["... (diff truncated)"]
    return "\n".join(diff)


def detect_changed_files(
    cache: Dict[str, FileSnapshot], *, max_diff_lines: int = 80
) -> List[ChangedFileNotice]:
    """Return notices for files whose mtime increased since they were read."""
    notices: List[ChangedFileNotice] = []

    # Iterate over a static list so we can mutate cache safely.
    for file_path, snapshot in list(cache.items()):
        try:
            current_mtime = os.path.getmtime(file_path)
        except OSError:
            notices.append(
                ChangedFileNotice(
                    file_path=file_path, summary="File was deleted or is no longer accessible."
                )
            )
            cache.pop(file_path, None)
            continue

        if current_mtime <= snapshot.timestamp:
            continue

        try:
            new_content = _read_portion(file_path, snapshot.offset, snapshot.limit)
        except (
            OSError,
            IOError,
            UnicodeDecodeError,
            ValueError,
        ) as exc:  # pragma: no cover - best-effort telemetry
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
            # Avoid spamming repeated errors by updating timestamp.
            snapshot.timestamp = current_mtime
            cache[file_path] = snapshot
            continue

        diff_summary = _build_diff_summary(
            snapshot.content, new_content, file_path, max_lines=max_diff_lines
        )
        notices.append(ChangedFileNotice(file_path=file_path, summary=diff_summary))
        # Update snapshot so we only notify on subsequent changes.
        record_snapshot(
            file_path,
            new_content,
            cache,
            offset=snapshot.offset,
            limit=snapshot.limit,
        )

    return notices
