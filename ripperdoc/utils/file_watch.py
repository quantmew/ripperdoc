"""Lightweight file-change tracking for notifying the model about user edits."""

from __future__ import annotations

import difflib
import os
import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

from ripperdoc.utils.log import get_logger

logger = get_logger()

# Default limits for BoundedFileCache
DEFAULT_MAX_ENTRIES = int(os.getenv("RIPPERDOC_FILE_CACHE_MAX_ENTRIES", "500"))
DEFAULT_MAX_MEMORY_MB = float(os.getenv("RIPPERDOC_FILE_CACHE_MAX_MEMORY_MB", "50"))


@dataclass
class FileSnapshot:
    """Snapshot of a file read by the agent."""

    content: str
    timestamp: float
    offset: int = 0
    limit: Optional[int] = None

    def memory_size(self) -> int:
        """Estimate memory usage of this snapshot in bytes."""
        # String memory: roughly 1 byte per char for ASCII, more for unicode
        # Plus object overhead (~50 bytes for dataclass)
        return sys.getsizeof(self.content) + 50


class BoundedFileCache:
    """Thread-safe LRU cache for FileSnapshots with memory and entry limits.

    This cache prevents unbounded memory growth in long sessions by:
    1. Limiting the maximum number of entries (LRU eviction)
    2. Limiting total memory usage
    3. Providing thread-safe access

    Usage:
        cache = BoundedFileCache(max_entries=500, max_memory_mb=50)
        cache["/path/to/file"] = FileSnapshot(content="...", timestamp=123.0)
        snapshot = cache.get("/path/to/file")
    """

    def __init__(
        self,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        max_memory_mb: float = DEFAULT_MAX_MEMORY_MB,
    ) -> None:
        """Initialize the bounded cache.

        Args:
            max_entries: Maximum number of file snapshots to keep
            max_memory_mb: Maximum total memory usage in megabytes
        """
        self._max_entries = max(1, max_entries)
        self._max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self._cache: OrderedDict[str, FileSnapshot] = OrderedDict()
        self._current_memory = 0
        self._lock = threading.RLock()
        self._eviction_count = 0

    @property
    def max_entries(self) -> int:
        """Maximum number of entries allowed."""
        return self._max_entries

    @property
    def max_memory_bytes(self) -> int:
        """Maximum memory in bytes."""
        return self._max_memory_bytes

    @property
    def current_memory(self) -> int:
        """Current estimated memory usage in bytes."""
        with self._lock:
            return self._current_memory

    @property
    def eviction_count(self) -> int:
        """Number of entries evicted due to limits."""
        with self._lock:
            return self._eviction_count

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._cache

    def __getitem__(self, key: str) -> FileSnapshot:
        with self._lock:
            if key not in self._cache:
                raise KeyError(key)
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return self._cache[key]

    def __setitem__(self, key: str, value: FileSnapshot) -> None:
        with self._lock:
            new_size = value.memory_size()

            # If key exists, remove old entry first (atomic pop to avoid TOCTOU)
            old_value = self._cache.pop(key, None)
            if old_value is not None:
                self._current_memory = max(0, self._current_memory - old_value.memory_size())

            # Evict entries if needed (memory limit)
            while self._current_memory + new_size > self._max_memory_bytes and self._cache:
                self._evict_oldest()

            # Evict entries if needed (entry limit)
            while len(self._cache) >= self._max_entries:
                self._evict_oldest()

            # Add new entry
            self._cache[key] = value
            self._current_memory += new_size

    def __delitem__(self, key: str) -> None:
        with self._lock:
            # Use atomic pop to avoid TOCTOU between check and delete
            old_value = self._cache.pop(key, None)
            if old_value is not None:
                self._current_memory = max(0, self._current_memory - old_value.memory_size())

    def _evict_oldest(self) -> None:
        """Evict the least recently used entry. Must be called with lock held."""
        if self._cache:
            oldest_key, oldest_value = self._cache.popitem(last=False)
            self._current_memory = max(0, self._current_memory - oldest_value.memory_size())
            self._eviction_count += 1
            logger.debug(
                "[file_cache] Evicted entry due to cache limits",
                extra={"evicted_path": oldest_key, "total_evictions": self._eviction_count},
            )

    def get(self, key: str, default: Optional[FileSnapshot] = None) -> Optional[FileSnapshot]:
        """Get a snapshot, returning default if not found."""
        with self._lock:
            if key not in self._cache:
                return default
            self._cache.move_to_end(key)
            return self._cache[key]

    def pop(self, key: str, default: Optional[FileSnapshot] = None) -> Optional[FileSnapshot]:
        """Remove and return a snapshot."""
        with self._lock:
            if key not in self._cache:
                return default
            value = self._cache.pop(key)
            self._current_memory = max(0, self._current_memory - value.memory_size())
            return value

    def setdefault(self, key: str, default: FileSnapshot) -> FileSnapshot:
        """Atomically get or set a snapshot.

        If key exists, return its value (and mark as recently used).
        If key doesn't exist, set it to default and return default.
        This provides a thread-safe get-or-create operation.
        """
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            # Key doesn't exist - add it
            new_size = default.memory_size()
            # Evict if needed
            while self._current_memory + new_size > self._max_memory_bytes and self._cache:
                self._evict_oldest()
            while len(self._cache) >= self._max_entries:
                self._evict_oldest()
            self._cache[key] = default
            self._current_memory += new_size
            return default

    def clear(self) -> None:
        """Remove all entries from the cache."""
        with self._lock:
            self._cache.clear()
            self._current_memory = 0

    def keys(self) -> List[str]:
        """Return list of cached file paths."""
        with self._lock:
            return list(self._cache.keys())

    def values(self) -> List[FileSnapshot]:
        """Return list of cached snapshots."""
        with self._lock:
            return list(self._cache.values())

    def items(self) -> List[Tuple[str, FileSnapshot]]:
        """Return list of (path, snapshot) pairs."""
        with self._lock:
            return list(self._cache.items())

    def __iter__(self) -> Iterator[str]:
        """Iterate over keys (not thread-safe for modifications during iteration)."""
        with self._lock:
            return iter(list(self._cache.keys()))

    def stats(self) -> Dict[str, int]:
        """Return cache statistics."""
        with self._lock:
            return {
                "entries": len(self._cache),
                "max_entries": self._max_entries,
                "memory_bytes": self._current_memory,
                "max_memory_bytes": self._max_memory_bytes,
                "eviction_count": self._eviction_count,
            }


@dataclass
class ChangedFileNotice:
    """Information about a file that changed after it was read."""

    file_path: str
    summary: str


# Type alias for cache - supports both Dict and BoundedFileCache
FileCacheType = Dict[str, FileSnapshot] | BoundedFileCache


def record_snapshot(
    file_path: str,
    content: str,
    cache: FileCacheType,
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
    cache: FileCacheType, *, max_diff_lines: int = 80
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
