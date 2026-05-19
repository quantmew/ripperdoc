"""File state cache — tracks file contents read by the agent.

Tracks file contents read by the agent with size-based eviction.

* ``FileState`` — snapshot of a file at a point in time
* ``FileStateCache`` — LRU cache with size-based eviction and path normalisation
* Helper functions for clone / merge / serialisation
"""

from __future__ import annotations

import os
import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from ripperdoc.utils.log import get_logger

logger = get_logger()

# ---------------------------------------------------------------------------
# FileState — snapshot of a file at a point in time
# ---------------------------------------------------------------------------


@dataclass
class FileState:
    """Snapshot of a file read by the agent.

    Fields:

    - ``content``      – full text content of the file (or the slice read)
    - ``timestamp``    – ``os.path.getmtime()`` at read time
    - ``offset``       – line offset passed to Read (``None`` = from start)
    - ``limit``        – line limit passed to Read (``None`` = to end)
    - ``is_partial_view`` – ``True`` when injected content was truncated or
      processed (e.g. CLAUDE.md with stripped comments).  When ``True``,
      Edit/Write must require an explicit full Read first.
    """

    content: str
    timestamp: float
    offset: Optional[int] = None
    limit: Optional[int] = None
    is_partial_view: bool = False

    def memory_size(self) -> int:
        """Estimate memory usage in bytes (for cache eviction)."""
        return sys.getsizeof(self.content) + 50


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

READ_FILE_STATE_CACHE_SIZE = int(
    os.getenv("RIPPERDOC_FILE_STATE_CACHE_SIZE", "100")
)
DEFAULT_MAX_CACHE_SIZE_BYTES = int(
    os.getenv("RIPPERDOC_FILE_STATE_CACHE_BYTES", str(25 * 1024 * 1024))
)


# ---------------------------------------------------------------------------
# FileStateCache — LRU cache for FileState entries
# ---------------------------------------------------------------------------


def _normalize_path(key: str) -> str:
    """Normalise a path key so ``foo/../bar`` and ``bar`` match."""
    try:
        return os.path.normpath(os.path.abspath(key))
    except (TypeError, ValueError):
        return key


class FileStateCache:
    """LRU cache for ``FileState`` entries with size-based eviction.

    Features:

    - Path normalisation on every ``get``/``set``/``has``/``delete``
    - Size-based eviction (bytes of ``content``)
    - Entry-count limit
    - ``dump``/``load`` for serialisation
    - ``clone``/``merge`` for fork-agent cache sharing
    """

    def __init__(
        self,
        max_entries: int = READ_FILE_STATE_CACHE_SIZE,
        max_size_bytes: int = DEFAULT_MAX_CACHE_SIZE_BYTES,
    ) -> None:
        self._max_entries = max(1, max_entries)
        self._max_size_bytes = max_size_bytes
        self._cache: OrderedDict[str, FileState] = OrderedDict()
        self._current_size = 0
        self._lock = threading.RLock()
        self._eviction_count = 0

    # -- properties --

    @property
    def max(self) -> int:
        return self._max_entries

    @property
    def max_size(self) -> int:
        return self._max_size_bytes

    @property
    def calculated_size(self) -> int:
        with self._lock:
            return self._current_size

    @property
    def eviction_count(self) -> int:
        with self._lock:
            return self._eviction_count

    # -- dict-like interface --

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return _normalize_path(key) in self._cache

    def __getitem__(self, key: str) -> FileState:
        with self._lock:
            norm = _normalize_path(key)
            if norm not in self._cache:
                raise KeyError(key)
            self._cache.move_to_end(norm)
            return self._cache[norm]

    def __setitem__(self, key: str, value: FileState) -> None:
        with self._lock:
            norm = _normalize_path(key)
            new_size = value.memory_size()

            old = self._cache.pop(norm, None)
            if old is not None:
                self._current_size = max(0, self._current_size - old.memory_size())

            while self._current_size + new_size > self._max_size_bytes and self._cache:
                self._evict_oldest()
            while len(self._cache) >= self._max_entries:
                self._evict_oldest()

            self._cache[norm] = value
            self._current_size += new_size

    def __delitem__(self, key: str) -> None:
        with self._lock:
            norm = _normalize_path(key)
            old = self._cache.pop(norm, None)
            if old is not None:
                self._current_size = max(0, self._current_size - old.memory_size())

    # -- public helpers --

    def get(self, key: str, default: Optional[FileState] = None) -> Optional[FileState]:
        with self._lock:
            norm = _normalize_path(key)
            if norm not in self._cache:
                return default
            self._cache.move_to_end(norm)
            return self._cache[norm]

    def has(self, key: str) -> bool:
        with self._lock:
            return _normalize_path(key) in self._cache

    def delete(self, key: str) -> bool:
        with self._lock:
            norm = _normalize_path(key)
            old = self._cache.pop(norm, None)
            if old is not None:
                self._current_size = max(0, self._current_size - old.memory_size())
                return True
            return False

    def pop(self, key: str, default: Optional[FileState] = None) -> Optional[FileState]:
        with self._lock:
            norm = _normalize_path(key)
            if norm not in self._cache:
                return default
            value = self._cache.pop(norm)
            self._current_size = max(0, self._current_size - value.memory_size())
            return value

    def setdefault(self, key: str, default: FileState) -> FileState:
        with self._lock:
            norm = _normalize_path(key)
            if norm in self._cache:
                self._cache.move_to_end(norm)
                return self._cache[norm]
            new_size = default.memory_size()
            while self._current_size + new_size > self._max_size_bytes and self._cache:
                self._evict_oldest()
            while len(self._cache) >= self._max_entries:
                self._evict_oldest()
            self._cache[norm] = default
            self._current_size += new_size
            return default

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._current_size = 0

    # -- iteration --

    def keys(self) -> List[str]:
        with self._lock:
            return list(self._cache.keys())

    def values(self) -> List[FileState]:
        with self._lock:
            return list(self._cache.values())

    def items(self) -> List[Tuple[str, FileState]]:
        with self._lock:
            return list(self._cache.items())

    def entries(self) -> List[Tuple[str, FileState]]:
        """Alias for ``items``."""
        return self.items()

    def __iter__(self) -> Iterator[str]:
        with self._lock:
            return iter(list(self._cache.keys()))

    # -- serialisation --

    def dump(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Serialise cache contents for persistence or fork inheritance."""
        with self._lock:
            result: List[Tuple[str, Dict[str, Any]]] = []
            for path, state in self._cache.items():
                result.append((
                    path,
                    {
                        "content": state.content,
                        "timestamp": state.timestamp,
                        "offset": state.offset,
                        "limit": state.limit,
                        "is_partial_view": state.is_partial_view,
                    },
                ))
            return result

    def load(self, entries: List[Tuple[str, Dict[str, Any]]]) -> None:
        """Deserialise into this cache (appends, does not clear first)."""
        for path, data in entries:
            state = FileState(
                content=data["content"],
                timestamp=data["timestamp"],
                offset=data.get("offset"),
                limit=data.get("limit"),
                is_partial_view=data.get("is_partial_view", False),
            )
            self[path] = state

    # -- stats --

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._cache),
                "max_entries": self._max_entries,
                "size_bytes": self._current_size,
                "max_size_bytes": self._max_size_bytes,
                "eviction_count": self._eviction_count,
            }

    # -- internal --

    def _evict_oldest(self) -> None:
        if self._cache:
            oldest_key, oldest_value = self._cache.popitem(last=False)
            self._current_size = max(0, self._current_size - oldest_value.memory_size())
            self._eviction_count += 1
            logger.debug(
                "[fileStateCache] Evicted entry",
                extra={"evicted_path": oldest_key, "total_evictions": self._eviction_count},
            )


# ---------------------------------------------------------------------------
# Factory & helpers
# ---------------------------------------------------------------------------


def create_file_state_cache_with_size_limit(
    max_entries: int = READ_FILE_STATE_CACHE_SIZE,
    max_size_bytes: int = DEFAULT_MAX_CACHE_SIZE_BYTES,
) -> FileStateCache:
    """Factory — mirrors ``createFileStateCacheWithSizeLimit``."""
    return FileStateCache(max_entries=max_entries, max_size_bytes=max_size_bytes)


def cache_to_object(cache: FileStateCache) -> Dict[str, FileState]:
    """Convert cache to plain dict — mirrors ``cacheToObject``."""
    return dict(cache.entries())


def cache_keys(cache: FileStateCache) -> List[str]:
    """Get all cached paths — mirrors ``cacheKeys``."""
    return cache.keys()


def clone_file_state_cache(cache: FileStateCache) -> FileStateCache:
    """Deep-clone a cache — mirrors ``cloneFileStateCache``."""
    cloned = create_file_state_cache_with_size_limit(cache.max, cache.max_size)
    cloned.load(cache.dump())
    return cloned


def merge_file_state_caches(
    first: FileStateCache,
    second: FileStateCache,
) -> FileStateCache:
    """Merge two caches, more recent entries win — mirrors ``mergeFileStateCaches``."""
    merged = clone_file_state_cache(first)
    for path, state in second.entries():
        existing = merged.get(path)
        if existing is None or state.timestamp > existing.timestamp:
            merged[path] = state
    return merged


# ---------------------------------------------------------------------------
# Backward-compatible type alias
# ---------------------------------------------------------------------------

FileCacheType = Union[Dict[str, FileState], FileStateCache]


# ---------------------------------------------------------------------------
# record_snapshot — convenience wrapper used by file_read / file_edit / file_write
# ---------------------------------------------------------------------------


def record_snapshot(
    file_path: str,
    content: str,
    cache: FileCacheType,
    *,
    offset: int = 0,
    limit: Optional[int] = None,
    is_partial_view: bool = False,
) -> None:
    """Store a ``FileState`` for *file_path* into *cache*."""
    try:
        timestamp = os.path.getmtime(file_path)
    except OSError:
        timestamp = 0.0

    cache[file_path] = FileState(
        content=content,
        timestamp=timestamp,
        offset=offset,
        limit=limit,
        is_partial_view=is_partial_view,
    )
