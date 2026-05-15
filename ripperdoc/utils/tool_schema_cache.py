"""Tool schema cache for Ripperdoc.

Provides session-level caching for tool schemas to avoid recomputing
tool descriptions and JSON schemas on every request.

Cache key: (tool_name, tool.input_schema.model_json_schema() as JSON str)
- When inputJSONSchema is present (MCP tools), it's included in the key
- Cached schemas are invalidated when tool inventory changes
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

from ripperdoc.utils.log import get_logger

logger = get_logger()

# Session-level cache: key -> cached schema dict
_cache: Dict[str, Dict[str, Any]] = {}

# Version counter incremented when tool inventory changes
_cache_version: int = 0


def _make_cache_key(tool_name: str, input_schema_json: str) -> str:
    """Build a Blake2b hash key for a tool schema."""
    raw = f"{tool_name}:{input_schema_json}"
    return hashlib.blake2b(raw.encode("utf-8"), digest_size=16).hexdigest()


def get_cached_schema(
    tool_name: str,
    input_schema_json: str,
) -> Optional[Dict[str, Any]]:
    """Return cached schema if available and current."""
    key = _make_cache_key(tool_name, input_schema_json)
    entry = _cache.get(key)
    if entry is None:
        return None
    # Check version validity
    if entry.get("_version", -1) != _cache_version:
        _cache.pop(key, None)
        return None
    return entry.get("schema")


def set_cached_schema(
    tool_name: str,
    input_schema_json: str,
    schema: Dict[str, Any],
) -> None:
    """Store a schema in the cache."""
    key = _make_cache_key(tool_name, input_schema_json)
    _cache[key] = {
        "schema": schema,
        "_version": _cache_version,
    }


def invalidate_cache() -> None:
    """Invalidate all cached schemas. Called when tool inventory changes."""
    global _cache_version
    _cache_version += 1
    # Clean stale entries
    stale_keys = [
        k for k, v in _cache.items()
        if v.get("_version", -1) != _cache_version
    ]
    for key in stale_keys:
        _cache.pop(key, None)


def clear_cache() -> None:
    """Clear all cached schemas entirely."""
    _cache.clear()


def get_cache_stats() -> Dict[str, Any]:
    """Return cache statistics for monitoring."""
    return {
        "entries": len(_cache),
        "version": _cache_version,
    }
