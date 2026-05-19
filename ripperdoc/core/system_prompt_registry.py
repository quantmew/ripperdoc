"""System prompt section registry.

Provides memoization for system prompt sections. Sections that don't set
cache_break are computed once and cached until clear_system_prompt_sections()
is called.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Dict, List, Optional, Union

from ripperdoc.utils.log import get_logger

logger = get_logger()

# Boundary marker separating static (cross-org cacheable) content from dynamic content.
# Everything BEFORE this marker can use provider-level cache scope.
# Everything AFTER contains user/session-specific content and should not use
# global cache scope.
SYSTEM_PROMPT_DYNAMIC_BOUNDARY = "__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__"

# Type for a section compute function (sync or async)
ComputeFn = Callable[[], Union[Optional[str], Awaitable[Optional[str]]]]


class SystemPromptSection:
    """A named section of the system prompt with caching control."""

    def __init__(
        self,
        name: str,
        compute: ComputeFn,
        cache_break: bool = False,
    ) -> None:
        self.name = name
        self.compute = compute
        self.cache_break = cache_break

    def __repr__(self) -> str:
        return (
            f"SystemPromptSection(name={self.name!r}, "
            f"cache_break={self.cache_break})"
        )


def system_prompt_section(
    name: str,
    compute: ComputeFn,
) -> SystemPromptSection:
    """Create a cached system prompt section.

    Computed once, cached until /clear or /compact.
    """
    return SystemPromptSection(name=name, compute=compute, cache_break=False)


def DANGEROUS_uncached_system_prompt_section(
    name: str,
    compute: ComputeFn,
    _reason: str,
) -> SystemPromptSection:
    """Create a volatile system prompt section that recomputes every turn.

    This WILL break the prompt cache when the value changes.
    The reason parameter documents why cache-breaking is necessary.
    """
    return SystemPromptSection(name=name, compute=compute, cache_break=True)


# In-memory cache for system prompt section results
_cache: Dict[str, Optional[str]] = {}


def resolve_system_prompt_sections(
    sections: List[SystemPromptSection],
) -> List[Optional[str]]:
    """Resolve all system prompt sections, returning prompt strings.

    Non-cache-breaking sections are cached; cache-breaking sections are
    recomputed each time.
    """
    results: List[Optional[str]] = []

    for section in sections:
        # Check cache for non-cache-breaking sections
        if not section.cache_break and section.name in _cache:
            results.append(_cache[section.name])
            continue

        # Compute the section value
        try:
            value = section.compute()
            # Handle async compute functions
            if asyncio.iscoroutine(value):
                # This won't work in a sync context; caller must handle
                # This is a simplified sync-only version
                logger.warning(
                    "[system_prompt_registry] Async compute detected in sync path: %s",
                    section.name,
                )
                results.append(None)
                continue

            assert value is None or isinstance(value, str)
            # Cache if not cache-breaking
            if not section.cache_break:
                _cache[section.name] = value

            results.append(value)
        except (RuntimeError, ValueError, OSError) as exc:
            logger.warning(
                "[system_prompt_registry] Failed to compute section %s: %s: %s",
                section.name,
                type(exc).__name__,
                exc,
            )
            results.append(None)

    return results


async def resolve_system_prompt_sections_async(
    sections: List[SystemPromptSection],
) -> List[Optional[str]]:
    """Async version of resolve_system_prompt_sections.

    Supports both sync and async compute functions.
    """
    results: List[Optional[str]] = []

    for section in sections:
        if not section.cache_break and section.name in _cache:
            results.append(_cache[section.name])
            continue

        try:
            value = section.compute()
            if asyncio.iscoroutine(value):
                value = await value

            assert value is None or isinstance(value, str)
            if not section.cache_break:
                _cache[section.name] = value

            results.append(value)
        except (RuntimeError, ValueError, OSError) as exc:
            logger.warning(
                "[system_prompt_registry] Failed to compute section %s: %s: %s",
                section.name,
                type(exc).__name__,
                exc,
            )
            results.append(None)

    return results


def clear_system_prompt_sections() -> None:
    """Clear all cached system prompt sections.

    Called on /clear and /compact to force recomputation on next request.
    """
    _cache.clear()


def get_system_prompt_section_cache() -> Dict[str, Optional[str]]:
    """Return the current cache contents (for inspection/debug)."""
    return dict(_cache)
