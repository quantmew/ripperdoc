"""Search backend abstraction for WebSearch tool."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import List, Optional

from pydantic import BaseModel


class SearchResult(BaseModel):
    """A single search result."""

    title: str
    url: str
    snippet: str


class SearchBackend(ABC):
    """Abstract base class for search backends."""

    @abstractmethod
    async def search(
        self,
        query: str,
        *,
        allowed_domains: Optional[List[str]] = None,
        blocked_domains: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """Execute a search query and return results."""
        ...


def get_search_backend() -> Optional[SearchBackend]:
    """Get the configured search backend from environment variables.

    Reads RIPPERDOC_SEARCH_BACKEND to determine which backend to use.
    Supported values: "brave", "searxng"
    """
    backend_name = os.getenv("RIPPERDOC_SEARCH_BACKEND", "").lower().strip()

    if backend_name == "brave":
        from ripperdoc.services.search.brave import BraveSearchBackend
        api_key = os.getenv("RIPPERDOC_SEARCH_API_KEY", "")
        if not api_key:
            return None
        return BraveSearchBackend(api_key=api_key)

    if backend_name == "searxng":
        from ripperdoc.services.search.searxng import SearXNGBackend
        endpoint = os.getenv("RIPPERDOC_SEARCH_ENDPOINT", "http://localhost:8080")
        return SearXNGBackend(endpoint=endpoint)

    return None
