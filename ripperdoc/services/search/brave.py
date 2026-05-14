"""Brave Search API backend."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import List, Optional

from ripperdoc.services.search import SearchBackend, SearchResult


class BraveSearchBackend(SearchBackend):
    """Search backend using the Brave Web Search API."""

    BASE_URL = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def search(
        self,
        query: str,
        *,
        allowed_domains: Optional[List[str]] = None,
        blocked_domains: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        params = {"q": query}
        if allowed_domains:
            params["q"] += " " + " ".join(f"site:{d}" for d in allowed_domains)
        if blocked_domains:
            params["q"] += " " + " ".join(f"-site:{d}" for d in blocked_domains)

        url = f"{self.BASE_URL}?{urllib.parse.urlencode(params)}"

        req = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": self.api_key,
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Brave Search API error: HTTP {exc.code}") from exc
        except Exception as exc:
            raise RuntimeError(f"Brave Search request failed: {exc}") from exc

        results: List[SearchResult] = []
        for item in data.get("web", {}).get("results", []):
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("description", ""),
                )
            )

        return results[:20]
