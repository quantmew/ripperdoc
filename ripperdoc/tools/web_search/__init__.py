"""WebSearch tool — searches the web for information."""

from __future__ import annotations

from typing import AsyncGenerator, List, Optional

from pydantic import BaseModel, Field

from ripperdoc.core.tool import Tool, ToolOutput, ToolResult, ToolUseContext, ValidationResult
from ripperdoc.services.search import SearchResult, get_search_backend

TOOL_NAME = "WebSearch"

# Session-level rate limiting
_MAX_SEARCHES_PER_SESSION = 8
_search_count = 0


class WebSearchToolInput(BaseModel):
    """Input for WebSearch."""

    query: str = Field(
        description="The search query to use",
        min_length=2,
    )
    allowed_domains: Optional[List[str]] = Field(
        default=None,
        description="Only include search results from these domains",
    )
    blocked_domains: Optional[List[str]] = Field(
        default=None,
        description="Never include search results from these domains",
    )


class WebSearchResult(BaseModel):
    """A single search result."""

    title: str
    url: str
    snippet: str


class WebSearchToolOutput(BaseModel):
    """Output for WebSearch."""

    query: str
    results: List[WebSearchResult] = Field(default_factory=list)
    error: Optional[str] = None


class WebSearchTool(Tool[WebSearchToolInput, WebSearchToolOutput]):
    """Search the web and return results formatted as markdown hyperlinks."""

    @property
    def name(self) -> str:
        return TOOL_NAME

    async def description(self) -> str:
        return "Search the web for information and return formatted results."

    @property
    def input_schema(self) -> type[WebSearchToolInput]:
        return WebSearchToolInput

    async def prompt(self, yolo_mode: bool = False) -> str:  # noqa: ARG002
        return (
            "Use this tool to search the web for up-to-date information. "
            "Returns search results formatted as markdown hyperlinks. "
            "IMPORTANT: After answering the user's question using search results, "
            "you MUST include a 'Sources:' section at the end of your response listing "
            "all relevant URLs as markdown hyperlinks: [Title](URL). "
            "This is MANDATORY — never skip including sources."
        )

    def needs_permissions(self, _input_data: Optional[WebSearchToolInput] = None) -> bool:
        return True

    async def validate_input(
        self,
        input_data: WebSearchToolInput,
        _context: Optional[ToolUseContext] = None,
    ) -> ValidationResult:
        global _search_count
        if not input_data.query.strip():
            return ValidationResult(result=False, message="query is required")
        if _search_count >= _MAX_SEARCHES_PER_SESSION:
            return ValidationResult(
                result=False,
                message=f"Web search rate limit reached ({_MAX_SEARCHES_PER_SESSION} searches per session).",
            )
        return ValidationResult(result=True)

    def render_result_for_assistant(self, output: WebSearchToolOutput) -> str:
        if output.error:
            return f"WebSearch failed for '{output.query}': {output.error}"
        if not output.results:
            return f"No search results found for '{output.query}'."

        lines = [f"Search results for '{output.query}':", ""]
        for result in output.results:
            lines.append(f"- [{result.title}]({result.url})")
            if result.snippet:
                lines.append(f"  {result.snippet}")
        return "\n".join(lines)

    def render_tool_use_message(
        self, input_data: WebSearchToolInput, _verbose: bool = False
    ) -> str:
        return f"Searching web for: {input_data.query}"

    async def call(
        self,
        input_data: WebSearchToolInput,
        _context: ToolUseContext,
    ) -> AsyncGenerator[ToolOutput, None]:
        global _search_count

        backend = get_search_backend()
        if backend is None:
            output = WebSearchToolOutput(
                query=input_data.query,
                results=[],
                error=(
                    "WebSearch requires a search backend to be configured. "
                    "Set RIPPERDOC_SEARCH_BACKEND=brave and RIPPERDOC_SEARCH_API_KEY=<key> "
                    "to enable web search."
                ),
            )
            yield ToolResult(
                data=output,
                result_for_assistant=self.render_result_for_assistant(output),
            )
            return

        try:
            raw_results = await backend.search(
                input_data.query,
                allowed_domains=input_data.allowed_domains,
                blocked_domains=input_data.blocked_domains,
            )

            results = [
                WebSearchResult(
                    title=r.title,
                    url=r.url,
                    snippet=r.snippet,
                )
                for r in raw_results
            ]

            _search_count += 1

            output = WebSearchToolOutput(
                query=input_data.query,
                results=results,
            )
            yield ToolResult(
                data=output,
                result_for_assistant=self.render_result_for_assistant(output),
            )

        except Exception as exc:
            output = WebSearchToolOutput(
                query=input_data.query,
                results=[],
                error=f"{type(exc).__name__}: {exc}",
            )
            yield ToolResult(
                data=output,
                result_for_assistant=self.render_result_for_assistant(output),
            )
