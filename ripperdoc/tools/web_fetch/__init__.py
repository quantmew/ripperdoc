"""WebFetch tool — fetches content from a URL."""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from typing import AsyncGenerator, Dict, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from ripperdoc.core.tool import Tool, ToolOutput, ToolResult, ToolUseContext, ValidationResult

TOOL_NAME = "WebFetch"

# In-memory response cache: url -> (content, timestamp)
_fetch_cache: Dict[str, tuple[str, float]] = {}
_CACHE_TTL_SECONDS = 15 * 60  # 15 minutes

# Domains that don't require explicit permission
PREAPPROVED_DOMAINS = frozenset({
    "docs.python.org",
    "pypi.org",
    "docs.rs",
    "pkg.go.dev",
    "developer.mozilla.org",
    "nodejs.org",
    "react.dev",
    "typescriptlang.org",
})


def _get_domain(url: str) -> str:
    """Extract domain from URL."""
    try:
        parsed = urlparse(url)
        return parsed.hostname or ""
    except Exception:
        return ""


class WebFetchToolInput(BaseModel):
    """Input for WebFetch."""

    url: str = Field(description="The URL to fetch content from")
    prompt: Optional[str] = Field(
        default=None,
        description="Optional prompt describing what to extract from the page",
    )
    raw: bool = Field(
        default=False,
        description="If true, return raw HTML/text instead of extracted content",
    )
    timeout: int = Field(
        default=20,
        description="Request timeout in seconds",
    )
    no_cache: bool = Field(
        default=False,
        description="Disable cache for this request",
    )
    return_format: str = Field(
        default="markdown",
        description="Reader response content type: 'markdown' or 'text'",
    )
    retain_images: bool = Field(
        default=True,
        description="Retain images in converted content",
    )


class WebFetchToolOutput(BaseModel):
    """Output for WebFetch."""

    url: str
    content: str
    status_code: int = 0
    error: Optional[str] = None
    from_cache: bool = False


class WebFetchTool(Tool[WebFetchToolInput, WebFetchToolOutput]):
    """Fetch content from a URL and return it in a model-friendly format."""

    @property
    def name(self) -> str:
        return TOOL_NAME

    async def description(self) -> str:
        return "Fetch and Convert URL to Large Model Friendly Input."

    @property
    def input_schema(self) -> type[WebFetchToolInput]:
        return WebFetchToolInput

    async def prompt(self, yolo_mode: bool = False) -> str:  # noqa: ARG002
        return (
            "Use this tool to fetch and read content from a URL. "
            "Returns the content converted to a format suitable for the model. "
            "Supports markdown and text output formats. "
            "You MUST follow the requirement: After answering the user's question, "
            "you MUST include a 'Sources:' section at the end of your response listing "
            "all relevant URLs as markdown hyperlinks."
        )

    def needs_permissions(self, input_data: Optional[WebFetchToolInput] = None) -> bool:
        if input_data is None:
            return True
        domain = _get_domain(input_data.url)
        if domain in PREAPPROVED_DOMAINS:
            return False
        return True

    async def validate_input(
        self,
        input_data: WebFetchToolInput,
        _context: Optional[ToolUseContext] = None,
    ) -> ValidationResult:
        if not input_data.url.strip():
            return ValidationResult(result=False, message="url is required")
        if not input_data.url.startswith(("http://", "https://")):
            return ValidationResult(
                result=False,
                message="url must start with http:// or https://",
            )
        return ValidationResult(result=True)

    def render_result_for_assistant(self, output: WebFetchToolOutput) -> str:
        if output.error:
            return f"WebFetch failed for {output.url}: {output.error}"
        cache_note = " (cached)" if output.from_cache else ""
        if len(output.content) > 8000:
            return output.content[:7977] + "...\n[Content truncated]"
        return output.content + cache_note

    def render_tool_use_message(
        self, input_data: WebFetchToolInput, _verbose: bool = False
    ) -> str:
        return f"Fetching {input_data.url}"

    def _convert_html_to_markdown(self, html: str) -> str:
        """Convert HTML to markdown using available libraries."""
        try:
            from markdownify import markdownify
            return markdownify(html, heading_style="ATX", strip=["script", "style"])
        except ImportError:
            pass

        try:
            import html2text
            converter = html2text.HTML2Text()
            converter.ignore_links = False
            converter.ignore_images = not True  # retain images by default
            converter.body_width = 0
            return converter.handle(html)
        except ImportError:
            pass

        # Fallback: strip HTML tags crudely
        import re
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", "", text)
        return text.strip()

    async def call(
        self,
        input_data: WebFetchToolInput,
        _context: ToolUseContext,
    ) -> AsyncGenerator[ToolOutput, None]:
        url = input_data.url

        # Check cache
        if not input_data.no_cache and url in _fetch_cache:
            cached_content, cached_time = _fetch_cache[url]
            if time.time() - cached_time < _CACHE_TTL_SECONDS:
                output = WebFetchToolOutput(
                    url=url,
                    content=cached_content,
                    status_code=200,
                    from_cache=True,
                )
                yield ToolResult(
                    data=output,
                    result_for_assistant=self.render_result_for_assistant(output),
                )
                return

        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; Ripperdoc/1.0)",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                },
            )
            with urllib.request.urlopen(req, timeout=input_data.timeout) as response:
                raw_content = response.read().decode("utf-8", errors="replace")
                content_type = response.headers.get("Content-Type", "")

            # Convert HTML to markdown if not raw mode
            if input_data.raw:
                content = raw_content
            elif "text/html" in content_type:
                content = self._convert_html_to_markdown(raw_content)
            else:
                content = raw_content

            # Cache the result
            _fetch_cache[url] = (content, time.time())

            output = WebFetchToolOutput(
                url=url,
                content=content,
                status_code=response.status,
            )
            yield ToolResult(
                data=output,
                result_for_assistant=self.render_result_for_assistant(output),
            )

        except urllib.error.HTTPError as exc:
            output = WebFetchToolOutput(
                url=url,
                content="",
                status_code=exc.code,
                error=f"HTTP {exc.code}: {exc.reason}",
            )
            yield ToolResult(
                data=output,
                result_for_assistant=self.render_result_for_assistant(output),
            )
        except Exception as exc:
            output = WebFetchToolOutput(
                url=url,
                content="",
                error=f"{type(exc).__name__}: {exc}",
            )
            yield ToolResult(
                data=output,
                result_for_assistant=self.render_result_for_assistant(output),
            )
