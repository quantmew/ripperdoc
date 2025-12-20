"""Tool search helper for deferred tool loading."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import Any, AsyncGenerator, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from ripperdoc.core.tool import (
    Tool,
    ToolOutput,
    ToolResult,
    ToolUseContext,
    ToolUseExample,
    ValidationResult,
    build_tool_description,
)
from ripperdoc.utils.log import get_logger


logger = get_logger()


class ToolSearchInput(BaseModel):
    """Input for tool search and activation."""

    query: Optional[str] = Field(
        default=None,
        description="Search phrase describing the capability or tool name you need.",
    )
    names: Optional[List[str]] = Field(
        default=None,
        description="Explicit tool names to activate. Use after seeing search results.",
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=25,
        description="Maximum number of matching tools to return.",
    )
    include_active: bool = Field(
        default=False,
        description="Include already-active tools in the search results.",
    )
    auto_activate: bool = Field(
        default=True,
        description="If true, activate the returned matches so they can be called immediately.",
    )
    include_examples: bool = Field(
        default=False,
        description="Include input examples in the returned tool descriptions.",
    )
    model_config = ConfigDict(extra="ignore")


class ToolSearchMatch(BaseModel):
    """Metadata about a matching tool."""

    name: str
    user_facing_name: Optional[str] = None
    description: Optional[str] = None
    active: bool = False
    deferred: bool = False


class ToolSearchOutput(BaseModel):
    """Search results and activation summary."""

    matches: List[ToolSearchMatch] = Field(default_factory=list)
    activated: List[str] = Field(default_factory=list)
    missing: List[str] = Field(default_factory=list)
    deferred_remaining: int = 0


class ToolSearchTool(Tool[ToolSearchInput, ToolSearchOutput]):
    """Search across available tools and activate deferred ones on demand."""

    @property
    def name(self) -> str:
        return "ToolSearch"

    async def description(self) -> str:
        return (
            "Search available tools by name or description, returning a small set of candidates. "
            "Use this when you suspect a capability exists but is not currently active. "
            "Matching deferred tools are automatically activated so you can call them next."
        )

    @property
    def input_schema(self) -> type[ToolSearchInput]:
        return ToolSearchInput

    def input_examples(self) -> List[ToolUseExample]:
        return [
            ToolUseExample(
                description="Search for notebook-related tools and activate top results",
                example={"query": "notebook", "max_results": 3},
            ),
            ToolUseExample(
                description="Activate a known tool by name",
                example={"names": ["mcp__search__query"], "auto_activate": True},
            ),
        ]

    async def prompt(self, yolo_mode: bool = False) -> str:  # noqa: ARG002
        return (
            "Search for a tool by providing a short description (e.g., 'query database', 'render notebook'). "
            "Use names to activate tools you've already discovered. "
            "Keep queries concise to retrieve the 3-5 most relevant tools."
        )

    def is_read_only(self) -> bool:
        return True

    def is_concurrency_safe(self) -> bool:
        return True

    def needs_permissions(self, input_data: Optional[ToolSearchInput] = None) -> bool:  # noqa: ARG002
        return False

    async def validate_input(
        self,
        input_data: ToolSearchInput,
        context: Optional[ToolUseContext] = None,  # noqa: ARG002
    ) -> ValidationResult:
        if not (input_data.query or input_data.names):
            return ValidationResult(
                result=False,
                message="Provide a search query or explicit tool names to load.",
            )
        return ValidationResult(result=True)

    def render_result_for_assistant(self, output: ToolSearchOutput) -> str:
        lines = []
        if output.activated:
            lines.append(f"Activated: {', '.join(sorted(output.activated))}")
        if output.matches:
            lines.append("Matches:")
            for match in output.matches:
                status = []
                if match.active:
                    status.append("active")
                if match.deferred:
                    status.append("deferred")
                status_text = f" ({', '.join(status)})" if status else ""
                lines.append(f"- {match.name}{status_text}: {match.description or ''}".strip())
        if output.missing:
            lines.append(f"Unknown tool names: {', '.join(sorted(output.missing))}")
        if output.deferred_remaining:
            lines.append(f"Deferred tools remaining: {output.deferred_remaining}")
        return "\n".join(lines) if lines else "No matching tools found."

    def render_tool_use_message(self, input_data: ToolSearchInput, verbose: bool = False) -> str:
        detail = f'"{input_data.query}"' if input_data.query else ", ".join(input_data.names or [])
        return f"Search tools for {detail}"

    async def _search(
        self,
        query: str,
        registry: Any,
        *,
        include_active: bool,
        include_examples: bool,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Regex + BM25-style search over tool metadata."""
        normalized = (query or "").strip().lower()
        if not normalized:
            return []

        regex: Optional[re.Pattern[str]] = None
        if normalized.startswith("/") and normalized.endswith("/") and len(normalized) > 2:
            try:
                regex = re.compile(normalized[1:-1], re.IGNORECASE)
            except re.error:
                regex = None
                logger.exception("[tool_search] Invalid regex search query", extra={"query": query})

        def _tokenize(text: str) -> List[str]:
            return re.findall(r"[a-z0-9]+", text.lower())

        corpus: List[tuple[str, Any, List[str], int, str]] = []
        for name, tool in registry.iter_named_tools():
            try:
                description = await build_tool_description(
                    tool, include_examples=include_examples, max_examples=2
                )
            except (OSError, RuntimeError, ValueError, TypeError, AttributeError, KeyError) as exc:
                description = ""
                logger.warning(
                    "[tool_search] Failed to build tool description: %s: %s",
                    type(exc).__name__,
                    exc,
                    extra={"tool_name": getattr(tool, "name", None)},
                )
            doc_text = " ".join([name, tool.user_facing_name(), description])
            tokens = _tokenize(doc_text)
            corpus.append((name, tool, tokens, len(tokens), description))

        if not corpus:
            return []

        avg_len = sum(doc_len for _, _, _, doc_len, _ in corpus) / len(corpus)
        query_terms = _tokenize(normalized)
        df: Dict[str, int] = defaultdict(int)
        for _, _, tokens, _, _ in corpus:
            seen_terms = set(tokens)
            for term in query_terms:
                if term in seen_terms:
                    df[term] += 1

        k1 = 1.5
        b = 0.75

        def _bm25_score(tokens: List[str], doc_len: int) -> float:
            score = 0.0
            counts = Counter(tokens)
            for term in query_terms:
                if term not in counts:
                    continue
                tf = counts[term]
                df_term = df.get(term, 0) or 1
                idf = math.log((len(corpus) - df_term + 0.5) / (df_term + 0.5) + 1)
                numerator = tf * (k1 + 1)
                denom = tf + k1 * (1 - b + b * (doc_len / (avg_len or 1)))
                score += idf * (numerator / denom)
            return score

        results: List[Dict[str, Any]] = []
        for name, tool, tokens, doc_len, description in corpus:
            if not include_active and registry.is_active(name):
                continue

            combined_text = " ".join([name, tool.user_facing_name(), description]).lower()
            score = _bm25_score(tokens, doc_len)
            if regex and regex.search(combined_text):
                score += 5.0
            if normalized in combined_text:
                score += 3.0
            score += SequenceMatcher(None, normalized, name.lower()).ratio() * 2
            score += SequenceMatcher(None, normalized, tool.user_facing_name().lower()).ratio()

            results.append(
                {
                    "name": name,
                    "user_facing_name": tool.user_facing_name(),
                    "active": registry.is_active(name),
                    "deferred": name in getattr(registry, "deferred_names", set()),
                    "description": description,
                    "input_schema": tool.input_schema.model_json_schema(),
                    "score": score,
                }
            )

        return sorted(results, key=lambda item: item.get("score", 0), reverse=True)[:limit]

    async def _describe_by_name(
        self,
        registry: Any,
        names: List[str],
        include_examples: bool,
        limit: int,
    ) -> List[Dict[str, Any]]:
        seen = set()
        results: List[Dict[str, Any]] = []
        for name in names:
            if not name or name in seen:
                continue
            seen.add(name)
            tool = registry.get(name) if hasattr(registry, "get") else None
            if not tool:
                continue
            description = await build_tool_description(
                tool, include_examples=include_examples, max_examples=2
            )
            results.append(
                {
                    "name": name,
                    "user_facing_name": tool.user_facing_name(),
                    "description": description,
                    "active": (
                        getattr(registry, "is_active", lambda *_: False)(name)
                        if hasattr(registry, "is_active")
                        else False
                    ),
                    "deferred": name in getattr(registry, "deferred_names", set()),
                    "score": 0.0,
                }
            )
            if len(results) >= limit:
                break
        return results

    async def call(
        self,
        input_data: ToolSearchInput,
        context: ToolUseContext,
    ) -> AsyncGenerator[ToolOutput, None]:
        registry = getattr(context, "tool_registry", None)
        if not registry:
            yield ToolResult(
                data=ToolSearchOutput(),
                result_for_assistant="Tool registry unavailable; cannot search tools.",
            )
            return

        matches: List[Dict[str, Any]] = []
        if input_data.query:
            matches = await self._search(
                input_data.query,
                registry,
                include_active=input_data.include_active,
                include_examples=input_data.include_examples,
                limit=input_data.max_results,
            )

        if input_data.names:
            named_matches = await self._describe_by_name(
                registry,
                input_data.names,
                input_data.include_examples,
                input_data.max_results,
            )
            # Merge in explicit names that weren't returned by the search query.
            known = {m["name"] for m in matches}
            matches.extend([m for m in named_matches if m["name"] not in known])

        if matches:
            matches = sorted(matches, key=lambda item: item.get("score", 0), reverse=True)
            if input_data.max_results:
                matches = matches[: input_data.max_results]

        max_description_chars = 600
        for match in matches:
            desc = match.get("description")
            if (
                max_description_chars
                and isinstance(desc, str)
                and len(desc) > max_description_chars
            ):
                match["description"] = desc[:max_description_chars] + "..."

        # Activate tools as requested.
        activation_targets: List[str] = []
        if input_data.names:
            activation_targets.extend(input_data.names)
        elif input_data.auto_activate:
            activation_targets.extend([match["name"] for match in matches])

        activated: List[str] = []
        missing: List[str] = []
        if activation_targets:
            activated, missing = registry.activate_tools(activation_targets)

        normalized_matches: List[ToolSearchMatch] = []
        for match in matches[: input_data.max_results]:
            normalized_matches.append(
                ToolSearchMatch(
                    name=match.get("name", ""),
                    user_facing_name=match.get("user_facing_name"),
                    description=match.get("description"),
                    active=bool(match.get("active")),
                    deferred=bool(match.get("deferred")),
                )
            )

        output = ToolSearchOutput(
            matches=normalized_matches,
            activated=activated,
            missing=missing,
            deferred_remaining=len(getattr(registry, "deferred_names", [])),
        )

        yield ToolResult(
            data=output,
            result_for_assistant=self.render_result_for_assistant(output),
        )
