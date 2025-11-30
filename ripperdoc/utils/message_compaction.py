"""Utilities for compacting conversation history when context grows too large."""

from __future__ import annotations

import json
import math
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from ripperdoc.core.config import GlobalConfig, ModelProfile
from ripperdoc.utils.messages import (
    AssistantMessage,
    MessageContent,
    ProgressMessage,
    UserMessage,
    normalize_messages_for_api,
)

ConversationMessage = Union[UserMessage, AssistantMessage, ProgressMessage]

# Defaults roughly match modern 200k context windows while still working for smaller models.
DEFAULT_CONTEXT_TOKENS = 200_000
MIN_CONTEXT_TOKENS = 20_000
WARNING_USAGE_RATIO = 0.85
AUTO_COMPACT_RATIO = 0.9
ERROR_USAGE_RATIO = 0.97
MAX_TOOL_USES_TO_PRESERVE = 3
COMPACT_PLACEHOLDER = "[Old tool result content cleared]"
# Only compact tools that usually return bulky outputs.
TOOL_NAMES_TO_COMPACT: Set[str] = {"Bash", "View", "Glob", "Grep", "LS"}


@dataclass
class ContextUsageStatus:
    """Snapshot of the current context usage."""

    total_tokens: int
    max_context_tokens: int
    tokens_left: int
    percent_used: float
    is_above_warning: bool
    is_above_error: bool
    should_auto_compact: bool


@dataclass
class CompactionResult:
    """Result of a compaction run."""

    messages: List[ConversationMessage]
    tokens_before: int
    tokens_after: int
    tokens_saved: int
    cleared_tool_ids: Set[str]
    was_compacted: bool


@dataclass
class ContextBreakdown:
    """Detailed breakdown of context usage for display."""

    max_context_tokens: int
    system_prompt_tokens: int
    mcp_tokens: int
    tool_schema_tokens: int
    memory_tokens: int
    message_tokens: int
    reserved_tokens: int
    message_count: int

    @property
    def reported_tokens(self) -> int:
        return (
            self.system_prompt_tokens
            + self.mcp_tokens
            + self.tool_schema_tokens
            + self.memory_tokens
            + self.message_tokens
        )

    @property
    def effective_tokens(self) -> int:
        """Tokens that count against the limit including any reserved buffer."""
        return min(self.max_context_tokens, self.reported_tokens + self.reserved_tokens)

    @property
    def free_tokens(self) -> int:
        return max(self.max_context_tokens - self.effective_tokens, 0)

    @property
    def percent_used(self) -> float:
        if self.max_context_tokens <= 0:
            return 0.0
        return min(100.0, (self.effective_tokens / self.max_context_tokens) * 100)

    def percent_of_limit(self, tokens: int) -> float:
        if self.max_context_tokens <= 0:
            return 0.0
        return min(100.0, (tokens / self.max_context_tokens) * 100)


def estimate_tokens_from_text(text: str) -> int:
    """Rough token estimate using a 4-characters-per-token rule."""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def _stringify_content(content: Union[str, List[MessageContent], None]) -> str:
    """Convert normalized content into plain text for estimation."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: List[str] = []
    for part in content:
        if isinstance(part, dict):
            parts.append(str(part.get("text", "")))
        elif hasattr(part, "text"):
            parts.append(str(getattr(part, "text", "")))
        else:
            parts.append(str(part))
    return "\n".join(parts)


def estimate_conversation_tokens(messages: Sequence[ConversationMessage]) -> int:
    """Estimate tokens for a conversation after normalization."""
    normalized = normalize_messages_for_api(list(messages))
    total = 0
    for message in normalized:
        total += estimate_tokens_from_text(_stringify_content(message.get("content")))
    return total


def _estimate_tool_schema_tokens(tools: Sequence[Any]) -> int:
    """Estimate tokens consumed by tool schemas."""
    total = 0
    for tool in tools:
        try:
            schema = tool.input_schema.model_json_schema()
            schema_text = json.dumps(schema, sort_keys=True)
            total += estimate_tokens_from_text(schema_text)
        except Exception:
            continue
    return total


def get_model_context_limit(
    model_profile: Optional[ModelProfile],
    explicit_limit: Optional[int] = None
) -> int:
    """Best-effort guess of the model context window."""
    env_override = os.getenv("RIPPERDOC_CONTEXT_TOKENS")
    if env_override:
        try:
            parsed = int(env_override)
            if parsed > 0:
                return parsed
        except ValueError:
            pass

    if explicit_limit and explicit_limit > 0:
        return explicit_limit

    if model_profile and getattr(model_profile, "context_window", None):
        try:
            configured = int(model_profile.context_window)  # type: ignore[arg-type]
            if configured > 0:
                return configured
        except (TypeError, ValueError):
            pass

    if model_profile and model_profile.model:
        name = model_profile.model.lower()
        if "claude" in name:
            return 200_000
        if "gpt-4o" in name or "gpt-4.1" in name or "gpt-4-turbo" in name:
            return 128_000
        if "gpt-4" in name:
            return 32_000
        if "gpt-3.5" in name:
            return 16_000
        if "deepseek" in name:
            return 64_000

    return DEFAULT_CONTEXT_TOKENS


def resolve_auto_compact_enabled(config: GlobalConfig) -> bool:
    """Return whether auto-compaction is enabled, honoring an env override."""
    env_override = os.getenv("RIPPERDOC_AUTO_COMPACT")
    if env_override is not None:
        normalized = env_override.strip().lower()
        return normalized not in {"0", "false", "no", "off"}
    return bool(config.auto_compact_enabled)


def get_context_usage_status(
    messages: Sequence[ConversationMessage],
    max_context_tokens: int,
    auto_compact_enabled: bool
) -> ContextUsageStatus:
    """Compute context usage and thresholds."""
    max_context_tokens = max(max_context_tokens, MIN_CONTEXT_TOKENS)
    total_tokens = estimate_conversation_tokens(messages)
    warning_tokens = int(max_context_tokens * WARNING_USAGE_RATIO)
    auto_compact_tokens = int(max_context_tokens * AUTO_COMPACT_RATIO)
    error_tokens = int(max_context_tokens * ERROR_USAGE_RATIO)

    return ContextUsageStatus(
        total_tokens=total_tokens,
        max_context_tokens=max_context_tokens,
        tokens_left=max(max_context_tokens - total_tokens, 0),
        percent_used=min(100.0, (total_tokens / max_context_tokens) * 100),
        is_above_warning=total_tokens >= warning_tokens,
        is_above_error=total_tokens >= error_tokens,
        should_auto_compact=auto_compact_enabled and total_tokens >= auto_compact_tokens,
    )


def summarize_context_usage(
    messages: Sequence[ConversationMessage],
    tools: Sequence[Any],
    system_prompt: str,
    max_context_tokens: int,
    auto_compact_enabled: bool,
    memory_tokens: int = 0,
    mcp_tokens: int = 0,
) -> ContextBreakdown:
    """Return a detailed breakdown of context usage.

    Includes estimates for the system prompt, tool schemas, conversation history,
    and any reserved buffer for auto-compaction.
    """
    max_context_tokens = max(max_context_tokens, MIN_CONTEXT_TOKENS)
    raw_system_tokens = estimate_tokens_from_text(system_prompt)
    base_prompt_tokens = max(0, raw_system_tokens - max(0, mcp_tokens))
    tool_schema_tokens = _estimate_tool_schema_tokens(tools)
    message_tokens = estimate_conversation_tokens(messages)
    message_count = len([m for m in messages if getattr(m, "type", "") != "progress"])
    reserved_tokens = (
        max_context_tokens - int(max_context_tokens * AUTO_COMPACT_RATIO)
        if auto_compact_enabled
        else 0
    )

    return ContextBreakdown(
        max_context_tokens=max_context_tokens,
        system_prompt_tokens=base_prompt_tokens,
        mcp_tokens=max(0, mcp_tokens),
        tool_schema_tokens=tool_schema_tokens,
        memory_tokens=max(0, memory_tokens),
        message_tokens=message_tokens,
        reserved_tokens=reserved_tokens,
        message_count=message_count,
    )


def _collect_tool_metadata(
    messages: Sequence[ConversationMessage]
) -> Tuple[List[str], Dict[str, str]]:
    """Collect tool use order and names keyed by tool_use_id."""
    tool_use_order: List[str] = []
    tool_names: Dict[str, str] = {}
    for message in messages:
        if getattr(message, "type", "") != "assistant":
            continue
        content = getattr(getattr(message, "message", None), "content", None)
        if not isinstance(content, list):
            continue
        for block in content:
            if getattr(block, "type", None) == "tool_use":
                tool_use_id = getattr(block, "tool_use_id", None) or getattr(block, "id", None)
                if tool_use_id:
                    tool_use_order.append(tool_use_id)
                    tool_names[tool_use_id] = getattr(block, "name", "") or ""
    return tool_use_order, tool_names


def compact_messages(
    messages: Sequence[ConversationMessage],
    max_context_tokens: int,
    *,
    preserve_tool_uses: int = MAX_TOOL_USES_TO_PRESERVE,
    force: bool = False
) -> CompactionResult:
    """Compact old tool results to save context tokens."""
    tokens_before = estimate_conversation_tokens(messages)
    tool_use_order, tool_names = _collect_tool_metadata(messages)
    order_lookup = {tool_id: idx for idx, tool_id in enumerate(tool_use_order)}
    preserved_ids = set(tool_use_order[-preserve_tool_uses:]) if preserve_tool_uses > 0 else set()

    # Aggregate tool result content by tool_use_id.
    results_by_tool_id: Dict[str, Dict[str, object]] = {}
    for message_index, message in enumerate(messages):
        if getattr(message, "type", "") != "user":
            continue
        content = getattr(getattr(message, "message", None), "content", None)
        if not isinstance(content, list):
            continue
        for content_index, block in enumerate(content):
            if getattr(block, "type", None) != "tool_result":
                continue
            tool_use_id = getattr(block, "tool_use_id", None)
            if not tool_use_id or tool_use_id in preserved_ids:
                continue
            tool_name = tool_names.get(tool_use_id, "")
            if TOOL_NAMES_TO_COMPACT and tool_name and tool_name not in TOOL_NAMES_TO_COMPACT:
                continue
            text = getattr(block, "text", None) or ""
            if not text or text == COMPACT_PLACEHOLDER:
                continue
            entry = results_by_tool_id.setdefault(
                tool_use_id,
                {
                    "tokens": 0,
                    "order": order_lookup.get(tool_use_id, len(order_lookup) + 1),
                    "occurrences": [],
                },
            )
            entry["tokens"] = int(entry["tokens"]) + estimate_tokens_from_text(text)
            entry["occurrences"].append((message_index, content_index))

    if not results_by_tool_id:
        return CompactionResult(
            messages=list(messages),
            tokens_before=tokens_before,
            tokens_after=tokens_before,
            tokens_saved=0,
            cleared_tool_ids=set(),
            was_compacted=False,
        )

    # Determine how many tokens we need to reclaim.
    target_tokens = max(
        MIN_CONTEXT_TOKENS,
        max_context_tokens - max(5_000, int(max_context_tokens * 0.05)),
    )
    tokens_to_reclaim = (
        sum(int(meta["tokens"]) for meta in results_by_tool_id.values())
        if force
        else max(0, tokens_before - target_tokens)
    )
    if tokens_to_reclaim <= 0 and not force:
        return CompactionResult(
            messages=list(messages),
            tokens_before=tokens_before,
            tokens_after=tokens_before,
            tokens_saved=0,
            cleared_tool_ids=set(),
            was_compacted=False,
        )

    sorted_results = sorted(
        results_by_tool_id.items(), key=lambda item: int(item[1]["order"])
    )

    ids_to_clear: Set[str] = set()
    reclaimed = 0
    for tool_use_id, meta in sorted_results:
        ids_to_clear.add(tool_use_id)
        reclaimed += int(meta["tokens"])
        if not force and reclaimed >= tokens_to_reclaim:
            break

    # Build compacted messages without mutating the originals.
    compacted_messages: List[ConversationMessage] = []
    for message_index, message in enumerate(messages):
        if getattr(message, "type", "") != "user":
            compacted_messages.append(message)
            continue

        original_content = getattr(getattr(message, "message", None), "content", None)
        if not isinstance(original_content, list):
            compacted_messages.append(message)
            continue

        new_content: List[MessageContent] = []
        modified = False
        for content_index, block in enumerate(original_content):
            if (
                getattr(block, "type", None) == "tool_result"
                and getattr(block, "tool_use_id", None) in ids_to_clear
            ):
                new_block = block.model_copy()
                new_block.text = COMPACT_PLACEHOLDER
                new_content.append(new_block)
                modified = True
            else:
                new_content.append(block)

        if modified:
            compacted_messages.append(
                UserMessage(
                    message=message.message.model_copy(update={"content": new_content}),
                    tool_use_result=getattr(message, "tool_use_result", None),
                    uuid=getattr(message, "uuid", None),
                )
            )
        else:
            compacted_messages.append(message)

    tokens_after = estimate_conversation_tokens(compacted_messages)
    return CompactionResult(
        messages=compacted_messages,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        tokens_saved=tokens_before - tokens_after,
        cleared_tool_ids=ids_to_clear,
        was_compacted=bool(ids_to_clear),
    )
