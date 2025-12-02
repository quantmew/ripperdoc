"""Utilities for compacting conversation history when context grows too large."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from ripperdoc.core.config import GlobalConfig, ModelProfile
from ripperdoc.utils.log import get_logger
from ripperdoc.utils.messages import (
    AssistantMessage,
    MessageContent,
    ProgressMessage,
    UserMessage,
    normalize_messages_for_api,
)


logger = get_logger()

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
            block_type = part.get("type")
            text_val = part.get("text")
            if text_val:
                parts.append(str(text_val))

            # Capture nested text for tool_result content blocks
            nested_content = part.get("content")
            if isinstance(nested_content, list):
                nested_text = _stringify_content(nested_content)
                if nested_text:
                    parts.append(nested_text)

            # Include tool payloads that otherwise don't have "text"
            if block_type == "tool_use" and part.get("input") is not None:
                try:
                    parts.append(json.dumps(part.get("input"), ensure_ascii=False))
                except Exception:
                    parts.append(str(part.get("input")))

            # OpenAI-style arguments blocks
            if part.get("arguments"):
                parts.append(str(part.get("arguments")))
        elif hasattr(part, "text"):
            text_val = getattr(part, "text", "")
            if text_val:
                parts.append(str(text_val))
        else:
            parts.append(str(part))
    # Filter out empty strings to avoid over-counting separators
    return "\n".join([p for p in parts if p])


def estimate_conversation_tokens(
    messages: Sequence[ConversationMessage], *, protocol: str = "anthropic"
) -> int:
    """Estimate tokens for a conversation after normalization."""
    normalized = normalize_messages_for_api(list(messages), protocol=protocol)
    total = 0
    for message in normalized:
        total += estimate_tokens_from_text(_stringify_content(message.get("content")))

        # Account for OpenAI-style tool_calls payloads (arguments + name)
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if not isinstance(call, dict):
                    total += estimate_tokens_from_text(str(call))
                    continue
                func = call.get("function")
                if isinstance(func, dict):
                    arguments = func.get("arguments")
                    if arguments:
                        total += estimate_tokens_from_text(str(arguments))
                    name = func.get("name")
                    if name:
                        total += estimate_tokens_from_text(str(name))
                else:
                    total += estimate_tokens_from_text(str(func))
    return total


def _estimate_tool_schema_tokens(tools: Sequence[Any]) -> int:
    """Estimate tokens consumed by tool schemas."""
    total = 0
    for tool in tools:
        try:
            schema = tool.input_schema.model_json_schema()
            schema_text = json.dumps(schema, sort_keys=True)
            total += estimate_tokens_from_text(schema_text)
        except Exception as exc:
            logger.error(f"Failed to estimate tokens for tool schema: {exc}")
            continue
    return total


def get_model_context_limit(
    model_profile: Optional[ModelProfile], explicit_limit: Optional[int] = None
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
    auto_compact_enabled: bool,
    *,
    protocol: str = "anthropic",
) -> ContextUsageStatus:
    """Compute context usage and thresholds."""
    max_context_tokens = max(max_context_tokens, MIN_CONTEXT_TOKENS)
    total_tokens = estimate_conversation_tokens(messages, protocol=protocol)
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
    *,
    protocol: str = "anthropic",
) -> ContextBreakdown:
    """Return a detailed breakdown of context usage.

    Includes estimates for the system prompt, tool schemas, conversation history,
    and any reserved buffer for auto-compaction.
    """
    max_context_tokens = max(max_context_tokens, MIN_CONTEXT_TOKENS)
    raw_system_tokens = estimate_tokens_from_text(system_prompt)
    base_prompt_tokens = max(0, raw_system_tokens - max(0, mcp_tokens))
    tool_schema_tokens = _estimate_tool_schema_tokens(tools)
    message_tokens = estimate_conversation_tokens(messages, protocol=protocol)
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
    messages: Sequence[ConversationMessage],
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
    force: bool = False,
    protocol: str = "anthropic",
) -> CompactionResult:
    """Compact old tool results to save context tokens."""
    tokens_before = estimate_conversation_tokens(messages, protocol=protocol)
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
            try:
                token_val = entry["tokens"]
                if token_val is None:
                    current_tokens = 0
                else:
                    current_tokens = int(token_val)
                entry["tokens"] = current_tokens + estimate_tokens_from_text(text)
            except (TypeError, ValueError):
                entry["tokens"] = estimate_tokens_from_text(text)
            occurrences_list = entry["occurrences"]
            if isinstance(occurrences_list, list):
                occurrences_list.append((message_index, content_index))
            else:
                entry["occurrences"] = [(message_index, content_index)]

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
        sum(
            int(meta["tokens"]) if isinstance(meta["tokens"], (int, float, str)) else 0
            for meta in results_by_tool_id.values()
        )
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
        results_by_tool_id.items(),
        key=lambda item: (
            int(item[1]["order"]) if isinstance(item[1]["order"], (int, float, str)) else 0
        ),
    )

    ids_to_clear: Set[str] = set()
    reclaimed = 0
    for tool_use_id, meta in sorted_results:
        ids_to_clear.add(tool_use_id)
        try:
            token_value = meta["tokens"]
            if token_value is not None:
                reclaimed += int(token_value)
        except (TypeError, ValueError):
            # If tokens is not convertible to int, skip it
            pass
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
            # Check if message has .message attribute (ProgressMessage doesn't)
            if hasattr(message, "message"):
                compacted_messages.append(
                    UserMessage(
                        message=message.message.model_copy(update={"content": new_content}),
                        tool_use_result=getattr(message, "tool_use_result", None),
                        uuid=getattr(message, "uuid", None),
                    )
                )
            else:
                # For ProgressMessage or other messages without .message, keep as is
                compacted_messages.append(message)
        else:
            compacted_messages.append(message)

    tokens_after = estimate_conversation_tokens(compacted_messages, protocol=protocol)
    return CompactionResult(
        messages=compacted_messages,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        tokens_saved=tokens_before - tokens_after,
        cleared_tool_ids=ids_to_clear,
        was_compacted=bool(ids_to_clear),
    )
