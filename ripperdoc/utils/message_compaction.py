"""Context compaction utilities"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Union

from ripperdoc.core.config import GlobalConfig, ModelProfile, get_global_config
from ripperdoc.utils.log import get_logger
from ripperdoc.utils.token_estimation import estimate_tokens
from ripperdoc.utils.messages import (
    AssistantMessage,
    MessageContent,
    ProgressMessage,
    UserMessage,
    normalize_messages_for_api,
)

logger = get_logger()

ConversationMessage = Union[UserMessage, AssistantMessage, ProgressMessage]

# Thresholds.
MAX_TOKENS_SOFT = 20_000
MAX_TOKENS_HARD = 40_000
MAX_TOOL_USES_TO_PRESERVE = 3
IMAGE_TOKEN_COST = 2_000
AUTO_COMPACT_BUFFER = 24_000
WARNING_THRESHOLD = 20_000
ERROR_THRESHOLD = 20_000
MICRO_PLACEHOLDER = "[Old tool result content cleared]"

# Context sizing.
DEFAULT_CONTEXT_TOKENS = 200_000
MIN_CONTEXT_TOKENS = 20_000

# Tools likely to generate large payloads.
TOOL_COMMANDS: Set[str] = {
    "Read",
    "Bash",
    "Grep",
    "Glob",
    "LS",
    "WebSearch",
    "WebFetch",
    "BashOutput",
    "ListMcpServers",
    "ListMcpResources",
    "ReadMcpResource",
    # "FileEdit",
    # "MultiEdit",
    # "NotebookEdit",
    # "FileWrite",
}

# State to avoid re-compacting the same tool results.
_processed_tool_use_ids: Set[str] = set()
_token_cache: Dict[str, int] = {}


@dataclass
class ContextUsageStatus:
    """Snapshot of current context usage."""

    used_tokens: int
    max_context_tokens: int
    tokens_left: int
    percent_left: float
    percent_used: float
    is_above_warning_threshold: bool
    is_above_error_threshold: bool
    is_above_auto_compact_threshold: bool

    @property
    def total_tokens(self) -> int:
        return self.used_tokens

    @property
    def is_above_warning(self) -> bool:
        return self.is_above_warning_threshold

    @property
    def is_above_error(self) -> bool:
        return self.is_above_error_threshold

    @property
    def should_auto_compact(self) -> bool:
        return self.is_above_auto_compact_threshold


@dataclass
class ContextBreakdown:
    """Detailed breakdown for UI display."""

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


@dataclass
class MicroCompactionResult:
    """Result of a micro-compaction pass."""

    messages: List[ConversationMessage]
    tokens_before: int
    tokens_after: int
    tokens_saved: int
    tools_compacted: int
    trigger_type: str
    was_compacted: bool


def _parse_truthy_env_value(value: Optional[str]) -> bool:
    if value is None:
        return False
    normalized = value.strip().lower()
    return normalized in {"1", "true", "yes", "on"}


def estimate_tokens_from_text(text: str) -> int:
    return estimate_tokens(text or "")


def _stringify_content(content: Union[str, List[MessageContent], None]) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: List[str] = []
    for part in content:
        if isinstance(part, dict):
            text_val = part.get("text") or part.get("content") or ""
            if text_val:
                parts.append(str(text_val))
            nested = part.get("content")
            if isinstance(nested, list):
                nested_text = _stringify_content(nested)
                if nested_text:
                    parts.append(nested_text)
            if part.get("arguments"):
                parts.append(str(part.get("arguments")))
        elif hasattr(part, "text"):
            text_val = getattr(part, "text", "") or ""
            if text_val:
                parts.append(text_val)
        else:
            parts.append(str(part))
    return "\n".join([p for p in parts if p])


def estimate_conversation_tokens(
    messages: Sequence[ConversationMessage], *, protocol: str = "anthropic"
) -> int:
    """Estimate tokens for a conversation after normalization."""
    normalized = normalize_messages_for_api(list(messages), protocol=protocol)
    total = 0
    for message in normalized:
        total += estimate_tokens_from_text(_stringify_content(message.get("content")))

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
    total = 0
    for tool in tools:
        try:
            schema = tool.input_schema.model_json_schema()
            schema_text = json.dumps(schema, sort_keys=True)
            total += estimate_tokens_from_text(schema_text)
        except (AttributeError, TypeError, KeyError, ValueError) as exc:
            logger.warning(
                "Failed to estimate tokens for tool schema: %s: %s",
                type(exc).__name__,
                exc,
                extra={"tool": getattr(tool, "name", None)},
            )
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

    try:
        model = getattr(model_profile, "model", None) or ""
    except Exception:
        model = ""

    # Fallback mapping; tuned for common providers.
    model = model.lower()
    if "1000k" in model or "1m" in model:
        return 1_000_000
    if "gpt-4o" in model or "gpt4o" in model:
        return 128_000
    if "gpt-4" in model:
        return 32_000
    if "deepseek" in model:
        return 128_000
    return DEFAULT_CONTEXT_TOKENS


def get_remaining_context_tokens(
    model_profile: Optional[ModelProfile], explicit_limit: Optional[int] = None
) -> int:
    """Context window minus configured output tokens."""
    context_limit = max(get_model_context_limit(model_profile, explicit_limit), MIN_CONTEXT_TOKENS)
    try:
        max_output_tokens = (
            int(getattr(model_profile, "max_tokens", 0) or 0) if model_profile else 0
        )
    except (TypeError, ValueError):
        max_output_tokens = 0
    return max(MIN_CONTEXT_TOKENS, context_limit - max(0, max_output_tokens))


def resolve_auto_compact_enabled(config: GlobalConfig) -> bool:
    env_override = os.getenv("RIPPERDOC_AUTO_COMPACT")
    if env_override is not None:
        normalized = env_override.strip().lower()
        return normalized not in {"0", "false", "no", "off"}
    return bool(config.auto_compact_enabled)


def get_context_usage_status(
    used_tokens: int,
    max_context_tokens: Optional[int],
    auto_compact_enabled: bool,
) -> ContextUsageStatus:
    """Compute usage thresholds."""
    context_limit = max(max_context_tokens or DEFAULT_CONTEXT_TOKENS, MIN_CONTEXT_TOKENS)
    effective_limit = (
        max(MIN_CONTEXT_TOKENS, context_limit - AUTO_COMPACT_BUFFER)
        if auto_compact_enabled
        else context_limit
    )

    tokens_left = max(effective_limit - used_tokens, 0)
    percent_left = (
        0.0 if effective_limit <= 0 else min(100.0, (tokens_left / effective_limit) * 100)
    )
    percent_used = 100.0 - percent_left

    warning_limit = max(0, effective_limit - WARNING_THRESHOLD)
    error_limit = max(0, effective_limit - ERROR_THRESHOLD)
    auto_compact_limit = max(MIN_CONTEXT_TOKENS, context_limit - AUTO_COMPACT_BUFFER)

    return ContextUsageStatus(
        used_tokens=used_tokens,
        max_context_tokens=context_limit,
        tokens_left=tokens_left,
        percent_left=percent_left,
        percent_used=percent_used,
        is_above_warning_threshold=used_tokens >= warning_limit,
        is_above_error_threshold=used_tokens >= error_limit,
        is_above_auto_compact_threshold=auto_compact_enabled and used_tokens >= auto_compact_limit,
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
    """Return a detailed breakdown of context usage."""
    max_context_tokens = max(max_context_tokens, MIN_CONTEXT_TOKENS)
    raw_system_tokens = estimate_tokens_from_text(system_prompt)
    base_prompt_tokens = max(0, raw_system_tokens - max(0, mcp_tokens))
    tool_schema_tokens = _estimate_tool_schema_tokens(tools)
    message_tokens = estimate_conversation_tokens(messages, protocol=protocol)
    message_count = len([m for m in messages if getattr(m, "type", "") != "progress"])
    reserved_tokens = AUTO_COMPACT_BUFFER if auto_compact_enabled else 0

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


def find_latest_assistant_usage_tokens(messages: Sequence[ConversationMessage]) -> int:
    """Best-effort extraction of usage tokens from the latest assistant message."""
    for message in reversed(messages):
        if getattr(message, "type", "") != "assistant":
            continue
        payload = getattr(message, "message", None) or getattr(message, "content", None)
        usage = getattr(payload, "usage", None)
        if usage is None and isinstance(payload, dict):
            usage = payload.get("usage")
        if not usage:
            continue
        try:
            tokens = 0
            for field in (
                "input_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
                "output_tokens",
                "prompt_tokens",
                "completion_tokens",
            ):
                value = getattr(usage, field, None)
                if value is None and isinstance(usage, dict):
                    value = usage.get(field)
                if value is not None:
                    tokens += int(value)
            if tokens > 0:
                return tokens
        except (TypeError, ValueError, AttributeError):
            logger.debug("[message_compaction] Failed to parse usage tokens")
            continue
    return 0


def estimate_used_tokens(
    messages: Sequence[ConversationMessage],
    *,
    protocol: str = "anthropic",
    precomputed_total_tokens: Optional[int] = None,
) -> int:
    usage_tokens = find_latest_assistant_usage_tokens(messages)
    if usage_tokens > 0:
        return usage_tokens
    if precomputed_total_tokens is not None:
        return precomputed_total_tokens
    return estimate_conversation_tokens(messages, protocol=protocol)


def _normalize_tool_use_id(block: Any) -> str:
    if block is None:
        return ""
    if isinstance(block, dict):
        return str(block.get("tool_use_id") or block.get("id") or "")
    return str(getattr(block, "tool_use_id", None) or getattr(block, "id", None) or "")


def _estimate_message_tokens(content_block: Any) -> int:
    """Estimate tokens for a single content block (text/image only)."""
    if content_block is None:
        return 0

    content = getattr(content_block, "content", None)
    if isinstance(content_block, dict) and content is None:
        content = content_block.get("content")

    if isinstance(content, str):
        return estimate_tokens_from_text(content)
    if isinstance(content, list):
        total = 0
        for part in content:
            part_type = getattr(part, "type", None) or (
                part.get("type") if isinstance(part, dict) else None
            )
            if part_type == "text":
                text_val = getattr(part, "text", None) if hasattr(part, "text") else None
                if text_val is None and isinstance(part, dict):
                    text_val = part.get("text")
                total += estimate_tokens_from_text(text_val or "")
            elif part_type == "image":
                total += IMAGE_TOKEN_COST
        return total

    text_val = getattr(content_block, "text", None)
    if text_val is None and isinstance(content_block, dict):
        text_val = content_block.get("text") or content_block.get("content")
    return estimate_tokens_from_text(text_val or "")


def _get_cached_token_count(cache_key: str, content_block: Any) -> int:
    estimated = _token_cache.get(cache_key)
    if estimated is None:
        estimated = _estimate_message_tokens(content_block)
        _token_cache[cache_key] = estimated
    return estimated


def micro_compact_messages(
    messages: Sequence[ConversationMessage],
    *,
    max_tokens: Optional[int] = None,
    context_limit: Optional[int] = None,
    auto_compact_enabled: Optional[bool] = None,
    protocol: str = "anthropic",
    trigger_type: str = "auto",
) -> MicroCompactionResult:
    """Micro-compaction: strip older tool_result payloads to keep context lean."""
    tokens_before = estimate_conversation_tokens(messages, protocol=protocol)

    if _parse_truthy_env_value(os.getenv("DISABLE_MICROCOMPACT")):
        return MicroCompactionResult(
            messages=list(messages),
            tokens_before=tokens_before,
            tokens_after=tokens_before,
            tokens_saved=0,
            tools_compacted=0,
            trigger_type=trigger_type,
            was_compacted=False,
        )

    # Legacy flag kept for parity with upstream behavior.
    _parse_truthy_env_value(os.getenv("USE_API_CONTEXT_MANAGEMENT"))

    is_max_tokens_specified = max_tokens is not None
    try:
        effective_max_tokens = int(max_tokens) if max_tokens is not None else MAX_TOKENS_HARD
    except (TypeError, ValueError):
        effective_max_tokens = MAX_TOKENS_HARD

    tool_use_ids_to_compact: List[str] = []
    token_counts_by_tool_use_id: Dict[str, int] = {}

    for message in messages:
        msg_type = getattr(message, "type", "")
        content = getattr(getattr(message, "message", None), "content", None)
        if msg_type not in {"user", "assistant"} or not isinstance(content, list):
            continue

        for content_block in content:
            block_type = getattr(content_block, "type", None) or (
                content_block.get("type") if isinstance(content_block, dict) else None
            )
            tool_use_id = _normalize_tool_use_id(content_block)
            tool_name = getattr(content_block, "name", None)
            if tool_name is None and isinstance(content_block, dict):
                tool_name = content_block.get("name")

            if block_type == "tool_use" and tool_name in TOOL_COMMANDS:
                if tool_use_id and tool_use_id not in _processed_tool_use_ids:
                    tool_use_ids_to_compact.append(tool_use_id)
            elif block_type == "tool_result" and tool_use_id in tool_use_ids_to_compact:
                token_count = _get_cached_token_count(tool_use_id, content_block)
                token_counts_by_tool_use_id[tool_use_id] = token_count

    latest_tool_use_ids = (
        tool_use_ids_to_compact[-MAX_TOOL_USES_TO_PRESERVE:]
        if MAX_TOOL_USES_TO_PRESERVE > 0
        else []
    )
    total_token_count = sum(token_counts_by_tool_use_id.values())

    total_tokens_removed = 0
    ids_to_remove: Set[str] = set()

    for tool_use_id in tool_use_ids_to_compact:
        if tool_use_id in latest_tool_use_ids:
            continue
        if total_token_count - total_tokens_removed > effective_max_tokens:
            ids_to_remove.add(tool_use_id)
            total_tokens_removed += token_counts_by_tool_use_id.get(tool_use_id, 0)

    if not is_max_tokens_specified:
        resolved_auto_compact = (
            auto_compact_enabled
            if auto_compact_enabled is not None
            else resolve_auto_compact_enabled(get_global_config())
        )
        usage_tokens = estimate_used_tokens(
            messages, protocol=protocol, precomputed_total_tokens=tokens_before
        )
        status = get_context_usage_status(
            usage_tokens,
            max_context_tokens=context_limit,
            auto_compact_enabled=resolved_auto_compact,
        )
        if not status.is_above_warning_threshold or total_tokens_removed < MAX_TOKENS_SOFT:
            ids_to_remove.clear()
            total_tokens_removed = 0

    def _should_remove(tool_use_id: str) -> bool:
        return tool_use_id in ids_to_remove or tool_use_id in _processed_tool_use_ids

    compacted_messages: List[ConversationMessage] = []

    for message in messages:
        msg_type = getattr(message, "type", "")
        content = getattr(getattr(message, "message", None), "content", None)

        if msg_type not in {"user", "assistant"} or not isinstance(content, list):
            compacted_messages.append(message)
            continue

        if msg_type == "assistant" and isinstance(message, AssistantMessage):
            compacted_messages.append(
                AssistantMessage(
                    message=message.message.model_copy(update={"content": list(content)}),
                    cost_usd=getattr(message, "cost_usd", 0.0),
                    duration_ms=getattr(message, "duration_ms", 0.0),
                    uuid=getattr(message, "uuid", None),
                    is_api_error_message=getattr(message, "is_api_error_message", False),
                )
            )
            continue

        filtered_content: List[MessageContent] = []
        modified = False

        for content_item in content:
            block_type = getattr(content_item, "type", None) or (
                content_item.get("type") if isinstance(content_item, dict) else None
            )
            tool_use_id = _normalize_tool_use_id(content_item)

            if block_type == "tool_result" and _should_remove(tool_use_id):
                modified = True
                if hasattr(content_item, "model_copy"):
                    new_block = content_item.model_copy()
                    new_block.text = MICRO_PLACEHOLDER
                else:
                    block_dict = (
                        dict(content_item)
                        if isinstance(content_item, dict)
                        else {"type": "tool_result"}
                    )
                    block_dict["text"] = MICRO_PLACEHOLDER
                    block_dict["tool_use_id"] = tool_use_id
                    new_block = MessageContent(**block_dict)
                filtered_content.append(new_block)
            else:
                if isinstance(content_item, MessageContent):
                    filtered_content.append(content_item)
                elif isinstance(content_item, dict):
                    filtered_content.append(MessageContent(**content_item))
                else:
                    filtered_content.append(
                        MessageContent(type=str(block_type or "text"), text=str(content_item))
                    )

        if modified and isinstance(message, UserMessage):
            compacted_messages.append(
                UserMessage(
                    message=message.message.model_copy(update={"content": filtered_content}),
                    tool_use_result=getattr(message, "tool_use_result", None),
                    uuid=getattr(message, "uuid", None),
                )
            )
        else:
            compacted_messages.append(message)

    for id_to_remove in ids_to_remove:
        _processed_tool_use_ids.add(id_to_remove)

    tokens_after = estimate_conversation_tokens(compacted_messages, protocol=protocol)
    tokens_saved = max(0, tokens_before - tokens_after)

    if ids_to_remove:
        logger.debug(
            "[message_compaction] Micro-compacted conversation",
            extra={
                "tokens_before": tokens_before,
                "tokens_after": tokens_after,
                "tokens_saved": tokens_saved,
                "cleared_tool_ids": list(ids_to_remove),
            },
        )

    return MicroCompactionResult(
        messages=compacted_messages,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        tokens_saved=tokens_saved,
        tools_compacted=len(ids_to_remove),
        trigger_type="manual" if is_max_tokens_specified else trigger_type,
        was_compacted=bool(ids_to_remove),
    )


def reset_micro_compaction_state() -> None:
    """Clear caches and processed IDs (useful for tests)."""
    _processed_tool_use_ids.clear()
    _token_cache.clear()
