"""Provider-aware message normalization strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence


ReasoningKeys = ("reasoning_content", "reasoning_details", "reasoning")


@dataclass
class _NormalizationStats:
    tool_results_seen: int = 0
    tool_uses_seen: int = 0
    skipped_tool_uses_no_result: int = 0
    skipped_tool_uses_no_id: int = 0
    skipped_tool_results_no_call: int = 0


def _msg_type(msg: Any) -> Optional[str]:
    if hasattr(msg, "type"):
        return getattr(msg, "type", None)
    if isinstance(msg, dict):
        return msg.get("type")
    return None


def _msg_content(msg: Any) -> Any:
    if hasattr(msg, "message"):
        return getattr(getattr(msg, "message", None), "content", None)
    if isinstance(msg, dict):
        message_payload = msg.get("message")
        if isinstance(message_payload, dict):
            return message_payload.get("content")
        if "content" in msg:
            return msg.get("content")
    return None


def _msg_metadata(msg: Any) -> Dict[str, Any]:
    message_obj = getattr(msg, "message", None)
    if message_obj is not None and hasattr(message_obj, "metadata"):
        try:
            meta = getattr(message_obj, "metadata", {}) or {}
            meta_dict = dict(meta) if isinstance(meta, dict) else {}
        except (TypeError, ValueError):
            meta_dict = {}
        reasoning_val = getattr(message_obj, "reasoning", None)
        if reasoning_val is not None and "reasoning" not in meta_dict:
            meta_dict["reasoning"] = reasoning_val
        return meta_dict
    if isinstance(msg, dict):
        message_payload = msg.get("message")
        if isinstance(message_payload, dict):
            meta = message_payload.get("metadata") or {}
            meta_dict = dict(meta) if isinstance(meta, dict) else {}
            if "reasoning" not in meta_dict and "reasoning" in message_payload:
                meta_dict["reasoning"] = message_payload.get("reasoning")
            return meta_dict
    return {}


def _block_value(block: Any, key: str, default: Any = None) -> Any:
    if isinstance(block, dict):
        return block.get(key, default)
    return getattr(block, key, default)


def _block_type(block: Any) -> Any:
    return _block_value(block, "type")


def _append_reasoning_meta(target: Dict[str, Any], meta: Dict[str, Any]) -> None:
    for key in ReasoningKeys:
        if key in meta and meta[key] is not None:
            target[key] = meta[key]


def _precompute_openai_positions(messages: Sequence[Any]) -> tuple[Dict[str, int], Dict[str, int]]:
    tool_result_positions: Dict[str, int] = {}
    tool_use_positions: Dict[str, int] = {}

    for idx, msg in enumerate(messages):
        msg_type = _msg_type(msg)
        content = _msg_content(msg)
        if not isinstance(content, list):
            continue

        if msg_type == "user":
            for block in content:
                if _block_type(block) != "tool_result":
                    continue
                tool_id = _block_value(block, "tool_use_id") or _block_value(block, "id")
                if tool_id and tool_id not in tool_result_positions:
                    tool_result_positions[str(tool_id)] = idx
        elif msg_type == "assistant":
            for block in content:
                if _block_type(block) != "tool_use":
                    continue
                tool_id = _block_value(block, "id") or _block_value(block, "tool_use_id")
                if tool_id and tool_id not in tool_use_positions:
                    tool_use_positions[str(tool_id)] = idx

    return tool_result_positions, tool_use_positions


def _normalize_openai_user_list(
    *,
    msg_index: int,
    user_content: list[Any],
    meta: Dict[str, Any],
    normalized: list[dict[str, Any]],
    stats: _NormalizationStats,
    tool_use_positions: Dict[str, int],
    to_openai: Callable[[Any], Dict[str, Any]],
) -> None:
    has_images = any(_block_type(block) == "image" for block in user_content)
    has_text_only = all(
        _block_type(block) in ("text", "image", "tool_result") for block in user_content
    )
    has_tool_result = any(_block_type(block) == "tool_result" for block in user_content)

    if has_images or (has_text_only and not has_tool_result):
        content_array: List[Dict[str, Any]] = []
        for block in user_content:
            block_type = _block_type(block)
            if block_type == "image":
                content_array.append(to_openai(block))
            elif block_type == "text":
                content_array.append({"type": "text", "text": _block_value(block, "text", "") or ""})
            elif block_type == "tool_result":
                stats.tool_results_seen += 1
                tool_id = _block_value(block, "tool_use_id") or _block_value(block, "id")
                if not tool_id:
                    stats.skipped_tool_results_no_call += 1
                    continue
                call_pos = tool_use_positions.get(str(tool_id))
                if call_pos is None or call_pos >= msg_index:
                    stats.skipped_tool_results_no_call += 1
                    continue
                mapped = to_openai(block)
                if mapped:
                    normalized.append(mapped)

        if content_array:
            user_msg: Dict[str, Any] = {"role": "user", "content": content_array}
            if meta:
                _append_reasoning_meta(user_msg, meta)
            normalized.append(user_msg)
        return

    openai_msgs: List[Dict[str, Any]] = []
    for block in user_content:
        if _block_type(block) == "tool_result":
            stats.tool_results_seen += 1
            tool_id = _block_value(block, "tool_use_id") or _block_value(block, "id")
            if not tool_id:
                stats.skipped_tool_results_no_call += 1
                continue
            call_pos = tool_use_positions.get(str(tool_id))
            if call_pos is None or call_pos >= msg_index:
                stats.skipped_tool_results_no_call += 1
                continue

        mapped = to_openai(block)
        if mapped:
            openai_msgs.append(mapped)

    if meta and openai_msgs:
        for candidate in openai_msgs:
            _append_reasoning_meta(candidate, meta)

    normalized.extend(openai_msgs)


def _normalize_openai_assistant_list(
    *,
    msg_index: int,
    asst_content: list[Any],
    meta: Dict[str, Any],
    normalized: list[dict[str, Any]],
    stats: _NormalizationStats,
    tool_result_positions: Dict[str, int],
    to_openai: Callable[[Any], Dict[str, Any]],
    thinking_mode: Optional[str],
    logger: Any,
) -> None:
    assistant_openai_msgs: List[Dict[str, Any]] = []
    tool_calls: List[Dict[str, Any]] = []
    text_parts: List[str] = []

    for block in asst_content:
        block_type = _block_type(block)
        if block_type == "tool_use":
            stats.tool_uses_seen += 1
            tool_id = _block_value(block, "tool_use_id") or _block_value(block, "id")
            if not tool_id:
                stats.skipped_tool_uses_no_id += 1
                continue

            result_pos = tool_result_positions.get(str(tool_id))
            if result_pos is None or result_pos <= msg_index:
                stats.skipped_tool_uses_no_result += 1
                continue

            mapped = to_openai(block)
            mapped_calls = mapped.get("tool_calls") if mapped else None
            if mapped_calls:
                tool_calls.extend(mapped_calls)
            continue

        if block_type == "text":
            text_parts.append(_block_value(block, "text", "") or "")
            continue

        mapped = to_openai(block)
        if mapped:
            assistant_openai_msgs.append(mapped)

    if tool_calls:
        tool_call_msg: Dict[str, Any] = {
            "role": "assistant",
            "content": "\n".join(text_parts) if text_parts else None,
            "tool_calls": tool_calls,
        }
        reasoning_content = meta.get("reasoning_content") if meta else None
        if reasoning_content is not None:
            tool_call_msg["reasoning_content"] = reasoning_content
            logger.debug(
                "[normalize_messages_for_api] Added reasoning_content to "
                "tool_call message (len=%s)",
                len(str(reasoning_content)),
            )
        elif thinking_mode == "deepseek":
            logger.warning(
                "[normalize_messages_for_api] DeepSeek mode: assistant "
                "message with tool_calls but no reasoning_content in metadata. "
                "meta_keys=%s",
                list(meta.keys()) if meta else [],
            )
        assistant_openai_msgs.append(tool_call_msg)
    elif text_parts:
        assistant_openai_msgs.append({"role": "assistant", "content": "\n".join(text_parts)})

    if meta and assistant_openai_msgs and not tool_calls:
        _append_reasoning_meta(assistant_openai_msgs[-1], meta)

    normalized.extend(assistant_openai_msgs)


def _normalize_openai_messages(
    *,
    messages: Sequence[Any],
    thinking_mode: Optional[str],
    to_openai: Callable[[Any], Dict[str, Any]],
    logger: Any,
) -> tuple[list[dict[str, Any]], _NormalizationStats, Dict[str, int], Dict[str, int]]:
    normalized: list[dict[str, Any]] = []
    stats = _NormalizationStats()
    tool_result_positions, tool_use_positions = _precompute_openai_positions(messages)

    for msg_index, msg in enumerate(messages):
        msg_type = _msg_type(msg)
        if msg_type in (None, "progress"):
            continue

        content = _msg_content(msg)
        meta = _msg_metadata(msg)

        if msg_type == "user":
            if isinstance(content, list):
                _normalize_openai_user_list(
                    msg_index=msg_index,
                    user_content=content,
                    meta=meta,
                    normalized=normalized,
                    stats=stats,
                    tool_use_positions=tool_use_positions,
                    to_openai=to_openai,
                )
            else:
                normalized.append({"role": "user", "content": content})
            continue

        if msg_type == "assistant":
            if isinstance(content, list):
                _normalize_openai_assistant_list(
                    msg_index=msg_index,
                    asst_content=content,
                    meta=meta,
                    normalized=normalized,
                    stats=stats,
                    tool_result_positions=tool_result_positions,
                    to_openai=to_openai,
                    thinking_mode=thinking_mode,
                    logger=logger,
                )
            else:
                normalized.append({"role": "assistant", "content": content})

    return normalized, stats, tool_result_positions, tool_use_positions


def _normalize_default_messages(
    *,
    messages: Sequence[Any],
    to_api: Callable[[Any], Dict[str, Any]],
    protocol: str,
) -> tuple[list[dict[str, Any]], _NormalizationStats]:
    normalized: list[dict[str, Any]] = []
    stats = _NormalizationStats()

    for msg in messages:
        msg_type = _msg_type(msg)
        if msg_type in (None, "progress"):
            continue

        content = _msg_content(msg)
        if msg_type == "user":
            if isinstance(content, list):
                api_blocks = []
                for block in content:
                    if _block_type(block) == "tool_result":
                        stats.tool_results_seen += 1
                    if protocol == "anthropic" and _block_type(block) in (
                        "thinking",
                        "redacted_thinking",
                    ):
                        signature = _block_value(block, "signature")
                        if not isinstance(signature, str) or not signature:
                            continue
                    mapped = to_api(block)
                    if mapped:
                        api_blocks.append(mapped)
                if api_blocks:
                    normalized.append({"role": "user", "content": api_blocks})
            else:
                normalized.append({"role": "user", "content": content})
            continue

        if msg_type == "assistant":
            if isinstance(content, list):
                api_blocks = []
                for block in content:
                    if _block_type(block) == "tool_use":
                        stats.tool_uses_seen += 1
                    if protocol == "anthropic" and _block_type(block) in (
                        "thinking",
                        "redacted_thinking",
                    ):
                        signature = _block_value(block, "signature")
                        if not isinstance(signature, str) or not signature:
                            continue
                    mapped = to_api(block)
                    if mapped:
                        api_blocks.append(mapped)
                if api_blocks:
                    normalized.append({"role": "assistant", "content": api_blocks})
            else:
                normalized.append({"role": "assistant", "content": content})

    return normalized, stats


def normalize_messages_for_api_impl(
    messages: Sequence[Any],
    *,
    protocol: str,
    tool_mode: str,
    thinking_mode: Optional[str],
    to_api: Callable[[Any], Dict[str, Any]],
    to_openai: Callable[[Any], Dict[str, Any]],
    apply_deepseek_reasoning_content: Callable[[list[dict[str, Any]], bool], list[dict[str, Any]]],
    logger: Any,
) -> list[dict[str, Any]]:
    """Normalize conversation messages into provider API payloads."""
    effective_tool_mode = (tool_mode or "native").lower()
    if effective_tool_mode not in {"native", "text"}:
        effective_tool_mode = "native"

    if protocol == "openai":
        normalized, stats, tool_result_positions, tool_use_positions = _normalize_openai_messages(
            messages=messages,
            thinking_mode=thinking_mode,
            to_openai=to_openai,
            logger=logger,
        )
    else:
        normalized, stats = _normalize_default_messages(
            messages=messages,
            to_api=to_api,
            protocol=protocol,
        )
        tool_result_positions = {}
        tool_use_positions = {}

    logger.debug(
        "[normalize_messages_for_api] protocol=%s tool_mode=%s thinking_mode=%s "
        "input_msgs=%s normalized=%s tool_results_seen=%s tool_uses_seen=%s "
        "tool_result_positions=%s tool_use_positions=%s "
        "skipped_tool_uses_no_result=%s skipped_tool_uses_no_id=%s "
        "skipped_tool_results_no_call=%s",
        protocol,
        effective_tool_mode,
        thinking_mode,
        len(messages),
        len(normalized),
        stats.tool_results_seen,
        stats.tool_uses_seen,
        len(tool_result_positions),
        len(tool_use_positions),
        stats.skipped_tool_uses_no_result,
        stats.skipped_tool_uses_no_id,
        stats.skipped_tool_results_no_call,
    )

    if thinking_mode == "deepseek":
        normalized = apply_deepseek_reasoning_content(normalized, False)

    return normalized
