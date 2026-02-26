"""Shared helpers for OpenAI Responses-style payloads and streams."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from ripperdoc.core.message_utils import normalize_tool_args, openai_usage_tokens


def extract_unsupported_parameter_name(message: str) -> Optional[str]:
    """Extract unsupported parameter name from API error text."""
    match = re.search(r"Unsupported parameter:\s*([A-Za-z0-9_.-]+)", message or "")
    if not match:
        return None
    return match.group(1).strip() or None


def convert_chat_function_tools_to_responses_tools(
    openai_tools: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Convert chat.completions function tool schema to Responses tool schema."""
    responses_tools: List[Dict[str, Any]] = []
    for tool in openai_tools:
        if not isinstance(tool, dict):
            continue
        tool_type = str(tool.get("type") or "").lower()
        if tool_type != "function":
            responses_tools.append(tool)
            continue

        function_obj = tool.get("function")
        if not isinstance(function_obj, dict):
            responses_tools.append(tool)
            continue

        name = function_obj.get("name")
        if not isinstance(name, str) or not name:
            continue

        responses_tool: Dict[str, Any] = {
            "type": "function",
            "name": name,
            "description": function_obj.get("description"),
            "parameters": function_obj.get("parameters"),
        }
        strict = function_obj.get("strict")
        if isinstance(strict, bool):
            responses_tool["strict"] = strict
        responses_tools.append(responses_tool)

    return responses_tools


def _flatten_message_text(content: Any) -> str:
    """Flatten assorted message content shapes into plain text."""
    parts: List[str] = []

    def _append_text(value: Any) -> None:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                parts.append(stripped)

    if isinstance(content, str):
        _append_text(content)
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, str):
                _append_text(item)
                continue
            if not isinstance(item, dict):
                continue
            part_type = str(item.get("type") or "").lower()
            text = item.get("text")
            if isinstance(text, str):
                _append_text(text)
                continue
            if part_type == "tool_use":
                name = str(item.get("name") or "tool")
                tool_input = item.get("input")
                if tool_input is None:
                    _append_text(f"[tool:{name}]")
                else:
                    try:
                        serialized = json.dumps(tool_input, ensure_ascii=False)
                    except (TypeError, ValueError):
                        serialized = str(tool_input)
                    _append_text(f"[tool:{name}] {serialized}")
                continue
            if part_type in {"tool_result", "tool"}:
                _append_text(item.get("content"))
    elif isinstance(content, dict):
        _append_text(content.get("text"))
        _append_text(content.get("content"))

    return "\n".join(parts)


def _block_text_value(block: Dict[str, Any]) -> Optional[str]:
    """Extract text from a normalized content block when available."""
    text_value = block.get("text")
    if isinstance(text_value, str):
        stripped = text_value.strip()
        if stripped:
            return stripped

    content_value = block.get("content")
    if isinstance(content_value, str):
        stripped = content_value.strip()
        if stripped:
            return stripped
    if isinstance(content_value, list):
        text_chunks: List[str] = []
        for part in content_value:
            if isinstance(part, str):
                stripped = part.strip()
                if stripped:
                    text_chunks.append(stripped)
                continue
            if not isinstance(part, dict):
                continue
            part_text = part.get("text")
            if isinstance(part_text, str):
                stripped = part_text.strip()
                if stripped:
                    text_chunks.append(stripped)
        if text_chunks:
            return "\n".join(text_chunks)
    return None


def _serialize_function_arguments(raw_input: Any) -> str:
    """Serialize tool input to JSON string for Responses function_call.arguments."""
    normalized_input = normalize_tool_args(raw_input)
    try:
        return json.dumps(normalized_input, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return "{}"


def _build_message_item(
    *,
    role: str,
    text: str,
    text_type: str,
    include_phase: bool,
    phase_value: Optional[str],
) -> Dict[str, Any]:
    """Create one Responses message input item."""
    item: Dict[str, Any] = {
        "role": role,
        "content": [{"type": text_type, "text": text}],
    }
    if include_phase and role == "assistant" and phase_value in {"commentary", "final_answer"}:
        item["phase"] = phase_value
    return item


def build_input_from_normalized_messages(
    normalized_messages: List[Dict[str, Any]],
    *,
    assistant_text_type: str = "output_text",
    include_phase: bool = False,
) -> List[Dict[str, Any]]:
    """Convert normalized chat history into Responses API input items.

    For tool history, this prefers native Responses item types:
    - assistant `tool_use` => `function_call`
    - user `tool_result` => `function_call_output`
    """
    items: List[Dict[str, Any]] = []
    resolved_assistant_text_type = (
        assistant_text_type if assistant_text_type in {"input_text", "output_text"} else "output_text"
    )
    for message in normalized_messages:
        role_raw = str(message.get("role") or "").strip().lower()
        if not role_raw:
            continue

        role = role_raw
        if role == "tool":
            role = "assistant"
        if role not in {"user", "assistant", "system", "developer"}:
            continue

        phase_normalized: Optional[str] = None
        if include_phase and role == "assistant":
            phase_value: Any = message.get("phase")
            if not isinstance(phase_value, str):
                metadata = message.get("metadata")
                if isinstance(metadata, dict):
                    phase_value = metadata.get("phase")
            if isinstance(phase_value, str):
                candidate = phase_value.strip().lower()
                if candidate in {"commentary", "final_answer"}:
                    phase_normalized = candidate

        text_type = resolved_assistant_text_type if role == "assistant" else "input_text"
        raw_content = message.get("content")

        if isinstance(raw_content, str):
            stripped = raw_content.strip()
            if stripped:
                items.append(
                    _build_message_item(
                        role=role,
                        text=stripped,
                        text_type=text_type,
                        include_phase=include_phase,
                        phase_value=phase_normalized,
                    )
                )
            continue

        if not isinstance(raw_content, list):
            fallback_text = _flatten_message_text(raw_content)
            if fallback_text:
                items.append(
                    _build_message_item(
                        role=role,
                        text=fallback_text,
                        text_type=text_type,
                        include_phase=include_phase,
                        phase_value=phase_normalized,
                    )
                )
            continue

        text_parts: List[str] = []

        def _flush_text_parts() -> None:
            if not text_parts:
                return
            merged_text = "\n".join(part for part in text_parts if part)
            text_parts.clear()
            if not merged_text:
                return
            items.append(
                _build_message_item(
                    role=role,
                    text=merged_text,
                    text_type=text_type,
                    include_phase=include_phase,
                    phase_value=phase_normalized,
                )
            )

        for part in raw_content:
            if isinstance(part, str):
                stripped = part.strip()
                if stripped:
                    text_parts.append(stripped)
                continue
            if not isinstance(part, dict):
                continue

            part_type = str(part.get("type") or "").strip().lower()
            if part_type in {"text", "input_text", "output_text"}:
                text_value = _block_text_value(part)
                if text_value:
                    text_parts.append(text_value)
                continue

            if part_type == "tool_use" and role == "assistant":
                _flush_text_parts()
                tool_name = part.get("name")
                if not isinstance(tool_name, str) or not tool_name:
                    fallback_text = _flatten_message_text(part)
                    if fallback_text:
                        text_parts.append(fallback_text)
                    continue
                call_id = part.get("tool_use_id") or part.get("id") or str(uuid4())
                items.append(
                    {
                        "type": "function_call",
                        "call_id": str(call_id),
                        "name": tool_name,
                        "arguments": _serialize_function_arguments(part.get("input")),
                    }
                )
                continue

            if part_type == "tool_result":
                _flush_text_parts()
                call_id = part.get("tool_use_id") or part.get("id")
                if call_id:
                    output_text = _block_text_value(part) or ""
                    items.append(
                        {
                            "type": "function_call_output",
                            "call_id": str(call_id),
                            "output": output_text,
                        }
                    )
                else:
                    fallback_text = _block_text_value(part) or _flatten_message_text(part)
                    if fallback_text:
                        text_parts.append(fallback_text)
                continue

            fallback_text = _block_text_value(part) or _flatten_message_text(part)
            if fallback_text:
                text_parts.append(fallback_text)

        _flush_text_parts()

    return items


def extract_content_blocks_from_output(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract text and tool_use blocks from Responses API payload output array."""
    blocks: List[Dict[str, Any]] = []
    output = payload.get("output")
    if not isinstance(output, list):
        return blocks

    for item in output:
        if not isinstance(item, dict):
            continue

        item_type = str(item.get("type") or "").lower()
        if item_type == "message":
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = str(part.get("type") or "").lower()
                text_value = part.get("text")
                if part_type in {"output_text", "text"} and isinstance(text_value, str) and text_value:
                    blocks.append({"type": "text", "text": text_value})
                    continue
                if part_type == "refusal":
                    refusal_text = part.get("refusal")
                    if isinstance(refusal_text, str) and refusal_text:
                        blocks.append({"type": "text", "text": refusal_text})
                    continue
                if part_type in {"function_call", "tool_call"}:
                    name = part.get("name")
                    if not isinstance(name, str) or not name:
                        continue
                    call_id = part.get("call_id") or part.get("id") or str(uuid4())
                    args = normalize_tool_args(
                        part.get("arguments")
                        if part.get("arguments") is not None
                        else part.get("input")
                    )
                    blocks.append(
                        {
                            "type": "tool_use",
                            "tool_use_id": str(call_id),
                            "name": name,
                            "input": args,
                        }
                    )
            continue

        if item_type in {"function_call", "tool_call"}:
            name = item.get("name")
            if not isinstance(name, str) or not name:
                continue
            call_id = item.get("call_id") or item.get("id") or str(uuid4())
            args = normalize_tool_args(
                item.get("arguments")
                if item.get("arguments") is not None
                else item.get("input")
            )
            blocks.append(
                {
                    "type": "tool_use",
                    "tool_use_id": str(call_id),
                    "name": name,
                    "input": args,
                }
            )

    return blocks


def parse_sse_json_events(raw_text: str) -> List[Dict[str, Any]]:
    """Parse SSE `data:` payloads into JSON events."""
    events: List[Dict[str, Any]] = []
    if not raw_text:
        return events

    data_lines: List[str] = []
    for line in raw_text.splitlines():
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
            continue
        if line.strip():
            continue
        if not data_lines:
            continue
        payload = "\n".join(data_lines).strip()
        data_lines = []
        if not payload or payload == "[DONE]":
            continue
        try:
            parsed = json.loads(payload)
        except (TypeError, ValueError):
            continue
        if isinstance(parsed, dict):
            events.append(parsed)

    if data_lines:
        payload = "\n".join(data_lines).strip()
        if payload and payload != "[DONE]":
            try:
                parsed = json.loads(payload)
            except (TypeError, ValueError):
                parsed = None
            if isinstance(parsed, dict):
                events.append(parsed)

    return events


def extract_text_usage_from_sse_events(
    events: List[Dict[str, Any]],
) -> Tuple[str, Dict[str, int], Optional[Dict[str, Any]]]:
    """Extract text/usage/final response from Responses SSE events."""
    text_parts: List[str] = []
    usage_tokens: Dict[str, int] = {"input_tokens": 0, "output_tokens": 0}
    final_response: Optional[Dict[str, Any]] = None

    for event in events:
        event_type = str(event.get("type") or "")
        if event_type == "response.output_text.delta":
            delta = event.get("delta")
            if isinstance(delta, str) and delta:
                text_parts.append(delta)
            continue
        if event_type == "response.completed":
            response_obj = event.get("response")
            if isinstance(response_obj, dict):
                final_response = response_obj
                usage_tokens = openai_usage_tokens(response_obj.get("usage"))
            continue

    return "".join(text_parts), usage_tokens, final_response
