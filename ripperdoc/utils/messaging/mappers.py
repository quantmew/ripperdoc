"""Message format conversion mappers."""

from __future__ import annotations

from typing import Any, Dict
from uuid import uuid4

from ripperdoc.utils.log import get_logger
from ripperdoc.utils.messaging.types.content import MessageContent

logger = get_logger()


def _content_block_to_api(block: MessageContent) -> Dict[str, Any]:
    """Convert a MessageContent block to API-ready dict for tool protocols."""

    def _to_plain_json(value: Any) -> Any:
        if value is None:
            return None
        if hasattr(value, "model_dump"):
            try:
                value = value.model_dump(mode="json")
            except (TypeError, ValueError):
                value = value.model_dump()
        elif hasattr(value, "dict"):
            value = value.dict()
        if isinstance(value, list):
            return [_to_plain_json(item) for item in value]
        if isinstance(value, tuple):
            return [_to_plain_json(item) for item in value]
        if isinstance(value, dict):
            return {str(key): _to_plain_json(item) for key, item in value.items()}
        return value

    block_type = getattr(block, "type", None)
    if block_type == "thinking":
        return {
            "type": "thinking",
            "thinking": getattr(block, "thinking", None) or getattr(block, "text", None) or "",
            "signature": getattr(block, "signature", None),
        }
    if block_type == "redacted_thinking":
        return {
            "type": "redacted_thinking",
            "data": getattr(block, "data", None) or getattr(block, "text", None) or "",
            "signature": getattr(block, "signature", None),
        }
    if block_type == "tool_use":
        input_value = getattr(block, "input", None) or {}
        # Ensure input is a dict, not a Pydantic model
        if hasattr(input_value, "model_dump"):
            input_value = input_value.model_dump()
        elif hasattr(input_value, "dict"):
            input_value = input_value.dict()
        elif not isinstance(input_value, dict):
            input_value = {"value": str(input_value)}
        return {
            "type": "tool_use",
            "id": getattr(block, "id", None) or getattr(block, "tool_use_id", "") or "",
            "name": getattr(block, "name", None) or "",
            "input": input_value,
        }
    if block_type == "server_tool_use":
        input_value = getattr(block, "input", None) or {}
        if hasattr(input_value, "model_dump"):
            input_value = input_value.model_dump()
        elif hasattr(input_value, "dict"):
            input_value = input_value.dict()
        elif not isinstance(input_value, dict):
            input_value = {"value": str(input_value)}
        return {
            "type": "server_tool_use",
            "id": getattr(block, "id", None) or getattr(block, "tool_use_id", "") or "",
            "name": getattr(block, "name", None) or "",
            "input": input_value,
        }
    if block_type == "tool_search_tool_result":
        payload = _to_plain_json(getattr(block, "content", None))
        if payload is None:
            payload = {}
        return {
            "type": "tool_search_tool_result",
            "tool_use_id": getattr(block, "tool_use_id", None) or getattr(block, "id", None) or "",
            "content": payload,
        }
    if block_type == "tool_reference":
        return {
            "type": "tool_reference",
            "tool_name": getattr(block, "tool_name", None) or getattr(block, "name", None) or "",
        }
    if block_type == "tool_result":
        content_value = _to_plain_json(getattr(block, "content", None))
        if content_value is None:
            content_value = [
                {
                    "type": "text",
                    "text": getattr(block, "text", None) or "",
                }
            ]
        elif isinstance(content_value, str):
            content_value = [{"type": "text", "text": content_value}]
        elif isinstance(content_value, dict):
            content_value = [content_value]
        result: Dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": getattr(block, "tool_use_id", None) or getattr(block, "id", None) or "",
            "content": content_value,
        }
        if getattr(block, "is_error", None) is not None:
            result["is_error"] = block.is_error
        return result
    if block_type == "image":
        return {
            "type": "image",
            "source": {
                "type": getattr(block, "source_type", None) or "base64",
                "media_type": getattr(block, "media_type", None) or "image/jpeg",
                "data": getattr(block, "image_data", None) or "",
            },
        }
    # Default to text block
    return {
        "type": "text",
        "text": getattr(block, "text", None) or getattr(block, "content", None) or str(block),
    }


def _content_block_to_openai(block: MessageContent) -> Dict[str, Any]:
    """Convert a MessageContent block to OpenAI chat-completions tool call format."""
    block_type = getattr(block, "type", None)
    if block_type in {"server_tool_use", "tool_search_tool_result", "tool_reference"}:
        # Anthropic-specific tool-search blocks are not valid OpenAI messages.
        return {}
    if block_type == "tool_use":
        import json

        args = getattr(block, "input", None) or {}
        try:
            args_str = json.dumps(args)
        except (TypeError, ValueError) as exc:
            logger.warning(
                "[_content_block_to_openai] Failed to serialize tool arguments: %s: %s",
                type(exc).__name__,
                exc,
            )
            args_str = "{}"
        tool_call_id = (
            getattr(block, "id", None) or getattr(block, "tool_use_id", "") or str(uuid4())
        )
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": getattr(block, "name", None) or "",
                        "arguments": args_str,
                    },
                }
            ],
        }
    if block_type == "tool_result":
        # OpenAI expects role=tool messages after a tool call
        tool_call_id = getattr(block, "tool_use_id", None) or getattr(block, "id", None) or ""
        if not tool_call_id:
            logger.debug("[_content_block_to_openai] Skipping tool_result without tool_call_id")
            return {}
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": getattr(block, "text", None) or getattr(block, "content", None) or "",
        }
    if block_type == "image":
        # OpenAI uses data URL format for images
        media_type = getattr(block, "media_type", None) or "image/jpeg"
        image_data = getattr(block, "image_data", None) or ""
        data_url = f"data:{media_type};base64,{image_data}"
        return {
            "type": "image_url",
            "image_url": {"url": data_url},
        }
    # Fallback text message
    return {
        "role": "assistant",
        "content": getattr(block, "text", None) or getattr(block, "content", None) or str(block),
    }
