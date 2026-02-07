"""Message conversion helpers for stdio protocol handler."""

from __future__ import annotations

from typing import Any

from ripperdoc.core.message_utils import resolve_model_profile
from ripperdoc.protocol.models import (
    AssistantMessageData,
    AssistantStreamMessage,
    UserMessageData,
    UserStreamMessage,
    model_to_dict,
)


class StdioMessageMixin:
    def _convert_message_to_sdk(self, message: Any) -> dict[str, Any] | None:
        """Convert internal message to SDK format.

        Args:
            message: The internal message object.

        Returns:
            A dictionary in SDK message format, or None if message should be skipped.
        """
        msg_type = getattr(message, "type", None)

        # Filter out progress messages (internal implementation detail)
        if msg_type == "progress":
            return None

        if msg_type == "assistant":
            content_blocks = []
            msg_content = getattr(message, "message", None)
            if msg_content:
                content = getattr(msg_content, "content", None)
                if content:
                    if isinstance(content, str):
                        content_blocks.append({"type": "text", "text": content})
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict):
                                if block.get("type") == "tool_use":
                                    content_blocks.append(self._normalize_tool_use_block(block))
                                else:
                                    content_blocks.append(block)
                            else:
                                # Convert MessageContent (Pydantic model) to dict
                                block_dict = self._convert_content_block(block)
                                if block_dict:
                                    content_blocks.append(block_dict)

            # Resolve model pointer to actual model name for SDK
            # The message may have model=None (unset), so fall back to QueryContext.model
            # Then resolve any pointer (e.g., "main") to the actual model name
            model_pointer = getattr(message, "model", None) or (
                self._query_context.model if self._query_context else None
            )  # type: ignore[union-attr]
            model_profile = resolve_model_profile(
                str(model_pointer) if model_pointer else "claude-opus-4-5-20251101"
            )
            actual_model = (
                model_profile.model
                if model_profile
                else (model_pointer or "claude-opus-4-5-20251101")
            )

            stream_message = AssistantStreamMessage(
                message=AssistantMessageData(
                    content=content_blocks,
                    model=actual_model,
                ),
                session_id=self._session_id,
                parent_tool_use_id=getattr(message, "parent_tool_use_id", None),
                uuid=getattr(message, "uuid", None),
            )
            return model_to_dict(stream_message)

        if msg_type == "user":
            msg_content = getattr(message, "message", None)
            content = getattr(msg_content, "content", "") if msg_content else ""
            tool_result_text: str | None = None
            tool_result_is_error = False

            # If content is a list of MessageContent objects (e.g., tool results),
            # convert it to SDK content blocks
            if isinstance(content, list):
                content_blocks = []
                for block in content:
                    if isinstance(block, dict):
                        block_type = block.get("type", "")
                        if block_type == "tool_result":
                            tool_use_id = block.get("tool_use_id") or block.get("id") or ""
                            text_value = self._normalize_tool_result_text(
                                block.get("text"), block.get("content")
                            )
                            normalized_block: dict[str, Any] = {
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": text_value,
                            }
                            if "is_error" in block:
                                normalized_block["is_error"] = block.get("is_error")
                                tool_result_is_error = bool(block.get("is_error"))
                            content_blocks.append(normalized_block)
                            if tool_result_text is None:
                                tool_result_text = str(text_value) if text_value is not None else ""
                        else:
                            if block_type == "tool_use":
                                content_blocks.append(self._normalize_tool_use_block(block))
                            else:
                                content_blocks.append(block)
                    else:
                        block_dict = self._convert_content_block(block)
                        if block_dict:
                            if block_dict.get("type") == "tool_result":
                                if tool_result_text is None:
                                    tool_result_text = str(block_dict.get("content") or "")
                                tool_result_is_error = bool(block_dict.get("is_error", False))
                            content_blocks.append(block_dict)
                content = content_blocks

            stream_message: UserStreamMessage | AssistantStreamMessage = UserStreamMessage(  # type: ignore[assignment,no-redef]
                message=UserMessageData(content=content),
                uuid=getattr(message, "uuid", None),
                session_id=self._session_id,
                parent_tool_use_id=getattr(message, "parent_tool_use_id", None),
                tool_use_result=(
                    (
                        self._format_tool_use_result(tool_result_text, tool_result_is_error)
                        if isinstance(content, list)
                        and tool_result_is_error
                        and tool_result_text is not None
                        else self._sanitize_for_json(getattr(message, "tool_use_result", None))
                    )
                ),
            )
            return model_to_dict(stream_message)

        # Unknown message type - return None to skip
        return None

    def _sanitize_for_json(self, obj: Any) -> Any:
        """Recursively convert objects to JSON-serializable types.

        This function ensures Pydantic models and other objects are converted
        to dictionaries/lists/primitives that can be JSON serialized.

        Args:
            obj: The object to sanitize.

        Returns:
            A JSON-serializable version of the object.
        """
        # None values
        if obj is None:
            return None

        # Primitives
        if isinstance(obj, (str, int, float, bool)):
            return obj

        # Lists and tuples
        if isinstance(obj, (list, tuple)):
            return [self._sanitize_for_json(item) for item in obj]

        # Dictionaries
        if isinstance(obj, dict):
            return {key: self._sanitize_for_json(value) for key, value in obj.items()}

        # Pydantic models
        if hasattr(obj, "model_dump"):
            try:
                dumped = obj.model_dump(exclude_none=True)
                return self._sanitize_for_json(dumped)
            except Exception:
                pass

        # Objects with dict() method
        if hasattr(obj, "dict"):
            try:
                dumped = obj.dict(exclude_none=True)
                return self._sanitize_for_json(dumped)
            except Exception:
                pass

        # Fallback: try to convert to string
        try:
            return str(obj)
        except Exception:
            return None

    def _normalize_tool_result_text(self, text_value: Any, content_value: Any) -> str:
        if isinstance(text_value, str):
            return text_value
        if isinstance(content_value, str):
            return content_value
        if isinstance(content_value, list):
            for item in content_value:
                if isinstance(item, dict) and item.get("type") == "text":
                    return str(item.get("text") or "")
            if content_value:
                return str(content_value[0])
        if content_value is None:
            return ""
        return str(content_value)

    def _format_tool_use_result(self, text_value: str | None, is_error: bool) -> str | None:
        if text_value is None:
            return None
        if is_error and not text_value.startswith("Error: "):
            return f"Error: {text_value}"
        return text_value

    def _summarize_task_prompt(self, prompt: str) -> str:
        line = prompt.strip().splitlines()[0] if prompt else ""
        if len(line) > 120:
            return f"{line[:117]}..."
        return line

    def _normalize_task_tool_input(self, input_data: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(input_data)
        subagent_type = normalized.get("subagent_type")
        if isinstance(subagent_type, str) and subagent_type in ("explore", "plan"):
            normalized["subagent_type"] = subagent_type.title()
        if "description" not in normalized:
            prompt = normalized.get("prompt")
            if isinstance(prompt, str) and prompt.strip():
                normalized["description"] = self._summarize_task_prompt(prompt)
        return normalized

    def _normalize_tool_use_block(self, block: dict[str, Any]) -> dict[str, Any]:
        tool_id = block.get("id") or block.get("tool_use_id") or ""
        name = block.get("name") or ""
        input_value = block.get("input") or {}
        if hasattr(input_value, "model_dump"):
            input_value = input_value.model_dump()
        elif hasattr(input_value, "dict"):
            input_value = input_value.dict()
        if not isinstance(input_value, dict):
            input_value = {"value": str(input_value)}
        if name == "Task":
            input_value = self._normalize_task_tool_input(input_value)
        normalized = dict(block)
        normalized.update(
            {
                "type": "tool_use",
                "id": tool_id,
                "name": name,
                "input": input_value,
            }
        )
        return normalized

    def _convert_content_block(self, block: Any) -> dict[str, Any] | None:
        """Convert a MessageContent block to dictionary.

        Uses the same logic as _content_block_to_api in messages.py
        to ensure consistency and proper field mapping.

        Args:
            block: The MessageContent object.

        Returns:
            A dictionary representation of the block.
        """
        block_type = getattr(block, "type", None)

        if block_type == "text":
            return {
                "type": "text",
                "text": getattr(block, "text", None) or "",
            }

        if block_type == "thinking":
            return {
                "type": "thinking",
                "thinking": getattr(block, "thinking", None) or getattr(block, "text", None) or "",
                "signature": getattr(block, "signature", None),
            }

        if block_type == "tool_use":
            # Use the same id extraction logic as _content_block_to_api
            # Try id first, then tool_use_id, then empty string
            tool_id = getattr(block, "id", None) or getattr(block, "tool_use_id", None) or ""
            name = getattr(block, "name", None) or ""
            input_value = getattr(block, "input", None) or {}
            if hasattr(input_value, "model_dump"):
                input_value = input_value.model_dump()
            elif hasattr(input_value, "dict"):
                input_value = input_value.dict()
            if not isinstance(input_value, dict):
                input_value = {"value": str(input_value)}
            if name == "Task" and isinstance(input_value, dict):
                input_value = self._normalize_task_tool_input(input_value)
            return {
                "type": "tool_use",
                "id": tool_id,
                "name": name,
                "input": input_value,
            }

        if block_type == "tool_result":
            text_value = (
                getattr(block, "text", None)
                or self._normalize_tool_result_text(None, getattr(block, "content", None))
                or ""
            )
            result_block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": getattr(block, "tool_use_id", None)
                or getattr(block, "id", None)
                or "",
                "content": text_value,
            }
            if getattr(block, "is_error", None) is not None:
                result_block["is_error"] = getattr(block, "is_error", None)
            return result_block

        if block_type == "image":
            return {
                "type": "image",
                "source": {
                    "type": getattr(block, "source_type", None) or "base64",
                    "media_type": getattr(block, "media_type", None) or "image/jpeg",
                    "data": getattr(block, "image_data", None) or "",
                },
            }

        # Unknown block type - try to convert with generic approach
        block_dict = {}
        if hasattr(block, "type"):
            block_dict["type"] = block.type
        if hasattr(block, "text"):
            block_dict["text"] = block.text
        if hasattr(block, "id"):
            block_dict["id"] = block.id
        if hasattr(block, "name"):
            block_dict["name"] = block.name
        if hasattr(block, "input"):
            block_dict["input"] = block.input
        if hasattr(block, "content"):
            block_dict["content"] = block.content
        if hasattr(block, "is_error"):
            block_dict["is_error"] = block.is_error
        return block_dict if block_dict else None
