"""Message parser for Ripperdoc SDK subprocess architecture.

This module parses JSON messages from the CLI into typed Message objects.
It follows Claude SDK's elegant pattern matching approach.
"""

import logging
from typing import Any

from ripperdoc.sdk._errors import MessageParseError
from ripperdoc.sdk.types import (
    AssistantMessage,
    ContentBlock,
    Message,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

logger = logging.getLogger(__name__)


def parse_message(data: dict[str, Any]) -> Message:
    """Parse message from CLI output into typed Message objects.

    Uses Python's match/case for clean pattern matching.

    Args:
        data: Raw message dictionary from CLI output

    Returns:
        Parsed Message object

    Raises:
        MessageParseError: If parsing fails or message type is unrecognized
    """
    if not isinstance(data, dict):
        raise MessageParseError(
            f"Invalid message data type (expected dict, got {type(data).__name__})",
            data,
        )

    message_type = data.get("type")
    if not message_type:
        raise MessageParseError("Message missing 'type' field", data)

    match message_type:
        case "user":
            try:
                parent_tool_use_id = data.get("parent_tool_use_id")
                tool_use_result = data.get("tool_use_result")
                uuid = data.get("uuid")

                # Parse content blocks
                content = data.get("message", {}).get("content", "")
                if isinstance(content, list):
                    content_blocks: list[ContentBlock] = []
                    for block in content:
                        block_type = block.get("type")
                        match block_type:
                            case "text":
                                content_blocks.append(
                                    TextBlock(text=block.get("text", ""))
                                )
                            case "tool_use":
                                content_blocks.append(
                                    ToolUseBlock(
                                        id=block.get("id", ""),
                                        name=block.get("name", ""),
                                        input=block.get("input", {}) or {},
                                    )
                                )
                            case "tool_result":
                                content_blocks.append(
                                    ToolResultBlock(
                                        tool_use_id=block.get("tool_use_id", ""),
                                        content=block.get("content"),
                                        is_error=block.get("is_error"),
                                    )
                                )
                            case _:
                                logger.warning(f"Unknown content block type: {block_type}")

                    return UserMessage(
                        content=content_blocks if content_blocks else content,
                        uuid=uuid,
                        parent_tool_use_id=parent_tool_use_id,
                        tool_use_result=tool_use_result,
                    )
                else:
                    # String content
                    return UserMessage(
                        content=content,
                        uuid=uuid,
                        parent_tool_use_id=parent_tool_use_id,
                        tool_use_result=tool_use_result,
                    )

            except KeyError as e:
                raise MessageParseError(
                    f"Missing required field in user message: {e}", data
                ) from e

        case "assistant":
            try:
                content_blocks: list[ContentBlock] = []
                for block in data.get("message", {}).get("content", []):
                    block_type = block.get("type")
                    match block_type:
                        case "text":
                            content_blocks.append(
                                TextBlock(text=block.get("text", ""))
                            )
                        case "thinking":
                            content_blocks.append(
                                ThinkingBlock(
                                    thinking=block.get("thinking", ""),
                                    signature=block.get("signature", ""),
                                )
                            )
                        case "tool_use":
                            content_blocks.append(
                                ToolUseBlock(
                                    id=block.get("id", ""),
                                    name=block.get("name", ""),
                                    input=block.get("input", {}) or {},
                                )
                            )
                        case "tool_result":
                            content_blocks.append(
                                ToolResultBlock(
                                    tool_use_id=block.get("tool_use_id", ""),
                                    content=block.get("content"),
                                    is_error=block.get("is_error"),
                                )
                            )
                        case _:
                            logger.warning(f"Unknown content block type: {block_type}")

                return AssistantMessage(
                    content=content_blocks,
                    model=data.get("message", {}).get("model", ""),
                    parent_tool_use_id=data.get("parent_tool_use_id"),
                    error=data.get("message", {}).get("error"),
                )

            except KeyError as e:
                raise MessageParseError(
                    f"Missing required field in assistant message: {e}", data
                ) from e

        case "progress":
            # Progress messages
            return SystemMessage(
                subtype="progress",
                data={
                    "tool_use_id": data.get("tool_use_id", ""),
                    "content": data.get("content"),
                },
            )

        case "result":
            try:
                return ResultMessage(
                    subtype="result",
                    duration_ms=data.get("duration_ms", 0),
                    duration_api_ms=data.get("duration_api_ms", 0),
                    is_error=data.get("is_error", False),
                    num_turns=data.get("num_turns", 0),
                    session_id=data.get("session_id", ""),
                    total_cost_usd=data.get("total_cost_usd"),
                    usage=data.get("usage"),
                    result=data.get("result"),
                )
            except KeyError as e:
                raise MessageParseError(
                    f"Missing required field in result message: {e}", data
                ) from e

        case "system":
            try:
                return SystemMessage(
                    subtype=data.get("subtype", ""),
                    data=data,
                )
            except KeyError as e:
                raise MessageParseError(
                    f"Missing required field in system message: {e}", data
                ) from e

        case _:
            raise MessageParseError(f"Unknown message type: {message_type}", data)


__all__ = ["parse_message"]
