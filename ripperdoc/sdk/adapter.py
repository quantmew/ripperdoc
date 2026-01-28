"""Message and options adapter for Claude SDK compatibility.

This module provides adapters to convert between Ripperdoc's internal
message types and Claude Agent SDK compatible message types.
"""

from __future__ import annotations

from collections.abc import AsyncIterable
from typing import Any, AsyncIterator

from ripperdoc.utils.messages import (
    UserMessage as RipperdocUserMessage,
    AssistantMessage as RipperdocAssistantMessage,
    ProgressMessage as RipperdocProgressMessage,
    MessageContent,
    Message,
)
from ripperdoc.utils.log import get_logger

from .types import (
    Message,
    UserMessage,
    AssistantMessage,
    SystemMessage,
    ResultMessage,
    StreamEvent,
    ContentBlock,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    ToolResultBlock,
)

logger = get_logger()


# =============================================================================
# Message Adapter
# =============================================================================

class MessageAdapter:
    """Adapter for converting between Ripperdoc and Claude SDK message types.

    Ripperdoc uses an internal message structure (with nested Message objects),
    while Claude SDK uses a flat ContentBlock-based structure. This adapter
    bridges the gap between the two formats.
    """

    @staticmethod
    def to_claude_message(
        msg: RipperdocUserMessage | RipperdocAssistantMessage | RipperdocProgressMessage,
        model: str | None = None,
        session_id: str | None = None,
    ) -> Message:
        """Convert a Ripperdoc message to a Claude SDK compatible message.

        Args:
            msg: The Ripperdoc message to convert
            model: The model name (for AssistantMessage)
            session_id: The session ID (for ResultMessage)

        Returns:
            A Claude SDK compatible Message
        """
        msg_type = getattr(msg, "type", None)

        # Use isinstance() checks as fallback when type field is not available
        # This handles cases where Pydantic models might have type field serialization issues
        if msg_type == "user" or (msg_type is None and isinstance(msg, RipperdocUserMessage)):
            return MessageAdapter._user_to_claude(msg)
        elif msg_type == "assistant" or (msg_type is None and isinstance(msg, RipperdocAssistantMessage)):
            return MessageAdapter._assistant_to_claude(msg, model or "")
        elif msg_type == "progress" or (msg_type is None and isinstance(msg, RipperdocProgressMessage)):
            # Progress messages don't have a direct Claude SDK equivalent
            # We could convert them to SystemMessage or skip them
            return MessageAdapter._progress_to_claude(msg)
        else:
            logger.warning(
                f"[MessageAdapter] Unknown message type: {msg_type}, "
                f"converting to SystemMessage"
            )
            return SystemMessage(
                subtype="unknown",
                data={"original_message": str(msg)},
            )

    @staticmethod
    def _user_to_claude(msg: RipperdocUserMessage) -> UserMessage:
        """Convert Ripperdoc UserMessage to Claude SDK UserMessage."""
        content: str | list[ContentBlock] = []
        has_blocks = False

        # Get the nested Message object
        inner_msg = getattr(msg, "message", None)
        if inner_msg is None:
            # Fallback to simple string content
            return UserMessage(
                content=getattr(msg, "content", str(msg)),
                uuid=getattr(msg, "uuid", None),
                parent_tool_use_id=getattr(msg, "parent_tool_use_id", None),
                tool_use_result=getattr(msg, "tool_use_result", None),
            )

        inner_content = getattr(inner_msg, "content", None)
        if isinstance(inner_content, str):
            content = inner_content
        elif isinstance(inner_content, list):
            # Convert MessageContent list to ContentBlock list
            content = []
            for item in inner_content:
                if isinstance(item, MessageContent):
                    block = MessageAdapter._content_to_block(item)
                    if block:
                        content.append(block)
                        has_blocks = True
                elif isinstance(item, dict):
                    # Handle dict-style content
                    block = MessageAdapter._dict_to_block(item)
                    if block:
                        content.append(block)
                        has_blocks = True

            if not has_blocks and not content:
                # Fallback to string if no blocks were converted
                content = str(inner_msg.content or "")

        return UserMessage(
            content=content,
            uuid=getattr(msg, "uuid", None),
            parent_tool_use_id=getattr(msg, "parent_tool_use_id", None),
            tool_use_result=getattr(msg, "tool_use_result", None),
        )

    @staticmethod
    def _assistant_to_claude(
        msg: RipperdocAssistantMessage, model: str
    ) -> AssistantMessage:
        """Convert Ripperdoc AssistantMessage to Claude SDK AssistantMessage."""
        content: list[ContentBlock] = []

        # Get the nested Message object
        inner_msg = getattr(msg, "message", None)
        if inner_msg is None:
            # Empty content but still valid
            return AssistantMessage(
                content=[],
                model=model,
                parent_tool_use_id=getattr(msg, "parent_tool_use_id", None),
                error=getattr(msg, "error", None),
            )

        inner_content = getattr(inner_msg, "content", None)
        if isinstance(inner_content, list):
            for item in inner_content:
                if isinstance(item, MessageContent):
                    block = MessageAdapter._content_to_block(item)
                    if block:
                        content.append(block)
                elif isinstance(item, dict):
                    block = MessageAdapter._dict_to_block(item)
                    if block:
                        content.append(block)
        elif isinstance(inner_content, str):
            content = [TextBlock(text=inner_content)]

        return AssistantMessage(
            content=content,
            model=model,
            parent_tool_use_id=getattr(msg, "parent_tool_use_id", None),
            error=getattr(msg, "is_api_error_message", None),
        )

    @staticmethod
    def _progress_to_claude(msg: RipperdocProgressMessage) -> SystemMessage:
        """Convert Ripperdoc ProgressMessage to Claude SDK SystemMessage."""
        return SystemMessage(
            subtype="progress",
            data={
                "tool_use_id": getattr(msg, "tool_use_id", ""),
                "content": getattr(msg, "content", None),
            },
        )

    @staticmethod
    def _content_to_block(content: MessageContent) -> ContentBlock | None:
        """Convert a MessageContent to a ContentBlock."""
        content_type = getattr(content, "type", None)

        if content_type == "text" or content_type is None:
            return TextBlock(text=getattr(content, "text", ""))

        elif content_type == "thinking":
            return ThinkingBlock(
                thinking=getattr(content, "thinking", ""),
                signature=getattr(content, "signature", ""),
            )

        elif content_type == "redacted_thinking":
            # Map redacted_thinking to ThinkingBlock with signature
            return ThinkingBlock(
                thinking=getattr(content, "data", ""),
                signature=getattr(content, "signature", ""),
            )

        elif content_type == "tool_use":
            return ToolUseBlock(
                id=getattr(content, "id", "")
                or getattr(content, "tool_use_id", "")
                or "",
                name=getattr(content, "name", ""),
                input=getattr(content, "input", {}) or {},
            )

        elif content_type == "tool_result":
            return ToolResultBlock(
                tool_use_id=getattr(content, "tool_use_id", "")
                or getattr(content, "id", "")
                or "",
                content=getattr(content, "text", None),
                is_error=getattr(content, "is_error", None),
            )

        elif content_type == "image":
            # Image blocks don't have a direct ContentBlock equivalent
            # Convert to a text block for now
            return TextBlock(text=f"[Image: {getattr(content, 'media_type', 'image')}]")

        else:
            logger.warning(
                f"[MessageAdapter] Unknown content type: {content_type}, "
                f"converting to TextBlock"
            )
            return TextBlock(text=str(content))

    @staticmethod
    def _dict_to_block(d: dict[str, Any]) -> ContentBlock | None:
        """Convert a dict to a ContentBlock."""
        block_type = d.get("type")

        if block_type == "text":
            return TextBlock(text=d.get("text", ""))

        elif block_type == "thinking":
            return ThinkingBlock(
                thinking=d.get("thinking", ""),
                signature=d.get("signature", ""),
            )

        elif block_type == "redacted_thinking":
            return ThinkingBlock(
                thinking=d.get("data", ""),
                signature=d.get("signature", ""),
            )

        elif block_type == "tool_use":
            return ToolUseBlock(
                id=d.get("id", "") or d.get("tool_use_id", "") or "",
                name=d.get("name", ""),
                input=d.get("input", {}) or {},
            )

        elif block_type == "tool_result":
            return ToolResultBlock(
                tool_use_id=d.get("tool_use_id", "") or d.get("id", "") or "",
                content=d.get("text", None) or d.get("content", None),
                is_error=d.get("is_error", None),
            )

        else:
            logger.warning(
                f"[MessageAdapter] Unknown dict block type: {block_type}, "
                f"converting to TextBlock"
            )
            return TextBlock(text=str(d))

    @staticmethod
    def from_claude_message(msg: Message) -> RipperdocUserMessage | RipperdocAssistantMessage:
        """Convert a Claude SDK message to a Ripperdoc message.

        This is the inverse of to_claude_message.

        Args:
            msg: The Claude SDK compatible message to convert

        Returns:
            A Ripperdoc internal message
        """
        if isinstance(msg, UserMessage):
            return MessageAdapter._claude_user_to_ripperdoc(msg)
        elif isinstance(msg, AssistantMessage):
            return MessageAdapter._claude_assistant_to_ripperdoc(msg)
        else:
            # SystemMessage, ResultMessage, StreamEvent don't have
            # direct Ripperdoc equivalents
            logger.warning(
                f"[MessageAdapter] Cannot convert {type(msg).__name__} "
                f"to Ripperdoc message, returning empty AssistantMessage"
            )
            return RipperdocAssistantMessage(
                message=Message(role="assistant", content=""),
            )

    @staticmethod
    def _claude_user_to_ripperdoc(msg: UserMessage) -> RipperdocUserMessage:
        """Convert Claude SDK UserMessage to Ripperdoc UserMessage."""
        content = msg.content

        # Convert ContentBlock list to MessageContent list
        if isinstance(content, list):
            message_contents = []
            for block in content:
                mc = MessageAdapter._block_to_content(block)
                if mc:
                    message_contents.append(mc)

            inner_msg = Message(role="user", content=message_contents)
        else:
            inner_msg = Message(role="user", content=content)

        return RipperdocUserMessage(
            message=inner_msg,
            uuid=msg.uuid,
            tool_use_result=msg.tool_use_result,
        )

    @staticmethod
    def _claude_assistant_to_ripperdoc(msg: AssistantMessage) -> RipperdocAssistantMessage:
        """Convert Claude SDK AssistantMessage to Ripperdoc AssistantMessage."""
        message_contents = []

        for block in msg.content:
            mc = MessageAdapter._block_to_content(block)
            if mc:
                message_contents.append(mc)

        inner_msg = Message(
            role="assistant",
            content=message_contents if message_contents else "",
        )

        return RipperdocAssistantMessage(
            message=inner_msg,
            model=msg.model,
            is_api_error_message=bool(msg.error),
        )

    @staticmethod
    def _block_to_content(block: ContentBlock) -> MessageContent | None:
        """Convert a ContentBlock to a MessageContent."""
        if isinstance(block, TextBlock):
            return MessageContent(type="text", text=block.text)

        elif isinstance(block, ThinkingBlock):
            return MessageContent(
                type="thinking",
                thinking=block.thinking,
                signature=block.signature,
            )

        elif isinstance(block, ToolUseBlock):
            return MessageContent(
                type="tool_use",
                id=block.id,
                name=block.name,
                input=block.input,
            )

        elif isinstance(block, ToolResultBlock):
            return MessageContent(
                type="tool_result",
                tool_use_id=block.tool_use_id,
                text=block.content if isinstance(block.content, str) else None,
                is_error=block.is_error,
            )

        else:
            logger.warning(
                f"[MessageAdapter] Unknown block type: {type(block).__name__}"
            )
            return None


# =============================================================================
# Async Message Stream Adapter
# =============================================================================

class AsyncMessageAdapter:
    """Async iterator that adapts Ripperdoc messages to Claude SDK messages."""

    def __init__(
        self,
        source: AsyncIterable[RipperdocUserMessage | RipperdocAssistantMessage | RipperdocProgressMessage],
        model: str | None = None,
    ):
        """Initialize the adapter.

        Args:
            source: The source async iterable of Ripperdoc messages
            model: The model name to use for AssistantMessage
        """
        self._source = source
        self._model = model

    def __aiter__(self) -> AsyncIterator[Message]:
        """Return self as an async iterator."""
        return self._adapt()

    async def _adapt(self) -> AsyncIterator[Message]:
        """Adapt messages from the source."""
        async for msg in self._source:
            yield MessageAdapter.to_claude_message(msg, self._model)


# =============================================================================
# ResultMessage Factory
# =============================================================================

class ResultMessageFactory:
    """Factory for creating ResultMessage objects.

    Ripperdoc doesn't have native ResultMessage, so we create them
    at the end of queries for compatibility.
    """

    @staticmethod
    def create(
        session_id: str,
        duration_ms: int,
        duration_api_ms: int = 0,
        is_error: bool = False,
        num_turns: int = 1,
        total_cost_usd: float | None = None,
        usage: dict[str, Any] | None = None,
        result: str | None = None,
    ) -> ResultMessage:
        """Create a ResultMessage.

        Args:
            session_id: The session identifier
            duration_ms: Total duration in milliseconds
            duration_api_ms: API duration in milliseconds
            is_error: Whether an error occurred
            num_turns: Number of conversation turns
            total_cost_usd: Total cost in USD
            usage: Token usage information
            result: Result text

        Returns:
            A ResultMessage instance
        """
        return ResultMessage(
            subtype="result",
            duration_ms=duration_ms,
            duration_api_ms=duration_api_ms,
            is_error=is_error,
            num_turns=num_turns,
            session_id=session_id,
            total_cost_usd=total_cost_usd,
            usage=usage,
            result=result,
        )
