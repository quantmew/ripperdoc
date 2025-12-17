"""Conversation compaction utilities for RichUI.

This module handles the /compact command logic, including:
- Extracting tool IDs from messages
- Ensuring tool_use/tool_result pairs are preserved together
- Summarizing conversations
"""

from typing import Any, List, Optional, Set, Tuple, Union

from ripperdoc.utils.messages import (
    UserMessage,
    AssistantMessage,
    ProgressMessage,
    create_user_message,
    create_assistant_message,
)
from ripperdoc.utils.message_compaction import (
    compact_messages,
    estimate_conversation_tokens,
)
from ripperdoc.core.query import query_llm
from ripperdoc.utils.log import get_logger

logger = get_logger()

# Type alias for conversation messages
ConversationMessage = Union[UserMessage, AssistantMessage, ProgressMessage]

# Keep a small window of recent messages alongside the summary after /compact
RECENT_MESSAGES_AFTER_COMPACT = 8


def extract_tool_ids_from_message(msg: ConversationMessage) -> Tuple[Set[str], Set[str]]:
    """Extract tool_use IDs and tool_result IDs from a message.

    Returns:
        A tuple of (tool_use_ids, tool_result_ids) found in the message.
    """
    tool_use_ids: Set[str] = set()
    tool_result_ids: Set[str] = set()

    content = getattr(getattr(msg, "message", None), "content", None)
    if not isinstance(content, list):
        return tool_use_ids, tool_result_ids

    for block in content:
        block_type = getattr(block, "type", None)
        if block_type == "tool_use":
            tool_id = getattr(block, "id", None) or getattr(block, "tool_use_id", None)
            if tool_id:
                tool_use_ids.add(tool_id)
        elif block_type == "tool_result":
            tool_id = getattr(block, "tool_use_id", None) or getattr(block, "id", None)
            if tool_id:
                tool_result_ids.add(tool_id)

    return tool_use_ids, tool_result_ids


def get_complete_tool_pairs_tail(
    messages: List[ConversationMessage], target_count: int
) -> List[ConversationMessage]:
    """Get the last N messages while ensuring complete tool_use/tool_result pairs.

    If the tail contains tool_results without their corresponding tool_use,
    expand the tail backwards to include the tool_use message.
    """
    if target_count <= 0 or not messages:
        return []

    # Start with the basic tail
    tail_start = max(0, len(messages) - target_count)
    tail = messages[tail_start:]

    # Collect all tool_result IDs in the tail
    result_ids_in_tail: Set[str] = set()
    for msg in tail:
        _, result_ids = extract_tool_ids_from_message(msg)
        result_ids_in_tail.update(result_ids)

    # Collect all tool_use IDs in the tail
    use_ids_in_tail: Set[str] = set()
    for msg in tail:
        use_ids, _ = extract_tool_ids_from_message(msg)
        use_ids_in_tail.update(use_ids)

    # Find orphan tool_results (results without their tool_use in tail)
    orphan_result_ids = result_ids_in_tail - use_ids_in_tail

    if not orphan_result_ids:
        return tail

    # Search backwards for messages containing the missing tool_uses
    for i in range(tail_start - 1, -1, -1):
        msg = messages[i]
        use_ids, _ = extract_tool_ids_from_message(msg)
        matched = use_ids & orphan_result_ids
        if matched:
            # Include this message and update tracking
            tail_start = i
            orphan_result_ids -= matched
            use_ids_in_tail.update(use_ids)

            # Also check if this message has tool_results that need their tool_uses
            _, new_result_ids = extract_tool_ids_from_message(msg)
            new_orphans = new_result_ids - use_ids_in_tail
            orphan_result_ids.update(new_orphans)

        if not orphan_result_ids:
            break

    return messages[tail_start:]


class ConversationCompactor:
    """Handles conversation compaction operations."""

    def __init__(
        self,
        console: Any,
        render_transcript_fn: Any,
        extract_assistant_text_fn: Any,
    ):
        """Initialize the compactor.

        Args:
            console: Rich console for output
            render_transcript_fn: Function to render messages as transcript text
            extract_assistant_text_fn: Function to extract text from assistant response
        """
        self.console = console
        self._render_transcript = render_transcript_fn
        self._extract_assistant_text = extract_assistant_text_fn

    async def compact(
        self,
        messages: List[ConversationMessage],
        custom_instructions: str,
        protocol: str = "anthropic",
    ) -> Optional[List[ConversationMessage]]:
        """Compact conversation history.

        Returns:
            New compacted conversation list, or None if compaction failed.
        """
        from rich.markup import escape
        from ripperdoc.cli.ui.spinner import Spinner

        if len(messages) < 2:
            self.console.print("[yellow]Not enough conversation history to compact.[/yellow]")
            return None

        tokens_before = estimate_conversation_tokens(messages, protocol=protocol)

        compaction = compact_messages(messages, protocol=protocol)
        messages_for_summary = compaction.messages

        spinner = Spinner(self.console, "Summarizing conversation...", spinner="dots")
        summary_text = ""
        try:
            spinner.start()
            summary_text = await self._summarize_conversation(
                messages_for_summary, custom_instructions
            )
        except Exception as e:
            import traceback
            self.console.print(f"[red]Error during compaction: {escape(str(e))}[/red]")
            logger.warning(
                "[compaction] Error during manual compaction: %s: %s\n%s",
                type(e).__name__, e, traceback.format_exc(),
            )
            self.console.print(f"[dim red]{traceback.format_exc()}[/dim red]")
            return None
        finally:
            spinner.stop()

        if not summary_text:
            logger.warning(
                "[compaction] _summarize_conversation returned empty/None. "
                "transcript may be empty or LLM returned nothing."
            )
            self.console.print("[red]Failed to summarize conversation for compaction.[/red]")
            return None

        if summary_text.strip() == "":
            self.console.print("[red]Summarization returned empty content; aborting compaction.[/red]")
            return None

        summary_message = create_assistant_message(
            f"Conversation summary (generated by /compact):\n{summary_text}"
        )
        non_progress_messages = [
            m for m in messages_for_summary if getattr(m, "type", "") != "progress"
        ]
        # Ensure tool_use/tool_result pairs are kept together
        recent_tail = get_complete_tool_pairs_tail(
            non_progress_messages, RECENT_MESSAGES_AFTER_COMPACT
        )
        new_conversation = [
            create_user_message(
                "Conversation compacted. Summary plus recent turns are kept; older tool output may "
                "be cleared."
            ),
            summary_message,
            *recent_tail,
        ]

        tokens_after = estimate_conversation_tokens(new_conversation, protocol=protocol)
        tokens_saved = max(0, tokens_before - tokens_after)
        self.console.print(
            f"[green]✓ Conversation compacted[/green] "
            f"(saved ~{tokens_saved} tokens). Use /resume to restore full history."
        )

        return new_conversation

    async def _summarize_conversation(
        self,
        messages: List[ConversationMessage],
        custom_instructions: str,
    ) -> str:
        """Summarize the given conversation using the configured model."""
        # Keep transcript bounded to recent turns to avoid blowing context.
        recent_messages = messages[-40:]
        transcript = self._render_transcript(recent_messages)

        logger.debug(
            "[compaction] _summarize_conversation: %d messages, transcript length=%d",
            len(recent_messages), len(transcript) if transcript else 0,
        )

        if not transcript.strip():
            logger.warning("[compaction] transcript is empty, cannot summarize")
            return ""

        instructions = (
            "You are a helpful assistant summarizing the prior conversation. "
            "Produce a concise bullet-list summary covering key decisions, important context, "
            "commands run, files touched, and pending TODOs. Include blockers or open questions. "
            "Keep it brief."
        )
        if custom_instructions.strip():
            instructions += f"\nCustom instructions: {custom_instructions.strip()}"

        user_content = (
            f"Summarize the following conversation between a user and an assistant:\n\n{transcript}"
        )

        logger.debug("[compaction] calling query_llm for summarization...")
        assistant_response = await query_llm(
            messages=[{"role": "user", "content": user_content}],
            system_prompt=instructions,
            tools=[],
            max_thinking_tokens=0,
            model="main",
        )

        logger.debug(
            "[compaction] query_llm returned: type=%s",
            type(assistant_response).__name__,
        )

        result = self._extract_assistant_text(assistant_response)
        logger.debug(
            "[compaction] _extract_assistant_text returned: length=%d",
            len(result) if result else 0,
        )
        return result
