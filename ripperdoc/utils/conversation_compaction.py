"""Conversation compaction (auto and manual)"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Set, Tuple, Union

from ripperdoc.core.query import query_llm
from ripperdoc.utils.log import get_logger
from ripperdoc.utils.message_compaction import (
    estimate_conversation_tokens,
    micro_compact_messages,
)
from ripperdoc.utils.message_formatting import (
    render_transcript,
    extract_assistant_text,
)
from ripperdoc.utils.messages import (
    AssistantMessage,
    ProgressMessage,
    UserMessage,
    create_user_message,
)

logger = get_logger()

ConversationMessage = Union[UserMessage, AssistantMessage, ProgressMessage]

RECENT_MESSAGES_AFTER_COMPACT = 8


# ─────────────────────────────────────────────────────────────────────────────
# Summary Prompt Generation
# ─────────────────────────────────────────────────────────────────────────────


def generate_summary_prompt(additional_instructions: Optional[str] = None) -> str:
    """Generate the system prompt for conversation summarization.

    This prompt guides the model to create a detailed, structured summary
    that preserves technical details essential for continuing development.

    Args:
        additional_instructions: Optional custom instructions to append.

    Returns:
        The complete summary prompt string.
    """
    base_prompt = """Your task is to create a detailed summary of the conversation so far, paying close attention to the user's explicit requests and your previous actions.
This summary should be thorough in capturing technical details, code patterns, and architectural decisions that would be essential for continuing development work without losing context.

Before providing your final summary, wrap your analysis in <analysis> tags to organize your thoughts and ensure you've covered all necessary points. In your analysis process:

1. Chronologically analyze each message and section of the conversation. For each section thoroughly identify:
   - The user's explicit requests and intents
   - Your approach to addressing the user's requests
   - Key decisions, technical concepts and code patterns
   - Specific details like:
     - file names
     - full code snippets
     - function signatures
     - file edits
   - Errors that you ran into and how you fixed them
   - Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.
2. Double-check for technical accuracy and completeness, addressing each required element thoroughly.

Your summary should include the following sections:

1. Primary Request and Intent: Capture all of the user's explicit requests and intents in detail
2. Key Technical Concepts: List all important technical concepts, technologies, and frameworks discussed.
3. Files and Code Sections: Enumerate specific files and code sections examined, modified, or created. Pay special attention to the most recent messages and include full code snippets where applicable and include a summary of why this file read or edit is important.
4. Errors and fixes: List all errors that you ran into, and how you fixed them. Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.
5. Problem Solving: Document problems solved and any ongoing troubleshooting efforts.
6. All user messages: List ALL user messages that are not tool results. These are critical for understanding the users' feedback and changing intent.
7. Pending Tasks: Outline any pending tasks that you have explicitly been asked to work on.
8. Current Work: Describe in detail precisely what was being worked on immediately before this summary request, paying special attention to the most recent messages from both user and assistant. Include file names and code snippets where applicable.
9. Optional Next Step: List the next step that you will take that is related to the most recent work you were doing. IMPORTANT: ensure that this step is DIRECTLY in line with the user's explicit requests, and the task you were working on immediately before this summary request. If your last task was concluded, then only list next steps if they are explicitly in line with the users request. Do not start on tangential requests without confirming with the user first.
   If there is a next step, include direct quotes from the most recent conversation showing exactly what task you were working on and where you left off. This should be verbatim to ensure there's no drift in task interpretation.

Here's an example of how your output should be structured:

<example>
<analysis>
[Your thought process, ensuring all points are covered thoroughly and accurately]
</analysis>

<summary>
1. Primary Request and Intent:
   [Detailed description]

2. Key Technical Concepts:
   - [Concept 1]
   - [Concept 2]
   - [...]

3. Files and Code Sections:
   - [File Name 1]
      - [Summary of why this file is important]
      - [Summary of the changes made to this file, if any]
      - [Important Code Snippet]
   - [File Name 2]
      - [Important Code Snippet]
   - [...]

4. Errors and fixes:
    - [Detailed description of error 1]:
      - [How you fixed the error]
      - [User feedback on the error if any]
    - [...]

5. Problem Solving:
   [Description of solved problems and ongoing troubleshooting]

6. All user messages:
    - [Detailed non tool use user message]
    - [...]

7. Pending Tasks:
   - [Task 1]
   - [Task 2]
   - [...]

8. Current Work:
   [Precise description of current work]

9. Optional Next Step:
   [Optional Next step to take]

</summary>
</example>

Please provide your summary based on the conversation so far, following this structure and ensuring precision and thoroughness in your response."""

    if additional_instructions and additional_instructions.strip():
        return f"{base_prompt}\n\nAdditional Instructions:\n{additional_instructions.strip()}"

    return base_prompt


def format_summary_response(raw_summary_text: str) -> str:
    """Format the summary response by extracting content from XML tags.

    Converts <analysis>...</analysis> and <summary>...</summary> tags
    to readable section headers.

    Args:
        raw_summary_text: The raw response from the model.

    Returns:
        Formatted summary text with clean section headers.
    """
    formatted_text = raw_summary_text

    # Extract and format analysis section
    analysis_match = re.search(r"<analysis>([\s\S]*?)</analysis>", formatted_text)
    if analysis_match:
        extracted_content = analysis_match.group(1) or ""
        formatted_text = re.sub(
            r"<analysis>[\s\S]*?</analysis>",
            f"Analysis:\n{extracted_content.strip()}",
            formatted_text,
        )

    # Extract and format summary section
    summary_match = re.search(r"<summary>([\s\S]*?)</summary>", formatted_text)
    if summary_match:
        summary_content = summary_match.group(1) or ""
        formatted_text = re.sub(
            r"<summary>[\s\S]*?</summary>",
            f"Summary:\n{summary_content.strip()}",
            formatted_text,
        )

    # Clean up excessive newlines
    formatted_text = re.sub(r"\n\n+", "\n\n", formatted_text)

    return formatted_text.strip()


def build_continuation_prompt(summary_text: str, should_continue: bool = False) -> str:
    """Build the continuation prompt for a compacted conversation.

    Args:
        summary_text: The formatted summary text.
        should_continue: If True, instructs the model to continue without asking.

    Returns:
        The continuation prompt to start the compacted conversation.
    """
    formatted_summary = format_summary_response(summary_text)
    prompt = f"""This session is being continued from a previous conversation that ran out of context. The conversation is summarized below:
{formatted_summary}"""

    if should_continue:
        return f"""{prompt}

Please continue the conversation from where we left it off without asking the user any further questions. Continue with the last task that you were asked to work on."""

    return prompt


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CompactionResult:
    """Result of a conversation compaction operation."""

    messages: List[ConversationMessage]
    summary_text: str
    continuation_prompt: str
    tokens_before: int
    tokens_after: int
    tokens_saved: int
    micro_tokens_saved: int
    was_compacted: bool


@dataclass
class CompactionError:
    """Error during compaction."""

    error_type: str  # "not_enough_messages", "empty_summary", "exception"
    message: str
    exception: Optional[Exception] = None


def extract_tool_ids_from_message(msg: ConversationMessage) -> Tuple[Set[str], Set[str]]:
    """Extract tool_use IDs and tool_result IDs from a message."""
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
    """Return the last N messages, expanding to keep tool_use/tool_result pairs together."""
    if target_count <= 0 or not messages:
        return []

    tail_start = max(0, len(messages) - target_count)
    tail = messages[tail_start:]

    result_ids_in_tail: Set[str] = set()
    for msg in tail:
        _, result_ids = extract_tool_ids_from_message(msg)
        result_ids_in_tail.update(result_ids)

    use_ids_in_tail: Set[str] = set()
    for msg in tail:
        use_ids, _ = extract_tool_ids_from_message(msg)
        use_ids_in_tail.update(use_ids)

    orphan_result_ids = result_ids_in_tail - use_ids_in_tail
    if not orphan_result_ids:
        return tail

    for i in range(tail_start - 1, -1, -1):
        msg = messages[i]
        use_ids, _ = extract_tool_ids_from_message(msg)
        matched = use_ids & orphan_result_ids
        if matched:
            tail_start = i
            orphan_result_ids -= matched
            use_ids_in_tail.update(use_ids)

            _, new_result_ids = extract_tool_ids_from_message(msg)
            new_orphans = new_result_ids - use_ids_in_tail
            orphan_result_ids.update(new_orphans)

        if not orphan_result_ids:
            break

    return messages[tail_start:]


async def summarize_conversation(
    messages: List[ConversationMessage],
    custom_instructions: str = "",
) -> str:
    """Summarize the given conversation using the configured model.

    Uses a detailed prompt structure to capture technical details, code patterns,
    and architectural decisions essential for continuing development work.

    Args:
        messages: The conversation messages to summarize (uses last 60).
        custom_instructions: Optional additional instructions for summarization.

    Returns:
        The summary text, or empty string if summarization fails.
    """
    recent_messages = messages[-60:]
    transcript = render_transcript(recent_messages)

    logger.debug(
        "[compaction] summarize_conversation: %d messages, transcript length=%d",
        len(recent_messages),
        len(transcript) if transcript else 0,
    )

    if not transcript.strip():
        logger.warning("[compaction] transcript is empty, cannot summarize")
        return ""

    # Use the detailed summary prompt from generate_summary_prompt
    system_prompt = "You are a helpful AI assistant tasked with summarizing conversations."
    user_prompt = generate_summary_prompt(custom_instructions)
    user_content = f"{user_prompt}\n\nHere is the conversation to summarize:\n\n{transcript}"

    assistant_response = await query_llm(
        messages=[create_user_message(user_content)],
        system_prompt=system_prompt,
        tools=[],
        max_thinking_tokens=0,
        model="main",
    )

    result = extract_assistant_text(assistant_response)
    logger.debug(
        "[compaction] summarize_conversation returned: length=%d",
        len(result) if result else 0,
    )
    return result


async def compact_conversation(
    messages: List[ConversationMessage],
    custom_instructions: str = "",
    protocol: str = "anthropic",
    tail_count: int = RECENT_MESSAGES_AFTER_COMPACT,
    attachment_provider: Optional[Callable[[], List[ConversationMessage]]] = None,
) -> Union["CompactionResult", "CompactionError"]:
    """Compact a conversation by summarizing and rebuilding.

    This is a pure logic function with no UI dependencies.

    Args:
        messages: The conversation messages to compact.
        custom_instructions: Optional instructions for the summarizer.
        protocol: The API protocol ("anthropic" or "openai").
        tail_count: Number of recent messages to preserve after compaction.
        attachment_provider: Optional callable to provide attachment messages.

    Returns:
        CompactionResult on success, CompactionError on failure.
    """
    if len(messages) < 2:
        return CompactionError(
            error_type="not_enough_messages",
            message="Not enough conversation history to compact.",
        )

    tokens_before = estimate_conversation_tokens(messages, protocol=protocol)

    micro = micro_compact_messages(messages, protocol=protocol)
    messages_for_summary = micro.messages

    # Summarize the conversation

    non_progress_messages = [
        m for m in messages_for_summary if getattr(m, "type", "") != "progress"
    ]
    try:
        summary_text = await summarize_conversation(non_progress_messages, custom_instructions)
    except Exception as exc:
        import traceback

        logger.warning(
            "[compaction] Error during compaction: %s: %s\n%s",
            type(exc).__name__,
            exc,
            traceback.format_exc(),
        )
        return CompactionError(
            error_type="exception",
            message=f"Error during compaction: {exc}",
            exception=exc,
        )

    if not summary_text.strip():
        return CompactionError(
            error_type="empty_summary",
            message="Failed to summarize conversation for compaction.",
        )

    # Build continuation prompt using the new structured format
    continuation_prompt = build_continuation_prompt(summary_text, should_continue=False)

    recent_tail = get_complete_tool_pairs_tail(non_progress_messages, tail_count)

    attachments: List[ConversationMessage] = []
    if callable(attachment_provider):
        try:
            attachments = attachment_provider() or []
        except Exception as exc:
            logger.warning(
                "[compaction] attachment_provider failed: %s: %s",
                type(exc).__name__,
                exc,
            )

    compacted_messages: List[ConversationMessage] = [create_user_message(continuation_prompt)]
    compacted_messages.extend(attachments)
    compacted_messages.extend(recent_tail)

    tokens_after = estimate_conversation_tokens(compacted_messages, protocol=protocol)
    tokens_saved = max(0, tokens_before - tokens_after)

    return CompactionResult(
        messages=compacted_messages,
        summary_text=summary_text,
        continuation_prompt=continuation_prompt,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        tokens_saved=tokens_saved,
        micro_tokens_saved=micro.tokens_saved,
        was_compacted=True,
    )


# Legacy class for backward compatibility
class ConversationCompactor:
    """Legacy wrapper for conversation compaction.

    Deprecated: Use compact_conversation() function directly instead.
    This class is kept for backward compatibility.
    """

    # Keep CompactionResult as a nested class for backward compatibility
    CompactionResult = CompactionResult

    def __init__(
        self,
        console: Optional[object] = None,
        render_transcript_fn: Optional[Callable] = None,
        extract_assistant_text_fn: Optional[Callable] = None,
        attachment_provider: Optional[Callable[[], List[ConversationMessage]]] = None,
    ):
        self._attachment_provider = attachment_provider
        # console and render functions are ignored - kept for API compatibility

    async def compact(
        self,
        messages: List[ConversationMessage],
        custom_instructions: str,
        protocol: str = "anthropic",
        tail_count: int = RECENT_MESSAGES_AFTER_COMPACT,
    ) -> Optional["CompactionResult"]:  # type: ignore[valid-type]
        """Compact the conversation. Returns None on error."""
        result = await compact_conversation(
            messages=messages,
            custom_instructions=custom_instructions,
            protocol=protocol,
            tail_count=tail_count,
            attachment_provider=self._attachment_provider,
        )
        if isinstance(result, CompactionError):
            return None
        return result
