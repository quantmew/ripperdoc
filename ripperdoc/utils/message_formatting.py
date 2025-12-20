"""Message content formatting utilities.

This module provides functions for converting message content to plain text,
with support for detailed tool information extraction. Used primarily for
conversation summarization and compaction.
"""

from typing import Any, List, Union

from ripperdoc.utils.messages import UserMessage, AssistantMessage, ProgressMessage

ConversationMessage = Union[UserMessage, AssistantMessage, ProgressMessage]


def stringify_message_content(content: Any, *, include_tool_details: bool = False) -> str:
    """Convert message content to plain string.

    Args:
        content: The message content to stringify.
        include_tool_details: If True, include tool input/output details
            instead of just placeholders. Useful for summarization.

    Returns:
        Plain text representation of the message content.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text = getattr(block, "text", None)
                if text:
                    parts.append(str(text))
            elif block_type == "tool_use":
                name = getattr(block, "name", "tool")
                if include_tool_details:
                    tool_input = getattr(block, "input", None)
                    parts.append(format_tool_use_detail(name, tool_input))
                else:
                    parts.append(f"[Called {name}]")
            elif block_type == "tool_result":
                if include_tool_details:
                    result_text = getattr(block, "text", None) or ""
                    is_error = getattr(block, "is_error", False)
                    parts.append(format_tool_result_detail(result_text, is_error))
                else:
                    parts.append("[Tool result]")
        return "\n".join(parts)
    return str(content)


def format_tool_use_detail(name: str, tool_input: Any) -> str:
    """Format tool_use block with input details for summarization.

    Args:
        name: The tool name.
        tool_input: The tool input dictionary.

    Returns:
        Formatted string like "[Called Bash(command=ls -la)]"
    """
    if not tool_input:
        return f"[Called {name}]"

    summary_parts: List[str] = []
    if isinstance(tool_input, dict):
        # Common patterns for different tools
        if name == "Bash":
            cmd = tool_input.get("command", "")
            if cmd:
                cmd_preview = cmd[:200] + "..." if len(cmd) > 200 else cmd
                summary_parts.append(f"command={cmd_preview}")
        elif name in ("Read", "Write", "Edit", "MultiEdit"):
            path = tool_input.get("file_path", "")
            if path:
                summary_parts.append(f"file={path}")
        elif name in ("Glob", "Grep"):
            pattern = tool_input.get("pattern", "")
            if pattern:
                summary_parts.append(f"pattern={pattern}")
        elif name == "Task":
            desc = tool_input.get("description", "")
            subagent = tool_input.get("subagent_type", "")
            if subagent:
                summary_parts.append(f"subagent={subagent}")
            if desc:
                summary_parts.append(f"desc={desc}")
        else:
            # Generic: show first few key-value pairs
            for key, value in list(tool_input.items())[:3]:
                val_str = str(value)
                if len(val_str) > 50:
                    val_str = val_str[:47] + "..."
                summary_parts.append(f"{key}={val_str}")

    if summary_parts:
        return f"[Called {name}({', '.join(summary_parts)})]"
    return f"[Called {name}]"


def format_tool_result_detail(result_text: str, is_error: bool = False) -> str:
    """Format tool_result block with output details for summarization.

    Args:
        result_text: The tool result text.
        is_error: Whether this is an error result.

    Returns:
        Formatted string like "[Tool result]: file contents..."
    """
    prefix = "[Tool error]" if is_error else "[Tool result]"
    if not result_text:
        return prefix

    # Truncate very long results but keep enough for context
    max_len = 500
    if len(result_text) > max_len:
        result_preview = result_text[:max_len] + f"... (truncated, {len(result_text)} chars total)"
    else:
        result_preview = result_text

    return f"{prefix}: {result_preview}"


def format_reasoning_preview(reasoning: Any, show_full_thinking: bool = False) -> str:
    """Return a short preview of reasoning/thinking content.

    Args:
        reasoning: The reasoning content (string, list, or other).
        show_full_thinking: If True, return full reasoning content without truncation.
            If False, return a truncated preview (max 250 chars).

    Returns:
        A short preview string or full reasoning content.
    """
    if reasoning is None:
        return ""
    if isinstance(reasoning, str):
        text = reasoning
    elif isinstance(reasoning, list):
        parts = []
        for block in reasoning:
            if isinstance(block, dict):
                parts.append(block.get("thinking") or block.get("summary") or "")
            elif hasattr(block, "thinking"):
                parts.append(getattr(block, "thinking", "") or "")
        text = "\n".join(p for p in parts if p)
    else:
        text = str(reasoning)
    
    if show_full_thinking:
        return text
    
    lines = text.strip().splitlines()
    if not lines:
        return ""
    preview = lines[0][:250]
    if len(lines) > 1 or len(lines[0]) > 250:
        preview += "..."
    return preview


def render_transcript(
    messages: List[ConversationMessage], *, include_tool_details: bool = True
) -> str:
    """Render conversation messages into a plain-text transcript.

    Args:
        messages: List of conversation messages to render.
        include_tool_details: If True (default), include tool input/output
            details for better summarization context.

    Returns:
        Plain text transcript of the conversation.
    """
    lines: List[str] = []
    for msg in messages:
        msg_type = getattr(msg, "type", "")
        if msg_type == "progress":
            continue
        role = "User" if msg_type == "user" else "Assistant"
        content = getattr(getattr(msg, "message", None), "content", None)
        text = stringify_message_content(content, include_tool_details=include_tool_details)
        if text.strip():
            lines.append(f"{role}: {text}")
    return "\n\n".join(lines)


def extract_assistant_text(assistant_message: Any) -> str:
    """Extract plain text from an assistant response object.

    Args:
        assistant_message: An AssistantMessage or similar object.

    Returns:
        Plain text content from the message.
    """
    # AssistantMessage has .message.content structure
    message = getattr(assistant_message, "message", None)
    if message is not None:
        content = getattr(message, "content", None)
    else:
        # Fallback: maybe it's a raw object with .content directly
        content = getattr(assistant_message, "content", None)

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if getattr(block, "type", None) == "text":
                text = getattr(block, "text", None)
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    return ""
