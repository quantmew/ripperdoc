"""Rendering helpers for streamed messages in the Rich UI."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Dict, Optional

from ripperdoc.utils.messages import (
    AssistantMessage,
    UserMessage,
    ProgressMessage,
    INTERRUPT_MESSAGE,
    INTERRUPT_MESSAGE_FOR_TOOL_USE,
    is_hook_notice_payload,
)
from ripperdoc.utils.token_estimation import estimate_tokens
from rich.markup import escape


def simplify_progress_suffix(content: Any) -> str:
    """Simplify progress message content for cleaner spinner display."""
    if not isinstance(content, str):
        return f"Working... {content}"

    # Hide stream labels from the spinner line; keep only useful payload text.
    for stream_label in ("stdout:", "stderr:"):
        if content.startswith(stream_label):
            stream_text = content[len(stream_label) :].strip()
            return f"Working... {stream_text}" if stream_text else "Working..."

    # Handle bash command progress: "Running... (10s)\nstdout..."
    if content.startswith("Running..."):
        # Extract just the "Running... (time)" part before any newline
        first_line = content.split("\n", 1)[0]
        return first_line

    # For other progress messages, limit length to avoid terminal wrapping
    max_length = 60
    if len(content) > max_length:
        return f"Working... {content[:max_length]}..."

    return f"Working... {content}"


def handle_assistant_message(
    ui: object,
    message: AssistantMessage,
    tool_registry: Dict[str, Dict[str, Any]],
    spinner: Optional[Any] = None,
) -> Optional[str]:
    """Handle an assistant message from the query stream."""
    pause = lambda: spinner.paused() if spinner else nullcontext()  # noqa: E731

    meta = getattr(getattr(message, "message", None), "metadata", {}) or {}
    reasoning_payload = (
        meta.get("reasoning_content") or meta.get("reasoning") or meta.get("reasoning_details")
    )
    if reasoning_payload:
        with pause():
            ui._print_reasoning(reasoning_payload)

    last_tool_name: Optional[str] = None

    if isinstance(message.message.content, str):
        if ui._esc_interrupt_seen and message.message.content.strip() in (
            INTERRUPT_MESSAGE,
            INTERRUPT_MESSAGE_FOR_TOOL_USE,
        ):
            return last_tool_name
        with pause():
            ui.display_message("Ripperdoc", message.message.content)
    elif isinstance(message.message.content, list):
        for block in message.message.content:
            if hasattr(block, "type") and block.type == "text" and block.text:
                with pause():
                    ui.display_message("Ripperdoc", block.text)
            elif hasattr(block, "type") and block.type == "tool_use":
                tool_name = getattr(block, "name", "unknown tool")
                tool_args = getattr(block, "input", {})
                tool_use_id = getattr(block, "tool_use_id", None) or getattr(block, "id", None)

                if tool_use_id:
                    tool_registry[tool_use_id] = {
                        "name": tool_name,
                        "args": tool_args,
                        "printed": False,
                    }

                if tool_name == "Task":
                    with pause():
                        ui.display_message(
                            tool_name, "", is_tool=True, tool_type="call", tool_args=tool_args
                        )
                    if tool_use_id:
                        tool_registry[tool_use_id]["printed"] = True

                last_tool_name = tool_name

    return last_tool_name


def handle_tool_result_message(
    ui: object,
    message: UserMessage,
    tool_registry: Dict[str, Dict[str, Any]],
    last_tool_name: Optional[str],
    spinner: Optional[Any] = None,
) -> None:
    """Handle a user message containing tool results."""
    if not isinstance(message.message.content, list):
        return

    pause = lambda: spinner.paused() if spinner else nullcontext()  # noqa: E731

    for block in message.message.content:
        if not (hasattr(block, "type") and block.type == "tool_result" and block.text):
            continue

        tool_name = "Tool"
        tool_data = getattr(message, "tool_use_result", None)
        is_error = bool(getattr(block, "is_error", False))
        tool_use_id = getattr(block, "tool_use_id", None)

        entry = tool_registry.get(tool_use_id) if tool_use_id else None
        if entry:
            tool_name = entry.get("name", tool_name)
            if not entry.get("printed"):
                with pause():
                    ui.display_message(
                        tool_name,
                        "",
                        is_tool=True,
                        tool_type="call",
                        tool_args=entry.get("args", {}),
                    )
                entry["printed"] = True
        elif last_tool_name:
            tool_name = last_tool_name

        with pause():
            ui.display_message(
                tool_name,
                block.text,
                is_tool=True,
                tool_type="result",
                tool_data=tool_data,
                tool_error=is_error,
            )


def handle_progress_message(
    ui: object,
    message: ProgressMessage,
    spinner: Any,
    output_token_est: int,
) -> int:
    """Handle a progress message and update spinner."""
    if is_hook_notice_payload(message.content):
        payload = message.content
        hook_event = payload.get("hook_event") or "Hook"
        tool_name = payload.get("tool_name")
        label = f"{hook_event}:{tool_name}" if tool_name else str(hook_event)
        text = payload.get("text", "")
        with spinner.paused():
            ui.console.print(
                f"[yellow]Hook {escape(str(label))}[/yellow] {escape(str(text))}"
            )
        return output_token_est

    if ui.verbose:
        with spinner.paused():
            ui.display_message("System", f"Progress: {message.content}", is_tool=True)
    elif message.content and isinstance(message.content, str):
        if message.content.startswith("Subagent: "):
            with spinner.paused():
                ui.display_message(
                    "Subagent", message.content[len("Subagent: ") :], is_tool=True
                )
        elif message.content.startswith("Subagent"):
            with spinner.paused():
                ui.display_message("Subagent", message.content, is_tool=True)

    if message.tool_use_id == "stream":
        delta_tokens = estimate_tokens(message.content)
        output_token_est += delta_tokens
        spinner.update_tokens(output_token_est)
    else:
        # Keep spinner minimal: do not surface tool progress text as spinner suffix.
        spinner.update_tokens(output_token_est)

    return output_token_est
