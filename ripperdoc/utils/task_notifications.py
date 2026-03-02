"""Helpers for structured task completion notifications."""

from __future__ import annotations

import json
import re
from html import escape, unescape
from typing import Any, Dict, Optional, TypedDict

from ripperdoc.utils.pending_messages import PendingMessageQueue


class ParsedTaskNotification(TypedDict, total=False):
    task_id: str
    status: str
    summary: str
    tool_use_id: str
    output_file: str
    usage: Dict[str, Any]


_TASK_NOTIFICATION_PATTERN = re.compile(
    r"<task-notification>([\s\S]*?)</task-notification>",
    flags=re.IGNORECASE,
)


def _extract_tag(content: str, tag: str) -> Optional[str]:
    pattern = re.compile(rf"<{re.escape(tag)}>([\s\S]*?)</{re.escape(tag)}>", flags=re.IGNORECASE)
    match = pattern.search(content)
    if not match:
        return None
    return unescape((match.group(1) or "").strip())


def parse_task_notification(payload: str) -> Optional[ParsedTaskNotification]:
    """Parse XML-like task notification payload into a dict."""
    if not isinstance(payload, str):
        return None
    match = _TASK_NOTIFICATION_PATTERN.search(payload)
    if not match:
        return None

    inner = match.group(1)
    parsed: ParsedTaskNotification = {}
    task_id = _extract_tag(inner, "task-id")
    status = _extract_tag(inner, "status")
    summary = _extract_tag(inner, "summary")
    tool_use_id = _extract_tag(inner, "tool-use-id")
    output_file = _extract_tag(inner, "output-file")
    usage_raw = _extract_tag(inner, "usage")

    if task_id:
        parsed["task_id"] = task_id
    if status:
        parsed["status"] = status
    if summary:
        parsed["summary"] = summary
    if tool_use_id:
        parsed["tool_use_id"] = tool_use_id
    if output_file:
        parsed["output_file"] = output_file
    if usage_raw:
        try:
            parsed_usage = json.loads(usage_raw)
        except json.JSONDecodeError:
            parsed_usage = {"raw": usage_raw}
        if isinstance(parsed_usage, dict):
            parsed["usage"] = parsed_usage

    return parsed


def format_task_notification_for_agent(payload: str) -> str:
    """Wrap a structured task notification for agent-visible conversation input."""
    return f"A background agent completed a task:\n{payload}"


def summarize_task_notification(payload: str) -> str:
    """Create a concise user-facing summary from a task notification payload."""
    parsed = parse_task_notification(payload)
    if not parsed:
        return "A background task completed."

    task_id = parsed.get("task_id") or "unknown"
    status = parsed.get("status") or "completed"
    summary = parsed.get("summary") or "No summary provided."
    return f"{task_id} [{status}] {summary}"


def format_task_notification(
    *,
    task_id: str,
    status: str,
    summary: str,
    tool_use_id: Optional[str] = None,
    output_file: Optional[str] = None,
    usage: Optional[Dict[str, Any]] = None,
) -> str:
    """Render a task notification payload with stable XML-like tags."""
    lines = ["<task-notification>"]
    lines.append(f"<task-id>{escape(task_id or '')}</task-id>")
    lines.append(f"<status>{escape(status or '')}</status>")
    lines.append(f"<summary>{escape(summary or '')}</summary>")
    if tool_use_id:
        lines.append(f"<tool-use-id>{escape(tool_use_id)}</tool-use-id>")
    if output_file:
        lines.append(f"<output-file>{escape(output_file)}</output-file>")
    if usage:
        usage_json = json.dumps(usage, ensure_ascii=False, separators=(",", ":"))
        lines.append(f"<usage>{escape(usage_json)}</usage>")
    lines.append("</task-notification>")
    return "\n".join(lines)


def enqueue_task_notification(
    queue: Optional[PendingMessageQueue],
    *,
    task_id: str,
    status: str,
    summary: str,
    tool_use_id: Optional[str] = None,
    output_file: Optional[str] = None,
    usage: Optional[Dict[str, Any]] = None,
    source: Optional[str] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """Enqueue a task notification into a pending-message queue."""
    if queue is None:
        return False

    payload = format_task_notification(
        task_id=task_id,
        status=status,
        summary=summary,
        tool_use_id=tool_use_id,
        output_file=output_file,
        usage=usage,
    )
    metadata: Dict[str, Any] = {
        "notification_type": "task_notification",
        "task_id": task_id,
        "status": status,
    }
    if source:
        metadata["source"] = source
    if tool_use_id:
        metadata["tool_use_id"] = tool_use_id
    if output_file:
        metadata["output_file"] = output_file
    if usage:
        metadata["usage"] = usage
    if extra_metadata:
        metadata.update(extra_metadata)

    queue.enqueue_text(payload, metadata=metadata)
    return True
