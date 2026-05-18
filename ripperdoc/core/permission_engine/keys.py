"""Permission keys and rule suggestion serialization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

from ripperdoc.core.tool import Tool
from ripperdoc.utils.log import get_logger
from ripperdoc.utils.permissions import ToolRule

logger = get_logger()


def permission_key(tool: Tool[Any, Any], parsed_input: Any) -> str:
    """Build a stable permission key for persistence."""
    if hasattr(parsed_input, "command"):
        return f"{tool.name}::command::{getattr(parsed_input, 'command')}"
    if hasattr(parsed_input, "file_path"):
        try:
            return f"{tool.name}::path::{Path(getattr(parsed_input, 'file_path')).resolve()}"
        except (OSError, RuntimeError) as exc:
            logger.warning(
                "[permissions] Failed to resolve file_path for permission key",
                extra={"tool": getattr(tool, "name", None), "error": str(exc)},
            )
            return f"{tool.name}::path::{getattr(parsed_input, 'file_path')}"
    if hasattr(parsed_input, "path"):
        try:
            return f"{tool.name}::path::{Path(getattr(parsed_input, 'path')).resolve()}"
        except (OSError, RuntimeError) as exc:
            logger.warning(
                "[permissions] Failed to resolve path for permission key",
                extra={"tool": getattr(tool, "name", None), "error": str(exc)},
            )
            return f"{tool.name}::path::{getattr(parsed_input, 'path')}"
    return tool.name


def _rule_strings(rule_suggestions: Optional[Any]) -> List[str]:
    """Normalize rule suggestions to simple strings."""
    if not rule_suggestions:
        return []
    rules: List[str] = []
    for suggestion in rule_suggestions:
        if isinstance(suggestion, ToolRule):
            rules.append(suggestion.rule_content)
        else:
            rules.append(str(suggestion))
    return [rule for rule in rules if rule]


def _serialize_permission_suggestions(rule_suggestions: Optional[Any]) -> Optional[List[Any]]:
    """Convert rule suggestions into hook-friendly structures."""
    if not rule_suggestions:
        return None
    suggestions: List[Any] = []
    for suggestion in rule_suggestions:
        if isinstance(suggestion, ToolRule):
            suggestions.append(
                {
                    "tool_name": suggestion.tool_name,
                    "rule": suggestion.rule_content,
                    "behavior": suggestion.behavior,
                }
            )
        else:
            suggestions.append(str(suggestion))
    return suggestions or None

