"""Permission handling for tool execution.

This module adds a lightweight permission system.
When safe mode is enabled, tool executions are checked against
the project's allowed list and will prompt the user before running.
Session approvals can be granted inline, and explicit project approvals
are persisted in `.ripperdoc/config.json`.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional, Set

from ripperdoc.core.config import config_manager
from ripperdoc.core.tool import Tool


@dataclass
class PermissionResult:
    """Result of a permission check."""

    result: bool
    message: Optional[str] = None


def _format_input_preview(parsed_input: Any) -> str:
    """Create a short, human-friendly preview for prompts."""
    if hasattr(parsed_input, "command"):
        return f"command='{getattr(parsed_input, 'command')}'"
    if hasattr(parsed_input, "file_path"):
        return f"file='{getattr(parsed_input, 'file_path')}'"
    if hasattr(parsed_input, "path"):
        return f"path='{getattr(parsed_input, 'path')}'"

    preview = str(parsed_input)
    if len(preview) > 140:
        return preview[:137] + "..."
    return preview


def permission_key(tool: Tool[Any, Any], parsed_input: Any) -> str:
    """Build a stable permission key for persistence."""
    if hasattr(parsed_input, "command"):
        return f"{tool.name}::command::{getattr(parsed_input, 'command')}"
    if hasattr(parsed_input, "file_path"):
        try:
            return f"{tool.name}::path::{Path(getattr(parsed_input, 'file_path')).resolve()}"
        except Exception:
            return f"{tool.name}::path::{getattr(parsed_input, 'file_path')}"
    if hasattr(parsed_input, "path"):
        try:
            return f"{tool.name}::path::{Path(getattr(parsed_input, 'path')).resolve()}"
        except Exception:
            return f"{tool.name}::path::{getattr(parsed_input, 'path')}"
    return tool.name


def _radiolist_choice(prompt: str, options: list[tuple[str, str]]) -> Optional[str]:
    """Try to show a compact arrow-key selector."""
    return None


def _render_options_prompt(prompt: str, options: list[tuple[str, str]]) -> str:
    """Render a simple numbered prompt when radiolist is unavailable."""
    border = "─" * 120
    lines = [border, prompt, ""]
    for idx, (_, label) in enumerate(options, start=1):
        prefix = "❯" if idx == 1 else " "
        lines.append(f"{prefix} {idx}. {label}")
    numeric_choices = "/".join(str(i) for i in range(1, len(options) + 1))
    shortcut_choices = "/".join(opt[0] for opt in options)
    lines.append(f"Choice ({numeric_choices} or {shortcut_choices}): ")
    return "\n".join(lines)


def make_permission_checker(
    project_path: Path,
    safe_mode: bool,
    prompt_fn: Optional[Callable[[str], str]] = None
) -> Callable[[Tool[Any, Any], Any], Awaitable[PermissionResult]]:
    """Create a permission checking function for the current project.

    Args:
        project_path: Path to the project root (where .ripperdoc lives)
        safe_mode: Whether safe mode is enabled
        prompt_fn: Optional prompt function for testing/injection
    """
    project_path = project_path.resolve()
    # Ensure the config manager is scoped to the current project
    config_manager.get_project_config(project_path)

    session_allowed: Set[str] = set()

    async def _prompt_user(prompt: str, options: Optional[list[tuple[str, str]]] = None) -> str:
        """Prompt the user without blocking the event loop."""
        loop = asyncio.get_running_loop()
        responder = prompt_fn or input

        def _ask() -> str:
            if options:
                rendered = _render_options_prompt(prompt, options)
                return responder(rendered)
            return responder(prompt)

        return await loop.run_in_executor(None, _ask)

    async def can_use_tool(tool: Tool[Any, Any], parsed_input: Any) -> PermissionResult:
        """Check and optionally persist permission for a tool invocation."""
        if not safe_mode:
            return PermissionResult(result=True)

        # Skip tools that do not require permissions (e.g., read-only tools)
        try:
            if hasattr(tool, "needs_permissions") and not tool.needs_permissions(parsed_input):
                return PermissionResult(result=True)
        except Exception:
            # Fail closed on unexpected errors
            return PermissionResult(
                result=False,
                message="Permission check failed for this tool invocation."
            )

        config = config_manager.get_project_config(project_path)
        allowed = set(config.allowed_tools or [])
        key_specific = permission_key(tool, parsed_input)
        key_generic = tool.name

        # Session approvals or persisted approvals
        if key_specific in allowed or key_generic in allowed:
            return PermissionResult(result=True)
        if key_specific in session_allowed or key_generic in session_allowed:
            return PermissionResult(result=True)

        input_preview = _format_input_preview(parsed_input)
        prompt_lines = [
            f"{tool.name}",
            "",
            f"  {input_preview}",
            "  Do you want to proceed?",
        ]
        prompt = "\n".join(prompt_lines)
        options = [
            ("y", "Yes"),
            ("a", "Yes, always for this session"),
            ("n", "No"),
        ]

        answer = (await _prompt_user(prompt, options=options)).strip().lower()

        # Support arrow selection (y/a/p/n) and numeric shortcuts
        if answer in ("1", "y", "yes"):
            session_allowed.add(key_specific or key_generic)
            return PermissionResult(result=True)

        if answer in ("2", "a", "always"):
            # Allow the entire tool for the current session
            session_allowed.add(key_generic)
            return PermissionResult(result=True)

        if answer in ("3", "n", "no"):
            return PermissionResult(
                result=False,
                message=f"Permission denied for tool '{tool.name}'."
            )

        return PermissionResult(
            result=False,
            message=f"Permission denied for tool '{tool.name}'."
        )

    return can_use_tool
