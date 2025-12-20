"""Permission handling for tool execution."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional, Set

from ripperdoc.core.config import config_manager
from ripperdoc.core.tool import Tool
from ripperdoc.utils.permissions import PermissionDecision, ToolRule
from ripperdoc.utils.log import get_logger

logger = get_logger()


@dataclass
class PermissionResult:
    """Result of a permission check."""

    result: bool
    message: Optional[str] = None
    updated_input: Any = None
    decision: Optional[PermissionDecision] = None


def _format_input_preview(parsed_input: Any, tool_name: Optional[str] = None) -> str:
    """Create a human-friendly preview for prompts.

    For Bash commands, shows full details for security review.
    For other tools, shows a concise preview.
    """
    # For Bash tool, show full command details for security review
    if tool_name == "Bash" and hasattr(parsed_input, "command"):
        lines = [f"Command: {getattr(parsed_input, 'command')}"]

        # Add other relevant parameters
        if hasattr(parsed_input, "timeout") and parsed_input.timeout:
            lines.append(f"Timeout: {parsed_input.timeout}ms")
        if hasattr(parsed_input, "sandbox"):
            lines.append(f"Sandbox: {parsed_input.sandbox}")
        if hasattr(parsed_input, "run_in_background"):
            lines.append(f"Background: {parsed_input.run_in_background}")
        if hasattr(parsed_input, "shell_executable") and parsed_input.shell_executable:
            lines.append(f"Shell: {parsed_input.shell_executable}")

        return "\n  ".join(lines)

    # For other tools with commands, show concise preview
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


def _render_options_prompt(prompt: str, options: list[tuple[str, str]]) -> str:
    """Render a simple numbered prompt."""
    border = "─" * 120
    lines = [border, prompt, ""]
    for idx, (_, label) in enumerate(options, start=1):
        prefix = "❯" if idx == 1 else " "
        lines.append(f"{prefix} {idx}. {label}")
    numeric_choices = "/".join(str(i) for i in range(1, len(options) + 1))
    shortcut_choices = "/".join(opt[0] for opt in options)
    lines.append(f"Choice ({numeric_choices} or {shortcut_choices}): ")
    return "\n".join(lines)


def _rule_strings(rule_suggestions: Optional[Any]) -> list[str]:
    """Normalize rule suggestions to simple strings."""
    if not rule_suggestions:
        return []
    rules: list[str] = []
    for suggestion in rule_suggestions:
        if isinstance(suggestion, ToolRule):
            rules.append(suggestion.rule_content)
        else:
            rules.append(str(suggestion))
    return [rule for rule in rules if rule]


def make_permission_checker(
    project_path: Path,
    yolo_mode: bool,
    prompt_fn: Optional[Callable[[str], str]] = None,
) -> Callable[[Tool[Any, Any], Any], Awaitable[PermissionResult]]:
    """Create a permission checking function for the current project.

    In yolo mode, all tool calls are allowed without prompting.
    """

    project_path = project_path.resolve()
    config_manager.get_project_config(project_path)

    session_allowed_tools: Set[str] = set()
    session_tool_rules: dict[str, Set[str]] = defaultdict(set)

    async def _prompt_user(prompt: str, options: list[tuple[str, str]]) -> str:
        """Prompt the user without blocking the event loop."""
        loop = asyncio.get_running_loop()
        responder = prompt_fn or input

        def _ask() -> str:
            rendered = _render_options_prompt(prompt, options)
            return responder(rendered)

        return await loop.run_in_executor(None, _ask)

    async def can_use_tool(tool: Tool[Any, Any], parsed_input: Any) -> PermissionResult:
        """Check and optionally persist permission for a tool invocation."""
        config = config_manager.get_project_config(project_path)

        if yolo_mode:
            return PermissionResult(result=True)

        try:
            if hasattr(tool, "needs_permissions") and not tool.needs_permissions(parsed_input):
                return PermissionResult(result=True)
        except (TypeError, AttributeError, ValueError) as exc:
            # Tool implementation error - log and deny for safety
            logger.warning(
                "[permissions] Tool needs_permissions check failed",
                extra={
                    "tool": getattr(tool, "name", None),
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            )
            return PermissionResult(
                result=False,
                message=f"Permission check failed: {type(exc).__name__}: {exc}",
            )

        allowed_tools = set(config.allowed_tools or [])
        allow_rules = {
            "Bash": set(config.bash_allow_rules or []) | session_tool_rules.get("Bash", set())
        }
        deny_rules = {"Bash": set(config.bash_deny_rules or [])}
        allowed_working_dirs = {
            str(project_path.resolve()),
            *[str(Path(p).resolve()) for p in config.working_directories or []],
        }

        # Persisted approvals
        if tool.name in allowed_tools or tool.name in session_allowed_tools:
            return PermissionResult(result=True)

        decision: Optional[PermissionDecision] = None
        if hasattr(tool, "check_permissions"):
            try:
                maybe_decision = tool.check_permissions(
                    parsed_input,
                    {
                        "allowed_rules": allow_rules.get(tool.name, set()),
                        "denied_rules": deny_rules.get(tool.name, set()),
                        "allowed_working_directories": allowed_working_dirs,
                    },
                )
                decision = (
                    await maybe_decision if asyncio.iscoroutine(maybe_decision) else maybe_decision
                )
                # Allow tools to return a plain dict shaped like PermissionDecision.
                if isinstance(decision, dict) and "behavior" in decision:
                    decision = PermissionDecision(**decision)
            except (TypeError, AttributeError, ValueError, KeyError) as exc:
                # Tool implementation error - fall back to asking user
                logger.warning(
                    "[permissions] Tool check_permissions failed",
                    extra={
                        "tool": getattr(tool, "name", None),
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    },
                )
                decision = PermissionDecision(
                    behavior="ask",
                    message=f"Error checking permissions: {type(exc).__name__}",
                    rule_suggestions=None,
                )

        if decision is None:
            decision = PermissionDecision(
                behavior="passthrough",
                message=f"Allow tool '{tool.name}'?",
                rule_suggestions=[ToolRule(tool_name=tool.name, rule_content=tool.name)],
            )

        if decision.behavior == "allow":
            return PermissionResult(
                result=True,
                message=decision.message,
                updated_input=decision.updated_input,
                decision=decision,
            )

        if decision.behavior == "deny":
            return PermissionResult(
                result=False,
                message=decision.message or f"Permission denied for tool '{tool.name}'.",
                decision=decision,
            )

        # Ask/passthrough flows prompt the user.
        input_preview = _format_input_preview(parsed_input, tool_name=tool.name)
        prompt_lines = [
            f"{tool.name}",
            "",
            f"  {input_preview}",
        ]
        if decision.message:
            prompt_lines.append(f"  {decision.message}")
        prompt_lines.append("  Do you want to proceed?")
        prompt = "\n".join(prompt_lines)

        options = [
            ("y", "Yes"),
            ("s", "Yes, for this session"),
            ("n", "No"),
        ]

        answer = (await _prompt_user(prompt, options=options)).strip().lower()
        logger.debug(
            "[permissions] User answer for permission prompt",
            extra={"answer": answer, "tool": getattr(tool, "name", None)},
        )
        rule_suggestions = _rule_strings(decision.rule_suggestions) or [
            permission_key(tool, parsed_input)
        ]

        if answer in ("1", "y", "yes"):
            return PermissionResult(
                result=True, updated_input=decision.updated_input, decision=decision
            )

        if answer in ("2", "s", "session", "a"):
            if tool.name == "Bash":
                session_tool_rules["Bash"].update(rule_suggestions)
            else:
                session_allowed_tools.add(tool.name)
            return PermissionResult(
                result=True, updated_input=decision.updated_input, decision=decision
            )

        return PermissionResult(
            result=False,
            message=decision.message or f"Permission denied for tool '{tool.name}'.",
            decision=decision,
        )

    return can_use_tool
