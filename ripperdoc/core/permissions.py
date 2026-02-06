"""Permission handling for tool execution."""

from __future__ import annotations

import asyncio
import html
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional, Set

from ripperdoc.cli.ui.choice import prompt_choice
from ripperdoc.core.config import config_manager
from ripperdoc.core.hooks.manager import hook_manager
from ripperdoc.core.tool import Tool
from ripperdoc.utils.permissions import PermissionDecision, ToolRule
from ripperdoc.utils.log import get_logger

if TYPE_CHECKING:
    from rich.console import Console
    from prompt_toolkit import PromptSession

logger = get_logger()


@dataclass
class PermissionResult:
    """Result of a permission check."""

    result: bool
    message: Optional[str] = None
    updated_input: Any = None
    decision: Optional[PermissionDecision] = None


@dataclass
class PermissionPreview:
    """Non-interactive preview of permission evaluation."""

    requires_user_input: bool
    result: Optional[PermissionResult] = None
    decision: Optional[PermissionDecision] = None


def _format_input_preview(parsed_input: Any, tool_name: Optional[str] = None) -> str:
    """Create a human-friendly preview for prompts.

    For Bash commands, shows full details for security review.
    For other tools, shows a concise preview.
    Returns HTML-formatted text with color tags.
    """
    # For Bash tool, show full command details for security review
    if tool_name == "Bash" and hasattr(parsed_input, "command"):
        command = html.escape(getattr(parsed_input, "command"))
        lines = [f"<label>Command:</label> <value>{command}</value>"]

        # Add other relevant parameters
        if hasattr(parsed_input, "timeout") and parsed_input.timeout:
            lines.append(f"<label>Timeout:</label> <value>{parsed_input.timeout}ms</value>")
        if hasattr(parsed_input, "sandbox"):
            lines.append(f"<label>Sandbox:</label> <value>{parsed_input.sandbox}</value>")
        if hasattr(parsed_input, "run_in_background"):
            lines.append(f"<label>Background:</label> <value>{parsed_input.run_in_background}</value>")
        if hasattr(parsed_input, "shell_executable") and parsed_input.shell_executable:
            lines.append(f"<label>Shell:</label> <value>{html.escape(parsed_input.shell_executable)}</value>")

        return "\n  ".join(lines)

    # For other tools with commands, show concise preview
    if hasattr(parsed_input, "command"):
        return f"<label>command:</label> <value>'{html.escape(getattr(parsed_input, 'command'))}'</value>"
    if hasattr(parsed_input, "file_path"):
        return f"<label>file:</label> <value>'{html.escape(getattr(parsed_input, 'file_path'))}'</value>"
    if hasattr(parsed_input, "path"):
        return f"<label>path:</label> <value>'{html.escape(getattr(parsed_input, 'path'))}'</value>"

    preview = str(parsed_input)
    if len(preview) > 140:
        preview = preview[:137] + "..."
    return f"<value>{html.escape(preview)}</value>"


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


def _serialize_permission_suggestions(rule_suggestions: Optional[Any]) -> Optional[list[Any]]:
    """Convert rule suggestions into hook-friendly structures."""
    if not rule_suggestions:
        return None
    suggestions: list[Any] = []
    for suggestion in rule_suggestions:
        if isinstance(suggestion, ToolRule):
            suggestions.append(
                {
                    "toolName": suggestion.tool_name,
                    "rule": suggestion.rule_content,
                    "behavior": suggestion.behavior,
                }
            )
        else:
            suggestions.append(str(suggestion))
    return suggestions or None


def _apply_updated_permissions(
    updated_permissions: Any,
    *,
    default_tool_name: str,
    session_allowed_tools: Set[str],
    session_tool_rules: dict[str, Set[str]],
) -> None:
    """Apply updatedPermissions output to in-session permission state."""
    if not updated_permissions:
        return

    def _apply_entry(entry: Any) -> None:
        if entry is None:
            return
        if isinstance(entry, str):
            if default_tool_name == "Bash":
                session_tool_rules.setdefault("Bash", set()).add(entry)
            else:
                session_allowed_tools.add(default_tool_name)
            return
        if not isinstance(entry, dict):
            return

        tool_name = (
            entry.get("toolName")
            or entry.get("tool_name")
            or entry.get("tool")
            or default_tool_name
        )
        behavior = (entry.get("behavior") or entry.get("decision") or "allow").lower()
        rule = entry.get("rule") or entry.get("rule_content") or entry.get("ruleContent")

        if behavior not in ("allow", "approve"):
            return

        if tool_name == "Bash":
            if isinstance(rule, str) and rule:
                session_tool_rules.setdefault("Bash", set()).add(rule)
            return

        if isinstance(tool_name, str) and tool_name:
            session_allowed_tools.add(tool_name)

    if isinstance(updated_permissions, list):
        for entry in updated_permissions:
            _apply_entry(entry)
        return

    if isinstance(updated_permissions, dict):
        allowed_tools = updated_permissions.get("allowedTools") or updated_permissions.get(
            "allowed_tools"
        )
        if isinstance(allowed_tools, list):
            session_allowed_tools.update(
                {str(name).strip() for name in allowed_tools if str(name).strip()}
            )

        bash_allow = (
            updated_permissions.get("bashAllowRules")
            or updated_permissions.get("bash_allow_rules")
            or updated_permissions.get("allowRules")
            or updated_permissions.get("allow_rules")
        )
        if isinstance(bash_allow, list):
            session_tool_rules.setdefault("Bash", set()).update(
                {str(rule).strip() for rule in bash_allow if str(rule).strip()}
            )

        if any(k in updated_permissions for k in ("toolName", "tool_name", "tool", "rule")):
            _apply_entry(updated_permissions)
        return

    _apply_entry(updated_permissions)


def _default_permission_decision(tool_name: str) -> PermissionDecision:
    """Return the fallback permission decision for a tool."""
    return PermissionDecision(
        behavior="passthrough",
        message=f"Allow tool '{tool_name}'?",
        rule_suggestions=[ToolRule(tool_name=tool_name, rule_content=tool_name)],
    )


def _permission_denied_message(tool_name: str, decision: PermissionDecision) -> str:
    """Return a user-facing deny message for a decision."""
    return decision.message or f"Permission denied for tool '{tool_name}'."


def _build_permission_policy(
    *,
    project_path: Path,
    config: Any,
    global_config: Any,
    local_config: Any,
    session_tool_rules: dict[str, Set[str]],
) -> dict[str, Any]:
    """Build merged permission policy inputs for tool-level evaluation."""
    allow_rules = {
        "Bash": (
            set(config.bash_allow_rules or [])
            | set(global_config.user_allow_rules or [])
            | set(local_config.local_allow_rules or [])
            | session_tool_rules.get("Bash", set())
        )
    }
    deny_rules = {
        "Bash": (
            set(config.bash_deny_rules or [])
            | set(global_config.user_deny_rules or [])
            | set(local_config.local_deny_rules or [])
        )
    }
    ask_rules = {
        "Bash": (
            set(config.bash_ask_rules or [])
            | set(global_config.user_ask_rules or [])
            | set(local_config.local_ask_rules or [])
        )
    }
    allowed_working_dirs = {
        str(project_path.resolve()),
        *[str(Path(p).resolve()) for p in config.working_directories or []],
    }

    return {
        "allow_rules": allow_rules,
        "deny_rules": deny_rules,
        "ask_rules": ask_rules,
        "allowed_working_dirs": allowed_working_dirs,
    }


def _coerce_permission_decision(raw_decision: Any) -> Optional[PermissionDecision]:
    """Normalize tool-provided permission decision payloads."""
    if isinstance(raw_decision, PermissionDecision):
        return raw_decision
    if isinstance(raw_decision, dict) and "behavior" in raw_decision:
        try:
            return PermissionDecision(**raw_decision)
        except TypeError:
            return PermissionDecision(
                behavior="ask",
                message="Error checking permissions: TypeError",
                rule_suggestions=None,
            )
    return None


async def _resolve_permission_decision(
    tool: Tool[Any, Any],
    parsed_input: Any,
    *,
    policy: dict[str, Any],
    log_errors: bool,
) -> PermissionDecision:
    """Resolve the tool decision from tool policy hooks/checkers."""
    if not hasattr(tool, "check_permissions"):
        return _default_permission_decision(tool.name)

    permission_context = {
        "allowed_rules": policy["allow_rules"].get(tool.name, set()),
        "denied_rules": policy["deny_rules"].get(tool.name, set()),
        "ask_rules": policy["ask_rules"].get(tool.name, set()),
        "allowed_working_directories": policy["allowed_working_dirs"],
    }

    try:
        maybe_decision = tool.check_permissions(parsed_input, permission_context)
        raw_decision = await maybe_decision if asyncio.iscoroutine(maybe_decision) else maybe_decision
        decision = _coerce_permission_decision(raw_decision)
        return decision or _default_permission_decision(tool.name)
    except (TypeError, AttributeError, ValueError, KeyError) as exc:
        if log_errors:
            logger.warning(
                "[permissions] Tool check_permissions failed",
                extra={
                    "tool": getattr(tool, "name", None),
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            )
        return PermissionDecision(
            behavior="ask",
            message=f"Error checking permissions: {type(exc).__name__}",
            rule_suggestions=None,
        )


def _run_permission_decision_engine(
    *,
    tool_name: str,
    yolo_mode: bool,
    force_prompt: bool,
    needs_permission: bool,
    is_preapproved: bool,
    decision: Optional[PermissionDecision],
) -> PermissionPreview:
    """Pure decision engine for permission outcomes.

    This function intentionally has no side effects and performs no IO.
    """
    if yolo_mode and not force_prompt:
        return PermissionPreview(requires_user_input=False, result=PermissionResult(result=True))

    if is_preapproved:
        return PermissionPreview(requires_user_input=False, result=PermissionResult(result=True))

    resolved_decision = decision or _default_permission_decision(tool_name)

    if not needs_permission and resolved_decision.behavior != "ask" and not force_prompt:
        if resolved_decision.behavior == "deny":
            return PermissionPreview(
                requires_user_input=False,
                result=PermissionResult(
                    result=False,
                    message=_permission_denied_message(tool_name, resolved_decision),
                    decision=resolved_decision,
                ),
                decision=resolved_decision,
            )
        return PermissionPreview(
            requires_user_input=False,
            result=PermissionResult(
                result=True,
                message=resolved_decision.message,
                updated_input=resolved_decision.updated_input,
                decision=resolved_decision,
            ),
            decision=resolved_decision,
        )

    if resolved_decision.behavior == "allow" and not force_prompt:
        return PermissionPreview(
            requires_user_input=False,
            result=PermissionResult(
                result=True,
                message=resolved_decision.message,
                updated_input=resolved_decision.updated_input,
                decision=resolved_decision,
            ),
            decision=resolved_decision,
        )

    if resolved_decision.behavior == "deny":
        return PermissionPreview(
            requires_user_input=False,
            result=PermissionResult(
                result=False,
                message=_permission_denied_message(tool_name, resolved_decision),
                decision=resolved_decision,
            ),
            decision=resolved_decision,
        )

    return PermissionPreview(
        requires_user_input=True,
        result=None,
        decision=resolved_decision,
    )


def make_permission_checker(
    project_path: Path,
    yolo_mode: bool,
    prompt_fn: Optional[Callable[[str], str]] = None,
    console: Optional["Console"] = None,  # noqa: ARG001 (kept for backward compatibility)
    prompt_session: Optional["PromptSession"] = None,  # noqa: ARG001 (kept for backward compatibility)
) -> Callable[[Tool[Any, Any], Any], Awaitable[PermissionResult]]:
    """Create a permission checking function for the current project.

    Args:
        project_path: Path to the project directory
        yolo_mode: If True, all tool calls are allowed without prompting
        prompt_fn: Optional function to use for prompting (defaults to input())
        console: (Deprecated) No longer used, kept for backward compatibility
        prompt_session: (Deprecated) No longer used, kept for backward compatibility

    In yolo mode, all tool calls are allowed without prompting.
    """

    _ = console, prompt_session  # Mark as intentionally unused
    project_path = project_path.resolve()
    config_manager.get_project_config(project_path)

    session_allowed_tools: Set[str] = set()
    session_tool_rules: dict[str, Set[str]] = defaultdict(set)

    async def _prompt_user(prompt: str, options: list[tuple[str, str]]) -> str:
        """Prompt the user with proper interrupt handling using unified choice component.

        Args:
            prompt: The prompt text to display (supports HTML formatting).
            options: List of (value, label) tuples for choices.
        """
        loop = asyncio.get_running_loop()

        def _ask() -> str:
            try:
                # If a custom prompt_fn is provided (e.g., for tests), use it directly
                responder = prompt_fn or None
                if responder is not None:
                    # Build a simple text prompt for the prompt_fn
                    numeric_choices = "/".join(str(i) for i in range(1, len(options) + 1))
                    shortcut_choices = "/".join(opt[0] for opt in options)
                    input_prompt = f"Choice ({numeric_choices} or {shortcut_choices}): "
                    return responder(input_prompt)

                # Use the unified choice component
                return prompt_choice(
                    message=prompt,
                    options=options,
                    allow_esc=True,
                    esc_value="n",  # ESC means no
                )
            except KeyboardInterrupt:
                logger.debug("[permissions] KeyboardInterrupt in choice")
                return "n"
            except EOFError:
                logger.debug("[permissions] EOFError in choice")
                return "n"

        return await loop.run_in_executor(None, _ask)

    async def _compute_permission_preview(
        tool: Tool[Any, Any],
        parsed_input: Any,
        *,
        force_prompt: bool,
        log_errors: bool,
    ) -> PermissionPreview:
        """Shared non-interactive permission evaluation path."""
        config = config_manager.get_project_config(project_path)
        allowed_tools = set(config.allowed_tools or [])

        try:
            needs_permission = True
            if hasattr(tool, "needs_permissions"):
                needs_permission = tool.needs_permissions(parsed_input)
            if force_prompt:
                needs_permission = True
        except (TypeError, AttributeError, ValueError) as exc:
            if log_errors:
                logger.warning(
                    "[permissions] Tool needs_permissions check failed",
                    extra={
                        "tool": getattr(tool, "name", None),
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    },
                )
            return PermissionPreview(
                requires_user_input=False,
                result=PermissionResult(
                    result=False,
                    message=f"Permission check failed: {type(exc).__name__}: {exc}",
                ),
            )

        policy = _build_permission_policy(
            project_path=project_path,
            config=config,
            global_config=config_manager.get_global_config(),
            local_config=config_manager.get_project_local_config(project_path),
            session_tool_rules=session_tool_rules,
        )
        is_preapproved = tool.name in allowed_tools or tool.name in session_allowed_tools
        decision = None if is_preapproved else await _resolve_permission_decision(
            tool,
            parsed_input,
            policy=policy,
            log_errors=log_errors,
        )

        return _run_permission_decision_engine(
            tool_name=tool.name,
            yolo_mode=yolo_mode,
            force_prompt=force_prompt,
            needs_permission=needs_permission,
            is_preapproved=is_preapproved,
            decision=decision,
        )

    async def _evaluate_permission(
        tool: Tool[Any, Any], parsed_input: Any, *, force_prompt: bool = False
    ) -> PermissionResult:
        """Check and optionally persist permission for a tool invocation."""
        preview = await _compute_permission_preview(
            tool,
            parsed_input,
            force_prompt=force_prompt,
            log_errors=True,
        )
        if not preview.requires_user_input and preview.result is not None:
            return preview.result
        decision = preview.decision or _default_permission_decision(tool.name)

        # Ask/passthrough flows prompt the user.
        tool_input_dict = (
            parsed_input.model_dump()
            if hasattr(parsed_input, "model_dump")
            else dict(parsed_input)
            if isinstance(parsed_input, dict)
            else {}
        )
        try:
            permission_suggestions = _serialize_permission_suggestions(
                decision.rule_suggestions if decision else None
            )
            hook_result = await hook_manager.run_permission_request_async(
                tool.name, tool_input_dict, permission_suggestions=permission_suggestions
            )
            if hook_result.outputs:
                _apply_updated_permissions(
                    hook_result.updated_permissions,
                    default_tool_name=tool.name,
                    session_allowed_tools=session_allowed_tools,
                    session_tool_rules=session_tool_rules,
                )
                updated_input = hook_result.updated_input or decision.updated_input
                if hook_result.should_block or not hook_result.should_continue:
                    reason = (
                        hook_result.block_reason
                        or hook_result.stop_reason
                        or decision.message
                        or f"Permission denied for tool '{tool.name}'."
                    )
                    return PermissionResult(
                        result=False,
                        message=reason,
                        updated_input=updated_input,
                        decision=decision,
                    )
                if hook_result.should_allow:
                    return PermissionResult(
                        result=True,
                        message=decision.message,
                        updated_input=updated_input,
                        decision=decision,
                    )
        except (RuntimeError, ValueError, TypeError, OSError) as exc:
            logger.warning(
                "[permissions] PermissionRequest hook failed: %s: %s",
                type(exc).__name__,
                exc,
                extra={"tool": getattr(tool, "name", None)},
            )

        input_preview = _format_input_preview(parsed_input, tool_name=tool.name)
        # Use inline styles for prompt_toolkit HTML formatting
        # The style names must match keys in the _permission_style() dict
        prompt_html = f"""<title>{html.escape(tool.name)}</title>

  <description>{input_preview}</description>"""
        if decision.message:
            # Use warning style for warning messages
            prompt_html += f"\n  <warning>{html.escape(decision.message)}</warning>"
        prompt_html += "\n  <question>Do you want to proceed?</question>"
        prompt = prompt_html

        options = [
            ("y", "<yes-option>Yes</yes-option>"),
            ("s", "<yes-option>Yes, for this session</yes-option>"),
            ("n", "<no-option>No</no-option>"),
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
            message=_permission_denied_message(tool.name, decision),
            decision=decision,
        )

    async def _preview_permission(
        tool: Tool[Any, Any], parsed_input: Any, *, force_prompt: bool = False
    ) -> PermissionPreview:
        """Preview permission outcome without hooks or interactive prompt.

        This mirrors rule evaluation logic used by _evaluate_permission and is
        intended for SDK bridges that must decide whether user input is needed.
        """
        return await _compute_permission_preview(
            tool,
            parsed_input,
            force_prompt=force_prompt,
            log_errors=False,
        )

    async def can_use_tool(tool: Tool[Any, Any], parsed_input: Any) -> PermissionResult:
        return await _evaluate_permission(tool, parsed_input, force_prompt=False)

    async def _force_prompt(tool: Tool[Any, Any], parsed_input: Any) -> PermissionResult:
        return await _evaluate_permission(tool, parsed_input, force_prompt=True)

    async def _preview(tool: Tool[Any, Any], parsed_input: Any) -> PermissionPreview:
        return await _preview_permission(tool, parsed_input, force_prompt=False)

    async def _preview_force_prompt(tool: Tool[Any, Any], parsed_input: Any) -> PermissionPreview:
        return await _preview_permission(tool, parsed_input, force_prompt=True)

    setattr(can_use_tool, "force_prompt", _force_prompt)
    setattr(can_use_tool, "preview", _preview)
    setattr(can_use_tool, "preview_force_prompt", _preview_force_prompt)

    return can_use_tool
