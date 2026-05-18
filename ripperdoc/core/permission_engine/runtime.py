"""Runtime permission checker assembly."""

from __future__ import annotations

import asyncio
import html
import sys
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, Iterable, List, Optional, Set, Tuple

from ripperdoc.cli.ui.choice import prompt_choice as default_prompt_choice
from ripperdoc.core.config import config_manager
from ripperdoc.core.hooks.manager import hook_manager
from ripperdoc.core.tool import Tool
from ripperdoc.utils.log import get_logger

from .decision import (
    _default_permission_decision,
    _normalize_permission_mode,
    _permission_denied_message,
    _permission_request_allow_can_override,
    _plan_mode_restriction_result,
    _resolve_permission_decision,
    _run_permission_decision_engine,
)
from .keys import _rule_strings, _serialize_permission_suggestions, permission_key
from .models import PermissionPreview, PermissionResult
from .policy import _apply_updated_permissions, _build_permission_policy
from .preview import _format_input_preview

if TYPE_CHECKING:
    from rich.console import Console
    from prompt_toolkit import PromptSession

logger = get_logger()


def _compat_prompt_choice():
    compat_module = sys.modules.get("ripperdoc.core.permission_engine")
    return getattr(compat_module, "prompt_choice", default_prompt_choice)


def make_permission_checker(
    project_path: Path,
    yolo_mode: bool,
    permission_mode: str = "default",
    is_bypass_permissions_mode_available: Optional[bool] = None,
    plan_file_path: Optional[str] = None,
    prompt_fn: Optional[Callable[[str], str]] = None,
    console: Optional["Console"] = None,  # noqa: ARG001 (kept for backward compatibility)
    prompt_session: Optional["PromptSession"] = None,  # noqa: ARG001 (kept for backward compatibility)
    session_additional_working_dirs: Optional[Iterable[str]] = None,
    session_allowed_tools: Optional[Iterable[str]] = None,
    session_disallowed_tools: Optional[Iterable[str]] = None,
) -> Callable[[Tool[Any, Any], Any], Awaitable[PermissionResult]]:
    """Create a permission checking function for the current project.

    Args:
        project_path: Path to the project directory
        yolo_mode: If True, all tool calls are allowed without prompting
        permission_mode: Permission mode for mode-specific behavior (e.g. dontAsk)
        is_bypass_permissions_mode_available: Whether plan mode can auto-bypass prompts
        prompt_fn: Optional function to use for prompting (defaults to input())
        console: (Deprecated) No longer used, kept for backward compatibility
        prompt_session: (Deprecated) No longer used, kept for backward compatibility

    In yolo mode, all tool calls are allowed without prompting.
    """

    _ = console, prompt_session  # Mark as intentionally unused
    project_path = project_path.resolve()
    permission_mode = _normalize_permission_mode(permission_mode)
    if is_bypass_permissions_mode_available is None:
        effective_config = config_manager.get_effective_config(project_path)
        is_bypass_permissions_mode_available = not bool(
            getattr(effective_config, "disable_bypass_permissions_mode", False)
        )
    bypass_permissions_mode_available = bool(is_bypass_permissions_mode_available)
    config_manager.get_project_config(project_path)

    session_allowed_tools_set: Set[str] = {
        str(name).strip()
        for name in (session_allowed_tools or [])
        if str(name).strip()
    }
    session_disallowed_tools_set: Set[str] = {
        str(name).strip()
        for name in (session_disallowed_tools or [])
        if str(name).strip()
    }
    session_tool_rules: Dict[str, Set[str]] = defaultdict(set)
    session_working_dirs: Set[str] = set()
    for raw_path in session_additional_working_dirs or []:
        try:
            path = Path(str(raw_path)).expanduser()
            if not path.is_absolute():
                path = project_path / path
            session_working_dirs.add(str(path.resolve()))
        except (OSError, RuntimeError, ValueError):
            continue

    async def _prompt_user(prompt: str, options: List[Tuple[str, str]]) -> str:
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
                return _compat_prompt_choice()(
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

        if permission_mode == "plan":
            plan_mode_preview = _plan_mode_restriction_result(
                tool=tool,
                parsed_input=parsed_input,
                plan_file_path=plan_file_path,
            )
            if plan_mode_preview is not None:
                return plan_mode_preview

        # Auto-deny takes precedence over auto-approve.
        if tool.name in session_disallowed_tools_set:
            return PermissionPreview(
                requires_user_input=False,
                result=PermissionResult(
                    result=False,
                    message=f"Tool '{tool.name}' is disallowed by session configuration.",
                ),
            )

        policy = _build_permission_policy(
            project_path=project_path,
            config=config,
            global_config=config_manager.get_global_config(),
            local_config=config_manager.get_project_local_config(project_path),
            session_tool_rules=session_tool_rules,
            session_working_dirs=session_working_dirs,
        )
        policy["permission_mode"] = permission_mode
        if policy.get("managed_permissions_only"):
            is_preapproved = False
        else:
            is_preapproved = tool.name in allowed_tools or tool.name in session_allowed_tools_set
        decision = None if is_preapproved else await _resolve_permission_decision(
            tool,
            parsed_input,
            policy=policy,
            log_errors=log_errors,
        )

        return _run_permission_decision_engine(
            tool_name=tool.name,
            yolo_mode=yolo_mode,
            permission_mode=permission_mode,
            is_bypass_permissions_mode_available=bypass_permissions_mode_available,
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
                    session_allowed_tools=session_allowed_tools_set,
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
                if hook_result.should_allow and _permission_request_allow_can_override(decision):
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
        # Append destructive-command warning for Bash if present
        if tool.name == "Bash":
            from ripperdoc.tools.bash.destructive_warning import get_destructive_command_warning
            command = getattr(parsed_input, "command", "") or ""
            destructive_warn = get_destructive_command_warning(command)
            if destructive_warn:
                prompt_html += f"\n  <warning>{html.escape(destructive_warn)}</warning>"
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
                session_allowed_tools_set.add(tool.name)
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

    def _add_working_directory(path: str) -> Optional[str]:
        """Add a session-scoped allowed working directory."""
        text = str(path).strip()
        if not text:
            return None
        try:
            resolved = Path(text).expanduser()
            if not resolved.is_absolute():
                resolved = project_path / resolved
            resolved_str = str(resolved.resolve())
        except (OSError, RuntimeError, ValueError):
            return None
        session_working_dirs.add(resolved_str)
        return resolved_str

    def _list_working_directories() -> Set[str]:
        """Return session-scoped additional working directories."""
        return set(session_working_dirs)

    setattr(can_use_tool, "add_working_directory", _add_working_directory)
    setattr(can_use_tool, "list_working_directories", _list_working_directories)

    return can_use_tool
