"""Pure permission decision logic."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, Optional

from ripperdoc.core.plan_mode import is_plan_file_path
from ripperdoc.core.tool import Tool
from ripperdoc.utils.log import get_logger
from ripperdoc.utils.memory import is_auto_memory_enabled, is_auto_memory_path
from ripperdoc.utils.permissions import PermissionDecision, ToolRule

from .constants import (
    _AUTO_MEMORY_WRITE_TOOLS,
    _PERMISSION_MODES,
    _PLAN_MODE_PLAN_FILE_EDIT_TOOLS,
    _PLAN_MODE_SPECIAL_ALLOWED_TOOLS,
)
from .models import PermissionPreview, PermissionResult
from .policy import _resolve_explicit_rule_decision

logger = get_logger()


def _extract_tool_target_path(tool_name: str, parsed_input: Any) -> Optional[str]:
    """Return the primary filesystem target path for mutating file tools."""

    if tool_name in {"Write", "Edit", "MultiEdit"} and hasattr(parsed_input, "file_path"):
        return str(getattr(parsed_input, "file_path"))
    if tool_name == "NotebookEdit" and hasattr(parsed_input, "notebook_path"):
        return str(getattr(parsed_input, "notebook_path"))
    return None


def _permission_request_allow_can_override(decision: PermissionDecision) -> bool:
    if decision.behavior == "deny":
        return False
    return not _is_rule_ask_decision(decision)


def _plan_mode_restriction_result(
    *,
    tool: Tool[Any, Any],
    parsed_input: Any,
    plan_file_path: Optional[str],
) -> Optional[PermissionPreview]:
    """Apply plan-mode write restrictions.

    In plan mode, only read-only tools, AskUserQuestion, ExitPlanMode, and edits
    to the active plan file are allowed. All other mutating operations are denied.
    """

    tool_name = str(getattr(tool, "name", "") or "")
    if tool.is_read_only():
        return None
    if tool_name in _PLAN_MODE_SPECIAL_ALLOWED_TOOLS:
        return None

    target_path = _extract_tool_target_path(tool_name, parsed_input)
    if (
        tool_name in _PLAN_MODE_PLAN_FILE_EDIT_TOOLS
        and is_plan_file_path(target_path, plan_file_path)
    ):
        return None

    plan_suffix = f" except the active plan file ({plan_file_path})" if plan_file_path else ""
    return PermissionPreview(
        requires_user_input=False,
        result=PermissionResult(
            result=False,
            message=(
                "Plan mode is read-only. "
                f"Tool '{tool_name}' is blocked while planning{plan_suffix}."
            ),
        ),
    )


def _extract_input_file_path(parsed_input: Any) -> Optional[str]:
    """Extract a candidate file path from a tool input payload."""
    if hasattr(parsed_input, "file_path"):
        file_path = getattr(parsed_input, "file_path")
        if isinstance(file_path, str) and file_path.strip():
            return file_path

    if isinstance(parsed_input, dict):
        file_path = parsed_input.get("file_path") or parsed_input.get("path")
        if isinstance(file_path, str) and file_path.strip():
            return file_path

    return None


def _resolve_auto_memory_write_decision(
    *,
    tool_name: str,
    parsed_input: Any,
    project_path: Path,
) -> Optional[PermissionDecision]:
    """Auto-allow write/edit operations that target auto-memory files."""
    if tool_name not in _AUTO_MEMORY_WRITE_TOOLS:
        return None

    file_path = _extract_input_file_path(parsed_input)
    if not file_path:
        return None

    try:
        if not is_auto_memory_enabled(project_path):
            return None
        if not is_auto_memory_path(file_path, project_path=project_path):
            return None
    except Exception:
        return None

    return PermissionDecision(
        behavior="allow",
        message="Auto memory files are allowed for writing.",
        decision_reason={"type": "auto_memory", "path": file_path},
    )


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


def _dont_ask_permission_denied_message(tool_name: str) -> str:
    """Return deny message used when running in dontAsk mode."""
    return (
        f"Permission denied for tool '{tool_name}' because permission mode is dontAsk."
    )


def _normalize_permission_mode(mode: str) -> str:
    normalized = str(mode or "").strip()
    if normalized in _PERMISSION_MODES:
        return normalized
    return "default"


def _is_rule_ask_decision(decision: PermissionDecision) -> bool:
    if decision.behavior != "ask":
        return False
    reason = decision.decision_reason or {}
    return isinstance(reason, dict) and reason.get("type") == "rule"



def _coerce_permission_decision(raw_decision: Any) -> Optional[PermissionDecision]:
    """Normalize tool-provided permission decision payloads."""
    if isinstance(raw_decision, PermissionDecision):
        return raw_decision

    # Handle security.PermissionResult from bash tool's check_permissions
    from ripperdoc.security import PermissionResult as SecurityPermissionResult
    if isinstance(raw_decision, SecurityPermissionResult):
        return PermissionDecision(
            behavior=raw_decision.behavior,
            message=raw_decision.message or None,
            updated_input=raw_decision.updated_input,
            decision_reason=raw_decision.decision_reason,
            rule_suggestions=raw_decision.suggestions,
        )

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
    policy: Dict[str, Any],
    log_errors: bool,
) -> PermissionDecision:
    """Resolve the tool decision from tool policy hooks/checkers."""
    explicit_rule_decision = _resolve_explicit_rule_decision(
        tool_name=tool.name,
        parsed_input=parsed_input,
        policy=policy,
    )
    if explicit_rule_decision is not None:
        return explicit_rule_decision

    auto_memory_decision = _resolve_auto_memory_write_decision(
        tool_name=tool.name,
        parsed_input=parsed_input,
        project_path=policy["project_path"],
    )
    if auto_memory_decision is not None:
        return auto_memory_decision

    if not hasattr(tool, "check_permissions"):
        return _default_permission_decision(tool.name)

    permission_context = {
        "mode": policy["permission_mode"],
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
                "[permissions] Tool %s check_permissions failed: %s: %s",
                getattr(tool, "name", None),
                type(exc).__name__,
                exc,
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
    permission_mode: str,
    is_bypass_permissions_mode_available: bool,
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

    if (
        permission_mode == "plan"
        and is_bypass_permissions_mode_available
        and not force_prompt
        and not _is_rule_ask_decision(resolved_decision)
    ):
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

    if permission_mode == "dontAsk":
        return PermissionPreview(
            requires_user_input=False,
            result=PermissionResult(
                result=False,
                message=_dont_ask_permission_denied_message(tool_name),
                decision=resolved_decision,
            ),
            decision=resolved_decision,
        )

    return PermissionPreview(
        requires_user_input=True,
        result=None,
        decision=resolved_decision,
    )

