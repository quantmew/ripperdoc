"""Permissions and security helpers for Bash tool."""

from __future__ import annotations

from typing import Any, Optional

from ripperdoc.utils.permissions.path_validation_utils import validate_shell_command_paths
from ripperdoc.utils.permissions.tool_permission_utils import (
    evaluate_shell_command_permissions,
    is_command_read_only,
)
from ripperdoc.utils.permissions import PermissionDecision
from ripperdoc.utils.filesystem.safe_get_cwd import safe_get_cwd
from ripperdoc.tools.bash._models import BashToolInput


def is_background_allowed(command: str) -> bool:
    """Skip backgrounding trivial ignored commands unless combined with other operators."""
    from ripperdoc.utils.shell.exit_code_handlers import IGNORED_COMMANDS

    normalized = command.strip()
    if not normalized:
        return True

    if any(op in normalized for op in ("&&", "||", "|", ";")):
        return True

    parts = normalized.split(maxsplit=1)
    if normalized in IGNORED_COMMANDS:
        return False
    if len(parts) == 1 and parts[0] in IGNORED_COMMANDS:
        return False
    return True


def detect_auto_background(command: str) -> tuple[str, bool]:
    """Detect trailing '&' requests and strip them for execution."""
    stripped = command.rstrip()
    if not stripped:
        return command, False

    if stripped.endswith("&") and not stripped.endswith("&&"):
        cleaned = stripped.rstrip("&").rstrip()
        return cleaned, True

    return command, False


async def check_permissions(
    input_data: BashToolInput, permission_context: dict[str, Any]
) -> Any:
    """Evaluate permissions using reference-style rules."""
    sandbox_requested = bool(getattr(input_data, "sandbox", False)) and not bool(
        getattr(input_data, "dangerously_disable_sandbox", False)
    )
    if sandbox_requested:
        return {"behavior": "allow", "updated_input": input_data}

    allow_rules = permission_context.get("allowed_rules") or set()
    deny_rules = permission_context.get("denied_rules") or set()
    ask_rules = permission_context.get("ask_rules") or set()
    allowed_dirs = permission_context.get("allowed_working_directories") or {safe_get_cwd()}

    cwd = safe_get_cwd()
    path_validation = validate_shell_command_paths(
        input_data.command,
        cwd,
        allowed_dirs,
    )
    if path_validation.behavior == "ask":
        return PermissionDecision(
            behavior="ask",
            message=path_validation.message,
            updated_input=input_data,
            decision_reason={"type": "sensitive_directory_access"},
            rule_suggestions=path_validation.rule_suggestions,
        )

    decision = evaluate_shell_command_permissions(
        input_data,
        allow_rules,
        deny_rules,
        ask_rules,
        allowed_dirs,
    )

    _, auto_background = detect_auto_background(input_data.command)
    if (input_data.run_in_background or auto_background) and getattr(
        decision, "behavior", None
    ) == "allow":
        reason = getattr(decision, "decision_reason", {}) or {}
        if reason.get("type") != "rule":
            return PermissionDecision(
                behavior="ask",
                message="Background bash commands require explicit approval.",
                updated_input=getattr(decision, "updated_input", None) or input_data,
                decision_reason=reason or None,
                rule_suggestions=getattr(decision, "rule_suggestions", None),
            )

    return decision
