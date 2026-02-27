"""Permission handling for tool execution."""

from __future__ import annotations

import asyncio
import difflib
import html
import os
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Iterable, Optional, Set

from ripperdoc.cli.ui.choice import prompt_choice
from ripperdoc.core.config import config_manager
from ripperdoc.core.hooks.manager import hook_manager
from ripperdoc.core.tool import Tool
from ripperdoc.tools.file_read_tool import detect_file_encoding
from ripperdoc.utils.diff_rendering import build_numbered_diff_layout, format_numbered_diff_text
from ripperdoc.utils.permissions import PermissionDecision, ToolRule
from ripperdoc.utils.permissions.rule_syntax import (
    ParsedPermissionRule,
    match_parsed_permission_rule,
    parse_permission_rule,
)
from ripperdoc.utils.log import get_logger

if TYPE_CHECKING:
    from rich.console import Console
    from prompt_toolkit import PromptSession

logger = get_logger()
_TOOL_RULE_HINT_RE = re.compile(r"^[A-Za-z0-9_-]+\s*\(.*\)\s*$")
_EDIT_PREVIEW_MAX_DIFF_LINES = 30
_EDIT_PREVIEW_MAX_BYTES = 1_500_000
_EDIT_PREVIEW_MATCH_SNIPPET_MAX = 140
_EDIT_PREVIEW_SEPARATOR = "-------------------"
_PERMISSION_PROMPT_RESERVED_LINES = 14
_PERMISSION_PROMPT_MIN_DIFF_LINES = 4
_PERMISSION_MODES = {"default", "acceptEdits", "plan", "bypassPermissions", "dontAsk"}


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

    if tool_name in {"Edit", "MultiEdit"} and hasattr(parsed_input, "file_path"):
        edit_preview = _build_edit_permission_preview(parsed_input, tool_name=tool_name)
        if edit_preview:
            return edit_preview

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


def _build_edit_permission_preview(parsed_input: Any, *, tool_name: str) -> str:
    """Render a before-apply preview for Edit/MultiEdit prompts."""
    file_path_raw = str(getattr(parsed_input, "file_path", "") or "")
    if not file_path_raw:
        return ""

    path = Path(file_path_raw).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()

    lines = [f"<label>file:</label> <value>{html.escape(str(path))}</value>"]

    if not path.exists():
        lines.append(
            "<warning>Preview unavailable: target file does not exist yet.</warning>"
        )
        return "\n  ".join(lines)

    if not path.is_file():
        lines.append("<warning>Preview unavailable: target path is not a file.</warning>")
        return "\n  ".join(lines)

    try:
        file_size = os.path.getsize(path)
    except OSError:
        file_size = None
    if file_size is not None and file_size > _EDIT_PREVIEW_MAX_BYTES:
        lines.append(
            f"<warning>Preview skipped: file is {file_size} bytes (> {_EDIT_PREVIEW_MAX_BYTES} bytes).</warning>"
        )
        return "\n  ".join(lines)

    try:
        detected_encoding, _ = detect_file_encoding(str(path))
        encoding = detected_encoding or "utf-8"
        with open(path, "r", encoding=encoding) as handle:
            original_content = handle.read()
    except (OSError, UnicodeDecodeError, LookupError) as exc:
        lines.append(f"<warning>Preview unavailable: {html.escape(str(exc))}</warning>")
        return "\n  ".join(lines)

    preview_result = _compute_edit_preview(
        original_content=original_content,
        parsed_input=parsed_input,
        tool_name=tool_name,
    )
    if preview_result["error"] is not None:
        lines.append(
            f"<warning>Preview unavailable: {html.escape(preview_result['error'])}</warning>"
        )
        return "\n  ".join(lines)

    diff_lines: list[str] = preview_result["diff_lines"]
    replacements = preview_result["replacements"]
    if not diff_lines:
        lines.append("<warning>No textual diff generated.</warning>")
        return "\n  ".join(lines)

    line_budget = _permission_preview_diff_line_budget()
    lines.append(
        f"<label>preview:</label> <value>{replacements} replacement(s), showing up to "
        f"{line_budget} diff lines</value>"
    )
    lines.append(f"<dim>{_EDIT_PREVIEW_SEPARATOR}</dim>")

    layout = build_numbered_diff_layout(diff_lines)
    clipped = layout.lines[:line_budget]
    for diff_line in clipped:
        rendered = format_numbered_diff_text(
            diff_line,
            old_width=layout.old_width,
            new_width=layout.new_width,
        )
        escaped_rendered = html.escape(rendered)
        if diff_line.kind == "hunk":
            lines.append(f"<diff-hunk>{escaped_rendered}</diff-hunk>")
            continue

        if diff_line.kind == "add":
            lines.append(f"<diff-add>{escaped_rendered}</diff-add>")
            continue
        if diff_line.kind == "del":
            lines.append(f"<diff-del>{escaped_rendered}</diff-del>")
            continue
        lines.append(f"<value>{escaped_rendered}</value>")

    if len(diff_lines) > line_budget:
        hidden = len(diff_lines) - line_budget
        lines.append(f"<dim>... ({hidden} more diff lines)</dim>")
    lines.append(f"<dim>{_EDIT_PREVIEW_SEPARATOR}</dim>")

    return "\n  ".join(lines)


def _permission_preview_diff_line_budget() -> int:
    """Compute diff preview line budget based on terminal height."""
    try:
        height = shutil.get_terminal_size(fallback=(80, 24)).lines
    except OSError:
        height = 24
    dynamic_budget = height - _PERMISSION_PROMPT_RESERVED_LINES
    dynamic_budget = max(_PERMISSION_PROMPT_MIN_DIFF_LINES, dynamic_budget)
    return min(_EDIT_PREVIEW_MAX_DIFF_LINES, dynamic_budget)


def _compact_preview_snippet(text: str, *, max_len: int = _EDIT_PREVIEW_MATCH_SNIPPET_MAX) -> str:
    """Short single-line snippet for permission preview messages."""
    single_line = text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n")
    if len(single_line) <= max_len:
        return single_line
    return single_line[: max_len - 3] + "..."


def _compute_edit_preview(
    *,
    original_content: str,
    parsed_input: Any,
    tool_name: str,
) -> dict[str, Any]:
    """Apply edit inputs in-memory and return diff preview payload."""
    operations = _normalize_edit_operations(parsed_input, tool_name=tool_name)
    if operations is None:
        return {"error": "invalid edit payload", "diff_lines": [], "replacements": 0}

    updated = original_content
    replacements = 0
    for op in operations:
        old = op["old_string"]
        new = op["new_string"]
        replace_all = op["replace_all"]

        if old == "":
            if updated != "":
                return {
                    "error": (
                        "empty old_string is only valid when creating content in an empty file"
                    ),
                    "diff_lines": [],
                    "replacements": 0,
                }
            updated = new
            replacements += 1 if new else 0
            continue

        occurrences = updated.count(old)
        if occurrences == 0:
            old_preview = _compact_preview_snippet(old)
            return {
                "error": f"old_string not found (snippet: {old_preview!r})",
                "diff_lines": [],
                "replacements": 0,
            }
        if not replace_all and occurrences > 1:
            return {
                "error": (
                    f"string appears {occurrences} times; provide a unique match or set replace_all=true"
                ),
                "diff_lines": [],
                "replacements": 0,
            }

        if replace_all:
            updated = updated.replace(old, new)
            replacements += occurrences
        else:
            updated = updated.replace(old, new, 1)
            replacements += 1

    diff = list(
        difflib.unified_diff(
            original_content.splitlines(keepends=True),
            updated.splitlines(keepends=True),
            fromfile="before",
            tofile="after",
            lineterm="",
        )
    )
    return {
        "error": None,
        "diff_lines": [line for line in diff[2:]],
        "replacements": replacements,
    }


def _normalize_edit_operations(parsed_input: Any, *, tool_name: str) -> Optional[list[dict[str, Any]]]:
    """Normalize Edit/MultiEdit payloads into a common in-memory operation list."""
    if tool_name == "Edit":
        return [
            {
                "old_string": str(getattr(parsed_input, "old_string", "")),
                "new_string": str(getattr(parsed_input, "new_string", "")),
                "replace_all": bool(getattr(parsed_input, "replace_all", False)),
            }
        ]

    if tool_name == "MultiEdit":
        edits = getattr(parsed_input, "edits", None)
        if edits is None:
            return None
        normalized: list[dict[str, Any]] = []
        for edit in edits:
            normalized.append(
                {
                    "old_string": str(getattr(edit, "old_string", "")),
                    "new_string": str(getattr(edit, "new_string", "")),
                    "replace_all": bool(getattr(edit, "replace_all", False)),
                }
            )
        return normalized

    return None


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
                    "tool_name": suggestion.tool_name,
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
            parsed = parse_permission_rule(
                entry,
                default_tool_name=default_tool_name,
                known_tool_names={default_tool_name},
            )
            if parsed is None:
                return
            if parsed.specifier is None:
                session_allowed_tools.add(parsed.tool_name)
                return
            session_tool_rules.setdefault(parsed.tool_name, set()).add(parsed.specifier)
            return
        if not isinstance(entry, dict):
            return

        tool_name = entry.get("tool_name") or entry.get("tool") or default_tool_name
        behavior = (entry.get("behavior") or entry.get("decision") or "allow").lower()
        rule = entry.get("rule") or entry.get("rule_content")

        if behavior != "allow":
            return

        if not isinstance(tool_name, str) or not tool_name:
            return

        if isinstance(rule, str) and rule.strip():
            if _TOOL_RULE_HINT_RE.match(rule.strip()):
                parsed = parse_permission_rule(
                    rule,
                    default_tool_name=tool_name,
                    known_tool_names={tool_name},
                )
            else:
                parsed = parse_permission_rule(
                    f"{tool_name}({rule})",
                    default_tool_name=tool_name,
                    known_tool_names={tool_name},
                )
            if parsed is None:
                return
            if parsed.specifier is None:
                session_allowed_tools.add(parsed.tool_name)
                return
            session_tool_rules.setdefault(parsed.tool_name, set()).add(parsed.specifier)
            return

        if isinstance(tool_name, str) and tool_name:
            session_allowed_tools.add(tool_name)

    if isinstance(updated_permissions, list):
        for entry in updated_permissions:
            _apply_entry(entry)
        return

    if isinstance(updated_permissions, dict):
        allowed_tools = updated_permissions.get("allowed_tools")
        if isinstance(allowed_tools, list):
            session_allowed_tools.update(
                {str(name).strip() for name in allowed_tools if str(name).strip()}
            )

        bash_allow = updated_permissions.get("bash_allow_rules")
        if isinstance(bash_allow, list):
            for rule in bash_allow:
                _apply_entry({"tool_name": "Bash", "rule": rule, "behavior": "allow"})

        allow_rules = updated_permissions.get("allow_rules")
        if isinstance(allow_rules, list):
            for rule in allow_rules:
                _apply_entry(rule)

        if any(k in updated_permissions for k in ("tool_name", "tool", "rule")):
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


def _parse_rule_collection(rules: Iterable[str]) -> list[ParsedPermissionRule]:
    parsed_rules: list[ParsedPermissionRule] = []
    for rule in rules:
        parsed = parse_permission_rule(rule)
        if parsed is not None:
            parsed_rules.append(parsed)
    return parsed_rules


def _extract_tool_specifier_rules(
    parsed_rules: Iterable[ParsedPermissionRule], tool_name: str
) -> set[str]:
    specifiers: set[str] = set()
    for rule in parsed_rules:
        if rule.tool_name != tool_name or rule.specifier is None:
            continue
        specifiers.add(rule.specifier)
    return specifiers


def _session_rule_to_raw(tool_name: str, rule: str) -> str:
    parsed = parse_permission_rule(rule, known_tool_names={tool_name})
    if parsed is not None and parsed.tool_name == tool_name:
        return parsed.canonical_rule
    if tool_name == "Bash":
        bash_parsed = parse_permission_rule(rule, known_tool_names={"Bash"})
        if bash_parsed is not None:
            return bash_parsed.canonical_rule
        return "Bash"
    return f"{tool_name}({str(rule).strip()})"


def _explicit_rule_decision(
    *,
    tool_name: str,
    parsed_input: Any,
    rules: Iterable[ParsedPermissionRule],
    behavior: str,
    project_path: Path,
) -> Optional[PermissionDecision]:
    for rule in sorted(rules, key=lambda item: item.canonical_rule):
        if match_parsed_permission_rule(
            rule,
            tool_name=tool_name,
            parsed_input=parsed_input,
            cwd=project_path,
        ):
            if behavior == "deny":
                return PermissionDecision(
                    behavior="deny",
                    message=f"Permission denied by rule: {rule.canonical_rule}",
                    decision_reason={"type": "rule", "rule": rule.canonical_rule},
                )
            if behavior == "ask":
                return PermissionDecision(
                    behavior="ask",
                    message=f"Command requires confirmation by rule: {rule.canonical_rule}",
                    decision_reason={"type": "rule", "rule": rule.canonical_rule},
                )
            return PermissionDecision(
                behavior="allow",
                message=f"Approved by rule: {rule.canonical_rule}",
                decision_reason={"type": "rule", "rule": rule.canonical_rule},
            )
    return None


def _resolve_explicit_rule_decision(
    *,
    tool_name: str,
    parsed_input: Any,
    policy: dict[str, Any],
) -> Optional[PermissionDecision]:
    project_path = policy["project_path"]
    deny_decision = _explicit_rule_decision(
        tool_name=tool_name,
        parsed_input=parsed_input,
        rules=policy.get("parsed_deny_rules", []),
        behavior="deny",
        project_path=project_path,
    )
    if deny_decision is not None:
        return deny_decision

    ask_decision = _explicit_rule_decision(
        tool_name=tool_name,
        parsed_input=parsed_input,
        rules=policy.get("parsed_ask_rules", []),
        behavior="ask",
        project_path=project_path,
    )
    if ask_decision is not None:
        return ask_decision

    return _explicit_rule_decision(
        tool_name=tool_name,
        parsed_input=parsed_input,
        rules=policy.get("parsed_allow_rules", []),
        behavior="allow",
        project_path=project_path,
    )


def _build_permission_policy(
    *,
    project_path: Path,
    config: Any,
    global_config: Any,
    local_config: Any,
    session_tool_rules: dict[str, Set[str]],
    session_working_dirs: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Build merged permission policy inputs for tool-level evaluation."""
    raw_allow_rules = (
        set(config.bash_allow_rules or [])
        | set(global_config.user_allow_rules or [])
        | set(local_config.local_allow_rules or [])
    )
    for tool_name, tool_rules in session_tool_rules.items():
        for rule in tool_rules:
            raw_allow_rules.add(_session_rule_to_raw(tool_name, rule))

    raw_deny_rules = (
        set(config.bash_deny_rules or [])
        | set(global_config.user_deny_rules or [])
        | set(local_config.local_deny_rules or [])
    )
    raw_ask_rules = (
        set(config.bash_ask_rules or [])
        | set(global_config.user_ask_rules or [])
        | set(local_config.local_ask_rules or [])
    )

    parsed_allow_rules = _parse_rule_collection(raw_allow_rules)
    parsed_deny_rules = _parse_rule_collection(raw_deny_rules)
    parsed_ask_rules = _parse_rule_collection(raw_ask_rules)

    # Keep per-tool Bash specifier sets for BashTool's internal security heuristics.
    allow_rules = {"Bash": _extract_tool_specifier_rules(parsed_allow_rules, "Bash")}
    deny_rules = {"Bash": _extract_tool_specifier_rules(parsed_deny_rules, "Bash")}
    ask_rules = {"Bash": _extract_tool_specifier_rules(parsed_ask_rules, "Bash")}

    allowed_working_dirs = {str(project_path.resolve())}
    for raw_path in config.working_directories or []:
        try:
            path = Path(raw_path).expanduser()
            if not path.is_absolute():
                path = project_path / path
            allowed_working_dirs.add(str(path.resolve()))
        except (OSError, RuntimeError, ValueError):
            continue
    for raw_path in session_working_dirs or []:
        try:
            path = Path(raw_path).expanduser()
            if not path.is_absolute():
                path = project_path / path
            allowed_working_dirs.add(str(path.resolve()))
        except (OSError, RuntimeError, ValueError):
            continue

    return {
        "allow_rules": allow_rules,
        "deny_rules": deny_rules,
        "ask_rules": ask_rules,
        "raw_allow_rules": raw_allow_rules,
        "raw_deny_rules": raw_deny_rules,
        "raw_ask_rules": raw_ask_rules,
        "parsed_allow_rules": parsed_allow_rules,
        "parsed_deny_rules": parsed_deny_rules,
        "parsed_ask_rules": parsed_ask_rules,
        "allowed_working_dirs": allowed_working_dirs,
        "project_path": project_path,
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
    explicit_rule_decision = _resolve_explicit_rule_decision(
        tool_name=tool.name,
        parsed_input=parsed_input,
        policy=policy,
    )
    if explicit_rule_decision is not None:
        return explicit_rule_decision

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


def make_permission_checker(
    project_path: Path,
    yolo_mode: bool,
    permission_mode: str = "default",
    is_bypass_permissions_mode_available: Optional[bool] = None,
    prompt_fn: Optional[Callable[[str], str]] = None,
    console: Optional["Console"] = None,  # noqa: ARG001 (kept for backward compatibility)
    prompt_session: Optional["PromptSession"] = None,  # noqa: ARG001 (kept for backward compatibility)
    session_additional_working_dirs: Optional[Iterable[str]] = None,
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

    session_allowed_tools: Set[str] = set()
    session_tool_rules: dict[str, Set[str]] = defaultdict(set)
    session_working_dirs: Set[str] = set()
    for raw_path in session_additional_working_dirs or []:
        try:
            path = Path(str(raw_path)).expanduser()
            if not path.is_absolute():
                path = project_path / path
            session_working_dirs.add(str(path.resolve()))
        except (OSError, RuntimeError, ValueError):
            continue

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
            session_working_dirs=session_working_dirs,
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

    def _add_working_directory(path: str) -> str | None:
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

    def _list_working_directories() -> set[str]:
        """Return session-scoped additional working directories."""
        return set(session_working_dirs)

    setattr(can_use_tool, "add_working_directory", _add_working_directory)
    setattr(can_use_tool, "list_working_directories", _list_working_directories)

    return can_use_tool
