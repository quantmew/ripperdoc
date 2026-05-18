"""Permission rule and policy construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from ripperdoc.services.managed_settings import get_managed_setting
from ripperdoc.utils.permissions import PermissionDecision
from ripperdoc.utils.permissions.rule_syntax import (
    ParsedPermissionRule,
    match_parsed_permission_rule,
    parse_permission_rule,
)

from .constants import _TOOL_RULE_HINT_RE


def _as_str_set(raw: Any) -> Set[str]:
    if isinstance(raw, str):
        value = raw.strip()
        return {value} if value else set()
    if not isinstance(raw, (list, tuple, set)):
        return set()
    out: Set[str] = set()
    for item in raw:
        value = str(item or "").strip()
        if value:
            out.add(value)
    return out


def _managed_permissions_only_enabled() -> bool:
    raw = get_managed_setting("managedPermissionsOnly")
    if raw is None:
        raw = get_managed_setting("managed_permissions_only")
    return bool(raw)

def _apply_updated_permissions(
    updated_permissions: Any,
    *,
    default_tool_name: str,
    session_allowed_tools: Set[str],
    session_tool_rules: Dict[str, Set[str]],
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



def _parse_rule_collection(rules: Iterable[str]) -> List[ParsedPermissionRule]:
    parsed_rules: List[ParsedPermissionRule] = []
    for rule in rules:
        parsed = parse_permission_rule(rule)
        if parsed is not None:
            parsed_rules.append(parsed)
    return parsed_rules


def _extract_tool_specifier_rules(
    parsed_rules: Iterable[ParsedPermissionRule], tool_name: str
) -> Set[str]:
    specifiers: Set[str] = set()
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
    policy: Dict[str, Any],
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
    session_tool_rules: Dict[str, Set[str]],
    session_working_dirs: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Build merged permission policy inputs for tool-level evaluation."""
    managed_permissions_only = _managed_permissions_only_enabled()
    managed_allow_rules = (
        _as_str_set(get_managed_setting("managedAllowRules"))
        | _as_str_set(get_managed_setting("user_allow_rules"))
    )
    managed_deny_rules = (
        _as_str_set(get_managed_setting("managedDenyRules"))
        | _as_str_set(get_managed_setting("user_deny_rules"))
    )
    managed_ask_rules = (
        _as_str_set(get_managed_setting("managedAskRules"))
        | _as_str_set(get_managed_setting("user_ask_rules"))
    )

    if managed_permissions_only:
        raw_allow_rules = set(managed_allow_rules)
        raw_deny_rules = set(managed_deny_rules)
        raw_ask_rules = set(managed_ask_rules)
    else:
        raw_allow_rules = (
            set(config.bash_allow_rules or [])
            | set(global_config.user_allow_rules or [])
            | set(local_config.local_allow_rules or [])
            | managed_allow_rules
        )
        for tool_name, tool_rules in session_tool_rules.items():
            for rule in tool_rules:
                raw_allow_rules.add(_session_rule_to_raw(tool_name, rule))

        raw_deny_rules = (
            set(config.bash_deny_rules or [])
            | set(global_config.user_deny_rules or [])
            | set(local_config.local_deny_rules or [])
            | managed_deny_rules
        )
        raw_ask_rules = (
            set(config.bash_ask_rules or [])
            | set(global_config.user_ask_rules or [])
            | set(local_config.local_ask_rules or [])
            | managed_ask_rules
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
        "permission_mode": getattr(config, "permission_mode", "default"),
        "managed_permissions_only": managed_permissions_only,
    }
