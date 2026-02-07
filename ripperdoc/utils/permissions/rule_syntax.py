"""Permission rule syntax parsing and matching helpers."""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import urlparse

from ripperdoc.utils.permissions.tool_permission_utils import match_rule


_TOOL_WITH_SPEC_RE = re.compile(r"^\s*([A-Za-z0-9_-]+)\s*\((.*)\)\s*$")
_TOOL_ONLY_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_WILDCARD_CHARS = {"*", "?", "["}


@dataclass(frozen=True)
class ParsedPermissionRule:
    """Parsed form of a permission rule."""

    tool_name: str
    specifier: Optional[str]
    canonical_rule: str
    used_legacy_bash_suffix: bool = False


def _is_wildcard_pattern(value: str) -> bool:
    return any(ch in value for ch in _WILDCARD_CHARS)


def normalize_legacy_bash_wildcard(specifier: str) -> tuple[str, bool]:
    """Convert deprecated `:*` wildcard suffix syntax to glob ` *`.

    Examples:
        "ls:*" -> "ls *"
        "git add:*" -> "git add *"
    """
    normalized = re.sub(r"(?<!/):\*", " *", specifier)
    return normalized, normalized != specifier


def _looks_like_tool_name(
    token: str, known_tool_names: Optional[Iterable[str]] = None, default_tool_name: str = "Bash"
) -> bool:
    known = {name for name in (known_tool_names or []) if isinstance(name, str)}
    if token in known:
        return True
    if token == default_tool_name:
        return True
    # Claude-style built-in tool names are typically capitalized.
    return bool(token) and token[0].isupper()


def parse_permission_rule(
    rule: str,
    *,
    default_tool_name: str = "Bash",
    known_tool_names: Optional[Iterable[str]] = None,
) -> Optional[ParsedPermissionRule]:
    """Parse a permission rule into `Tool` or `Tool(specifier)` form.

    Backward compatibility:
    - Bare values that do not look like tool names are treated as Bash specifiers.
    - Deprecated Bash `:*` suffix syntax is accepted and normalized.
    """
    text = str(rule).strip()
    if not text:
        return None

    match = _TOOL_WITH_SPEC_RE.match(text)
    if match:
        tool_name = match.group(1).strip()
        inner = match.group(2).strip()
        used_legacy = False
        if tool_name == "Bash" and inner:
            inner, used_legacy = normalize_legacy_bash_wildcard(inner)
            inner = inner.strip()
        specifier = inner or None
        if specifier == "*":
            specifier = None
        canonical = tool_name if specifier is None else f"{tool_name}({specifier})"
        return ParsedPermissionRule(
            tool_name=tool_name,
            specifier=specifier,
            canonical_rule=canonical,
            used_legacy_bash_suffix=used_legacy,
        )

    if _TOOL_ONLY_RE.match(text) and _looks_like_tool_name(
        text, known_tool_names=known_tool_names, default_tool_name=default_tool_name
    ):
        return ParsedPermissionRule(tool_name=text, specifier=None, canonical_rule=text)

    used_legacy = False
    bash_specifier = text
    if default_tool_name == "Bash":
        bash_specifier, used_legacy = normalize_legacy_bash_wildcard(bash_specifier)
        bash_specifier = bash_specifier.strip()
    specifier = bash_specifier or None
    if specifier == "*":
        specifier = None
    canonical = default_tool_name if specifier is None else f"{default_tool_name}({specifier})"
    return ParsedPermissionRule(
        tool_name=default_tool_name,
        specifier=specifier,
        canonical_rule=canonical,
        used_legacy_bash_suffix=used_legacy,
    )


def normalize_permission_rule(
    rule: str,
    *,
    default_tool_name: str = "Bash",
    known_tool_names: Optional[Iterable[str]] = None,
) -> str:
    """Return the canonical text form of a permission rule."""
    parsed = parse_permission_rule(
        rule,
        default_tool_name=default_tool_name,
        known_tool_names=known_tool_names,
    )
    return parsed.canonical_rule if parsed else str(rule).strip()


def _get_attr_or_key(data: Any, field: str) -> Optional[str]:
    if isinstance(data, dict):
        value = data.get(field)
        return str(value) if isinstance(value, (str, Path)) else None
    if hasattr(data, field):
        value = getattr(data, field)
        return str(value) if isinstance(value, (str, Path)) else None
    return None


def _extract_url_value(parsed_input: Any) -> Optional[str]:
    for field in ("url", "uri"):
        value = _get_attr_or_key(parsed_input, field)
        if value:
            return value.strip()
    return None


def _extract_domain(parsed_input: Any) -> Optional[str]:
    url_value = _extract_url_value(parsed_input)
    if not url_value:
        return None
    parsed = urlparse(url_value)
    if parsed.hostname:
        return parsed.hostname.lower()
    guessed = url_value.split("/")[0].strip().lower()
    return guessed or None


def _normalize_path_text(value: str) -> str:
    return value.replace("\\", "/")


def _path_candidates(path_value: str, cwd: Optional[Path]) -> set[str]:
    candidates = {_normalize_path_text(path_value.strip())}
    try:
        resolved = Path(path_value).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        resolved = None
    if resolved is not None:
        abs_text = _normalize_path_text(str(resolved))
        candidates.add(abs_text)
        if cwd is not None:
            try:
                rel = resolved.relative_to(cwd.resolve())
                rel_text = _normalize_path_text(str(rel))
                candidates.add(rel_text)
                candidates.add(f"./{rel_text}" if rel_text else ".")
            except (OSError, RuntimeError, ValueError):
                pass
    return {c for c in candidates if c}


def _matches_text_specifier(specifier: str, values: Iterable[str]) -> bool:
    if _is_wildcard_pattern(specifier):
        return any(fnmatch.fnmatch(value, specifier) for value in values)
    return any(value == specifier for value in values)


def match_parsed_permission_rule(
    rule: ParsedPermissionRule,
    *,
    tool_name: str,
    parsed_input: Any,
    cwd: Optional[Path] = None,
) -> bool:
    """Return whether a parsed rule matches a tool invocation."""
    if rule.tool_name != tool_name:
        return False
    if rule.specifier is None:
        return True

    specifier = rule.specifier

    if tool_name == "Bash":
        command = _get_attr_or_key(parsed_input, "command")
        if not command:
            return False
        return match_rule(command, specifier)

    if specifier.startswith("domain:"):
        pattern = specifier[len("domain:") :].strip().lower()
        if not pattern:
            return False
        domain = _extract_domain(parsed_input)
        if not domain:
            return False
        return fnmatch.fnmatch(domain, pattern) if _is_wildcard_pattern(pattern) else domain == pattern

    path_value = _get_attr_or_key(parsed_input, "file_path") or _get_attr_or_key(parsed_input, "path")
    if path_value:
        candidates = _path_candidates(path_value, cwd)
        return _matches_text_specifier(_normalize_path_text(specifier), candidates)

    url_value = _extract_url_value(parsed_input)
    if url_value:
        return _matches_text_specifier(specifier, [url_value.strip()])

    command_value = _get_attr_or_key(parsed_input, "command")
    if command_value:
        return _matches_text_specifier(specifier, [command_value.strip()])

    if hasattr(parsed_input, "model_dump"):
        try:
            dumped = parsed_input.model_dump()
            if isinstance(dumped, dict):
                text_values = [str(v) for v in dumped.values() if isinstance(v, (str, Path))]
                if text_values and _matches_text_specifier(specifier, text_values):
                    return True
        except (TypeError, ValueError, AttributeError):
            pass

    return _matches_text_specifier(specifier, [str(parsed_input)])

