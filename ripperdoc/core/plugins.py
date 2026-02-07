"""Plugin discovery and configuration for Ripperdoc.

Ripperdoc plugins follow a Claude-style layout with support for:
- optional manifest: .ripperdoc-plugin/plugin.json (or .claude-plugin/plugin.json)
- commands/, skills/, agents/, hooks/hooks.json, .mcp.json, .lsp.json
- optional manifest component path fields that supplement defaults

Plugins can be enabled via:
- runtime-only directories (e.g., --plugin-dir)
- RIPPERDOC_PLUGIN_DIR (os.pathsep-separated)
- settings files:
  - ~/.ripperdoc/plugins.json
  - .ripperdoc/plugins.json
  - .ripperdoc/plugins.local.json
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ripperdoc.utils.log import get_logger

logger = get_logger()


PLUGIN_MANIFEST_CANDIDATES: tuple[tuple[str, str], ...] = (
    (".ripperdoc-plugin", "plugin.json"),
    (".claude-plugin", "plugin.json"),
)
PLUGIN_SETTINGS_FILE = "plugins.json"
PLUGIN_SETTINGS_LOCAL_FILE = "plugins.local.json"
PLUGIN_ROOT_ENV_VAR = "RIPPERDOC_PLUGIN_ROOT"
PLUGIN_ROOT_ENV_COMPAT_VAR = "CLAUDE_PLUGIN_ROOT"
PLUGIN_DIRS_ENV_VAR = "RIPPERDOC_PLUGIN_DIR"
_PLUGIN_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")

# Runtime-only plugin dirs set by CLI flags (e.g., --plugin-dir)
_runtime_plugin_dirs: tuple[Path, ...] = ()


class PluginSettingsScope(str, Enum):
    USER = "user"
    PROJECT = "project"
    LOCAL = "local"


@dataclass(frozen=True)
class PluginLoadError:
    path: Path
    reason: str


@dataclass
class PluginDefinition:
    name: str
    root: Path
    manifest_path: Optional[Path] = None
    description: str = ""
    version: Optional[str] = None
    commands_paths: List[Path] = field(default_factory=list)
    skills_paths: List[Path] = field(default_factory=list)
    agents_paths: List[Path] = field(default_factory=list)
    hooks_paths: List[Path] = field(default_factory=list)
    hooks_inline: List[Dict[str, Any]] = field(default_factory=list)
    mcp_paths: List[Path] = field(default_factory=list)
    mcp_inline: List[Dict[str, Any]] = field(default_factory=list)
    lsp_paths: List[Path] = field(default_factory=list)
    lsp_inline: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class PluginLoadResult:
    plugins: List[PluginDefinition]
    errors: List[PluginLoadError]


def _settings_paths(
    project_path: Optional[Path], home: Optional[Path]
) -> Dict[PluginSettingsScope, Path]:
    home_dir = (home or Path.home()).expanduser()
    project_dir = (project_path or Path.cwd()).resolve()
    return {
        PluginSettingsScope.USER: home_dir / ".ripperdoc" / PLUGIN_SETTINGS_FILE,
        PluginSettingsScope.PROJECT: project_dir / ".ripperdoc" / PLUGIN_SETTINGS_FILE,
        PluginSettingsScope.LOCAL: project_dir / ".ripperdoc" / PLUGIN_SETTINGS_LOCAL_FILE,
    }


def _entry_base_dir_for_scope(scope: PluginSettingsScope, settings_path: Path) -> Path:
    if scope in (PluginSettingsScope.PROJECT, PluginSettingsScope.LOCAL):
        parent = settings_path.parent
        if parent.name == ".ripperdoc":
            return parent.parent
    if scope == PluginSettingsScope.USER:
        parent = settings_path.parent
        if parent.name == ".ripperdoc":
            return parent.parent
    return settings_path.parent


def get_plugin_settings_path(
    scope: PluginSettingsScope,
    project_path: Optional[Path] = None,
    home: Optional[Path] = None,
) -> Path:
    """Resolve plugin settings path for a specific scope."""
    return _settings_paths(project_path, home)[scope]


def set_runtime_plugin_dirs(
    directories: Optional[Sequence[str | Path]],
    *,
    base_dir: Optional[Path] = None,
) -> None:
    """Set runtime-only plugin directories (typically from --plugin-dir)."""
    global _runtime_plugin_dirs
    resolved: List[Path] = []
    cwd = (base_dir or Path.cwd()).resolve()
    for raw in directories or []:
        if not raw:
            continue
        try:
            candidate = Path(raw).expanduser()
            if not candidate.is_absolute():
                candidate = (cwd / candidate).resolve()
            resolved.append(candidate.resolve())
        except (OSError, RuntimeError, ValueError):
            continue
    _runtime_plugin_dirs = tuple(dict.fromkeys(resolved))


def get_runtime_plugin_dirs() -> List[Path]:
    """Return runtime-only plugin directories."""
    return list(_runtime_plugin_dirs)


def clear_runtime_plugin_dirs() -> None:
    """Clear runtime-only plugin directory overrides."""
    set_runtime_plugin_dirs([])


def expand_plugin_root_vars_in_string(text: str, plugin_root: Path) -> str:
    """Expand supported plugin-root variables in a string."""
    root = str(plugin_root.resolve())
    text = text.replace(f"${{{PLUGIN_ROOT_ENV_VAR}}}", root)
    text = text.replace(f"${{{PLUGIN_ROOT_ENV_COMPAT_VAR}}}", root)
    return text


def expand_plugin_root_vars(value: Any, plugin_root: Path) -> Any:
    """Recursively expand plugin-root variables in mapping/list/string values."""
    if isinstance(value, str):
        return expand_plugin_root_vars_in_string(value, plugin_root)
    if isinstance(value, list):
        return [expand_plugin_root_vars(item, plugin_root) for item in value]
    if isinstance(value, dict):
        return {str(k): expand_plugin_root_vars(v, plugin_root) for k, v in value.items()}
    return value


def _coerce_settings_entries(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if not isinstance(raw, list):
        return []
    values: List[str] = []
    for item in raw:
        if isinstance(item, str):
            values.append(item)
        elif isinstance(item, dict):
            candidate = item.get("path") or item.get("source") or item.get("dir")
            if isinstance(candidate, str):
                values.append(candidate)
    return values


def _load_settings_file(path: Path) -> Tuple[List[str], List[str]]:
    if not path.exists():
        return [], []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, IOError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.warning(
            "[plugins] Failed to read settings file %s: %s: %s",
            path,
            type(exc).__name__,
            exc,
        )
        return [], []

    if isinstance(raw, list):
        return _coerce_settings_entries(raw), []
    if not isinstance(raw, dict):
        return [], []

    enabled = (
        _coerce_settings_entries(raw.get("enabledPlugins"))
        or _coerce_settings_entries(raw.get("enabled_plugins"))
        or _coerce_settings_entries(raw.get("plugins"))
    )
    disabled = _coerce_settings_entries(raw.get("disabledPlugins")) or _coerce_settings_entries(
        raw.get("disabled_plugins")
    )
    return enabled, disabled


def _resolve_plugin_path(raw: str, base_dir: Path) -> Optional[Path]:
    value = raw.strip()
    if not value:
        return None
    try:
        candidate = Path(value).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()
        return (base_dir / candidate).resolve()
    except (OSError, RuntimeError, ValueError):
        return None


def _configured_plugin_dirs(project_path: Optional[Path], home: Optional[Path]) -> List[Path]:
    paths = _settings_paths(project_path, home)
    ordered_scopes = [
        PluginSettingsScope.USER,
        PluginSettingsScope.PROJECT,
        PluginSettingsScope.LOCAL,
    ]

    enabled: List[Path] = []
    disabled: set[Path] = set()
    for scope in ordered_scopes:
        settings_path = paths[scope]
        entries, disabled_entries = _load_settings_file(settings_path)
        base_dir = _entry_base_dir_for_scope(scope, settings_path)

        for raw in entries:
            resolved = _resolve_plugin_path(raw, base_dir)
            if resolved is not None:
                enabled.append(resolved)
        for raw in disabled_entries:
            resolved = _resolve_plugin_path(raw, base_dir)
            if resolved is not None:
                disabled.add(resolved)

    filtered = [path for path in enabled if path not in disabled]
    return list(dict.fromkeys(filtered))


def _env_plugin_dirs() -> List[Path]:
    raw = os.getenv(PLUGIN_DIRS_ENV_VAR, "")
    if not raw.strip():
        return []
    result: List[Path] = []
    for item in raw.split(os.pathsep):
        value = item.strip()
        if not value:
            continue
        try:
            result.append(Path(value).expanduser().resolve())
        except (OSError, RuntimeError, ValueError):
            continue
    return list(dict.fromkeys(result))


def resolve_enabled_plugin_dirs(
    project_path: Optional[Path] = None, home: Optional[Path] = None
) -> List[Path]:
    """Resolve all enabled plugin directories in merge order.

    Merge order:
    1) settings scopes (user -> project -> local, with local disable support)
    2) RIPPERDOC_PLUGIN_DIR
    3) runtime plugin dirs (--plugin-dir)
    """
    merged: List[Path] = []
    merged.extend(_configured_plugin_dirs(project_path, home))
    merged.extend(_env_plugin_dirs())
    merged.extend(get_runtime_plugin_dirs())
    return list(dict.fromkeys(merged))


def _safe_plugin_name(raw: str) -> str:
    value = raw.strip().lower()
    if _PLUGIN_NAME_RE.match(value):
        return value

    value = value.replace("_", "-").replace(" ", "-")
    value = re.sub(r"[^a-z0-9-]+", "-", value).strip("-")
    value = re.sub(r"-{2,}", "-", value)
    if not value:
        value = "plugin"
    if len(value) > 64:
        value = value[:64].rstrip("-")
    if not _PLUGIN_NAME_RE.match(value):
        value = f"plugin-{abs(hash(raw)) % 100000:05d}"
    return value


def _find_manifest(plugin_root: Path) -> Optional[Path]:
    for folder, file_name in PLUGIN_MANIFEST_CANDIDATES:
        candidate = plugin_root / folder / file_name
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _load_manifest(manifest_path: Path) -> Tuple[Dict[str, Any], Optional[str]]:
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, IOError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {}, f"Invalid manifest JSON: {type(exc).__name__}: {exc}"
    if not isinstance(data, dict):
        return {}, "Invalid manifest format: expected JSON object"
    return data, None


def _resolve_manifest_path(
    plugin_root: Path,
    raw_path: str,
    *,
    field_name: str,
) -> Tuple[Optional[Path], Optional[str]]:
    value = raw_path.strip()
    if not value:
        return None, None
    if value.startswith("${"):
        # Keep support for explicit root vars in manifests.
        value = expand_plugin_root_vars_in_string(value, plugin_root)
    path_obj = Path(value)
    if path_obj.is_absolute():
        return None, f"{field_name}: absolute paths are not allowed ({value})"
    resolved = (plugin_root / path_obj).resolve()
    try:
        resolved.relative_to(plugin_root.resolve())
    except ValueError:
        return None, f"{field_name}: path escapes plugin root ({value})"
    return resolved, None


def _coerce_path_values(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, str)]


def _coerce_mixed_values(raw: Any) -> List[Any]:
    if raw is None:
        return []
    if isinstance(raw, (str, dict)):
        return [raw]
    if not isinstance(raw, list):
        return []
    return list(raw)


def _dedupe_paths(paths: Iterable[Path]) -> List[Path]:
    deduped: List[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    return deduped


def _default_component_paths(plugin_root: Path) -> Dict[str, List[Path]]:
    return {
        "commands": [plugin_root / "commands"],
        "skills": [plugin_root / "skills"],
        "agents": [plugin_root / "agents"],
        "hooks": [plugin_root / "hooks" / "hooks.json"],
        "mcp": [plugin_root / ".mcp.json"],
        "lsp": [plugin_root / ".lsp.json"],
    }


def _build_plugin_definition(
    plugin_root: Path,
) -> Tuple[Optional[PluginDefinition], List[PluginLoadError]]:
    errors: List[PluginLoadError] = []
    manifest_path = _find_manifest(plugin_root)
    manifest: Dict[str, Any] = {}
    if manifest_path is not None:
        manifest, err = _load_manifest(manifest_path)
        if err:
            return None, [PluginLoadError(path=manifest_path, reason=err)]

    raw_name = manifest.get("name")
    if isinstance(raw_name, str) and raw_name.strip():
        plugin_name = _safe_plugin_name(raw_name)
    else:
        plugin_name = _safe_plugin_name(plugin_root.name)

    defaults = _default_component_paths(plugin_root)

    commands_paths: List[Path] = list(defaults["commands"])
    skills_paths: List[Path] = list(defaults["skills"])
    agents_paths: List[Path] = list(defaults["agents"])
    hooks_paths: List[Path] = list(defaults["hooks"])
    hooks_inline: List[Dict[str, Any]] = []
    mcp_paths: List[Path] = list(defaults["mcp"])
    mcp_inline: List[Dict[str, Any]] = []
    lsp_paths: List[Path] = list(defaults["lsp"])
    lsp_inline: List[Dict[str, Any]] = []

    for field_name, store in (
        ("commands", commands_paths),
        ("skills", skills_paths),
        ("agents", agents_paths),
    ):
        for raw in _coerce_path_values(manifest.get(field_name)):
            resolved, err = _resolve_manifest_path(plugin_root, raw, field_name=field_name)
            if err:
                errors.append(PluginLoadError(path=plugin_root, reason=err))
                continue
            if resolved is not None:
                store.append(resolved)

    for item in _coerce_mixed_values(manifest.get("hooks")):
        if isinstance(item, dict):
            hooks_inline.append(item)
            continue
        if isinstance(item, str):
            resolved, err = _resolve_manifest_path(plugin_root, item, field_name="hooks")
            if err:
                errors.append(PluginLoadError(path=plugin_root, reason=err))
                continue
            if resolved is not None:
                hooks_paths.append(resolved)

    for item in _coerce_mixed_values(manifest.get("mcpServers")):
        if isinstance(item, dict):
            mcp_inline.append(item)
            continue
        if isinstance(item, str):
            resolved, err = _resolve_manifest_path(plugin_root, item, field_name="mcpServers")
            if err:
                errors.append(PluginLoadError(path=plugin_root, reason=err))
                continue
            if resolved is not None:
                mcp_paths.append(resolved)

    for item in _coerce_mixed_values(manifest.get("lspServers")):
        if isinstance(item, dict):
            lsp_inline.append(item)
            continue
        if isinstance(item, str):
            resolved, err = _resolve_manifest_path(plugin_root, item, field_name="lspServers")
            if err:
                errors.append(PluginLoadError(path=plugin_root, reason=err))
                continue
            if resolved is not None:
                lsp_paths.append(resolved)

    plugin = PluginDefinition(
        name=plugin_name,
        root=plugin_root.resolve(),
        manifest_path=manifest_path,
        description=str(manifest.get("description") or ""),
        version=str(manifest.get("version")) if manifest.get("version") is not None else None,
        commands_paths=_dedupe_paths(commands_paths),
        skills_paths=_dedupe_paths(skills_paths),
        agents_paths=_dedupe_paths(agents_paths),
        hooks_paths=_dedupe_paths(hooks_paths),
        hooks_inline=[entry for entry in hooks_inline if isinstance(entry, dict)],
        mcp_paths=_dedupe_paths(mcp_paths),
        mcp_inline=[entry for entry in mcp_inline if isinstance(entry, dict)],
        lsp_paths=_dedupe_paths(lsp_paths),
        lsp_inline=[entry for entry in lsp_inline if isinstance(entry, dict)],
    )
    return plugin, errors


def discover_plugins(
    project_path: Optional[Path] = None, home: Optional[Path] = None
) -> PluginLoadResult:
    """Discover enabled plugins and parse their manifests/components."""
    plugin_dirs = resolve_enabled_plugin_dirs(project_path=project_path, home=home)
    plugins: List[PluginDefinition] = []
    errors: List[PluginLoadError] = []
    seen_roots: set[Path] = set()

    for plugin_dir in plugin_dirs:
        root = plugin_dir.resolve()
        if root in seen_roots:
            continue
        seen_roots.add(root)

        if not root.exists():
            errors.append(PluginLoadError(path=root, reason="Plugin directory does not exist"))
            continue
        if not root.is_dir():
            errors.append(PluginLoadError(path=root, reason="Plugin path is not a directory"))
            continue

        plugin, plugin_errors = _build_plugin_definition(root)
        errors.extend(plugin_errors)
        if plugin is not None:
            plugins.append(plugin)

    return PluginLoadResult(plugins=plugins, errors=errors)


def _plugin_settings_payload(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, IOError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _serialize_plugin_path(path: Path, base_dir: Path) -> str:
    try:
        rel = path.resolve().relative_to(base_dir.resolve())
    except ValueError:
        return str(path.resolve())
    rel_text = rel.as_posix()
    if not rel_text.startswith("."):
        rel_text = f"./{rel_text}"
    return rel_text


def list_enabled_plugin_entries_for_scope(
    scope: PluginSettingsScope,
    project_path: Optional[Path] = None,
    home: Optional[Path] = None,
) -> List[str]:
    """Return raw enabled plugin entries from one scope settings file."""
    settings_path = get_plugin_settings_path(scope, project_path=project_path, home=home)
    payload = _plugin_settings_payload(settings_path)
    enabled = (
        _coerce_settings_entries(payload.get("enabledPlugins"))
        or _coerce_settings_entries(payload.get("enabled_plugins"))
        or _coerce_settings_entries(payload.get("plugins"))
    )
    return enabled


def add_enabled_plugin_for_scope(
    plugin_dir: Path,
    scope: PluginSettingsScope = PluginSettingsScope.PROJECT,
    project_path: Optional[Path] = None,
    home: Optional[Path] = None,
) -> Path:
    """Add a plugin directory to a scope settings file."""
    settings_path = get_plugin_settings_path(scope, project_path=project_path, home=home)
    payload = _plugin_settings_payload(settings_path)

    current = (
        _coerce_settings_entries(payload.get("enabledPlugins"))
        or _coerce_settings_entries(payload.get("enabled_plugins"))
        or _coerce_settings_entries(payload.get("plugins"))
    )

    base_dir = _entry_base_dir_for_scope(scope, settings_path)
    serialized = _serialize_plugin_path(plugin_dir.resolve(), base_dir)
    if serialized not in current:
        current.append(serialized)

    payload["enabledPlugins"] = current
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return settings_path


def remove_enabled_plugin_for_scope(
    plugin_dir: Path,
    scope: PluginSettingsScope = PluginSettingsScope.PROJECT,
    project_path: Optional[Path] = None,
    home: Optional[Path] = None,
) -> Tuple[Path, bool]:
    """Remove a plugin directory from a scope settings file."""
    settings_path = get_plugin_settings_path(scope, project_path=project_path, home=home)
    payload = _plugin_settings_payload(settings_path)

    entries = (
        _coerce_settings_entries(payload.get("enabledPlugins"))
        or _coerce_settings_entries(payload.get("enabled_plugins"))
        or _coerce_settings_entries(payload.get("plugins"))
    )
    if not entries:
        return settings_path, False

    base_dir = _entry_base_dir_for_scope(scope, settings_path)
    target = plugin_dir.resolve()
    kept: List[str] = []
    removed = False
    for raw in entries:
        resolved = _resolve_plugin_path(raw, base_dir)
        if resolved is not None and resolved == target:
            removed = True
            continue
        kept.append(raw)

    payload["enabledPlugins"] = kept
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return settings_path, removed


__all__ = [
    "PLUGIN_ROOT_ENV_VAR",
    "PLUGIN_ROOT_ENV_COMPAT_VAR",
    "PLUGIN_DIRS_ENV_VAR",
    "PluginSettingsScope",
    "PluginLoadError",
    "PluginDefinition",
    "PluginLoadResult",
    "get_plugin_settings_path",
    "set_runtime_plugin_dirs",
    "get_runtime_plugin_dirs",
    "clear_runtime_plugin_dirs",
    "expand_plugin_root_vars_in_string",
    "expand_plugin_root_vars",
    "resolve_enabled_plugin_dirs",
    "discover_plugins",
    "list_enabled_plugin_entries_for_scope",
    "add_enabled_plugin_for_scope",
    "remove_enabled_plugin_for_scope",
]
