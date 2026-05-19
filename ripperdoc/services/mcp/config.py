"""
MCP configuration loading — mirrors reference: services/mcp/config.ts.

Loads server configs from multiple scopes (user, project, enterprise, managed),
supports plugin discovery, runtime overrides, and JSON/file parsing.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ripperdoc.services.mcp.types import (
    McpServerInfo,
)
from ripperdoc.services.plugins import discover_plugins, expand_plugin_root_vars
from ripperdoc.utils.filesystem.config_paths import config_file_for_scope
from ripperdoc.utils.log import get_logger

logger = get_logger()


def _load_json_file(path: Path) -> Dict[str, Any]:
    """Load and parse a JSON file, returning empty dict on failure."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            return data
        return {}
    except (OSError, json.JSONDecodeError):
        logger.exception("Failed to load JSON", extra={"path": str(path)})
        return {}


def _ensure_str_dict(raw: object) -> Dict[str, str]:
    """Coerce a raw value to a string-to-string dict."""
    if not isinstance(raw, dict):
        return {}
    result: Dict[str, str] = {}
    for key, value in raw.items():
        try:
            result[str(key)] = str(value)
        except (TypeError, ValueError) as exc:
            logger.warning(
                "[mcp] Failed to coerce env/header value to string: %s: %s",
                type(exc).__name__,
                exc,
                extra={"key": key},
            )
            continue
    return result


def _normalize_command(
    raw_command: Any, raw_args: Any
) -> Tuple[Optional[str], List[str]]:
    """Normalize MCP server command/args.

    Supports:
    - command as list → first element is executable, rest are args
    - command as string with spaces → shlex.split into executable/args (when args empty)
    - command as plain string → used as-is
    """
    args: List[str] = []
    if isinstance(raw_args, list):
        args = [str(a) for a in raw_args]

    if isinstance(raw_command, list):
        tokens = [str(t) for t in raw_command if str(t)]
        if not tokens:
            return None, args
        return tokens[0], tokens[1:] + args

    if not isinstance(raw_command, str):
        return None, args

    command_str = raw_command.strip()
    if not command_str:
        return None, args

    if not args and (" " in command_str or "\t" in command_str):
        try:
            tokens = shlex.split(command_str)
        except ValueError:
            tokens = [command_str]
        if tokens:
            return tokens[0], tokens[1:]

    return command_str, args


def parse_server(name: str, raw: Dict[str, Any]) -> McpServerInfo:
    """Parse a single MCP server config dict into an McpServerInfo."""
    server_type = str(raw.get("type") or raw.get("transport") or "").strip().lower()
    command, args = _normalize_command(raw.get("command"), raw.get("args"))
    url = str(raw.get("url") or raw.get("uri") or "").strip() or None

    if not server_type:
        if url:
            server_type = "sse"
        elif command:
            server_type = "stdio"
        else:
            server_type = "stdio"

    description = str(raw.get("description") or "")
    env = _ensure_str_dict(raw.get("env"))
    headers = _ensure_str_dict(raw.get("headers"))
    instructions = raw.get("instructions")
    headers_helper = raw.get("headersHelper") or raw.get("headers_helper")

    return McpServerInfo(
        name=name,
        type=server_type,
        url=url,
        description=description,
        command=command,
        args=[str(a) for a in args] if args else [],
        env=env,
        headers=headers,
        headers_helper=str(headers_helper) if headers_helper else None,
        instructions=str(instructions) if isinstance(instructions, str) else None,
    )


def parse_servers(data: Dict[str, Any]) -> Dict[str, McpServerInfo]:
    """Parse MCP server definitions from a config dict.

    Supports both ``{ "servers": { ... } }`` and ``{ "mcpServers": { ... } }``
    top-level keys.
    """
    servers: Dict[str, McpServerInfo] = {}
    for key in ("servers", "mcpServers"):
        raw_servers = data.get(key)
        if not isinstance(raw_servers, dict):
            continue
        for raw_name, raw in raw_servers.items():
            if not isinstance(raw, dict):
                continue
            server_name = str(raw_name).strip()
            if not server_name:
                continue
            servers[server_name] = parse_server(server_name, raw)
    if servers:
        return servers

    # Support direct top-level map of server_name -> config
    for name, raw in data.items():
        if not isinstance(raw, dict):
            continue
        if not any(
            key in raw for key in ("command", "args", "url", "uri", "type", "transport")
        ):
            continue
        server_name = str(name).strip()
        if not server_name:
            continue
        servers[server_name] = parse_server(server_name, raw)
    return servers


def parse_mcp_config_option(
    raw_value: Union[str, Path, None],
    *,
    base_dir: Optional[Path] = None,
) -> Dict[str, McpServerInfo]:
    """Parse ``--mcp-config`` style JSON/path input into server configs."""
    if raw_value is None:
        return {}
    if isinstance(raw_value, Path):
        raw_text = raw_value.read_text(encoding="utf-8")
    else:
        candidate = str(raw_value).strip()
        if not candidate:
            return {}
        candidate_path = Path(candidate)
        if not candidate.lstrip().startswith("{") and not candidate.lstrip().startswith(
            "["
        ):
            if not candidate_path.is_absolute() and base_dir is not None:
                candidate_path = (base_dir / candidate_path).resolve()
            if candidate_path.exists():
                raw_text = candidate_path.read_text(encoding="utf-8")
            else:
                raw_text = candidate
        else:
            raw_text = candidate
    parsed = json.loads(raw_text)
    return parse_mcp_server_configs(parsed)


def parse_mcp_server_configs(raw: Any) -> Dict[str, McpServerInfo]:
    """Parse MCP server config payloads from control requests."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        if "servers" in raw or "mcpServers" in raw:
            return parse_servers(raw)
        if all(isinstance(value, dict) for value in raw.values()):
            return parse_servers({"servers": raw})
        if "name" in raw and isinstance(raw.get("name"), str):
            entry_name = str(raw.get("name") or "").strip()
            if not entry_name:
                return {}
            entry: Dict[str, Any] = dict(raw)
            entry.pop("name", None)
            return {entry_name: parse_server(entry_name, entry)}
        return {}

    if isinstance(raw, list):
        parsed: Dict[str, McpServerInfo] = {}
        for item in raw:
            if not isinstance(item, dict):
                continue
            item_name = str(item.get("name") or "").strip()
            if not item_name:
                continue
            entry = dict(item)
            entry.pop("name", None)
            parsed[item_name] = parse_server(item_name, entry)
        return parsed

    return {}


# Runtime overrides (injected by control requests, tests, etc.)
_mcp_runtime_server_overrides: Dict[str, Dict[str, McpServerInfo]] = {}
_mcp_runtime_disabled_servers: Dict[str, Set[str]] = {}


def _project_scope_key(project_path: Optional[Path]) -> str:
    path = project_path or Path.cwd()
    try:
        return str(path.resolve())
    except (OSError, RuntimeError):
        return str(path)


def set_mcp_runtime_overrides(
    project_path: Optional[Path] = None,
    *,
    servers: Optional[Dict[str, McpServerInfo]] = None,
    disabled: Optional[Set[str]] = None,
) -> None:
    """Set runtime-only MCP server overrides for a project scope."""
    key = _project_scope_key(project_path)
    if servers is None:
        _mcp_runtime_server_overrides.pop(key, None)
    else:
        from dataclasses import replace
        _mcp_runtime_server_overrides[key] = {
            str(name): replace(server) for name, server in servers.items()
        }

    if disabled is None:
        _mcp_runtime_disabled_servers.pop(key, None)
    else:
        _mcp_runtime_disabled_servers[key] = {
            str(name) for name in disabled if str(name).strip()
        }


def clear_mcp_runtime_overrides(project_path: Optional[Path] = None) -> None:
    """Clear runtime-only MCP server overrides for a project scope."""
    key = _project_scope_key(project_path)
    _mcp_runtime_server_overrides.pop(key, None)
    _mcp_runtime_disabled_servers.pop(key, None)


def load_server_configs(project_path: Optional[Path]) -> Dict[str, McpServerInfo]:
    """Load effective MCP server configs (disk + plugin + runtime overrides)."""
    from dataclasses import replace

    project_path = project_path or Path.cwd()
    managed_mcp_path = config_file_for_scope(
        "managed", "managed-mcp.json", project_path=project_path
    )
    candidates = [
        config_file_for_scope("user", "mcp.json"),
        Path.home() / ".mcp.json",
        config_file_for_scope("project", "mcp.json", project_path=project_path),
        project_path / ".mcp.json",
    ]

    merged: Dict[str, McpServerInfo] = {}
    for path in candidates:
        data = _load_json_file(path)
        merged.update(parse_servers(data))

    plugin_result = discover_plugins(project_path=project_path)
    for plugin_error in plugin_result.errors:
        logger.warning(
            "[mcp] Plugin discovery warning: %s (%s)",
            plugin_error.path,
            plugin_error.reason,
        )

    for plugin in plugin_result.plugins:
        for mcp_path in plugin.mcp_paths:
            resolved_path = mcp_path
            if resolved_path.is_dir():
                if (resolved_path / ".mcp.json").exists():
                    resolved_path = resolved_path / ".mcp.json"
                elif (resolved_path / "mcp.json").exists():
                    resolved_path = resolved_path / "mcp.json"
            data = _load_json_file(resolved_path)
            if not data:
                continue
            expanded = expand_plugin_root_vars(data, plugin.root)
            if isinstance(expanded, dict):
                merged.update(parse_servers(expanded))

        for inline_entry in plugin.mcp_inline:
            expanded_inline = expand_plugin_root_vars(inline_entry, plugin.root)
            if isinstance(expanded_inline, dict):
                merged.update(parse_servers(expanded_inline))

    # Managed MCP has highest precedence
    managed_payload = _load_json_file(managed_mcp_path)
    if managed_payload:
        merged.update(parse_servers(managed_payload))

    logger.debug(
        "[mcp] Loaded MCP server configs",
        extra={
            "project_path": str(project_path),
            "server_count": len(merged),
        },
    )

    key = _project_scope_key(project_path)
    overrides = _mcp_runtime_server_overrides.get(key)
    if overrides is not None:
        merged = {
            str(name): replace(server)
            for name, server in overrides.items()
            if str(name).strip()
        }

    disabled = _mcp_runtime_disabled_servers.get(key)
    if disabled:
        for server_name in list(disabled):
            merged.pop(server_name, None)

    return merged


def load_mcp_server_configs(
    project_path: Optional[Path] = None,
) -> Dict[str, McpServerInfo]:
    """Load effective MCP server configs (disk + plugin + runtime overrides)."""
    return load_server_configs(project_path)
