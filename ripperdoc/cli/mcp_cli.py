"""Top-level `ripperdoc mcp` command group."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import click

_PROJECT_SCOPE = "project"
_USER_SCOPE = "user"
_TOP_LEVEL_SERVERS_KEY = "__top_level_servers__"
_SUPPORTED_TRANSPORTS = ("stdio", "sse", "http", "streamable-http")


def _config_path_for_scope(project_path: Path, scope: str) -> Path:
    if scope == _USER_SCOPE:
        return Path.home().expanduser() / ".ripperdoc" / "mcp.json"
    return project_path / ".ripperdoc" / "mcp.json"


def _candidate_paths(project_path: Path, scope: str = "all") -> list[Path]:
    user_paths = [
        (Path.home().expanduser() / ".ripperdoc" / "mcp.json").resolve(),
        (Path.home().expanduser() / ".mcp.json").resolve(),
    ]
    project_paths = [
        (project_path / ".ripperdoc" / "mcp.json").resolve(),
        (project_path / ".mcp.json").resolve(),
    ]
    if scope == _USER_SCOPE:
        return user_paths
    if scope == _PROJECT_SCOPE:
        return project_paths
    return [*user_paths, *project_paths]


def _scope_for_path(project_path: Path, path: Path) -> str:
    project_paths = {
        (project_path / ".ripperdoc" / "mcp.json").resolve(),
        (project_path / ".mcp.json").resolve(),
    }
    user_paths = {
        (Path.home().expanduser() / ".ripperdoc" / "mcp.json").resolve(),
        (Path.home().expanduser() / ".mcp.json").resolve(),
    }
    resolved = path.resolve()
    if resolved in project_paths:
        return _PROJECT_SCOPE
    if resolved in user_paths:
        return _USER_SCOPE
    return "unknown"


def _looks_like_server_entry(raw: Any) -> bool:
    if not isinstance(raw, dict):
        return False
    known_keys = {"command", "args", "url", "uri", "type", "transport", "env", "headers"}
    return any(key in raw for key in known_keys)


def _load_mcp_json(path: Path) -> tuple[dict[str, Any], str]:
    if not path.exists():
        return {}, "servers"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, IOError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise click.ClickException(f"Invalid MCP config JSON at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise click.ClickException(f"Invalid MCP config format at {path}: root must be an object.")
    if "servers" in payload:
        if not isinstance(payload["servers"], dict):
            raise click.ClickException(f"Invalid MCP config at {path}: 'servers' must be an object.")
        return payload, "servers"
    if "mcpServers" in payload:
        if not isinstance(payload["mcpServers"], dict):
            raise click.ClickException(
                f"Invalid MCP config at {path}: 'mcpServers' must be an object."
            )
        return payload, "mcpServers"
    if any(_looks_like_server_entry(value) for value in payload.values()):
        return payload, _TOP_LEVEL_SERVERS_KEY
    return payload, "servers"


def _resolve_servers_container(data: dict[str, Any], servers_key: str) -> dict[str, Any]:
    if servers_key == _TOP_LEVEL_SERVERS_KEY:
        return data
    raw = data.get(servers_key)
    if raw is None:
        servers: dict[str, Any] = {}
        data[servers_key] = servers
        return servers
    if not isinstance(raw, dict):
        raise click.ClickException(f"Invalid MCP config: '{servers_key}' must be an object.")
    return raw


def _write_config(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _resolve_server_key(servers: dict[str, Any], requested: str) -> Optional[str]:
    if requested in servers:
        return requested
    lowered = [name for name in servers if name.lower() == requested.lower()]
    if len(lowered) == 1:
        return lowered[0]
    return None


def _parse_env_entries(raw_entries: tuple[str, ...]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw in raw_entries:
        if "=" not in raw:
            raise click.ClickException(
                f"Invalid --env entry '{raw}'. Expected KEY=VALUE format."
            )
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise click.ClickException(f"Invalid --env entry '{raw}'. Empty key is not allowed.")
        parsed[key] = value
    return parsed


def _parse_header_entries(raw_entries: tuple[str, ...]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw in raw_entries:
        if ":" in raw:
            key, value = raw.split(":", 1)
        elif "=" in raw:
            key, value = raw.split("=", 1)
        else:
            raise click.ClickException(
                f"Invalid --header entry '{raw}'. Use 'Name: Value' or 'Name=Value'."
            )
        key = key.strip()
        value = value.strip()
        if not key:
            raise click.ClickException(f"Invalid --header entry '{raw}'. Empty header name.")
        parsed[key] = value
    return parsed


def _infer_transport(transport: Optional[str], command_or_url: Optional[str], args: list[str]) -> str:
    if transport:
        return transport
    first = (command_or_url or (args[0] if args else "")).strip().lower()
    if first.startswith(("http://", "https://")):
        return "http"
    return "stdio"


def _format_stdio_target(entry: dict[str, Any]) -> str:
    command = entry.get("command")
    args = entry.get("args")
    parts: list[str] = []
    if isinstance(command, str) and command.strip():
        parts.append(command.strip())
    elif isinstance(command, list):
        parts.extend(str(item) for item in command if str(item).strip())
    if isinstance(args, list):
        parts.extend(str(item) for item in args if str(item).strip())
    return " ".join(parts)


def _entry_transport(entry: dict[str, Any]) -> str:
    transport = str(entry.get("type") or entry.get("transport") or "").strip().lower()
    if transport in _SUPPORTED_TRANSPORTS:
        return transport
    if entry.get("url") or entry.get("uri"):
        return "sse"
    return "stdio"


def _load_merged_servers(project_path: Path) -> dict[str, tuple[dict[str, Any], Path]]:
    merged: dict[str, tuple[dict[str, Any], Path]] = {}
    for path in _candidate_paths(project_path, scope="all"):
        if not path.exists():
            continue
        data, servers_key = _load_mcp_json(path)
        servers = _resolve_servers_container(data, servers_key)
        for raw_name, raw_entry in servers.items():
            if not isinstance(raw_entry, dict):
                continue
            name = str(raw_name).strip()
            if not name:
                continue
            merged[name] = (dict(raw_entry), path)
    return merged


def _upsert_server(path: Path, name: str, entry: dict[str, Any], *, overwrite: bool) -> bool:
    data, servers_key = _load_mcp_json(path)
    servers = _resolve_servers_container(data, servers_key)
    existing_key = _resolve_server_key(servers, name)
    updated = existing_key is not None
    if updated and not overwrite:
        raise click.ClickException(
            f"Server '{name}' already exists in {path}. Re-run with --overwrite to replace it."
        )
    if existing_key and existing_key != name:
        del servers[existing_key]
    servers[name] = entry
    if servers_key != _TOP_LEVEL_SERVERS_KEY:
        data[servers_key] = servers
    _write_config(path, data)
    return updated


def _remove_server(path: Path, name: str) -> bool:
    data, servers_key = _load_mcp_json(path)
    servers = _resolve_servers_container(data, servers_key)
    target_key = _resolve_server_key(servers, name)
    if target_key is None:
        return False
    del servers[target_key]
    if servers_key != _TOP_LEVEL_SERVERS_KEY:
        data[servers_key] = servers
    _write_config(path, data)
    return True


@click.group(name="mcp", invoke_without_command=True, help="Configure and manage MCP servers.")
@click.pass_context
def mcp_group(ctx: click.Context) -> None:
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@mcp_group.command(name="list")
@click.option("--json", "json_output", is_flag=True, help="Output machine-readable JSON.")
def list_servers(json_output: bool) -> None:
    project_path = Path.cwd()
    merged = _load_merged_servers(project_path)
    records: list[dict[str, Any]] = []
    for name in sorted(merged, key=str.lower):
        entry, path = merged[name]
        records.append(
            {
                "name": name,
                "transport": _entry_transport(entry),
                "scope": _scope_for_path(project_path, path),
                "config_path": str(path),
                "config": entry,
            }
        )
    if json_output:
        click.echo(json.dumps(records, indent=2, ensure_ascii=False))
        return
    if not records:
        click.echo("No MCP servers configured.")
        return
    click.echo("Configured MCP servers:")
    for row in records:
        click.echo(f"- {row['name']} [{row['transport']}]")
        click.echo(f"  Scope: {row['scope']} ({row['config_path']})")
        config = row["config"]
        if row["transport"] == "stdio":
            target = _format_stdio_target(config)
            if target:
                click.echo(f"  Command: {target}")
        else:
            url = str(config.get("url") or config.get("uri") or "").strip()
            if url:
                click.echo(f"  URL: {url}")


@mcp_group.command(name="get")
@click.argument("name")
@click.option("--json", "json_output", is_flag=True, help="Output machine-readable JSON.")
def get_server(name: str, json_output: bool) -> None:
    project_path = Path.cwd()
    merged = _load_merged_servers(project_path)
    resolved_name = _resolve_server_key({key: None for key in merged}, name)
    if resolved_name is None:
        known = ", ".join(sorted(merged, key=str.lower))
        suffix = f" Known servers: {known}" if known else ""
        raise click.ClickException(f"MCP server '{name}' not found.{suffix}")
    entry, path = merged[resolved_name]
    payload = {
        "name": resolved_name,
        "transport": _entry_transport(entry),
        "scope": _scope_for_path(project_path, path),
        "config_path": str(path),
        "config": entry,
    }
    if json_output:
        click.echo(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    click.echo(f"Name: {payload['name']}")
    click.echo(f"Transport: {payload['transport']}")
    click.echo(f"Scope: {payload['scope']}")
    click.echo(f"Config path: {payload['config_path']}")
    click.echo("Config:")
    click.echo(json.dumps(entry, indent=2, ensure_ascii=False))


@mcp_group.command(name="add")
@click.option(
    "--transport",
    type=click.Choice(list(_SUPPORTED_TRANSPORTS)),
    default=None,
    help="Transport type. Defaults to auto-detect from command/url.",
)
@click.option(
    "--scope",
    type=click.Choice([_PROJECT_SCOPE, _USER_SCOPE]),
    default=_PROJECT_SCOPE,
    show_default=True,
    help="Where to save server config.",
)
@click.option("-e", "--env", "env_entries", multiple=True, help="Environment entry KEY=VALUE.")
@click.option(
    "--header",
    "header_entries",
    multiple=True,
    help="HTTP header entry (`Name: Value` or `Name=Value`).",
)
@click.option("--description", default="", help="Optional server description.")
@click.option("--overwrite", is_flag=True, help="Overwrite if server already exists.")
@click.argument("name")
@click.argument("command_or_url", required=False)
@click.argument("args", nargs=-1)
def add_server(
    transport: Optional[str],
    scope: str,
    env_entries: tuple[str, ...],
    header_entries: tuple[str, ...],
    description: str,
    overwrite: bool,
    name: str,
    command_or_url: Optional[str],
    args: tuple[str, ...],
) -> None:
    clean_name = name.strip()
    if not clean_name:
        raise click.ClickException("Server name is required.")
    if " " in clean_name:
        raise click.ClickException("Server name must not contain spaces.")

    args_list = list(args)
    resolved_transport = _infer_transport(transport, command_or_url, args_list)
    env_map = _parse_env_entries(env_entries) if env_entries else {}
    header_map = _parse_header_entries(header_entries) if header_entries else {}
    entry: dict[str, Any] = {}

    if resolved_transport == "stdio":
        if header_map:
            raise click.ClickException(
                "--header is only supported for HTTP/SSE transports (not stdio)."
            )
        command_value = (command_or_url or "").strip()
        if not command_value:
            if not args_list:
                raise click.ClickException("Missing stdio command. Provide a command after <name>.")
            command_value = args_list.pop(0)
        entry["command"] = command_value
        if args_list:
            entry["args"] = args_list
        if env_map:
            entry["env"] = env_map
    else:
        if command_or_url:
            url_value = command_or_url.strip()
            if args_list:
                raise click.ClickException(
                    "Unexpected extra arguments for URL transports. "
                    "Use: ripperdoc mcp add --transport http <name> <url>"
                )
        else:
            if not args_list:
                raise click.ClickException("Missing URL for HTTP/SSE transport.")
            url_value = args_list.pop(0).strip()
            if args_list:
                raise click.ClickException(
                    "Unexpected extra arguments for URL transports. "
                    "Use: ripperdoc mcp add --transport http <name> <url>"
                )
        if not url_value:
            raise click.ClickException("URL cannot be empty.")
        entry["type"] = resolved_transport
        entry["url"] = url_value
        if header_map:
            entry["headers"] = header_map
        if env_map:
            entry["env"] = env_map

    if description.strip():
        entry["description"] = description.strip()

    path = _config_path_for_scope(Path.cwd(), scope)
    updated = _upsert_server(path, clean_name, entry, overwrite=overwrite)
    action = "Updated" if updated else "Added"
    click.echo(f"{action} MCP server '{clean_name}' in {path}.")


@mcp_group.command(name="add-json")
@click.option(
    "--scope",
    type=click.Choice([_PROJECT_SCOPE, _USER_SCOPE]),
    default=_PROJECT_SCOPE,
    show_default=True,
    help="Where to save server config.",
)
@click.option("--overwrite", is_flag=True, help="Overwrite if server already exists.")
@click.argument("name")
@click.argument("json_payload")
def add_server_json(scope: str, overwrite: bool, name: str, json_payload: str) -> None:
    clean_name = name.strip()
    if not clean_name:
        raise click.ClickException("Server name is required.")
    try:
        parsed = json.loads(json_payload)
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"Invalid JSON payload: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise click.ClickException("JSON payload must be an object.")
    path = _config_path_for_scope(Path.cwd(), scope)
    updated = _upsert_server(path, clean_name, dict(parsed), overwrite=overwrite)
    action = "Updated" if updated else "Added"
    click.echo(f"{action} MCP server '{clean_name}' in {path}.")


@mcp_group.command(name="remove")
@click.option(
    "--scope",
    type=click.Choice([_PROJECT_SCOPE, _USER_SCOPE, "all"]),
    default="all",
    show_default=True,
    help="Search and remove server from selected scope(s).",
)
@click.argument("name")
def remove_server(scope: str, name: str) -> None:
    clean_name = name.strip()
    if not clean_name:
        raise click.ClickException("Server name is required.")
    removed_paths: list[Path] = []
    for path in _candidate_paths(Path.cwd(), scope=scope):
        if not path.exists():
            continue
        if _remove_server(path, clean_name):
            removed_paths.append(path)
    if not removed_paths:
        raise click.ClickException(f"MCP server '{clean_name}' was not found in scope '{scope}'.")
    for path in removed_paths:
        click.echo(f"Removed MCP server '{clean_name}' from {path}.")


@mcp_group.command(name="reset-project-choices")
def reset_project_choices() -> None:
    project_path = Path.cwd()
    candidates = [
        project_path / ".ripperdoc" / "mcp_choices.json",
        project_path / ".ripperdoc" / "mcp_project_choices.json",
        project_path / ".ripperdoc" / "mcp_server_choices.json",
    ]
    deleted: list[Path] = []
    for path in candidates:
        if not path.exists():
            continue
        try:
            path.unlink()
            deleted.append(path)
        except OSError as exc:
            raise click.ClickException(f"Failed to remove {path}: {exc}") from exc
    if not deleted:
        click.echo("No project MCP choice cache found.")
        return
    click.echo(f"Reset {len(deleted)} project MCP choice file(s).")
    for path in deleted:
        click.echo(f"- {path}")


@mcp_group.command(name="serve")
def serve_command() -> None:
    click.echo(
        "ripperdoc mcp serve is not implemented yet. "
        "Use ripperdoc --output-format stream-json for protocol integration."
    )


__all__ = ["mcp_group"]
