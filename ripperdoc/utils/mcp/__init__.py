"""
MCP configuration loader, connection manager, and prompt helpers.

This module re-exports from the refactored ``services/mcp/`` package for
backward compatibility.
"""

from pathlib import Path as _Path

from ripperdoc.services.mcp.types import (
    ConfigScope,
    McpResourceInfo,
    McpServerInfo,
    McpToolInfo,
    TransportType,
)
from ripperdoc.services.mcp.config import (
    clear_mcp_runtime_overrides,
    load_mcp_server_configs,
    parse_mcp_config_option,
    parse_mcp_server_configs,
    set_mcp_runtime_overrides,
    load_server_configs,
    parse_server,
    parse_servers,
    _ensure_str_dict,
    _load_json_file,
    _normalize_command,
    # _parse_server and _parse_servers renamed to parse_server/parse_servers
    _project_scope_key,
)
from ripperdoc.services.mcp import types as _ripperdoc_mcp_types
MCP_AVAILABLE = _ripperdoc_mcp_types.MCP_AVAILABLE
from ripperdoc.services.mcp.client import (  # noqa: E402
    _global_runtime,
    _mcp_circuit_states,
    McpRuntime,
    _SdkMcpSession,
    ensure_mcp_runtime,
    get_existing_mcp_runtime,
    get_mcp_stderr_log_path,
    get_mcp_stderr_mode,
    shutdown_mcp_runtime,
    clear_sdk_mcp_request_sender,
    set_sdk_mcp_request_sender,
    get_sdk_mcp_request_sender,
)
from ripperdoc.services.mcp.utils import (  # noqa: E402
    estimate_mcp_tokens,
    find_mcp_resource,
    format_mcp_instructions,
    load_mcp_servers,
    load_mcp_servers_async,
)
from ripperdoc.services.mcp.mcp_string_utils import (  # noqa: E402
    build_mcp_tool_name,
    get_mcp_prefix,
    mcp_info_from_string,
)
from ripperdoc.services.mcp.normalization import (  # noqa: E402
    normalize_name_for_mcp,
)
from ripperdoc.services.mcp.env_expansion import (  # noqa: E402
    expand_env_vars_in_string,
)

_load_server_configs = load_server_configs

# Backward-compat aliases
_parse_server = parse_server
_parse_servers = parse_servers

# Keep _coerce_sdk_schema here since it's locally defined
def _coerce_sdk_schema(value: object) -> dict:
    """Coerce Agent SDK shorthand schemas into JSON Schema."""
    if value is None:
        return {}

    if hasattr(value, "model_json_schema") and callable(value.model_json_schema):
        try:
            json_schema = value.model_json_schema()
            if isinstance(json_schema, dict):
                return json_schema
        except (TypeError, ValueError, AttributeError):
            pass

    from typing import Dict, List, cast, get_args, get_origin

    def _looks_like_json_schema(val: object) -> bool:
        if not isinstance(val, dict):
            return False
        schema_keys = {
            "$schema", "$defs", "$ref", "type", "properties", "required",
            "items", "anyOf", "oneOf", "allOf", "enum", "additionalProperties",
        }
        return bool(schema_keys & set(val.keys()))

    if _looks_like_json_schema(value):
        return cast(dict, value)

    origin = get_origin(value)
    if origin is not None:
        args = [arg for arg in get_args(value) if arg is not type(None)]
        if origin in (list, List):
            items = _coerce_sdk_schema(args[0]) if args else {}
            return {"type": "array", "items": items}
        if origin in (dict, Dict):
            additional = _coerce_sdk_schema(args[1]) if len(args) > 1 else {}
            return {"type": "object", "additionalProperties": additional}

    if isinstance(value, dict):
        properties: dict = {}
        required: list = []
        for key, item in value.items():
            properties[str(key)] = _coerce_sdk_schema(item)
            required.append(str(key))
        schema: dict = {"type": "object", "properties": properties}
        if required:
            schema["required"] = required
        return schema

    if isinstance(value, (list, tuple)):
        items = _coerce_sdk_schema(value[0]) if len(value) == 1 else {}
        return {"type": "array", "items": items}

    type_map = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
    }
    for typ, schema_val in type_map.items():
        if value is typ:
            return schema_val

    if value is list:
        return {"type": "array", "items": {}}
    if value is dict:
        return {"type": "object", "additionalProperties": {}}

    if isinstance(value, type):
        return {}

    return {}


Path = _Path


__all__ = [
    "McpServerInfo",
    "McpToolInfo",
    "McpResourceInfo",
    "ConfigScope",
    "TransportType",
    "load_mcp_server_configs",
    "parse_mcp_server_configs",
    "parse_mcp_config_option",
    "set_mcp_runtime_overrides",
    "clear_mcp_runtime_overrides",
    "get_existing_mcp_runtime",
    "load_mcp_servers",
    "load_mcp_servers_async",
    "ensure_mcp_runtime",
    "shutdown_mcp_runtime",
    "find_mcp_resource",
    "format_mcp_instructions",
    "estimate_mcp_tokens",
    "get_mcp_stderr_mode",
    "get_mcp_stderr_log_path",
    "set_sdk_mcp_request_sender",
    "clear_sdk_mcp_request_sender",
    "get_sdk_mcp_request_sender",
    "normalize_name_for_mcp",
    "mcp_info_from_string",
    "build_mcp_tool_name",
    "get_mcp_prefix",
    "expand_env_vars_in_string",
    "_coerce_sdk_schema",
    "_ensure_str_dict",
    "_load_json_file",
    "_normalize_command",
    "_project_scope_key",
    "_global_runtime",
    "_mcp_circuit_states",
    "McpRuntime",
    "_SdkMcpSession",
]
