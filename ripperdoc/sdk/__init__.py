"""Lightweight Python SDK for using Ripperdoc headlessly."""

from ripperdoc.sdk.client import (
    AgentConfig,
    HookCallback,
    HookMatcher,
    McpServerConfig,
    PermissionMode,
    RipperdocClient,
    RipperdocOptions,
    SettingSource,
    StderrCallback,
    clear_programmatic_registries,
    get_programmatic_agents,
    get_programmatic_hooks,
    query,
)

__all__ = [
    # Core classes
    "RipperdocClient",
    "RipperdocOptions",
    "query",
    # Enums
    "PermissionMode",
    "SettingSource",
    # Configuration types
    "McpServerConfig",
    "AgentConfig",
    "HookMatcher",
    # Callback types
    "HookCallback",
    "StderrCallback",
    # Registry access
    "get_programmatic_agents",
    "get_programmatic_hooks",
    "clear_programmatic_registries",
]
