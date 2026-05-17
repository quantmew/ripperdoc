"""
Pure utility functions for MCP name normalization.
Mirrors reference: services/mcp/normalization.ts

This file has minimal dependencies to keep it lightweight.
"""

import re

# Claude.ai server names are prefixed with this string
CLAUDEAI_SERVER_PREFIX = "claude.ai "


def normalize_name_for_mcp(name: str) -> str:
    """Normalize server names to be compatible with the API pattern ^[a-zA-Z0-9_-]{1,64}$.

    Replaces any invalid characters (including dots and spaces) with underscores.

    For claude.ai servers (names starting with "claude.ai "), also collapses
    consecutive underscores and strips leading/trailing underscores to prevent
    interference with the __ delimiter used in MCP tool names.
    """
    normalized = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    if name.startswith(CLAUDEAI_SERVER_PREFIX):
        normalized = re.sub(r"_+", "_", normalized).strip("_")
    # Truncate to reasonable length
    if len(normalized) > 64:
        normalized = normalized[:64]
    return normalized
