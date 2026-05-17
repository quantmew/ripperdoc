"""
Shared utilities for expanding environment variables in MCP server configurations.
Mirrors reference: services/mcp/envExpansion.ts
"""

import os
import re
from typing import Dict, List, Tuple


def expand_env_vars_in_string(value: str) -> Tuple[str, List[str]]:
    """Expand environment variables in a string value.

    Handles ``${VAR}`` and ``${VAR:-default}`` syntax.

    Returns a tuple of ``(expanded_string, list_of_missing_vars)``.
    """
    missing_vars: List[str] = []

    def _replacer(match: re.Match) -> str:
        var_content = match.group(1)
        # Split on :- to support default values
        parts = var_content.split(":-", 1)
        var_name = parts[0]
        default_value = parts[1] if len(parts) > 1 else None

        env_value = os.environ.get(var_name)
        if env_value is not None:
            return env_value
        if default_value is not None:
            return default_value

        missing_vars.append(var_name)
        return match.group(0)

    expanded = re.sub(r"\$\{([^}]+)\}", _replacer, value)
    return expanded, missing_vars
