"""User-Agent generation for Ripperdoc API requests.

Format: ripperdoc-cli/{version} (external, {source}) agent-sdk/{sdk_version}

Examples:
- CLI: ripperdoc-cli/0.4.4 (external, cli) agent-sdk/0.4.4
- Python SDK: ripperdoc-cli/0.4.4 (external, sdk-py) agent-sdk/0.4.4
- TypeScript SDK: ripperdoc-cli/0.4.4 (external, sdk-ts) agent-sdk/0.4.4
- VSCode extension: ripperdoc-cli/0.4.4 (external, vscode) agent-sdk/0.4.4
"""

from __future__ import annotations

import os
from typing import Literal

from ripperdoc import __version__

# Source types for user-agent
UserAgentSource = Literal["cli", "sdk-py", "sdk-ts", "sdk-cli", "vscode"]

# Environment variables
RIPPERDOC_CLIENT_SOURCE_ENV = "RIPPERDOC_CLIENT_SOURCE"
RIPPERDOC_AGENT_SDK_VERSION_ENV = "RIPPERDOC_AGENT_SDK_VERSION"

# Default source when not specified
DEFAULT_SOURCE: UserAgentSource = "cli"


def get_client_source() -> UserAgentSource:
    """Get the client source type from environment or default.

    Returns:
        The client source type (cli, sdk-py, sdk-ts, sdk-cli, vscode)
    """
    source = os.environ.get(RIPPERDOC_CLIENT_SOURCE_ENV, "").lower()
    valid_sources: set[UserAgentSource] = {"cli", "sdk-py", "sdk-ts", "sdk-cli", "vscode"}
    if source in valid_sources:
        return source  # type: ignore
    return DEFAULT_SOURCE


def build_user_agent(source: UserAgentSource | None = None) -> str:
    """Build the User-Agent header value.

    Args:
        source: Optional source type override. If not provided, uses environment
                variable or defaults to "cli".

    Returns:
        User-Agent string in format: ripperdoc-cli/{version} (external, {source}) agent-sdk/{sdk_version}
    """
    if source is None:
        source = get_client_source()

    version = __version__
    sdk_version = os.environ.get(RIPPERDOC_AGENT_SDK_VERSION_ENV, version)
    if source != "cli":
        return f"ripperdoc-cli/{version} (external, {source}, agent-sdk/{sdk_version})"
    else:
        return f"ripperdoc-cli/{version} (external, {source})"


# Pre-built user-agents for common use cases
USER_AGENT_CLI = build_user_agent("cli")
USER_AGENT_SDK_PY = build_user_agent("sdk-py")
USER_AGENT_SDK_TS = build_user_agent("sdk-ts")
USER_AGENT_SDK_CLI = build_user_agent("sdk-cli")
USER_AGENT_VSCODE = build_user_agent("vscode")
