"""User-Agent generation for Ripperdoc API requests.

Format: ripperdoc-cli/{version} (external, {source}) agent-sdk/{sdk_version}

Examples:
- CLI: ripperdoc-cli/0.4.4 (external, cli) agent-sdk/0.4.4
- Python SDK: ripperdoc-cli/0.4.4 (external, sdk-py) agent-sdk/0.4.4
- TypeScript SDK: ripperdoc-cli/0.4.4 (external, sdk-ts) agent-sdk/0.4.4
- VSCode extension: ripperdoc-cli/0.4.4 (external, vscode) agent-sdk/0.4.4
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Literal, Optional

from ripperdoc import __version__

# Source types for user-agent
UserAgentSource = Literal["cli", "sdk-py", "sdk-ts", "sdk-cli", "vscode"]

# Environment variables
RIPPERDOC_CLIENT_SOURCE_ENV = "RIPPERDOC_CLIENT_SOURCE"
RIPPERDOC_AGENT_SDK_VERSION_ENV = "RIPPERDOC_AGENT_SDK_VERSION"
RIPPERDOC_CUSTOM_USER_AGENT_ENV = "RIPPERDOC_CUSTOM_USER_AGENT"
RIPPERDOC_CUSTOM_HEADERS_ENV = "RIPPERDOC_CUSTOM_HEADERS"

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

    If RIPPERDOC_CUSTOM_USER_AGENT is set, returns that value directly.

    Args:
        source: Optional source type override. If not provided, uses environment
                variable or defaults to "cli".

    Returns:
        User-Agent string in format: ripperdoc-cli/{version} (external, {source}) agent-sdk/{sdk_version}
    """
    custom = os.environ.get(RIPPERDOC_CUSTOM_USER_AGENT_ENV, "").strip()
    if custom:
        return custom

    if source is None:
        source = get_client_source()

    version = __version__
    sdk_version = os.environ.get(RIPPERDOC_AGENT_SDK_VERSION_ENV, version)
    if source != "cli":
        return f"ripperdoc-cli/{version} (external, {source}, agent-sdk/{sdk_version})"
    else:
        return f"ripperdoc-cli/{version} (external, {source})"


def load_custom_headers_env() -> Dict[str, str]:
    """Load custom headers from the RIPPERDOC_CUSTOM_HEADERS environment variable.

    The value should be a JSON object: '{"Header-Name": "value", ...}'

    Returns:
        Dictionary of custom headers, or empty dict if not set or invalid.
    """
    raw = os.environ.get(RIPPERDOC_CUSTOM_HEADERS_ENV, "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(k): str(v) for k, v in parsed.items() if isinstance(k, str)}


def build_request_headers(
    *,
    profile_headers: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Build merged headers for an outgoing API request.

    Merge order (later wins):
    1. Default User-Agent (from build_user_agent)
    2. RIPPERDOC_CUSTOM_HEADERS env var (global extra headers)
    3. Profile-level headers from config.json

    Returns:
        Merged header dict.
    """
    headers: Dict[str, Any] = {"User-Agent": build_user_agent()}

    # Layer 2: global env var headers
    env_headers = load_custom_headers_env()
    if env_headers:
        headers.update(env_headers)

    # Layer 3: per-profile headers from config
    if profile_headers:
        headers.update(profile_headers)

    return headers


# Pre-built user-agents for common use cases
USER_AGENT_CLI = build_user_agent("cli")
USER_AGENT_SDK_PY = build_user_agent("sdk-py")
USER_AGENT_SDK_TS = build_user_agent("sdk-ts")
USER_AGENT_SDK_CLI = build_user_agent("sdk-cli")
USER_AGENT_VSCODE = build_user_agent("vscode")
