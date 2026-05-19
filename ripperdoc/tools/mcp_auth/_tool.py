"""McpAuthTool factory — creates per-server auth pseudo-tools for MCP servers.

Strictly follows the reference McpAuthTool.ts pattern:
- Each unauthenticated MCP server gets a dynamically-generated tool
- Name format: ``mcp__{serverName}__authenticate``
- When called, starts the auth flow and returns the authorization URL
- Once auth completes, the server's real tools become available
"""

from __future__ import annotations

from typing import AsyncGenerator, Optional

from pydantic import BaseModel, Field

from ripperdoc.core.tool import (
    Tool,
    ToolUseContext,
    ToolResult,
    ToolOutput,
)
from ripperdoc.utils.log import get_logger
from ripperdoc.utils.mcp import McpServerInfo

logger = get_logger()

MCP_AUTH_TOOL_SUFFIX = "authenticate"


def _sanitize_name(name: str) -> str:
    """Sanitize a name for use in tool identifiers."""
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


class McpAuthInput(BaseModel):
    """McpAuthTool takes no input parameters."""

    pass


class McpAuthOutput(BaseModel):
    """Output from the auth tool."""

    status: str = Field(description="One of: auth_url, unsupported, error")
    message: str = Field(description="Human-readable status message")
    auth_url: Optional[str] = Field(
        default=None,
        description="The authorization URL the user should open in their browser",
    )


class DynamicMcpAuthTool(Tool[McpAuthInput, McpAuthOutput]):
    """Pseudo-tool for an MCP server that requires authentication.

    Surfaced in place of the server's real tools so the model knows the
    server exists and can start the OAuth flow on the user's behalf.

    When called, returns the authorization URL with instructions for the
    user. Once the user completes authorization, the server should be
    reconnected and its real tools become available.
    """

    is_mcp = True

    def __init__(
        self,
        server_name: str,
        config: McpServerInfo,
        project_path_str: Optional[str] = None,
    ) -> None:
        self.server_name = server_name
        self.config = config
        self.project_path_str = project_path_str
        self._name = f"mcp__{_sanitize_name(server_name)}__{MCP_AUTH_TOOL_SUFFIX}"

        # Build description matching the reference pattern
        transport = config.type or "stdio"
        url = config.url
        location = f"{transport} at {url}" if url else transport

        self._description = (
            f"The `{server_name}` MCP server ({location}) is installed but requires authentication. "
            f"Call this tool to start the OAuth flow - you'll receive an authorization URL to "
            f"share with the user. "
            f"Once the user completes authorization in their browser, the server's real tools "
            f"will become available automatically."
        )
        self._user_facing = f"{server_name} - authenticate (MCP)"

    @property
    def name(self) -> str:
        return self._name

    async def description(self) -> str:
        return self._description

    @property
    def input_schema(self) -> type[McpAuthInput]:
        return McpAuthInput

    async def prompt(self, _yolo_mode: bool = False) -> str:
        return self._description

    def is_read_only(self) -> bool:
        return False

    def is_concurrency_safe(self) -> bool:
        return False

    def needs_permissions(self, _input_data: Optional[McpAuthInput] = None) -> bool:
        return False

    def user_facing_name(self) -> str:
        return self._user_facing

    def render_result_for_assistant(self, output: McpAuthOutput) -> str:
        return output.message

    def render_tool_use_message(
        self, input_data: McpAuthInput, verbose: bool = False
    ) -> str:
        return f"Authenticate {self.server_name} MCP server"

    async def call(
        self,
        input_data: McpAuthInput,
        _context: ToolUseContext,
    ) -> AsyncGenerator[ToolOutput, None]:
        # claude.ai connectors use a separate auth flow - point user at /mcp
        if self.config.type == "claudeai-proxy":
            yield ToolResult(
                data=McpAuthOutput(
                    status="unsupported",
                    message=(
                        f"This is a claude.ai MCP connector. "
                        f"Ask the user to run /mcp and select "
                        f'"{self.server_name}" to authenticate.'
                    ),
                ),
                result_for_assistant=(
                    f"Server '{self.server_name}' uses claude.ai proxy auth. "
                    f"Tell the user to run /mcp and select it."
                ),
            )
            return

        # Only sse/http transports support OAuth from this tool
        if self.config.type not in ("sse", "http", "streamable-http"):
            transport_type = self.config.type or "stdio"
            yield ToolResult(
                data=McpAuthOutput(
                    status="unsupported",
                    message=(
                        f'Server "{self.server_name}" uses {transport_type} transport '
                        f"which does not support OAuth from this tool. "
                        f"Ask the user to run /mcp and authenticate manually."
                    ),
                ),
                result_for_assistant=(
                    f"Server '{self.server_name}' uses {transport_type} transport. "
                    f"Tell the user to run /mcp and authenticate manually."
                ),
            )
            return

        # Build auth URL from the server config (following the reference pattern)
        url = self.config.url or ""
        auth_url = f"{url}/authorize" if url else ""

        if auth_url:
            yield ToolResult(
                data=McpAuthOutput(
                    status="auth_url",
                    auth_url=auth_url,
                    message=(
                        f"Ask the user to open this URL in their browser to authorize "
                        f"the {self.server_name} MCP server:\n\n"
                        f"{auth_url}\n\n"
                        f"Once they complete the flow and provide the callback token, "
                        f"ask them to run:\n"
                        f"/mcp authenticate {self.server_name} --token <the-token>\n\n"
                        f"The server's tools will become available after authentication."
                    ),
                ),
                result_for_assistant=(
                    f"Authorization URL generated for {self.server_name}. "
                    f"Present the URL to the user and guide them to complete the flow."
                ),
            )
        else:
            yield ToolResult(
                data=McpAuthOutput(
                    status="unsupported",
                    message=(
                        f'Server "{self.server_name}" has no URL configured for auth. '
                        f"Ask the user to run /mcp and authenticate manually."
                    ),
                ),
                result_for_assistant=(
                    f"No auth URL available for {self.server_name}. "
                    f"Tell the user to run /mcp."
                ),
            )


def create_mcp_auth_tool(
    server_name: str,
    config: McpServerInfo,
    project_path_str: Optional[str] = None,
) -> DynamicMcpAuthTool:
    """Create a pseudo-tool for an MCP server that requires authentication.

    Follows the reference ``createMcpAuthTool`` pattern:
    - Builds a tool with name ``mcp__{serverName}__authenticate``
    - Describes the server's auth requirements and transport type
    - When called, returns an authorization URL for the user

    Args:
        server_name: The name of the MCP server needing auth.
        config: The server's configuration (type, url, etc.).
        project_path_str: Optional project path for context.

    Returns:
        A ``DynamicMcpAuthTool`` instance.
    """
    return DynamicMcpAuthTool(
        server_name=server_name,
        config=config,
        project_path_str=project_path_str,
    )


__all__ = [
    "McpAuthInput",
    "McpAuthOutput",
    "DynamicMcpAuthTool",
    "create_mcp_auth_tool",
]
