"""MCP-related tools for listing servers, resources, and invoking MCP tools."""

from __future__ import annotations

import base64
import binascii
import json
import os
import tempfile
from typing import Any, AsyncGenerator, List, Optional

from pydantic import BaseModel, Field

from ripperdoc.core.tool import (
    Tool,
    ToolUseContext,
    ToolResult,
    ToolOutput,
    ValidationResult,
)
from ripperdoc.utils.log import get_logger
from ripperdoc.utils.mcp import (
    McpResourceInfo,
    McpServerInfo,
    ensure_mcp_runtime,
    find_mcp_resource,
    format_mcp_instructions,
    load_mcp_servers_async,
)
from ripperdoc.utils.token_estimation import estimate_tokens


logger = get_logger()

try:
    import mcp.types as mcp_types  # type: ignore
except (ImportError, ModuleNotFoundError):  # pragma: no cover - SDK may be missing at runtime
    mcp_types = None  # type: ignore[assignment]
    logger.debug("[mcp_tools] MCP SDK unavailable during import")

DEFAULT_MAX_MCP_OUTPUT_TOKENS = 25_000
MIN_MCP_OUTPUT_TOKENS = 1_000
DEFAULT_MCP_WARNING_FRACTION = 0.8


# =============================================================================
# Base class for MCP tools to reduce code duplication
# =============================================================================


class BaseMcpTool(Tool):  # type: ignore[type-arg]
    """Base class for MCP tools with common default implementations.

    Provides default implementations for common MCP tool behaviors:
    - is_read_only() returns True (MCP tools typically don't modify local state)
    - is_concurrency_safe() returns True (MCP operations can run in parallel)
    - needs_permissions() returns False (MCP tools handle their own auth)
    """

    def is_read_only(self) -> bool:
        """MCP tools are read-only by default."""
        return True

    def is_concurrency_safe(self) -> bool:
        """MCP operations can safely run concurrently."""
        return True

    def needs_permissions(self, input_data: Optional[Any] = None) -> bool:
        """MCP tools don't require additional permissions."""
        return False

    async def validate_server_exists(self, server_name: str) -> ValidationResult:
        """Validate that a server exists in the MCP runtime.

        Common validation helper for tools that take a server parameter.
        """
        runtime = await ensure_mcp_runtime()
        server_names = {s.name for s in runtime.servers}
        if server_name not in server_names:
            return ValidationResult(result=False, message=f"Unknown MCP server '{server_name}'.")
        return ValidationResult(result=True)


# =============================================================================
# End BaseMcpTool
# =============================================================================


def _get_mcp_token_limits() -> tuple[int, int]:
    """Compute warning and hard limits for MCP output size."""
    max_tokens = os.getenv("RIPPERDOC_MCP_MAX_OUTPUT_TOKENS")
    try:
        max_tokens_int = int(max_tokens) if max_tokens else DEFAULT_MAX_MCP_OUTPUT_TOKENS
    except (TypeError, ValueError):
        max_tokens_int = DEFAULT_MAX_MCP_OUTPUT_TOKENS
    max_tokens_int = max(MIN_MCP_OUTPUT_TOKENS, max_tokens_int)

    warn_env = os.getenv("RIPPERDOC_MCP_WARNING_TOKENS")
    try:
        warn_tokens_int = (
            int(warn_env) if warn_env else int(max_tokens_int * DEFAULT_MCP_WARNING_FRACTION)
        )
    except (TypeError, ValueError):
        warn_tokens_int = int(max_tokens_int * DEFAULT_MCP_WARNING_FRACTION)
    warn_tokens_int = max(MIN_MCP_OUTPUT_TOKENS, min(warn_tokens_int, max_tokens_int))
    return warn_tokens_int, max_tokens_int


def _evaluate_mcp_output_size(
    result_text: Optional[str],
    server_name: str,
    tool_name: str,
) -> tuple[Optional[str], Optional[str], int]:
    """Return (warning, error, token_estimate) for an MCP result text."""
    warn_tokens, max_tokens = _get_mcp_token_limits()
    token_estimate = estimate_tokens(result_text or "")

    if token_estimate > max_tokens:
        error_text = (
            f"MCP response from {server_name}:{tool_name} is ~{token_estimate:,} tokens, "
            f"which exceeds the configured limit of {max_tokens}. "
            "Refine the request (pagination/filtering) or raise RIPPERDOC_MCP_MAX_OUTPUT_TOKENS."
        )
        return None, error_text, token_estimate

    warning_text = None
    if result_text and token_estimate >= warn_tokens:
        line_count = result_text.count("\n") + 1
        warning_text = (
            f"WARNING: Large MCP response (~{token_estimate:,} tokens, {line_count:,} lines). "
            "This can fill the context quickly; consider pagination or filters."
        )
    return warning_text, None, token_estimate


class ListMcpServersInput(BaseModel):
    """Input for listing MCP servers."""

    server: Optional[str] = Field(default=None, description="Optional server name to filter")


class ListMcpServersOutput(BaseModel):
    """Server summary."""

    servers: List[dict]


class ListMcpServersTool(BaseMcpTool, Tool[ListMcpServersInput, ListMcpServersOutput]):
    """List configured MCP servers and their tools."""

    @property
    def name(self) -> str:
        return "ListMcpServers"

    async def description(self) -> str:
        return "List configured MCP servers and their available tools."

    @property
    def input_schema(self) -> type[ListMcpServersInput]:
        return ListMcpServersInput

    async def prompt(self, _yolo_mode: bool = False) -> str:
        servers = await load_mcp_servers_async()
        return format_mcp_instructions(servers)

    def render_result_for_assistant(self, output: ListMcpServersOutput) -> str:
        if not output.servers:
            return "No MCP servers configured."
        lines = ["Configured MCP servers:"]
        for server in output.servers:
            name = server.get("name", "unknown")
            status = server.get("status", "unknown")
            tool_names = server.get("tools", [])
            tool_part = ", ".join(tool_names) if tool_names else "no tools"
            lines.append(f"- {name} ({status}) tools: {tool_part}")
        return "\n".join(lines)

    def render_tool_use_message(
        self, input_data: ListMcpServersInput, _verbose: bool = False
    ) -> str:
        return f"List MCP servers{f' for {input_data.server}' if input_data.server else ''}"

    async def call(
        self,
        input_data: ListMcpServersInput,
        _context: ToolUseContext,
    ) -> AsyncGenerator[ToolOutput, None]:
        runtime = await ensure_mcp_runtime()
        servers: List[McpServerInfo] = runtime.servers
        if input_data.server:
            servers = [s for s in servers if s.name == input_data.server]

        payload = []
        for server in servers:
            payload.append(
                {
                    "name": server.name,
                    "status": server.status,
                    "command": server.command,
                    "args": server.args,
                    "tools": [tool.name for tool in server.tools],
                    "resources": [resource.uri for resource in server.resources],
                    "error": server.error,
                }
            )

        yield ToolResult(
            data=ListMcpServersOutput(servers=payload),
            result_for_assistant=self.render_result_for_assistant(
                ListMcpServersOutput(servers=payload)
            ),
        )


class ListMcpResourcesInput(BaseModel):
    """Input for listing MCP resources."""

    server: Optional[str] = Field(default=None, description="Optional server name to filter")


class ListMcpResourcesOutput(BaseModel):
    resources: List[dict]


class ListMcpResourcesTool(BaseMcpTool, Tool[ListMcpResourcesInput, ListMcpResourcesOutput]):
    """List resources exposed by MCP servers."""

    @property
    def name(self) -> str:
        return "ListMcpResources"

    async def description(self) -> str:
        return (
            "Lists available resources from configured MCP servers.\n"
            "Each resource object includes a 'server' field indicating which server it's from.\n\n"
            "Usage examples:\n"
            "- List all resources from all servers: `listMcpResources`\n"
            '- List resources from a specific server: `listMcpResources({ server: "myserver" })`'
        )

    @property
    def input_schema(self) -> type[ListMcpResourcesInput]:
        return ListMcpResourcesInput

    async def prompt(self, _yolo_mode: bool = False) -> str:
        return (
            "List available resources from configured MCP servers.\n"
            "Each returned resource will include all standard MCP resource fields plus a 'server' field\n"
            "indicating which server the resource belongs to.\n\n"
            "Parameters:\n"
            "- server (optional): The name of a specific MCP server to get resources from. If not provided,\n"
            "  resources from all servers will be returned."
        )

    async def validate_input(
        self, input_data: ListMcpResourcesInput, _context: Optional[ToolUseContext] = None
    ) -> ValidationResult:
        if input_data.server:
            return await self.validate_server_exists(input_data.server)
        return ValidationResult(result=True)

    def render_result_for_assistant(self, output: ListMcpResourcesOutput) -> str:
        if not output.resources:
            return "No MCP resources found."
        try:
            return json.dumps(output.resources, indent=2, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            logger.warning(
                "[mcp_tools] Failed to serialize MCP resources for assistant output: %s: %s",
                type(exc).__name__,
                exc,
            )
            return str(output.resources)

    def render_tool_use_message(
        self, input_data: ListMcpResourcesInput, _verbose: bool = False
    ) -> str:
        return f"List MCP resources{f' for {input_data.server}' if input_data.server else ''}"

    async def call(
        self,
        input_data: ListMcpResourcesInput,
        _context: ToolUseContext,
    ) -> AsyncGenerator[ToolOutput, None]:
        runtime = await ensure_mcp_runtime()
        servers = runtime.servers
        resources: List[dict] = []
        for server in servers:
            if input_data.server and server.name != input_data.server:
                continue

            session = runtime.sessions.get(server.name) if runtime else None
            fetched: List[McpResourceInfo] = []

            if (
                session
                and mcp_types
                and getattr(session, "list_resources", None)
                and server.capabilities.get("resources", False)
            ):
                try:
                    response = await session.list_resources()
                    fetched = [
                        McpResourceInfo(
                            uri=str(res.uri),
                            name=getattr(res, "name", None),
                            description=getattr(res, "description", "") or "",
                            mime_type=getattr(res, "mimeType", None),
                            size=getattr(res, "size", None),
                        )
                        for res in response.resources
                    ]
                except (OSError, RuntimeError, ConnectionError, ValueError) as exc:
                    # pragma: no cover - runtime errors
                    logger.warning(
                        "Failed to fetch resources from MCP server: %s: %s",
                        type(exc).__name__,
                        exc,
                        extra={"server": server.name},
                    )
                    fetched = []

            candidate_resources = fetched if fetched else server.resources

            for resource in candidate_resources:
                resources.append(
                    {
                        "server": server.name,
                        "uri": getattr(resource, "uri", None),
                        "name": getattr(resource, "name", None),
                        "description": getattr(resource, "description", None),
                        "mime_type": getattr(resource, "mime_type", None),
                        "size": getattr(resource, "size", None),
                    }
                )

        result = ListMcpResourcesOutput(resources=resources)
        yield ToolResult(
            data=result,
            result_for_assistant=self.render_result_for_assistant(result),
        )


class ReadMcpResourceInput(BaseModel):
    """Input for reading a single MCP resource."""

    server: str = Field(description="Server name")
    uri: str = Field(description="Resource URI")
    save_blobs: bool = Field(
        default=False,
        description="If true, binary resource contents will be written to a temporary file in addition to Base64.",
    )


class ResourceContentPart(BaseModel):
    """Structured representation for resource content blocks."""

    type: str
    uri: Optional[str] = None
    mime_type: Optional[str] = None
    text: Optional[str] = None
    size: Optional[int] = None
    base64_data: Optional[str] = None
    saved_path: Optional[str] = None


class ReadMcpResourceOutput(BaseModel):
    server: str
    uri: str
    content: Optional[str] = None
    contents: List[ResourceContentPart] = Field(default_factory=list)
    token_estimate: Optional[int] = None
    warning: Optional[str] = None
    is_error: bool = False


class ReadMcpResourceTool(BaseMcpTool, Tool[ReadMcpResourceInput, ReadMcpResourceOutput]):
    """Read a resource defined in MCP configuration."""

    @property
    def name(self) -> str:
        return "ReadMcpResource"

    async def description(self) -> str:
        return (
            "Reads a specific resource from an MCP server.\n"
            "- server: The name of the MCP server to read from\n"
            "- uri: The URI of the resource to read\n"
            "- save_blobs: Save binary content to a temporary file in addition to Base64 output\n\n"
            "Usage examples:\n"
            '- Read a resource from a server: `readMcpResource({ server: "myserver", uri: "my-resource-uri" })`'
        )

    @property
    def input_schema(self) -> type[ReadMcpResourceInput]:
        return ReadMcpResourceInput

    async def prompt(self, _yolo_mode: bool = False) -> str:
        return (
            "Reads a specific resource from an MCP server, identified by server name and resource URI.\n\n"
            "Parameters:\n"
            "- server (required): The name of the MCP server from which to read the resource\n"
            "- uri (required): The URI of the resource to read\n"
            "- save_blobs (optional): If true, write binary content to a temporary file and include the path"
        )

    async def validate_input(
        self, input_data: ReadMcpResourceInput, _context: Optional[ToolUseContext] = None
    ) -> ValidationResult:
        # First validate the server exists
        server_result = await self.validate_server_exists(input_data.server)
        if not server_result.result:
            return server_result

        # Then validate the resource exists on that server
        runtime = await ensure_mcp_runtime()
        resource = find_mcp_resource(runtime.servers, input_data.server, input_data.uri)
        if not resource:
            return ValidationResult(
                result=False,
                message=f"Resource '{input_data.uri}' not found on server '{input_data.server}'.",
            )
        return ValidationResult(result=True)

    def render_result_for_assistant(self, output: ReadMcpResourceOutput) -> str:
        if output.contents:
            texts = [part.text for part in output.contents if part.text]
            blob_notes = [
                f"[binary {part.mime_type or 'unknown'} {part.size or 'unknown'} bytes{f'; saved to {part.saved_path}' if part.saved_path else ''}]"
                for part in output.contents
                if part.type == "blob"
            ]
            if texts or blob_notes:
                return "\n".join([*texts, *blob_notes]).strip()
        if not output.content:
            return f"MCP resource {output.uri} on {output.server} has no content."
        return output.content

    def render_tool_use_message(
        self, input_data: ReadMcpResourceInput, _verbose: bool = False
    ) -> str:
        return f"Read MCP resource {input_data.uri} from {input_data.server}"

    async def call(
        self,
        input_data: ReadMcpResourceInput,
        _context: ToolUseContext,
    ) -> AsyncGenerator[ToolOutput, None]:
        runtime = await ensure_mcp_runtime()
        session = runtime.sessions.get(input_data.server) if runtime else None

        content_text = None
        parts: List[ResourceContentPart] = []

        if session and mcp_types:
            try:
                # Convert string to AnyUrl
                from mcp.types import AnyUrl

                uri = AnyUrl(input_data.uri)
                result = await session.read_resource(uri)
                for item in result.contents:
                    if isinstance(item, mcp_types.TextResourceContents):
                        parts.append(
                            ResourceContentPart(
                                type="text",
                                uri=getattr(item, "uri", None),
                                mime_type=getattr(item, "mimeType", None),
                                text=item.text,
                            )
                        )
                    elif isinstance(item, mcp_types.BlobResourceContents):
                        blob_data = getattr(item, "blob", None)
                        base64_data = None
                        saved_path = None
                        if isinstance(blob_data, (bytes, bytearray)):
                            base64_data = base64.b64encode(blob_data).decode("ascii")
                            raw_bytes = blob_data
                        elif isinstance(blob_data, str):
                            base64_data = blob_data
                            try:
                                raw_bytes = base64.b64decode(blob_data)
                            except (ValueError, binascii.Error) as exc:
                                logger.warning(
                                    "[mcp_tools] Failed to decode base64 blob content: %s: %s",
                                    type(exc).__name__,
                                    exc,
                                    extra={"server": input_data.server, "uri": input_data.uri},
                                )
                                raw_bytes = None
                        else:
                            raw_bytes = None

                        if input_data.save_blobs and raw_bytes:
                            suffix = ""
                            mime = getattr(item, "mimeType", None) or ""
                            if "/" in mime:
                                suffix = "." + mime.split("/")[-1]
                            fd, path = tempfile.mkstemp(prefix="ripperdoc_mcp_", suffix=suffix)
                            with os.fdopen(fd, "wb") as handle:
                                handle.write(raw_bytes)
                            saved_path = path

                        parts.append(
                            ResourceContentPart(
                                type="blob",
                                uri=getattr(item, "uri", None),
                                mime_type=getattr(item, "mimeType", None),
                                size=getattr(item, "size", None),
                                base64_data=base64_data,
                                saved_path=saved_path,
                            )
                        )
                text_parts = [p.text for p in parts if p.text]
                content_text = "\n".join([p for p in text_parts if p]) or None
            except (OSError, RuntimeError, ConnectionError, ValueError, KeyError) as exc:
                # pragma: no cover - runtime errors
                logger.warning(
                    "Error reading MCP resource: %s: %s",
                    type(exc).__name__,
                    exc,
                    extra={"server": input_data.server, "uri": input_data.uri},
                )
                content_text = f"Error reading MCP resource: {exc}"
        else:
            resource = find_mcp_resource(runtime.servers, input_data.server, input_data.uri)
            content_text = resource.text if resource else None
            if resource:
                parts.append(
                    ResourceContentPart(
                        type="text",
                        uri=resource.uri,
                        mime_type=resource.mime_type,
                        text=resource.text,
                        size=resource.size,
                    )
                )

        read_result: Any = ReadMcpResourceOutput(
            server=input_data.server, uri=input_data.uri, content=content_text, contents=parts
        )
        assistant_text = self.render_result_for_assistant(read_result)  # type: ignore[arg-type]
        warning_text, error_text, token_estimate = _evaluate_mcp_output_size(
            assistant_text, input_data.server, f"resource:{input_data.uri}"
        )

        if error_text:
            limited_result = ReadMcpResourceOutput(
                server=input_data.server,
                uri=input_data.uri,
                content=None,
                contents=[],
                token_estimate=token_estimate,
                warning=None,
                is_error=True,
            )
            yield ToolResult(data=limited_result, result_for_assistant=error_text)
            return

        annotated_result = read_result.model_copy(
            update={"token_estimate": token_estimate, "warning": warning_text}
        )

        final_text = assistant_text or ""
        if not final_text and warning_text:
            final_text = warning_text

        yield ToolResult(
            data=annotated_result,
            result_for_assistant=final_text,  # type: ignore[arg-type]
        )


# Re-export DynamicMcpTool and related functions from the dedicated module
# for backward compatibility. The imports are placed at the end to avoid
# circular import issues since dynamic_mcp_tool.py imports _evaluate_mcp_output_size.
from ripperdoc.tools.dynamic_mcp_tool import (  # noqa: E402
    DynamicMcpTool,
    McpToolCallOutput,
    load_dynamic_mcp_tools_async,
    load_dynamic_mcp_tools_sync,
    merge_tools_with_dynamic,
)

__all__ = [
    "ListMcpServersTool",
    "ListMcpResourcesTool",
    "ReadMcpResourceTool",
    "DynamicMcpTool",
    "McpToolCallOutput",
    "load_dynamic_mcp_tools_async",
    "load_dynamic_mcp_tools_sync",
    "merge_tools_with_dynamic",
]
