"""MCP-related tools for listing servers, resources, and invoking MCP tools."""

from __future__ import annotations

import base64
import binascii
import json
from typing import Any, AsyncGenerator, List, Optional

from pydantic import BaseModel, Field

from ripperdoc.core.tool import Tool
from ripperdoc.utils.log import get_logger
from ripperdoc.utils.mcp import (
    McpResourceInfo,
    McpServerInfo,
    ensure_mcp_runtime,
    find_mcp_resource,
    format_mcp_instructions,
    load_mcp_servers_async,
)
from ripperdoc.utils.filesystem.temp_paths import ripperdoc_mkstemp
from ripperdoc.tools.mcp.mcp_output_limits import evaluate_mcp_output_size

logger = get_logger()

try:
    import mcp.types as mcp_types  # type: ignore
except (ImportError, ModuleNotFoundError):
    mcp_types = None  # type: ignore[assignment]
    logger.debug("[mcp_tools] MCP SDK unavailable during import")


class BaseMcpTool(Tool):  # type: ignore[type-arg]
    """Base class for MCP tools with common default implementations."""

    def is_read_only(self) -> bool:
        return True

    def is_concurrency_safe(self) -> bool:
        return True

    def needs_permissions(self, _input_data: Optional[BaseModel] = None) -> bool:
        return False


class ListMcpServersToolInput(BaseModel):
    """Input schema for ListMcpServersTool."""

    server: Optional[str] = Field(
        default=None,
        description="Optional server name to filter",
    )


class ListMcpServersToolOutput(BaseModel):
    """Output from listing MCP servers."""

    servers: List[McpServerInfo]
    server: Optional[str] = None


class ListMcpServersTool(BaseMcpTool[  # type: ignore[type-arg]
    ListMcpServersToolInput, ListMcpServersToolOutput
]):
    """Tool for listing MCP servers."""

    @property
    def name(self) -> str:
        return "ListMcpServers"

    async def description(self) -> str:
        return "List configured MCP servers and their available tools."

    @property
    def input_schema(self) -> type[ListMcpServersToolInput]:
        return ListMcpServersToolInput

    async def call(self, input_data: ListMcpServersToolInput, _context: Any) -> AsyncGenerator[Any, None]:  # pragma: no cover
        from ripperdoc.core.tool import ToolResult
        runtime = ensure_mcp_runtime()
        servers = await runtime.list_servers()
        if input_data.server:
            servers = [s for s in servers if s.name == input_data.server]
        output = ListMcpServersToolOutput(servers=servers, server=input_data.server)
        yield ToolResult(data=output, result_for_assistant=f"Found {len(servers)} MCP server(s).")


class ListMcpResourcesToolInput(BaseModel):
    """Input schema for ListMcpResourcesTool."""

    server: Optional[str] = Field(
        default=None,
        description="Optional server name to filter",
    )


class ListMcpResourcesToolOutput(BaseModel):
    """Output from listing MCP resources."""

    resources: List[McpResourceInfo]
    server: Optional[str] = None


class ListMcpResourcesTool(BaseMcpTool[  # type: ignore[type-arg]
    ListMcpResourcesToolInput, ListMcpResourcesToolOutput
]):
    """Tool for listing MCP resources."""

    @property
    def name(self) -> str:
        return "ListMcpResources"

    async def description(self) -> str:
        return "Lists available resources from configured MCP servers."

    @property
    def input_schema(self) -> type[ListMcpResourcesToolInput]:
        return ListMcpResourcesToolInput

    async def call(self, input_data: ListMcpResourcesToolInput, _context: Any) -> AsyncGenerator[Any, None]:  # pragma: no cover
        from ripperdoc.core.tool import ToolResult
        runtime = ensure_mcp_runtime()
        resources = await runtime.list_resources(server_name=input_data.server)
        output = ListMcpResourcesToolOutput(resources=resources, server=input_data.server)
        yield ToolResult(data=output, result_for_assistant=f"Found {len(resources)} MCP resource(s).")


class ReadMcpResourceToolInput(BaseModel):
    """Input schema for ReadMcpResourceTool."""

    server: str = Field(description="Server name")
    uri: str = Field(description="Resource URI")
    save_blobs: bool = Field(
        default=False,
        description="If true, binary resource contents will be written to a temporary file in addition to Base64.",
    )


class ReadMcpResourceToolOutput(BaseModel):
    """Output from reading an MCP resource."""

    server: str
    uri: str
    content_type: str = ""
    text: Optional[str] = None
    blob: Optional[str] = None
    temp_file: Optional[str] = None


class ReadMcpResourceTool(BaseMcpTool[  # type: ignore[type-arg]
    ReadMcpResourceToolInput, ReadMcpResourceToolOutput
]):
    """Tool for reading MCP resources."""

    @property
    def name(self) -> str:
        return "ReadMcpResource"

    async def description(self) -> str:
        return "Reads a specific resource from an MCP server."

    @property
    def input_schema(self) -> type[ReadMcpResourceToolInput]:
        return ReadMcpResourceToolInput

    async def call(self, input_data: ReadMcpResourceToolInput, _context: Any) -> AsyncGenerator[Any, None]:  # pragma: no cover
        from ripperdoc.core.tool import ToolResult
        runtime = ensure_mcp_runtime()
        try:
            result = await runtime.read_resource(
                input_data.server, input_data.uri,
            )
        except (ValueError, RuntimeError) as exc:
            output = ReadMcpResourceToolOutput(
                server=input_data.server,
                uri=input_data.uri,
                text=f"Error reading resource: {exc}",
            )
            yield ToolResult(data=output, result_for_assistant=f"Error: {exc}")
            return

        text: Optional[str] = None
        blob: Optional[str] = None
        temp_file: Optional[str] = None
        content_type = ""

        if isinstance(result, bytes):
            blob = base64.b64encode(result).decode("utf-8")
            content_type = "application/octet-stream"
            if input_data.save_blobs:
                try:
                    fd, path = ripperdoc_mkstemp(suffix=".bin", prefix="mcp_resource_")
                    with open(fd, "wb") as f:
                        f.write(result)
                    temp_file = path
                except (OSError, IOError, RuntimeError) as exc:
                    logger.warning(
                        "[mcp_tools] Failed to write blob to temp file: %s", exc,
                    )
        elif isinstance(result, str):
            text = result
            content_type = "text/plain"
        else:
            text = str(result)
            content_type = "text/plain"

        output = ReadMcpResourceToolOutput(
            server=input_data.server,
            uri=input_data.uri,
            content_type=content_type,
            text=text,
            blob=blob,
            temp_file=temp_file,
        )
        yield ToolResult(
            data=output,
            result_for_assistant=(
                f"Read MCP resource from {input_data.server}/{input_data.uri}: "
                f"type={content_type}, "
                f"{'text length ' + str(len(text)) if text else ''}"
                f"{'blob length ' + str(len(blob)) if blob else ''}"
                f"{', saved to ' + temp_file if temp_file else ''}"
            ),
        )


class MCPToolArgs(BaseModel):
    """Arguments for invoking an MCP tool."""

    name: str
    arguments: Optional[dict[str, Any]] = None


class MCPToolInput(BaseModel):
    """Input schema for MCPTool (dynamic tool invocation)."""

    server: str = Field(description="The name of the MCP server to call")
    tool_name: str = Field(
        description="The name of the MCP tool to invoke",
        validation_alias="toolName",
        serialization_alias="toolName",
    )
    arguments: Optional[dict[str, Any]] = Field(
        default=None,
        description="Arguments to pass to the MCP tool",
    )
    save_blobs: bool = Field(
        default=False,
        description="Save binary content to a temporary file in addition to Base64 output",
        validation_alias="saveBlobs",
        serialization_alias="saveBlobs",
    )


class MCPToolOutput(BaseModel):
    """Output from invoking an MCP tool."""

    server: str
    tool_name: str
    content: list[dict[str, Any]] = Field(default_factory=list)
    is_error: bool = False
    temp_files: list[str] = Field(default_factory=list)
    output_truncated: bool = False


class MCPTool(BaseMcpTool[MCPToolInput, MCPToolOutput]):  # type: ignore[type-arg]
    """Tool for invoking MCP tools."""

    @property
    def name(self) -> str:
        return "MCPTool"

    async def description(self) -> str:
        return "Invoke an MCP tool on a specific server."

    @property
    def input_schema(self) -> type[MCPToolInput]:
        return MCPToolInput

    async def call(self, input_data: MCPToolInput, _context: Any) -> AsyncGenerator[Any, None]:  # pragma: no cover
        from ripperdoc.core.tool import ToolResult
        runtime = ensure_mcp_runtime()
        try:
            result = await runtime.call_tool(
                input_data.server,
                input_data.tool_name,
                arguments=input_data.arguments,
            )
        except (ValueError, RuntimeError, ConnectionError) as exc:
            output = MCPToolOutput(
                server=input_data.server,
                tool_name=input_data.tool_name,
                is_error=True,
                content=[{"type": "text", "text": f"MCP error: {exc}"}],
            )
            yield ToolResult(data=output, result_for_assistant=f"Error: {exc}")
            return

        temp_files: list[str] = []
        content_parts = []
        has_text = False
        for item in (result.content if hasattr(result, "content") else []):
            item_dict: dict[str, Any] = {}
            try:
                if hasattr(item, "model_dump"):
                    item_dict = item.model_dump()
                elif isinstance(item, dict):
                    item_dict = item
                else:
                    item_dict = {"type": "text", "text": str(item)}
            except (ValueError, TypeError, RuntimeError):
                item_dict = {"type": "text", "text": str(item)}

            content_type = item_dict.get("type", "text")
            if content_type == "text":
                has_text = True
            elif content_type in ("image", "audio", "video", "blob"):
                data_field = item_dict.get("data", "")
                if isinstance(data_field, str):
                    try:
                        decoded = base64.b64decode(data_field)
                        mime_type = item_dict.get("mimeType", "application/octet-stream")
                        ext = _mime_to_ext(mime_type)
                        if input_data.save_blobs:
                            try:
                                fd, path = ripperdoc_mkstemp(suffix=ext, prefix="mcp_output_")
                                with open(fd, "wb") as f:
                                    f.write(decoded)
                                temp_files.append(path)
                            except (OSError, IOError) as exc:
                                logger.warning(
                                    "[mcp_tools] Failed to save blob: %s", exc,
                                )
                    except (binascii.Error, ValueError, TypeError) as exc:
                        logger.warning(
                            "[mcp_tools] Failed to decode blob: %s", exc,
                        )

            content_parts.append(item_dict)

        if not has_text:
            content_parts.append({"type": "text", "text": "(binary content)"})

        size_ok, size_info = evaluate_mcp_output_size(content_parts)
        output_truncated = not size_ok

        output = MCPToolOutput(
            server=input_data.server,
            tool_name=input_data.tool_name,
            content=content_parts,
            is_error=getattr(result, "isError", False) if hasattr(result, "isError") else False,
            temp_files=temp_files,
            output_truncated=output_truncated,
        )

        if output_truncated:
            yield ToolResult(
                data=output,
                result_for_assistant=(
                    f"MCP tool {input_data.server}/{input_data.tool_name} returned "
                    f"large output ({size_info}). "
                    "Output has been truncated per MCP output limits."
                ),
            )
        else:
            result_text = _summarize_mcp_result(content_parts)
            yield ToolResult(
                data=output,
                result_for_assistant=(
                    f"Invoked MCP tool {input_data.server}/{input_data.tool_name}. "
                    f"Result: {result_text}"
                ),
            )


def _mime_to_ext(mime_type: str) -> str:
    _MIME_EXT_MAP = {
        "image/png": ".png", "image/jpeg": ".jpg", "image/gif": ".gif",
        "image/webp": ".webp", "image/svg+xml": ".svg",
        "audio/mpeg": ".mp3", "audio/wav": ".wav", "audio/ogg": ".ogg",
        "video/mp4": ".mp4", "video/webm": ".webm",
        "application/pdf": ".pdf", "application/json": ".json",
    }
    return _MIME_EXT_MAP.get(mime_type, ".bin")


def _summarize_mcp_result(content_parts: list[dict[str, Any]]) -> str:
    text_count = 0
    other_count = 0
    total_chars = 0
    for part in content_parts:
        ptype = part.get("type", "text")
        if ptype == "text":
            text_count += 1
            total_chars += len(part.get("text", ""))
        else:
            other_count += 1
    parts = []
    if text_count:
        parts.append(f"{text_count} text part(s) ({total_chars} chars)")
    if other_count:
        parts.append(f"{other_count} other part(s)")
    return ", ".join(parts) if parts else "empty result"
