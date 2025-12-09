"""MCP-related tools for listing servers, resources, and invoking MCP tools."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import tempfile
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict

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
    get_existing_mcp_runtime,
    load_mcp_servers_async,
    shutdown_mcp_runtime,
)
from ripperdoc.utils.token_estimation import estimate_tokens


logger = get_logger()

try:
    import mcp.types as mcp_types  # type: ignore
except Exception:  # pragma: no cover - SDK may be missing at runtime
    mcp_types = None  # type: ignore[assignment]
    logger.exception("[mcp_tools] MCP SDK unavailable during import")

DEFAULT_MAX_MCP_OUTPUT_TOKENS = 25_000
MIN_MCP_OUTPUT_TOKENS = 1_000
DEFAULT_MCP_WARNING_FRACTION = 0.8


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
        warn_tokens_int = int(warn_env) if warn_env else int(max_tokens_int * DEFAULT_MCP_WARNING_FRACTION)
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


def _content_block_to_text(block: Any) -> str:
    block_type = getattr(block, "type", None) or (
        block.get("type") if isinstance(block, dict) else None
    )
    if block_type == "text":
        return str(getattr(block, "text", None) or block.get("text", ""))
    if block_type == "resource":
        resource = getattr(block, "resource", None) or block.get("resource")
        prefix = "resource"
        if isinstance(resource, dict):
            uri = resource.get("uri") or ""
            text = resource.get("text") or ""
            blob = resource.get("blob")
            if text:
                return f"[Resource {uri}] {text}"
            if blob:
                return f"[Resource {uri}] (binary content {len(str(blob))} chars)"
        if hasattr(resource, "uri"):
            uri = getattr(resource, "uri", "")
            text = getattr(resource, "text", None)
            blob = getattr(resource, "blob", None)
            if text:
                return f"[Resource {uri}] {text}"
            if blob:
                return f"[Resource {uri}] (binary content {len(str(blob))} chars)"
        return prefix
    if block_type == "resource_link":
        uri = getattr(block, "uri", None) or (block.get("uri") if isinstance(block, dict) else None)
        return f"[Resource link] {uri}" if uri else "[Resource link]"
    if block_type == "image":
        mime = getattr(block, "mimeType", None) or (
            block.get("mimeType") if isinstance(block, dict) else None
        )
        return f"[Image content {mime or ''}]".strip()
    if block_type == "audio":
        mime = getattr(block, "mimeType", None) or (
            block.get("mimeType") if isinstance(block, dict) else None
        )
        return f"[Audio content {mime or ''}]".strip()
    return str(block)


def _render_content_blocks(blocks: List[Any]) -> str:
    if not blocks:
        return ""
    parts = [_content_block_to_text(block) for block in blocks]
    return "\n".join([p for p in parts if p])


def _normalize_content_block(block: Any) -> Any:
    """Convert MCP content blocks to JSON-serializable structures."""
    if isinstance(block, dict):
        return block
    result: Dict[str, Any] = {}
    for attr in (
        "type",
        "text",
        "mimeType",
        "data",
        "name",
        "uri",
        "description",
        "resource",
        "blob",
    ):
        if hasattr(block, attr):
            result[attr] = getattr(block, attr)
    if result:
        return result
    return str(block)


def _normalize_content_blocks(blocks: Optional[List[Any]]) -> Optional[List[Any]]:
    if not blocks:
        return None
    return [_normalize_content_block(block) for block in blocks]


class ListMcpServersInput(BaseModel):
    """Input for listing MCP servers."""

    server: Optional[str] = Field(default=None, description="Optional server name to filter")


class ListMcpServersOutput(BaseModel):
    """Server summary."""

    servers: List[dict]


class ListMcpServersTool(Tool[ListMcpServersInput, ListMcpServersOutput]):
    """List configured MCP servers and their tools."""

    @property
    def name(self) -> str:
        return "ListMcpServers"

    async def description(self) -> str:
        return "List configured MCP servers and their available tools."

    @property
    def input_schema(self) -> type[ListMcpServersInput]:
        return ListMcpServersInput

    async def prompt(self, safe_mode: bool = False) -> str:
        servers = await load_mcp_servers_async()
        return format_mcp_instructions(servers)

    def is_read_only(self) -> bool:
        return True

    def is_concurrency_safe(self) -> bool:
        return True

    def needs_permissions(self, input_data: Optional[ListMcpServersInput] = None) -> bool:
        return False

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
        self, input_data: ListMcpServersInput, verbose: bool = False
    ) -> str:
        return f"List MCP servers{f' for {input_data.server}' if input_data.server else ''}"

    async def call(
        self,
        input_data: ListMcpServersInput,
        context: ToolUseContext,
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


class ListMcpResourcesTool(Tool[ListMcpResourcesInput, ListMcpResourcesOutput]):
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

    async def prompt(self, safe_mode: bool = False) -> str:
        return (
            "List available resources from configured MCP servers.\n"
            "Each returned resource will include all standard MCP resource fields plus a 'server' field\n"
            "indicating which server the resource belongs to.\n\n"
            "Parameters:\n"
            "- server (optional): The name of a specific MCP server to get resources from. If not provided,\n"
            "  resources from all servers will be returned."
        )

    def is_read_only(self) -> bool:
        return True

    def is_concurrency_safe(self) -> bool:
        return True

    def needs_permissions(self, input_data: Optional[ListMcpResourcesInput] = None) -> bool:
        return False

    async def validate_input(
        self, input_data: ListMcpResourcesInput, context: Optional[ToolUseContext] = None
    ) -> ValidationResult:
        runtime = await ensure_mcp_runtime()
        server_names = {s.name for s in runtime.servers}
        if input_data.server and input_data.server not in server_names:
            return ValidationResult(
                result=False, message=f"Unknown MCP server '{input_data.server}'."
            )
        return ValidationResult(result=True)

    def render_result_for_assistant(self, output: ListMcpResourcesOutput) -> str:
        if not output.resources:
            return "No MCP resources found."
        try:
            return json.dumps(output.resources, indent=2, ensure_ascii=False)
        except Exception:
            logger.exception("[mcp_tools] Failed to serialize MCP resources for assistant output")
            return str(output.resources)

    def render_tool_use_message(
        self, input_data: ListMcpResourcesInput, verbose: bool = False
    ) -> str:
        return f"List MCP resources{f' for {input_data.server}' if input_data.server else ''}"

    async def call(
        self,
        input_data: ListMcpResourcesInput,
        context: ToolUseContext,
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
                except Exception as exc:  # pragma: no cover - runtime errors
                    logger.exception(
                        "Failed to fetch resources from MCP server",
                        extra={"server": server.name, "error": str(exc)},
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


class McpToolCallOutput(BaseModel):
    """Standardized output for MCP tool calls."""

    server: str
    tool: str
    content: Optional[str] = None
    text: Optional[str] = None
    content_blocks: Optional[List[Any]] = None
    structured_content: Optional[dict] = None
    is_error: bool = False
    token_estimate: Optional[int] = None
    warning: Optional[str] = None


class ReadMcpResourceTool(Tool[ReadMcpResourceInput, ReadMcpResourceOutput]):
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

    async def prompt(self, safe_mode: bool = False) -> str:
        return (
            "Reads a specific resource from an MCP server, identified by server name and resource URI.\n\n"
            "Parameters:\n"
            "- server (required): The name of the MCP server from which to read the resource\n"
            "- uri (required): The URI of the resource to read\n"
            "- save_blobs (optional): If true, write binary content to a temporary file and include the path"
        )

    def is_read_only(self) -> bool:
        return True

    def is_concurrency_safe(self) -> bool:
        return True

    def needs_permissions(self, input_data: Optional[ReadMcpResourceInput] = None) -> bool:
        return False

    async def validate_input(
        self, input_data: ReadMcpResourceInput, context: Optional[ToolUseContext] = None
    ) -> ValidationResult:
        runtime = await ensure_mcp_runtime()
        server_names = {s.name for s in runtime.servers}
        if input_data.server not in server_names:
            return ValidationResult(
                result=False, message=f"Unknown MCP server '{input_data.server}'."
            )
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
        self, input_data: ReadMcpResourceInput, verbose: bool = False
    ) -> str:
        return f"Read MCP resource {input_data.uri} from {input_data.server}"

    async def call(
        self,
        input_data: ReadMcpResourceInput,
        context: ToolUseContext,
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
                            except Exception:
                                logger.exception(
                                    "[mcp_tools] Failed to decode base64 blob content",
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
            except Exception as exc:  # pragma: no cover - runtime errors
                logger.exception(
                    "Error reading MCP resource",
                    extra={"server": input_data.server, "uri": input_data.uri, "error": str(exc)},
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


def _sanitize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def _create_dynamic_input_model(schema: Optional[Dict[str, Any]]) -> type[BaseModel]:
    raw_schema = schema if isinstance(schema, dict) else {"type": "object"}
    raw_schema = raw_schema or {"type": "object"}

    class DynamicMcpInput(BaseModel):
        model_config = ConfigDict(extra="allow")

        @classmethod
        def model_json_schema(cls, *args: Any, **kwargs: Any) -> Dict[str, Any]:
            return raw_schema

    DynamicMcpInput.__name__ = (
        f"McpInput_{abs(hash(json.dumps(raw_schema, sort_keys=True, default=str))) % 10_000_000}"
    )
    return DynamicMcpInput


def _annotation_flag(tool_info: Any, key: str) -> bool:
    annotations = getattr(tool_info, "annotations", {}) or {}
    if hasattr(annotations, "get"):
        try:
            return bool(annotations.get(key, False))
        except Exception:
            logger.debug("[mcp_tools] Failed to read annotation flag", exc_info=True)
            return False
    return False


def _render_mcp_tool_result_for_assistant(output: McpToolCallOutput) -> str:
    if output.text or output.content:
        return output.text or output.content or ""
    if output.is_error:
        return "MCP tool call failed."
    return f"MCP tool '{output.tool}' returned no content."


class DynamicMcpTool(Tool[BaseModel, McpToolCallOutput]):
    """Runtime wrapper for an MCP tool exposed by a connected server."""

    is_mcp = True

    def __init__(self, server_name: str, tool_info: Any, project_path: Path) -> None:
        self.server_name = server_name
        self.tool_info = tool_info
        self.project_path = project_path
        self._input_model = _create_dynamic_input_model(getattr(tool_info, "input_schema", None))
        self._name = f"mcp__{_sanitize_name(server_name)}__{_sanitize_name(tool_info.name)}"
        self._user_facing = (
            f"{server_name} - {getattr(tool_info, 'description', '') or tool_info.name} (MCP)"
        )

    @property
    def name(self) -> str:
        return self._name

    async def description(self) -> str:
        desc = getattr(self.tool_info, "description", "") or ""
        schema = getattr(self.tool_info, "input_schema", None)
        schema_snippet = json.dumps(schema, indent=2) if schema else ""
        if schema_snippet:
            schema_snippet = (
                schema_snippet if len(schema_snippet) < 800 else schema_snippet[:800] + "..."
            )
            return f"{desc}\n\n[MCP tool]\nServer: {self.server_name}\nTool: {self.tool_info.name}\nInput schema:\n{schema_snippet}"
        return f"{desc}\n\n[MCP tool]\nServer: {self.server_name}\nTool: {self.tool_info.name}"

    @property
    def input_schema(self) -> type[BaseModel]:
        return self._input_model

    async def prompt(self, safe_mode: bool = False) -> str:
        return await self.description()

    def is_read_only(self) -> bool:
        return _annotation_flag(self.tool_info, "readOnlyHint")

    def is_concurrency_safe(self) -> bool:
        return self.is_read_only()

    def is_destructive(self) -> bool:
        return _annotation_flag(self.tool_info, "destructiveHint")

    def is_open_world(self) -> bool:
        return _annotation_flag(self.tool_info, "openWorldHint")

    def defer_loading(self) -> bool:
        """Avoid loading all MCP tools into the initial context."""
        return True

    def needs_permissions(self, input_data: Optional[BaseModel] = None) -> bool:
        return not self.is_read_only()

    def render_result_for_assistant(self, output: McpToolCallOutput) -> str:
        return _render_mcp_tool_result_for_assistant(output)

    def render_tool_use_message(self, input_data: BaseModel, verbose: bool = False) -> str:
        args = input_data.model_dump(exclude_none=True)
        arg_preview = json.dumps(args) if verbose and args else ""
        suffix = f" with args {arg_preview}" if arg_preview else ""
        return f"MCP {self.server_name}:{self.tool_info.name}{suffix}"

    def user_facing_name(self) -> str:
        return self._user_facing

    async def call(
        self,
        input_data: BaseModel,
        context: ToolUseContext,
    ) -> AsyncGenerator[ToolOutput, None]:
        runtime = await ensure_mcp_runtime(self.project_path)
        session = runtime.sessions.get(self.server_name) if runtime else None
        if not session:
            result = McpToolCallOutput(
                server=self.server_name,
                tool=self.tool_info.name,
                content=None,
                text=None,
                content_blocks=None,
                structured_content=None,
                is_error=True,
            )
            yield ToolResult(
                data=result,
                result_for_assistant=f"MCP server '{self.server_name}' is not connected.",
            )
            return

        try:
            args = input_data.model_dump(exclude_none=True)
            call_result = await session.call_tool(
                self.tool_info.name,
                args or {},
            )
            raw_blocks = getattr(call_result, "content", None)
            content_blocks = _normalize_content_blocks(raw_blocks)
            content_text = _render_content_blocks(content_blocks) if content_blocks else None
            structured = (
                call_result.structuredContent if hasattr(call_result, "structuredContent") else None
            )
            assistant_text = content_text
            if structured:
                assistant_text = (assistant_text + "\n" if assistant_text else "") + json.dumps(
                    structured, indent=2
                )
            output = McpToolCallOutput(
                server=self.server_name,
                tool=self.tool_info.name,
                content=assistant_text or None,
                text=content_text,
                content_blocks=content_blocks,
                structured_content=structured,
                is_error=getattr(call_result, "isError", False),
            )
            base_result_text = self.render_result_for_assistant(output)
            warning_text, error_text, token_estimate = _evaluate_mcp_output_size(
                base_result_text, self.server_name, self.tool_info.name
            )

            if error_text:
                limited_output = McpToolCallOutput(
                    server=self.server_name,
                    tool=self.tool_info.name,
                    content=None,
                    text=None,
                    content_blocks=None,
                    structured_content=None,
                    is_error=True,
                    token_estimate=token_estimate,
                    warning=None,
                )
                yield ToolResult(data=limited_output, result_for_assistant=error_text)
                return

            annotated_output = output.model_copy(
                update={"token_estimate": token_estimate, "warning": warning_text}
            )

            final_text = base_result_text or ""
            if not final_text and warning_text:
                final_text = warning_text

            yield ToolResult(
                data=annotated_output,
                result_for_assistant=final_text,
            )
        except Exception as exc:  # pragma: no cover - runtime errors
            output = McpToolCallOutput(
                server=self.server_name,
                tool=self.tool_info.name,
                content=None,
                text=None,
                content_blocks=None,
                structured_content=None,
                is_error=True,
            )
            logger.exception(
                "Error calling MCP tool",
                extra={
                    "server": self.server_name,
                    "tool": self.tool_info.name,
                    "error": str(exc),
                },
            )
            yield ToolResult(
                data=output,
                result_for_assistant=f"Error calling MCP tool '{self.tool_info.name}' on '{self.server_name}': {exc}",
            )


def _build_dynamic_mcp_tools(runtime: Optional[Any]) -> List[DynamicMcpTool]:
    if not runtime or not getattr(runtime, "servers", None):
        return []
    tools: List[DynamicMcpTool] = []
    for server in runtime.servers:
        if getattr(server, "status", "") != "connected":
            continue
        if not getattr(server, "tools", None):
            continue
        for tool in server.tools:
            tools.append(
                DynamicMcpTool(server.name, tool, getattr(runtime, "project_path", Path.cwd()))
            )
    return tools


def load_dynamic_mcp_tools_sync(project_path: Optional[Path] = None) -> List[DynamicMcpTool]:
    """Best-effort synchronous loader for MCP tools."""
    runtime = get_existing_mcp_runtime()
    if runtime and not getattr(runtime, "_closed", False):
        return _build_dynamic_mcp_tools(runtime)

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            return []
    except RuntimeError:
        pass

    async def _load_and_cleanup() -> List[DynamicMcpTool]:
        runtime = await ensure_mcp_runtime(project_path)
        try:
            return _build_dynamic_mcp_tools(runtime)
        finally:
            # Close the runtime inside the same event loop to avoid asyncgen
            # shutdown errors when asyncio.run tears down the loop.
            await shutdown_mcp_runtime()

    try:
        return asyncio.run(_load_and_cleanup())
    except Exception as exc:  # pragma: no cover - SDK/runtime failures
        logger.exception(
            "Failed to initialize MCP runtime for dynamic tools (sync)",
            extra={"error": str(exc)},
        )
        return []


async def load_dynamic_mcp_tools_async(project_path: Optional[Path] = None) -> List[DynamicMcpTool]:
    """Async loader for MCP tools when already in an event loop."""
    try:
        runtime = await ensure_mcp_runtime(project_path)
    except Exception as exc:  # pragma: no cover - SDK/runtime failures
        logger.exception(
            "Failed to initialize MCP runtime for dynamic tools (async)",
            extra={"error": str(exc)},
        )
        return []
    return _build_dynamic_mcp_tools(runtime)


def merge_tools_with_dynamic(base_tools: List[Any], dynamic_tools: List[Any]) -> List[Any]:
    """Merge dynamic MCP tools into the existing tool list and rebuild the Task tool."""
    from ripperdoc.tools.task_tool import TaskTool  # Local import to avoid cycles

    base_without_task = [tool for tool in base_tools if getattr(tool, "name", None) != "Task"]
    existing_names = {getattr(tool, "name", None) for tool in base_without_task}

    for tool in dynamic_tools:
        if getattr(tool, "name", None) in existing_names:
            continue
        base_without_task.append(tool)
        existing_names.add(getattr(tool, "name", None))

    task_tool = TaskTool(lambda: base_without_task)
    return base_without_task + [task_tool]


__all__ = [
    "ListMcpServersTool",
    "ListMcpResourcesTool",
    "ReadMcpResourceTool",
    "DynamicMcpTool",
    "load_dynamic_mcp_tools_async",
    "load_dynamic_mcp_tools_sync",
    "merge_tools_with_dynamic",
]
