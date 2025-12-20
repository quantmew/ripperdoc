"""Dynamic MCP tool wrapper for runtime MCP server tools.

This module provides the DynamicMcpTool class that wraps MCP server tools
at runtime, along with helper functions for loading and merging these tools.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from pydantic import BaseModel, ConfigDict

from ripperdoc.core.tool import (
    Tool,
    ToolUseContext,
    ToolResult,
    ToolOutput,
)
from ripperdoc.utils.log import get_logger
from ripperdoc.utils.mcp import (
    ensure_mcp_runtime,
    get_existing_mcp_runtime,
)

logger = get_logger()


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


def _sanitize_name(name: str) -> str:
    """Sanitize a name for use in tool identifiers."""
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def _create_dynamic_input_model(schema: Optional[Dict[str, Any]]) -> type[BaseModel]:
    """Create a dynamic Pydantic model from a JSON schema."""
    raw_schema = schema if isinstance(schema, dict) else {"type": "object"}
    raw_schema = raw_schema or {"type": "object"}

    class DynamicMcpInput(BaseModel):
        model_config = ConfigDict(extra="allow")

        @classmethod
        def model_json_schema(cls, *_args: Any, **_kwargs: Any) -> Dict[str, Any]:
            return raw_schema

    DynamicMcpInput.__name__ = (
        f"McpInput_{abs(hash(json.dumps(raw_schema, sort_keys=True, default=str))) % 10_000_000}"
    )
    return DynamicMcpInput


def _annotation_flag(tool_info: Any, key: str) -> bool:
    """Extract a boolean flag from tool annotations."""
    annotations = getattr(tool_info, "annotations", {}) or {}
    if hasattr(annotations, "get"):
        try:
            return bool(annotations.get(key, False))
        except (AttributeError, TypeError, KeyError) as exc:
            logger.debug(
                "[mcp_tools] Failed to read annotation flag: %s: %s",
                type(exc).__name__,
                exc,
            )
            return False
    return False


def _render_mcp_tool_result_for_assistant(output: McpToolCallOutput) -> str:
    """Render MCP tool output for the assistant."""
    if output.text or output.content:
        return output.text or output.content or ""
    if output.is_error:
        return "MCP tool call failed."
    return f"MCP tool '{output.tool}' returned no content."


def _content_block_to_text(block: Any) -> str:
    """Convert a content block to text representation."""
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
    """Render multiple content blocks to text."""
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
    """Normalize a list of content blocks."""
    if not blocks:
        return None
    return [_normalize_content_block(block) for block in blocks]


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

    async def prompt(self, _yolo_mode: bool = False) -> str:
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

    def needs_permissions(self, _input_data: Optional[BaseModel] = None) -> bool:
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
        _context: ToolUseContext,
    ) -> AsyncGenerator[ToolOutput, None]:
        from ripperdoc.tools.mcp_tools import _evaluate_mcp_output_size

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
        except (
            OSError,
            RuntimeError,
            ConnectionError,
            ValueError,
            KeyError,
            TypeError,
        ) as exc:  # pragma: no cover - runtime errors
            output = McpToolCallOutput(
                server=self.server_name,
                tool=self.tool_info.name,
                content=None,
                text=None,
                content_blocks=None,
                structured_content=None,
                is_error=True,
            )
            logger.warning(
                "Error calling MCP tool: %s: %s",
                type(exc).__name__,
                exc,
                extra={
                    "server": self.server_name,
                    "tool": self.tool_info.name,
                },
            )
            yield ToolResult(
                data=output,
                result_for_assistant=f"Error calling MCP tool '{self.tool_info.name}' on '{self.server_name}': {exc}",
            )


def _build_dynamic_mcp_tools(runtime: Optional[Any]) -> List[DynamicMcpTool]:
    """Build DynamicMcpTool instances from a runtime's connected servers."""
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

    async def _load_and_keep() -> List[DynamicMcpTool]:
        runtime = await ensure_mcp_runtime(project_path)
        return _build_dynamic_mcp_tools(runtime)

    try:
        return asyncio.run(_load_and_keep())
    except (
        OSError,
        RuntimeError,
        ConnectionError,
        ValueError,
    ) as exc:  # pragma: no cover - SDK/runtime failures
        logger.warning(
            "Failed to initialize MCP runtime for dynamic tools (sync): %s: %s",
            type(exc).__name__,
            exc,
        )
        return []


async def load_dynamic_mcp_tools_async(project_path: Optional[Path] = None) -> List[DynamicMcpTool]:
    """Async loader for MCP tools when already in an event loop."""
    try:
        runtime = await ensure_mcp_runtime(project_path)
    except (
        OSError,
        RuntimeError,
        ConnectionError,
        ValueError,
    ) as exc:  # pragma: no cover - SDK/runtime failures
        logger.warning(
            "Failed to initialize MCP runtime for dynamic tools (async): %s: %s",
            type(exc).__name__,
            exc,
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
    "DynamicMcpTool",
    "McpToolCallOutput",
    "load_dynamic_mcp_tools_async",
    "load_dynamic_mcp_tools_sync",
    "merge_tools_with_dynamic",
]
