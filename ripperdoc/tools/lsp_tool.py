"""LSP tool for code intelligence queries."""

from __future__ import annotations

from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Literal

from pydantic import BaseModel, ConfigDict, Field

from ripperdoc.core.tool import (
    Tool,
    ToolOutput,
    ToolResult,
    ToolUseContext,
    ToolUseExample,
    ValidationResult,
)
from ripperdoc.utils.log import get_logger
from ripperdoc.utils.path_ignore import check_path_for_tool
from ripperdoc.utils.lsp import (
    LspLaunchError,
    LspProtocolError,
    LspRequestError,
    ensure_lsp_manager,
    uri_to_path,
)


logger = get_logger()

LSP_USAGE = (
    "Interact with Language Server Protocol (LSP) servers to get code intelligence features.\n\n"
    "Supported operations:\n"
    "- goToDefinition: Find where a symbol is defined\n"
    "- findReferences: Find all references to a symbol\n"
    "- hover: Get hover information (documentation, type info) for a symbol\n"
    "- documentSymbol: Get all symbols (functions, classes, variables) in a document\n"
    "- workspaceSymbol: Search for symbols across the entire workspace\n"
    "- goToImplementation: Find implementations of an interface or abstract method\n\n"
    "All operations require:\n"
    "- filePath: The file to operate on\n"
    "- line: The line number (1-based, as shown in editors)\n"
    "- character: The character offset (1-based, as shown in editors)\n\n"
    "Note: LSP servers must be configured for the file type. "
    "If no server is available, an error will be returned."
)

MAX_RESULTS = 50

SYMBOL_KIND_NAMES = {
    1: "File",
    2: "Module",
    3: "Namespace",
    4: "Package",
    5: "Class",
    6: "Method",
    7: "Property",
    8: "Field",
    9: "Constructor",
    10: "Enum",
    11: "Interface",
    12: "Function",
    13: "Variable",
    14: "Constant",
    15: "String",
    16: "Number",
    17: "Boolean",
    18: "Array",
    19: "Object",
    20: "Key",
    21: "Null",
    22: "EnumMember",
    23: "Struct",
    24: "Event",
    25: "Operator",
    26: "TypeParameter",
}


def _resolve_file_path(raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def _display_path(file_path: Path, verbose: bool) -> str:
    if verbose:
        return str(file_path)
    try:
        rel = file_path.resolve().relative_to(Path.cwd().resolve())
    except (ValueError, OSError):
        return str(file_path)
    rel_str = str(rel)
    return rel_str if rel_str != "." else str(file_path)


def _read_text(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8", errors="replace")


def _normalize_position(
    lines: List[str], line: int, character: int
) -> Tuple[int, int, str]:
    if not lines:
        return 0, 0, ""
    line_index = max(0, min(line - 1, len(lines) - 1))
    line_text = lines[line_index]
    char_index = max(0, min(character - 1, len(line_text)))
    return line_index, char_index, line_text


def _extract_symbol_at_position(line_text: str, char_index: int) -> Optional[str]:
    if not line_text:
        return None
    if char_index >= len(line_text):
        char_index = len(line_text) - 1
    if char_index < 0:
        return None

    if not line_text[char_index].isalnum() and line_text[char_index] != "_":
        if char_index > 0 and (line_text[char_index - 1].isalnum() or line_text[char_index - 1] == "_"):
            char_index -= 1
        else:
            return None

    start = char_index
    while start > 0 and (line_text[start - 1].isalnum() or line_text[start - 1] == "_"):
        start -= 1
    end = char_index
    while end + 1 < len(line_text) and (line_text[end + 1].isalnum() or line_text[end + 1] == "_"):
        end += 1
    symbol = line_text[start : end + 1].strip()
    return symbol or None


def _symbol_kind_name(kind: Any) -> str:
    try:
        kind_value = int(kind)
    except (TypeError, ValueError):
        return "Unknown"
    return SYMBOL_KIND_NAMES.get(kind_value, "Unknown")


def _location_to_path_line_char(location: Optional[Dict[str, Any]]) -> Tuple[str, int, int]:
    if not location:
        return "<unknown>", 0, 0
    uri = location.get("uri") or location.get("targetUri")
    range_info = location.get("range") or location.get("targetRange") or location.get(
        "targetSelectionRange"
    )
    path = "<unknown>"
    if isinstance(uri, str):
        file_path = uri_to_path(uri)
        if file_path:
            path = str(file_path)
    line = 0
    character = 0
    if isinstance(range_info, dict):
        start = range_info.get("start")
        if isinstance(start, dict):
            line = int(start.get("line", 0)) + 1
            character = int(start.get("character", 0)) + 1
    return path, line, character


def _format_locations(
    label: str, locations: List[Dict[str, Any]]
) -> Tuple[str, int, int]:
    if not locations:
        return f"No {label} found.", 0, 0

    unique_files = set()
    lines: List[str] = []
    for loc in locations[:MAX_RESULTS]:
        path, line, char = _location_to_path_line_char(loc)
        unique_files.add(path)
        lines.append(f"{path}:{line}:{char}")

    omitted = len(locations) - len(lines)
    if omitted > 0:
        lines.append(f"... {omitted} more result(s) not shown")

    summary = f"{len(locations)} {label} found in {len(unique_files)} file(s)."
    return f"{summary}\n\n" + "\n".join(lines), len(locations), len(unique_files)


def _format_hover(result: Any) -> Tuple[str, int, int]:
    if not result:
        return "No hover information found.", 0, 0

    if isinstance(result, str):
        text = result.strip()
        return (text, 1, 1) if text else ("No hover information found.", 0, 0)

    contents = result.get("contents") if isinstance(result, dict) else None
    if not contents:
        return "No hover information found.", 0, 0

    if isinstance(contents, dict):
        value = contents.get("value")
        text = value if isinstance(value, str) else str(contents)
    elif isinstance(contents, list):
        parts = []
        for item in contents:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                value = item.get("value")
                parts.append(value if isinstance(value, str) else str(item))
            else:
                parts.append(str(item))
        text = "\n".join([part for part in parts if part])
    else:
        text = str(contents)

    text = text.strip()
    if not text:
        return "No hover information found.", 0, 0
    return text, 1, 1


def _flatten_document_symbols(
    symbols: List[Dict[str, Any]],
    depth: int = 0,
    lines: Optional[List[str]] = None,
) -> Tuple[List[str], int]:
    if lines is None:
        lines = []
    count = 0

    for symbol in symbols:
        count += 1
        name = symbol.get("name", "<unknown>")
        detail = symbol.get("detail")
        kind = _symbol_kind_name(symbol.get("kind"))
        selection = symbol.get("selectionRange") or symbol.get("range") or {}
        start = selection.get("start") if isinstance(selection, dict) else {}
        line = int(start.get("line", 0)) + 1 if isinstance(start, dict) else 0
        char = int(start.get("character", 0)) + 1 if isinstance(start, dict) else 0
        prefix = "  " * depth
        detail_text = f" - {detail}" if detail else ""
        lines.append(f"{prefix}{name}{detail_text} ({kind}) @ {line}:{char}")

        children = symbol.get("children")
        if isinstance(children, list) and children:
            child_lines, child_count = _flatten_document_symbols(children, depth + 1, lines)
            count += child_count
            lines = child_lines

    return lines, count


def _format_document_symbols(result: Any) -> Tuple[str, int, int]:
    if not result:
        return "No document symbols found.", 0, 0

    symbols: List[Dict[str, Any]] = []
    if isinstance(result, list):
        symbols = [s for s in result if isinstance(s, dict)]
    if not symbols:
        return "No document symbols found.", 0, 0

    lines, count = _flatten_document_symbols(symbols)
    if len(lines) > MAX_RESULTS:
        omitted = len(lines) - MAX_RESULTS
        lines = lines[:MAX_RESULTS] + [f"... {omitted} more result(s) not shown"]

    summary = f"{count} symbol(s) found in document."
    return f"{summary}\n\n" + "\n".join(lines), count, 1


def _format_workspace_symbols(result: Any) -> Tuple[str, int, int]:
    if not result:
        return "No workspace symbols found.", 0, 0

    symbols: List[Dict[str, Any]] = []
    if isinstance(result, list):
        symbols = [s for s in result if isinstance(s, dict)]
    if not symbols:
        return "No workspace symbols found.", 0, 0

    unique_files = set()
    lines: List[str] = []
    for symbol in symbols[:MAX_RESULTS]:
        name = symbol.get("name", "<unknown>")
        kind = _symbol_kind_name(symbol.get("kind"))
        container = symbol.get("containerName")
        location = None
        if isinstance(symbol.get("location"), dict):
            location = symbol.get("location")
        else:
            locations = symbol.get("locations")
            if isinstance(locations, list) and locations:
                first = locations[0]
                if isinstance(first, dict):
                    location = first
        path, line, char = _location_to_path_line_char(location)
        unique_files.add(path)
        container_text = f" ({container})" if container else ""
        lines.append(f"{name}{container_text} ({kind}) {path}:{line}:{char}")

    omitted = len(symbols) - len(lines)
    if omitted > 0:
        lines.append(f"... {omitted} more result(s) not shown")

    summary = f"{len(symbols)} symbol(s) found in {len(unique_files)} file(s)."
    return f"{summary}\n\n" + "\n".join(lines), len(symbols), len(unique_files)


class LspToolInput(BaseModel):
    """Input schema for LspTool."""

    model_config = ConfigDict(populate_by_name=True)

    operation: Literal[
        "goToDefinition",
        "findReferences",
        "hover",
        "documentSymbol",
        "workspaceSymbol",
        "goToImplementation",
    ] = Field(description="The LSP operation to perform.")
    file_path: str = Field(
        validation_alias="filePath",
        serialization_alias="filePath",
        description="The absolute or relative path to the file",
    )
    line: int = Field(ge=1, description="The line number (1-based, as shown in editors)")
    character: int = Field(ge=1, description="The character offset (1-based, as shown in editors)")


class LspToolOutput(BaseModel):
    """Output from LspTool."""

    model_config = ConfigDict(populate_by_name=True)

    operation: str
    result: str
    file_path: str = Field(validation_alias="filePath", serialization_alias="filePath")
    is_error: bool = Field(
        default=False,
        validation_alias="is_error",
        serialization_alias="is_error",
        description="Whether the LSP operation failed.",
    )
    result_count: Optional[int] = Field(
        default=None,
        validation_alias="resultCount",
        serialization_alias="resultCount",
    )
    file_count: Optional[int] = Field(
        default=None,
        validation_alias="fileCount",
        serialization_alias="fileCount",
    )


class LspTool(Tool[LspToolInput, LspToolOutput]):
    """Tool for LSP-backed code intelligence."""

    @property
    def name(self) -> str:
        return "LSP"

    async def description(self) -> str:
        return LSP_USAGE

    @property
    def input_schema(self) -> type[LspToolInput]:
        return LspToolInput

    def input_examples(self) -> List[ToolUseExample]:
        return [
            ToolUseExample(
                description="Jump to a symbol definition",
                example={
                    "operation": "goToDefinition",
                    "filePath": "src/main.py",
                    "line": 12,
                    "character": 8,
                },
            ),
            ToolUseExample(
                description="Find references to a function",
                example={
                    "operation": "findReferences",
                    "filePath": "src/main.py",
                    "line": 12,
                    "character": 8,
                },
            ),
            ToolUseExample(
                description="List document symbols",
                example={
                    "operation": "documentSymbol",
                    "filePath": "src/main.py",
                    "line": 1,
                    "character": 1,
                },
            ),
        ]

    async def prompt(self, _yolo_mode: bool = False) -> str:
        return LSP_USAGE

    def is_read_only(self) -> bool:
        return True

    def is_concurrency_safe(self) -> bool:
        return True

    def needs_permissions(self, _input_data: Optional[LspToolInput] = None) -> bool:
        return False

    async def validate_input(
        self, input_data: LspToolInput, _context: Optional[ToolUseContext] = None
    ) -> ValidationResult:
        try:
            resolved_path = _resolve_file_path(input_data.file_path)
        except (OSError, RuntimeError, ValueError) as exc:
            return ValidationResult(result=False, message=str(exc))

        if not resolved_path.exists():
            return ValidationResult(result=False, message=f"File not found: {input_data.file_path}")
        if not resolved_path.is_file():
            return ValidationResult(
                result=False, message=f"Path is not a file: {input_data.file_path}"
            )

        should_proceed, warning_msg = check_path_for_tool(
            resolved_path, tool_name="LSP", warn_only=True
        )
        if warning_msg:
            logger.info("[lsp_tool] %s", warning_msg)
        if not should_proceed:
            return ValidationResult(result=False, message=warning_msg or "Access denied.")

        return ValidationResult(result=True)

    def render_result_for_assistant(self, output: LspToolOutput) -> str:
        return output.result

    def render_tool_use_message(self, input_data: LspToolInput, verbose: bool = False) -> str:
        try:
            file_path = _resolve_file_path(input_data.file_path)
        except (OSError, RuntimeError, ValueError):
            file_path = Path(input_data.file_path)

        symbol = None
        if input_data.operation in {
            "goToDefinition",
            "findReferences",
            "hover",
            "goToImplementation",
            "workspaceSymbol",
        }:
            try:
                text = _read_text(file_path)
                lines = text.splitlines()
                _line_index, char_index, line_text = _normalize_position(
                    lines, input_data.line, input_data.character
                )
                symbol = _extract_symbol_at_position(line_text, char_index)
            except (OSError, RuntimeError, UnicodeDecodeError):
                symbol = None

        parts = [f'operation: "{input_data.operation}"']
        if symbol:
            parts.append(f'symbol: "{symbol}"')
        parts.append(f'file: "{_display_path(file_path, verbose)}"')
        if not symbol:
            parts.append(f"position: {input_data.line}:{input_data.character}")
        return ", ".join(parts)

    async def call(
        self, input_data: LspToolInput, _context: ToolUseContext
    ) -> AsyncGenerator[ToolOutput, None]:
        try:
            file_path = _resolve_file_path(input_data.file_path)
            text = _read_text(file_path)
            lines = text.splitlines()
            line_index, char_index, line_text = _normalize_position(
                lines, input_data.line, input_data.character
            )
            symbol = _extract_symbol_at_position(line_text, char_index)
        except (OSError, RuntimeError, UnicodeDecodeError, ValueError) as exc:
            output = LspToolOutput(
                operation=input_data.operation,
                result=f"Error reading file for LSP: {exc}",
                file_path=input_data.file_path,
                is_error=True,
            )
            yield ToolResult(data=output, result_for_assistant=output.result)
            return

        operation = input_data.operation
        method: Optional[str] = None
        params: Optional[Dict[str, Any]] = None

        position = {"line": line_index, "character": char_index}
        text_document = {"uri": file_path.resolve().as_uri()}

        if operation == "goToDefinition":
            method = "textDocument/definition"
            params = {"textDocument": text_document, "position": position}
        elif operation == "findReferences":
            method = "textDocument/references"
            params = {
                "textDocument": text_document,
                "position": position,
                "context": {"includeDeclaration": True},
            }
        elif operation == "hover":
            method = "textDocument/hover"
            params = {"textDocument": text_document, "position": position}
        elif operation == "documentSymbol":
            method = "textDocument/documentSymbol"
            params = {"textDocument": text_document}
        elif operation == "workspaceSymbol":
            if not symbol:
                output = LspToolOutput(
                    operation=operation,
                    result="No symbol found at the given position to search in workspace.",
                    file_path=input_data.file_path,
                )
                yield ToolResult(data=output, result_for_assistant=output.result)
                return
            method = "workspace/symbol"
            params = {"query": symbol}
        elif operation == "goToImplementation":
            method = "textDocument/implementation"
            params = {"textDocument": text_document, "position": position}
        else:
            output = LspToolOutput(
                operation=operation,
                result=f"Unknown LSP operation: {operation}",
                file_path=input_data.file_path,
                is_error=True,
            )
            yield ToolResult(data=output, result_for_assistant=output.result)
            return

        manager = await ensure_lsp_manager(Path.cwd())
        server_info = await manager.server_for_path(file_path)
        if not server_info:
            output = LspToolOutput(
                operation=operation,
                result=(
                    f"No LSP server available for file type: {file_path.suffix or 'unknown'}. "
                    "Configure servers in ~/.ripperdoc/lsp.json, ~/.lsp.json, "
                    ".ripperdoc/lsp.json, or .lsp.json."
                ),
                file_path=input_data.file_path,
                is_error=True,
            )
            yield ToolResult(data=output, result_for_assistant=output.result)
            return

        server, _config, language_id = server_info

        try:
            await server.ensure_initialized()
            if method.startswith("textDocument/"):
                await server.ensure_document_open(file_path, text, language_id)
            result = await server.request(method, params)
        except (LspLaunchError, LspProtocolError, LspRequestError) as exc:
            output = LspToolOutput(
                operation=operation,
                result=f"Error performing {operation}: {exc}",
                file_path=input_data.file_path,
                is_error=True,
            )
            yield ToolResult(data=output, result_for_assistant=output.result)
            return

        formatted: str
        result_count: Optional[int] = None
        file_count: Optional[int] = None

        if operation == "goToDefinition":
            if isinstance(result, dict):
                result = [result]
            formatted, result_count, file_count = _format_locations(
                "definition(s)", result or []
            )
        elif operation == "findReferences":
            formatted, result_count, file_count = _format_locations(
                "reference(s)", result or []
            )
        elif operation == "hover":
            formatted, result_count, file_count = _format_hover(result or {})
        elif operation == "documentSymbol":
            formatted, result_count, file_count = _format_document_symbols(result)
        elif operation == "workspaceSymbol":
            formatted, result_count, file_count = _format_workspace_symbols(result)
        elif operation == "goToImplementation":
            if isinstance(result, dict):
                result = [result]
            formatted, result_count, file_count = _format_locations(
                "implementation(s)", result or []
            )
        else:
            formatted = str(result)

        output = LspToolOutput(
            operation=operation,
            result=formatted,
            file_path=input_data.file_path,
            result_count=result_count,
            file_count=file_count,
        )
        yield ToolResult(data=output, result_for_assistant=output.result)
