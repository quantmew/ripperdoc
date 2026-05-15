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
from ripperdoc.utils.filesystem.path_ignore import check_path_for_tool
from ripperdoc.utils.lsp import (
    LspLaunchError,
    LspProtocolError,
    LspRequestError,
    ensure_lsp_manager,
    uri_to_path,
)
from ripperdoc.tools.lsp._formatters import (
    format_code_actions,
    format_diagnostics,
    format_document_symbols,
    format_hover,
    format_locations,
    format_workspace_symbols,
    _symbol_kind_name,
)
from ripperdoc.tools.lsp._prompt import LSP_PROMPT as LSP_USAGE

logger = get_logger()

MAX_RESULTS = 50


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


def _normalize_position(lines: List[str], line: int, character: int) -> Tuple[int, int, str]:
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
        if char_index > 0 and (
            line_text[char_index - 1].isalnum() or line_text[char_index - 1] == "_"
        ):
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
        "codeAction",
        "diagnostics",
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
            resolved_path, tool_name="LSP", warn_only=True, warn_on_gitignore=False
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
            "codeAction",
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
        elif operation == "codeAction":
            method = "textDocument/codeAction"
            line_end = {"line": line_index, "character": len(line_text) if line_text else 0}
            params = {
                "textDocument": text_document,
                "range": {"start": position, "end": line_end},
                "context": {"diagnostics": []},
            }
        elif operation == "diagnostics":
            method = "textDocument/diagnostic"
            params = {"textDocument": text_document}
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
            formatted, result_count, file_count = format_locations("definition(s)", result or [])
        elif operation == "findReferences":
            formatted, result_count, file_count = format_locations("reference(s)", result or [])
        elif operation == "hover":
            formatted, result_count, file_count = format_hover(result or {})
        elif operation == "documentSymbol":
            formatted, result_count, file_count = format_document_symbols(result)
        elif operation == "workspaceSymbol":
            formatted, result_count, file_count = format_workspace_symbols(result)
        elif operation == "goToImplementation":
            if isinstance(result, dict):
                result = [result]
            formatted, result_count, file_count = format_locations(
                "implementation(s)", result or []
            )
        elif operation == "codeAction":
            formatted, result_count, file_count = format_code_actions(result)
        elif operation == "diagnostics":
            if isinstance(result, dict):
                diag_items = result.get("items", [])
            elif isinstance(result, list):
                diag_items = result
            else:
                diag_items = []
            formatted, result_count, file_count = format_diagnostics(diag_items)
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
