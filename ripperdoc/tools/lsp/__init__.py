"""LSP tool for code intelligence queries."""

from ripperdoc.tools.lsp._formatters import (
    MAX_RESULTS,
    SYMBOL_KIND_NAMES,
    _DIAGNOSTIC_SEVERITY,
    _flatten_document_symbols,
    _location_to_path_line_char,
    _symbol_kind_name,
    format_code_actions,
    format_diagnostics,
    format_document_symbols,
    format_hover,
    format_locations,
    format_workspace_symbols,
)
from ripperdoc.tools.lsp._tool import (
    LspTool,
    LspToolInput,
    LspToolOutput,
    _display_path,
    _extract_symbol_at_position,
    _normalize_position,
    _read_text,
    _resolve_file_path,
)

_format_code_actions = format_code_actions
_format_diagnostics = format_diagnostics
_format_document_symbols = format_document_symbols
_format_hover = format_hover
_format_locations = format_locations
_format_workspace_symbols = format_workspace_symbols

__all__ = [
    "LspTool",
    "LspToolInput",
    "LspToolOutput",
    "MAX_RESULTS",
    "SYMBOL_KIND_NAMES",
    "_DIAGNOSTIC_SEVERITY",
    "_display_path",
    "_extract_symbol_at_position",
    "_flatten_document_symbols",
    "_format_code_actions",
    "_format_diagnostics",
    "_format_document_symbols",
    "_format_hover",
    "_format_locations",
    "_format_workspace_symbols",
    "_location_to_path_line_char",
    "_normalize_position",
    "_read_text",
    "_resolve_file_path",
    "_symbol_kind_name",
    "format_code_actions",
    "format_diagnostics",
    "format_document_symbols",
    "format_hover",
    "format_locations",
    "format_workspace_symbols",
]
