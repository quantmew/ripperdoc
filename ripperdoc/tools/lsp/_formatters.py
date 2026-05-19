"""Formatters for LSP tool results."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ripperdoc.utils.lsp import uri_to_path

MAX_RESULTS = 50

SYMBOL_KIND_NAMES = {
    1: "File", 2: "Module", 3: "Namespace", 4: "Package",
    5: "Class", 6: "Method", 7: "Property", 8: "Field",
    9: "Constructor", 10: "Enum", 11: "Interface", 12: "Function",
    13: "Variable", 14: "Constant", 15: "String", 16: "Number",
    17: "Boolean", 18: "Array", 19: "Object", 20: "Key",
    21: "Null", 22: "EnumMember", 23: "Struct", 24: "Event",
    25: "Operator", 26: "TypeParameter",
}

_DIAGNOSTIC_SEVERITY = {
    1: "Error", 2: "Warning", 3: "Information", 4: "Hint",
}


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
    range_info = (
        location.get("range") or location.get("targetRange") or location.get("targetSelectionRange")
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


def format_locations(label: str, locations: List[Dict[str, Any]]) -> Tuple[str, int, int]:
    """Format location results (definitions, references, implementations)."""
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


def format_hover(result: Any) -> Tuple[str, int, int]:
    """Format hover result."""
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


def format_document_symbols(result: Any) -> Tuple[str, int, int]:
    """Format document symbol results."""
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


def format_workspace_symbols(result: Any) -> Tuple[str, int, int]:
    """Format workspace symbol results."""
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


def format_diagnostics(diagnostics: Any) -> Tuple[str, int, int]:
    """Format LSP diagnostic items."""
    if not diagnostics:
        return "No diagnostics found.", 0, 0

    items: List[Dict[str, Any]] = []
    if isinstance(diagnostics, list):
        items = [d for d in diagnostics if isinstance(d, dict)]
    if not items:
        return "No diagnostics found.", 0, 0

    lines: List[str] = []
    for diag in items[:MAX_RESULTS]:
        severity = _DIAGNOSTIC_SEVERITY.get(diag.get("severity", 0), "Unknown")
        range_info = diag.get("range", {})
        start = range_info.get("start", {})
        line_num = int(start.get("line", 0)) + 1
        char_num = int(start.get("character", 0)) + 1
        message = diag.get("message", "No message")
        source = diag.get("source")
        source_text = f" [{source}]" if source else ""
        lines.append(f"  {severity}{source_text} @ line {line_num}:{char_num}: {message}")

    omitted = len(items) - len(lines)
    if omitted > 0:
        lines.append(f"... {omitted} more diagnostic(s) not shown")

    summary = f"{len(items)} diagnostic(s) found."
    return f"{summary}\n" + "\n".join(lines), len(items), 1


def format_code_actions(actions: Any) -> Tuple[str, int, int]:
    """Format LSP code action items."""
    if not actions:
        return "No code actions available.", 0, 0

    items: List[Dict[str, Any]] = []
    if isinstance(actions, list):
        items = [a for a in actions if isinstance(a, dict)]
    if not items:
        return "No code actions available.", 0, 0

    lines: List[str] = []
    for action in items[:MAX_RESULTS]:
        title = action.get("title", "Untitled action")
        kind = action.get("kind", "")
        is_preferred = action.get("isPreferred", False)
        kind_text = f" ({kind})" if kind else ""
        preferred_text = " [preferred]" if is_preferred else ""
        lines.append(f"  - {title}{kind_text}{preferred_text}")

    omitted = len(items) - len(lines)
    if omitted > 0:
        lines.append(f"... {omitted} more action(s) not shown")

    summary = f"{len(items)} code action(s) available."
    return f"{summary}\n" + "\n".join(lines), len(items), 1
