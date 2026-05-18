"""Permission prompt preview rendering."""

from __future__ import annotations

import difflib
import html
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from ripperdoc.tools.file_read import detect_file_encoding
from ripperdoc.utils.diff_rendering import build_numbered_diff_layout, format_numbered_diff_text

from .constants import (
    _EDIT_PREVIEW_MAX_BYTES,
    _EDIT_PREVIEW_MAX_DIFF_LINES,
    _EDIT_PREVIEW_MATCH_SNIPPET_MAX,
    _EDIT_PREVIEW_SEPARATOR,
    _PERMISSION_PROMPT_MIN_DIFF_LINES,
    _PERMISSION_PROMPT_RESERVED_LINES,
)


def _format_input_preview(parsed_input: Any, tool_name: Optional[str] = None) -> str:
    """Create a human-friendly preview for prompts.

    For Bash commands, shows full details for security review.
    For other tools, shows a concise preview.
    Returns HTML-formatted text with color tags.
    """
    # For Bash tool, show full command details for security review
    if tool_name == "Bash" and hasattr(parsed_input, "command"):
        command = html.escape(getattr(parsed_input, "command"))
        lines = [f"<label>Command:</label> <value>{command}</value>"]

        # Add other relevant parameters
        if hasattr(parsed_input, "timeout") and parsed_input.timeout:
            lines.append(f"<label>Timeout:</label> <value>{parsed_input.timeout}ms</value>")
        if hasattr(parsed_input, "sandbox"):
            lines.append(f"<label>Sandbox:</label> <value>{parsed_input.sandbox}</value>")
        if hasattr(parsed_input, "run_in_background"):
            lines.append(f"<label>Background:</label> <value>{parsed_input.run_in_background}</value>")
        if hasattr(parsed_input, "shell_executable") and parsed_input.shell_executable:
            lines.append(f"<label>Shell:</label> <value>{html.escape(parsed_input.shell_executable)}</value>")

        return "\n  ".join(lines)

    if tool_name == "Edit" and hasattr(parsed_input, "file_path"):
        edit_preview = _build_edit_permission_preview(parsed_input, tool_name=tool_name)
        if edit_preview:
            return edit_preview

    if tool_name == "Write" and hasattr(parsed_input, "file_path"):
        write_preview = _build_write_permission_preview(parsed_input)
        if write_preview:
            return write_preview

    # For other tools with commands, show concise preview
    if hasattr(parsed_input, "command"):
        return f"<label>command:</label> <value>'{html.escape(getattr(parsed_input, 'command'))}'</value>"
    if hasattr(parsed_input, "file_path"):
        return f"<label>file:</label> <value>'{html.escape(getattr(parsed_input, 'file_path'))}'</value>"
    if hasattr(parsed_input, "path"):
        return f"<label>path:</label> <value>'{html.escape(getattr(parsed_input, 'path'))}'</value>"

    preview = str(parsed_input)
    if len(preview) > 140:
        preview = preview[:137] + "..."
    return f"<value>{html.escape(preview)}</value>"


def _build_edit_permission_preview(parsed_input: Any, *, tool_name: str) -> str:
    """Render a before-apply preview for Edit prompts."""
    file_path_raw = str(getattr(parsed_input, "file_path", "") or "")
    if not file_path_raw:
        return ""

    path = Path(file_path_raw).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()

    lines = [f"<label>file:</label> <value>{html.escape(str(path))}</value>"]

    if not path.exists():
        lines.append(
            "<warning>Preview unavailable: target file does not exist yet.</warning>"
        )
        return "\n  ".join(lines)

    if not path.is_file():
        lines.append("<warning>Preview unavailable: target path is not a file.</warning>")
        return "\n  ".join(lines)

    try:
        file_size = os.path.getsize(path)
    except OSError:
        file_size = None
    if file_size is not None and file_size > _EDIT_PREVIEW_MAX_BYTES:
        lines.append(
            f"<warning>Preview skipped: file is {file_size} bytes (> {_EDIT_PREVIEW_MAX_BYTES} bytes).</warning>"
        )
        return "\n  ".join(lines)

    try:
        detected_encoding, _ = detect_file_encoding(str(path))
        encoding = detected_encoding or "utf-8"
        with open(path, "r", encoding=encoding) as handle:
            original_content = handle.read()
    except (OSError, UnicodeDecodeError, LookupError) as exc:
        lines.append(f"<warning>Preview unavailable: {html.escape(str(exc))}</warning>")
        return "\n  ".join(lines)

    preview_result = _compute_edit_preview(
        original_content=original_content,
        parsed_input=parsed_input,
        tool_name=tool_name,
    )
    if preview_result["error"] is not None:
        lines.append(
            f"<warning>Preview unavailable: {html.escape(preview_result['error'])}</warning>"
        )
        return "\n  ".join(lines)

    diff_lines: List[str] = preview_result["diff_lines"]
    replacements = preview_result["replacements"]
    if not diff_lines:
        lines.append("<warning>No textual diff generated.</warning>")
        return "\n  ".join(lines)

    line_budget = _permission_preview_diff_line_budget()
    lines.append(
        f"<label>preview:</label> <value>{replacements} replacement(s), showing up to "
        f"{line_budget} diff lines</value>"
    )
    lines.append(f"<dim>{_EDIT_PREVIEW_SEPARATOR}</dim>")

    layout = build_numbered_diff_layout(diff_lines)
    clipped = layout.lines[:line_budget]
    for diff_line in clipped:
        rendered = format_numbered_diff_text(
            diff_line,
            old_width=layout.old_width,
            new_width=layout.new_width,
        )
        escaped_rendered = html.escape(rendered)
        if diff_line.kind == "hunk":
            lines.append(f"<diff-hunk>{escaped_rendered}</diff-hunk>")
            continue

        if diff_line.kind == "add":
            lines.append(f"<diff-add>{escaped_rendered}</diff-add>")
            continue
        if diff_line.kind == "del":
            lines.append(f"<diff-del>{escaped_rendered}</diff-del>")
            continue
        lines.append(f"<value>{escaped_rendered}</value>")

    if len(diff_lines) > line_budget:
        hidden = len(diff_lines) - line_budget
        lines.append(f"<dim>... ({hidden} more diff lines)</dim>")
    lines.append(f"<dim>{_EDIT_PREVIEW_SEPARATOR}</dim>")

    return "\n  ".join(lines)


def _build_write_permission_preview(parsed_input: Any) -> str:
    """Render a before-apply preview for Write prompts."""
    file_path_raw = str(getattr(parsed_input, "file_path", "") or "")
    if not file_path_raw:
        return ""

    new_content = str(getattr(parsed_input, "content", "") or "")

    path = Path(file_path_raw).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()

    lines = [f"<label>file:</label> <value>{html.escape(str(path))}</value>"]

    file_exists = path.exists()
    if file_exists and not path.is_file():
        lines.append("<warning>Preview unavailable: target path is not a file.</warning>")
        return "\n  ".join(lines)

    new_content_bytes = len(new_content.encode("utf-8", errors="replace"))
    if new_content_bytes > _EDIT_PREVIEW_MAX_BYTES:
        lines.append(
            f"<warning>Preview skipped: new content is {new_content_bytes} bytes (> {_EDIT_PREVIEW_MAX_BYTES} bytes).</warning>"
        )
        return "\n  ".join(lines)

    original_content = ""
    if file_exists:
        try:
            file_size = os.path.getsize(path)
        except OSError:
            file_size = None
        if file_size is not None and file_size > _EDIT_PREVIEW_MAX_BYTES:
            lines.append(
                f"<warning>Preview skipped: file is {file_size} bytes (> {_EDIT_PREVIEW_MAX_BYTES} bytes).</warning>"
            )
            return "\n  ".join(lines)

        try:
            detected_encoding, _ = detect_file_encoding(str(path))
            encoding = detected_encoding or "utf-8"
            with open(path, "r", encoding=encoding) as handle:
                original_content = handle.read()
        except (OSError, UnicodeDecodeError, LookupError) as exc:
            lines.append(f"<warning>Preview unavailable: {html.escape(str(exc))}</warning>")
            return "\n  ".join(lines)

    diff = list(
        difflib.unified_diff(
            original_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile="before",
            tofile="after",
            lineterm="",
        )
    )
    diff_lines = [line for line in diff[2:]]
    if not diff_lines:
        if not file_exists and not new_content:
            lines.append("<warning>Preview: file will be created empty.</warning>")
        else:
            lines.append("<warning>No textual diff generated.</warning>")
        return "\n  ".join(lines)

    additions = sum(1 for line in diff_lines if line.startswith("+") and not line.startswith("+++"))
    deletions = sum(1 for line in diff_lines if line.startswith("-") and not line.startswith("---"))

    line_budget = _permission_preview_diff_line_budget()
    operation = "create" if not file_exists else "overwrite"
    lines.append(
        f"<label>preview:</label> <value>{operation}, +{additions}/-{deletions} lines, "
        f"showing up to {line_budget} diff lines</value>"
    )
    lines.append(f"<dim>{_EDIT_PREVIEW_SEPARATOR}</dim>")

    layout = build_numbered_diff_layout(diff_lines)
    clipped = layout.lines[:line_budget]
    for diff_line in clipped:
        rendered = format_numbered_diff_text(
            diff_line,
            old_width=layout.old_width,
            new_width=layout.new_width,
        )
        escaped_rendered = html.escape(rendered)
        if diff_line.kind == "hunk":
            lines.append(f"<diff-hunk>{escaped_rendered}</diff-hunk>")
            continue
        if diff_line.kind == "add":
            lines.append(f"<diff-add>{escaped_rendered}</diff-add>")
            continue
        if diff_line.kind == "del":
            lines.append(f"<diff-del>{escaped_rendered}</diff-del>")
            continue
        lines.append(f"<value>{escaped_rendered}</value>")

    if len(diff_lines) > line_budget:
        hidden = len(diff_lines) - line_budget
        lines.append(f"<dim>... ({hidden} more diff lines)</dim>")
    lines.append(f"<dim>{_EDIT_PREVIEW_SEPARATOR}</dim>")

    return "\n  ".join(lines)


def _permission_preview_diff_line_budget() -> int:
    """Compute diff preview line budget based on terminal height."""
    try:
        height = shutil.get_terminal_size(fallback=(80, 24)).lines
    except OSError:
        height = 24
    dynamic_budget = height - _PERMISSION_PROMPT_RESERVED_LINES
    dynamic_budget = max(_PERMISSION_PROMPT_MIN_DIFF_LINES, dynamic_budget)
    return min(_EDIT_PREVIEW_MAX_DIFF_LINES, dynamic_budget)



def _compact_preview_snippet(text: str, *, max_len: int = _EDIT_PREVIEW_MATCH_SNIPPET_MAX) -> str:
    """Short single-line snippet for permission preview messages."""
    single_line = text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n")
    if len(single_line) <= max_len:
        return single_line
    return single_line[: max_len - 3] + "..."


def _compute_edit_preview(
    *,
    original_content: str,
    parsed_input: Any,
    tool_name: str,
) -> Dict[str, Any]:
    """Apply edit inputs in-memory and return diff preview payload."""
    operations = _normalize_edit_operations(parsed_input, tool_name=tool_name)
    if operations is None:
        return {"error": "invalid edit payload", "diff_lines": [], "replacements": 0}

    updated = original_content
    replacements = 0
    for op in operations:
        old = op["old_string"]
        new = op["new_string"]
        replace_all = op["replace_all"]

        if old == "":
            if updated != "":
                return {
                    "error": (
                        "empty old_string is only valid when creating content in an empty file"
                    ),
                    "diff_lines": [],
                    "replacements": 0,
                }
            updated = new
            replacements += 1 if new else 0
            continue

        occurrences = updated.count(old)
        if occurrences == 0:
            old_preview = _compact_preview_snippet(old)
            return {
                "error": f"old_string not found (snippet: {old_preview!r})",
                "diff_lines": [],
                "replacements": 0,
            }
        if not replace_all and occurrences > 1:
            return {
                "error": (
                    f"string appears {occurrences} times; provide a unique match or set replace_all=true"
                ),
                "diff_lines": [],
                "replacements": 0,
            }

        if replace_all:
            updated = updated.replace(old, new)
            replacements += occurrences
        else:
            updated = updated.replace(old, new, 1)
            replacements += 1

    diff = list(
        difflib.unified_diff(
            original_content.splitlines(keepends=True),
            updated.splitlines(keepends=True),
            fromfile="before",
            tofile="after",
            lineterm="",
        )
    )
    return {
        "error": None,
        "diff_lines": [line for line in diff[2:]],
        "replacements": replacements,
    }


def _normalize_edit_operations(parsed_input: Any, *, tool_name: str) -> Optional[List[Dict[str, Any]]]:
    """Normalize Edit payload into a common in-memory operation list."""
    if tool_name == "Edit":
        return [
            {
                "old_string": str(getattr(parsed_input, "old_string", "")),
                "new_string": str(getattr(parsed_input, "new_string", "")),
                "replace_all": bool(getattr(parsed_input, "replace_all", False)),
            }
        ]

    return None
