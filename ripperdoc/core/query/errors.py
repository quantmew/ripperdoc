"""Shared error and notice helpers for query execution."""

from typing import List

from ripperdoc.utils.file_watch import ChangedFileNotice


def _format_changed_file_notice(notices: List[ChangedFileNotice]) -> str:
    """Render a system notice about files that changed on disk."""
    lines: List[str] = [
        "System notice: Files you previously read have changed on disk.",
        "Please re-read the affected files before making further edits.",
        "",
    ]
    for notice in notices:
        lines.append(f"- {notice.file_path}")
        summary = (notice.summary or "").rstrip()
        if summary:
            indented = "\n".join(f"    {line}" for line in summary.splitlines())
            lines.append(indented)
    return "\n".join(lines)
