"""Tests for unified diff layout/format helpers."""

from ripperdoc.utils.diff_rendering import build_numbered_diff_layout, format_numbered_diff_text


def test_numbered_diff_layout_uses_max_old_and_new_widths() -> None:
    diff_lines = [
        "@@ -120,2 +997,3 @@",
        " keep_a",
        "-removed",
        "+added",
        "+tail",
    ]

    layout = build_numbered_diff_layout(diff_lines)

    assert layout.old_width == 3
    assert layout.new_width == 3
    rendered = [
        format_numbered_diff_text(line, old_width=layout.old_width, new_width=layout.new_width)
        for line in layout.lines
        if line.kind != "hunk"
    ]
    context_line, deleted_line, added_line, _ = rendered
    assert context_line.index("keep_a") == deleted_line.index("removed")
    assert context_line.index("keep_a") == added_line.index("added")
    assert context_line.startswith("120   997")
