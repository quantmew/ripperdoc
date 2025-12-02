"""Shared helpers for rendering context/token usage in CLI output."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from ripperdoc.utils.message_compaction import ContextBreakdown


def format_tokens(tokens: int) -> str:
    """Render token counts in a compact human-readable form."""
    if tokens >= 1_000_000:
        value = tokens / 1_000_000
        suffix = "M"
    elif tokens >= 1_000:
        value = tokens / 1_000
        suffix = "k"
    else:
        return str(tokens)
    text = f"{value:.1f}"
    if text.endswith(".0"):
        text = text[:-2]
    return f"{text}{suffix}"


def styled_symbol(symbol: str, color: str) -> str:
    return f"[{color}]{symbol}[/]"


def visible_length(text: str) -> int:
    """Best-effort length calculation that ignores Rich markup."""
    if not text:
        return 0
    return len(re.sub(r"\[/?[^\]]+\]", "", text))


def make_icon_bar(
    used_tokens: int,
    reserved_tokens: int,
    max_tokens: int,
    *,
    segments: int = 10,
    glyph_used: str = "⛁",
    glyph_partial: str = "⛀",
    glyph_reserved: str = "⛝",
    glyph_empty: str = "⛶",
    color_used: Optional[str] = None,
    color_reserved: Optional[str] = None,
    color_empty: Optional[str] = "grey50",
) -> str:
    """Render a segmented bar using contextual glyphs."""
    if max_tokens <= 0:
        max_tokens = 1

    used_tokens = max(0, min(used_tokens, max_tokens))
    reserved_tokens = max(0, min(reserved_tokens, max_tokens - used_tokens))

    def style(symbol: str, color: Optional[str]) -> str:
        return f"[{color}]{symbol}[/]" if color else symbol

    bar: List[str] = []
    used_slots_float = (used_tokens / max_tokens) * segments
    used_full = int(used_slots_float)
    used_partial = used_slots_float - used_full
    bar.extend([style(glyph_used, color_used)] * min(used_full, segments))
    if used_partial > 0.05 and len(bar) < segments:
        bar.append(style(glyph_partial, color_used))

    reserved_slots_float = (reserved_tokens / max_tokens) * segments
    reserved_full = int(reserved_slots_float)
    reserved_partial = reserved_slots_float - reserved_full
    remaining = segments - len(bar)
    if remaining > 0:
        bar.extend([style(glyph_reserved, color_reserved)] * min(reserved_full, remaining))
    remaining = segments - len(bar)
    if reserved_partial > 0.05 and remaining > 0:
        bar.append(style(glyph_reserved, color_reserved))

    remaining = segments - len(bar)
    if remaining > 0:
        bar.extend([style(glyph_empty, color_empty)] * remaining)

    return " ".join(bar[:segments])


def make_segment_grid(
    breakdown: ContextBreakdown,
    *,
    per_row: int = 10,
) -> List[str]:
    """Build a 10x10 proportional grid (left→right for main sections, reserved pinned to bottom-right)."""
    total_slots = max(1, per_row * per_row)  # default 100 slots

    categories: List[Dict[str, Any]] = [
        {
            "label": "System prompt",
            "glyph": "⛁",
            "color": "grey58",
            "tokens": breakdown.system_prompt_tokens,
        },
        {
            "label": "MCP instructions",
            "glyph": "⛁",
            "color": "cyan",
            "tokens": getattr(breakdown, "mcp_tokens", 0),
        },
        {
            "label": "System tools",
            "glyph": "⛁",
            "color": "green3",
            "tokens": breakdown.tool_schema_tokens,
        },
        {
            "label": "Memory files",
            "glyph": "⛁",
            "color": "dark_orange3",
            "tokens": breakdown.memory_tokens,
        },
        {
            "label": "Messages",
            "glyph": "⛁",
            "color": "medium_purple",
            "tokens": breakdown.message_tokens,
        },
        {
            "label": "Free space",
            "glyph": "⛶",
            "color": "grey46",
            "tokens": max(breakdown.free_tokens, 0),
        },
        {
            "label": "Autocompact buffer",
            "glyph": "⛝",
            "color": "yellow3",
            "tokens": max(breakdown.reserved_tokens, 0),
        },
    ]

    max_tokens = max(breakdown.max_context_tokens, 1)

    # First compute slot counts for all categories so totals match the grid size.
    allocations: List[int] = []
    remainders: List[tuple[float, int]] = []
    allocated = 0
    for idx, category in enumerate(categories):
        token_value_int = int(category["tokens"])
        token_value = max(0, token_value_int)
        raw_slots = (token_value / max_tokens) * total_slots
        base = int(raw_slots)
        if token_value > 0 and base == 0:
            base = 1  # ensure tiny but nonzero sections are visible
        allocations.append(base)
        allocated += base
        remainders.append((raw_slots - base, idx))

    min_allowed = [1 if int(cat["tokens"]) > 0 else 0 for cat in categories]

    while allocated > total_slots:
        for _, idx in sorted(remainders, key=lambda x: x[0]):
            if allocated <= total_slots:
                break
            if allocations[idx] > min_allowed[idx]:
                allocations[idx] -= 1
                allocated -= 1
        else:
            break

    while allocated < total_slots:
        for _, idx in sorted(remainders, key=lambda x: x[0], reverse=True):
            if allocated >= total_slots:
                break
            allocations[idx] += 1
            allocated += 1
        else:
            break

    # Place all non-reserved sections from the top-left; place reserved from the bottom-right.
    reserved_count = allocations[-1]
    forward_categories = categories[:-1]
    forward_allocations = allocations[:-1]

    icons: List[Optional[str]] = [None] * total_slots
    cursor = 0
    for category, count in zip(forward_categories, forward_allocations):
        for _ in range(count):
            if cursor >= total_slots:
                break
            icons[cursor] = styled_symbol(str(category["glyph"]), str(category["color"]))
            cursor += 1

    end_cursor = total_slots - 1
    reserved_symbol = styled_symbol(str(categories[-1]["glyph"]), str(categories[-1]["color"]))
    for _ in range(reserved_count):
        if end_cursor < 0:
            break
        icons[end_cursor] = reserved_symbol
        end_cursor -= 1

    # Fill any gaps (should not normally happen) with free-space icons.
    free_symbol = styled_symbol("⛶", "grey46")
    for idx, value in enumerate(icons):
        if value is None:
            icons[idx] = free_symbol

    rows: List[str] = []
    for start in range(0, total_slots, per_row):
        row_icons = [icon for icon in icons[start : start + per_row] if icon is not None]
        rows.append(" ".join(row_icons))
    return rows


def context_usage_lines(
    breakdown: ContextBreakdown, model_label: str, auto_compact_enabled: bool
) -> List[str]:
    """Build a stylized context usage block using a fixed 10x10 grid."""
    grid_lines: List[str] = []
    grid_lines.append("  ⎿   Context Usage")

    grid_rows = make_segment_grid(breakdown, per_row=10)
    if grid_rows:
        header_row = grid_rows[0]
        grid_lines.append(
            f"     {header_row}   {model_label} · "
            f"{format_tokens(breakdown.effective_tokens)}/"
            f"{format_tokens(breakdown.max_context_tokens)} tokens "
            f"({breakdown.percent_used:.1f}%)"
        )
        for row in grid_rows[1:]:
            grid_lines.append(f"     {row}")

    # Textual stats (without additional mini bars).
    stats: List[Tuple[str, Optional[int], Optional[float]]] = [
        (
            f"{styled_symbol('⛁', 'grey58')} System prompt",
            breakdown.system_prompt_tokens,
            breakdown.percent_of_limit(breakdown.system_prompt_tokens),
        ),
        (
            f"{styled_symbol('⛁', 'cyan')} MCP instructions",
            getattr(breakdown, "mcp_tokens", 0),
            breakdown.percent_of_limit(getattr(breakdown, "mcp_tokens", 0)),
        ),
        (
            f"{styled_symbol('⛁', 'green3')} System tools",
            breakdown.tool_schema_tokens,
            breakdown.percent_of_limit(breakdown.tool_schema_tokens),
        ),
        (
            f"{styled_symbol('⛁', 'dark_orange3')} Memory files",
            breakdown.memory_tokens,
            breakdown.percent_of_limit(breakdown.memory_tokens),
        ),
        (
            f"{styled_symbol('⛁', 'medium_purple')} Messages",
            breakdown.message_tokens,
            breakdown.percent_of_limit(breakdown.message_tokens),
        ),
        (
            f"{styled_symbol('⛶', 'grey46')} Free space",
            breakdown.free_tokens,
            breakdown.percent_of_limit(breakdown.free_tokens),
        ),
    ]

    reserved_label = f"{styled_symbol('⛝', 'yellow3')} Autocompact buffer"
    if auto_compact_enabled and breakdown.reserved_tokens:
        stats.append(
            (
                reserved_label,
                breakdown.reserved_tokens,
                breakdown.percent_of_limit(breakdown.reserved_tokens),
            )
        )
    else:
        stats.append((reserved_label, None, None))

    stats_lines: List[str] = []
    for label, tokens, percent in stats:
        if tokens is None:
            stats_lines.append(f"{label}: disabled")
        else:
            stats_lines.append(f"{label}: {format_tokens(tokens)} tokens ({percent:.1f}%)")

    total_rows = max(len(grid_lines), len(stats_lines))
    padded_grid = [""] * (total_rows - len(grid_lines)) + grid_lines
    padded_stats = [""] * (total_rows - len(stats_lines)) + stats_lines

    combined: List[str] = []
    for left, right in zip(padded_grid, padded_stats):
        # left_pad = " " * max(0, grid_width - visible_length(left))
        left_pad = ""
        if right:
            combined.append(f"{left}{left_pad}   {right}")
        else:
            combined.append(f"{left}")

    return combined
