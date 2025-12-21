"""Activity heatmap visualization for session statistics."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict

from rich.console import Console
from rich.text import Text


def _get_intensity_char(count: int, max_count: int) -> str:
    """Get unicode block character based on activity intensity.

    Returns character and color code for 8 intensity levels.
    """
    if count == 0:
        return "·"
    if max_count == 0:
        return "█"

    # Calculate intensity (0-1)
    intensity = count / max_count

    # Use unicode block characters for different intensities (8 levels)
    if intensity <= 0.125:
        return "░"
    elif intensity <= 0.25:
        return "░"
    elif intensity <= 0.375:
        return "▒"
    elif intensity <= 0.5:
        return "▒"
    elif intensity <= 0.625:
        return "▓"
    elif intensity <= 0.75:
        return "▓"
    elif intensity <= 0.875:
        return "█"
    else:
        return "█"


def _get_intensity_color(count: int, max_count: int) -> str:
    """Get color style based on activity intensity (8 levels)."""
    if count == 0:
        return "dim white"
    if max_count == 0:
        return "bold color(46)"

    # Calculate intensity (0-1)
    intensity = count / max_count

    # 8-level green gradient from very light to very dark
    if intensity <= 0.125:
        return "color(22)"  # Very light green
    elif intensity <= 0.25:
        return "color(28)"  # Light green
    elif intensity <= 0.375:
        return "color(34)"  # Light-medium green
    elif intensity <= 0.5:
        return "color(40)"  # Medium green
    elif intensity <= 0.625:
        return "color(46)"  # Medium-dark green
    elif intensity <= 0.75:
        return "color(82)"  # Dark green
    elif intensity <= 0.875:
        return "bold color(46)"  # Bright green with bold
    else:
        return "bold color(82)"  # Very dark green with bold


def _get_week_grid(
    daily_activity: Dict[str, int], weeks_count: int = 52
) -> tuple[list[list[tuple[str, int]]], int]:
    """Build a grid of weeks for heatmap display.

    Args:
        daily_activity: Dictionary mapping date strings to activity counts
        weeks_count: Number of weeks to display (default 52 for full year)

    Returns:
        Tuple of (grid, max_count) where grid is a list of weeks,
        each week is a list of (date_str, count) tuples.
    """
    # Calculate total days to display
    total_days = weeks_count * 7

    # Start from today and go back
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=total_days - 1)

    # Find max count for intensity calculation
    max_count = max(daily_activity.values()) if daily_activity else 1

    # Build grid by weeks (Sunday to Saturday)
    weeks: list[list[tuple[str, int]]] = []
    current_week: list[tuple[str, int]] = []

    current_date = start_date
    # Pad the start to align with week start (Sunday = 6 in Python's weekday, we want 0)
    weekday = (current_date.weekday() + 1) % 7  # Convert to Sunday=0
    if weekday > 0:
        # Add empty days at the start
        for _ in range(weekday):
            current_week.append(("", 0))

    while current_date <= end_date:
        date_str = current_date.isoformat()
        count = daily_activity.get(date_str, 0)
        current_week.append((date_str, count))

        # If we've completed a week (7 days), start a new week
        if len(current_week) == 7:
            weeks.append(current_week)
            current_week = []

        current_date += timedelta(days=1)

    # Add remaining days
    if current_week:
        # Pad to complete the week
        while len(current_week) < 7:
            current_week.append(("", 0))
        weeks.append(current_week)

    return weeks, max_count


def render_heatmap(
    console: Console, daily_activity: Dict[str, int], weeks_count: int = 52
) -> None:
    """Render activity heatmap to console.

    Args:
        console: Rich console for output
        daily_activity: Dictionary mapping date strings to activity counts
        weeks_count: Number of weeks to display (default 52 for full year)
    """
    # Alignment constant: width of weekday labels column
    WEEKDAY_LABEL_WIDTH = 8

    weeks, max_count = _get_week_grid(daily_activity, weeks_count)
    if not weeks:
        console.print("[dim]No activity data[/dim]")
        return

    # Build month labels row
    # Each week column is exactly 1 character wide in the heatmap
    # Month labels are 3 characters wide (e.g., "Dec", "Jan", "Feb")
    month_positions = []  # List of (week_idx, month_name) tuples
    current_month = None

    for week_idx, week in enumerate(weeks):
        # Check first non-empty day in week for month
        for date_str, _ in week:
            if date_str:
                date = datetime.fromisoformat(date_str).date()
                month_str = date.strftime("%b")

                # Record position when month changes
                if month_str != current_month:
                    month_positions.append((week_idx, month_str))
                    current_month = month_str
                break

    # Build month label string with precise alignment
    month_chars = [" "] * len(weeks)

    for week_idx, month_name in month_positions:
        # Check if we have space for the full month name (3 chars)
        can_place = True
        for i in range(3):
            if week_idx + i < len(month_chars) and month_chars[week_idx + i] != " ":
                can_place = False
                break

        if can_place and week_idx < len(month_chars):
            # Place the month name starting at this week position
            for i, char in enumerate(month_name):
                if week_idx + i < len(month_chars):
                    month_chars[week_idx + i] = char

    # Print month labels with proper alignment
    month_row = " " * WEEKDAY_LABEL_WIDTH + "".join(month_chars)
    console.print(month_row)

    # Weekday labels - show Mon, Wed, Fri
    weekday_labels = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    show_weekdays = [1, 3, 5]  # Show Mon, Wed, Fri

    # Render each row (weekday)
    for weekday_idx in range(7):
        row = Text()

        # Add weekday label - exactly WEEKDAY_LABEL_WIDTH characters
        if weekday_idx in show_weekdays:
            label = weekday_labels[weekday_idx]
            # Right-align the label within the width, then add a space
            row.append(f"{label:>3} ", style="dim")  # "Mon " = 4 chars
            row.append(" " * (WEEKDAY_LABEL_WIDTH - 4))  # Fill remaining space
        else:
            row.append(" " * WEEKDAY_LABEL_WIDTH)

        # Add activity cells for this weekday across all weeks
        # Each week column is exactly 1 character wide
        for week in weeks:
            if weekday_idx < len(week):
                date_str, count = week[weekday_idx]
                if date_str:
                    char = _get_intensity_char(count, max_count)
                    color = _get_intensity_color(count, max_count)
                    row.append(char, style=color)
                else:
                    row.append("·", style="dim white")
            else:
                row.append(" ")

        console.print(row)

    # Legend with 8-level green gradient
    console.print()
    legend = Text(" " * WEEKDAY_LABEL_WIDTH + "Less ", style="dim")
    # Show representative levels from the 8-level gradient
    legend.append("░", style="color(22)")  # Level 1: Very light green
    legend.append(" ", style="dim")
    legend.append("░", style="color(28)")  # Level 2: Light green
    legend.append(" ", style="dim")
    legend.append("▒", style="color(34)")  # Level 3: Light-medium green
    legend.append(" ", style="dim")
    legend.append("▒", style="color(40)")  # Level 4: Medium green
    legend.append(" ", style="dim")
    legend.append("▓", style="color(46)")  # Level 5: Medium-dark green
    legend.append(" ", style="dim")
    legend.append("▓", style="color(82)")  # Level 6: Dark green
    legend.append(" ", style="dim")
    legend.append("█", style="bold color(46)")  # Level 7: Bright green
    legend.append(" ", style="dim")
    legend.append("█", style="bold color(82)")  # Level 8: Very dark green
    legend.append(" More", style="dim")
    console.print(legend)


__all__ = ["render_heatmap"]
