"""Statistics command to show session usage patterns."""

from typing import Any

from rich.panel import Panel
from rich.table import Table

from ripperdoc.utils.session_heatmap import render_heatmap
from ripperdoc.utils.session_stats import (
    collect_session_stats,
    format_duration,
    format_large_number,
)

from .base import SlashCommand


def _format_number(n: int) -> str:
    """Format number with thousands separator."""
    return f"{n:,}"


def _truncate_model_name(model: str, max_len: int = 15) -> str:
    """Truncate model name with ellipsis if too long."""
    if len(model) <= max_len:
        return model
    return model[: max_len - 1] + "…"


def _get_token_comparison(total_tokens: int) -> str:
    """Get a fun comparison for total tokens based on famous books.

    Reference data from famous books' approximate token counts.
    """
    books = [
        ("The Old Man and the Sea", 35000),
        ("Animal Farm", 39000),
        ("The Great Gatsby", 62000),
        ("Brave New World", 83000),
        ("Harry Potter and the Philosopher's Stone", 103000),
        ("The Hobbit", 123000),
        ("1984", 123000),
        ("To Kill a Mockingbird", 130000),
        ("Pride and Prejudice", 156000),
        ("Anna Karenina", 468000),
        ("Don Quixote", 520000),
        ("The Lord of the Rings", 576000),
        ("War and Peace", 730000),
    ]

    if total_tokens == 0:
        return "Start chatting to build your stats!"

    # Find the best matching book
    for i, (name, tokens) in enumerate(books):
        if total_tokens < tokens:
            if i == 0:
                # Less than the smallest book
                percentage = (total_tokens / tokens) * 100
                return f"You've used {percentage:.0f}% of the tokens in {name}"
            else:
                # Between two books
                prev_name, prev_tokens = books[i - 1]
                if total_tokens >= prev_tokens * 1.5:
                    # Closer to current book
                    multiplier = total_tokens / tokens
                    return f"You've used ~{multiplier:.1f}x the tokens in {name}"
                else:
                    # Closer to previous book
                    multiplier = total_tokens / prev_tokens
                    return f"You've used ~{multiplier:.1f}x the tokens in {prev_name}"

    # More than the largest book
    largest_name, largest_tokens = books[-1]
    multiplier = total_tokens / largest_tokens
    return f"You've used ~{multiplier:.1f}x the tokens in {largest_name}!"


def _get_duration_comparison(duration_minutes: int) -> str:
    """Get a fun comparison for session duration based on common activities.

    Reference data for activity durations in minutes.
    """
    activities = [
        ("a TED talk", 18),
        ("an episode of The Office", 22),
        ("a half marathon (average time)", 120),
        ("the movie Inception", 148),
        ("a transatlantic flight", 420),
    ]

    if duration_minutes == 0:
        return ""

    # Find the best matching activity
    for i, (name, minutes) in enumerate(activities):
        if duration_minutes < minutes:
            if i == 0:
                # Less than the shortest activity
                percentage = (duration_minutes / minutes) * 100
                return f"Longest session: {percentage:.0f}% of {name}"
            else:
                # Between two activities
                prev_name, prev_minutes = activities[i - 1]
                if duration_minutes >= prev_minutes * 1.5:
                    # Closer to current activity
                    multiplier = duration_minutes / minutes
                    return f"Longest session: ~{multiplier:.1f}x {name}"
                else:
                    # Closer to previous activity
                    multiplier = duration_minutes / prev_minutes
                    return f"Longest session: ~{multiplier:.1f}x {prev_name}"

    # More than the longest activity
    longest_name, longest_minutes = activities[-1]
    multiplier = duration_minutes / longest_minutes
    return f"Longest session: ~{multiplier:.1f}x {longest_name}!"


def _handle(ui: Any, _args: str) -> bool:
    """Handle the stats command."""
    # Get project path from UI
    project_path = getattr(ui, "project_path", None)
    if project_path is None:
        ui.console.print("[yellow]No project context available[/yellow]")
        return True

    # Collect statistics
    console = ui.console
    days = 365  # Look back full year for complete heatmap
    weeks_count = 52  # Display 52 weeks (1 year)
    stats = collect_session_stats(project_path, days=days)

    if stats.total_sessions == 0:
        console.print("[yellow]No session data found[/yellow]")
        return True

    # Create overview panel
    console.print()
    console.print("[bold]Activity Heatmap[/bold]")
    console.print()

    # Render heatmap with full year (52 weeks)
    render_heatmap(console, stats.daily_activity, weeks_count=weeks_count)
    console.print()

    # Create two-column layout for statistics
    # Left column: Model and tokens
    # Right column: Sessions and streaks
    table = Table.grid(padding=(0, 4))
    table.add_column(style="cyan", justify="left")
    table.add_column(justify="left")
    table.add_column(style="cyan", justify="left")
    table.add_column(justify="left")

    # Row 1: Favorite model | Sessions
    favorite_model_display = ""
    if stats.favorite_model:
        favorite_model_display = _truncate_model_name(stats.favorite_model)
    table.add_row(
        "Favorite model:",
        favorite_model_display or "N/A",
        "Sessions:",
        f"{_format_number(stats.total_sessions)}",
    )

    # Row 2: Total tokens | Longest session
    total_tokens_display = format_large_number(stats.total_tokens) if stats.total_tokens > 0 else "0"
    longest_session_display = ""
    if stats.longest_session_duration.total_seconds() > 0:
        longest_session_display = format_duration(stats.longest_session_duration)
    table.add_row(
        "Total tokens:",
        total_tokens_display,
        "Longest session:",
        longest_session_display or "N/A",
    )

    # Row 3: empty | Current streak
    table.add_row(
        "",
        "",
        "Current streak:",
        f"{stats.current_streak} days",
    )

    # Row 4: empty | Longest streak
    table.add_row(
        "",
        "",
        "Longest streak:",
        f"{stats.longest_streak} days",
    )

    # Row 5: empty | Active days
    table.add_row(
        "",
        "",
        "Active days:",
        f"{stats.active_days}/{stats.total_days}",
    )

    # Row 6: empty | Peak hour
    peak_hour_str = f"{stats.peak_hour:02d}:00-{stats.peak_hour + 1:02d}:00"
    table.add_row(
        "",
        "",
        "Peak hour:",
        peak_hour_str,
    )

    # Create the main panel with statistics
    # Match heatmap width: WEEKDAY_LABEL_WIDTH (8) + weeks_count (52) = 60
    heatmap_width = 8 + weeks_count + 16
    console.print(
        Panel(table, title="Statistics", border_style="blue", width=heatmap_width)
    )
    console.print()

    # Fun comparisons
    # Token comparison
    token_comparison = _get_token_comparison(stats.total_tokens)
    if token_comparison:
        console.print(f"[dim]{token_comparison}[/dim]")

    # Duration comparison
    duration_minutes = int(stats.longest_session_duration.total_seconds() / 60)
    duration_comparison = _get_duration_comparison(duration_minutes)
    if duration_comparison:
        console.print(f"[dim]{duration_comparison}[/dim]")

    console.print(f"[dim]Stats from the last {days} days[/dim]")
    console.print()

    return True


command = SlashCommand(
    name="stats",
    description="Show session statistics and activity patterns",
    handler=_handle,
)

__all__ = ["command"]
