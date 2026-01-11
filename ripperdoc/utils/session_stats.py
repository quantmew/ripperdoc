"""Session statistics collection and analysis."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

from ripperdoc.utils.log import get_logger
from ripperdoc.utils.session_history import list_session_summaries

logger = get_logger()


@dataclass
class SessionStats:
    """Aggregated statistics across all sessions."""

    # Basic counts
    total_sessions: int = 0
    total_messages: int = 0
    total_cost_usd: float = 0.0

    # Token statistics
    total_tokens: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_creation_tokens: int = 0

    # Model statistics (model -> count)
    model_usage: Dict[str, int] = field(default_factory=dict)
    favorite_model: str = ""

    # Time statistics
    longest_session_duration: timedelta = field(default_factory=lambda: timedelta(0))
    earliest_session: datetime | None = None
    latest_session: datetime | None = None

    # Streak statistics
    current_streak: int = 0
    longest_streak: int = 0
    active_days: int = 0
    total_days: int = 0

    # Activity patterns (hour -> count)
    hourly_activity: Dict[int, int] = field(default_factory=dict)

    # Daily activity (date_str -> count)
    daily_activity: Dict[str, int] = field(default_factory=dict)

    # Weekday activity (0=Monday -> 6=Sunday, count)
    weekday_activity: Dict[int, int] = field(default_factory=dict)

    # Peak hour (hour with most activity)
    peak_hour: int = 0


def _calculate_streaks(active_dates: List[datetime]) -> Tuple[int, int]:
    """Calculate current and longest streak from sorted active dates."""
    if not active_dates:
        return 0, 0

    # Sort dates
    sorted_dates = sorted(set(d.date() for d in active_dates))

    # Calculate longest streak
    longest = 1
    current = 1
    for i in range(1, len(sorted_dates)):
        if (sorted_dates[i] - sorted_dates[i - 1]).days == 1:
            current += 1
            longest = max(longest, current)
        else:
            current = 1

    # Calculate current streak (from today backwards)
    today = datetime.now().date()
    current_streak = 0

    # Check if today or yesterday has activity
    if sorted_dates[-1] == today:
        current_streak = 1
        check_date = today - timedelta(days=1)
    elif sorted_dates[-1] == today - timedelta(days=1):
        current_streak = 1
        check_date = sorted_dates[-1] - timedelta(days=1)
    else:
        return 0, longest

    # Count backwards
    i = len(sorted_dates) - 2
    while i >= 0:
        if sorted_dates[i] == check_date:
            current_streak += 1
            check_date -= timedelta(days=1)
            i -= 1
        else:
            break

    return current_streak, longest


def collect_session_stats(project_path: Path, days: int = 32) -> SessionStats:
    """Collect statistics from session history.

    Args:
        project_path: Project root directory
        days: Number of days to look back (default 32)
    """
    stats = SessionStats(
        hourly_activity=defaultdict(int),
        daily_activity=defaultdict(int),
        weekday_activity=defaultdict(int),
        model_usage=defaultdict(int),
    )

    summaries = list_session_summaries(project_path)
    if not summaries:
        return stats

    # Filter by date range (use timezone-aware cutoff if needed)
    from datetime import timezone
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    # Ensure comparison works with both naive and aware datetimes
    recent_summaries = []
    for s in summaries:
        # Make updated_at timezone-aware if it's naive
        updated_at = s.updated_at
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)
        if updated_at >= cutoff:
            recent_summaries.append(s)

    if not recent_summaries:
        return stats

    # Basic counts
    stats.total_sessions = len(recent_summaries)
    stats.total_messages = sum(s.message_count for s in recent_summaries)

    # Time statistics
    stats.earliest_session = min(s.created_at for s in recent_summaries)
    stats.latest_session = max(s.updated_at for s in recent_summaries)
    stats.total_days = (stats.latest_session - stats.earliest_session).days + 1

    # Calculate longest session and activity patterns in single pass
    active_dates: List[datetime] = []
    date_set: set[str] = set()

    for summary in recent_summaries:
        # Longest session
        duration = summary.updated_at - summary.created_at
        if duration > stats.longest_session_duration:
            stats.longest_session_duration = duration

        # Track dates
        date_str = summary.updated_at.date().isoformat()
        if date_str not in date_set:
            date_set.add(date_str)
            active_dates.append(summary.updated_at)

        # Hourly activity
        stats.hourly_activity[summary.updated_at.hour] += 1

        # Daily activity (for heatmap)
        stats.daily_activity[date_str] += 1

        # Weekday activity
        stats.weekday_activity[summary.updated_at.weekday()] += 1

    # Active days
    stats.active_days = len(date_set)

    # Streaks
    stats.current_streak, stats.longest_streak = _calculate_streaks(active_dates)

    # Peak hour
    if stats.hourly_activity:
        stats.peak_hour = max(stats.hourly_activity.items(), key=lambda x: x[1])[0]

    # Load detailed session data for token and model statistics
    import json

    for summary in recent_summaries:
        session_file = summary.path
        if not session_file.exists():
            continue

        try:
            with session_file.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    # Quick string check before full JSON parse
                    if '"type":"assistant"' not in line and '"type": "assistant"' not in line:
                        continue
                    try:
                        entry = json.loads(line)
                        payload = entry.get("payload", {})

                        # Only process assistant messages
                        if payload.get("type") != "assistant":
                            continue

                        # Extract model and token information
                        model = payload.get("model")
                        if model:
                            stats.model_usage[model] += 1

                        # Extract token counts
                        input_tokens = payload.get("input_tokens", 0)
                        output_tokens = payload.get("output_tokens", 0)
                        cache_read = payload.get("cache_read_tokens", 0)
                        cache_creation = payload.get("cache_creation_tokens", 0)

                        stats.total_input_tokens += input_tokens
                        stats.total_output_tokens += output_tokens
                        stats.total_cache_read_tokens += cache_read
                        stats.total_cache_creation_tokens += cache_creation

                        # Extract cost
                        cost = payload.get("cost_usd", 0.0)
                        stats.total_cost_usd += cost

                    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                        continue
        except (OSError, IOError):
            continue

    # Calculate total tokens
    stats.total_tokens = (
        stats.total_input_tokens
        + stats.total_output_tokens
        + stats.total_cache_read_tokens
        + stats.total_cache_creation_tokens
    )

    # Determine favorite model
    if stats.model_usage:
        stats.favorite_model = max(stats.model_usage.items(), key=lambda x: x[1])[0]

    return stats


def format_duration(td: timedelta) -> str:
    """Format timedelta as human-readable string."""
    total_seconds = int(td.total_seconds())
    if total_seconds < 60:
        return f"{total_seconds}s"

    minutes = total_seconds // 60
    if minutes < 60:
        seconds = total_seconds % 60
        return f"{minutes}m {seconds}s"

    hours = minutes // 60
    remaining_mins = minutes % 60
    if hours < 24:
        return f"{hours}h {remaining_mins}m"

    days = hours // 24
    remaining_hours = hours % 24
    return f"{days}d {remaining_hours}h {remaining_mins}m"


def format_large_number(num: int) -> str:
    """Format large numbers with k/m/b suffix.

    Examples:
        1234 -> "1.2k"
        1234567 -> "1.2m"
        1234567890 -> "1.2b"
    """
    if num < 1000:
        return str(num)
    elif num < 1_000_000:
        return f"{num / 1000:.1f}k"
    elif num < 1_000_000_000:
        return f"{num / 1_000_000:.1f}m"
    else:
        return f"{num / 1_000_000_000:.1f}b"


__all__ = [
    "SessionStats",
    "collect_session_stats",
    "format_duration",
    "format_large_number",
]
