#!/usr/bin/env python3
"""Handle session end events.

This hook runs when a Ripperdoc session ends.
Useful for cleanup, logging, or reporting.

Configuration example (.ripperdoc/hooks.json):
{
  "hooks": {
    "SessionEnd": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python $RIPPERDOC_PROJECT_DIR/.ripperdoc/hooks/session_end.py"
          }
        ]
      }
    ]
  }
}
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path


def format_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    if seconds is None:
        return "unknown"

    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}h {minutes}m"


def main() -> None:
    # Read hook input from stdin
    try:
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON input: {e}", file=sys.stderr)
        sys.exit(1)

    trigger = input_data.get("trigger", "unknown")  # clear, logout, prompt_input_exit, other
    session_id = input_data.get("session_id", "unknown")
    project_dir = input_data.get("project_dir", os.getcwd())
    duration_seconds = input_data.get("duration_seconds")
    message_count = input_data.get("message_count", 0)

    # Log session end
    log_dir = Path(project_dir) / ".ripperdoc" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "sessions.log"

    timestamp = datetime.now().isoformat()
    duration_str = format_duration(duration_seconds) if duration_seconds else "N/A"

    log_entry = (
        f"[{timestamp}] SESSION_END trigger={trigger} session={session_id} "
        f"duration={duration_str} messages={message_count}\n"
    )

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_entry)

    # Print summary (goes to stderr so it's visible but doesn't affect hook output)
    print(
        f"Session ended: {duration_str}, {message_count} messages",
        file=sys.stderr
    )

    sys.exit(0)


if __name__ == "__main__":
    main()
