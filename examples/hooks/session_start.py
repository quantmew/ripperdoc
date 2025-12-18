#!/usr/bin/env python3
"""Handle session start events.

This hook runs when a Ripperdoc session starts, resumes, or clears.
Useful for initialization, logging, or environment setup.

Configuration example (.ripperdoc/hooks.json):
{
  "hooks": {
    "SessionStart": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python $RIPPERDOC_PROJECT_DIR/.ripperdoc/hooks/session_start.py"
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


def main():
    # Read hook input from stdin
    try:
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON input: {e}", file=sys.stderr)
        sys.exit(1)

    trigger = input_data.get("trigger", "unknown")  # startup, resume, clear, compact
    session_id = input_data.get("session_id", "unknown")
    project_dir = input_data.get("project_dir", os.getcwd())

    # Log session start
    log_dir = Path(project_dir) / ".ripperdoc" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "sessions.log"

    timestamp = datetime.now().isoformat()
    log_entry = f"[{timestamp}] SESSION_START trigger={trigger} session={session_id}\n"

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_entry)

    # Provide context based on trigger
    if trigger == "startup":
        context = f"Session started at {datetime.now().strftime('%H:%M:%S')}"
    elif trigger == "resume":
        context = "Session resumed from previous state"
    elif trigger == "clear":
        context = "Session cleared - fresh start"
    elif trigger == "compact":
        context = "Session compacted - context summarized"
    else:
        context = f"Session trigger: {trigger}"

    # Output additional context
    print(context)
    sys.exit(0)


if __name__ == "__main__":
    main()
