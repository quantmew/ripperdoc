#!/usr/bin/env python3
"""Demo hook showing all available hook input fields.

This hook simply logs the full input it receives, useful for
understanding what data is available in each hook event.

Configuration example (.ripperdoc/hooks.json):
{
  "hooks": {
    "PreToolUse": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python $RIPPERDOC_PROJECT_DIR/.ripperdoc/hooks/demo.py"
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

    # Get project directory for logging
    project_dir = input_data.get("project_dir", os.getcwd())

    # Log the full input
    log_dir = Path(project_dir) / ".ripperdoc" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "hook_debug.log"

    timestamp = datetime.now().isoformat()
    event_name = input_data.get("event_name", "unknown")

    log_entry = f"\n{'='*60}\n"
    log_entry += f"[{timestamp}] Event: {event_name}\n"
    log_entry += f"{'='*60}\n"
    log_entry += json.dumps(input_data, indent=2, default=str)
    log_entry += "\n"

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_entry)

    # Print info to stderr (doesn't affect hook decision)
    print(f"Hook demo: received {event_name} event", file=sys.stderr)

    # Allow the operation to proceed
    sys.exit(0)


if __name__ == "__main__":
    main()
