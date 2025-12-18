#!/usr/bin/env python3
"""Log all Bash commands to a file.

This hook logs Bash tool invocations for auditing purposes.
Useful for tracking what commands were run during a session.

Configuration example (.ripperdoc/hooks.json):
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "python $RIPPERDOC_PROJECT_DIR/.ripperdoc/hooks/log_bash.py"
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

    tool_input = input_data.get("tool_input", {})
    command = tool_input.get("command", "")
    description = tool_input.get("description", "No description")
    session_id = input_data.get("session_id", "unknown")
    project_dir = input_data.get("project_dir", os.getcwd())

    # Determine log file location
    log_dir = Path(project_dir) / ".ripperdoc" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "bash_commands.log"

    # Format log entry
    timestamp = datetime.now().isoformat()
    log_entry = f"[{timestamp}] session={session_id}\n"
    log_entry += f"  command: {command}\n"
    log_entry += f"  description: {description}\n\n"

    # Append to log file
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_entry)

    # Allow the command to proceed (exit 0)
    sys.exit(0)


if __name__ == "__main__":
    main()
