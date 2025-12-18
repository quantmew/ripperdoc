#!/usr/bin/env python3
"""Validate and restrict Bash commands.

This hook checks Bash commands before execution and blocks
potentially dangerous operations.

Configuration example (.ripperdoc/hooks.json):
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "python $RIPPERDOC_PROJECT_DIR/.ripperdoc/hooks/check_bash.py"
          }
        ]
      }
    ]
  }
}
"""

import json
import re
import sys

# Commands that are always blocked
BLOCKED_COMMANDS = [
    # System destruction
    r"\brm\s+-rf\s+/(?!\w)",  # rm -rf / (but allow rm -rf /path/to/something)
    r"\brm\s+-rf\s+~",  # rm -rf ~
    r"\bmkfs\b",
    r"\bdd\s+.*of=/dev/",
    r"\b:(){ :\|:& };:",  # Fork bomb

    # Network attacks
    r"\bnmap\b.*-sS",  # SYN scan
    r"\bhping3?\b",

    # Privilege escalation attempts
    r"\bchmod\s+[0-7]*777\s+/",  # chmod 777 on root paths
    r"\bsudo\s+su\b",

    # Credential theft
    r"\bcat\s+.*/(passwd|shadow|sudoers)",
    r"\b/etc/shadow\b",

    # Cleanup protection
    r"\bgit\s+push\s+.*--force.*main\b",
    r"\bgit\s+push\s+.*--force.*master\b",
    r"\bgit\s+reset\s+--hard\s+HEAD~",
]

# Commands that require user confirmation
ASK_COMMANDS = [
    (r"\bgit\s+push\b", "Push to remote repository?"),
    (r"\bgit\s+reset\s+--hard\b", "Hard reset will lose uncommitted changes"),
    (r"\brm\s+-rf\b", "Recursive deletion requested"),
    (r"\bsudo\b", "Root privileges requested"),
    (r"\bdocker\s+rm\b", "Docker container removal"),
    (r"\bdocker\s+system\s+prune\b", "Docker system cleanup"),
    (r"\bnpm\s+publish\b", "Publish to npm registry?"),
    (r"\bpip\s+install\s+--user\b", "Install Python package globally?"),
]


def check_command(command: str) -> tuple:
    """Check if a command should be blocked or needs confirmation.

    Returns:
        (decision, reason) where decision is 'allow', 'deny', or 'ask'
    """
    command_lower = command.lower()

    # Check blocked commands
    for pattern in BLOCKED_COMMANDS:
        if re.search(pattern, command_lower):
            return "deny", f"Blocked dangerous command pattern: {pattern}"

    # Check commands that need confirmation
    for pattern, message in ASK_COMMANDS:
        if re.search(pattern, command_lower):
            return "ask", message

    return "allow", None


def main():
    # Read hook input from stdin
    try:
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON input: {e}", file=sys.stderr)
        sys.exit(1)

    tool_input = input_data.get("tool_input", {})
    command = tool_input.get("command", "")

    if not command:
        sys.exit(0)

    decision, reason = check_command(command)

    if decision == "deny":
        output = {
            "decision": "deny",
            "reason": f"Security policy: {reason}. This command has been blocked for safety.",
        }
        print(json.dumps(output))
        sys.exit(0)

    if decision == "ask":
        output = {
            "decision": "ask",
            "reason": reason,
        }
        print(json.dumps(output))
        sys.exit(0)

    # Allow the command
    sys.exit(0)


if __name__ == "__main__":
    main()
