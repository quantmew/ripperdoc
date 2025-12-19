#!/usr/bin/env python3
"""Protect sensitive files from being written.

This hook blocks Write and Edit operations on sensitive files
like .env, credentials, private keys, etc.

Configuration example (.ripperdoc/hooks.json):
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Edit|Write|MultiEdit",
        "hooks": [
          {
            "type": "command",
            "command": "python $RIPPERDOC_PROJECT_DIR/.ripperdoc/hooks/check_write.py"
          }
        ]
      }
    ]
  }
}
"""

import json
import sys
from pathlib import Path

# Patterns for sensitive files (case-insensitive matching)
SENSITIVE_PATTERNS = [
    ".env",
    ".env.local",
    ".env.production",
    ".env.development",
    "credentials.json",
    "secrets.json",
    "secrets.yaml",
    "secrets.yml",
    ".git/",
    ".ssh/",
    "id_rsa",
    "id_ed25519",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
]

# Directories that should never be modified
PROTECTED_DIRS = [
    ".git",
    ".ssh",
    "node_modules",
]


def is_sensitive_path(file_path: str) -> bool:
    """Check if a file path matches sensitive patterns."""
    path = Path(file_path)
    path_str = str(path).lower()
    name = path.name.lower()

    # Check for protected directories
    for protected in PROTECTED_DIRS:
        if f"/{protected}/" in path_str or path_str.startswith(f"{protected}/"):
            return True

    # Check exact filename matches
    for pattern in SENSITIVE_PATTERNS:
        if pattern.startswith("*."):
            # Wildcard extension pattern
            ext = pattern[1:]  # .pem, .key, etc.
            if name.endswith(ext):
                return True
        elif pattern.endswith("/"):
            # Directory pattern
            if pattern[:-1] in path_str:
                return True
        else:
            # Exact match
            if name == pattern.lower():
                return True

    return False


def main() -> None:
    # Read hook input from stdin
    try:
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON input: {e}", file=sys.stderr)
        sys.exit(1)

    tool_input = input_data.get("tool_input", {})
    file_path = tool_input.get("file_path", "")

    if not file_path:
        # No file path specified, allow
        sys.exit(0)

    if is_sensitive_path(file_path):
        # Block the operation
        output = {
            "decision": "deny",
            "reason": f"Security policy: Cannot modify sensitive file '{file_path}'. "
            "If you need to modify this file, please do so manually.",
        }
        print(json.dumps(output))
        sys.exit(0)

    # Allow the operation
    sys.exit(0)


if __name__ == "__main__":
    main()
