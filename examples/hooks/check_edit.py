#!/usr/bin/env python3
"""Validate Edit tool operations.

This hook performs additional validation on file edit operations,
such as checking for potentially dangerous patterns being introduced.

Configuration example (.ripperdoc/hooks.json):
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Edit|MultiEdit",
        "hooks": [
          {
            "type": "command",
            "command": "python $RIPPERDOC_PROJECT_DIR/.ripperdoc/hooks/check_edit.py"
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
from pathlib import Path

# Patterns that might indicate security issues in code
SUSPICIOUS_PATTERNS = [
    # Hardcoded credentials
    (r"password\s*=\s*['\"][^'\"]+['\"]", "hardcoded password"),
    (r"api_key\s*=\s*['\"][^'\"]+['\"]", "hardcoded API key"),
    (r"secret\s*=\s*['\"][^'\"]+['\"]", "hardcoded secret"),

    # Dangerous functions
    (r"\beval\s*\(", "eval() usage"),
    (r"\bexec\s*\(", "exec() usage"),
    (r"__import__\s*\(", "dynamic import"),

    # SQL injection risks
    (r"f['\"].*SELECT.*{", "possible SQL injection"),
    (r"\.format\(.*\).*(?:SELECT|INSERT|UPDATE|DELETE)", "possible SQL injection"),

    # Command injection risks
    (r"subprocess\..*shell\s*=\s*True", "shell=True in subprocess"),
    (r"os\.system\s*\(", "os.system usage"),
    (r"os\.popen\s*\(", "os.popen usage"),
]

# File extensions to check
CHECKABLE_EXTENSIONS = {".py", ".js", ".ts", ".jsx", ".tsx", ".rb", ".php", ".java"}


def analyze_new_content(new_string: str, file_ext: str) -> list:
    """Analyze new content for suspicious patterns.

    Returns list of warnings.
    """
    if file_ext not in CHECKABLE_EXTENSIONS:
        return []

    warnings = []
    for pattern, description in SUSPICIOUS_PATTERNS:
        if re.search(pattern, new_string, re.IGNORECASE):
            warnings.append(description)

    return warnings


def main() -> None:
    # Read hook input from stdin
    try:
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON input: {e}", file=sys.stderr)
        sys.exit(1)

    tool_input = input_data.get("tool_input", {})
    file_path = tool_input.get("file_path", "")
    new_string = tool_input.get("new_string", "")

    if not file_path or not new_string:
        sys.exit(0)

    # Get file extension
    file_ext = Path(file_path).suffix.lower()

    # Analyze the new content
    warnings = analyze_new_content(new_string, file_ext)

    if warnings:
        # Found suspicious patterns - warn but don't block
        # You can change this to "ask" or "deny" for stricter enforcement
        warning_list = ", ".join(warnings)
        output = {
            "additionalContext": f"⚠️ Code quality warning: The edit introduces patterns that may need review: {warning_list}",
        }
        print(json.dumps(output))
        sys.exit(0)

    # Allow the edit
    sys.exit(0)


if __name__ == "__main__":
    main()
