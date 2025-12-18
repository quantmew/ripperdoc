#!/usr/bin/env python3
"""Filter and block prompts containing sensitive information.

This hook scans user prompts for potential secrets like passwords,
API keys, tokens, etc. and blocks them from being processed.

Configuration example (.ripperdoc/hooks.json):
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python $RIPPERDOC_PROJECT_DIR/.ripperdoc/hooks/check_prompt.py"
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
from datetime import datetime

# Patterns that might indicate secrets in prompts
SENSITIVE_PATTERNS = [
    # Password patterns
    (r"(?i)\b(password|passwd|pwd)\s*[:=]\s*\S+", "password"),
    (r"(?i)\bpassword\s+is\s+['\"]?\S+", "password"),

    # API key patterns
    (r"(?i)\b(api[_-]?key|apikey)\s*[:=]\s*\S+", "API key"),
    (r"(?i)\b(secret[_-]?key|secretkey)\s*[:=]\s*\S+", "secret key"),

    # Token patterns
    (r"(?i)\b(auth[_-]?token|bearer)\s*[:=]\s*\S+", "auth token"),
    (r"(?i)\b(access[_-]?token)\s*[:=]\s*\S+", "access token"),

    # AWS patterns
    (r"AKIA[0-9A-Z]{16}", "AWS access key"),
    (r"(?i)aws[_-]?secret[_-]?access[_-]?key\s*[:=]\s*\S+", "AWS secret key"),

    # Private key patterns
    (r"-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----", "private key"),

    # Database connection strings
    (r"(?i)(mysql|postgres|mongodb)://[^\s]+:[^\s]+@", "database connection string"),

    # Generic secret patterns
    (r"(?i)\bsecret\s*[:=]\s*['\"]?\S{10,}", "secret value"),
]


def check_for_secrets(prompt: str) -> list:
    """Check prompt for potential secrets.

    Returns list of (pattern_name, matched_text) tuples.
    """
    findings = []
    for pattern, name in SENSITIVE_PATTERNS:
        matches = re.findall(pattern, prompt)
        if matches:
            findings.append(name)
    return findings


def main():
    # Read hook input from stdin
    try:
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON input: {e}", file=sys.stderr)
        sys.exit(1)

    prompt = input_data.get("prompt", "")

    if not prompt:
        sys.exit(0)

    # Check for sensitive patterns
    findings = check_for_secrets(prompt)

    if findings:
        # Found potential secrets - block the prompt
        finding_list = ", ".join(set(findings))
        output = {
            "decision": "block",
            "reason": f"Security policy violation: Prompt appears to contain sensitive information ({finding_list}). "
            "Please rephrase your request without including secrets, passwords, or credentials.",
        }
        print(json.dumps(output))
        sys.exit(0)

    # Add timestamp context (optional feature)
    context = f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    print(context)
    sys.exit(0)


if __name__ == "__main__":
    main()
