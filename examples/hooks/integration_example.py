#!/usr/bin/env python3
"""Complete integration example for Ripperdoc hooks.

This script demonstrates a comprehensive hooks configuration with:
- Bash command logging
- File protection
- Prompt validation
- Session tracking

Copy this file and the corresponding hooks to your project.

Usage:
1. Copy examples/hooks/*.py to .ripperdoc/hooks/
2. Copy the configuration below to .ripperdoc/hooks.local.json
3. Restart Ripperdoc
"""

import json
from pathlib import Path

# Complete hooks configuration
HOOKS_CONFIG = {
    "hooks": {
        # Log and validate Bash commands
        "PreToolUse": [
            {
                "matcher": "Bash",
                "hooks": [
                    {
                        "type": "command",
                        "command": "python $RIPPERDOC_PROJECT_DIR/.ripperdoc/hooks/log_bash.py",
                        "timeout": 5
                    },
                    {
                        "type": "command",
                        "command": "python $RIPPERDOC_PROJECT_DIR/.ripperdoc/hooks/check_bash.py",
                        "timeout": 5
                    }
                ]
            },
            # Protect sensitive files from modification
            {
                "matcher": "Edit|Write|MultiEdit",
                "hooks": [
                    {
                        "type": "command",
                        "command": "python $RIPPERDOC_PROJECT_DIR/.ripperdoc/hooks/check_write.py",
                        "timeout": 5
                    }
                ]
            }
        ],

        # Code quality checks after edits
        "PostToolUse": [
            {
                "matcher": "Edit|MultiEdit",
                "hooks": [
                    {
                        "type": "command",
                        "command": "python $RIPPERDOC_PROJECT_DIR/.ripperdoc/hooks/check_edit.py",
                        "timeout": 5
                    }
                ]
            }
        ],

        # Validate user prompts for sensitive content
        "UserPromptSubmit": [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": "python $RIPPERDOC_PROJECT_DIR/.ripperdoc/hooks/check_prompt.py",
                        "timeout": 5
                    }
                ]
            }
        ],

        # Desktop notifications
        "Notification": [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": "python $RIPPERDOC_PROJECT_DIR/.ripperdoc/hooks/log_notification.py",
                        "timeout": 10
                    }
                ]
            }
        ],

        # Session lifecycle tracking
        "SessionStart": [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": "python $RIPPERDOC_PROJECT_DIR/.ripperdoc/hooks/session_start.py",
                        "timeout": 5
                    }
                ]
            }
        ],

        "SessionEnd": [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": "python $RIPPERDOC_PROJECT_DIR/.ripperdoc/hooks/session_end.py",
                        "timeout": 5
                    }
                ]
            }
        ]
    }
}


def setup_hooks(project_dir: Path):
    """Set up hooks configuration in a project directory."""
    hooks_dir = project_dir / ".ripperdoc" / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)

    # Write configuration
    config_file = project_dir / ".ripperdoc" / "hooks.local.json"
    config_file.write_text(json.dumps(HOOKS_CONFIG, indent=2))
    print(f"Created {config_file}")

    # Add hooks.local.json to gitignore
    gitignore_file = project_dir / ".ripperdoc" / ".gitignore"
    gitignore_entries = ["hooks.local.json", "logs/"]

    existing = set()
    if gitignore_file.exists():
        existing = set(gitignore_file.read_text().splitlines())

    with gitignore_file.open("a") as f:
        for entry in gitignore_entries:
            if entry not in existing:
                f.write(f"{entry}\n")

    print(f"Updated {gitignore_file}")
    print("\nHooks configured! Copy the example scripts to:")
    print(f"  {hooks_dir}/")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        project_dir = Path(sys.argv[1])
    else:
        project_dir = Path.cwd()

    print(f"Setting up hooks in: {project_dir}")
    setup_hooks(project_dir)

    print("\nExample configuration:")
    print(json.dumps(HOOKS_CONFIG, indent=2))
