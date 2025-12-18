# Ripperdoc Hooks Examples

This directory contains example hook scripts for Ripperdoc's hook system.

## What are Hooks?

Hooks are custom scripts that run at specific points during Ripperdoc's operation:

- **PreToolUse**: Before a tool is called (can block execution)
- **PostToolUse**: After a tool completes
- **UserPromptSubmit**: When user submits a prompt
- **Notification**: When Ripperdoc sends a notification
- **Stop**: When the agent stops responding
- **SubagentStop**: When a subagent task completes
- **PreCompact**: Before conversation compaction
- **SessionStart**: When a session starts or resumes
- **SessionEnd**: When a session ends

## Configuration

Hooks are configured in JSON files:

- `~/.ripperdoc/hooks.json` - Global hooks (apply to all projects)
- `.ripperdoc/hooks.json` - Project hooks (checked into git)
- `.ripperdoc/hooks.local.json` - Local hooks (git-ignored)

### Configuration Format

```json
{
  "hooks": {
    "EventName": [
      {
        "matcher": "ToolPattern",
        "hooks": [
          {
            "type": "command",
            "command": "your-command-here",
            "timeout": 60
          }
        ]
      }
    ]
  }
}
```

### Matcher Patterns

For `PreToolUse` and `PostToolUse` events:

- Exact match: `"Bash"`, `"Write"`, `"Edit"`
- Regex pattern: `"Edit|Write"`, `"mcp__.*__write.*"`
- Match all: `"*"` or omit/empty

### Environment Variables

Hooks receive these environment variables:
- `RIPPERDOC_PROJECT_DIR`: Path to the project directory
- `CLAUDE_PROJECT_DIR`: Alias for compatibility

### Input/Output

Hooks receive JSON input via stdin containing event-specific data.

Hooks can output:
- Plain text (added as context)
- JSON with decision control:

```json
{
  "decision": "allow|deny|ask|block",
  "reason": "Explanation shown to user/model",
  "suppressOutput": false,
  "additionalContext": "Context to add to conversation"
}
```

### Exit Codes

- `0`: Success, continue normally
- `2`: Block/deny the operation
- Other: Error occurred

## Examples

### log_bash.py
Logs all Bash commands to a file.

### check_write.py
Prevents writing to sensitive files.

### check_prompt.py
Filters sensitive information from prompts.

### session_start.py / session_end.py
Track session lifecycle events.

### log_notification.py
Send desktop notifications.

## Quick Start

1. Copy example scripts to your project's `.ripperdoc/hooks/` directory

2. Create `.ripperdoc/hooks.json`:
```json
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
```

3. Restart Ripperdoc to load the new configuration.
