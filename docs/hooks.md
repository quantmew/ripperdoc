# Ripperdoc Hooks System

Hooks allow you to execute custom scripts at various points during Ripperdoc's operation.
This is compatible with Claude Code CLI's hooks system.

## Table of Contents

- [Overview](#overview)
- [Configuration](#configuration)
- [Hook Events](#hook-events)
- [Input/Output Format](#inputoutput-format)
- [Decision Control](#decision-control)
- [Examples](#examples)
- [Best Practices](#best-practices)

## Overview

Hooks are user-defined commands that run at specific points:

- **PreToolUse**: Before a tool is called (can block/allow/ask)
- **PermissionRequest**: When user is shown permission dialog (can allow/deny)
- **PostToolUse**: After a tool completes
- **UserPromptSubmit**: When user submits a prompt (can block)
- **Notification**: When Ripperdoc sends a notification
- **Stop**: When the agent stops responding (can block to continue)
- **SubagentStop**: When a subagent task completes (can block to continue)
- **PreCompact**: Before conversation compaction
- **SessionStart**: When a session starts/resumes
- **SessionEnd**: When a session ends

## Configuration

Hooks are configured in JSON files with this hierarchy:

1. **Global**: `~/.ripperdoc/hooks.json` - applies to all projects
2. **Project**: `.ripperdoc/hooks.json` - project-specific, checked into git
3. **Local**: `.ripperdoc/hooks.local.json` - local overrides, git-ignored

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

### Fields

| Field | Description | Required |
|-------|-------------|----------|
| `matcher` | Pattern to match tool names (PreToolUse/PermissionRequest/PostToolUse only) | No |
| `hooks` | Array of hook commands to run | Yes |
| `type` | Hook type: "command" or "prompt" | Yes |
| `command` | Shell command to execute (for type="command") | Required for command hooks |
| `prompt` | LLM prompt template (for type="prompt") | Required for prompt hooks |
| `timeout` | Timeout in seconds (default: 60) | No |

### Hook Types

#### Command Hooks (type: "command")

Execute a shell command. The command receives hook input via stdin as JSON.

```json
{
  "type": "command",
  "command": "python check_command.py",
  "timeout": 60
}
```

#### Prompt Hooks (type: "prompt")

Use an LLM to evaluate the hook. The prompt can include `$ARGUMENTS` which will be replaced with the JSON input.

```json
{
  "type": "prompt",
  "prompt": "Evaluate if this tool call should proceed: $ARGUMENTS. Respond with JSON containing 'decision' (approve/block) and 'reason'.",
  "timeout": 30
}
```

**Supported Events for Prompt Hooks:**
- `Stop`
- `SubagentStop`
- `UserPromptSubmit`
- `PreToolUse`
- `PermissionRequest`

**Prompt Hook Response Format:**
```json
{
  "decision": "approve|block",
  "reason": "explanation of the decision",
  "continue": true,
  "stopReason": "optional message if blocking",
  "systemMessage": "optional warning"
}
```

### Matcher Patterns

For `PreToolUse`, `PermissionRequest`, and `PostToolUse`:

- **Exact match**: `"Bash"`, `"Write"`, `"Edit"`
- **Regex**: `"Edit|Write"`, `"mcp__.*__write.*"`
- **Match all**: `"*"` or omit the field

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [{"type": "command", "command": "echo 'Bash called'"}]
      },
      {
        "matcher": "Edit|Write|MultiEdit",
        "hooks": [{"type": "command", "command": "echo 'File operation'"}]
      },
      {
        "matcher": "mcp__.*",
        "hooks": [{"type": "command", "command": "echo 'MCP tool called'"}]
      }
    ]
  }
}
```

## Hook Events

### PreToolUse

Runs after Agent creates tool parameters and before processing the tool call. Can block, allow, or modify the execution.

**Matchers**: Tool-specific (Bash, Write, Edit, Read, Glob, Grep, etc.)

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {"type": "command", "command": "python check_command.py"}
        ]
      }
    ]
  }
}
```

**Input fields**:
- `hook_event_name`: "PreToolUse"
- `tool_name`: Name of the tool
- `tool_input`: Tool parameters (object)
- `tool_use_id`: Unique identifier for this tool call
- `session_id`: Current session ID
- `transcript_path`: Path to conversation JSON file
- `cwd`: Current working directory
- `permission_mode`: Current mode ("default", "plan", "acceptEdits", "bypassPermissions")

**Output decisions**:
- `allow`: Bypass permission system, auto-approve
- `deny`: Block the tool call, inform model
- `ask`: Prompt user for confirmation

**Output can also include**:
- `updatedInput`: Modified tool input parameters
- `additionalContext`: Extra info for the model

### PermissionRequest

Runs when the user is shown a permission dialog.

**Matchers**: Tool-specific

**Input fields**: Same as PreToolUse

**Output decisions**:
- `allow`: Auto-approve the permission
- `deny`: Auto-deny the permission

### PostToolUse

Runs immediately after a tool completes successfully.

**Input fields**:
- `hook_event_name`: "PostToolUse"
- `tool_name`: Name of the tool
- `tool_input`: Tool parameters (object)
- `tool_response`: Tool's response/output
- `tool_use_id`: Unique identifier for this tool call
- `session_id`, `transcript_path`, `cwd`, `permission_mode`

**Output decisions**:
- `block`: Auto-prompt the model about the issue

### UserPromptSubmit

Runs when the user submits a prompt, before Agent processes it.

**Input fields**:
- `hook_event_name`: "UserPromptSubmit"
- `prompt`: User's prompt text
- `session_id`, `transcript_path`, `cwd`, `permission_mode`

**Output decisions**:
- `block`: Reject the prompt, show reason to user

### Notification

Runs when Ripperdoc sends notifications.

**Input fields**:
- `hook_event_name`: "Notification"
- `message`: Notification content
- `notification_type`: Type of notification
  - `permission_prompt`: Permission requests
  - `idle_prompt`: Waiting for user input (after 60+ seconds idle)
  - `auth_success`: Authentication success
  - `elicitation_dialog`: MCP tool elicitation
- `session_id`, `transcript_path`, `cwd`, `permission_mode`

### Stop

Runs when the main agent has finished responding. Does not run if stoppage occurred due to user interrupt.

**Input fields**:
- `hook_event_name`: "Stop"
- `stop_hook_active`: True if already continuing from a stop hook
- `session_id`, `transcript_path`, `cwd`, `permission_mode`

**Output decisions**:
- `block`: Prevent stopping, reason tells model how to continue

### SubagentStop

Runs when a subagent (Task tool call) has finished responding.

**Input fields**:
- `hook_event_name`: "SubagentStop"
- `stop_hook_active`: True if already continuing from a stop hook
- `session_id`, `transcript_path`, `cwd`, `permission_mode`

**Output decisions**:
- `block`: Prevent stopping, reason tells model how to continue

### PreCompact

Runs before a compact operation.

**Input fields**:
- `hook_event_name`: "PreCompact"
- `trigger`: "manual" (from /compact) or "auto" (from full context window)
- `custom_instructions`: Custom instructions passed to /compact
- `session_id`, `transcript_path`, `cwd`, `permission_mode`

### SessionStart

Runs when a session starts or resumes.

**Source values**:
- `startup`: Fresh start
- `resume`: From --resume, --continue, or /resume
- `clear`: From /clear
- `compact`: From auto or manual compact

**Input fields**:
- `hook_event_name`: "SessionStart"
- `source`: Start source type
- `session_id`, `transcript_path`, `cwd`, `permission_mode`

**Environment variables**:
SessionStart hooks have access to `RIPPERDOC_ENV_FILE`, which provides a file path where environment variables can be persisted as JSON.

### SessionEnd

Runs when a session ends.

**Reason values**:
- `clear`: Session cleared with /clear command
- `logout`: User logged out
- `prompt_input_exit`: User exited while prompt input was visible
- `other`: Other exit reasons

**Input fields**:
- `hook_event_name`: "SessionEnd"
- `reason`: End reason type
- `session_id`, `transcript_path`, `cwd`, `permission_mode`

## Input/Output Format

### Input

Hooks receive JSON input via **stdin**:

```json
{
  "hook_event_name": "PreToolUse",
  "tool_name": "Bash",
  "tool_input": {
    "command": "ls -la",
    "description": "List files"
  },
  "tool_use_id": "toolu_abc123",
  "session_id": "session-uuid",
  "transcript_path": "/path/to/transcript.json",
  "cwd": "/current/working/dir",
  "permission_mode": "default"
}
```

### Output

Hooks can output:

1. **Plain text**: Added as additional context
2. **JSON**: For decision control

#### Simple JSON Output

```json
{
  "decision": "allow",
  "reason": "Auto-approved safe operation"
}
```

#### Extended JSON Output

```json
{
  "continue": true,
  "stopReason": null,
  "suppressOutput": false,
  "systemMessage": null,
  "hookSpecificOutput": {
    "hookEventName": "PreToolUse",
    "permissionDecision": "allow",
    "permissionDecisionReason": "Safe operation",
    "updatedInput": {
      "command": "ls -la --color=never"
    },
    "additionalContext": "Extra info for the model"
  }
}
```

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success. stdout processed, JSON parsed if present |
| 2 | Blocking error. stderr is used as error message, JSON ignored |
| Other | Non-blocking error. stderr shown to user |

## Decision Control

### PreToolUse/PermissionRequest Decisions

| Decision | Effect |
|----------|--------|
| `allow` | Bypass permission system, auto-approve |
| `deny` | Block the tool call, inform model |
| `ask` | Prompt user for confirmation (PreToolUse only) |

### PostToolUse Decisions

| Decision | Effect |
|----------|--------|
| `block` | Auto-prompt model about issue |

### UserPromptSubmit Decisions

| Decision | Effect |
|----------|--------|
| `block` | Reject the prompt, show reason |

### Stop/SubagentStop Decisions

| Decision | Effect |
|----------|--------|
| `block` | Prevent stopping, reason tells model how to continue |

## Examples

### Log Bash Commands

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

### Protect Sensitive Files

```python
#!/usr/bin/env python3
import json
import sys

PROTECTED = [".env", "credentials.json", ".git/"]

input_data = json.load(sys.stdin)
file_path = input_data.get("tool_input", {}).get("file_path", "")

for pattern in PROTECTED:
    if pattern in file_path:
        print(json.dumps({
            "decision": "deny",
            "reason": f"Cannot modify protected file: {file_path}"
        }))
        sys.exit(0)

sys.exit(0)
```

### Modify Tool Input

```python
#!/usr/bin/env python3
"""PreToolUse hook that modifies Bash commands."""
import json
import sys

input_data = json.load(sys.stdin)
tool_input = input_data.get("tool_input", {})
command = tool_input.get("command", "")

# Always add --color=never to ls commands
if command.strip().startswith("ls "):
    if "--color" not in command:
        tool_input["command"] = command + " --color=never"

print(json.dumps({
    "hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "allow",
        "updatedInput": tool_input
    }
}))
```

### Desktop Notifications

```python
#!/usr/bin/env python3
import json
import subprocess
import sys

input_data = json.load(sys.stdin)
message = input_data.get("message", "")
notification_type = input_data.get("notification_type", "info")

# Linux
subprocess.run(["notify-send", "Ripperdoc", message])

# macOS
# subprocess.run(["osascript", "-e", f'display notification "{message}" with title "Ripperdoc"'])

sys.exit(0)
```

### Block on Exit Code 2

```python
#!/usr/bin/env python3
"""Block dangerous commands using exit code 2."""
import json
import sys

input_data = json.load(sys.stdin)
command = input_data.get("tool_input", {}).get("command", "")

DANGEROUS = ["rm -rf /", "dd if=", "mkfs", "> /dev/"]

for pattern in DANGEROUS:
    if pattern in command:
        # Exit code 2 blocks with stderr as reason
        print(f"Dangerous command blocked: {pattern}", file=sys.stderr)
        sys.exit(2)

sys.exit(0)
```

### LLM-Based Prompt Hook

Use an LLM to evaluate whether a command should proceed:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "prompt",
            "prompt": "You are a security reviewer. Analyze this bash command and decide if it should be allowed. Command details: $ARGUMENTS\n\nRespond with JSON: {\"decision\": \"approve\" or \"block\", \"reason\": \"your reasoning\"}",
            "timeout": 30
          }
        ]
      }
    ]
  }
}
```

## Environment Variables

Hooks have access to:

| Variable | Description |
|----------|-------------|
| `RIPPERDOC_PROJECT_DIR` | Project directory path |
| `RIPPERDOC_SESSION_ID` | Current session ID |
| `RIPPERDOC_TRANSCRIPT_PATH` | Path to conversation transcript |
| `RIPPERDOC_ENV_FILE` | Environment file path (SessionStart only) |

Use in commands:
```json
{
  "command": "python $RIPPERDOC_PROJECT_DIR/.ripperdoc/hooks/myhook.py"
}
```

## Best Practices

1. **Keep hooks fast**: Use short timeouts for critical hooks
2. **Handle errors gracefully**: Don't crash on unexpected input
3. **Log for debugging**: Write to `.ripperdoc/logs/` for troubleshooting
4. **Use project-local config**: Put experimental hooks in `hooks.local.json`
5. **Test thoroughly**: Use the demo hook to understand input format
6. **Fail safe**: When in doubt, allow operations to proceed
7. **Use exit code 2**: For blocking errors, use exit code 2 with stderr message

## Programmatic Usage

```python
from pathlib import Path
from ripperdoc.core.hooks import (
    HookManager,
    HookResult,
    init_hook_manager,
    hook_manager,
    LLMCallback,
)

# Define LLM callback for prompt hooks (optional)
async def my_llm_callback(prompt: str) -> str:
    # Call your LLM here
    response = await call_your_llm(prompt)
    return response

# Initialize with project context
init_hook_manager(
    project_dir=Path("/path/to/project"),
    session_id="my-session",
    transcript_path="/path/to/transcript.json",
    permission_mode="default",
    llm_callback=my_llm_callback,  # Required for prompt hooks
)

# Check before tool execution
result: HookResult = await hook_manager.run_pre_tool_use_async(
    tool_name="Bash",
    tool_input={"command": "ls -la"},
    tool_use_id="toolu_123",
)

if result.should_block:
    print(f"Blocked: {result.block_reason}")
elif result.should_allow:
    print("Auto-approved by hook")
elif result.updated_input:
    print(f"Input modified: {result.updated_input}")

# Run after tool execution
await hook_manager.run_post_tool_use_async(
    tool_name="Bash",
    tool_input={"command": "ls -la"},
    tool_response="file1.txt\nfile2.txt",
    tool_use_id="toolu_123",
)

# Clean up on session end
hook_manager.cleanup()
```

## Troubleshooting

### Hooks not running

1. Check configuration file location and syntax
2. Verify the command path is correct
3. Check file permissions (scripts must be executable)
4. Look for errors in Ripperdoc logs

### Hook blocks unexpectedly

1. Run the hook command manually with sample input
2. Check the exit code
3. Verify JSON output format

### Timeout issues

Increase the timeout in configuration:
```json
{
  "timeout": 120
}
```

### View configured hooks

Use the `/hooks` command in Ripperdoc to see all configured hooks.

### Manage hooks from the CLI

Use the guided editors to create and maintain hooks without hand-editing JSON:

- `/hooks add [scope]` walks through scope selection (local/project/global), event, matcher, command or prompt text, and timeout. Defaults to `.ripperdoc/hooks.local.json`.
- `/hooks edit [scope]` lets you pick an existing hook and update its fields.
- `/hooks delete [scope]` removes a hook entry and cleans up empty matchers.

Scopes match the files described in [Configuration](#configuration); omit the scope to be prompted.
