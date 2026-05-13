"""Hook-related constants."""

# Default timeout for hook commands (in seconds)
DEFAULT_HOOK_TIMEOUT = 60

# Hook events that match on tool names (tool_name in input).
TOOL_MATCHER_EVENTS = {
    "PreToolUse",
    "PermissionRequest",
    "PostToolUse",
    "PostToolUseFailure",
}

# Hook events with fixed matcher value options.
MATCHER_VALUE_OPTIONS: dict[str, list[str]] = {
    "SessionStart": ["startup", "resume", "clear", "compact"],
    "PreCompact": ["manual", "auto"],
    "Notification": [
        "permission_prompt",
        "idle_prompt",
        "auth_success",
        "elicitation_dialog",
    ],
    "SessionEnd": ["clear", "logout", "prompt_input_exit", "other"],
    "Setup": ["init", "maintenance"],
}

# Hook events where matchers are accepted but ignored (always match).
ALWAYS_MATCHER_EVENTS = {
    "Stop",
}
