"""Permission-related constants."""

# Edit preview limits
EDIT_PREVIEW_MAX_DIFF_LINES = 30
EDIT_PREVIEW_MAX_BYTES = 1_500_000
EDIT_PREVIEW_MATCH_SNIPPET_MAX = 140
EDIT_PREVIEW_SEPARATOR = "-------------------"

# Permission prompt layout
PERMISSION_PROMPT_RESERVED_LINES = 14
PERMISSION_PROMPT_MIN_DIFF_LINES = 4

# Permission modes
PERMISSION_MODES = {"default", "acceptEdits", "plan", "bypassPermissions", "dontAsk"}

# Tools that auto-memory write operations apply to
AUTO_MEMORY_WRITE_TOOLS = {"Write", "Edit", "MultiEdit"}

# Plan mode tool restrictions
PLAN_MODE_SPECIAL_ALLOWED_TOOLS = {"AskUserQuestion", "ExitPlanMode"}
PLAN_MODE_PLAN_FILE_EDIT_TOOLS = {"Write", "Edit", "MultiEdit"}
