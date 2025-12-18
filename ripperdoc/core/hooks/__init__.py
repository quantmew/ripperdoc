"""Hooks system for Ripperdoc.

This module provides a hook system similar to Claude Code CLI's hooks,
allowing users to execute custom scripts at various points in the workflow.

Hook events:
- PreToolUse: Before a tool is called (can block/allow/ask)
- PermissionRequest: When user is shown permission dialog (can allow/deny)
- PostToolUse: After a tool completes
- UserPromptSubmit: When user submits a prompt (can block)
- Notification: When a notification is sent
- Stop: When the agent stops responding (can block to continue)
- SubagentStop: When a subagent task completes (can block to continue)
- PreCompact: Before conversation compaction
- SessionStart: When a session starts or resumes
- SessionEnd: When a session ends

Configuration is stored in:
- ~/.ripperdoc/hooks.json (global)
- .ripperdoc/hooks.json (project)
- .ripperdoc/hooks.local.json (local, git-ignored)
"""

from ripperdoc.core.hooks.events import (
    HookEvent,
    HookDecision,
    HookInput,
    HookOutput,
    PreToolUseInput,
    PermissionRequestInput,
    PostToolUseInput,
    UserPromptSubmitInput,
    NotificationInput,
    StopInput,
    SubagentStopInput,
    PreCompactInput,
    SessionStartInput,
    SessionEndInput,
    PreToolUseHookOutput,
    PermissionRequestHookOutput,
    PermissionRequestDecision,
    PostToolUseHookOutput,
    UserPromptSubmitHookOutput,
    SessionStartHookOutput,
)
from ripperdoc.core.hooks.config import (
    HookDefinition,
    HookMatcher,
    HooksConfig,
    load_hooks_config,
    get_merged_hooks_config,
    get_global_hooks_path,
    get_project_hooks_path,
    get_project_local_hooks_path,
)
from ripperdoc.core.hooks.executor import HookExecutor, LLMCallback
from ripperdoc.core.hooks.manager import HookManager, HookResult, hook_manager, init_hook_manager

__all__ = [
    # Events
    "HookEvent",
    "HookDecision",
    "HookInput",
    "HookOutput",
    "PreToolUseInput",
    "PermissionRequestInput",
    "PostToolUseInput",
    "UserPromptSubmitInput",
    "NotificationInput",
    "StopInput",
    "SubagentStopInput",
    "PreCompactInput",
    "SessionStartInput",
    "SessionEndInput",
    # Hook-specific outputs
    "PreToolUseHookOutput",
    "PermissionRequestHookOutput",
    "PermissionRequestDecision",
    "PostToolUseHookOutput",
    "UserPromptSubmitHookOutput",
    "SessionStartHookOutput",
    # Config
    "HookDefinition",
    "HookMatcher",
    "HooksConfig",
    "load_hooks_config",
    "get_merged_hooks_config",
    "get_global_hooks_path",
    "get_project_hooks_path",
    "get_project_local_hooks_path",
    # Executor
    "HookExecutor",
    "LLMCallback",
    # Manager
    "HookManager",
    "HookResult",
    "hook_manager",
    "init_hook_manager",
]
