"""Hook configuration loading and management.

This module handles loading hooks configuration from:
- ~/.ripperdoc/hooks.json (global/user-level)
- .ripperdoc/hooks.json (project-level, checked into git)
- .ripperdoc/hooks.local.json (local, git-ignored)

Configuration format:
{
  "hooks": {
    "EventName": [
      {
        "matcher": "ToolPattern",  // Only for PreToolUse/PermissionRequest/PostToolUse
        "hooks": [
          {
            "type": "command",
            "command": "your-command-here",
            "timeout": 100
          },
          {
            "type": "prompt",
            "prompt": "Evaluate if this should proceed: $ARGUMENTS",
            "timeout": 30
          }
        ]
      }
    ]
  }
}
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

from ripperdoc.core.hooks.events import HookEvent
from ripperdoc.utils.log import get_logger

logger = get_logger()

# Default timeout for hook commands (in seconds)
DEFAULT_HOOK_TIMEOUT = 60

# Hook events that support prompt-based hooks
PROMPT_SUPPORTED_EVENTS = {
    "Stop",
    "SubagentStop",
    "UserPromptSubmit",
    "PreToolUse",
    "PermissionRequest",
}


class HookDefinition(BaseModel):
    """Definition of a single hook.

    Supports two types:
    - command: Execute a shell command
    - prompt: Use LLM to evaluate (for supported events only)
    """

    type: Literal["command", "prompt"] = "command"
    command: Optional[str] = None  # Shell command (for type="command")
    prompt: Optional[str] = None  # LLM prompt (for type="prompt"), use $ARGUMENTS for input JSON
    timeout: int = DEFAULT_HOOK_TIMEOUT  # Timeout in seconds

    def is_command_hook(self) -> bool:
        """Check if this is a command-based hook."""
        return self.type == "command" and self.command is not None

    def is_prompt_hook(self) -> bool:
        """Check if this is a prompt-based hook."""
        return self.type == "prompt" and self.prompt is not None


class HookMatcher(BaseModel):
    """A matcher that groups hooks for a specific pattern.

    For PreToolUse/PostToolUse events, the matcher can be:
    - A specific tool name (e.g., "Bash", "Write", "Edit")
    - A regex pattern (e.g., "Edit|Write", "mcp__.*__write.*")
    - "*" or "" to match all tools
    """

    matcher: Optional[str] = None  # None or empty means match all
    hooks: List[HookDefinition] = Field(default_factory=list)

    def matches(self, tool_name: Optional[str] = None) -> bool:
        """Check if this matcher matches the given tool name.

        For events that don't use tool names, always returns True if matcher is empty.
        """
        if not self.matcher or self.matcher == "*":
            return True

        if tool_name is None:
            return True

        # Try exact match first (case-sensitive)
        if self.matcher == tool_name:
            return True

        # Try regex match
        try:
            pattern = re.compile(self.matcher)
            return bool(pattern.match(tool_name))
        except re.error:
            # Invalid regex, fall back to simple string comparison
            return False


class HooksConfig(BaseModel):
    """Configuration for all hooks."""

    hooks: Dict[str, List[HookMatcher]] = Field(default_factory=dict)

    def get_hooks_for_event(
        self, event: HookEvent, tool_name: Optional[str] = None
    ) -> List[HookDefinition]:
        """Get all hooks that should run for a given event and optional tool name."""
        event_name = event.value
        if event_name not in self.hooks:
            return []

        result = []
        for matcher in self.hooks[event_name]:
            if matcher.matches(tool_name):
                result.extend(matcher.hooks)
        return result

    def merge_with(self, other: "HooksConfig") -> "HooksConfig":
        """Merge this config with another, with 'other' taking precedence for additions."""
        merged_hooks: Dict[str, List[HookMatcher]] = {}

        # Copy all events from this config
        for event_name, matchers in self.hooks.items():
            merged_hooks[event_name] = list(matchers)

        # Add/extend with hooks from other config
        for event_name, matchers in other.hooks.items():
            if event_name not in merged_hooks:
                merged_hooks[event_name] = []
            merged_hooks[event_name].extend(matchers)

        return HooksConfig(hooks=merged_hooks)


def _parse_hooks_file(data: Dict[str, Any]) -> HooksConfig:
    """Parse a hooks configuration from a dictionary.

    Supports two formats:
    1. With 'hooks' wrapper: {"hooks": {"PreToolUse": [...]}}
    2. Without wrapper: {"PreToolUse": [...]}
    """
    # Support both formats: with or without "hooks" wrapper
    hooks_data = data.get("hooks", {})
    if not hooks_data:
        # Try treating the entire data as hooks config (no wrapper)
        # Check if any top-level key looks like an event name
        for key in data.keys():
            try:
                HookEvent(key)
                hooks_data = data
                break
            except ValueError:
                continue

    parsed_hooks: Dict[str, List[HookMatcher]] = {}

    for event_name, matchers_list in hooks_data.items():
        # Validate event name
        try:
            HookEvent(event_name)
        except ValueError:
            logger.warning(f"Unknown hook event: {event_name}")
            continue

        if not isinstance(matchers_list, list):
            logger.warning(f"Invalid hooks config for {event_name}: expected list")
            continue

        parsed_matchers: List[HookMatcher] = []
        for matcher_data in matchers_list:
            if not isinstance(matcher_data, dict):
                continue

            matcher_pattern = matcher_data.get("matcher")
            hooks_list = matcher_data.get("hooks", [])

            if not isinstance(hooks_list, list):
                continue

            hook_definitions = []
            for hook_data in hooks_list:
                if not isinstance(hook_data, dict):
                    continue

                hook_type = hook_data.get("type", "command")

                # Validate hook type
                if hook_type not in ("command", "prompt"):
                    logger.warning(f"Unknown hook type: {hook_type}")
                    continue

                # For command hooks, require command field
                if hook_type == "command":
                    if "command" not in hook_data:
                        continue
                    hook_def = HookDefinition(
                        type="command",
                        command=hook_data["command"],
                        timeout=hook_data.get("timeout", DEFAULT_HOOK_TIMEOUT),
                    )
                # For prompt hooks, require prompt field and validate event
                elif hook_type == "prompt":
                    if "prompt" not in hook_data:
                        continue
                    # Warn if prompt hooks used on unsupported events
                    if event_name not in PROMPT_SUPPORTED_EVENTS:
                        logger.warning(
                            f"Prompt hooks not supported for {event_name} event, skipping"
                        )
                        continue
                    hook_def = HookDefinition(
                        type="prompt",
                        prompt=hook_data["prompt"],
                        timeout=hook_data.get("timeout", DEFAULT_HOOK_TIMEOUT),
                    )
                else:
                    continue

                hook_definitions.append(hook_def)

            if hook_definitions:
                parsed_matchers.append(HookMatcher(matcher=matcher_pattern, hooks=hook_definitions))

        if parsed_matchers:
            parsed_hooks[event_name] = parsed_matchers

    return HooksConfig(hooks=parsed_hooks)


def load_hooks_config(config_path: Path) -> HooksConfig:
    """Load hooks configuration from a file."""
    if not config_path.exists():
        return HooksConfig()

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        config = _parse_hooks_file(data)
        logger.debug(
            f"Loaded hooks config from {config_path}",
            extra={"event_count": len(config.hooks)},
        )
        return config
    except json.JSONDecodeError as e:
        logger.warning(f"Invalid JSON in hooks config {config_path}: {e}")
        return HooksConfig()
    except (OSError, IOError) as e:
        logger.warning(f"Error reading hooks config {config_path}: {e}")
        return HooksConfig()


def get_global_hooks_path() -> Path:
    """Get the path to the global hooks configuration."""
    return Path.home() / ".ripperdoc" / "hooks.json"


def get_project_hooks_path(project_path: Path) -> Path:
    """Get the path to the project hooks configuration."""
    return project_path / ".ripperdoc" / "hooks.json"


def get_project_local_hooks_path(project_path: Path) -> Path:
    """Get the path to the local (git-ignored) project hooks configuration."""
    return project_path / ".ripperdoc" / "hooks.local.json"


def get_merged_hooks_config(project_path: Optional[Path] = None) -> HooksConfig:
    """Get the merged hooks configuration from all sources.

    Order of precedence (later overrides earlier):
    1. Global config (~/.ripperdoc/hooks.json)
    2. Project config (.ripperdoc/hooks.json)
    3. Local project config (.ripperdoc/hooks.local.json)
    """
    # Start with global config
    config = load_hooks_config(get_global_hooks_path())

    # Merge project config if available
    if project_path:
        project_config = load_hooks_config(get_project_hooks_path(project_path))
        config = config.merge_with(project_config)

        # Merge local config
        local_config = load_hooks_config(get_project_local_hooks_path(project_path))
        config = config.merge_with(local_config)

    return config
