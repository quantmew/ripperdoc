"""Hook configuration loading and management.

This module handles loading hooks configuration from:
- ~/.ripperdoc/hooks.json (user-level)
- .ripperdoc/hooks.json (project-level, checked into git)
- .ripperdoc/hooks.local.json (local, git-ignored)

Configuration format:
{
  "hooks": {
    "EventName": [
      {
        "matcher": "Pattern",  // Optional per event; see matcher rules below
        "hooks": [
          {
            "type": "command",
            "command": "your-command-here",
            "async": false,
            "statusMessage": "Running hook command",
            "timeout": 100
          },
          {
            "type": "prompt",
            "prompt": "Evaluate if this should proceed: $ARGUMENTS",
            "timeout": 30
          },
          {
            "type": "agent",
            "prompt": "Verify with tools: $ARGUMENTS",
            "model": "quick",
            "timeout": 60
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
from pydantic import BaseModel, ConfigDict, Field

from ripperdoc.core.hooks.events import HookEvent
from ripperdoc.utils.log import get_logger

logger = get_logger()

# Default timeout for hook commands (in seconds)
DEFAULT_HOOK_TIMEOUT = 60

# Prompt/agent hooks are supported for all hook events.
PROMPT_SUPPORTED_EVENTS = {event.value for event in HookEvent}

# Agent hooks support the same events as prompt hooks.
AGENT_SUPPORTED_EVENTS = set(PROMPT_SUPPORTED_EVENTS)

# Hook events that match on tool names (tool_name in input).
TOOL_MATCHER_EVENTS = {
    HookEvent.PRE_TOOL_USE.value,
    HookEvent.PERMISSION_REQUEST.value,
    HookEvent.POST_TOOL_USE.value,
    HookEvent.POST_TOOL_USE_FAILURE.value,
}

# Hook events with fixed matcher value options.
MATCHER_VALUE_OPTIONS: Dict[str, List[str]] = {
    HookEvent.SESSION_START.value: ["startup", "resume", "clear", "compact"],
    HookEvent.PRE_COMPACT.value: ["manual", "auto"],
    HookEvent.NOTIFICATION.value: [
        "permission_prompt",
        "idle_prompt",
        "auth_success",
        "elicitation_dialog",
    ],
    HookEvent.SESSION_END.value: ["clear", "logout", "prompt_input_exit", "other"],
    HookEvent.SETUP.value: ["init", "maintenance"],
}

# Hook events that allow free-text matchers (non-tool).
TEXT_MATCHER_EVENTS = {
    HookEvent.USER_PROMPT_SUBMIT.value,
    HookEvent.SUBAGENT_START.value,
    HookEvent.SUBAGENT_STOP.value,
}

# Hook events where matchers are accepted but ignored (always match).
ALWAYS_MATCHER_EVENTS = {
    HookEvent.STOP.value,
}


class HookDefinition(BaseModel):
    """Definition of a single hook.

    Supports hook types:
    - command: Execute a shell command
    - prompt: Use LLM to evaluate
    - agent: Spawn a subagent to evaluate with tool access
    """

    type: Literal["command", "prompt", "agent"] = "command"
    command: Optional[str] = None  # Shell command (for type="command")
    prompt: Optional[str] = None  # LLM prompt (for type="prompt"), use $ARGUMENTS for input JSON
    timeout: int = DEFAULT_HOOK_TIMEOUT  # Timeout in seconds
    model: Optional[str] = None  # Model pointer for agent hooks (e.g., "quick")
    run_async: bool = Field(default=False, alias="async")  # Background execution (command only)
    run_once: bool = Field(default=False, alias="once")  # Execute only once per session
    hook_id: Optional[str] = Field(default=None, alias="id")  # Optional stable hook identity
    status_message: Optional[str] = Field(default=None, alias="statusMessage")

    model_config = ConfigDict(populate_by_name=True)

    def is_command_hook(self) -> bool:
        """Check if this is a command-based hook."""
        return self.type == "command" and self.command is not None

    def is_prompt_hook(self) -> bool:
        """Check if this is a prompt-based hook."""
        return self.type == "prompt" and self.prompt is not None

    def is_agent_hook(self) -> bool:
        """Check if this is an agent-based hook."""
        return self.type == "agent" and self.prompt is not None


class HookMatcher(BaseModel):
    """A matcher that groups hooks for a specific pattern.

    Matcher value depends on event:
    - Tool lifecycle events: tool_name (exact match or regex).
    - Notification: notification_type
    - PreCompact: trigger ("manual" or "auto")
    - SessionStart: source ("startup", "resume", "clear", "compact")
    - SessionEnd: reason ("clear", "logout", "prompt_input_exit", "other")
    - Setup: trigger ("init" or "maintenance")
    - SubagentStart: subagent_type
    - UserPromptSubmit: prompt text

    Use "*" or "" to match all.
    """

    matcher: Optional[str] = None  # None or empty means match all
    hooks: List[HookDefinition] = Field(default_factory=list)

    def matches(self, matcher_value: Optional[str] = None) -> bool:
        """Check if this matcher matches the given matcher value."""
        if not self.matcher or self.matcher == "*":
            return True

        if matcher_value is None:
            return True

        # Try exact match first (case-sensitive)
        if self.matcher == matcher_value:
            return True

        # Try regex match
        try:
            pattern = re.compile(self.matcher)
            return bool(pattern.match(matcher_value))
        except re.error:
            # Invalid regex, fall back to simple string comparison
            return False


class HooksConfig(BaseModel):
    """Configuration for all hooks."""

    hooks: Dict[str, List[HookMatcher]] = Field(default_factory=dict)

    def get_hooks_for_event(
        self, event: HookEvent, matcher_value: Optional[str] = None
    ) -> List[HookDefinition]:
        """Get all hooks that should run for a given event and optional matcher value."""
        event_name = event.value
        if event_name not in self.hooks:
            return []

        if event_name in ALWAYS_MATCHER_EVENTS:
            result: List[HookDefinition] = []
            for matcher in self.hooks[event_name]:
                result.extend(matcher.hooks)
            return result

        result = []
        for matcher in self.hooks[event_name]:
            if matcher.matches(matcher_value):
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
                if hook_type not in ("command", "prompt", "agent"):
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
                        run_async=bool(hook_data.get("async", False)),
                        run_once=bool(hook_data.get("once", False)),
                        hook_id=hook_data.get("id"),
                        status_message=hook_data.get("statusMessage"),
                    )
                # For prompt hooks, require prompt field
                elif hook_type == "prompt":
                    if "prompt" not in hook_data:
                        continue
                    if hook_data.get("async"):
                        logger.warning(
                            "Async hooks are only supported for type='command'; ignoring async flag."
                        )
                    hook_def = HookDefinition(
                        type="prompt",
                        prompt=hook_data["prompt"],
                        timeout=hook_data.get("timeout", DEFAULT_HOOK_TIMEOUT),
                        run_once=bool(hook_data.get("once", False)),
                        hook_id=hook_data.get("id"),
                        status_message=hook_data.get("statusMessage"),
                    )
                # For agent hooks, require prompt field and validate event
                elif hook_type == "agent":
                    if "prompt" not in hook_data:
                        continue
                    if hook_data.get("async"):
                        logger.warning(
                            "Async hooks are only supported for type='command'; ignoring async flag."
                        )
                    if hook_data.get("once"):
                        logger.warning("once is not supported for agent hooks; ignoring.")
                    hook_def = HookDefinition(
                        type="agent",
                        prompt=hook_data["prompt"],
                        timeout=hook_data.get("timeout", DEFAULT_HOOK_TIMEOUT),
                        model=hook_data.get("model"),
                        run_once=False,
                        hook_id=hook_data.get("id"),
                        status_message=hook_data.get("statusMessage"),
                    )
                else:
                    continue

                hook_definitions.append(hook_def)

            if hook_definitions:
                parsed_matchers.append(HookMatcher(matcher=matcher_pattern, hooks=hook_definitions))

        if parsed_matchers:
            parsed_hooks[event_name] = parsed_matchers

    return HooksConfig(hooks=parsed_hooks)


def parse_hooks_config(data: Any, *, source: Optional[str] = None) -> HooksConfig:
    """Parse hooks configuration from in-memory data.

    Accepts either a full hooks payload (with optional "hooks" wrapper) or a
    mapping of event names to matcher lists.
    """
    if data is None:
        return HooksConfig()
    if not isinstance(data, dict):
        label = f" for {source}" if source else ""
        logger.warning(f"Invalid hooks config{label}: expected mapping")
        return HooksConfig()
    try:
        return _parse_hooks_file(data)
    except Exception as exc:  # pragma: no cover - defensive
        label = f" for {source}" if source else ""
        logger.warning(
            f"Failed to parse hooks config{label}: {type(exc).__name__}: {exc}"
        )
        return HooksConfig()


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
    """Get the path to the user-level hooks configuration."""
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
    1. User config (~/.ripperdoc/hooks.json)
    2. Project config (.ripperdoc/hooks.json)
    3. Local project config (.ripperdoc/hooks.local.json)
    """
    # Start with user config
    config = load_hooks_config(get_global_hooks_path())

    # Merge project config if available
    if project_path:
        project_config = load_hooks_config(get_project_hooks_path(project_path))
        config = config.merge_with(project_config)

        # Merge local config
        local_config = load_hooks_config(get_project_local_hooks_path(project_path))
        config = config.merge_with(local_config)

    return config
