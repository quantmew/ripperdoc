"""Hook event types and data structures.

This module defines the event types that can trigger hooks,
as well as the input/output data structures for each event type.
"""

import json
from enum import Enum
from typing import Any, Dict, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field


class HookEvent(str, Enum):
    """Hook event types that can trigger user-defined hooks."""

    # Tool lifecycle events
    PRE_TOOL_USE = "PreToolUse"
    PERMISSION_REQUEST = "PermissionRequest"
    POST_TOOL_USE = "PostToolUse"

    # User interaction events
    USER_PROMPT_SUBMIT = "UserPromptSubmit"
    NOTIFICATION = "Notification"

    # Completion events
    STOP = "Stop"
    SUBAGENT_STOP = "SubagentStop"

    # Session lifecycle events
    PRE_COMPACT = "PreCompact"
    SESSION_START = "SessionStart"
    SESSION_END = "SessionEnd"


class HookDecision(str, Enum):
    """Decision values that hooks can return to control flow.

    PreToolUse/PermissionRequest decisions:
    - allow: Bypass permission system, auto-approve the tool call
    - deny: Block the tool call, inform the model
    - ask: Prompt user for confirmation (PreToolUse only)

    PostToolUse decisions:
    - block: Auto-prompt the model about the issue

    UserPromptSubmit decisions:
    - block: Reject the prompt, show reason to user

    Stop/SubagentStop decisions:
    - block: Prevent stopping, reason tells model how to continue
    """

    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"
    BLOCK = "block"

    # Legacy aliases (deprecated but supported)
    APPROVE = "approve"  # Use 'allow' instead


class HookInput(BaseModel):
    """Base class for hook input data.

    Common fields present in all hook inputs.
    """

    # Common fields for all hooks
    session_id: Optional[str] = None
    transcript_path: Optional[str] = None  # Path to conversation JSON
    cwd: Optional[str] = None  # Current working directory
    permission_mode: str = "default"  # "default", "plan", "acceptEdits", "bypassPermissions"
    hook_event_name: str = ""

    model_config = ConfigDict(populate_by_name=True)


class PreToolUseInput(HookInput):
    """Input data for PreToolUse hooks.

    Runs after Agent creates tool parameters and before processing the tool call.
    """

    hook_event_name: str = "PreToolUse"
    tool_name: str = ""
    tool_input: Dict[str, Any] = Field(default_factory=dict)
    tool_use_id: Optional[str] = None


class PermissionRequestInput(HookInput):
    """Input data for PermissionRequest hooks.

    Runs when the user is shown a permission dialog.
    """

    hook_event_name: str = "PermissionRequest"
    tool_name: str = ""
    tool_input: Dict[str, Any] = Field(default_factory=dict)
    tool_use_id: Optional[str] = None


class PostToolUseInput(HookInput):
    """Input data for PostToolUse hooks.

    Runs immediately after a tool completes successfully.
    """

    hook_event_name: str = "PostToolUse"
    tool_name: str = ""
    tool_input: Dict[str, Any] = Field(default_factory=dict)
    tool_response: Any = None  # Tool's response/output
    tool_use_id: Optional[str] = None


class UserPromptSubmitInput(HookInput):
    """Input data for UserPromptSubmit hooks.

    Runs when the user submits a prompt, before Agent processes it.
    """

    hook_event_name: str = "UserPromptSubmit"
    prompt: str = ""


class NotificationInput(HookInput):
    """Input data for Notification hooks.

    Runs when Ripperdoc sends notifications.

    notification_type values:
    - permission_prompt: Permission requests
    - idle_prompt: Waiting for user input (after 60+ seconds idle)
    - auth_success: Authentication success
    - elicitation_dialog: MCP tool elicitation
    """

    hook_event_name: str = "Notification"
    message: str = ""
    notification_type: str = ""  # Used as matcher


class StopInput(HookInput):
    """Input data for Stop hooks.

    Runs when the main agent has finished responding.
    Does not run if stoppage occurred due to user interrupt.
    """

    hook_event_name: str = "Stop"
    stop_hook_active: bool = False  # True if already continuing from a stop hook
    reason: Optional[str] = None
    stop_sequence: Optional[str] = None


class SubagentStopInput(HookInput):
    """Input data for SubagentStop hooks.

    Runs when a subagent (Task tool call) has finished responding.
    """

    hook_event_name: str = "SubagentStop"
    stop_hook_active: bool = False


class PreCompactInput(HookInput):
    """Input data for PreCompact hooks.

    Runs before a compact operation.

    trigger values:
    - manual: Invoked from /compact
    - auto: Invoked from auto-compact (due to full context window)
    """

    hook_event_name: str = "PreCompact"
    trigger: str = ""  # "manual" or "auto"
    custom_instructions: str = ""  # Custom instructions passed to /compact


class SessionStartInput(HookInput):
    """Input data for SessionStart hooks.

    Runs when a session starts or resumes.

    source values:
    - startup: Fresh start
    - resume: From --resume, --continue, or /resume
    - clear: From /clear
    - compact: From auto or manual compact

    SessionStart hooks have access to RIPPERDOC_ENV_FILE environment variable,
    which provides a file path where environment variables can be persisted.
    """

    hook_event_name: str = "SessionStart"
    source: str = ""  # "startup", "resume", "clear", "compact"


class SessionEndInput(HookInput):
    """Input data for SessionEnd hooks.

    Runs when a session ends.

    reason values:
    - clear: Session cleared with /clear command
    - logout: User logged out
    - prompt_input_exit: User exited while prompt input was visible
    - other: Other exit reasons
    """

    hook_event_name: str = "SessionEnd"
    reason: str = ""  # "clear", "logout", "prompt_input_exit", "other"
    duration_seconds: Optional[float] = None
    message_count: Optional[int] = None


# ─────────────────────────────────────────────────────────────────────────────
# Hook Output Types
# ─────────────────────────────────────────────────────────────────────────────


class PreToolUseHookOutput(BaseModel):
    """Hook-specific output for PreToolUse."""

    hook_event_name: Literal["PreToolUse"] = "PreToolUse"
    permission_decision: Optional[str] = Field(
        default=None, alias="permissionDecision"
    )  # "allow", "deny", "ask"
    permission_decision_reason: Optional[str] = Field(
        default=None, alias="permissionDecisionReason"
    )
    updated_input: Optional[Dict[str, Any]] = Field(
        default=None, alias="updatedInput"
    )  # Modified tool input
    additional_context: Optional[str] = Field(default=None, alias="additionalContext")

    model_config = ConfigDict(populate_by_name=True)


class PermissionRequestDecision(BaseModel):
    """Decision object for PermissionRequest hooks."""

    behavior: str = ""  # "allow" or "deny"
    updated_input: Optional[Dict[str, Any]] = Field(default=None, alias="updatedInput")
    message: Optional[str] = None
    interrupt: bool = False

    model_config = ConfigDict(populate_by_name=True)


class PermissionRequestHookOutput(BaseModel):
    """Hook-specific output for PermissionRequest."""

    hook_event_name: Literal["PermissionRequest"] = "PermissionRequest"
    decision: Optional[PermissionRequestDecision] = None

    model_config = ConfigDict(populate_by_name=True)


class PostToolUseHookOutput(BaseModel):
    """Hook-specific output for PostToolUse."""

    hook_event_name: Literal["PostToolUse"] = "PostToolUse"
    additional_context: Optional[str] = Field(default=None, alias="additionalContext")

    model_config = ConfigDict(populate_by_name=True)


class UserPromptSubmitHookOutput(BaseModel):
    """Hook-specific output for UserPromptSubmit."""

    hook_event_name: Literal["UserPromptSubmit"] = "UserPromptSubmit"
    additional_context: Optional[str] = Field(default=None, alias="additionalContext")

    model_config = ConfigDict(populate_by_name=True)


class SessionStartHookOutput(BaseModel):
    """Hook-specific output for SessionStart."""

    hook_event_name: Literal["SessionStart"] = "SessionStart"
    additional_context: Optional[str] = Field(default=None, alias="additionalContext")

    model_config = ConfigDict(populate_by_name=True)


HookSpecificOutput = Union[
    PreToolUseHookOutput,
    PermissionRequestHookOutput,
    PostToolUseHookOutput,
    UserPromptSubmitHookOutput,
    SessionStartHookOutput,
    Dict[str, Any],  # Fallback for unknown types
]


class HookOutput(BaseModel):
    """Output from a hook execution.

    Hooks can output:
    - Plain text (treated as additional context or info)
    - JSON with decision control

    JSON output is only processed when hook exits with code 0.
    Exit code 2 uses stderr directly as the error message.
    """

    # Common JSON fields
    continue_execution: bool = Field(default=True, alias="continue")
    stop_reason: Optional[str] = Field(default=None, alias="stopReason")
    suppress_output: bool = Field(default=False, alias="suppressOutput")
    system_message: Optional[str] = Field(default=None, alias="systemMessage")

    # Decision control (for backwards compatibility)
    decision: Optional[HookDecision] = None
    reason: Optional[str] = None

    # Hook-specific output
    hook_specific_output: Optional[HookSpecificOutput] = Field(
        default=None, alias="hookSpecificOutput"
    )

    # Additional context to inject
    additional_context: Optional[str] = Field(default=None, alias="additionalContext")

    # Raw output (for non-JSON responses)
    raw_output: Optional[str] = None

    # Error info
    error: Optional[str] = None
    stderr: Optional[str] = None
    exit_code: int = 0
    timed_out: bool = False

    model_config = ConfigDict(populate_by_name=True)

    @classmethod
    def from_raw(
        cls, stdout: str, stderr: str, exit_code: int, timed_out: bool = False
    ) -> "HookOutput":
        """Parse hook output from raw command output.

        Exit code behavior:
        - 0: Success. stdout processed, JSON parsed if present.
        - 2: Blocking error. stderr is used as error message, JSON ignored.
        - Other: Non-blocking error. stderr shown to user.
        """
        output = cls(exit_code=exit_code, stderr=stderr, timed_out=timed_out)

        if timed_out:
            output.error = "Hook timed out"
            return output

        # Exit code 2: Blocking error - use stderr directly
        if exit_code == 2:
            output.decision = HookDecision.DENY
            output.reason = stderr.strip() if stderr.strip() else "Blocked by hook"
            output.error = stderr.strip() if stderr.strip() else None
            return output

        # Other non-zero exit codes: non-blocking error
        if exit_code != 0:
            output.error = stderr or f"Hook exited with code {exit_code}"
            return output

        # Exit code 0: Process stdout
        stdout = stdout.strip()
        if not stdout:
            return output

        # Try to parse as JSON
        try:
            data = json.loads(stdout)
            if isinstance(data, dict):
                output = cls._parse_json_output(data, stderr)
                return output
        except json.JSONDecodeError:
            pass

        # Not JSON, treat as raw output / additional context
        output.raw_output = stdout
        output.additional_context = stdout
        return output

    @classmethod
    def _parse_json_output(cls, data: Dict[str, Any], stderr: str) -> "HookOutput":
        """Parse JSON output from hook."""
        output = cls(stderr=stderr)

        # Common fields
        output.continue_execution = data.get("continue", True)
        output.stop_reason = data.get("stopReason")
        output.suppress_output = data.get("suppressOutput", False)
        output.system_message = data.get("systemMessage")

        # Legacy decision field (backwards compatibility)
        if "decision" in data:
            decision_str = str(data["decision"]).lower()
            # Handle legacy aliases
            if decision_str == "approve":
                decision_str = "allow"
            try:
                output.decision = HookDecision(decision_str)
            except ValueError:
                pass

        output.reason = data.get("reason")

        # Parse hook-specific output
        if "hookSpecificOutput" in data:
            hso = data["hookSpecificOutput"]
            if isinstance(hso, dict):
                event_name = hso.get("hookEventName")

                # Handle PreToolUse specific fields
                if event_name == "PreToolUse":
                    output.hook_specific_output = PreToolUseHookOutput(
                        permissionDecision=hso.get("permissionDecision"),
                        permissionDecisionReason=hso.get("permissionDecisionReason"),
                        updatedInput=hso.get("updatedInput"),
                        additionalContext=hso.get("additionalContext"),
                    )
                    # Map permissionDecision to decision
                    perm_decision = hso.get("permissionDecision")
                    if perm_decision:
                        perm_decision = perm_decision.lower()
                        if perm_decision == "approve":
                            perm_decision = "allow"
                        try:
                            output.decision = HookDecision(perm_decision)
                        except ValueError:
                            pass
                        output.reason = hso.get("permissionDecisionReason")

                # Handle PermissionRequest specific fields
                elif event_name == "PermissionRequest":
                    decision_obj = hso.get("decision", {})
                    decision_data = None
                    if isinstance(decision_obj, dict):
                        decision_data = PermissionRequestDecision(
                            behavior=decision_obj.get("behavior", ""),
                            updatedInput=decision_obj.get("updatedInput"),
                            message=decision_obj.get("message"),
                            interrupt=decision_obj.get("interrupt", False),
                        )
                        behavior = decision_obj.get("behavior")
                        if behavior == "allow":
                            output.decision = HookDecision.ALLOW
                        elif behavior == "deny":
                            output.decision = HookDecision.DENY
                        if decision_obj.get("message"):
                            output.reason = decision_obj.get("message")
                    output.hook_specific_output = PermissionRequestHookOutput(
                        decision=decision_data,
                    )

                # Handle PostToolUse specific fields
                elif event_name == "PostToolUse":
                    output.hook_specific_output = PostToolUseHookOutput(
                        additionalContext=hso.get("additionalContext"),
                    )
                    if hso.get("additionalContext"):
                        output.additional_context = hso["additionalContext"]

                # Handle UserPromptSubmit specific fields
                elif event_name == "UserPromptSubmit":
                    output.hook_specific_output = UserPromptSubmitHookOutput(
                        additionalContext=hso.get("additionalContext"),
                    )
                    if hso.get("additionalContext"):
                        output.additional_context = hso["additionalContext"]

                # Handle SessionStart specific fields
                elif event_name == "SessionStart":
                    output.hook_specific_output = SessionStartHookOutput(
                        additionalContext=hso.get("additionalContext"),
                    )
                    if hso.get("additionalContext"):
                        output.additional_context = hso["additionalContext"]

                # Fallback for unknown types
                else:
                    output.hook_specific_output = hso
                    if hso.get("additionalContext"):
                        output.additional_context = hso["additionalContext"]

        # Direct additional context
        if "additionalContext" in data:
            output.additional_context = data["additionalContext"]

        return output

    @property
    def should_block(self) -> bool:
        """Check if hook requests blocking."""
        return self.decision in (HookDecision.DENY, HookDecision.BLOCK)

    @property
    def should_allow(self) -> bool:
        """Check if hook requests allowing."""
        return self.decision in (HookDecision.ALLOW, HookDecision.APPROVE)

    @property
    def should_ask(self) -> bool:
        """Check if hook requests user confirmation."""
        return self.decision == HookDecision.ASK

    @property
    def should_continue(self) -> bool:
        """Check if execution should continue."""
        return self.continue_execution

    @property
    def updated_input(self) -> Optional[Dict[str, Any]]:
        """Get updated input from PreToolUse hook."""
        if isinstance(self.hook_specific_output, PreToolUseHookOutput):
            return self.hook_specific_output.updated_input
        if isinstance(self.hook_specific_output, dict):
            return self.hook_specific_output.get("updatedInput")
        return None


# Type alias for any hook input
AnyHookInput = Union[
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
]
