"""Hook manager for coordinating hook execution.

This module provides the main interface for triggering hooks
throughout the application lifecycle.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ripperdoc.core.hooks.config import (
    HooksConfig,
    HookDefinition,
    get_merged_hooks_config,
)
from ripperdoc.core.hooks.events import (
    HookEvent,
    HookDecision,
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
)
from ripperdoc.core.hooks.executor import HookExecutor, LLMCallback
from ripperdoc.utils.log import get_logger

logger = get_logger()


class HookResult:
    """Result of running hooks for an event.

    Aggregates results from all hooks and provides convenience methods
    for checking the overall decision.
    """

    def __init__(self, outputs: List[HookOutput]):
        self.outputs = outputs

    @property
    def should_block(self) -> bool:
        """Check if any hook returned a blocking decision."""
        return any(o.decision in (HookDecision.DENY, HookDecision.BLOCK) for o in self.outputs)

    @property
    def should_allow(self) -> bool:
        """Check if any hook returned an allow decision."""
        return any(o.decision == HookDecision.ALLOW for o in self.outputs)

    @property
    def should_ask(self) -> bool:
        """Check if any hook returned an ask decision."""
        return any(o.decision == HookDecision.ASK for o in self.outputs)

    @property
    def should_continue(self) -> bool:
        """Check if execution should continue (no hook set continue=false)."""
        return all(o.should_continue for o in self.outputs)

    @property
    def block_reason(self) -> Optional[str]:
        """Get the reason for blocking, if any."""
        for o in self.outputs:
            if o.decision in (HookDecision.DENY, HookDecision.BLOCK) and o.reason:
                return o.reason
        return None

    @property
    def stop_reason(self) -> Optional[str]:
        """Get the stop reason from hooks, if any."""
        for o in self.outputs:
            if o.stop_reason:
                return o.stop_reason
        return None

    @property
    def additional_context(self) -> Optional[str]:
        """Get combined additional context from all hooks."""
        contexts = []
        for o in self.outputs:
            if o.additional_context:
                contexts.append(o.additional_context)
        return "\n".join(contexts) if contexts else None

    @property
    def system_message(self) -> Optional[str]:
        """Get system message from hooks, if any."""
        for o in self.outputs:
            if o.system_message:
                return o.system_message
        return None

    @property
    def updated_input(self) -> Optional[Dict[str, Any]]:
        """Get updated tool input from PreToolUse hooks, if any."""
        for o in self.outputs:
            if o.updated_input:
                return o.updated_input
        return None

    @property
    def has_errors(self) -> bool:
        """Check if any hook had an error."""
        return any(o.error for o in self.outputs)

    @property
    def errors(self) -> List[str]:
        """Get all error messages."""
        return [o.error for o in self.outputs if o.error]


class HookManager:
    """Manages hook configuration and execution.

    This is the main interface for triggering hooks in the application.
    It loads configuration, finds matching hooks, and executes them.
    """

    def __init__(
        self,
        project_dir: Optional[Path] = None,
        session_id: Optional[str] = None,
        transcript_path: Optional[str] = None,
        permission_mode: str = "default",
        llm_callback: Optional[LLMCallback] = None,
    ):
        """Initialize the hook manager.

        Args:
            project_dir: The project directory
            session_id: Current session ID for hook input
            transcript_path: Path to the conversation transcript JSON
            permission_mode: Current permission mode (default, plan, acceptEdits, bypassPermissions)
            llm_callback: Async callback for prompt-based hooks. Takes prompt string,
                         returns LLM response string. If not set, prompt hooks will
                         be skipped with a warning.
        """
        self.project_dir = project_dir
        self.session_id = session_id
        self.transcript_path = transcript_path
        self.permission_mode = permission_mode
        self.llm_callback = llm_callback
        self._config: Optional[HooksConfig] = None
        self._executor: Optional[HookExecutor] = None

    @property
    def config(self) -> HooksConfig:
        """Get the hooks configuration (lazy loaded)."""
        if self._config is None:
            self._config = get_merged_hooks_config(self.project_dir)
        return self._config

    @property
    def executor(self) -> HookExecutor:
        """Get the hook executor (lazy created)."""
        if self._executor is None:
            self._executor = HookExecutor(
                project_dir=self.project_dir,
                session_id=self.session_id,
                transcript_path=self.transcript_path,
                llm_callback=self.llm_callback,
            )
        return self._executor

    def reload_config(self) -> None:
        """Reload hooks configuration from files."""
        self._config = None
        logger.debug("Hooks configuration will be reloaded on next access")

    def set_project_dir(self, project_dir: Optional[Path]) -> None:
        """Update the project directory and reload config."""
        self.project_dir = project_dir
        self._config = None
        self._executor = None

    def set_session_id(self, session_id: Optional[str]) -> None:
        """Update the session ID."""
        self.session_id = session_id
        if self._executor:
            self._executor.session_id = session_id

    def set_transcript_path(self, transcript_path: Optional[str]) -> None:
        """Update the transcript path."""
        self.transcript_path = transcript_path
        if self._executor:
            self._executor.transcript_path = transcript_path

    def set_permission_mode(self, mode: str) -> None:
        """Update the permission mode."""
        self.permission_mode = mode

    def set_llm_callback(self, callback: Optional[LLMCallback]) -> None:
        """Update the LLM callback for prompt hooks."""
        self.llm_callback = callback
        if self._executor:
            self._executor.llm_callback = callback

    def _get_cwd(self) -> Optional[str]:
        """Get current working directory."""
        try:
            return os.getcwd()
        except OSError:
            return str(self.project_dir) if self.project_dir else None

    def _get_hooks(self, event: HookEvent, tool_name: Optional[str] = None) -> List[HookDefinition]:
        """Get hooks that should run for an event."""
        return self.config.get_hooks_for_event(event, tool_name)

    def cleanup(self) -> None:
        """Clean up resources (call on session end)."""
        if self._executor:
            self._executor.cleanup_env_file()

    # --- Pre Tool Use ---

    def run_pre_tool_use(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_use_id: Optional[str] = None,
    ) -> HookResult:
        """Run PreToolUse hooks synchronously.

        Args:
            tool_name: Name of the tool being called
            tool_input: Input parameters for the tool
            tool_use_id: Unique ID for this tool use

        Returns:
            HookResult with decision information
        """
        hooks = self._get_hooks(HookEvent.PRE_TOOL_USE, tool_name)
        if not hooks:
            return HookResult([])

        input_data = PreToolUseInput(
            tool_name=tool_name,
            tool_input=tool_input,
            tool_use_id=tool_use_id,
            session_id=self.session_id,
            transcript_path=self.transcript_path,
            cwd=self._get_cwd(),
            permission_mode=self.permission_mode,
        )

        outputs = self.executor.execute_hooks_sync(hooks, input_data)
        return HookResult(outputs)

    async def run_pre_tool_use_async(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_use_id: Optional[str] = None,
    ) -> HookResult:
        """Run PreToolUse hooks asynchronously."""
        hooks = self._get_hooks(HookEvent.PRE_TOOL_USE, tool_name)
        if not hooks:
            return HookResult([])

        input_data = PreToolUseInput(
            tool_name=tool_name,
            tool_input=tool_input,
            tool_use_id=tool_use_id,
            session_id=self.session_id,
            transcript_path=self.transcript_path,
            cwd=self._get_cwd(),
            permission_mode=self.permission_mode,
        )

        outputs = await self.executor.execute_hooks_async(hooks, input_data)
        return HookResult(outputs)

    # --- Permission Request ---

    def run_permission_request(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_use_id: Optional[str] = None,
    ) -> HookResult:
        """Run PermissionRequest hooks synchronously.

        Args:
            tool_name: Name of the tool requesting permission
            tool_input: Input parameters for the tool
            tool_use_id: Unique ID for this tool use

        Returns:
            HookResult with decision information
        """
        hooks = self._get_hooks(HookEvent.PERMISSION_REQUEST, tool_name)
        if not hooks:
            return HookResult([])

        input_data = PermissionRequestInput(
            tool_name=tool_name,
            tool_input=tool_input,
            tool_use_id=tool_use_id,
            session_id=self.session_id,
            transcript_path=self.transcript_path,
            cwd=self._get_cwd(),
            permission_mode=self.permission_mode,
        )

        outputs = self.executor.execute_hooks_sync(hooks, input_data)
        return HookResult(outputs)

    async def run_permission_request_async(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_use_id: Optional[str] = None,
    ) -> HookResult:
        """Run PermissionRequest hooks asynchronously."""
        hooks = self._get_hooks(HookEvent.PERMISSION_REQUEST, tool_name)
        if not hooks:
            return HookResult([])

        input_data = PermissionRequestInput(
            tool_name=tool_name,
            tool_input=tool_input,
            tool_use_id=tool_use_id,
            session_id=self.session_id,
            transcript_path=self.transcript_path,
            cwd=self._get_cwd(),
            permission_mode=self.permission_mode,
        )

        outputs = await self.executor.execute_hooks_async(hooks, input_data)
        return HookResult(outputs)

    # --- Post Tool Use ---

    def run_post_tool_use(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_response: Any = None,
        tool_use_id: Optional[str] = None,
    ) -> HookResult:
        """Run PostToolUse hooks synchronously."""
        hooks = self._get_hooks(HookEvent.POST_TOOL_USE, tool_name)
        if not hooks:
            return HookResult([])

        input_data = PostToolUseInput(
            tool_name=tool_name,
            tool_input=tool_input,
            tool_response=tool_response,
            tool_use_id=tool_use_id,
            session_id=self.session_id,
            transcript_path=self.transcript_path,
            cwd=self._get_cwd(),
            permission_mode=self.permission_mode,
        )

        outputs = self.executor.execute_hooks_sync(hooks, input_data)
        return HookResult(outputs)

    async def run_post_tool_use_async(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_response: Any = None,
        tool_use_id: Optional[str] = None,
    ) -> HookResult:
        """Run PostToolUse hooks asynchronously."""
        hooks = self._get_hooks(HookEvent.POST_TOOL_USE, tool_name)
        if not hooks:
            return HookResult([])

        input_data = PostToolUseInput(
            tool_name=tool_name,
            tool_input=tool_input,
            tool_response=tool_response,
            tool_use_id=tool_use_id,
            session_id=self.session_id,
            transcript_path=self.transcript_path,
            cwd=self._get_cwd(),
            permission_mode=self.permission_mode,
        )

        outputs = await self.executor.execute_hooks_async(hooks, input_data)
        return HookResult(outputs)

    # --- User Prompt Submit ---

    def run_user_prompt_submit(self, prompt: str) -> HookResult:
        """Run UserPromptSubmit hooks synchronously."""
        hooks = self._get_hooks(HookEvent.USER_PROMPT_SUBMIT)
        if not hooks:
            return HookResult([])

        input_data = UserPromptSubmitInput(
            prompt=prompt,
            session_id=self.session_id,
            transcript_path=self.transcript_path,
            cwd=self._get_cwd(),
            permission_mode=self.permission_mode,
        )

        outputs = self.executor.execute_hooks_sync(hooks, input_data)
        return HookResult(outputs)

    async def run_user_prompt_submit_async(self, prompt: str) -> HookResult:
        """Run UserPromptSubmit hooks asynchronously."""
        hooks = self._get_hooks(HookEvent.USER_PROMPT_SUBMIT)
        if not hooks:
            return HookResult([])

        input_data = UserPromptSubmitInput(
            prompt=prompt,
            session_id=self.session_id,
            transcript_path=self.transcript_path,
            cwd=self._get_cwd(),
            permission_mode=self.permission_mode,
        )

        outputs = await self.executor.execute_hooks_async(hooks, input_data)
        return HookResult(outputs)

    # --- Notification ---

    def run_notification(self, message: str, notification_type: str = "info") -> HookResult:
        """Run Notification hooks synchronously.

        Args:
            message: The notification message
            notification_type: Type of notification (permission_prompt, idle_prompt, auth_success, elicitation_dialog)
        """
        hooks = self._get_hooks(HookEvent.NOTIFICATION)
        if not hooks:
            return HookResult([])

        input_data = NotificationInput(
            message=message,
            notification_type=notification_type,
            session_id=self.session_id,
            transcript_path=self.transcript_path,
            cwd=self._get_cwd(),
            permission_mode=self.permission_mode,
        )

        outputs = self.executor.execute_hooks_sync(hooks, input_data)
        return HookResult(outputs)

    async def run_notification_async(
        self, message: str, notification_type: str = "info"
    ) -> HookResult:
        """Run Notification hooks asynchronously."""
        hooks = self._get_hooks(HookEvent.NOTIFICATION)
        if not hooks:
            return HookResult([])

        input_data = NotificationInput(
            message=message,
            notification_type=notification_type,
            session_id=self.session_id,
            transcript_path=self.transcript_path,
            cwd=self._get_cwd(),
            permission_mode=self.permission_mode,
        )

        outputs = await self.executor.execute_hooks_async(hooks, input_data)
        return HookResult(outputs)

    # --- Stop ---

    def run_stop(
        self,
        stop_hook_active: bool = False,
        reason: Optional[str] = None,
        stop_sequence: Optional[str] = None,
    ) -> HookResult:
        """Run Stop hooks synchronously.

        Args:
            stop_hook_active: True if already continuing from a stop hook
            reason: Reason for stopping
            stop_sequence: Stop sequence that triggered the stop
        """
        hooks = self._get_hooks(HookEvent.STOP)
        if not hooks:
            return HookResult([])

        input_data = StopInput(
            stop_hook_active=stop_hook_active,
            reason=reason,
            stop_sequence=stop_sequence,
            session_id=self.session_id,
            transcript_path=self.transcript_path,
            cwd=self._get_cwd(),
            permission_mode=self.permission_mode,
        )

        outputs = self.executor.execute_hooks_sync(hooks, input_data)
        return HookResult(outputs)

    async def run_stop_async(
        self,
        stop_hook_active: bool = False,
        reason: Optional[str] = None,
        stop_sequence: Optional[str] = None,
    ) -> HookResult:
        """Run Stop hooks asynchronously."""
        hooks = self._get_hooks(HookEvent.STOP)
        if not hooks:
            return HookResult([])

        input_data = StopInput(
            stop_hook_active=stop_hook_active,
            reason=reason,
            stop_sequence=stop_sequence,
            session_id=self.session_id,
            transcript_path=self.transcript_path,
            cwd=self._get_cwd(),
            permission_mode=self.permission_mode,
        )

        outputs = await self.executor.execute_hooks_async(hooks, input_data)
        return HookResult(outputs)

    # --- Subagent Stop ---

    def run_subagent_stop(self, stop_hook_active: bool = False) -> HookResult:
        """Run SubagentStop hooks synchronously.

        Args:
            stop_hook_active: True if already continuing from a stop hook
        """
        hooks = self._get_hooks(HookEvent.SUBAGENT_STOP)
        if not hooks:
            return HookResult([])

        input_data = SubagentStopInput(
            stop_hook_active=stop_hook_active,
            session_id=self.session_id,
            transcript_path=self.transcript_path,
            cwd=self._get_cwd(),
            permission_mode=self.permission_mode,
        )

        outputs = self.executor.execute_hooks_sync(hooks, input_data)
        return HookResult(outputs)

    async def run_subagent_stop_async(self, stop_hook_active: bool = False) -> HookResult:
        """Run SubagentStop hooks asynchronously."""
        hooks = self._get_hooks(HookEvent.SUBAGENT_STOP)
        if not hooks:
            return HookResult([])

        input_data = SubagentStopInput(
            stop_hook_active=stop_hook_active,
            session_id=self.session_id,
            transcript_path=self.transcript_path,
            cwd=self._get_cwd(),
            permission_mode=self.permission_mode,
        )

        outputs = await self.executor.execute_hooks_async(hooks, input_data)
        return HookResult(outputs)

    # --- Pre Compact ---

    def run_pre_compact(self, trigger: str, custom_instructions: str = "") -> HookResult:
        """Run PreCompact hooks synchronously.

        Args:
            trigger: "manual" or "auto"
            custom_instructions: Custom instructions passed to /compact
        """
        hooks = self._get_hooks(HookEvent.PRE_COMPACT)
        if not hooks:
            return HookResult([])

        input_data = PreCompactInput(
            trigger=trigger,
            custom_instructions=custom_instructions,
            session_id=self.session_id,
            transcript_path=self.transcript_path,
            cwd=self._get_cwd(),
            permission_mode=self.permission_mode,
        )

        outputs = self.executor.execute_hooks_sync(hooks, input_data)
        return HookResult(outputs)

    async def run_pre_compact_async(
        self, trigger: str, custom_instructions: str = ""
    ) -> HookResult:
        """Run PreCompact hooks asynchronously."""
        hooks = self._get_hooks(HookEvent.PRE_COMPACT)
        if not hooks:
            return HookResult([])

        input_data = PreCompactInput(
            trigger=trigger,
            custom_instructions=custom_instructions,
            session_id=self.session_id,
            transcript_path=self.transcript_path,
            cwd=self._get_cwd(),
            permission_mode=self.permission_mode,
        )

        outputs = await self.executor.execute_hooks_async(hooks, input_data)
        return HookResult(outputs)

    # --- Session Start ---

    def run_session_start(self, source: str) -> HookResult:
        """Run SessionStart hooks synchronously.

        Args:
            source: "startup", "resume", "clear", or "compact"
        """
        hooks = self._get_hooks(HookEvent.SESSION_START)
        if not hooks:
            return HookResult([])

        input_data = SessionStartInput(
            source=source,
            session_id=self.session_id,
            transcript_path=self.transcript_path,
            cwd=self._get_cwd(),
            permission_mode=self.permission_mode,
        )

        outputs = self.executor.execute_hooks_sync(hooks, input_data)
        return HookResult(outputs)

    async def run_session_start_async(self, source: str) -> HookResult:
        """Run SessionStart hooks asynchronously."""
        hooks = self._get_hooks(HookEvent.SESSION_START)
        if not hooks:
            return HookResult([])

        input_data = SessionStartInput(
            source=source,
            session_id=self.session_id,
            transcript_path=self.transcript_path,
            cwd=self._get_cwd(),
            permission_mode=self.permission_mode,
        )

        outputs = await self.executor.execute_hooks_async(hooks, input_data)
        return HookResult(outputs)

    # --- Session End ---

    def run_session_end(
        self,
        reason: str,
        duration_seconds: Optional[float] = None,
        message_count: Optional[int] = None,
    ) -> HookResult:
        """Run SessionEnd hooks synchronously.

        Args:
            reason: "clear", "logout", "prompt_input_exit", or "other"
            duration_seconds: How long the session lasted
            message_count: Number of messages in the session
        """
        hooks = self._get_hooks(HookEvent.SESSION_END)
        if not hooks:
            return HookResult([])

        input_data = SessionEndInput(
            reason=reason,
            duration_seconds=duration_seconds,
            message_count=message_count,
            session_id=self.session_id,
            transcript_path=self.transcript_path,
            cwd=self._get_cwd(),
            permission_mode=self.permission_mode,
        )

        outputs = self.executor.execute_hooks_sync(hooks, input_data)
        return HookResult(outputs)

    async def run_session_end_async(
        self,
        reason: str,
        duration_seconds: Optional[float] = None,
        message_count: Optional[int] = None,
    ) -> HookResult:
        """Run SessionEnd hooks asynchronously."""
        hooks = self._get_hooks(HookEvent.SESSION_END)
        if not hooks:
            return HookResult([])

        input_data = SessionEndInput(
            reason=reason,
            duration_seconds=duration_seconds,
            message_count=message_count,
            session_id=self.session_id,
            transcript_path=self.transcript_path,
            cwd=self._get_cwd(),
            permission_mode=self.permission_mode,
        )

        outputs = await self.executor.execute_hooks_async(hooks, input_data)
        return HookResult(outputs)


# Global instance for convenience
hook_manager = HookManager()


def get_hook_manager() -> HookManager:
    """Get the global hook manager instance."""
    return hook_manager


def init_hook_manager(
    project_dir: Optional[Path] = None,
    session_id: Optional[str] = None,
    transcript_path: Optional[str] = None,
    permission_mode: str = "default",
    llm_callback: Optional[LLMCallback] = None,
) -> HookManager:
    """Initialize the global hook manager with project context.

    Args:
        project_dir: The project directory
        session_id: Current session ID
        transcript_path: Path to the conversation transcript JSON
        permission_mode: Current permission mode
        llm_callback: Async callback for prompt-based hooks

    Returns:
        The initialized global hook manager
    """
    hook_manager.set_project_dir(project_dir)
    hook_manager.set_session_id(session_id)
    hook_manager.set_transcript_path(transcript_path)
    hook_manager.set_permission_mode(permission_mode)
    hook_manager.set_llm_callback(llm_callback)
    return hook_manager
