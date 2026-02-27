"""Hook manager for coordinating hook execution.

This module provides the main interface for triggering hooks
throughout the application lifecycle.
"""

import os
import asyncio
import hashlib
import json
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
    PostToolUseFailureInput,
    UserPromptSubmitInput,
    NotificationInput,
    StopInput,
    SubagentStartInput,
    SubagentStopInput,
    PreCompactInput,
    SessionStartInput,
    SessionEndInput,
    SetupInput,
)
from ripperdoc.core.hooks.executor import HookExecutor, LLMCallback, HookCallback
from ripperdoc.core.hooks.state import (
    hooks_suspended,
    get_hook_scopes,
    get_pending_message_queue,
    get_hook_status_emitter,
)
from ripperdoc.utils.messages import create_hook_notice_message, create_user_message
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
        """Check if hooks collectively allow execution.

        A deny/block decision always takes priority over allow.
        """
        has_allow = False
        for output in self.outputs:
            if output.decision in (HookDecision.DENY, HookDecision.BLOCK):
                return False
            if output.decision == HookDecision.ALLOW:
                has_allow = True
        return has_allow

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
    def updated_permissions(self) -> Optional[Any]:
        """Get updated permissions from PermissionRequest hooks, if any."""
        for o in self.outputs:
            if o.updated_permissions is not None:
                return o.updated_permissions
        return None

    @property
    def updated_mcp_tool_output(self) -> Optional[Any]:
        """Get updated MCP tool output from PostToolUse hooks, if any."""
        for o in self.outputs:
            if o.updated_mcp_tool_output is not None:
                return o.updated_mcp_tool_output
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
        hook_callback: Optional["HookCallback"] = None,
    ):
        """Initialize the hook manager.

        Args:
            project_dir: The project directory
            session_id: Current session ID for hook input
            transcript_path: Path to the conversation transcript JSON
            permission_mode: Current permission mode (default, plan, dontAsk, acceptEdits, bypassPermissions)
            llm_callback: Async callback for prompt-based hooks. Takes prompt string,
                         returns LLM response string. If not set, prompt hooks will
                         be skipped with a warning.
            hook_callback: Async callback for external hook execution (e.g., SDK hooks).
        """
        self.project_dir = project_dir
        self.session_id = session_id
        self.transcript_path = transcript_path
        self.permission_mode = permission_mode
        self.llm_callback = llm_callback
        self.hook_callback = hook_callback
        self._config: Optional[HooksConfig] = None
        self._executor: Optional[HookExecutor] = None
        self._setup_ran_for_project: Optional[Path] = None
        self._once_executed: set[str] = set()

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
                hook_callback=self.hook_callback,
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
        if session_id != self.session_id:
            self._once_executed.clear()
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

    def set_hook_callback(self, callback: Optional["HookCallback"]) -> None:
        """Update the external hook callback handler."""
        self.hook_callback = callback
        if self._executor:
            self._executor.hook_callback = callback

    def _run_setup_if_needed(self) -> None:
        if not self.project_dir or self.project_dir == self._setup_ran_for_project:
            return
        try:
            self.run_setup("init")
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(
                "[hook_manager] Setup hook failed: %s: %s",
                type(exc).__name__,
                exc,
                extra={"project_dir": str(self.project_dir)},
            )
        self._setup_ran_for_project = self.project_dir

    async def _run_setup_if_needed_async(self) -> None:
        if not self.project_dir or self.project_dir == self._setup_ran_for_project:
            return
        try:
            await self.run_setup_async("init")
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(
                "[hook_manager] Setup hook failed: %s: %s",
                type(exc).__name__,
                exc,
                extra={"project_dir": str(self.project_dir)},
            )
        self._setup_ran_for_project = self.project_dir

    def _get_cwd(self) -> Optional[str]:
        """Get current working directory."""
        try:
            return os.getcwd()
        except OSError:
            return str(self.project_dir) if self.project_dir else None

    def _get_hooks(
        self, event: HookEvent, matcher_value: Optional[str] = None
    ) -> List[HookDefinition]:
        """Get hooks that should run for an event."""
        if hooks_suspended():
            return []
        config = self.config
        scoped = get_hook_scopes()
        if scoped:
            for scope in scoped:
                config = config.merge_with(scope)
        hooks = config.get_hooks_for_event(event, matcher_value)
        return self._select_hooks_for_execution(event, hooks)

    def _hook_identity(self, event: HookEvent, hook: HookDefinition) -> str:
        if hook.hook_id:
            return f"id:{hook.hook_id}"
        payload = {
            "event": event.value,
            "type": hook.type,
            "command": hook.command,
            "prompt": hook.prompt,
            "model": hook.model,
            "timeout": hook.timeout,
            "async": hook.run_async,
            "once": hook.run_once,
            "statusMessage": hook.status_message,
            "callbackId": hook.callback_id,
        }
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _select_hooks_for_execution(
        self, event: HookEvent, hooks: List[HookDefinition]
    ) -> List[HookDefinition]:
        selected: List[HookDefinition] = []
        seen: set[str] = set()
        for hook in hooks:
            key = self._hook_identity(event, hook)
            if key in seen:
                continue
            seen.add(key)
            if hook.run_once:
                if key in self._once_executed:
                    continue
                # Mark once as soon as we schedule it to avoid concurrent re-entry.
                self._once_executed.add(key)
            selected.append(hook)
        return selected

    def _enqueue_async_hook_output(
        self,
        output: HookOutput,
        input_data: Any,
    ) -> None:
        queue = get_pending_message_queue()
        if queue is None:
            logger.debug("[hook_manager] Async hook output dropped: no pending message queue")
            return

        event = getattr(input_data, "hook_event_name", "Hook")
        detail = event
        tool_name = getattr(input_data, "tool_name", None)
        if tool_name:
            detail = f"{detail}:{tool_name}"
        metadata = {"source": "hook_async", "hook_event": event, "tool_name": tool_name}

        if output.system_message:
            try:
                notice = create_hook_notice_message(
                    str(output.system_message),
                    hook_event=event,
                    tool_name=tool_name,
                )
                queue.enqueue(notice)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "[hook_manager] Failed to enqueue async hook notice: %s: %s",
                    type(exc).__name__,
                    exc,
                )

        if output.additional_context:
            try:
                message = create_user_message(
                    f"[async hook {detail}]\n{output.additional_context}"
                )
                message.message.metadata.update(metadata)
                queue.enqueue(message)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "[hook_manager] Failed to enqueue async hook output: %s: %s",
                    type(exc).__name__,
                    exc,
                )

    def _start_async_command_hook(self, hook: HookDefinition, input_data: Any) -> None:
        async def _runner() -> None:
            try:
                output = await self.executor._execute_command_async(hook, input_data)
                self._enqueue_async_hook_output(output, input_data)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "[hook_manager] Async hook failed: %s: %s",
                    type(exc).__name__,
                    exc,
                )

        try:
            task = asyncio.create_task(_runner())
        except RuntimeError as exc:  # pragma: no cover - defensive
            logger.warning(
                "[hook_manager] Async hook requested but no running loop: %s", exc
            )
            return

        def _handle_task_done(t: asyncio.Task[None]) -> None:
            if t.cancelled():
                return
            exc = t.exception()
            if exc:
                logger.warning(
                    "[hook_manager] Async hook task error: %s: %s",
                    type(exc).__name__,
                    exc,
                )

        task.add_done_callback(_handle_task_done)

    async def _execute_hooks_async(
        self,
        hooks: List[HookDefinition],
        input_data: Any,
        *,
        allow_async_commands: bool = True,
    ) -> List[HookOutput]:
        results: List[HookOutput] = []
        for hook in hooks:
            self._emit_status(hook, input_data)
            if hook.is_command_hook() and hook.run_async and allow_async_commands:
                self._start_async_command_hook(hook, input_data)
                results.append(HookOutput())
                continue
            results.append(await self.executor.execute_async(hook, input_data))
        return results

    def _execute_hooks_sync(
        self,
        hooks: List[HookDefinition],
        input_data: Any,
    ) -> List[HookOutput]:
        results: List[HookOutput] = []
        for hook in hooks:
            self._emit_status(hook, input_data)
            results.append(self.executor.execute_sync(hook, input_data))
        return results

    def _emit_status(self, hook: HookDefinition, input_data: Any) -> None:
        if not hook.status_message:
            return
        emitter = get_hook_status_emitter()
        if emitter:
            try:
                emitter(str(hook.status_message))
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(
                    "[hook_manager] status emitter failed: %s: %s",
                    type(exc).__name__,
                    exc,
                )
        logger.info(
            "[hook_manager] %s",
            hook.status_message,
            extra={
                "hook_event": getattr(input_data, "hook_event_name", None),
                "hook_type": hook.type,
            },
        )

    def cleanup(self) -> None:
        """Clean up resources (call on session end)."""
        if self._executor:
            self._executor.cleanup_env_file()

    def _apply_session_env(self) -> None:
        """Load and apply environment variables produced by SessionStart hooks."""
        if not self._executor:
            return
        updates = self._executor.load_env_from_file()
        if not updates:
            return
        os.environ.update(updates)
        logger.info(
            "[hook_manager] Loaded SessionStart environment updates",
            extra={"count": len(updates)},
        )

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

        outputs = self._execute_hooks_sync(hooks, input_data)
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

        outputs = await self._execute_hooks_async(hooks, input_data)
        return HookResult(outputs)

    # --- Permission Request ---

    def run_permission_request(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_use_id: Optional[str] = None,
        permission_suggestions: Optional[List[Any]] = None,
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
            permission_suggestions=permission_suggestions,
            session_id=self.session_id,
            transcript_path=self.transcript_path,
            cwd=self._get_cwd(),
            permission_mode=self.permission_mode,
        )

        outputs = self._execute_hooks_sync(hooks, input_data)
        return HookResult(outputs)

    async def run_permission_request_async(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_use_id: Optional[str] = None,
        permission_suggestions: Optional[List[Any]] = None,
    ) -> HookResult:
        """Run PermissionRequest hooks asynchronously."""
        hooks = self._get_hooks(HookEvent.PERMISSION_REQUEST, tool_name)
        if not hooks:
            return HookResult([])

        input_data = PermissionRequestInput(
            tool_name=tool_name,
            tool_input=tool_input,
            tool_use_id=tool_use_id,
            permission_suggestions=permission_suggestions,
            session_id=self.session_id,
            transcript_path=self.transcript_path,
            cwd=self._get_cwd(),
            permission_mode=self.permission_mode,
        )

        outputs = await self._execute_hooks_async(hooks, input_data)
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

        outputs = self._execute_hooks_sync(hooks, input_data)
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

        outputs = await self._execute_hooks_async(hooks, input_data)
        return HookResult(outputs)

    # --- Post Tool Use Failure ---

    def run_post_tool_use_failure(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_response: Any = None,
        tool_error: Optional[str] = None,
        tool_use_id: Optional[str] = None,
        *,
        error: Optional[str] = None,
        is_interrupt: Optional[bool] = None,
    ) -> HookResult:
        """Run PostToolUseFailure hooks synchronously."""
        hooks = self._get_hooks(HookEvent.POST_TOOL_USE_FAILURE, tool_name)
        if not hooks:
            return HookResult([])

        error_value = error if error is not None else tool_error
        input_data = PostToolUseFailureInput(
            tool_name=tool_name,
            tool_input=tool_input,
            tool_response=tool_response,
            error=error_value,
            is_interrupt=is_interrupt,
            tool_use_id=tool_use_id,
            session_id=self.session_id,
            transcript_path=self.transcript_path,
            cwd=self._get_cwd(),
            permission_mode=self.permission_mode,
        )

        outputs = self._execute_hooks_sync(hooks, input_data)
        return HookResult(outputs)

    async def run_post_tool_use_failure_async(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_response: Any = None,
        tool_error: Optional[str] = None,
        tool_use_id: Optional[str] = None,
        *,
        error: Optional[str] = None,
        is_interrupt: Optional[bool] = None,
    ) -> HookResult:
        """Run PostToolUseFailure hooks asynchronously."""
        hooks = self._get_hooks(HookEvent.POST_TOOL_USE_FAILURE, tool_name)
        if not hooks:
            return HookResult([])

        error_value = error if error is not None else tool_error
        input_data = PostToolUseFailureInput(
            tool_name=tool_name,
            tool_input=tool_input,
            tool_response=tool_response,
            error=error_value,
            is_interrupt=is_interrupt,
            tool_use_id=tool_use_id,
            session_id=self.session_id,
            transcript_path=self.transcript_path,
            cwd=self._get_cwd(),
            permission_mode=self.permission_mode,
        )

        outputs = await self._execute_hooks_async(hooks, input_data)
        return HookResult(outputs)

    # --- User Prompt Submit ---

    def run_user_prompt_submit(self, prompt: str) -> HookResult:
        """Run UserPromptSubmit hooks synchronously."""
        hooks = self._get_hooks(HookEvent.USER_PROMPT_SUBMIT, prompt)
        if not hooks:
            return HookResult([])

        input_data = UserPromptSubmitInput(
            prompt=prompt,
            session_id=self.session_id,
            transcript_path=self.transcript_path,
            cwd=self._get_cwd(),
            permission_mode=self.permission_mode,
        )

        outputs = self._execute_hooks_sync(hooks, input_data)
        return HookResult(outputs)

    async def run_user_prompt_submit_async(self, prompt: str) -> HookResult:
        """Run UserPromptSubmit hooks asynchronously."""
        hooks = self._get_hooks(HookEvent.USER_PROMPT_SUBMIT, prompt)
        if not hooks:
            return HookResult([])

        input_data = UserPromptSubmitInput(
            prompt=prompt,
            session_id=self.session_id,
            transcript_path=self.transcript_path,
            cwd=self._get_cwd(),
            permission_mode=self.permission_mode,
        )

        outputs = await self._execute_hooks_async(hooks, input_data)
        return HookResult(outputs)

    # --- Notification ---

    def run_notification(
        self,
        message: str,
        notification_type: str = "info",
        title: Optional[str] = None,
    ) -> HookResult:
        """Run Notification hooks synchronously.

        Args:
            message: The notification message
            notification_type: Type of notification (permission_prompt, idle_prompt, auth_success, elicitation_dialog)
            title: Optional notification title
        """
        hooks = self._get_hooks(HookEvent.NOTIFICATION, notification_type)
        if not hooks:
            return HookResult([])

        input_data = NotificationInput(
            message=message,
            title=title,
            notification_type=notification_type,
            session_id=self.session_id,
            transcript_path=self.transcript_path,
            cwd=self._get_cwd(),
            permission_mode=self.permission_mode,
        )

        outputs = self._execute_hooks_sync(hooks, input_data)
        return HookResult(outputs)

    async def run_notification_async(
        self,
        message: str,
        notification_type: str = "info",
        title: Optional[str] = None,
    ) -> HookResult:
        """Run Notification hooks asynchronously."""
        hooks = self._get_hooks(HookEvent.NOTIFICATION, notification_type)
        if not hooks:
            return HookResult([])

        input_data = NotificationInput(
            message=message,
            title=title,
            notification_type=notification_type,
            session_id=self.session_id,
            transcript_path=self.transcript_path,
            cwd=self._get_cwd(),
            permission_mode=self.permission_mode,
        )

        outputs = await self._execute_hooks_async(hooks, input_data)
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
        hooks = self._get_hooks(HookEvent.STOP, reason)
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

        outputs = self._execute_hooks_sync(hooks, input_data)
        return HookResult(outputs)

    async def run_stop_async(
        self,
        stop_hook_active: bool = False,
        reason: Optional[str] = None,
        stop_sequence: Optional[str] = None,
    ) -> HookResult:
        """Run Stop hooks asynchronously."""
        logger.debug("[hook_manager] run_stop_async ENTER")
        hooks = self._get_hooks(HookEvent.STOP, reason)
        logger.debug(f"[hook_manager] run_stop_async: got {len(hooks)} hooks")
        if not hooks:
            logger.debug("[hook_manager] run_stop_async: no hooks, returning empty HookResult")
            return HookResult([])

        logger.debug("[hook_manager] run_stop_async: creating StopInput")
        input_data = StopInput(
            stop_hook_active=stop_hook_active,
            reason=reason,
            stop_sequence=stop_sequence,
            session_id=self.session_id,
            transcript_path=self.transcript_path,
            cwd=self._get_cwd(),
            permission_mode=self.permission_mode,
        )

        logger.debug("[hook_manager] run_stop_async: calling executor.execute_hooks_async")
        outputs = await self._execute_hooks_async(hooks, input_data)
        logger.debug("[hook_manager] run_stop_async: execute_hooks_async returned")
        return HookResult(outputs)

    # --- Subagent Start ---

    def run_subagent_start(
        self,
        subagent_type: str,
        prompt: Optional[str] = None,
        resume: Optional[str] = None,
        run_in_background: bool = False,
        tool_use_id: Optional[str] = None,
    ) -> HookResult:
        """Run SubagentStart hooks synchronously."""
        hooks = self._get_hooks(HookEvent.SUBAGENT_START, subagent_type)
        if not hooks:
            return HookResult([])

        input_data = SubagentStartInput(
            subagent_type=subagent_type,
            prompt=prompt,
            resume=resume,
            run_in_background=run_in_background,
            tool_use_id=tool_use_id,
            session_id=self.session_id,
            transcript_path=self.transcript_path,
            cwd=self._get_cwd(),
            permission_mode=self.permission_mode,
        )

        outputs = self._execute_hooks_sync(hooks, input_data)
        return HookResult(outputs)

    async def run_subagent_start_async(
        self,
        subagent_type: str,
        prompt: Optional[str] = None,
        resume: Optional[str] = None,
        run_in_background: bool = False,
        tool_use_id: Optional[str] = None,
    ) -> HookResult:
        """Run SubagentStart hooks asynchronously."""
        hooks = self._get_hooks(HookEvent.SUBAGENT_START, subagent_type)
        if not hooks:
            return HookResult([])

        input_data = SubagentStartInput(
            subagent_type=subagent_type,
            prompt=prompt,
            resume=resume,
            run_in_background=run_in_background,
            tool_use_id=tool_use_id,
            session_id=self.session_id,
            transcript_path=self.transcript_path,
            cwd=self._get_cwd(),
            permission_mode=self.permission_mode,
        )

        outputs = await self._execute_hooks_async(hooks, input_data)
        return HookResult(outputs)

    # --- Subagent Stop ---

    def run_subagent_stop(
        self,
        stop_hook_active: bool = False,
        subagent_type: Optional[str] = None,
    ) -> HookResult:
        """Run SubagentStop hooks synchronously.

        Args:
            stop_hook_active: True if already continuing from a stop hook
            subagent_type: Subagent type name
        """
        hooks = self._get_hooks(HookEvent.SUBAGENT_STOP, subagent_type)
        if not hooks:
            return HookResult([])

        input_data = SubagentStopInput(
            subagent_type=subagent_type or "",
            stop_hook_active=stop_hook_active,
            session_id=self.session_id,
            transcript_path=self.transcript_path,
            cwd=self._get_cwd(),
            permission_mode=self.permission_mode,
        )

        outputs = self._execute_hooks_sync(hooks, input_data)
        return HookResult(outputs)

    async def run_subagent_stop_async(
        self,
        stop_hook_active: bool = False,
        subagent_type: Optional[str] = None,
    ) -> HookResult:
        """Run SubagentStop hooks asynchronously."""
        hooks = self._get_hooks(HookEvent.SUBAGENT_STOP, subagent_type)
        if not hooks:
            return HookResult([])

        input_data = SubagentStopInput(
            subagent_type=subagent_type or "",
            stop_hook_active=stop_hook_active,
            session_id=self.session_id,
            transcript_path=self.transcript_path,
            cwd=self._get_cwd(),
            permission_mode=self.permission_mode,
        )

        outputs = await self._execute_hooks_async(hooks, input_data)
        return HookResult(outputs)

    # --- Setup ---

    def run_setup(self, trigger: str = "init") -> HookResult:
        """Run Setup hooks synchronously."""
        hooks = self._get_hooks(HookEvent.SETUP, trigger)
        if not hooks:
            return HookResult([])

        input_data = SetupInput(
            trigger=trigger,
            session_id=self.session_id,
            transcript_path=self.transcript_path,
            cwd=self._get_cwd(),
            permission_mode=self.permission_mode,
        )

        outputs = self._execute_hooks_sync(hooks, input_data)
        return HookResult(outputs)

    async def run_setup_async(self, trigger: str = "init") -> HookResult:
        """Run Setup hooks asynchronously."""
        hooks = self._get_hooks(HookEvent.SETUP, trigger)
        if not hooks:
            return HookResult([])

        input_data = SetupInput(
            trigger=trigger,
            session_id=self.session_id,
            transcript_path=self.transcript_path,
            cwd=self._get_cwd(),
            permission_mode=self.permission_mode,
        )

        outputs = await self._execute_hooks_async(hooks, input_data)
        return HookResult(outputs)

    # --- Pre Compact ---

    def run_pre_compact(self, trigger: str, custom_instructions: str = "") -> HookResult:
        """Run PreCompact hooks synchronously.

        Args:
            trigger: "manual" or "auto"
            custom_instructions: Custom instructions passed to /compact
        """
        hooks = self._get_hooks(HookEvent.PRE_COMPACT, trigger)
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

        outputs = self._execute_hooks_sync(hooks, input_data)
        return HookResult(outputs)

    async def run_pre_compact_async(
        self, trigger: str, custom_instructions: str = ""
    ) -> HookResult:
        """Run PreCompact hooks asynchronously."""
        hooks = self._get_hooks(HookEvent.PRE_COMPACT, trigger)
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

        outputs = await self._execute_hooks_async(hooks, input_data)
        return HookResult(outputs)

    # --- Session Start ---

    def run_session_start(self, source: str) -> HookResult:
        """Run SessionStart hooks synchronously.

        Args:
            source: "startup", "resume", "clear", or "compact"
        """
        if source == "startup":
            self._run_setup_if_needed()
        hooks = self._get_hooks(HookEvent.SESSION_START, source)
        if not hooks:
            return HookResult([])

        input_data = SessionStartInput(
            source=source,
            session_id=self.session_id,
            transcript_path=self.transcript_path,
            cwd=self._get_cwd(),
            permission_mode=self.permission_mode,
        )

        outputs = self._execute_hooks_sync(hooks, input_data)
        self._apply_session_env()
        return HookResult(outputs)

    async def run_session_start_async(self, source: str) -> HookResult:
        """Run SessionStart hooks asynchronously."""
        if source == "startup":
            await self._run_setup_if_needed_async()
        hooks = self._get_hooks(HookEvent.SESSION_START, source)
        if not hooks:
            return HookResult([])

        input_data = SessionStartInput(
            source=source,
            session_id=self.session_id,
            transcript_path=self.transcript_path,
            cwd=self._get_cwd(),
            permission_mode=self.permission_mode,
        )

        # SessionStart should wait for command hooks so env updates are deterministic.
        outputs = await self._execute_hooks_async(
            hooks,
            input_data,
            allow_async_commands=False,
        )
        self._apply_session_env()
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
        hooks = self._get_hooks(HookEvent.SESSION_END, reason)
        if not hooks:
            self.cleanup()
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

        try:
            outputs = self._execute_hooks_sync(hooks, input_data)
            return HookResult(outputs)
        finally:
            self.cleanup()

    async def run_session_end_async(
        self,
        reason: str,
        duration_seconds: Optional[float] = None,
        message_count: Optional[int] = None,
    ) -> HookResult:
        """Run SessionEnd hooks asynchronously."""
        hooks = self._get_hooks(HookEvent.SESSION_END, reason)
        if not hooks:
            self.cleanup()
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

        try:
            outputs = await self._execute_hooks_async(hooks, input_data)
            return HookResult(outputs)
        finally:
            self.cleanup()


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
    hook_callback: Optional[HookCallback] = None,
) -> HookManager:
    """Initialize the global hook manager with project context.

    Args:
        project_dir: The project directory
        session_id: Current session ID
        transcript_path: Path to the conversation transcript JSON
        permission_mode: Current permission mode
        llm_callback: Async callback for prompt-based hooks
        hook_callback: Async callback for external hook execution

    Returns:
        The initialized global hook manager
    """
    hook_manager.set_project_dir(project_dir)
    hook_manager.set_session_id(session_id)
    hook_manager.set_transcript_path(transcript_path)
    hook_manager.set_permission_mode(permission_mode)
    hook_manager.set_llm_callback(llm_callback)
    hook_manager.set_hook_callback(hook_callback)
    return hook_manager
