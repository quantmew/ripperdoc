"""Integration helpers for hooks with tool execution.

This module provides convenient integration points for running hooks
as part of tool execution flows.
"""

from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union

from ripperdoc.core.hooks.manager import HookManager, hook_manager
from ripperdoc.utils.log import get_logger

logger = get_logger()

T = TypeVar("T")


class HookInterceptor:
    """Provides hook interception for tool execution.

    This class wraps tool execution with pre/post hooks,
    handling blocking decisions and context injection.
    """

    def __init__(self, manager: Optional[HookManager] = None):
        """Initialize the interceptor.

        Args:
            manager: HookManager to use, defaults to global instance
        """
        self.manager = manager or hook_manager

    def check_pre_tool_use(
        self, tool_name: str, tool_input: Dict[str, Any]
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Check if a tool call should proceed based on PreToolUse hooks.

        Args:
            tool_name: Name of the tool
            tool_input: Tool input parameters

        Returns:
            Tuple of (should_proceed, block_reason, additional_context)
            - should_proceed: True if tool should execute
            - block_reason: Reason if blocked, None otherwise
            - additional_context: Additional context to add if any
        """
        result = self.manager.run_pre_tool_use(tool_name, tool_input)

        if result.should_block:
            logger.info(
                f"Tool {tool_name} blocked by hook",
                extra={"reason": result.block_reason},
            )
            return False, result.block_reason, result.additional_context

        if result.should_ask:
            # For 'ask' decision, we return a special flag
            # The caller should prompt the user for confirmation
            return False, "USER_CONFIRMATION_REQUIRED", result.additional_context

        return True, None, result.additional_context

    async def check_pre_tool_use_async(
        self, tool_name: str, tool_input: Dict[str, Any]
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Async version of check_pre_tool_use."""
        result = await self.manager.run_pre_tool_use_async(tool_name, tool_input)

        if result.should_block:
            logger.info(
                f"Tool {tool_name} blocked by hook",
                extra={"reason": result.block_reason},
            )
            return False, result.block_reason, result.additional_context

        if result.should_ask:
            return False, "USER_CONFIRMATION_REQUIRED", result.additional_context

        return True, None, result.additional_context

    def run_post_tool_use(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_output: Any = None,
        tool_error: Optional[str] = None,
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Run PostToolUse hooks after tool execution.

        Args:
            tool_name: Name of the tool
            tool_input: Tool input parameters
            tool_output: Output from the tool
            tool_error: Error message if tool failed

        Returns:
            Tuple of (should_continue, block_reason, additional_context)
        """
        result = self.manager.run_post_tool_use(tool_name, tool_input, tool_output, tool_error)

        if result.should_block:
            return False, result.block_reason, result.additional_context

        return True, None, result.additional_context

    async def run_post_tool_use_async(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_output: Any = None,
        tool_error: Optional[str] = None,
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Async version of run_post_tool_use."""
        result = await self.manager.run_post_tool_use_async(
            tool_name, tool_input, tool_output, tool_error
        )

        if result.should_block:
            return False, result.block_reason, result.additional_context

        return True, None, result.additional_context

    def wrap_tool_execution(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        execute_fn: Callable[[], T],
    ) -> Tuple[bool, Union[T, str, None], Optional[str]]:
        """Wrap synchronous tool execution with pre/post hooks.

        Args:
            tool_name: Name of the tool
            tool_input: Tool input parameters
            execute_fn: Function to execute the tool

        Returns:
            Tuple of (success, result_or_error, additional_context)
        """
        # Run pre-tool hooks
        should_proceed, block_reason, pre_context = self.check_pre_tool_use(tool_name, tool_input)

        if not should_proceed:
            return False, block_reason or "Blocked by hook", pre_context

        # Execute the tool
        try:
            result = execute_fn()
            tool_error = None
        except Exception as e:
            result = None
            tool_error = str(e)

        # Run post-tool hooks
        _, _, post_context = self.run_post_tool_use(tool_name, tool_input, result, tool_error)

        # Combine contexts
        combined_context = None
        if pre_context or post_context:
            parts = [c for c in [pre_context, post_context] if c]
            combined_context = "\n".join(parts) if parts else None

        if tool_error:
            return False, tool_error or "", combined_context

        return True, result, combined_context

    async def wrap_tool_execution_async(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        execute_fn: Callable[[], T],
    ) -> Tuple[bool, Union[T, str, None], Optional[str]]:
        """Wrap async tool execution with pre/post hooks."""
        # Run pre-tool hooks
        should_proceed, block_reason, pre_context = await self.check_pre_tool_use_async(
            tool_name, tool_input
        )

        if not should_proceed:
            return False, block_reason or "Blocked by hook", pre_context

        # Execute the tool
        try:
            import asyncio

            if asyncio.iscoroutinefunction(execute_fn):
                result = await execute_fn()
            else:
                result = execute_fn()
            tool_error = None
        except Exception as e:
            result = None
            tool_error = str(e)

        # Run post-tool hooks
        _, _, post_context = await self.run_post_tool_use_async(
            tool_name, tool_input, result, tool_error
        )

        # Combine contexts
        combined_context = None
        if pre_context or post_context:
            parts = [c for c in [pre_context, post_context] if c]
            combined_context = "\n".join(parts) if parts else None

        if tool_error:
            return False, tool_error or "", combined_context

        return True, result, combined_context


# Global interceptor instance
hook_interceptor = HookInterceptor()


def check_pre_tool_use(
    tool_name: str, tool_input: Dict[str, Any]
) -> Tuple[bool, Optional[str], Optional[str]]:
    """Convenience function to check pre-tool hooks using global interceptor."""
    return hook_interceptor.check_pre_tool_use(tool_name, tool_input)


async def check_pre_tool_use_async(
    tool_name: str, tool_input: Dict[str, Any]
) -> Tuple[bool, Optional[str], Optional[str]]:
    """Async convenience function to check pre-tool hooks."""
    return await hook_interceptor.check_pre_tool_use_async(tool_name, tool_input)


def run_post_tool_use(
    tool_name: str,
    tool_input: Dict[str, Any],
    tool_output: Any = None,
    tool_error: Optional[str] = None,
) -> Tuple[bool, Optional[str], Optional[str]]:
    """Convenience function to run post-tool hooks using global interceptor."""
    return hook_interceptor.run_post_tool_use(tool_name, tool_input, tool_output, tool_error)


async def run_post_tool_use_async(
    tool_name: str,
    tool_input: Dict[str, Any],
    tool_output: Any = None,
    tool_error: Optional[str] = None,
) -> Tuple[bool, Optional[str], Optional[str]]:
    """Async convenience function to run post-tool hooks."""
    return await hook_interceptor.run_post_tool_use_async(
        tool_name, tool_input, tool_output, tool_error
    )


def check_user_prompt(prompt: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """Check if a user prompt should be processed.

    Args:
        prompt: The user's prompt text

    Returns:
        Tuple of (should_process, block_reason, additional_context)
    """
    result = hook_manager.run_user_prompt_submit(prompt)

    if result.should_block:
        return False, result.block_reason, result.additional_context

    return True, None, result.additional_context


async def check_user_prompt_async(
    prompt: str,
) -> Tuple[bool, Optional[str], Optional[str]]:
    """Async version of check_user_prompt."""
    result = await hook_manager.run_user_prompt_submit_async(prompt)

    if result.should_block:
        return False, result.block_reason, result.additional_context

    return True, None, result.additional_context


def notify_session_start(trigger: str) -> Optional[str]:
    """Notify hooks that a session is starting.

    Args:
        trigger: "startup", "resume", "clear", or "compact"

    Returns:
        Additional context from hooks, if any
    """
    result = hook_manager.run_session_start(trigger)
    return result.additional_context


def notify_session_end(
    trigger: str,
    duration_seconds: Optional[float] = None,
    message_count: int = 0,
) -> Optional[str]:
    """Notify hooks that a session is ending.

    Args:
        trigger: "clear", "logout", "prompt_input_exit", or "other"
        duration_seconds: How long the session lasted
        message_count: Number of messages in the session

    Returns:
        Additional context from hooks, if any
    """
    result = hook_manager.run_session_end(trigger, duration_seconds, message_count)
    return result.additional_context


def check_stop(
    reason: Optional[str] = None, stop_sequence: Optional[str] = None
) -> Tuple[bool, Optional[str]]:
    """Check if the agent should stop.

    Args:
        reason: Why the agent wants to stop
        stop_sequence: The stop sequence encountered, if any

    Returns:
        Tuple of (should_stop, continue_reason)
        - should_stop: True if agent should stop
        - continue_reason: Reason to continue if blocked
    """
    result = hook_manager.run_stop(False, reason, stop_sequence)

    if result.should_block:
        return False, result.block_reason

    return True, None


async def check_stop_async(
    reason: Optional[str] = None, stop_sequence: Optional[str] = None
) -> Tuple[bool, Optional[str]]:
    """Async version of check_stop."""
    result = await hook_manager.run_stop_async(False, reason, stop_sequence)

    if result.should_block:
        return False, result.block_reason

    return True, None
