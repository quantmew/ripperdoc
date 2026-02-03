"""Tool execution helpers for the query loop."""

import asyncio
import os
import sys
from asyncio import CancelledError
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from ripperdoc.core.hooks.manager import hook_manager
from ripperdoc.core.tool import Tool, ToolProgress, ToolResult, ToolUseContext
from ripperdoc.utils.log import get_logger
from ripperdoc.core.query_utils import tool_result_message
from ripperdoc.utils.messages import (
    ProgressMessage,
    UserMessage,
    create_progress_message,
    create_user_message,
)

from .context import _append_hook_context

logger = get_logger()

# Timeout for individual tool execution (can be overridden per tool if needed)
DEFAULT_TOOL_TIMEOUT_SEC = float(os.getenv("RIPPERDOC_TOOL_TIMEOUT", "300"))  # 5 minutes
# Timeout for concurrent tool execution (total for all tools)
DEFAULT_CONCURRENT_TOOL_TIMEOUT_SEC = float(
    os.getenv("RIPPERDOC_CONCURRENT_TOOL_TIMEOUT", "600")
)  # 10 minutes


def _resolve_tool(
    tool_registry: "ToolRegistry", tool_name: str, tool_use_id: str
) -> tuple[Optional[Tool[Any, Any]], Optional[UserMessage]]:
    """Find a tool by name and return an error message if missing."""
    tool = tool_registry.get(tool_name)
    if tool:
        tool_registry.activate_tools([tool_name])
        return tool, None
    return None, tool_result_message(
        tool_use_id, f"Error: Tool '{tool_name}' not found", is_error=True
    )


async def _run_tool_use_generator(
    tool: Tool[Any, Any],
    tool_use_id: str,
    tool_name: str,
    parsed_input: Any,
    sibling_ids: set[str],
    tool_context: ToolUseContext,
    context: Dict[str, str],
) -> AsyncGenerator[Union[UserMessage, ProgressMessage], None]:
    """Execute a single tool_use and yield progress/results."""
    logger.debug(
        "[query] _run_tool_use_generator ENTER: tool='%s' tool_use_id=%s",
        tool_name,
        tool_use_id,
    )
    # Get tool input as dict for hooks
    tool_input_dict = (
        parsed_input.model_dump()
        if hasattr(parsed_input, "model_dump")
        else dict(parsed_input)
        if isinstance(parsed_input, dict)
        else {}
    )

    # Run PreToolUse hooks
    pre_result = await hook_manager.run_pre_tool_use_async(
        tool_name, tool_input_dict, tool_use_id=tool_use_id
    )
    if pre_result.should_block:
        block_reason = pre_result.block_reason or f"Blocked by hook: {tool_name}"
        logger.info(
            f"[query] Tool {tool_name} blocked by PreToolUse hook",
            extra={"tool_use_id": tool_use_id, "reason": block_reason},
        )
        yield tool_result_message(tool_use_id, f"Hook blocked: {block_reason}", is_error=True)
        return

    # Handle updated input from hooks
    if pre_result.updated_input:
        logger.debug(
            f"[query] PreToolUse hook modified input for {tool_name}",
            extra={"tool_use_id": tool_use_id},
        )
        # Re-parse the input with the updated values
        try:
            # Ensure updated_input is a dict, not a Pydantic model
            updated_input = pre_result.updated_input
            if hasattr(updated_input, "model_dump"):
                updated_input = updated_input.model_dump()
            elif not isinstance(updated_input, dict):
                updated_input = {"value": str(updated_input)}
            parsed_input = tool.input_schema(**updated_input)
            tool_input_dict = updated_input
        except (ValueError, TypeError) as exc:
            logger.warning(
                f"[query] Failed to apply updated input from hook: {exc}",
                extra={"tool_use_id": tool_use_id},
            )

    # Add hook context if provided
    if pre_result.additional_context:
        logger.debug(
            f"[query] PreToolUse hook added context for {tool_name}",
            extra={"context": pre_result.additional_context[:100]},
        )
        _append_hook_context(context, f"PreToolUse:{tool_name}", pre_result.additional_context)
    if pre_result.system_message:
        _append_hook_context(context, f"PreToolUse:{tool_name}:system", pre_result.system_message)

    tool_output = None

    try:
        logger.debug("[query] _run_tool_use_generator: BEFORE tool.call() for '%s'", tool_name)
        # Wrap tool execution with timeout to prevent hangs
        try:
            async with asyncio.timeout(DEFAULT_TOOL_TIMEOUT_SEC):
                async for output in tool.call(parsed_input, tool_context):
                    logger.debug(
                        "[query] _run_tool_use_generator: tool='%s' yielded output type=%s",
                        tool_name,
                        type(output).__name__,
                    )
                    if isinstance(output, ToolProgress):
                        yield create_progress_message(
                            tool_use_id=tool_use_id,
                            sibling_tool_use_ids=sibling_ids,
                            content=output.content,
                            is_subagent_message=getattr(output, "is_subagent_message", False),
                        )
                        logger.debug(
                            f"[query] Progress from tool_use_id={tool_use_id}: {output.content}"
                        )
                    elif isinstance(output, ToolResult):
                        tool_output = output.data
                        result_content = output.result_for_assistant or str(output.data)
                        result_msg = tool_result_message(
                            tool_use_id, result_content, tool_use_result=output.data
                        )
                        yield result_msg
                        logger.debug(
                            f"[query] Tool completed tool_use_id={tool_use_id} name={tool_name} "
                            f"result_len={len(result_content)}"
                        )
        except asyncio.TimeoutError:
            logger.error(
                f"[query] Tool '{tool_name}' timed out after {DEFAULT_TOOL_TIMEOUT_SEC}s",
                extra={"tool": tool_name, "tool_use_id": tool_use_id},
            )
            yield tool_result_message(
                tool_use_id,
                f"Tool '{tool_name}' timed out after {DEFAULT_TOOL_TIMEOUT_SEC:.0f} seconds",
                is_error=True,
            )
            return  # Exit early on timeout
        logger.debug("[query] _run_tool_use_generator: AFTER tool.call() loop for '%s'", tool_name)
    except CancelledError:
        logger.debug("[query] _run_tool_use_generator: tool='%s' CANCELLED", tool_name)
        raise  # Don't suppress task cancellation
    except (RuntimeError, ValueError, TypeError, OSError, IOError, AttributeError, KeyError) as exc:
        logger.warning(
            "Error executing tool '%s': %s: %s",
            tool_name,
            type(exc).__name__,
            exc,
            extra={"tool": tool_name, "tool_use_id": tool_use_id},
        )
        yield tool_result_message(tool_use_id, f"Error executing tool: {str(exc)}", is_error=True)

    # Run PostToolUse hooks
    post_result = await hook_manager.run_post_tool_use_async(
        tool_name, tool_input_dict, tool_response=tool_output, tool_use_id=tool_use_id
    )
    if post_result.additional_context:
        _append_hook_context(context, f"PostToolUse:{tool_name}", post_result.additional_context)
    if post_result.system_message:
        _append_hook_context(context, f"PostToolUse:{tool_name}:system", post_result.system_message)
    if post_result.should_block:
        reason = post_result.block_reason or post_result.stop_reason or "Blocked by hook."
        yield create_user_message(f"PostToolUse hook blocked: {reason}")

    logger.debug(
        "[query] _run_tool_use_generator DONE: tool='%s' tool_use_id=%s", tool_name, tool_use_id
    )


def _group_tool_calls_by_concurrency(prepared_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group consecutive tool calls by their concurrency safety."""
    groups: List[Dict[str, Any]] = []
    for call in prepared_calls:
        is_safe = bool(call.get("is_concurrency_safe"))
        if groups and groups[-1]["is_concurrency_safe"] == is_safe:
            groups[-1]["items"].append(call)
        else:
            groups.append({"is_concurrency_safe": is_safe, "items": [call]})
    return groups


async def _execute_tools_sequentially(
    items: List[Dict[str, Any]], tool_results: List[UserMessage]
) -> AsyncGenerator[Union[UserMessage, ProgressMessage], None]:
    """Run tool generators one by one."""
    for item in items:
        gen = item.get("generator")
        if not gen:
            continue
        async for message in gen:
            if isinstance(message, UserMessage):
                tool_results.append(message)
            yield message


async def _execute_tools_in_parallel(
    items: List[Dict[str, Any]], tool_results: List[UserMessage]
) -> AsyncGenerator[Union[UserMessage, ProgressMessage], None]:
    """Run tool generators concurrently."""
    logger.debug("[query] _execute_tools_in_parallel ENTER: %d items", len(items))
    valid_items = [call for call in items if call.get("generator")]
    generators = [call["generator"] for call in valid_items]
    tool_names = [call.get("tool_name", "unknown") for call in valid_items]
    logger.debug(
        "[query] _execute_tools_in_parallel: %d valid generators, tools=%s",
        len(generators),
        tool_names,
    )
    async for message in _run_concurrent_tool_uses(generators, tool_names, tool_results):
        yield message
    logger.debug("[query] _execute_tools_in_parallel DONE")


async def _run_tools_concurrently(
    prepared_calls: List[Dict[str, Any]], tool_results: List[UserMessage]
) -> AsyncGenerator[Union[UserMessage, ProgressMessage], None]:
    """Run tools grouped by concurrency safety (parallel for safe groups)."""
    for group in _group_tool_calls_by_concurrency(prepared_calls):
        if group["is_concurrency_safe"]:
            logger.debug(
                f"[query] Executing {len(group['items'])} concurrency-safe tool(s) in parallel"
            )
            async for message in _execute_tools_in_parallel(group["items"], tool_results):
                yield message
        else:
            logger.debug(
                f"[query] Executing {len(group['items'])} tool(s) sequentially (not concurrency safe)"
            )
            async for message in _run_tools_serially(group["items"], tool_results):
                yield message


async def _run_tools_serially(
    prepared_calls: List[Dict[str, Any]], tool_results: List[UserMessage]
) -> AsyncGenerator[Union[UserMessage, ProgressMessage], None]:
    """Run all tools sequentially (helper for clarity)."""
    async for message in _execute_tools_sequentially(prepared_calls, tool_results):
        yield message


async def _run_concurrent_tool_uses(
    generators: List[AsyncGenerator[Union[UserMessage, ProgressMessage], None]],
    tool_names: List[str],
    tool_results: List[UserMessage],
) -> AsyncGenerator[Union[UserMessage, ProgressMessage], None]:
    """Drain multiple tool generators concurrently and stream outputs with overall timeout."""
    logger.debug(
        "[query] _run_concurrent_tool_uses ENTER: %d generators, tools=%s, timeout=%s",
        len(generators),
        tool_names,
        DEFAULT_CONCURRENT_TOOL_TIMEOUT_SEC,
    )
    if not generators:
        logger.debug("[query] _run_concurrent_tool_uses: no generators, returning")
        return
        yield  # Make this a proper async generator that yields nothing (unreachable but required)

    queue: asyncio.Queue[Optional[Union[UserMessage, ProgressMessage]]] = asyncio.Queue()

    async def _consume(
        gen: AsyncGenerator[Union[UserMessage, ProgressMessage], None],
        gen_index: int,
        tool_name: str,
    ) -> Optional[Exception]:
        """Consume a tool generator and return any exception that occurred."""
        logger.debug(
            "[query] _consume START: tool='%s' index=%d gen=%s",
            tool_name,
            gen_index,
            type(gen).__name__,
        )
        captured_exception: Optional[Exception] = None
        message_count = 0
        try:
            logger.debug("[query] _consume: entering async for loop for '%s'", tool_name)
            async for message in gen:
                message_count += 1
                msg_type = type(message).__name__
                logger.debug(
                    "[query] _consume: tool='%s' received message #%d type=%s",
                    tool_name,
                    message_count,
                    msg_type,
                )
                await queue.put(message)
                logger.debug("[query] _consume: tool='%s' put message to queue", tool_name)
            logger.debug(
                "[query] _consume: tool='%s' async for loop finished, total messages=%d",
                tool_name,
                message_count,
            )
        except asyncio.CancelledError:
            logger.debug("[query] _consume: tool='%s' was CANCELLED", tool_name)
            raise  # Don't suppress cancellation
        except (StopAsyncIteration, GeneratorExit):
            logger.debug("[query] _consume: tool='%s' StopAsyncIteration/GeneratorExit", tool_name)
            pass  # Normal generator termination
        except Exception as exc:
            # Capture exception for reporting to caller
            captured_exception = exc
            logger.warning(
                "[query] Error while consuming tool '%s' (task %d): %s: %s",
                tool_name,
                gen_index,
                type(exc).__name__,
                exc,
            )
        finally:
            logger.debug("[query] _consume FINALLY: tool='%s' putting None to queue", tool_name)
            await queue.put(None)
            logger.debug("[query] _consume DONE: tool='%s' messages=%d", tool_name, message_count)
        return captured_exception

    logger.debug("[query] _run_concurrent_tool_uses: creating %d tasks", len(generators))
    tasks = [
        asyncio.create_task(_consume(gen, i, tool_names[i])) for i, gen in enumerate(generators)
    ]
    active = len(tasks)
    logger.debug("[query] _run_concurrent_tool_uses: %d tasks created, entering while loop", active)

    try:
        # Add overall timeout for entire concurrent execution
        async with asyncio.timeout(DEFAULT_CONCURRENT_TOOL_TIMEOUT_SEC):
            while active:
                logger.debug(
                    "[query] _run_concurrent_tool_uses: waiting for queue.get(), active=%d", active
                )
                try:
                    message = await asyncio.wait_for(
                        queue.get(), timeout=DEFAULT_CONCURRENT_TOOL_TIMEOUT_SEC
                    )
                except asyncio.TimeoutError:
                    logger.error(
                        "[query] Concurrent tool execution timed out waiting for messages"
                    )
                    # Cancel all remaining tasks
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    raise

                logger.debug(
                    "[query] _run_concurrent_tool_uses: got message type=%s, active=%d",
                    type(message).__name__ if message else "None",
                    active,
                )
                if message is None:
                    active -= 1
                    logger.debug(
                        "[query] _run_concurrent_tool_uses: None received, active now=%d", active
                    )
                    continue
                if isinstance(message, UserMessage):
                    tool_results.append(message)
                yield message
            logger.debug("[query] _run_concurrent_tool_uses: while loop finished, all tools done")
    except asyncio.TimeoutError:
        logger.error(
            f"[query] Concurrent tool execution timed out after {DEFAULT_CONCURRENT_TOOL_TIMEOUT_SEC}s",
            extra={"tool_names": tool_names},
        )
        # Ensure all tasks are cancelled
        for task in tasks:
            if not task.done():
                task.cancel()
        raise
    finally:
        # Wait for all tasks and collect any exceptions
        results = await asyncio.gather(*tasks, return_exceptions=True)
        exceptions_found: List[tuple[int, str, BaseException]] = []
        for i, result in enumerate(results):
            if isinstance(result, asyncio.CancelledError):
                continue
            elif isinstance(result, Exception):
                # Exception from gather itself (shouldn't happen with return_exceptions=True)
                exceptions_found.append((i, tool_names[i], result))
            elif result is not None:
                # Exception returned by _consume
                exceptions_found.append((i, tool_names[i], result))

        # Log all exceptions for debugging
        for i, name, exc in exceptions_found:
            logger.warning(
                "[query] Concurrent tool '%s' (task %d) failed: %s: %s",
                name,
                i,
                type(exc).__name__,
                exc,
            )

        # Re-raise first exception if any occurred, so caller knows something failed
        # Only raise here if no outer exception is already in flight (e.g., timeout/cancel).
        if exceptions_found and sys.exc_info()[1] is None:
            first_name = exceptions_found[0][1]
            first_exc = exceptions_found[0][2]
            logger.error(
                "[query] %d tool(s) failed during concurrent execution, first error in '%s': %s",
                len(exceptions_found),
                first_name,
                first_exc,
            )
            raise first_exc
