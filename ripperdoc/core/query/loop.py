"""Main query loop and LLM interaction helpers."""

import asyncio
import os
import sys
import time
from asyncio import CancelledError
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, List, Optional, Union, cast

from pydantic import ValidationError

from ripperdoc.core.config import ModelProfile, provider_protocol
from ripperdoc.core.hooks.manager import HookResult, hook_manager
from ripperdoc.core.hooks.state import bind_hook_scopes, bind_pending_message_queue
from ripperdoc.core.providers import ProviderClient, get_provider_client
from ripperdoc.core.query_utils import (
    build_full_system_prompt,
    determine_tool_mode,
    extract_tool_use_blocks,
    format_pydantic_errors,
    log_openai_messages,
    resolve_model_profile,
    text_mode_history,
    tool_result_message,
)
from ripperdoc.core.tool import Tool, ToolUseContext
from ripperdoc.utils.context_length_errors import detect_context_length_error
from ripperdoc.utils.file_watch import detect_changed_files
from ripperdoc.utils.log import get_logger
from ripperdoc.utils.messages import (
    AssistantMessage,
    MessageContent,
    ProgressMessage,
    UserMessage,
    create_assistant_message,
    create_hook_notice_message,
    create_progress_message,
    create_user_message,
    normalize_messages_for_api,
    INTERRUPT_MESSAGE,
    INTERRUPT_MESSAGE_FOR_TOOL_USE,
)

from .context import QueryContext, _append_hook_context, _apply_skill_context_updates
from .errors import _format_changed_file_notice
from .permissions import ToolPermissionCallable, _check_tool_permissions
from .tools import (
    _resolve_tool,
    _run_hook_call_with_status,
    _run_tool_use_generator,
    _run_tools_concurrently,
)

logger = get_logger()

DEFAULT_REQUEST_TIMEOUT_SEC = float(os.getenv("RIPPERDOC_API_TIMEOUT", "120"))
MAX_LLM_RETRIES = int(os.getenv("RIPPERDOC_MAX_RETRIES", "10"))


def infer_thinking_mode(model_profile: ModelProfile) -> Optional[str]:
    """Infer thinking mode from ModelProfile if not explicitly configured.

    This function checks the model_profile.thinking_mode first. If it's set,
    returns that value. Otherwise, auto-detects based on api_base and model name.

    Args:
        model_profile: The model profile to analyze

    Returns:
        Thinking mode string ("deepseek", "qwen", "openrouter", "gemini_openai")
        or None if no thinking mode should be applied.
    """
    # Use explicit config if set
    explicit_mode = model_profile.thinking_mode
    if explicit_mode:
        # "none", "disabled", "off" means thinking is explicitly disabled
        if explicit_mode.lower() in ("disabled", "off"):
            return None
        return explicit_mode

    # Auto-detect based on API base and model name
    base = (model_profile.api_base or "").lower()
    name = (model_profile.model or "").lower()

    if "deepseek" in base or name.startswith("deepseek"):
        return "deepseek"
    if "dashscope" in base or "qwen" in name:
        return "qwen"
    if "openrouter.ai" in base:
        return "openrouter"
    if "generativelanguage.googleapis.com" in base or name.startswith("gemini"):
        return "gemini_openai"
    if "openai" in base:
        return "openai"

    return None


def _prepare_request_messages(
    messages: List[Union[UserMessage, AssistantMessage, ProgressMessage]],
    model_profile: ModelProfile,
) -> tuple[str, str, List[Union[UserMessage, AssistantMessage, ProgressMessage]]]:
    """Prepare protocol/tool-mode and conversation payload for provider request."""
    protocol = provider_protocol(model_profile.provider)
    tool_mode = determine_tool_mode(model_profile)
    if tool_mode == "text":
        messages_for_model = cast(
            List[Union[UserMessage, AssistantMessage, ProgressMessage]],
            text_mode_history(messages),
        )
    else:
        messages_for_model = messages
    return protocol, tool_mode, messages_for_model


def _resolve_request_thinking_mode(
    protocol: str,
    model_profile: ModelProfile,
    max_thinking_tokens: int,
) -> Optional[str]:
    """Resolve provider-specific thinking mode for the current request."""
    if protocol != "openai":
        return None
    model_name = (model_profile.model or "").lower()
    is_reasoning_model = "reasoner" in model_name or "r1" in model_name
    if max_thinking_tokens > 0 or is_reasoning_model:
        return infer_thinking_mode(model_profile)
    return None


def _create_api_error_message(
    content: Any,
    *,
    duration_ms: float,
    model: Optional[str],
    metadata: Optional[Dict[str, Any]] = None,
) -> AssistantMessage:
    """Create a standardized assistant API error message."""
    error_msg = create_assistant_message(
        content=content,
        duration_ms=duration_ms,
        metadata=metadata,
        model=model,
    )
    error_msg.is_api_error_message = True
    return error_msg


def _resolve_provider_client_or_error(
    model_profile: ModelProfile,
    start_time: float,
) -> tuple[Optional[ProviderClient], Optional[AssistantMessage]]:
    """Resolve provider client or return a user-facing error message."""
    try:
        client: Optional[ProviderClient] = get_provider_client(model_profile.provider)
    except RuntimeError as exc:
        duration_ms = (time.time() - start_time) * 1000
        return None, _create_api_error_message(
            str(exc),
            duration_ms=duration_ms,
            model=model_profile.model,
        )

    if client is None:
        duration_ms = (time.time() - start_time) * 1000
        provider_label = getattr(model_profile.provider, "value", None) or str(
            model_profile.provider
        )
        return None, _create_api_error_message(
            (
                f"No provider client available for '{provider_label}'. "
                "Check your model configuration and provider dependencies."
            ),
            duration_ms=duration_ms,
            model=model_profile.model,
        )

    return client, None


def _build_provider_error_metadata(provider_response: Any) -> Dict[str, Any]:
    """Build metadata payload for provider-level API errors."""
    metadata: Dict[str, Any] = {
        "api_error": True,
        "error_code": provider_response.error_code,
        "error_message": provider_response.error_message,
    }
    if provider_response.error_code == "context_length_exceeded":
        metadata["context_length_exceeded"] = True
    return metadata


async def query_llm(
    messages: List[Union[UserMessage, AssistantMessage, ProgressMessage]],
    system_prompt: str,
    tools: List[Tool[Any, Any]],
    max_thinking_tokens: int = 0,
    model: str = "main",
    _abort_signal: Optional[asyncio.Event] = None,
    *,
    progress_callback: Optional[Callable[[str], Awaitable[None]]] = None,
    request_timeout: Optional[float] = None,
    max_retries: int = MAX_LLM_RETRIES,
    stream: bool = True,
) -> AssistantMessage:
    """Query the AI model and return the response.

    Args:
        messages: Conversation history
        system_prompt: System prompt for the model
        tools: Available tools
        max_thinking_tokens: Maximum tokens for thinking (0 = disabled)
        model: Model pointer to use
        _abort_signal: Event to signal abortion (currently unused, reserved for future)
        progress_callback: Optional async callback invoked with streamed text chunks
        request_timeout: Max seconds to wait for a provider response before retrying
        max_retries: Number of retries on timeout/errors (total attempts = retries + 1)
        stream: Enable streaming for providers that support it (text-only mode)

    Returns:
        AssistantMessage with the model's response
    """
    request_timeout = request_timeout or DEFAULT_REQUEST_TIMEOUT_SEC
    model_profile = resolve_model_profile(model)
    protocol, tool_mode, messages_for_model = _prepare_request_messages(messages, model_profile)
    thinking_mode = _resolve_request_thinking_mode(protocol, model_profile, max_thinking_tokens)

    normalized_messages: List[Dict[str, Any]] = normalize_messages_for_api(
        messages_for_model,
        protocol=protocol,
        tool_mode=tool_mode,
        thinking_mode=thinking_mode,
    )
    logger.info(
        "[query_llm] Preparing model request",
        extra={
            "model_pointer": model,
            "provider": getattr(model_profile.provider, "value", str(model_profile.provider)),
            "model": model_profile.model,
            "normalized_messages": len(normalized_messages),
            "tool_count": len(tools),
            "max_thinking_tokens": max_thinking_tokens,
            "thinking_mode": thinking_mode,
            "tool_mode": tool_mode,
        },
    )

    if protocol == "openai":
        log_openai_messages(normalized_messages)

    logger.debug(
        f"[query_llm] Sending {len(normalized_messages)} messages to model pointer "
        f"'{model}' with {len(tools)} tool schemas; "
        f"max_thinking_tokens={max_thinking_tokens} protocol={protocol}"
    )

    # Make the API call
    start_time = time.time()

    try:
        client, client_error = _resolve_provider_client_or_error(model_profile, start_time)
        if client_error is not None:
            return client_error
        if client is None:
            return _create_api_error_message(
                "No provider client available.",
                duration_ms=(time.time() - start_time) * 1000,
                model=model_profile.model,
            )

        provider_response = await client.call(
            model_profile=model_profile,
            system_prompt=system_prompt,
            normalized_messages=normalized_messages,
            tools=tools,
            tool_mode=tool_mode,
            stream=stream,
            progress_callback=progress_callback,
            request_timeout=request_timeout,
            max_retries=max_retries,
            max_thinking_tokens=max_thinking_tokens,
        )

        # Check if provider returned an error response
        if provider_response.is_error:
            logger.warning(
                "[query_llm] Provider returned error response",
                extra={
                    "model": model_profile.model,
                    "error_code": provider_response.error_code,
                    "error_message": provider_response.error_message,
                },
            )
            return _create_api_error_message(
                provider_response.content_blocks,
                duration_ms=provider_response.duration_ms,
                model=model_profile.model,
                metadata=_build_provider_error_metadata(provider_response),
            )

        return create_assistant_message(
            content=provider_response.content_blocks,
            cost_usd=provider_response.cost_usd,
            duration_ms=provider_response.duration_ms,
            metadata=provider_response.metadata,
            model=model_profile.model,
            input_tokens=provider_response.usage_tokens.get("input_tokens", 0),
            output_tokens=provider_response.usage_tokens.get("output_tokens", 0),
            cache_read_tokens=provider_response.usage_tokens.get("cache_read_input_tokens", 0),
            cache_creation_tokens=provider_response.usage_tokens.get(
                "cache_creation_input_tokens", 0
            ),
        )

    except CancelledError:
        raise  # Don't suppress task cancellation
    except (RuntimeError, ValueError, TypeError, OSError, ConnectionError, TimeoutError) as e:
        # Return error message
        logger.warning(
            "Error querying AI model: %s: %s",
            type(e).__name__,
            e,
            extra={
                "model": getattr(model_profile, "model", None),
                "model_pointer": model,
                "provider": (
                    getattr(model_profile.provider, "value", None) if model_profile else None
                ),
            },
        )
        duration_ms = (time.time() - start_time) * 1000
        context_error = detect_context_length_error(e)
        error_metadata: Optional[Dict[str, Any]] = None
        content = f"Error querying AI model: {str(e)}"

        if context_error:
            content = f"The request exceeded the model's context window. {context_error.message}"
            error_metadata = {
                "context_length_exceeded": True,
                "context_length_provider": context_error.provider,
                "context_length_error_code": context_error.error_code,
                "context_length_status_code": context_error.status_code,
            }
            logger.info(
                "[query_llm] Detected context-length error; consider compacting history",
                extra={
                    "provider": context_error.provider,
                    "error_code": context_error.error_code,
                    "status_code": context_error.status_code,
                },
            )

        return _create_api_error_message(
            content,
            duration_ms=duration_ms,
            model=model_profile.model,
            metadata=error_metadata,
        )


MAX_QUERY_ITERATIONS = int(os.getenv("RIPPERDOC_MAX_QUERY_ITERATIONS", "1024"))


@dataclass
class IterationResult:
    """Result of a single query iteration.

    This is used as an "out parameter" to communicate results from
    _run_query_iteration back to the main query loop.
    """

    assistant_message: Optional[AssistantMessage] = None
    tool_results: List[UserMessage] = field(default_factory=list)
    should_stop: bool = False  # True means exit the query loop entirely


@dataclass
class _PreparedToolCalls:
    """Intermediate state while preparing tool calls for execution."""

    prepared_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[UserMessage] = field(default_factory=list)
    permission_denied: bool = False


def _normalize_tool_input_payload(raw_input: Any) -> Dict[str, Any]:
    """Normalize tool input payload to a plain dictionary."""
    if raw_input and hasattr(raw_input, "model_dump"):
        normalized = raw_input.model_dump()
    elif raw_input and hasattr(raw_input, "dict") and callable(getattr(raw_input, "dict")):
        normalized = raw_input.dict()
    elif raw_input and not isinstance(raw_input, dict):
        normalized = {"value": str(raw_input)}
    elif isinstance(raw_input, dict):
        normalized = raw_input
    else:
        normalized = {}
    return cast(Dict[str, Any], normalized)


async def _handle_iteration_stop_hook(
    context: Dict[str, str],
    query_context: QueryContext,
    result: IterationResult,
) -> AsyncGenerator[Union[UserMessage, ProgressMessage], None]:
    """Run stop hook when assistant response has no tool calls."""
    logger.debug("[query] No tool_use blocks; running stop hook and returning response to user.")
    stop_hook = query_context.stop_hook
    logger.debug(f"[query] stop_hook={stop_hook}, stop_hook_active={query_context.stop_hook_active}")
    logger.debug("[query] BEFORE calling hook_manager.run_stop_async")
    stop_result = (
        await hook_manager.run_subagent_stop_async(
            stop_hook_active=query_context.stop_hook_active,
            subagent_type=getattr(query_context, "subagent_type", None),
        )
        if stop_hook == "subagent"
        else await hook_manager.run_stop_async(stop_hook_active=query_context.stop_hook_active)
    )
    logger.debug("[query] AFTER calling hook_manager.run_stop_async")
    logger.debug("[query] Checking additional_context")
    if stop_result.additional_context:
        _append_hook_context(context, f"{stop_hook}:context", stop_result.additional_context)
    logger.debug("[query] Checking system_message")
    if stop_result.system_message:
        hook_event = "SubagentStop" if stop_hook == "subagent" else "Stop"
        yield create_hook_notice_message(
            str(stop_result.system_message),
            hook_event=hook_event,
            tool_name=getattr(query_context, "subagent_type", None),
        )
    logger.debug("[query] Checking should_block")
    if stop_result.should_block:
        reason = stop_result.block_reason or stop_result.stop_reason or "Blocked by hook."
        result.tool_results = [create_user_message(f"{stop_hook} hook blocked: {reason}")]
        for msg in result.tool_results:
            yield msg
        query_context.stop_hook_active = True
        result.should_stop = False
        return
    logger.debug("[query] Setting should_stop=True and returning")
    query_context.stop_hook_active = False
    result.should_stop = True


async def _prepare_iteration_tool_calls(
    *,
    tool_use_blocks: List[MessageContent],
    messages: List[Union[UserMessage, AssistantMessage, ProgressMessage]],
    context: Dict[str, str],
    query_context: QueryContext,
    can_use_tool_fn: Optional[ToolPermissionCallable],
    out: _PreparedToolCalls,
) -> AsyncGenerator[Union[UserMessage, ProgressMessage], None]:
    """Resolve tools, run hooks/permissions, and build runnable tool generators."""
    logger.debug(f"[query] Executing {len(tool_use_blocks)} tool_use block(s).")
    sibling_ids = set(
        getattr(t, "tool_use_id", None) or getattr(t, "id", None) or "" for t in tool_use_blocks
    )

    for tool_use in tool_use_blocks:
        tool_name = tool_use.name
        if not tool_name:
            continue
        tool_use_id = getattr(tool_use, "tool_use_id", None) or getattr(tool_use, "id", None) or ""
        tool_input = _normalize_tool_input_payload(getattr(tool_use, "input", {}) or {})

        tool, missing_msg = _resolve_tool(query_context.tool_registry, tool_name, tool_use_id)
        if missing_msg:
            logger.warning(f"[query] Tool '{tool_name}' not found for tool_use_id={tool_use_id}")
            out.tool_results.append(missing_msg)
            yield missing_msg
            continue
        if tool is None:
            raise RuntimeError(f"Tool '{tool_name}' resolved to None unexpectedly")

        try:
            parsed_input = tool.input_schema(**tool_input)
            logger.debug(
                f"[query] tool_use_id={tool_use_id} name={tool_name} parsed_input="
                f"{str(parsed_input)[:500]}"
            )

            tool_context = ToolUseContext(
                message_id=tool_use_id,  # Set message_id for parent_tool_use_id tracking
                yolo_mode=query_context.yolo_mode,
                verbose=query_context.verbose,
                permission_checker=can_use_tool_fn,
                tool_registry=query_context.tool_registry,
                file_state_cache=query_context.file_state_cache,
                conversation_messages=messages,
                abort_signal=query_context.abort_controller,
                pause_ui=query_context.pause_ui,
                resume_ui=query_context.resume_ui,
                pending_message_queue=query_context.pending_message_queue,
            )

            validation = await tool.validate_input(parsed_input, tool_context)
            if not validation.result:
                logger.debug(
                    f"[query] Validation failed for tool_use_id={tool_use_id}: {validation.message}"
                )
                result_msg = tool_result_message(
                    tool_use_id,
                    validation.message or "Tool input validation failed.",
                    is_error=True,
                )
                out.tool_results.append(result_msg)
                yield result_msg
                continue

            tool_input_dict = (
                parsed_input.model_dump()
                if hasattr(parsed_input, "model_dump")
                else dict(parsed_input)
                if isinstance(parsed_input, dict)
                else {}
            )

            # Run PreToolUse hooks before permission checks.
            pre_result: Optional[HookResult] = None
            async for item in _run_hook_call_with_status(
                hook_manager.run_pre_tool_use_async(
                    tool_name, tool_input_dict, tool_use_id=tool_use_id
                ),
                tool_use_id,
                sibling_ids,
            ):
                if isinstance(item, ProgressMessage):
                    yield item
                else:
                    pre_result = item
            if pre_result is None:
                pre_result = HookResult([])
            if pre_result.should_block or not pre_result.should_continue:
                reason = (
                    pre_result.block_reason
                    or pre_result.stop_reason
                    or f"Blocked by hook: {tool_name}"
                )
                result_msg = tool_result_message(tool_use_id, f"Hook blocked: {reason}", is_error=True)
                out.tool_results.append(result_msg)
                yield result_msg
                continue

            if pre_result.updated_input:
                logger.debug(
                    f"[query] PreToolUse hook modified input for {tool_name}",
                    extra={"tool_use_id": tool_use_id},
                )
                try:
                    normalized_input = _normalize_tool_input_payload(pre_result.updated_input)
                    parsed_input = tool.input_schema(**normalized_input)
                    tool_input_dict = normalized_input
                except ValidationError as ve:
                    detail_text = format_pydantic_errors(ve)
                    error_msg = tool_result_message(
                        tool_use_id,
                        f"Invalid PreToolUse-updated input for tool '{tool_name}': {detail_text}",
                        is_error=True,
                    )
                    out.tool_results.append(error_msg)
                    yield error_msg
                    continue
                except (TypeError, ValueError) as exc:
                    logger.warning(
                        "[query] Failed to apply updated input from PreToolUse hook: %s",
                        exc,
                        extra={"tool_use_id": tool_use_id},
                    )
                    error_msg = tool_result_message(
                        tool_use_id,
                        f"Invalid PreToolUse-updated input for tool '{tool_name}'.",
                        is_error=True,
                    )
                    out.tool_results.append(error_msg)
                    yield error_msg
                    continue
                validation = await tool.validate_input(parsed_input, tool_context)
                if not validation.result:
                    error_msg = tool_result_message(
                        tool_use_id,
                        validation.message or "Tool input validation failed.",
                        is_error=True,
                    )
                    out.tool_results.append(error_msg)
                    yield error_msg
                    continue

            if pre_result.additional_context:
                _append_hook_context(context, f"PreToolUse:{tool_name}", pre_result.additional_context)
            if pre_result.system_message:
                yield create_hook_notice_message(
                    str(pre_result.system_message),
                    hook_event="PreToolUse",
                    tool_name=tool_name,
                    tool_use_id=tool_use_id,
                    sibling_tool_use_ids=sibling_ids,
                )

            force_permission_prompt = pre_result.should_ask
            bypass_permissions = pre_result.should_allow and not force_permission_prompt

            if not bypass_permissions and (
                not query_context.yolo_mode or can_use_tool_fn is not None or force_permission_prompt
            ):
                allowed, denial_message, updated_input = await _check_tool_permissions(
                    tool,
                    parsed_input,
                    query_context,
                    can_use_tool_fn,
                    force_prompt=force_permission_prompt,
                )
                if not allowed:
                    logger.debug(
                        f"[query] Permission denied for tool_use_id={tool_use_id}: {denial_message}"
                    )
                    denial_text = denial_message or f"User aborted the tool invocation: {tool_name}"
                    denial_msg = tool_result_message(tool_use_id, denial_text, is_error=True)
                    out.tool_results.append(denial_msg)
                    yield denial_msg
                    out.permission_denied = True
                    break
                if updated_input:
                    try:
                        normalized_input = _normalize_tool_input_payload(updated_input)
                        parsed_input = tool.input_schema(**normalized_input)
                    except ValidationError as ve:
                        detail_text = format_pydantic_errors(ve)
                        error_msg = tool_result_message(
                            tool_use_id,
                            f"Invalid permission-updated input for tool '{tool_name}': {detail_text}",
                            is_error=True,
                        )
                        out.tool_results.append(error_msg)
                        yield error_msg
                        continue
                    validation = await tool.validate_input(parsed_input, tool_context)
                    if not validation.result:
                        error_msg = tool_result_message(
                            tool_use_id,
                            validation.message or "Tool input validation failed.",
                            is_error=True,
                        )
                        out.tool_results.append(error_msg)
                        yield error_msg
                        continue

            out.prepared_calls.append(
                {
                    "tool_name": tool_name,
                    "is_concurrency_safe": tool.is_concurrency_safe(),
                    "generator": _run_tool_use_generator(
                        tool,
                        tool_use_id,
                        tool_name,
                        parsed_input,
                        sibling_ids,
                        tool_context,
                        context,
                    ),
                }
            )

        except ValidationError as ve:
            detail_text = format_pydantic_errors(ve)
            error_msg = tool_result_message(
                tool_use_id,
                f"Invalid input for tool '{tool_name}': {detail_text}",
                is_error=True,
            )
            out.tool_results.append(error_msg)
            yield error_msg
            continue
        except CancelledError:
            raise  # Don't suppress task cancellation
        except (
            RuntimeError,
            ValueError,
            TypeError,
            OSError,
            IOError,
            AttributeError,
            KeyError,
        ) as e:
            logger.warning(
                "Error executing tool '%s': %s: %s",
                tool_name,
                type(e).__name__,
                e,
                extra={"tool": tool_name, "tool_use_id": tool_use_id},
            )
            error_msg = tool_result_message(tool_use_id, f"Error executing tool: {str(e)}", is_error=True)
            out.tool_results.append(error_msg)
            yield error_msg

        if out.permission_denied:
            break


async def _run_query_iteration(
    messages: List[Union[UserMessage, AssistantMessage, ProgressMessage]],
    system_prompt: str,
    context: Dict[str, str],
    query_context: QueryContext,
    can_use_tool_fn: Optional[ToolPermissionCallable],
    iteration: int,
    result: IterationResult,
) -> AsyncGenerator[Union[UserMessage, AssistantMessage, ProgressMessage], None]:
    """Run a single iteration of the query loop.

    This function handles one round of:
    1. Calling the LLM
    2. Streaming progress
    3. Processing tool calls (if any)

    Args:
        messages: Current conversation history
        system_prompt: Base system prompt
        context: Additional context dictionary
        query_context: Query configuration
        can_use_tool_fn: Optional function to check tool permissions
        iteration: Current iteration number (for logging)
        result: IterationResult object to store results

    Yields:
        Messages (progress, assistant, tool results) as they are generated
    """
    logger.info(f"[query] Starting iteration {iteration}/{MAX_QUERY_ITERATIONS}")

    # Check for file changes at the start of each iteration
    change_notices = detect_changed_files(query_context.file_state_cache)
    if change_notices:
        messages.append(create_user_message(_format_changed_file_notice(change_notices)))

    model_profile = resolve_model_profile(query_context.model)
    tool_mode = determine_tool_mode(model_profile)
    tools_for_model: List[Tool[Any, Any]] = [] if tool_mode == "text" else query_context.all_tools()

    full_system_prompt = build_full_system_prompt(
        system_prompt, context, tool_mode, query_context.all_tools()
    )
    logger.debug(
        "[query] Built system prompt",
        extra={
            "prompt_chars": len(full_system_prompt),
            "context_entries": len(context),
            "tool_count": len(tools_for_model),
        },
    )

    # Stream LLM response
    progress_queue: asyncio.Queue[Optional[ProgressMessage]] = asyncio.Queue(maxsize=1000)

    async def _stream_progress(chunk: str) -> None:
        if not chunk:
            return
        try:
            msg = create_progress_message(
                tool_use_id="stream",
                sibling_tool_use_ids=set(),
                content=chunk,
            )
            try:
                progress_queue.put_nowait(msg)
            except asyncio.QueueFull:
                # Queue full - wait with timeout instead of dropping immediately
                try:
                    await asyncio.wait_for(progress_queue.put(msg), timeout=0.5)
                except asyncio.TimeoutError:
                    logger.warning("[query] Progress queue full after timeout, dropping chunk")
        except (RuntimeError, ValueError) as exc:
            logger.warning("[query] Failed to enqueue stream progress chunk: %s", exc)

    query_module = sys.modules.get("ripperdoc.core.query")
    query_llm_fn = getattr(query_module, "query_llm", query_llm) if query_module else query_llm

    assistant_task = asyncio.create_task(
        query_llm_fn(
            messages,
            full_system_prompt,
            tools_for_model,
            query_context.max_thinking_tokens,
            query_context.model,
            query_context.abort_controller,
            progress_callback=_stream_progress,
            request_timeout=DEFAULT_REQUEST_TIMEOUT_SEC,
            max_retries=MAX_LLM_RETRIES,
            stream=True,
        )
    )

    logger.debug("[query] Created query_llm task, waiting for response...")

    assistant_message: Optional[AssistantMessage] = None

    # Wait for LLM response while yielding progress
    while True:
        if query_context.abort_controller.is_set():
            assistant_task.cancel()
            try:
                await assistant_task
            except CancelledError:
                pass
            yield create_assistant_message(INTERRUPT_MESSAGE, model=model_profile.model)
            result.should_stop = True
            return
        if assistant_task.done():
            assistant_message = await assistant_task
            break
        try:
            progress = progress_queue.get_nowait()
        except asyncio.QueueEmpty:
            waiter = asyncio.create_task(progress_queue.get())
            abort_waiter = asyncio.create_task(query_context.abort_controller.wait())
            done, pending = await asyncio.wait(
                {assistant_task, waiter, abort_waiter},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                # Don't cancel assistant_task here - it should only be cancelled
                # through abort_controller in the main loop
                if task is not assistant_task:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            if abort_waiter in done:
                continue
            if assistant_task in done:
                assistant_message = await assistant_task
                break
            progress = waiter.result()
        if progress:
            yield progress

    # Drain remaining progress messages
    while not progress_queue.empty():
        residual = progress_queue.get_nowait()
        if residual:
            yield residual

    if assistant_message is None:
        raise RuntimeError("assistant_message was unexpectedly None after LLM query")
    result.assistant_message = assistant_message

    # Check for abort
    if query_context.abort_controller.is_set():
        yield create_assistant_message(INTERRUPT_MESSAGE, model=model_profile.model)
        result.should_stop = True
        return

    yield assistant_message

    # Extract and process tool calls
    tool_use_blocks: List[MessageContent] = extract_tool_use_blocks(assistant_message)
    text_blocks = (
        len(assistant_message.message.content)
        if isinstance(assistant_message.message.content, list)
        else 1
    )
    logger.debug(
        f"[query] Assistant message received: text_blocks={text_blocks}, "
        f"tool_use_blocks={len(tool_use_blocks)}"
    )

    if not tool_use_blocks:
        async for msg in _handle_iteration_stop_hook(context, query_context, result):
            yield msg
        return

    prepared = _PreparedToolCalls()
    async for msg in _prepare_iteration_tool_calls(
        tool_use_blocks=tool_use_blocks,
        messages=messages,
        context=context,
        query_context=query_context,
        can_use_tool_fn=can_use_tool_fn,
        out=prepared,
    ):
        yield msg

    if prepared.permission_denied:
        result.tool_results = prepared.tool_results
        result.should_stop = True
        return

    if prepared.prepared_calls:
        async for message in _run_tools_concurrently(prepared.prepared_calls, prepared.tool_results):
            yield message

    _apply_skill_context_updates(prepared.tool_results, query_context)

    # Check for abort after tools
    if query_context.abort_controller.is_set():
        yield create_assistant_message(INTERRUPT_MESSAGE_FOR_TOOL_USE, model=model_profile.model)
        result.tool_results = prepared.tool_results
        result.should_stop = True
        return

    result.tool_results = prepared.tool_results
    # should_stop remains False, indicating the loop should continue


async def query(
    messages: List[Union[UserMessage, AssistantMessage, ProgressMessage]],
    system_prompt: str,
    context: Dict[str, str],
    query_context: QueryContext,
    can_use_tool_fn: Optional[ToolPermissionCallable] = None,
) -> AsyncGenerator[Union[UserMessage, AssistantMessage, ProgressMessage], None]:
    """Execute a query with tool support.

    This is the main query loop that:
    1. Sends messages to the AI
    2. Handles tool use responses
    3. Executes tools
    4. Continues the conversation in a loop until no more tool calls

    Args:
        messages: Conversation history
        system_prompt: Base system prompt
        context: Additional context dictionary
        query_context: Query configuration
        can_use_tool_fn: Optional function to check tool permissions

    Yields:
        Messages (user, assistant, progress) as they are generated
    """
    # Resolve model once for use in messages (e.g., max iterations, errors)
    model_profile = resolve_model_profile(query_context.model)
    """Execute a query with tool support.

    This is the main query loop that:
    1. Sends messages to the AI
    2. Handles tool use responses
    3. Executes tools
    4. Continues the conversation in a loop until no more tool calls

    Args:
        messages: Conversation history
        system_prompt: Base system prompt
        context: Additional context dictionary
        query_context: Query configuration
        can_use_tool_fn: Optional function to check tool permissions

    Yields:
        Messages (user, assistant, progress) as they are generated
    """
    with bind_pending_message_queue(query_context.pending_message_queue), bind_hook_scopes(
        query_context.hook_scopes
    ):
        logger.info(
            "[query] Starting query loop",
            extra={
                "message_count": len(messages),
                "tool_count": len(query_context.tools),
                "yolo_mode": query_context.yolo_mode,
                "model_pointer": query_context.model,
                "max_turns": query_context.max_turns,
                "permission_mode": query_context.permission_mode,
            },
        )
        # Work on a copy so external mutations (e.g., UI appending messages while consuming)
        # do not interfere with the loop or normalization.
        messages = list(messages)

        max_iterations = MAX_QUERY_ITERATIONS
        if query_context.max_turns and query_context.max_turns > 0:
            max_iterations = min(MAX_QUERY_ITERATIONS, query_context.max_turns)

        for iteration in range(1, max_iterations + 1):
            # Inject any pending messages queued by background events or user interjections
            pending_messages = query_context.drain_pending_messages()
            if pending_messages:
                messages.extend(pending_messages)
                for pending in pending_messages:
                    yield pending

            result = IterationResult()

            async for msg in _run_query_iteration(
                messages,
                system_prompt,
                context,
                query_context,
                can_use_tool_fn,
                iteration,
                result,
            ):
                yield msg

            if result.should_stop:
                # Before stopping, check if new pending messages arrived during this iteration.
                trailing_pending = query_context.drain_pending_messages()
                if trailing_pending:
                    # type: ignore[operator,list-item]
                    next_messages = (
                        messages + [result.assistant_message] + result.tool_results
                        if result.assistant_message is not None
                        else messages + result.tool_results  # type: ignore[operator]
                    )  # type: ignore[operator]
                    next_messages = next_messages + trailing_pending  # type: ignore[operator,list-item]
                    for pending in trailing_pending:
                        yield pending
                    messages = next_messages
                    continue
                return

            # Update messages for next iteration
            if result.assistant_message is not None:
                messages = messages + [result.assistant_message] + result.tool_results  # type: ignore[operator]
            else:
                messages = messages + result.tool_results  # type: ignore[operator]

            logger.debug(
                f"[query] Continuing loop with {len(messages)} messages after tools; "
                f"tool_results_count={len(result.tool_results)}"
            )

        # Reached max iterations
        logger.warning(
            f"[query] Reached maximum iterations ({max_iterations}), stopping query loop"
        )
        yield create_assistant_message(
            f"Reached maximum query iterations ({max_iterations}). "
            "Please continue the conversation to proceed.",
            model=model_profile.model,
        )
