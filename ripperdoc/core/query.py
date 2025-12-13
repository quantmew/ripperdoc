"""AI query system for Ripperdoc.

This module handles communication with AI models and manages
the query-response loop including tool execution.
"""

import asyncio
import inspect
import os
import time
from asyncio import CancelledError
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

from pydantic import ValidationError

from ripperdoc.core.config import provider_protocol
from ripperdoc.core.providers import ProviderClient, get_provider_client
from ripperdoc.core.permissions import PermissionResult
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
from ripperdoc.core.tool import Tool, ToolProgress, ToolResult, ToolUseContext
from ripperdoc.utils.file_watch import ChangedFileNotice, FileSnapshot, detect_changed_files
from ripperdoc.utils.log import get_logger
from ripperdoc.utils.messages import (
    AssistantMessage,
    MessageContent,
    ProgressMessage,
    UserMessage,
    create_assistant_message,
    create_user_message,
    create_progress_message,
    normalize_messages_for_api,
    INTERRUPT_MESSAGE,
    INTERRUPT_MESSAGE_FOR_TOOL_USE,
)


logger = get_logger()

DEFAULT_REQUEST_TIMEOUT_SEC = float(os.getenv("RIPPERDOC_API_TIMEOUT", "120"))
MAX_LLM_RETRIES = int(os.getenv("RIPPERDOC_MAX_RETRIES", "10"))


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


ToolPermissionCallable = Callable[
    [Tool[Any, Any], Any],
    Union[
        PermissionResult,
        Dict[str, Any],
        Tuple[bool, Optional[str]],
        bool,
        Awaitable[Union[PermissionResult, Dict[str, Any], Tuple[bool, Optional[str]], bool]],
    ],
]


async def _check_tool_permissions(
    tool: Tool[Any, Any],
    parsed_input: Any,
    query_context: "QueryContext",
    can_use_tool_fn: Optional[ToolPermissionCallable],
) -> tuple[bool, Optional[str]]:
    """Evaluate whether a tool call is allowed."""
    try:
        if can_use_tool_fn is not None:
            decision = can_use_tool_fn(tool, parsed_input)
            if inspect.isawaitable(decision):
                decision = await decision
            if isinstance(decision, PermissionResult):
                return decision.result, decision.message
            if isinstance(decision, dict) and "result" in decision:
                return bool(decision.get("result")), decision.get("message")
            if isinstance(decision, tuple) and len(decision) == 2:
                return bool(decision[0]), decision[1]
            return bool(decision), None

        if query_context.safe_mode and tool.needs_permissions(parsed_input):
            loop = asyncio.get_running_loop()
            input_preview = (
                parsed_input.model_dump()
                if hasattr(parsed_input, "model_dump")
                else str(parsed_input)
            )
            prompt = f"Allow tool '{tool.name}' with input {input_preview}? [y/N]: "
            response = await loop.run_in_executor(None, lambda: input(prompt))
            return response.strip().lower() in ("y", "yes"), None

        return True, None
    except Exception:
        logger.exception(
            f"Error checking permissions for tool '{tool.name}'",
            extra={"tool": getattr(tool, "name", None)},
        )
        return False, None


def _format_changed_file_notice(notices: List[ChangedFileNotice]) -> str:
    """Render a system notice about files that changed on disk."""
    lines: List[str] = [
        "System notice: Files you previously read have changed on disk.",
        "Please re-read the affected files before making further edits.",
        "",
    ]
    for notice in notices:
        lines.append(f"- {notice.file_path}")
        summary = (notice.summary or "").rstrip()
        if summary:
            indented = "\n".join(f"    {line}" for line in summary.splitlines())
            lines.append(indented)
    return "\n".join(lines)


async def _run_tool_use_generator(
    tool: Tool[Any, Any],
    tool_use_id: str,
    tool_name: str,
    parsed_input: Any,
    sibling_ids: set[str],
    tool_context: ToolUseContext,
) -> AsyncGenerator[Union[UserMessage, ProgressMessage], None]:
    """Execute a single tool_use and yield progress/results."""
    try:
        async for output in tool.call(parsed_input, tool_context):
            if isinstance(output, ToolProgress):
                yield create_progress_message(
                    tool_use_id=tool_use_id,
                    sibling_tool_use_ids=sibling_ids,
                    content=output.content,
                )
                logger.debug(f"[query] Progress from tool_use_id={tool_use_id}: {output.content}")
            elif isinstance(output, ToolResult):
                result_content = output.result_for_assistant or str(output.data)
                result_msg = tool_result_message(
                    tool_use_id, result_content, tool_use_result=output.data
                )
                yield result_msg
                logger.debug(
                    f"[query] Tool completed tool_use_id={tool_use_id} name={tool_name} "
                    f"result_len={len(result_content)}"
                )
    except Exception as exc:
        logger.exception(
            f"Error executing tool '{tool_name}'",
            extra={"tool": tool_name, "tool_use_id": tool_use_id},
        )
        yield tool_result_message(tool_use_id, f"Error executing tool: {str(exc)}", is_error=True)


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
    generators = [call["generator"] for call in items if call.get("generator")]
    async for message in _run_concurrent_tool_uses(generators, tool_results):
        yield message


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
    tool_results: List[UserMessage],
) -> AsyncGenerator[Union[UserMessage, ProgressMessage], None]:
    """Drain multiple tool generators concurrently and stream outputs."""
    if not generators:
        return

    queue: asyncio.Queue[Optional[Union[UserMessage, ProgressMessage]]] = asyncio.Queue()

    async def _consume(gen: AsyncGenerator[Union[UserMessage, ProgressMessage], None]) -> None:
        try:
            async for message in gen:
                await queue.put(message)
        except Exception:
            logger.exception("[query] Unexpected error while consuming tool generator")
        finally:
            await queue.put(None)

    tasks = [asyncio.create_task(_consume(gen)) for gen in generators]
    active = len(tasks)

    try:
        while active:
            message = await queue.get()
            if message is None:
                active -= 1
                continue
            if isinstance(message, UserMessage):
                tool_results.append(message)
            yield message
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)


class ToolRegistry:
    """Track available tools, including deferred ones, and expose search/activation helpers."""

    def __init__(self, tools: List[Tool[Any, Any]]) -> None:
        self._tool_map: Dict[str, Tool[Any, Any]] = {}
        self._order: List[str] = []
        self._deferred: set[str] = set()
        self._active: List[str] = []
        self._active_set: set[str] = set()
        self.replace_tools(tools)

    def replace_tools(self, tools: List[Tool[Any, Any]]) -> None:
        """Replace all known tools and rebuild active/deferred lists."""
        seen = set()
        self._tool_map.clear()
        self._order.clear()
        self._deferred.clear()
        self._active.clear()
        self._active_set.clear()

        for tool in tools:
            name = getattr(tool, "name", None)
            if not name or name in seen:
                continue
            seen.add(name)
            self._tool_map[name] = tool
            self._order.append(name)
            try:
                deferred = tool.defer_loading()
            except Exception:
                logger.exception(
                    "[tool_registry] Tool.defer_loading failed",
                    extra={"tool": getattr(tool, "name", None)},
                )
                deferred = False
            if deferred:
                self._deferred.add(name)
            else:
                self._active.append(name)
                self._active_set.add(name)

    @property
    def active_tools(self) -> List[Tool[Any, Any]]:
        """Return active (non-deferred) tools in original order."""
        return [self._tool_map[name] for name in self._order if name in self._active_set]

    @property
    def all_tools(self) -> List[Tool[Any, Any]]:
        """Return all known tools in registration order."""
        return [self._tool_map[name] for name in self._order]

    @property
    def deferred_names(self) -> set[str]:
        """Return the set of deferred tool names."""
        return set(self._deferred)

    def get(self, name: str) -> Optional[Tool[Any, Any]]:
        """Lookup a tool by name."""
        return self._tool_map.get(name)

    def is_active(self, name: str) -> bool:
        """Check if a tool is currently active."""
        return name in self._active_set

    def activate_tools(self, names: Iterable[str]) -> Tuple[List[str], List[str]]:
        """Activate deferred tools by name."""
        activated: List[str] = []
        missing: List[str] = []
        for raw_name in names:
            name = (raw_name or "").strip()
            if not name:
                continue
            if name in self._active_set:
                continue
            tool = self._tool_map.get(name)
            if tool:
                self._active.append(name)
                self._active_set.add(name)
                self._deferred.discard(name)
                activated.append(name)
            else:
                missing.append(name)
        return activated, missing

    def iter_named_tools(self) -> Iterable[tuple[str, Tool[Any, Any]]]:
        """Yield (name, tool) for all known tools in registration order."""
        for name in self._order:
            tool = self._tool_map.get(name)
            if tool:
                yield name, tool


class QueryContext:
    """Context for a query session."""

    def __init__(
        self,
        tools: List[Tool[Any, Any]],
        max_thinking_tokens: int = 0,
        safe_mode: bool = False,
        model: str = "main",
        verbose: bool = False,
        pause_ui: Optional[Callable[[], None]] = None,
        resume_ui: Optional[Callable[[], None]] = None,
    ) -> None:
        self.tool_registry = ToolRegistry(tools)
        self.max_thinking_tokens = max_thinking_tokens
        self.safe_mode = safe_mode
        self.model = model
        self.verbose = verbose
        self.abort_controller = asyncio.Event()
        self.file_state_cache: Dict[str, FileSnapshot] = {}
        self.pause_ui = pause_ui
        self.resume_ui = resume_ui

    @property
    def tools(self) -> List[Tool[Any, Any]]:
        """Active tools available for the current request."""
        return self.tool_registry.active_tools

    @tools.setter
    def tools(self, tools: List[Tool[Any, Any]]) -> None:
        """Replace tool inventory and recompute active/deferred sets."""
        self.tool_registry.replace_tools(tools)

    def activate_tools(self, names: Iterable[str]) -> Tuple[List[str], List[str]]:
        """Activate deferred tools by name."""
        return self.tool_registry.activate_tools(names)

    def all_tools(self) -> List[Tool[Any, Any]]:
        """Return all known tools (active + deferred)."""
        return self.tool_registry.all_tools


async def query_llm(
    messages: List[Union[UserMessage, AssistantMessage, ProgressMessage]],
    system_prompt: str,
    tools: List[Tool[Any, Any]],
    max_thinking_tokens: int = 0,
    model: str = "main",
    abort_signal: Optional[asyncio.Event] = None,
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
        abort_signal: Event to signal abortion
        progress_callback: Optional async callback invoked with streamed text chunks
        request_timeout: Max seconds to wait for a provider response before retrying
        max_retries: Number of retries on timeout/errors (total attempts = retries + 1)
        stream: Enable streaming for providers that support it (text-only mode)

    Returns:
        AssistantMessage with the model's response
    """
    request_timeout = request_timeout or DEFAULT_REQUEST_TIMEOUT_SEC
    request_timeout = request_timeout or DEFAULT_REQUEST_TIMEOUT_SEC
    model_profile = resolve_model_profile(model)

    # Normalize messages based on protocol family (Anthropic allows tool blocks; OpenAI-style prefers text-only)
    protocol = provider_protocol(model_profile.provider)
    tool_mode = determine_tool_mode(model_profile)
    messages_for_model: List[Union[UserMessage, AssistantMessage, ProgressMessage]]
    if tool_mode == "text":
        messages_for_model = cast(
            List[Union[UserMessage, AssistantMessage, ProgressMessage]],
            text_mode_history(messages),
        )
    else:
        messages_for_model = messages

    normalized_messages: List[Dict[str, Any]] = normalize_messages_for_api(
        messages_for_model, protocol=protocol, tool_mode=tool_mode
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
        client: Optional[ProviderClient] = get_provider_client(model_profile.provider)
        if client is None:
            duration_ms = (time.time() - start_time) * 1000
            error_msg = create_assistant_message(
                content=(
                    "Gemini protocol is not supported yet in Ripperdoc. "
                    "Please configure an Anthropic or OpenAI-compatible model."
                ),
                duration_ms=duration_ms,
            )
            error_msg.is_api_error_message = True
            return error_msg

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
        )

        return create_assistant_message(
            content=provider_response.content_blocks,
            cost_usd=provider_response.cost_usd,
            duration_ms=provider_response.duration_ms,
        )

    except Exception as e:
        # Return error message
        logger.exception(
            "Error querying AI model",
            extra={
                "model": getattr(model_profile, "model", None),
                "model_pointer": model,
                "provider": (
                    getattr(model_profile.provider, "value", None) if model_profile else None
                ),
            },
        )
        duration_ms = (time.time() - start_time) * 1000
        error_msg = create_assistant_message(
            content=f"Error querying AI model: {str(e)}", duration_ms=duration_ms
        )
        error_msg.is_api_error_message = True
        return error_msg


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
    4. Recursively continues the conversation

    Args:
        messages: Conversation history
        system_prompt: Base system prompt
        context: Additional context dictionary
        query_context: Query configuration
        can_use_tool_fn: Optional function to check tool permissions

    Yields:
        Messages (user, assistant, progress) as they are generated
    """
    logger.info(
        "[query] Starting query loop",
        extra={
            "message_count": len(messages),
            "tool_count": len(query_context.tools),
            "safe_mode": query_context.safe_mode,
            "model_pointer": query_context.model,
        },
    )
    # Work on a copy so external mutations (e.g., UI appending messages while consuming)
    # do not interfere with recursion or normalization.
    messages = list(messages)
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

    progress_queue: asyncio.Queue[Optional[ProgressMessage]] = asyncio.Queue()

    async def _stream_progress(chunk: str) -> None:
        if not chunk:
            return
        try:
            await progress_queue.put(
                create_progress_message(
                    tool_use_id="stream",
                    sibling_tool_use_ids=set(),
                    content=chunk,
                )
            )
        except Exception:
            logger.exception("[query] Failed to enqueue stream progress chunk")

    assistant_task = asyncio.create_task(
        query_llm(
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

    assistant_message: Optional[AssistantMessage] = None

    while True:
        if query_context.abort_controller.is_set():
            assistant_task.cancel()
            try:
                await assistant_task
            except CancelledError:
                pass
            yield create_assistant_message(INTERRUPT_MESSAGE)
            return
        if assistant_task.done():
            assistant_message = await assistant_task
            break
        try:
            progress = progress_queue.get_nowait()
        except asyncio.QueueEmpty:
            waiter = asyncio.create_task(progress_queue.get())
            done, pending = await asyncio.wait(
                {assistant_task, waiter}, return_when=asyncio.FIRST_COMPLETED
            )
            if assistant_task in done:
                for task in pending:
                    task.cancel()
                assistant_message = await assistant_task
                break
            progress = waiter.result()
        if progress:
            yield progress

    while not progress_queue.empty():
        residual = progress_queue.get_nowait()
        if residual:
            yield residual

    assert assistant_message is not None

    # Check for abort
    if query_context.abort_controller.is_set():
        yield create_assistant_message(INTERRUPT_MESSAGE)
        return

    yield assistant_message

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
        logger.debug("[query] No tool_use blocks; returning response to user.")
        return

    logger.debug(f"[query] Executing {len(tool_use_blocks)} tool_use block(s).")
    tool_results: List[UserMessage] = []
    permission_denied = False
    sibling_ids = set(
        getattr(t, "tool_use_id", None) or getattr(t, "id", None) or "" for t in tool_use_blocks
    )
    prepared_calls: List[Dict[str, Any]] = []

    for tool_use in tool_use_blocks:
        tool_name = tool_use.name
        if not tool_name:
            continue
        tool_use_id = getattr(tool_use, "tool_use_id", None) or getattr(tool_use, "id", None) or ""
        tool_input = getattr(tool_use, "input", {}) or {}

        tool, missing_msg = _resolve_tool(query_context.tool_registry, tool_name, tool_use_id)
        if missing_msg:
            logger.warning(f"[query] Tool '{tool_name}' not found for tool_use_id={tool_use_id}")
            tool_results.append(missing_msg)
            yield missing_msg
            continue
        assert tool is not None

        try:
            parsed_input = tool.input_schema(**tool_input)
            logger.debug(
                f"[query] tool_use_id={tool_use_id} name={tool_name} parsed_input="
                f"{str(parsed_input)[:500]}"
            )

            tool_context = ToolUseContext(
                safe_mode=query_context.safe_mode,
                verbose=query_context.verbose,
                permission_checker=can_use_tool_fn,
                tool_registry=query_context.tool_registry,
                file_state_cache=query_context.file_state_cache,
                abort_signal=query_context.abort_controller,
                pause_ui=query_context.pause_ui,
                resume_ui=query_context.resume_ui,
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
                tool_results.append(result_msg)
                yield result_msg
                continue

            if query_context.safe_mode or can_use_tool_fn is not None:
                allowed, denial_message = await _check_tool_permissions(
                    tool, parsed_input, query_context, can_use_tool_fn
                )
                if not allowed:
                    logger.debug(
                        f"[query] Permission denied for tool_use_id={tool_use_id}: {denial_message}"
                    )
                    denial_text = denial_message or f"User aborted the tool invocation: {tool_name}"
                    denial_msg = tool_result_message(tool_use_id, denial_text, is_error=True)
                    tool_results.append(denial_msg)
                    yield denial_msg
                    permission_denied = True
                    break

            prepared_calls.append(
                {
                    "is_concurrency_safe": tool.is_concurrency_safe(),
                    "generator": _run_tool_use_generator(
                        tool,
                        tool_use_id,
                        tool_name,
                        parsed_input,
                        sibling_ids,
                        tool_context,
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
            tool_results.append(error_msg)
            yield error_msg
            continue
        except Exception as e:
            logger.exception(
                f"Error executing tool '{tool_name}'",
                extra={"tool": tool_name, "tool_use_id": tool_use_id},
            )
            error_msg = tool_result_message(
                tool_use_id, f"Error executing tool: {str(e)}", is_error=True
            )
            tool_results.append(error_msg)
            yield error_msg

        if permission_denied:
            break

    if permission_denied:
        return

    if prepared_calls:
        async for message in _run_tools_concurrently(prepared_calls, tool_results):
            yield message

    # Check for abort after tools
    if query_context.abort_controller.is_set():
        yield create_assistant_message(INTERRUPT_MESSAGE_FOR_TOOL_USE)
        return

    if permission_denied:
        return

    new_messages = messages + [assistant_message] + tool_results
    logger.debug(
        f"[query] Recursing with {len(new_messages)} messages after tools; "
        f"tool_results_count={len(tool_results)}"
    )

    async for msg in query(new_messages, system_prompt, context, query_context, can_use_tool_fn):
        yield msg
