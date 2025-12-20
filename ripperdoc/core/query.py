"""AI query system for Ripperdoc.

This module handles communication with AI models and manages
the query-response loop including tool execution.
"""

import asyncio
import inspect
import os
import time
from asyncio import CancelledError
from dataclasses import dataclass, field
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

from ripperdoc.core.config import ModelProfile, provider_protocol
from ripperdoc.core.providers import ProviderClient, get_provider_client
from ripperdoc.core.permissions import PermissionResult
from ripperdoc.core.hooks.manager import hook_manager
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
from ripperdoc.utils.coerce import parse_optional_int
from ripperdoc.utils.context_length_errors import detect_context_length_error
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

        if not query_context.yolo_mode and tool.needs_permissions(parsed_input):
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
    except (TypeError, AttributeError, ValueError) as exc:
        logger.warning(
            f"Error checking permissions for tool '{tool.name}': {type(exc).__name__}: {exc}",
            extra={"tool": getattr(tool, "name", None), "error_type": type(exc).__name__},
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
            parsed_input = tool.input_schema(**pre_result.updated_input)
            tool_input_dict = pre_result.updated_input
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

    tool_output = None

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
    except CancelledError:
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
    await hook_manager.run_post_tool_use_async(
        tool_name, tool_input_dict, tool_response=tool_output, tool_use_id=tool_use_id
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
        except asyncio.CancelledError:
            raise  # Don't suppress cancellation
        except (StopAsyncIteration, GeneratorExit):
            pass  # Normal generator termination
        except (RuntimeError, ValueError, TypeError) as exc:
            logger.warning(
                "[query] Error while consuming tool generator: %s: %s",
                type(exc).__name__,
                exc,
            )
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
            except (TypeError, AttributeError) as exc:
                logger.warning(
                    "[tool_registry] Tool.defer_loading failed: %s: %s",
                    type(exc).__name__,
                    exc,
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


def _apply_skill_context_updates(
    tool_results: List[UserMessage], query_context: "QueryContext"
) -> None:
    """Update query context based on Skill tool outputs."""
    for message in tool_results:
        data = getattr(message, "tool_use_result", None)
        if not isinstance(data, dict):
            continue
        skill_name = (
            data.get("skill")
            or data.get("command_name")
            or data.get("commandName")
            or data.get("command")
        )
        if not skill_name:
            continue

        allowed_tools = data.get("allowed_tools") or data.get("allowedTools") or []
        if allowed_tools and getattr(query_context, "tool_registry", None):
            try:
                query_context.tool_registry.activate_tools(
                    [tool for tool in allowed_tools if isinstance(tool, str) and tool.strip()]
                )
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning(
                    "[query] Failed to activate tools listed in skill output: %s: %s",
                    type(exc).__name__,
                    exc,
                )

        model_hint = data.get("model")
        if isinstance(model_hint, str) and model_hint.strip():
            logger.debug(
                "[query] Applying model hint from skill",
                extra={"skill": skill_name, "model": model_hint},
            )
            query_context.model = model_hint.strip()

        max_tokens = data.get("max_thinking_tokens")
        if max_tokens is None:
            max_tokens = data.get("maxThinkingTokens")
        parsed_max = parse_optional_int(max_tokens)
        if parsed_max is not None:
            logger.debug(
                "[query] Applying max thinking tokens from skill",
                extra={"skill": skill_name, "max_thinking_tokens": parsed_max},
            )
            query_context.max_thinking_tokens = parsed_max


class QueryContext:
    """Context for a query session."""

    def __init__(
        self,
        tools: List[Tool[Any, Any]],
        max_thinking_tokens: int = 0,
        yolo_mode: bool = False,
        model: str = "main",
        verbose: bool = False,
        pause_ui: Optional[Callable[[], None]] = None,
        resume_ui: Optional[Callable[[], None]] = None,
    ) -> None:
        self.tool_registry = ToolRegistry(tools)
        self.max_thinking_tokens = max_thinking_tokens
        self.yolo_mode = yolo_mode
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

    # Get thinking_mode for provider-specific handling
    # Apply when thinking is enabled (max_thinking_tokens > 0) OR when using a
    # reasoning model like deepseek-reasoner which has thinking enabled by default
    thinking_mode: Optional[str] = None
    if protocol == "openai":
        model_name = (model_profile.model or "").lower()
        # DeepSeek Reasoner models have thinking enabled by default
        is_reasoning_model = "reasoner" in model_name or "r1" in model_name
        if max_thinking_tokens > 0 or is_reasoning_model:
            thinking_mode = infer_thinking_mode(model_profile)

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
            metadata: Dict[str, Any] = {
                "api_error": True,
                "error_code": provider_response.error_code,
                "error_message": provider_response.error_message,
            }
            # Add context length info if applicable
            if provider_response.error_code == "context_length_exceeded":
                metadata["context_length_exceeded"] = True

            error_msg = create_assistant_message(
                content=provider_response.content_blocks,
                duration_ms=provider_response.duration_ms,
                metadata=metadata,
            )
            error_msg.is_api_error_message = True
            return error_msg

        return create_assistant_message(
            content=provider_response.content_blocks,
            cost_usd=provider_response.cost_usd,
            duration_ms=provider_response.duration_ms,
            metadata=provider_response.metadata,
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

        error_msg = create_assistant_message(
            content=content, duration_ms=duration_ms, metadata=error_metadata
        )
        error_msg.is_api_error_message = True
        return error_msg


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
    logger.debug(f"[query] Iteration {iteration}/{MAX_QUERY_ITERATIONS}")

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
        except asyncio.QueueFull:
            logger.warning("[query] Progress queue full, dropping chunk")
        except (RuntimeError, ValueError) as exc:
            logger.warning("[query] Failed to enqueue stream progress chunk: %s", exc)

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

    # Wait for LLM response while yielding progress
    while True:
        if query_context.abort_controller.is_set():
            assistant_task.cancel()
            try:
                await assistant_task
            except CancelledError:
                pass
            yield create_assistant_message(INTERRUPT_MESSAGE)
            result.should_stop = True
            return
        if assistant_task.done():
            assistant_message = await assistant_task
            break
        try:
            progress = progress_queue.get_nowait()
        except asyncio.QueueEmpty:
            waiter = asyncio.create_task(progress_queue.get())
            # Use timeout to periodically check abort_controller during LLM request
            done, pending = await asyncio.wait(
                {assistant_task, waiter},
                return_when=asyncio.FIRST_COMPLETED,
                timeout=0.1,  # Check abort_controller every 100ms
            )
            if not done:
                # Timeout - cancel waiter and continue loop to check abort_controller
                waiter.cancel()
                try:
                    await waiter
                except asyncio.CancelledError:
                    pass
                continue
            if assistant_task in done:
                for task in pending:
                    task.cancel()
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

    assert assistant_message is not None
    result.assistant_message = assistant_message

    # Check for abort
    if query_context.abort_controller.is_set():
        yield create_assistant_message(INTERRUPT_MESSAGE)
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
        logger.debug("[query] No tool_use blocks; returning response to user.")
        result.should_stop = True
        return

    # Process tool calls
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
                yolo_mode=query_context.yolo_mode,
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

            if not query_context.yolo_mode or can_use_tool_fn is not None:
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
            error_msg = tool_result_message(
                tool_use_id, f"Error executing tool: {str(e)}", is_error=True
            )
            tool_results.append(error_msg)
            yield error_msg

        if permission_denied:
            break

    if permission_denied:
        result.tool_results = tool_results
        result.should_stop = True
        return

    if prepared_calls:
        async for message in _run_tools_concurrently(prepared_calls, tool_results):
            yield message

    _apply_skill_context_updates(tool_results, query_context)

    # Check for abort after tools
    if query_context.abort_controller.is_set():
        yield create_assistant_message(INTERRUPT_MESSAGE_FOR_TOOL_USE)
        result.tool_results = tool_results
        result.should_stop = True
        return

    result.tool_results = tool_results
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
    logger.info(
        "[query] Starting query loop",
        extra={
            "message_count": len(messages),
            "tool_count": len(query_context.tools),
            "yolo_mode": query_context.yolo_mode,
            "model_pointer": query_context.model,
        },
    )
    # Work on a copy so external mutations (e.g., UI appending messages while consuming)
    # do not interfere with the loop or normalization.
    messages = list(messages)

    for iteration in range(1, MAX_QUERY_ITERATIONS + 1):
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
        f"[query] Reached maximum iterations ({MAX_QUERY_ITERATIONS}), stopping query loop"
    )
    yield create_assistant_message(
        f"Reached maximum query iterations ({MAX_QUERY_ITERATIONS}). "
        "Please continue the conversation to proceed."
    )
