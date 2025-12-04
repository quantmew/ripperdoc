"""AI query system for Ripperdoc.

This module handles communication with AI models and manages
the query-response loop including tool execution.
"""

import asyncio
import inspect
import time
from typing import Any, AsyncGenerator, Dict, Iterable, List, Optional, Tuple, Union, cast

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from pydantic import ValidationError

from ripperdoc.core.config import ProviderType, provider_protocol
from ripperdoc.core.permissions import PermissionResult
from ripperdoc.core.query_utils import (
    anthropic_usage_tokens,
    build_anthropic_tool_schemas,
    build_full_system_prompt,
    build_openai_tool_schemas,
    content_blocks_from_anthropic_response,
    content_blocks_from_openai_choice,
    determine_tool_mode,
    extract_tool_use_blocks,
    format_pydantic_errors,
    log_openai_messages,
    openai_usage_tokens,
    resolve_model_profile,
    text_mode_history,
    tool_result_message,
)
from ripperdoc.core.tool import Tool, ToolProgress, ToolResult, ToolUseContext
from ripperdoc.utils.log import get_logger
from ripperdoc.utils.messages import (
    AssistantMessage,
    ProgressMessage,
    UserMessage,
    create_assistant_message,
    create_progress_message,
    normalize_messages_for_api,
    INTERRUPT_MESSAGE,
    INTERRUPT_MESSAGE_FOR_TOOL_USE,
)
from ripperdoc.utils.session_usage import record_usage


logger = get_logger()


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


async def _check_tool_permissions(
    tool: Tool[Any, Any],
    parsed_input: Any,
    query_context: "QueryContext",
    can_use_tool_fn: Optional[Any],
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
    ) -> None:
        self.tool_registry = ToolRegistry(tools)
        self.max_thinking_tokens = max_thinking_tokens
        self.safe_mode = safe_mode
        self.model = model
        self.verbose = verbose
        self.abort_controller = asyncio.Event()

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
) -> AssistantMessage:
    """Query the AI model and return the response.

    Args:
        messages: Conversation history
        system_prompt: System prompt for the model
        tools: Available tools
        max_thinking_tokens: Maximum tokens for thinking (0 = disabled)
        model: Model pointer to use
        abort_signal: Event to signal abortion

    Returns:
        AssistantMessage with the model's response
    """
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

    normalized_messages = normalize_messages_for_api(
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
        # Create the appropriate client based on provider
        if model_profile.provider == ProviderType.ANTHROPIC:
            async with AsyncAnthropic(api_key=model_profile.api_key) as client:
                tool_schemas = await build_anthropic_tool_schemas(tools)
                response = await client.messages.create(
                    model=model_profile.model,
                    max_tokens=model_profile.max_tokens,
                    system=system_prompt,
                    messages=normalized_messages,  # type: ignore[arg-type]
                    tools=tool_schemas if tool_schemas else None,  # type: ignore
                    temperature=model_profile.temperature,
                )

                duration_ms = (time.time() - start_time) * 1000

                usage_tokens = anthropic_usage_tokens(getattr(response, "usage", None))
                record_usage(model_profile.model, duration_ms=duration_ms, **usage_tokens)

                # Calculate cost (simplified, should use actual pricing)
                cost_usd = 0.0  # TODO: Implement cost calculation

                content_blocks = content_blocks_from_anthropic_response(response, tool_mode)
                tool_use_blocks = [
                    block for block in response.content if getattr(block, "type", None) == "tool_use"
                ]
                logger.info(
                    "[query_llm] Received response from Anthropic",
                    extra={
                        "model": model_profile.model,
                        "duration_ms": round(duration_ms, 2),
                        "usage_tokens": usage_tokens,
                        "tool_use_blocks": len(tool_use_blocks),
                    },
                )

                return create_assistant_message(
                    content=content_blocks,
                    cost_usd=cost_usd,
                    duration_ms=duration_ms,
                )

        elif model_profile.provider == ProviderType.OPENAI_COMPATIBLE:
            # OpenAI-compatible APIs (OpenAI, DeepSeek, Mistral, etc.)
            async with AsyncOpenAI(api_key=model_profile.api_key, base_url=model_profile.api_base) as client:
                openai_tools = await build_openai_tool_schemas(tools)

                # Prepare messages for OpenAI format
                openai_messages = [
                    {"role": "system", "content": system_prompt}
                ] + normalized_messages

                # Make the API call
                openai_response: Any = await client.chat.completions.create(
                    model=model_profile.model,
                    messages=openai_messages,
                    tools=openai_tools if openai_tools else None,  # type: ignore[arg-type]
                    temperature=model_profile.temperature,
                    max_tokens=model_profile.max_tokens,
                )

                duration_ms = (time.time() - start_time) * 1000
                usage_tokens = openai_usage_tokens(getattr(openai_response, "usage", None))
                record_usage(model_profile.model, duration_ms=duration_ms, **usage_tokens)
                cost_usd = 0.0  # TODO: Implement cost calculation

                # Convert OpenAI response to our format
                content_blocks = []
                choice = openai_response.choices[0]

                logger.info(
                    "[query_llm] Received response from OpenAI-compatible provider",
                    extra={
                        "model": model_profile.model,
                        "duration_ms": round(duration_ms, 2),
                        "usage_tokens": usage_tokens,
                        "finish_reason": getattr(choice, "finish_reason", None),
                    },
                )

                content_blocks = content_blocks_from_openai_choice(choice, tool_mode)

                return create_assistant_message(
                    content=content_blocks, cost_usd=cost_usd, duration_ms=duration_ms
                )

        elif model_profile.provider == ProviderType.GEMINI:
            raise NotImplementedError("Gemini protocol is not yet supported.")
        else:
            raise NotImplementedError(f"Provider {model_profile.provider} not yet implemented")

    except Exception as e:
        # Return error message
        logger.exception(
            "Error querying AI model",
            extra={
                "model": getattr(model_profile, "model", None),
                "model_pointer": model,
                "provider": getattr(model_profile.provider, "value", None)
                if model_profile
                else None,
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
    can_use_tool_fn: Optional[Any] = None,
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

    assistant_message = await query_llm(
        messages,
        full_system_prompt,
        tools_for_model,
        query_context.max_thinking_tokens,
        query_context.model,
        query_context.abort_controller,
    )

    # Check for abort
    if query_context.abort_controller.is_set():
        yield create_assistant_message(INTERRUPT_MESSAGE)
        return

    yield assistant_message

    tool_use_blocks = extract_tool_use_blocks(assistant_message)
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

        tool_context = ToolUseContext(
            safe_mode=query_context.safe_mode,
            verbose=query_context.verbose,
            permission_checker=can_use_tool_fn,
            tool_registry=query_context.tool_registry,
            abort_signal=query_context.abort_controller,
        )

        try:
            parsed_input = tool.input_schema(**tool_input)
            logger.debug(
                f"[query] tool_use_id={tool_use_id} name={tool_name} parsed_input="
                f"{str(parsed_input)[:500]}"
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

            async for output in tool.call(parsed_input, tool_context):
                if isinstance(output, ToolProgress):
                    progress = create_progress_message(
                        tool_use_id=tool_use_id,
                        sibling_tool_use_ids=sibling_ids,
                        content=output.content,
                    )
                    yield progress
                    logger.debug(f"[query] Progress from tool_use_id={tool_use_id}: {output.content}")
                elif isinstance(output, ToolResult):
                    result_content = output.result_for_assistant or str(output.data)
                    result_msg = tool_result_message(
                        tool_use_id, result_content, tool_use_result=output.data
                    )
                    tool_results.append(result_msg)
                    yield result_msg
                    logger.debug(
                        f"[query] Tool completed tool_use_id={tool_use_id} name={tool_name} "
                        f"result_len={len(result_content)}"
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
