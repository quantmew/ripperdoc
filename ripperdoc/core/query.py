"""AI query system for Ripperdoc.

This module handles communication with AI models and manages
the query-response loop including tool execution.
"""

import asyncio
import inspect
from typing import AsyncGenerator, List, Optional, Dict, Any, Union
from anthropic import Anthropic, AsyncAnthropic
from anthropic.types import Message as APIMessage, ToolUseBlock
from openai import AsyncOpenAI

from ripperdoc.core.tool import Tool, ToolUseContext, ToolResult, ToolProgress
from ripperdoc.utils.log import get_logger
from ripperdoc.utils.messages import (
    Message,
    MessageContent,
    UserMessage,
    AssistantMessage,
    ProgressMessage,
    create_user_message,
    create_assistant_message,
    create_progress_message,
    normalize_messages_for_api,
    INTERRUPT_MESSAGE,
    INTERRUPT_MESSAGE_FOR_TOOL_USE,
    create_tool_result_stop_message,
)
from ripperdoc.core.permissions import PermissionResult
from ripperdoc.core.config import get_global_config, ModelProfile, ProviderType
from ripperdoc.utils.session_usage import record_usage

import time


logger = get_logger()


def _safe_int(value: Any) -> int:
    """Best-effort int conversion for usage counters."""
    try:
        if value is None:
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def _get_usage_field(usage: Any, field: str) -> int:
    """Fetch a usage field from either a dict or object."""
    if usage is None:
        return 0
    if isinstance(usage, dict):
        return _safe_int(usage.get(field))
    return _safe_int(getattr(usage, field, 0))


def _anthropic_usage_tokens(usage: Any) -> Dict[str, int]:
    """Extract token counts from an Anthropic response usage payload."""
    return {
        "input_tokens": _get_usage_field(usage, "input_tokens"),
        "output_tokens": _get_usage_field(usage, "output_tokens"),
        "cache_read_input_tokens": _get_usage_field(usage, "cache_read_input_tokens"),
        "cache_creation_input_tokens": _get_usage_field(usage, "cache_creation_input_tokens"),
    }


def _openai_usage_tokens(usage: Any) -> Dict[str, int]:
    """Extract token counts from an OpenAI-compatible response usage payload."""
    prompt_details = None
    if isinstance(usage, dict):
        prompt_details = usage.get("prompt_tokens_details")
    else:
        prompt_details = getattr(usage, "prompt_tokens_details", None)

    cache_read_tokens = _get_usage_field(prompt_details, "cached_tokens") if prompt_details else 0

    return {
        "input_tokens": _get_usage_field(usage, "prompt_tokens"),
        "output_tokens": _get_usage_field(usage, "completion_tokens"),
        "cache_read_input_tokens": cache_read_tokens,
        "cache_creation_input_tokens": 0,
    }


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
        self.tools = tools
        self.max_thinking_tokens = max_thinking_tokens
        self.safe_mode = safe_mode
        self.model = model
        self.verbose = verbose
        self.abort_controller = asyncio.Event()


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
    config = get_global_config()

    # Get the model profile
    profile_name = getattr(config.model_pointers, model, None)
    if profile_name is None:
        profile_name = model

    model_profile = config.model_profiles.get(profile_name)
    if model_profile is None:
        fallback_profile = getattr(config.model_pointers, "main", "default")
        model_profile = config.model_profiles.get(fallback_profile)

    if not model_profile:
        raise ValueError(f"No model profile found for pointer: {model}")

    # Normalize messages based on provider (Anthropic allows tool blocks; others prefer text-only)
    protocol = "anthropic" if model_profile.provider == ProviderType.ANTHROPIC else "openai"
    normalized_messages = normalize_messages_for_api(
        messages,
        protocol=protocol,
    )

    if protocol == "openai":
        summary_parts = []
        for idx, m in enumerate(normalized_messages):
            role = m.get("role")
            tool_calls = m.get("tool_calls")
            tc_ids = []
            if tool_calls:
                tc_ids = [tc.get("id") for tc in tool_calls]
            tool_call_id = m.get("tool_call_id")
            summary_parts.append(
                f"{idx}:{role}"
                + (f" tool_calls={tc_ids}" if tc_ids else "")
                + (f" tool_call_id={tool_call_id}" if tool_call_id else "")
            )
        logger.debug(f"[query_llm] OpenAI normalized messages: {' | '.join(summary_parts)}")

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
                # Build tool schemas
                tool_schemas = []
                for tool in tools:
                    tool_schemas.append(
                        {
                            "name": tool.name,
                            "description": await tool.description(),
                            "input_schema": tool.input_schema.model_json_schema(),
                        }
                    )

                # Make the API call
                response = await client.messages.create(
                    model=model_profile.model,
                    max_tokens=model_profile.max_tokens,
                    system=system_prompt,
                    messages=normalized_messages,
                    tools=tool_schemas if tool_schemas else None,  # type: ignore
                    temperature=model_profile.temperature,
                )

                duration_ms = (time.time() - start_time) * 1000

                usage_tokens = _anthropic_usage_tokens(getattr(response, "usage", None))
                record_usage(model_profile.model, duration_ms=duration_ms, **usage_tokens)

                # Calculate cost (simplified, should use actual pricing)
                cost_usd = 0.0  # TODO: Implement cost calculation

                # Convert response to our format
                content_blocks = []
                for block in response.content:
                    if block.type == "text":
                        content_blocks.append({"type": "text", "text": block.text})
                    elif block.type == "tool_use":
                        content_blocks.append(
                            {
                                "type": "tool_use",
                                "tool_use_id": block.id,
                                "name": block.name,
                                "input": block.input,
                            }
                        )

                return create_assistant_message(
                    content=content_blocks, cost_usd=cost_usd, duration_ms=duration_ms
                )

        elif model_profile.provider == ProviderType.OPENAI:
            # OpenAI (including DeepSeek and other compatible APIs)
            async with AsyncOpenAI(
                api_key=model_profile.api_key, base_url=model_profile.api_base
            ) as client:
                # Build tool schemas for OpenAI format
                openai_tools = []
                for tool in tools:
                    openai_tools.append(
                        {
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": await tool.description(),
                                "parameters": tool.input_schema.model_json_schema(),
                            },
                        }
                    )

                # Prepare messages for OpenAI format
                openai_messages = [
                    {"role": "system", "content": system_prompt}
                ] + normalized_messages

                # Make the API call
                response = await client.chat.completions.create(
                    model=model_profile.model,
                    messages=openai_messages,
                    tools=openai_tools if openai_tools else None,
                    temperature=model_profile.temperature,
                    max_tokens=model_profile.max_tokens,
                )

                duration_ms = (time.time() - start_time) * 1000
                usage_tokens = _openai_usage_tokens(getattr(response, "usage", None))
                record_usage(model_profile.model, duration_ms=duration_ms, **usage_tokens)
                cost_usd = 0.0  # TODO: Implement cost calculation

                # Convert OpenAI response to our format
                content_blocks = []
                choice = response.choices[0]

                if choice.message.content:
                    content_blocks.append({"type": "text", "text": choice.message.content})

                if choice.message.tool_calls:
                    for tool_call in choice.message.tool_calls:
                        import json

                        content_blocks.append(
                            {
                                "type": "tool_use",
                                "tool_use_id": tool_call.id,
                                "name": tool_call.function.name,
                                "input": json.loads(tool_call.function.arguments),
                            }
                        )

                return create_assistant_message(
                    content=content_blocks, cost_usd=cost_usd, duration_ms=duration_ms
                )

        else:
            raise NotImplementedError(f"Provider {model_profile.provider} not yet implemented")

    except Exception as e:
        # Return error message
        logger.error(f"Error querying AI model: {e}")
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
    # Work on a copy so external mutations (e.g., UI appending messages while consuming)
    # do not interfere with recursion or normalization.
    messages = list(messages)

    async def _check_permissions(
        tool: Tool[Any, Any], parsed_input: Any
    ) -> tuple[bool, Optional[str]]:
        """Check permissions for tool execution."""
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
        except Exception as exc:
            # Fail closed on any errors
            logger.error(f"Error checking permissions for tool '{tool.name}': {exc}")
            return False, None

    # Build full system prompt with context
    full_system_prompt = system_prompt
    if context:
        context_str = "\n".join(f"{k}: {v}" for k, v in context.items())
        full_system_prompt = f"{system_prompt}\n\nContext:\n{context_str}"

    assistant_message = await query_llm(
        messages,
        full_system_prompt,
        query_context.tools,
        query_context.max_thinking_tokens,
        query_context.model,
        query_context.abort_controller,
    )

    # Check for abort
    if query_context.abort_controller.is_set():
        yield create_assistant_message(INTERRUPT_MESSAGE)
        return

    yield assistant_message

    tool_block_count = 0
    if isinstance(assistant_message.message.content, list):
        tool_block_count = sum(
            1
            for block in assistant_message.message.content
            if hasattr(block, "type") and block.type == "tool_use"
        )
    logger.debug(
        f"[query] Assistant message received: "
        f"text_blocks={len(assistant_message.message.content) if isinstance(assistant_message.message.content, list) else 1}, "
        f"tool_use_blocks={tool_block_count}"
    )

    # Check for tool use
    tool_use_blocks = []
    if isinstance(assistant_message.message.content, list):
        for block in assistant_message.message.content:
            normalized_block = MessageContent(**block) if isinstance(block, dict) else block
            if hasattr(normalized_block, "type") and normalized_block.type == "tool_use":
                tool_use_blocks.append(normalized_block)

    # If no tool use, we're done
    if not tool_use_blocks:
        logger.debug("[query] No tool_use blocks; returning response to user.")
        return

    # Execute tools
    tool_results: List[UserMessage] = []

    logger.debug(f"[query] Executing {len(tool_use_blocks)} tool_use block(s).")

    for tool_use in tool_use_blocks:
        tool_name = tool_use.name
        tool_id = getattr(tool_use, "tool_use_id", None) or getattr(tool_use, "id", None) or ""
        tool_input = getattr(tool_use, "input", {}) or {}

        # Find the tool
        tool = next((t for t in query_context.tools if t.name == tool_name), None)

        if not tool:
            # Tool not found
            logger.warning(f"[query] Tool '{tool_name}' not found for tool_use_id={tool_id}")
            result_msg = create_user_message(
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "text": f"Error: Tool '{tool_name}' not found",
                        "is_error": True,
                    }
                ]
            )
            tool_results.append(result_msg)
            yield result_msg
            continue

        # Execute the tool
        tool_context = ToolUseContext(
            safe_mode=query_context.safe_mode,
            verbose=query_context.verbose,
            permission_checker=can_use_tool_fn,
        )

        try:
            # Parse input using tool's schema
            parsed_input = tool.input_schema(**tool_input)
            logger.debug(
                f"[query] tool_use_id={tool_id} name={tool_name} parsed_input="
                f"{str(parsed_input)[:500]}"
            )

            # Validate input before execution
            validation = await tool.validate_input(parsed_input, tool_context)
            if not validation.result:
                logger.debug(
                    f"[query] Validation failed for tool_use_id={tool_id}: {validation.message}"
                )
                result_msg = create_user_message(
                    [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "text": validation.message or "Tool input validation failed.",
                            "is_error": True,
                        }
                    ]
                )
                tool_results.append(result_msg)
                yield result_msg
                continue

            # Permission check (safe mode or custom checker)
            if query_context.safe_mode or can_use_tool_fn is not None:
                allowed, denial_message = await _check_permissions(tool, parsed_input)
                if not allowed:
                    logger.debug(
                        f"[query] Permission denied for tool_use_id={tool_id}: {denial_message}"
                    )
                    denial_text = denial_message or f"Permission denied for tool '{tool_name}'."
                    result_msg = create_user_message(
                        [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_id,
                                "text": denial_text,
                                "is_error": True,
                            }
                        ]
                    )
                    tool_results.append(result_msg)
                    yield result_msg
                    continue

            # Execute tool
            async for output in tool.call(parsed_input, tool_context):
                if isinstance(output, ToolProgress):
                    # Yield progress
                    progress = create_progress_message(
                        tool_use_id=tool_id,
                        sibling_tool_use_ids=set(
                            getattr(t, "tool_use_id", None) or getattr(t, "id", None) or ""
                            for t in tool_use_blocks
                        ),
                        content=output.content,
                    )
                    yield progress
                    logger.debug(f"[query] Progress from tool_use_id={tool_id}: {output.content}")
                elif isinstance(output, ToolResult):
                    # Tool completed
                    result_content = output.result_for_assistant or str(output.data)
                    result_msg = create_user_message(
                        [{"type": "tool_result", "tool_use_id": tool_id, "text": result_content}],
                        tool_use_result=output.data,
                    )
                    tool_results.append(result_msg)
                    yield result_msg
                    logger.debug(
                        f"[query] Tool completed tool_use_id={tool_id} name={tool_name} "
                        f"result_len={len(result_content)}"
                    )

        except Exception as e:
            # Tool execution failed
            logger.error(f"Error executing tool '{tool_name}': {e}")
            error_msg = create_user_message(
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "text": f"Error executing tool: {str(e)}",
                        "is_error": True,
                    }
                ]
            )
            tool_results.append(error_msg)
            yield error_msg

    # Check for abort after tools
    if query_context.abort_controller.is_set():
        yield create_assistant_message(INTERRUPT_MESSAGE_FOR_TOOL_USE)
        return

    # Continue conversation with tool results
    new_messages = messages + [assistant_message] + tool_results
    logger.debug(
        f"[query] Recursing with {len(new_messages)} messages after tools; "
        f"tool_results_count={len(tool_results)}"
    )

    async for msg in query(new_messages, system_prompt, context, query_context, can_use_tool_fn):
        yield msg
