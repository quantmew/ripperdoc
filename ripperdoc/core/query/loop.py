"""Main query loop and LLM interaction helpers."""

import asyncio
import json
import os
import sys
import time
from asyncio import CancelledError
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Sequence,
    Union,
    cast,
)

from pydantic import ValidationError

from ripperdoc.core.config import ModelProfile, provider_protocol
from ripperdoc.core.hooks.manager import HookResult, hook_manager
from ripperdoc.core.hooks.state import bind_hook_scopes, bind_pending_message_queue
from ripperdoc.core.providers import ProviderClient, get_provider_client
from ripperdoc.core.message_utils import (
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
from ripperdoc.utils.teams import (
    get_active_team_name,
    list_teams,
    drain_team_inbox_messages,
    register_team_message_listener,
    unregister_team_message_listener,
)
from ripperdoc.utils.pending_messages import PendingMessageQueue
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
ConversationMessage = Union[UserMessage, AssistantMessage, ProgressMessage]


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
    explicit_mode = model_profile.thinking_mode
    if explicit_mode:
        return None if explicit_mode.lower() in ("disabled", "off") else explicit_mode

    base = (model_profile.api_base or "").lower()
    name = (model_profile.model or "").lower()

    thinking_patterns = [
        ("deepseek", lambda b, n: "deepseek" in b or n.startswith("deepseek")),
        ("qwen", lambda b, n: "dashscope" in b or "qwen" in n),
        ("openrouter", lambda b, n: "openrouter.ai" in b),
        ("gemini_openai", lambda b, n: "generativelanguage.googleapis.com" in b or n.startswith("gemini")),
        ("openai", lambda b, n: "openai" in b),
    ]

    for mode, matcher in thinking_patterns:
        if matcher(base, name):
            return mode

    return None


def _prepare_request_messages(
    messages: List[Union[UserMessage, AssistantMessage, ProgressMessage]],
    model_profile: ModelProfile,
) -> tuple[str, str, List[Union[UserMessage, AssistantMessage, ProgressMessage]]]:
    """Prepare protocol/tool-mode and conversation payload for provider request."""
    protocol = provider_protocol(model_profile.protocol)
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
        client: Optional[ProviderClient] = get_provider_client(model_profile.protocol)
    except RuntimeError as exc:
        duration_ms = (time.time() - start_time) * 1000
        return None, _create_api_error_message(
            str(exc),
            duration_ms=duration_ms,
            model=model_profile.model,
        )

    if client is None:
        duration_ms = (time.time() - start_time) * 1000
        provider_label = getattr(model_profile.protocol, "value", None) or str(
            model_profile.protocol
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
            "provider": getattr(model_profile.protocol, "value", str(model_profile.protocol)),
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
                    getattr(model_profile.protocol, "value", None) if model_profile else None
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
                "context_length_protocol": context_error.protocol,
                "context_length_error_code": context_error.error_code,
                "context_length_status_code": context_error.status_code,
            }
            logger.info(
                "[query_llm] Detected context-length error; consider compacting history",
                extra={
                    "protocol": context_error.protocol,
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


@dataclass
class _IterationPlan:
    """Resolved model/prompt state for one query iteration."""

    model_profile: ModelProfile
    full_system_prompt: str
    tools_for_model: List[Tool[Any, Any]]


@dataclass
class _AssistantWaitState:
    """Mutable state produced while waiting for query_llm to finish."""

    assistant_message: Optional[AssistantMessage] = None
    aborted: bool = False


def _compose_next_iteration_messages(
    base_messages: Sequence[ConversationMessage],
    *,
    assistant_message: Optional[AssistantMessage],
    tool_results: Sequence[UserMessage],
    extra_messages: Optional[Sequence[ConversationMessage]] = None,
) -> list[ConversationMessage]:
    """Build next-iteration history in a type-safe way."""
    next_messages: list[ConversationMessage] = list(base_messages)
    if assistant_message is not None:
        next_messages.append(assistant_message)
    next_messages.extend(tool_results)
    if extra_messages:
        next_messages.extend(extra_messages)
    return next_messages


def _resolve_query_team_name(query_context: QueryContext) -> Optional[str]:
    """Resolve the active team name for this query context."""
    context_team = (query_context.team_name or "").strip() if query_context.team_name else ""
    if context_team:
        return context_team

    env_team = os.getenv("RIPPERDOC_TEAM_NAME")
    if env_team and env_team.strip():
        return env_team.strip()

    disk_team = get_active_team_name()
    if disk_team:
        return disk_team

    teams = list_teams()
    if len(teams) == 1:
        return teams[0].name

    return None


def _resolve_query_team_listener_participant(query_context: QueryContext) -> str:
    """Select the listener participant key for this query context."""
    if query_context.teammate_name:
        return query_context.teammate_name
    if query_context.agent_id:
        return query_context.agent_id
    return "team-lead"


def _xml_attr_escape(value: str) -> str:
    """Escape special XML characters in attribute values."""
    return (
        value.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _wrap_teammate_message(
    *,
    sender_name: str,
    body: str,
    summary_text: str,
) -> str:
    """Wrap a teammate message in XML format."""
    teammate_id = _xml_attr_escape(sender_name or "teammate")
    summary_part = (
        f' summary="{_xml_attr_escape(summary_text)}"'
        if summary_text
        else ""
    )
    return (
        f'<teammate-message teammate_id="{teammate_id}"{summary_part}>\n'
        + body
        + "\n</teammate-message>"
    )


def _extract_shutdown_request_metadata(text: str, metadata_dict: dict) -> None:
    """Extract and merge shutdown request metadata from JSON payload."""
    try:
        parsed_payload = json.loads(text)
        if not isinstance(parsed_payload, dict):
            return
    except (TypeError, ValueError, json.JSONDecodeError):
        return

    mappings = {
        "request_id": ("request_id", "requestId"),
        "sender": ("sender", "from"),
        "content": ("content", "reason"),
    }

    for metadata_key, payload_keys in mappings.items():
        if metadata_dict.get(metadata_key):
            continue
        for payload_key in payload_keys:
            value = parsed_payload.get(payload_key)
            if value:
                metadata_dict[metadata_key] = value
                break


def _prioritize_shutdown_entries(
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Prioritize shutdown_request entries at the front of the list."""
    shutdown_entries: list[dict[str, Any]] = []
    normal_entries: list[dict[str, Any]] = []

    for candidate in entries:
        if isinstance(candidate, dict) and str(candidate.get("message_type")) == "shutdown_request":
            shutdown_entries.append(candidate)
        else:
            normal_entries.append(candidate)

    return shutdown_entries + normal_entries


def _normalize_entry_content(
    content: Any,
) -> str:
    """Normalize entry content to string."""
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(content)


def _inject_team_inbox_messages(
    team_name: str,
    participant: str,
    queue: PendingMessageQueue,
) -> int:
    """Load unread mailbox messages and enqueue them as pending user messages."""
    clean_team = (team_name or "").strip()
    clean_participant = (participant or "").strip()
    if not clean_team or not clean_participant:
        return 0

    inbox_entries = drain_team_inbox_messages(clean_team, clean_participant)
    if not inbox_entries:
        return 0

    ordered_entries = _prioritize_shutdown_entries(inbox_entries)

    injected = 0
    for entry in ordered_entries:
        if not isinstance(entry, dict):
            continue

        metadata = entry.get("metadata")
        metadata_dict = metadata if isinstance(metadata, dict) else {}
        team_name_value = entry.get("team_name") or clean_team
        message_type = str(
            entry.get("message_type")
            or metadata_dict.get("team_message_type")
            or "message"
        ).strip()
        sender = str(entry.get("sender") or metadata_dict.get("sender") or "").strip()
        summary = str(metadata_dict.get("summary") or "").strip()

        content = entry.get("content")
        if content is None:
            continue

        text = _normalize_entry_content(content)

        if message_type == "shutdown_request":
            _extract_shutdown_request_metadata(text, metadata_dict)

        if not text.lstrip().startswith("<teammate-message"):
            rendered_sender = sender or str(metadata_dict.get("sender") or "team-lead").strip()
            text = _wrap_teammate_message(
                sender_name=rendered_sender or "team-lead",
                body=text,
                summary_text=summary,
            )

        user_message = create_user_message(text)
        if metadata_dict:
            user_message.message.metadata.update(metadata_dict)
        user_message.message.metadata.setdefault(
            "team",
            {
                "team_name": team_name_value,
                "sender": entry.get("sender"),
                "recipient": entry.get("recipient"),
                "message_type": entry.get("message_type"),
                "message_id": entry.get("id"),
                "created_at": entry.get("created_at"),
            },
        )
        user_message.message.metadata.setdefault("team_message_type", entry.get("message_type"))
        user_message.message.metadata.setdefault("team_name", team_name_value)
        user_message.message.metadata.setdefault("sender", entry.get("sender"))
        user_message.message.metadata.setdefault("recipient", entry.get("recipient"))
        user_message.message.metadata.setdefault("request_id", entry.get("request_id"))
        user_message.message.metadata.setdefault("approve", entry.get("approve"))

        queue.enqueue(user_message)
        injected += 1

    return injected


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


def _append_changed_file_notice_if_needed(
    messages: List[Union[UserMessage, AssistantMessage, ProgressMessage]],
    query_context: QueryContext,
) -> None:
    """Inject a user-visible changed-file notice at iteration start when needed."""
    change_notices = detect_changed_files(query_context.file_state_cache)
    if change_notices:
        messages.append(create_user_message(_format_changed_file_notice(change_notices)))


def _build_iteration_plan(
    *,
    system_prompt: str,
    context: Dict[str, str],
    query_context: QueryContext,
) -> _IterationPlan:
    """Resolve model/tool mode and build the full system prompt for one iteration."""
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
    return _IterationPlan(
        model_profile=model_profile,
        full_system_prompt=full_system_prompt,
        tools_for_model=tools_for_model,
    )


async def _enqueue_stream_progress(
    progress_queue: asyncio.Queue[Optional[ProgressMessage]],
    chunk: str,
) -> None:
    """Convert streamed text chunks to progress messages and enqueue them."""
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
            try:
                await asyncio.wait_for(progress_queue.put(msg), timeout=0.5)
            except asyncio.TimeoutError:
                logger.warning("[query] Progress queue full after timeout, dropping chunk")
    except (RuntimeError, ValueError) as exc:
        logger.warning("[query] Failed to enqueue stream progress chunk: %s", exc)


def _resolve_query_llm_callable() -> Callable[..., Coroutine[Any, Any, AssistantMessage]]:
    """Resolve query_llm from package namespace for monkeypatch compatibility in tests."""
    query_module = sys.modules.get("ripperdoc.core.query")
    if query_module:
        resolved = getattr(query_module, "query_llm", query_llm)
        return cast(Callable[..., Coroutine[Any, Any, AssistantMessage]], resolved)
    return query_llm


async def _cancel_assistant_task(assistant_task: asyncio.Task[AssistantMessage]) -> None:
    """Cancel an assistant task and swallow expected cancellation errors."""
    assistant_task.cancel()
    try:
        await assistant_task
    except CancelledError:
        pass


async def _cancel_pending_waiters(pending: set[asyncio.Task[Any]], assistant_task: asyncio.Task[Any]) -> None:
    """Cancel helper waiters while leaving the primary assistant task intact."""
    for task in pending:
        if task is assistant_task:
            continue
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


async def _poll_progress_or_completion(
    *,
    assistant_task: asyncio.Task[AssistantMessage],
    progress_queue: asyncio.Queue[Optional[ProgressMessage]],
    abort_controller: asyncio.Event,
) -> tuple[Optional[ProgressMessage], Optional[AssistantMessage], bool]:
    """Poll one progress/assistant/abort event.

    Returns:
        (progress, assistant_message, should_continue_loop)
    """
    try:
        progress = progress_queue.get_nowait()
        return progress, None, False
    except asyncio.QueueEmpty:
        waiter = asyncio.create_task(progress_queue.get())
        abort_waiter = asyncio.create_task(abort_controller.wait())
        done, pending = await asyncio.wait(
            {assistant_task, waiter, abort_waiter},
            return_when=asyncio.FIRST_COMPLETED,
        )
        await _cancel_pending_waiters(pending, assistant_task)
        if abort_waiter in done:
            return None, None, True
        if assistant_task in done:
            return None, await assistant_task, False
        return waiter.result(), None, False


def _drain_progress_queue(
    progress_queue: asyncio.Queue[Optional[ProgressMessage]],
) -> List[ProgressMessage]:
    """Drain any residual progress chunks left in queue order."""
    residuals: List[ProgressMessage] = []
    while not progress_queue.empty():
        residual = progress_queue.get_nowait()
        if residual:
            residuals.append(residual)
    return residuals


async def _wait_for_assistant_with_progress(
    *,
    assistant_task: asyncio.Task[AssistantMessage],
    progress_queue: asyncio.Queue[Optional[ProgressMessage]],
    query_context: QueryContext,
    model_name: Optional[str],
    out: _AssistantWaitState,
) -> AsyncGenerator[Union[ProgressMessage, AssistantMessage], None]:
    """Yield progress while waiting for query_llm, then emit final assistant (or interrupt)."""
    while True:
        if query_context.abort_controller.is_set():
            await _cancel_assistant_task(assistant_task)
            out.aborted = True
            yield create_assistant_message(INTERRUPT_MESSAGE, model=model_name)
            return

        if assistant_task.done():
            out.assistant_message = await assistant_task
            break

        progress, completed_message, should_continue = await _poll_progress_or_completion(
            assistant_task=assistant_task,
            progress_queue=progress_queue,
            abort_controller=query_context.abort_controller,
        )
        if should_continue:
            continue
        if completed_message is not None:
            out.assistant_message = completed_message
            break
        if progress:
            yield progress

    for residual in _drain_progress_queue(progress_queue):
        yield residual

    if out.assistant_message is not None:
        yield out.assistant_message


async def _process_iteration_assistant_message(
    *,
    assistant_message: AssistantMessage,
    messages: List[Union[UserMessage, AssistantMessage, ProgressMessage]],
    context: Dict[str, str],
    query_context: QueryContext,
    can_use_tool_fn: Optional[ToolPermissionCallable],
    result: IterationResult,
    model_name: Optional[str],
) -> AsyncGenerator[Union[UserMessage, AssistantMessage, ProgressMessage], None]:
    """Process assistant tool calls, including hooks, permissions, and tool execution."""
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

    if query_context.abort_controller.is_set():
        yield create_assistant_message(INTERRUPT_MESSAGE_FOR_TOOL_USE, model=model_name)
        result.tool_results = prepared.tool_results
        result.should_stop = True
        return

    result.tool_results = prepared.tool_results


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
        if not tool_use.name:
            continue
        async for msg in _prepare_single_iteration_tool_call(
            tool_use=tool_use,
            sibling_ids=sibling_ids,
            messages=messages,
            context=context,
            query_context=query_context,
            can_use_tool_fn=can_use_tool_fn,
            out=out,
        ):
            yield msg

        if out.permission_denied:
            break


async def _prepare_single_iteration_tool_call(
    *,
    tool_use: MessageContent,
    sibling_ids: set[str],
    messages: List[Union[UserMessage, AssistantMessage, ProgressMessage]],
    context: Dict[str, str],
    query_context: QueryContext,
    can_use_tool_fn: Optional[ToolPermissionCallable],
    out: _PreparedToolCalls,
) -> AsyncGenerator[Union[UserMessage, ProgressMessage], None]:
    """Prepare one tool_use block for execution (resolve, validate, hook, permissions)."""
    tool_name = cast(str, tool_use.name)
    tool_use_id = getattr(tool_use, "tool_use_id", None) or getattr(tool_use, "id", None) or ""
    tool_input = _normalize_tool_input_payload(getattr(tool_use, "input", {}) or {})

    tool, resolution_error = _resolve_tool_for_iteration_call(
        query_context=query_context,
        tool_name=tool_name,
        tool_use_id=tool_use_id,
    )
    if resolution_error is not None:
        out.tool_results.append(resolution_error)
        yield resolution_error
        return
    tool = cast(Tool[Any, Any], tool)

    try:
        parsed_input, tool_context, tool_input_dict, validation_error = (
            await _parse_and_validate_tool_input_for_call(
                tool=tool,
                tool_name=tool_name,
                tool_use_id=tool_use_id,
                tool_input=tool_input,
                query_context=query_context,
                can_use_tool_fn=can_use_tool_fn,
                messages=messages,
            )
        )
        if validation_error is not None:
            out.tool_results.append(validation_error)
            yield validation_error
            return

        pre_result, pre_progress = await _run_pre_tool_use_hook_with_progress(
            tool_name=tool_name,
            tool_use_id=tool_use_id,
            tool_input_dict=tool_input_dict,
            sibling_ids=sibling_ids,
        )
        for message in pre_progress:
            yield message
        if pre_result.should_block or not pre_result.should_continue:
            reason = pre_result.block_reason or pre_result.stop_reason or f"Blocked by hook: {tool_name}"
            result_msg = tool_result_message(tool_use_id, f"Hook blocked: {reason}", is_error=True)
            out.tool_results.append(result_msg)
            yield result_msg
            return

        parsed_input, tool_input_dict, hook_input_error = await _apply_pre_hook_input_update(
            pre_result=pre_result,
            tool=tool,
            parsed_input=parsed_input,
            tool_context=tool_context,
            tool_name=tool_name,
            tool_use_id=tool_use_id,
            tool_input_dict=tool_input_dict,
        )
        if hook_input_error is not None:
            out.tool_results.append(hook_input_error)
            yield hook_input_error
            return

        for notice in _collect_pre_tool_hook_notices(
            pre_result=pre_result,
            context=context,
            tool_name=tool_name,
            tool_use_id=tool_use_id,
            sibling_ids=sibling_ids,
        ):
            yield notice

        parsed_input, permission_denied, permission_error = await _apply_permission_updates(
            tool=tool,
            parsed_input=parsed_input,
            query_context=query_context,
            can_use_tool_fn=can_use_tool_fn,
            pre_result=pre_result,
            tool_context=tool_context,
            tool_name=tool_name,
            tool_use_id=tool_use_id,
        )
        if permission_error is not None:
            out.tool_results.append(permission_error)
            out.permission_denied = out.permission_denied or permission_denied
            yield permission_error
            return

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
    except CancelledError:
        raise
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


def _resolve_tool_for_iteration_call(
    *,
    query_context: QueryContext,
    tool_name: str,
    tool_use_id: str,
) -> tuple[Optional[Tool[Any, Any]], Optional[UserMessage]]:
    """Resolve tool instance for one tool_use call and return user-facing error if missing."""
    tool, missing_msg = _resolve_tool(query_context.tool_registry, tool_name, tool_use_id)
    if missing_msg:
        logger.warning(f"[query] Tool '{tool_name}' not found for tool_use_id={tool_use_id}")
        return None, missing_msg
    if tool is None:
        raise RuntimeError(f"Tool '{tool_name}' resolved to None unexpectedly")
    return tool, None


async def _parse_and_validate_tool_input_for_call(
    *,
    tool: Tool[Any, Any],
    tool_name: str,
    tool_use_id: str,
    tool_input: Dict[str, Any],
    query_context: QueryContext,
    can_use_tool_fn: Optional[ToolPermissionCallable],
    messages: List[Union[UserMessage, AssistantMessage, ProgressMessage]],
) -> tuple[Any, ToolUseContext, Dict[str, Any], Optional[UserMessage]]:
    """Parse tool input and run the first validation pass."""
    try:
        parsed_input = tool.input_schema(**tool_input)
    except ValidationError as ve:
        detail_text = format_pydantic_errors(ve)
        return (
            {},
            ToolUseContext(
                message_id=tool_use_id,
                agent_id=query_context.agent_id,
                team_name=query_context.team_name,
                teammate_name=query_context.teammate_name,
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
            ),
            {},
            tool_result_message(
                tool_use_id,
                f"Invalid input for tool '{tool_name}': {detail_text}",
                is_error=True,
            ),
        )
    logger.debug(
        f"[query] tool_use_id={tool_use_id} name={tool_name} parsed_input="
        f"{str(parsed_input)[:500]}"
    )
    tool_context = ToolUseContext(
        message_id=tool_use_id,
        agent_id=query_context.agent_id,
        team_name=query_context.team_name,
        teammate_name=query_context.teammate_name,
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
        logger.debug(f"[query] Validation failed for tool_use_id={tool_use_id}: {validation.message}")
        return (
            parsed_input,
            tool_context,
            {},
            tool_result_message(
                tool_use_id,
                validation.message or "Tool input validation failed.",
                is_error=True,
            ),
        )
    tool_input_dict = (
        parsed_input.model_dump()
        if hasattr(parsed_input, "model_dump")
        else dict(parsed_input)
        if isinstance(parsed_input, dict)
        else {}
    )
    return parsed_input, tool_context, tool_input_dict, None


def _collect_pre_tool_hook_notices(
    *,
    pre_result: HookResult,
    context: Dict[str, str],
    tool_name: str,
    tool_use_id: str,
    sibling_ids: set[str],
) -> List[Union[UserMessage, ProgressMessage]]:
    """Apply pre-hook context updates and build any user-visible notices."""
    notices: List[Union[UserMessage, ProgressMessage]] = []
    if pre_result.additional_context:
        _append_hook_context(context, f"PreToolUse:{tool_name}", pre_result.additional_context)
    if pre_result.system_message:
        notices.append(
            create_hook_notice_message(
                str(pre_result.system_message),
                hook_event="PreToolUse",
                tool_name=tool_name,
                tool_use_id=tool_use_id,
                sibling_tool_use_ids=sibling_ids,
            )
        )
    return notices


async def _run_pre_tool_use_hook_with_progress(
    *,
    tool_name: str,
    tool_use_id: str,
    tool_input_dict: Dict[str, Any],
    sibling_ids: set[str],
) -> tuple[HookResult, List[ProgressMessage]]:
    """Run PreToolUse hook and collect emitted progress messages."""
    progress_messages: List[ProgressMessage] = []
    pre_result: Optional[HookResult] = None
    async for item in _run_hook_call_with_status(
        hook_manager.run_pre_tool_use_async(tool_name, tool_input_dict, tool_use_id=tool_use_id),
        tool_use_id,
        sibling_ids,
    ):
        if isinstance(item, ProgressMessage):
            progress_messages.append(item)
        else:
            pre_result = item
    return pre_result or HookResult([]), progress_messages


async def _apply_pre_hook_input_update(
    *,
    pre_result: HookResult,
    tool: Tool[Any, Any],
    parsed_input: Any,
    tool_context: ToolUseContext,
    tool_name: str,
    tool_use_id: str,
    tool_input_dict: Dict[str, Any],
) -> tuple[Any, Dict[str, Any], Optional[UserMessage]]:
    """Apply PreToolUse updated_input (if any) and re-validate."""
    if not pre_result.updated_input:
        return parsed_input, tool_input_dict, None

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
        return (
            parsed_input,
            tool_input_dict,
            tool_result_message(
                tool_use_id,
                f"Invalid PreToolUse-updated input for tool '{tool_name}': {detail_text}",
                is_error=True,
            ),
        )
    except (TypeError, ValueError) as exc:
        logger.warning(
            "[query] Failed to apply updated input from PreToolUse hook: %s",
            exc,
            extra={"tool_use_id": tool_use_id},
        )
        return (
            parsed_input,
            tool_input_dict,
            tool_result_message(
                tool_use_id,
                f"Invalid PreToolUse-updated input for tool '{tool_name}'.",
                is_error=True,
            ),
        )

    validation = await tool.validate_input(parsed_input, tool_context)
    if not validation.result:
        return (
            parsed_input,
            tool_input_dict,
            tool_result_message(
                tool_use_id,
                validation.message or "Tool input validation failed.",
                is_error=True,
            ),
        )
    return parsed_input, tool_input_dict, None


async def _apply_permission_updates(
    *,
    tool: Tool[Any, Any],
    parsed_input: Any,
    query_context: QueryContext,
    can_use_tool_fn: Optional[ToolPermissionCallable],
    pre_result: HookResult,
    tool_context: ToolUseContext,
    tool_name: str,
    tool_use_id: str,
) -> tuple[Any, bool, Optional[UserMessage]]:
    """Run permission gate and apply permission-updated input when provided."""
    force_permission_prompt = pre_result.should_ask
    bypass_permissions = pre_result.should_allow and not force_permission_prompt
    should_check_permissions = not bypass_permissions and (
        not query_context.yolo_mode or can_use_tool_fn is not None or force_permission_prompt
    )
    if not should_check_permissions:
        return parsed_input, False, None

    allowed, denial_message, updated_input = await _check_tool_permissions(
        tool,
        parsed_input,
        query_context,
        can_use_tool_fn,
        force_prompt=force_permission_prompt,
    )
    if not allowed:
        logger.debug(f"[query] Permission denied for tool_use_id={tool_use_id}: {denial_message}")
        denial_text = denial_message or f"User aborted the tool invocation: {tool_name}"
        return parsed_input, True, tool_result_message(tool_use_id, denial_text, is_error=True)

    if not updated_input:
        return parsed_input, False, None

    try:
        normalized_input = _normalize_tool_input_payload(updated_input)
        parsed_input = tool.input_schema(**normalized_input)
    except ValidationError as ve:
        detail_text = format_pydantic_errors(ve)
        return (
            parsed_input,
            False,
            tool_result_message(
                tool_use_id,
                f"Invalid permission-updated input for tool '{tool_name}': {detail_text}",
                is_error=True,
            ),
        )

    validation = await tool.validate_input(parsed_input, tool_context)
    if not validation.result:
        return (
            parsed_input,
            False,
            tool_result_message(
                tool_use_id,
                validation.message or "Tool input validation failed.",
                is_error=True,
            ),
        )
    return parsed_input, False, None


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

    _append_changed_file_notice_if_needed(messages, query_context)
    plan = _build_iteration_plan(
        system_prompt=system_prompt,
        context=context,
        query_context=query_context,
    )

    progress_queue: asyncio.Queue[Optional[ProgressMessage]] = asyncio.Queue(maxsize=1000)
    query_llm_fn = _resolve_query_llm_callable()

    assistant_task: asyncio.Task[AssistantMessage] = asyncio.create_task(
        query_llm_fn(
            messages,
            plan.full_system_prompt,
            plan.tools_for_model,
            query_context.max_thinking_tokens,
            query_context.model,
            query_context.abort_controller,
            progress_callback=lambda chunk: _enqueue_stream_progress(progress_queue, chunk),
            request_timeout=DEFAULT_REQUEST_TIMEOUT_SEC,
            max_retries=MAX_LLM_RETRIES,
            stream=True,
        )
    )

    logger.debug("[query] Created query_llm task, waiting for response...")
    wait_state = _AssistantWaitState()
    async for item in _wait_for_assistant_with_progress(
        assistant_task=assistant_task,
        progress_queue=progress_queue,
        query_context=query_context,
        model_name=plan.model_profile.model,
        out=wait_state,
    ):
        if getattr(item, "type", None) == "progress":
            yield item
            continue
        if wait_state.aborted:
            yield item
            result.should_stop = True
            return
        wait_state.assistant_message = cast(AssistantMessage, item)

    assistant_message = wait_state.assistant_message
    if assistant_message is None:
        raise RuntimeError("assistant_message was unexpectedly None after LLM query")

    result.assistant_message = assistant_message
    yield assistant_message

    async for msg in _process_iteration_assistant_message(
        assistant_message=assistant_message,
        messages=messages,
        context=context,
        query_context=query_context,
        can_use_tool_fn=can_use_tool_fn,
        result=result,
        model_name=plan.model_profile.model,
    ):
        yield msg


async def query(
    messages: List[ConversationMessage],
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
    team_name = _resolve_query_team_name(query_context)
    listener_token: Optional[tuple[str, str, int]] = None
    listener_participant = _resolve_query_team_listener_participant(query_context)
    if team_name:
        try:
            listener_token = register_team_message_listener(
                team_name=team_name,
                participant=listener_participant,
                queue=query_context.pending_message_queue,
            )
        except ValueError as exc:
            logger.debug(
                "[query] Team message listener registration skipped: %s: %s",
                type(exc).__name__,
                exc,
                extra={"team_name": team_name, "participant": listener_participant},
            )
    # Resolve model once for use in messages (e.g., max iterations, errors)
    model_profile = resolve_model_profile(query_context.model)
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

        try:
            for iteration in range(1, max_iterations + 1):
                # Inject any pending messages queued by background events or user interjections
                if team_name:
                    _inject_team_inbox_messages(
                        team_name,
                        listener_participant,
                        query_context.pending_message_queue,
                    )
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
                        next_messages = _compose_next_iteration_messages(
                            messages,
                            assistant_message=result.assistant_message,
                            tool_results=result.tool_results,
                            extra_messages=trailing_pending,
                        )
                        for pending in trailing_pending:
                            yield pending
                        messages = next_messages
                        continue
                    return

                # Update messages for next iteration
                messages = _compose_next_iteration_messages(
                    messages,
                    assistant_message=result.assistant_message,
                    tool_results=result.tool_results,
                )

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
        finally:
            if listener_token is not None and team_name:
                unregister_team_message_listener(
                    team_name=team_name,
                    participant=listener_participant,
                    queue=query_context.pending_message_queue,
                )
