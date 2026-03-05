"""Query execution for stdio protocol handler."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from pydantic import ValidationError

from ripperdoc.core.hooks.manager import hook_manager
from ripperdoc.core.hooks.state import bind_pending_message_queue, bind_hook_scopes
from ripperdoc.core.query import query
from ripperdoc.core.message_utils import estimate_cost_usd, resolve_model_profile
from ripperdoc.protocol.models import (
    JsonRpcErrorCodes,
    SamplingRequest,
    SamplingResult,
    UsageInfo,
    model_to_dict,
)
from ripperdoc.utils.asyncio_compat import asyncio_timeout
from ripperdoc.utils.mcp import format_mcp_instructions, load_mcp_servers_async
from ripperdoc.utils.messaging.messages import (
    create_assistant_message,
    create_hook_additional_context_message,
    create_hook_notice_payload,
    create_user_message,
    is_hook_notice_payload,
)
from ripperdoc.utils.sessions.session_history import SessionHistory

from .timeouts import (
    STDIO_HOOK_TIMEOUT_SEC,
    STDIO_QUERY_TIMEOUT_SEC,
    STDIO_WATCHDOG_INTERVAL_SEC,
)
from .error_codes import resolve_query_initialize_error_code
from .watchdog import OperationWatchdog

logger = logging.getLogger("ripperdoc.protocol.stdio.handler")


@dataclass
class _QueryRuntimeState:
    """Mutable runtime state for one stdio query."""

    start_time: float
    result_sent: bool = False
    num_turns: int = 0
    is_error: bool = False
    stop_reason: str = "endTurn"
    final_result_text: Optional[str] = None
    final_result_content: list[dict[str, Any]] | str | dict[str, Any] | None = None
    final_model: str | None = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_creation_tokens: int = 0

    def elapsed_ms(self) -> int:
        return int((time.time() - self.start_time) * 1000)

    def usage_tokens(self) -> dict[str, int]:
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "cache_read_input_tokens": self.total_cache_read_tokens,
            "cache_creation_input_tokens": self.total_cache_creation_tokens,
        }

    def build_usage_info(self) -> UsageInfo | None:
        if not (self.total_input_tokens or self.total_output_tokens):
            return None
        return UsageInfo(
            input_tokens=self.total_input_tokens,
            cache_creation_input_tokens=self.total_cache_creation_tokens,
            cache_read_input_tokens=self.total_cache_read_tokens,
            output_tokens=self.total_output_tokens,
        )


@dataclass
class _PreparedQuery:
    """Prepared inputs needed for query execution stage."""

    prompt: str
    messages: list[Any]
    session_history: SessionHistory
    system_prompt: str
    context: dict[str, Any]


class StdioQueryMixin:
    _session_history: SessionHistory | None
    _session_id: str | None
    _json_schema: dict[str, Any] | None
    _clear_context_after_turn: bool

    async def _handle_query(self, request: dict[str, Any], request_id: str) -> None:
        """Handle sampling/createMessage request from SDK."""
        state = _QueryRuntimeState(start_time=time.time())

        prepared = await self._prepare_query_stage(request, request_id, state)
        if prepared is None:
            return

        try:
            await self._execute_query_stage(prepared, state)
            await self._summarize_query_stage(request_id, state)
        except asyncio.TimeoutError as e:
            state.is_error = True
            state.stop_reason = "maxTokens"
            if not state.final_result_text:
                state.final_result_text = f"Query timed out after {STDIO_QUERY_TIMEOUT_SEC}s"
            await self._summarize_query_stage(request_id, state)
            self._finalize_query_stage(request_id, state)
            logger.warning(
                "[stdio] Query timed out: %s",
                e,
            )
            return
        except Exception as e:
            state.is_error = True
            state.stop_reason = "error"
            logger.error(f"[stdio] Handle query failed: {type(e).__name__}: {e}", exc_info=True)
            error_code = resolve_query_initialize_error_code(
                e,
                default=JsonRpcErrorCodes.InvalidParams,
            )
            await self._fail_query_request(
                request_id,
                state,
                str(e) or "Query failed",
                error_code=error_code,
            )
            return

        self._finalize_query_stage(request_id, state)

    async def _prepare_query_stage(
        self,
        request: dict[str, Any],
        request_id: str,
        state: _QueryRuntimeState,
    ) -> _PreparedQuery | None:
        """Prepare messages/system prompt and optional hook notices for this request."""
        if not self._initialized:
            await self._fail_query_request(request_id, state, "Not initialized")
            return None

        try:
            coerce_request = self._coerce_query_request(request)
            sampling_request = SamplingRequest.model_validate(coerce_request)
        except ValidationError as exc:
            error_code = resolve_query_initialize_error_code(
                exc,
                default=JsonRpcErrorCodes.InvalidParams,
            )
            await self._fail_query_request(
                request_id,
                state,
                f"Invalid sampling/createMessage request: {exc}",
                error_code=error_code,
            )
            return None
        except Exception as exc:
            error_code = resolve_query_initialize_error_code(
                exc,
                default=JsonRpcErrorCodes.InvalidParams,
            )
            await self._fail_query_request(
                request_id,
                state,
                f"Invalid sampling/createMessage request: {exc}",
                error_code=error_code,
            )
            return None

        if not sampling_request.messages:
            await self._fail_query_request(request_id, state, "messages is required")
            return None

        request_messages = [msg.model_dump(by_alias=True) for msg in sampling_request.messages]
        if not self._validate_tool_message_sequence(request_messages):
            await self._fail_query_request(
                request_id,
                state,
                "Invalid sampling message sequence: tool_result must match preceding tool_use blocks",
            )
            return None

        try:
            messages = self._coerce_sampling_messages(request_messages)
        except ValueError as exc:
            await self._fail_query_request(request_id, state, str(exc))
            return None

        prompt = self._extract_latest_user_prompt(messages, request_messages)
        if not prompt:
            await self._fail_query_request(request_id, state, "No user content found in messages")
            return None

        logger.info(
            "[stdio] Starting query handling",
            extra={
                "request_id": request_id,
                "prompt_length": len(prompt),
                "session_id": self._session_id,
                "conversation_messages": len(messages),
                "query_timeout": STDIO_QUERY_TIMEOUT_SEC,
            },
        )

        session_history = self._ensure_session_history()
        hook_manager.set_transcript_path(str(session_history.path))

        # Build base messages and context for this run.
        conversation_messages = list(self._conversation_messages) + list(messages)

        for message in messages:
            session_history.append(message)

        self._conversation_messages = conversation_messages

        hook_context_messages, hook_notices, blocked_reason = await self._collect_prepare_inputs(
            prompt
        )
        if blocked_reason:
            await self._fail_query_request(request_id, state, str(blocked_reason))
            return None
        if hook_context_messages:
            conversation_messages = [*hook_context_messages, *conversation_messages]
            for hook_message in hook_context_messages:
                session_history.append(hook_message)

        servers = await load_mcp_servers_async(self._project_path)
        mcp_instructions = format_mcp_instructions(servers)
        system_prompt = sampling_request.systemPrompt or self._resolve_system_prompt(
            self._query_context.tools if self._query_context else [],
            prompt,
            mcp_instructions,
            [],
        )

        await self._emit_hook_notices(hook_notices)

        request_context = dict(sampling_request.meta or {})

        return _PreparedQuery(
            prompt=prompt,
            messages=conversation_messages,
            session_history=session_history,
            system_prompt=system_prompt,
            context=request_context,
        )

    def _coerce_query_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Normalize query payload from JSON-RPC, control, or prompt shorthand."""
        request_body: dict[str, Any] = {}
        if isinstance(request, dict):
            request_body = dict(request)

        # Control-request style payload may be wrapped under `request`.
        wrapped_request = request_body.get("request")
        if isinstance(wrapped_request, dict):
            request_body = {**request_body, **wrapped_request}

        request_body.pop("type", None)
        request_body.pop("subtype", None)
        request_body.pop("method", None)

        # Some callers may include a raw `prompt`.
        prompt = request_body.get("prompt")
        if "messages" not in request_body and isinstance(prompt, str):
            text = prompt.strip()
            if text:
                request_body["messages"] = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": text,
                            }
                        ],
                    }
                ]
                request_body.setdefault("maxTokens", request_body.get("maxTokens", 1024) or 1024)

        if "messages" in request_body and not isinstance(request_body["messages"], list):
            del request_body["messages"]

        if request_body.get("maxTokens") is None:
            request_body["maxTokens"] = 1024

        return request_body

    async def _execute_query_stage(
        self,
        prepared: _PreparedQuery,
        state: _QueryRuntimeState,
    ) -> None:
        """Run query stream loop and update runtime state incrementally."""
        context: dict[str, Any] = dict(prepared.context)
        self._query_in_progress = True
        self._ensure_task_notification_poller()
        logger.debug(
            "[stdio] Preparing query execution",
            extra={
                "messages_count": len(prepared.messages),
                "system_prompt_length": len(prepared.system_prompt),
                "query_timeout": STDIO_QUERY_TIMEOUT_SEC,
            },
        )

        try:
            async with OperationWatchdog(
                timeout_sec=STDIO_QUERY_TIMEOUT_SEC,
                check_interval=STDIO_WATCHDOG_INTERVAL_SEC,
            ) as watchdog:
                async with asyncio_timeout(STDIO_QUERY_TIMEOUT_SEC):
                    async for message in query(
                        prepared.messages,
                        prepared.system_prompt,
                        context,
                        self._query_context or {},  # type: ignore[arg-type]
                        self._can_use_tool,
                    ):
                        watchdog.ping()
                        await self._process_stream_message(
                            message=message,
                            messages=prepared.messages,
                            session_history=prepared.session_history,
                            state=state,
                        )
            logger.debug("[stdio] Query loop ended successfully")
        except Exception as query_error:
            state.is_error = True
            if isinstance(query_error, asyncio.TimeoutError):
                state.stop_reason = "maxTokens"
                state.final_result_text = f"Query timed out after {STDIO_QUERY_TIMEOUT_SEC}s"
            elif isinstance(query_error, asyncio.CancelledError):
                state.stop_reason = "error"
                state.final_result_text = "Query was cancelled"
            else:
                state.stop_reason = "error"
                state.final_result_text = str(query_error)
            logger.error(
                f"[stdio] Query execution error: {type(query_error).__name__}: {query_error}",
                exc_info=True,
            )
            raise
        finally:
            self._query_in_progress = False

    async def _summarize_query_stage(
        self,
        request_id: str,
        state: _QueryRuntimeState,
    ) -> None:
        """Build and send SamplingResult."""
        if state.result_sent:
            return

        result_text = state.final_result_text or ""
        content = state.final_result_content
        if content is None:
            content = {"type": "text", "text": result_text}

        if isinstance(content, str):
            content = {"type": "text", "text": content}

        if isinstance(content, list):
            has_tool_use = any(
                isinstance(item, dict) and item.get("type") == "tool_use" for item in content
            )
            if has_tool_use:
                state.stop_reason = "toolUse"
        elif isinstance(content, dict) and content.get("type") == "tool_use":
            state.stop_reason = "toolUse"

        model_profile = self._query_context.model if self._query_context else "main"
        if state.final_model:
            model = state.final_model
        else:
            model = model_profile

        usage = state.build_usage_info()
        total_cost_usd = 0.0
        if usage is not None:
            try:
                total_cost_usd = estimate_cost_usd(resolve_model_profile(str(model)), usage.model_dump())
            except Exception as cost_error:
                logger.debug(
                    "[stdio] Failed to compute query cost summary: %s",
                    cost_error,
                    exc_info=True,
                )

        result = SamplingResult(
            model=str(model),
            stopReason=state.stop_reason,
            role="assistant",
            content=content,
            usage=usage,
        )

        result_subtype = "error_during_execution" if state.is_error else "success"
        session_id = self._session_id or str(uuid.uuid4())

        await self._write_message_stream(
            {
                "type": "result",
                "subtype": result_subtype,
                "duration_ms": state.elapsed_ms(),
                "duration_api_ms": state.elapsed_ms(),
                "is_error": state.is_error,
                "result": result_text,
                "num_turns": state.num_turns,
                "session_id": session_id,
                "stop_reason": state.stop_reason,
                "total_cost_usd": total_cost_usd,
            }
        )
        await self._write_control_response(request_id, response=model_to_dict(result))
        state.result_sent = True

    def _finalize_query_stage(self, request_id: str, state: _QueryRuntimeState) -> None:
        """Finalize request with terminal log event."""
        if self._clear_context_after_turn:
            self._conversation_messages = self._clear_context_messages(self._conversation_messages)
            self._clear_context_after_turn = False
        logger.info(
            "[stdio] Query completed",
            extra={
                "request_id": request_id,
                "duration_ms": state.elapsed_ms(),
                "num_turns": state.num_turns,
                "is_error": state.is_error,
                "input_tokens": state.total_input_tokens,
                "output_tokens": state.total_output_tokens,
            },
        )

    def _ensure_session_history(self) -> SessionHistory:
        """Return active session history object, creating it lazily."""
        if self._session_history is None:
            self._session_history = SessionHistory(
                self._project_path,
                self._session_id or str(uuid.uuid4()),
                session_persistence=getattr(self, "_session_persistence_enabled", True),
            )
        return self._session_history

    async def _collect_prepare_inputs(
        self,
        prompt: str,
    ) -> tuple[list[Any], list[dict[str, Any]], str | None]:
        """Collect hook notices and model-visible hook context messages before execution."""
        hook_context_messages: list[Any] = []
        hook_notices: list[dict[str, Any]] = []
        blocked_reason: str | None = None

        queue = self._query_context.pending_message_queue if self._query_context else None
        hook_scopes = self._query_context.hook_scopes if self._query_context else []
        with bind_pending_message_queue(queue), bind_hook_scopes(hook_scopes):
            if self._session_hook_messages:
                hook_context_messages.extend(self._session_hook_messages)

            try:
                async with asyncio_timeout(STDIO_HOOK_TIMEOUT_SEC):
                    prompt_hook_result = await hook_manager.run_user_prompt_submit_async(prompt)
                    if hasattr(prompt_hook_result, "should_block") and prompt_hook_result.should_block:
                        blocked_reason = (
                            prompt_hook_result.block_reason
                            if hasattr(prompt_hook_result, "block_reason")
                            else "Prompt blocked by hook."
                        )
                        return hook_context_messages, hook_notices, blocked_reason
                    if (
                        hasattr(prompt_hook_result, "system_message")
                        and prompt_hook_result.system_message
                    ):
                        hook_notices.append(
                            create_hook_notice_payload(
                                text=str(prompt_hook_result.system_message),
                                hook_event="UserPromptSubmit",
                            )
                        )
                    if (
                        hasattr(prompt_hook_result, "additional_context")
                        and prompt_hook_result.additional_context
                    ):
                        message = create_hook_additional_context_message(
                            str(prompt_hook_result.additional_context),
                            hook_name="UserPromptSubmit",
                            hook_event="UserPromptSubmit",
                        )
                        if message is not None:
                            hook_context_messages.append(message)
            except asyncio.TimeoutError:
                logger.warning(f"[stdio] Prompt submit hook timed out after {STDIO_HOOK_TIMEOUT_SEC}s")
            except Exception as e:
                logger.warning(f"[stdio] Prompt submit hook failed: {e}")

        return hook_context_messages, hook_notices, blocked_reason

    async def _emit_hook_notices(self, notices: list[dict[str, Any]]) -> None:
        """Stream prepared hook notices to SDK."""
        for notice in notices:
            stream_message = self._build_hook_notice_stream_message(
                str(notice.get("text", "")),
                str(notice.get("hook_event", "")),
                tool_name=notice.get("tool_name"),
                level=notice.get("level") or "info",
            )
            await self._write_message_stream(model_to_dict(stream_message))

    def _coerce_sampling_messages(self, messages: list[dict[str, Any]]) -> list[Any]:
        """Normalize `sampling/createMessage` message payload into internal Message models."""
        parsed_messages: list[Any] = []

        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("Each message must be an object")

            role = message.get("role")
            if role not in {"user", "assistant"}:
                raise ValueError(f"Unsupported message role: {role}")

            raw_content = message.get("content")
            if raw_content is None:
                raw_content = ""

            normalized_content: list[dict[str, Any]]
            if isinstance(raw_content, list):
                normalized_content = []
                for item in raw_content:
                    if isinstance(item, dict):
                        normalized_content.append(item)
                    else:
                        normalized_content.append({"type": "text", "text": str(item)})
            elif isinstance(raw_content, str):
                normalized_content = [{"type": "text", "text": raw_content}]
            else:
                normalized_content = [{"type": "text", "text": str(raw_content)}]

            if role == "user":
                parsed_messages.append(
                    create_user_message(
                        normalized_content,
                        parent_tool_use_id=message.get("parent_tool_use_id"),
                    )
                )
            else:
                parsed_messages.append(
                    create_assistant_message(
                        normalized_content,
                        model=str(message.get("model") or self._query_context.model if self._query_context else ""),
                        parent_tool_use_id=message.get("parent_tool_use_id"),
                    )
                )

        return parsed_messages

    def _validate_tool_message_sequence(self, messages: list[dict[str, Any]]) -> bool:
        """Validate tool_result/tool_use pairing across the final two messages."""
        if len(messages) < 2:
            return True

        last_message = messages[-1]
        previous_message = messages[-2]

        last_content = self._normalize_content_list(last_message.get("content"))
        previous_content = self._normalize_content_list(previous_message.get("content"))

        if not last_content:
            return True

        has_tool_result = any(
            item.get("type") == "tool_result" for item in last_content
        )
        has_tool_use = any(
            item.get("type") == "tool_use" for item in previous_content
        )

        if not has_tool_result:
            return True

        if any(item.get("type") != "tool_result" for item in last_content):
            return False

        if not has_tool_use:
            return False

        tool_use_ids = {
            str(item.get("id") or "")
            for item in previous_content
            if item.get("type") == "tool_use" and str(item.get("id") or "").strip()
        }
        tool_result_ids = {
            str(item.get("toolUseId") or item.get("tool_use_id") or item.get("id") or "")
            for item in last_content
            if item.get("type") == "tool_result" and str(
                item.get("toolUseId") or item.get("tool_use_id") or item.get("id") or ""
            ).strip()
        }

        if not tool_result_ids:
            return False
        if len(tool_use_ids) != len(tool_result_ids):
            return False
        return tool_result_ids == tool_use_ids

    def _extract_latest_user_prompt(self, messages: list[Any], request_messages: list[dict[str, Any]]) -> str | None:
        for request_message in reversed(request_messages):
            if request_message.get("role") != "user":
                continue

            content = request_message.get("content")
            if isinstance(content, str):
                text = content.strip()
                if text:
                    return text
            if isinstance(content, list):
                parts: list[str] = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        text = str(block.get("text") or "").strip()
                        if text:
                            parts.append(text)
                if parts:
                    return "\n".join(parts)

        for message in reversed(messages):
            if getattr(message, "type", None) != "user":
                continue
            text = self._extract_text_from_user_message(message)
            if text:
                return text
        return None

    def _extract_text_from_user_message(self, message: Any) -> str:
        msg_content = getattr(message, "message", None)
        if not msg_content:
            return ""
        content = getattr(msg_content, "content", "")
        if isinstance(content, str):
            return content.strip()
        if not isinstance(content, list):
            return ""
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text = str(block.get("text") or "").strip()
                    if text:
                        parts.append(text)
            elif getattr(block, "type", None) == "text":
                text = str(getattr(block, "text", "") or "").strip()
                if text:
                    parts.append(text)
        return "\n".join(parts)

    async def _process_stream_message(
        self,
        *,
        message: Any,
        messages: list[Any],
        session_history: SessionHistory,
        state: _QueryRuntimeState,
    ) -> None:
        """Process one streamed message from query loop."""
        msg_type = getattr(message, "type", None)
        logger.debug(
            f"[stdio] Received message of type: {msg_type}, "
            f"num_turns={state.num_turns}, "
            f"elapsed_ms={state.elapsed_ms()}"
        )

        if msg_type == "progress":
            await self._handle_progress_stream_message(message, state)
            return

        if msg_type == "assistant":
            self._track_assistant_usage(message, state)
            self._update_final_result_content(message, state)
            model_name = getattr(message, "model", None)
            if model_name:
                state.final_model = str(model_name)

        message_dict = self._convert_message_to_sdk(message)
        if message_dict is None:
            return
        await self._write_message_stream(message_dict)

        if msg_type in ("assistant", "user"):
            self._conversation_messages.append(message)
        if msg_type == "assistant":
            state.num_turns += 1

        messages.append(message)  # type: ignore[arg-type]
        session_history.append(message)  # type: ignore[arg-type]

    async def _handle_progress_stream_message(
        self,
        progress_message: Any,
        state: _QueryRuntimeState,
    ) -> None:
        """Handle progress stream payloads, including hook notices and subagent relays."""
        notice_content = getattr(progress_message, "content", None)
        if is_hook_notice_payload(notice_content) and isinstance(notice_content, dict):
            notice_payload: dict[str, Any] = notice_content
            stream_message = self._build_hook_notice_stream_message(
                str(notice_payload.get("text", "")),
                str(notice_payload.get("hook_event", "")),
                tool_name=notice_payload.get("tool_name"),
                level=notice_payload.get("level") or "info",
            )
            await self._write_message_stream(model_to_dict(stream_message))
            return

        if not getattr(progress_message, "is_subagent_message", False):
            return

        subagent_message = getattr(progress_message, "content", None)
        if not (subagent_message and hasattr(subagent_message, "type")):
            return

        logger.debug(
            "[stdio] Forwarding subagent message: type=%s",
            getattr(subagent_message, "type", "unknown"),
        )
        message_dict = self._convert_message_to_sdk(subagent_message)
        if message_dict is None:
            return
        await self._write_message_stream(message_dict)

        if getattr(subagent_message, "type", "") == "assistant":
            self._conversation_messages.append(subagent_message)
            state.num_turns += 1
            self._track_assistant_usage(subagent_message, state)

    def _track_assistant_usage(self, assistant_message: Any, state: _QueryRuntimeState) -> None:
        """Accumulate token usage from assistant-like messages."""
        state.total_input_tokens += getattr(assistant_message, "input_tokens", 0)
        state.total_output_tokens += getattr(assistant_message, "output_tokens", 0)
        state.total_cache_read_tokens += getattr(assistant_message, "cache_read_tokens", 0)
        state.total_cache_creation_tokens += getattr(assistant_message, "cache_creation_tokens", 0)

    def _update_final_result_content(self, assistant_message: Any, state: _QueryRuntimeState) -> None:
        """Backward-compatible state tracking for final text snippet."""
        msg_content = getattr(assistant_message, "message", None)
        if not msg_content:
            return

        content = getattr(msg_content, "content", None)
        if not content:
            return

        if isinstance(content, str):
            state.final_result_text = content
            state.final_result_content = {"type": "text", "text": content}
            return

        if not isinstance(content, list):
            return

        text_parts = self._extract_final_text_parts(content)
        state.final_result_text = "\n".join(text_parts)
        state.final_result_content = [
            block
            for block in self._convert_content_blocks_to_dict(content)
        ]

        for block in content:
            if (block.get("type") == "tool_use" if isinstance(block, dict) else getattr(block, "type", None) == "tool_use"):
                state.stop_reason = "toolUse"

    def _update_final_result_text(self, assistant_message: Any, state: _QueryRuntimeState) -> None:
        """Backward-compatible wrapper for older callers.

        The implementation keeps behavior aligned with previous method names by
        delegating to `_update_final_result_content`.
        """
        self._update_final_result_content(assistant_message, state)

    def _convert_content_blocks_to_dict(self, content_blocks: list[Any]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for block in content_blocks:
            if isinstance(block, dict):
                normalized.append(block)
                continue
            if getattr(block, "type", None):
                normalized_block = self._convert_content_block(block)
                if normalized_block:
                    normalized.append(normalized_block)
        return normalized

    def _convert_content_block(self, block: Any) -> dict[str, Any] | None:
        """Convert a message content block to SDK-compatible dictionary."""
        block_type = getattr(block, "type", None)

        if block_type == "text":
            return {
                "type": "text",
                "text": getattr(block, "text", None) or "",
            }

        if block_type == "thinking":
            return {
                "type": "thinking",
                "thinking": getattr(block, "thinking", None) or getattr(block, "text", None) or "",
                "signature": getattr(block, "signature", None),
            }

        if block_type == "tool_use":
            tool_id = getattr(block, "id", None) or getattr(block, "tool_use_id", None) or ""
            name = getattr(block, "name", None) or ""
            input_value = getattr(block, "input", None) or {}
            if hasattr(input_value, "model_dump"):
                input_value = input_value.model_dump()
            elif hasattr(input_value, "dict"):
                input_value = input_value.dict()
            if not isinstance(input_value, dict):
                input_value = {"value": str(input_value)}
            return {
                "type": "tool_use",
                "id": tool_id,
                "name": name,
                "input": input_value,
            }

        if block_type == "tool_result":
            text_value = getattr(block, "text", None)
            if not text_value:
                content_value = getattr(block, "content", None)
                if isinstance(content_value, list):
                    for item in content_value:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_value = item.get("text") or ""
                            break
                        if isinstance(item, str):
                            text_value = item
                            break
                elif content_value is not None:
                    text_value = content_value
            result_block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": getattr(block, "tool_use_id", None)
                or getattr(block, "id", None)
                or "",
                "content": str(text_value) if text_value is not None else "",
            }
            if getattr(block, "is_error", None) is not None:
                result_block["is_error"] = getattr(block, "is_error", None)
            return result_block

        if block_type == "image":
            return {
                "type": "image",
                "source": {
                    "type": getattr(block, "source_type", None) or "base64",
                    "media_type": getattr(block, "media_type", None) or "image/jpeg",
                    "data": getattr(block, "image_data", None) or "",
                },
            }

        block_dict: dict[str, Any] = {}
        if hasattr(block, "type"):
            block_dict["type"] = block.type
        if hasattr(block, "text"):
            block_dict["text"] = block.text
        if hasattr(block, "id"):
            block_dict["id"] = block.id
        if hasattr(block, "name"):
            block_dict["name"] = block.name
        if hasattr(block, "input"):
            block_dict["input"] = block.input
        if hasattr(block, "content"):
            block_dict["content"] = block.content
        if hasattr(block, "is_error"):
            block_dict["is_error"] = block.is_error
        return block_dict if block_dict else None

    def _clear_context_messages(self, messages: list[Any]) -> list[Any]:
        retained = [msg for msg in messages if getattr(msg, "type", None) in {"user", "assistant"}]
        if not retained:
            return []
        return retained[-4:]

    def _extract_final_text_parts(self, content_blocks: list[Any]) -> list[str]:
        """Extract user-visible text for the final result summary."""
        text_parts: list[str] = []
        for block in content_blocks:
            block_type = getattr(block, "type", None)
            if isinstance(block, dict):
                block_type = block.get("type")
                if block_type == "text":
                    text_parts.append(str(block.get("text", "")))
                    continue
                if block_type == "tool_use":
                    text_parts.clear()
                    continue
            if block_type == "text":
                text_parts.append(str(getattr(block, "text", "")))
            elif block_type == "tool_use":
                text_parts.clear()
        return text_parts

    def _normalize_content_list(self, content: Any) -> list[dict[str, Any]]:
        """Normalize message content to list[dict]."""
        normalized: list[dict[str, Any]] = []
        if content is None:
            return normalized
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        if isinstance(content, dict):
            return [content]
        if not isinstance(content, list):
            return [{"type": "text", "text": str(content)}]

        for item in content:
            if isinstance(item, dict):
                normalized.append(item)
            elif isinstance(item, str):
                normalized.append({"type": "text", "text": item})
            elif item is not None:
                normalized.append({"type": "text", "text": str(item)})
        return normalized

    async def _fail_query_request(
        self,
        request_id: str,
        state: _QueryRuntimeState,
        error_msg: str,
        error_code: int = JsonRpcErrorCodes.InvalidParams,
    ) -> None:
        """Send control error response."""
        del state
        try:
            await self._write_control_response(
                request_id,
                error={
                    "code": int(error_code),
                    "message": error_msg,
                },
            )
        except Exception as write_error:
            logger.error(
                f"[stdio] Failed to send control response: {write_error}",
                exc_info=True,
            )
