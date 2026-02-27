"""Query execution for stdio protocol handler."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import traceback
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from ripperdoc.core.hooks.manager import hook_manager
from ripperdoc.core.hooks.state import bind_pending_message_queue, bind_hook_scopes
from ripperdoc.core.query import query
from ripperdoc.core.message_utils import estimate_cost_usd, resolve_model_profile
from ripperdoc.protocol.models import ResultMessage, UsageInfo, model_to_dict
from ripperdoc.utils.asyncio_compat import asyncio_timeout
from ripperdoc.utils.mcp import format_mcp_instructions, load_mcp_servers_async
from ripperdoc.utils.messages import (
    create_hook_notice_payload,
    create_user_message,
    is_hook_notice_payload,
)
from ripperdoc.utils.session_history import SessionHistory

from .timeouts import (
    STDIO_HOOK_TIMEOUT_SEC,
    STDIO_QUERY_TIMEOUT_SEC,
    STDIO_WATCHDOG_INTERVAL_SEC,
)
from .watchdog import OperationWatchdog

logger = logging.getLogger("ripperdoc.protocol.stdio.handler")


@dataclass
class _QueryRuntimeState:
    """Mutable runtime state for one stdio query."""

    start_time: float
    result_message_sent: bool = False
    num_turns: int = 0
    is_error: bool = False
    final_result_text: Optional[str] = None
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


class StdioQueryMixin:
    _session_history: SessionHistory | None
    _session_id: str | None
    _json_schema: dict[str, Any] | None
    _clear_context_after_turn: bool

    async def _handle_query(self, request: dict[str, Any], request_id: str) -> None:
        """Handle query request from SDK with comprehensive timeout and error handling."""
        state = _QueryRuntimeState(start_time=time.time())

        try:
            prepared = await self._prepare_query_stage(request, request_id, state)
            if prepared is None:
                return

            await self._execute_query_stage(prepared, state)
            await self._summarize_query_stage(state)
            self._finalize_query_stage(request_id, state)
        except Exception as e:
            logger.error(f"[stdio] Handle query failed: {type(e).__name__}: {e}", exc_info=True)
            await self._write_control_response(request_id, error=str(e))

            # Ensure ResultMessage is sent even if everything else fails.
            if not state.result_message_sent:
                try:
                    await self._send_error_result(state, str(e), e)
                except Exception as send_error:
                    logger.error(
                        f"[stdio] Critical: Failed to send error ResultMessage: {send_error}",
                        exc_info=True,
                    )

    async def _prepare_query_stage(
        self,
        request: dict[str, Any],
        request_id: str,
        state: _QueryRuntimeState,
    ) -> _PreparedQuery | None:
        """Prepare messages/system prompt and emit query-start stream events."""
        if not self._initialized:
            await self._fail_query_request(request_id, state, "Not initialized")
            return None

        prompt = request.get("prompt", "")
        if not prompt:
            await self._fail_query_request(request_id, state, "Prompt is required")
            return None

        logger.info(
            "[stdio] Starting query handling",
            extra={
                "request_id": request_id,
                "prompt_length": len(prompt),
                "session_id": self._session_id,
                "conversation_messages": len(self._conversation_messages),
                "query_timeout": STDIO_QUERY_TIMEOUT_SEC,
            },
        )

        session_history = self._ensure_session_history()
        hook_manager.set_transcript_path(str(session_history.path))

        # Prepare user turn and base message list.
        user_message = create_user_message(prompt)
        self._conversation_messages.append(user_message)
        session_history.append(user_message)
        messages = list(self._conversation_messages)

        additional_instructions, hook_notices, blocked_reason = await self._collect_prepare_inputs(prompt)
        if blocked_reason:
            await self._fail_query_request(request_id, state, str(blocked_reason))
            return None

        servers = await load_mcp_servers_async(self._project_path)
        mcp_instructions = format_mcp_instructions(servers)
        system_prompt = self._resolve_system_prompt(
            self._query_context.tools if self._query_context else [],
            prompt,
            mcp_instructions,
            additional_instructions,
        )

        await self._write_control_response(
            request_id,
            response={"status": "querying", "session_id": self._session_id},
        )
        await self._emit_hook_notices(hook_notices)

        return _PreparedQuery(
            prompt=prompt,
            messages=messages,
            session_history=session_history,
            system_prompt=system_prompt,
        )

    async def _execute_query_stage(self, prepared: _PreparedQuery, state: _QueryRuntimeState) -> None:
        """Run query stream loop and update runtime state incrementally."""
        context: dict[str, Any] = {}
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
        except asyncio.TimeoutError:
            logger.error(f"[stdio] Query execution timed out after {STDIO_QUERY_TIMEOUT_SEC}s")
            await self._send_error_result(state, f"Query timed out after {STDIO_QUERY_TIMEOUT_SEC}s")
        except asyncio.CancelledError:
            logger.warning("[stdio] Query was cancelled")
            await self._send_error_result(state, "Query was cancelled")
        except Exception as query_error:
            state.is_error = True
            logger.error(
                f"[stdio] Query execution error: {type(query_error).__name__}: {query_error}",
                exc_info=True,
            )
            await self._send_error_result(state, str(query_error), query_error)

    async def _summarize_query_stage(self, state: _QueryRuntimeState) -> None:
        """Build and send final success ResultMessage when execution succeeded."""
        if state.is_error or state.result_message_sent:
            return

        logger.debug("[stdio] Building usage info")
        model_pointer = self._query_context.model if self._query_context else "main"
        model_profile = resolve_model_profile(model_pointer)
        total_cost_usd = estimate_cost_usd(model_profile, state.usage_tokens())

        result_message = ResultMessage(
            duration_ms=state.elapsed_ms(),
            duration_api_ms=state.elapsed_ms(),
            is_error=state.is_error,
            subtype="success",
            num_turns=state.num_turns,
            session_id=self._session_id or "",
            total_cost_usd=round(total_cost_usd, 8) if total_cost_usd > 0 else None,
            usage=state.build_usage_info(),
            result=state.final_result_text,
            structured_output=self._coerce_structured_output(state.final_result_text),
        )
        await self._send_final_result(state, result_message)

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
                "result_sent": state.result_message_sent,
            },
        )

    def _ensure_session_history(self) -> SessionHistory:
        """Return active session history object, creating it lazily."""
        if self._session_history is None:
            self._session_history = SessionHistory(
                self._project_path,
                self._session_id or str(uuid.uuid4()),
            )
        return self._session_history

    async def _collect_prepare_inputs(
        self,
        prompt: str,
    ) -> tuple[list[str], list[dict[str, Any]], str | None]:
        """Collect hook notices and additional system instructions before execution."""
        additional_instructions: list[str] = []
        hook_notices: list[dict[str, Any]] = []
        blocked_reason: str | None = None

        queue = self._query_context.pending_message_queue if self._query_context else None
        hook_scopes = self._query_context.hook_scopes if self._query_context else []
        with bind_pending_message_queue(queue), bind_hook_scopes(hook_scopes):
            if self._session_hook_contexts:
                additional_instructions.extend(self._session_hook_contexts)

            try:
                async with asyncio_timeout(STDIO_HOOK_TIMEOUT_SEC):
                    prompt_hook_result = await hook_manager.run_user_prompt_submit_async(prompt)
                    if hasattr(prompt_hook_result, "should_block") and prompt_hook_result.should_block:
                        blocked_reason = (
                            prompt_hook_result.block_reason
                            if hasattr(prompt_hook_result, "block_reason")
                            else "Prompt blocked by hook."
                        )
                        return additional_instructions, hook_notices, blocked_reason
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
                        additional_instructions.append(str(prompt_hook_result.additional_context))
            except asyncio.TimeoutError:
                logger.warning(f"[stdio] Prompt submit hook timed out after {STDIO_HOOK_TIMEOUT_SEC}s")
            except Exception as e:
                logger.warning(f"[stdio] Prompt submit hook failed: {e}")

        return additional_instructions, hook_notices, blocked_reason

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
            self._update_final_result_text(message, state)

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

    def _update_final_result_text(self, assistant_message: Any, state: _QueryRuntimeState) -> None:
        """Update final result text candidate from assistant content blocks."""
        msg_content = getattr(assistant_message, "message", None)
        if not msg_content:
            return

        content = getattr(msg_content, "content", None)
        if not content:
            return

        if isinstance(content, str):
            state.final_result_text = content
            return

        if not isinstance(content, list):
            return

        text_parts = self._extract_final_text_parts(content)

        if text_parts:
            state.final_result_text = "\n".join(text_parts)

    def _clear_context_messages(self, messages: list[Any]) -> list[Any]:
        retained = [msg for msg in messages if getattr(msg, "type", None) in {"user", "assistant"}]
        if not retained:
            return []
        return retained[-4:]

    def _extract_final_text_parts(self, content_blocks: list[Any]) -> list[str]:
        """Extract user-visible text for the final result summary.

        A tool call resets text aggregation because pre-tool explanatory text should
        not become the final answer text after a tool execution turn.
        """
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

    def _coerce_structured_output(self, result_text: Optional[str]) -> Any:
        """Best-effort JSON structured output when json_schema is configured."""
        if not self._json_schema or not result_text:
            return None

        stripped = result_text.strip()
        if not stripped:
            return None

        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return None

        schema_type = self._json_schema.get("type")
        if isinstance(schema_type, str):
            if schema_type == "integer" and isinstance(parsed, bool):
                return None
            if schema_type == "number" and isinstance(parsed, bool):
                return None
            type_checks: dict[str, tuple[type, ...]] = {
                "object": (dict,),
                "array": (list,),
                "string": (str,),
                "number": (int, float),
                "boolean": (bool,),
                "integer": (int,),
            }
            expected = type_checks.get(schema_type)
            if expected and not isinstance(parsed, expected):
                logger.debug(
                    "[stdio] Structured output did not match top-level schema type",
                    extra={"expected_type": schema_type, "actual_type": type(parsed).__name__},
                )
                return None

        return parsed

    async def _send_final_result(self, state: _QueryRuntimeState, result: ResultMessage) -> None:
        """Send ResultMessage and mark as sent."""
        if state.result_message_sent:
            logger.warning("[stdio] ResultMessage already sent, skipping duplicate")
            return
        logger.debug("[stdio] Sending ResultMessage")
        try:
            await self._write_message_stream(model_to_dict(result))
            state.result_message_sent = True
            logger.debug("[stdio] ResultMessage sent successfully")
        except Exception as e:
            logger.error(f"[stdio] Failed to send ResultMessage: {e}", exc_info=True)
            state.result_message_sent = True

    async def _send_error_result(
        self,
        state: _QueryRuntimeState,
        error_msg: str,
        exc: Exception | None = None,
    ) -> None:
        """Send error ResultMessage with full stack trace."""
        if exc:
            tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            error_detail = f"{type(exc).__name__}: {error_msg}\n\nStack trace:\n{tb_str}"
        else:
            error_detail = error_msg

        result = ResultMessage(
            duration_ms=state.elapsed_ms(),
            duration_api_ms=0,
            is_error=True,
            subtype="error_during_execution",
            num_turns=state.num_turns,
            session_id=self._session_id or "",
            total_cost_usd=None,
            usage=None,
            result=error_detail[:50000] if len(error_detail) > 50000 else error_detail,
            structured_output=None,
        )
        await self._send_final_result(state, result)

    async def _fail_query_request(
        self,
        request_id: str,
        state: _QueryRuntimeState,
        error_msg: str,
        exc: Exception | None = None,
    ) -> None:
        """Send control error response and always follow with ResultMessage."""
        try:
            await self._write_control_response(request_id, error=error_msg)
        except Exception as write_error:
            logger.error(
                f"[stdio] Failed to send control response: {write_error}",
                exc_info=True,
            )
        await self._send_error_result(state, error_msg, exc)
