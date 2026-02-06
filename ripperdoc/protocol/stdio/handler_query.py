"""Query execution for stdio protocol handler."""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
import uuid
from typing import Any

from ripperdoc.core.hooks.manager import hook_manager
from ripperdoc.core.hooks.state import bind_pending_message_queue, bind_hook_scopes
from ripperdoc.core.query import query
from ripperdoc.protocol.models import ResultMessage, UsageInfo, model_to_dict
from ripperdoc.utils.mcp import format_mcp_instructions, load_mcp_servers_async
from ripperdoc.utils.asyncio_compat import asyncio_timeout
from ripperdoc.utils.messages import create_hook_notice_payload, create_user_message, is_hook_notice_payload
from ripperdoc.utils.session_history import SessionHistory

from .timeouts import (
    STDIO_HOOK_TIMEOUT_SEC,
    STDIO_QUERY_TIMEOUT_SEC,
    STDIO_WATCHDOG_INTERVAL_SEC,
)
from .watchdog import OperationWatchdog

logger = logging.getLogger("ripperdoc.protocol.stdio.handler")


class StdioQueryMixin:
    _session_history: SessionHistory | None
    _session_id: str | None

    async def _handle_query(self, request: dict[str, Any], request_id: str) -> None:
        """Handle query request from SDK with comprehensive timeout and error handling.

        This method ensures ResultMessage is ALWAYS sent on any error/exception/timeout,
        including detailed stack traces for debugging.

        Args:
            request: The query request data.
            request_id: The request ID.
        """
        start_time = time.time()
        result_message_sent = [False]  # [bool] - Track if we've sent ResultMessage
        # Variables to track query state (using lists for mutable reference across async contexts)
        num_turns = [0]  # [int]
        is_error = [False]  # [bool]
        final_result_text: list[str | None] = [None]

        # Track token usage
        total_input_tokens = [0]  # [int]
        total_output_tokens = [0]  # [int]
        total_cache_read_tokens = [0]  # [int]
        total_cache_creation_tokens = [0]  # [int]

        async def send_final_result(result: ResultMessage) -> None:
            """Send ResultMessage and mark as sent."""
            if result_message_sent[0]:
                logger.warning("[stdio] ResultMessage already sent, skipping duplicate")
                return
            logger.debug("[stdio] Sending ResultMessage")
            try:
                await self._write_message_stream(model_to_dict(result))
                result_message_sent[0] = True
                logger.debug("[stdio] ResultMessage sent successfully")
            except Exception as e:
                logger.error(f"[stdio] Failed to send ResultMessage: {e}", exc_info=True)
                result_message_sent[0] = True  # Mark as sent to avoid retries

        async def send_error_result(error_msg: str, exc: Exception | None = None) -> None:
            """Send error ResultMessage with full stack trace."""
            if exc:
                tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                error_detail = f"{type(exc).__name__}: {error_msg}\n\nStack trace:\n{tb_str}"
            else:
                error_detail = error_msg

            result = ResultMessage(
                duration_ms=int((time.time() - start_time) * 1000),
                duration_api_ms=0,
                is_error=True,
                subtype="error_during_execution",
                num_turns=num_turns[0],
                session_id=self._session_id or "",
                total_cost_usd=None,
                usage=None,
                result=error_detail[:50000] if len(error_detail) > 50000 else error_detail,  # Limit size
                structured_output=None,
            )
            await send_final_result(result)

        async def fail_request(error_msg: str, exc: Exception | None = None) -> None:
            """Send control error response and always follow with ResultMessage."""
            try:
                await self._write_control_response(request_id, error=error_msg)
            except Exception as write_error:
                logger.error(
                    f"[stdio] Failed to send control response: {write_error}",
                    exc_info=True,
                )
            await send_error_result(error_msg, exc)

        if not self._initialized:
            await fail_request("Not initialized")
            return

        try:
            prompt = request.get("prompt", "")
            if not prompt:
                await fail_request("Prompt is required")
                return

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

            # Create session history (one per session)
            if self._session_history is None:
                self._session_history = SessionHistory(
                    self._project_path, self._session_id or str(uuid.uuid4())
                )
            session_history = self._session_history
            hook_manager.set_transcript_path(str(session_history.path))

            # Create initial user message
            user_message = create_user_message(prompt)
            self._conversation_messages.append(user_message)
            session_history.append(user_message)

            # Use the conversation history for messages
            messages = list(self._conversation_messages)

            # Build system prompt
            additional_instructions: list[str] = []
            hook_notices: list[dict[str, Any]] = []

            queue = self._query_context.pending_message_queue if self._query_context else None
            hook_scopes = self._query_context.hook_scopes if self._query_context else []
            with bind_pending_message_queue(queue), bind_hook_scopes(hook_scopes):
                if self._session_hook_contexts:
                    additional_instructions.extend(self._session_hook_contexts)

                # Run prompt submit hooks with timeout
                try:
                    async with asyncio_timeout(STDIO_HOOK_TIMEOUT_SEC):
                        prompt_hook_result = await hook_manager.run_user_prompt_submit_async(prompt)
                        if hasattr(prompt_hook_result, "should_block") and prompt_hook_result.should_block:
                            reason = (
                                prompt_hook_result.block_reason
                                if hasattr(prompt_hook_result, "block_reason")
                                else "Prompt blocked by hook."
                            )
                            await fail_request(str(reason))
                            return
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

            # Build final system prompt
            servers = await load_mcp_servers_async(self._project_path)
            mcp_instructions = format_mcp_instructions(servers)

            system_prompt = self._resolve_system_prompt(
                self._query_context.tools if self._query_context else [],
                prompt,
                mcp_instructions,
                additional_instructions,
            )

            # Send acknowledgment that query is starting
            await self._write_control_response(
                request_id, response={"status": "querying", "session_id": self._session_id}
            )
            for notice in hook_notices:
                stream_message = self._build_hook_notice_stream_message(
                    str(notice.get("text", "")),
                    str(notice.get("hook_event", "")),
                    tool_name=notice.get("tool_name"),
                    level=notice.get("level") or "info",
                )
                await self._write_message_stream(model_to_dict(stream_message))

            # Execute query with comprehensive timeout and error handling
            try:
                context: dict[str, Any] = {}

                logger.debug(
                    "[stdio] Preparing query execution",
                    extra={
                        "messages_count": len(messages),
                        "system_prompt_length": len(system_prompt),
                        "query_timeout": STDIO_QUERY_TIMEOUT_SEC,
                    },
                )

                # Create watchdog for monitoring query progress
                async with OperationWatchdog(
                    timeout_sec=STDIO_QUERY_TIMEOUT_SEC, check_interval=STDIO_WATCHDOG_INTERVAL_SEC
                ) as watchdog:
                    # Execute query with overall timeout
                    async with asyncio_timeout(STDIO_QUERY_TIMEOUT_SEC):
                        async for message in query(
                            messages,
                            system_prompt,
                            context,
                            self._query_context or {},  # type: ignore[arg-type]
                            self._can_use_tool,
                        ):
                            watchdog.ping()
                            msg_type = getattr(message, "type", None)
                            logger.debug(
                                f"[stdio] Received message of type: {msg_type}, "
                                f"num_turns={num_turns[0]}, "
                                f"elapsed_ms={int((time.time() - start_time) * 1000)}"
                            )
                            num_turns[0] += 1

                            # Handle progress messages
                            if msg_type == "progress":
                                notice_content = getattr(message, "content", None)
                                if is_hook_notice_payload(notice_content) and isinstance(
                                    notice_content, dict
                                ):
                                    notice_payload: dict[str, Any] = notice_content
                                    stream_message = self._build_hook_notice_stream_message(
                                        str(notice_payload.get("text", "")),
                                        str(notice_payload.get("hook_event", "")),
                                        tool_name=notice_payload.get("tool_name"),
                                        level=notice_payload.get("level") or "info",
                                    )
                                    await self._write_message_stream(model_to_dict(stream_message))
                                    continue
                                # Check if this is a subagent message that should be forwarded to SDK
                                is_subagent_msg = getattr(message, "is_subagent_message", False)
                                if is_subagent_msg:
                                    # Extract the subagent message from content
                                    subagent_message = getattr(message, "content", None)
                                    if subagent_message and hasattr(subagent_message, "type"):
                                        logger.debug(
                                            f"[stdio] Forwarding subagent message: type={getattr(subagent_message, 'type', 'unknown')}"
                                        )
                                        # Convert and forward the subagent message to SDK
                                        message_dict = self._convert_message_to_sdk(subagent_message)
                                        if message_dict:
                                            await self._write_message_stream(message_dict)

                                            # Add subagent messages to conversation history
                                            subagent_msg_type = getattr(subagent_message, "type", "")
                                            if subagent_msg_type == "assistant":
                                                self._conversation_messages.append(subagent_message)

                                                # Track token usage from subagent assistant messages
                                                total_input_tokens[0] += getattr(subagent_message, "input_tokens", 0)
                                                total_output_tokens[0] += getattr(subagent_message, "output_tokens", 0)
                                                total_cache_read_tokens[0] += getattr(subagent_message, "cache_read_tokens", 0)
                                                total_cache_creation_tokens[0] += getattr(
                                                    subagent_message, "cache_creation_tokens", 0
                                                )
                                # Continue to filter out normal progress messages
                                continue

                            # Track token usage from assistant messages
                            if msg_type == "assistant":
                                total_input_tokens[0] += getattr(message, "input_tokens", 0)
                                total_output_tokens[0] += getattr(message, "output_tokens", 0)
                                total_cache_read_tokens[0] += getattr(message, "cache_read_tokens", 0)
                                total_cache_creation_tokens[0] += getattr(
                                    message, "cache_creation_tokens", 0
                                )

                                msg_content = getattr(message, "message", None)
                                if msg_content:
                                    content = getattr(msg_content, "content", None)
                                    if content:
                                        # Extract text blocks for result field
                                        if isinstance(content, str):
                                            final_result_text[0] = content
                                        elif isinstance(content, list):
                                            text_parts: list[str] = []
                                            for block in content:
                                                if isinstance(block, dict):
                                                    if block.get("type") == "text":
                                                        text_parts.append(str(block.get("text", "")))
                                                    elif block.get("type") == "tool_use":
                                                        text_parts.clear()
                                                elif hasattr(block, "type"):
                                                    if block.type == "text":
                                                        text_parts.append(str(getattr(block, "text", "")))
                                            if text_parts:
                                                final_result_text[0] = "\n".join(text_parts)

                            # Convert message to SDK format
                            message_dict = self._convert_message_to_sdk(message)
                            if message_dict is None:
                                continue
                            await self._write_message_stream(message_dict)

                            # Add to conversation history
                            if msg_type in ("assistant", "user"):
                                self._conversation_messages.append(message)

                            # Add to local history and session history
                            messages.append(message)  # type: ignore[arg-type]
                            session_history.append(message)  # type: ignore[arg-type]

                logger.debug("[stdio] Query loop ended successfully")

            except asyncio.TimeoutError:
                logger.error(f"[stdio] Query execution timed out after {STDIO_QUERY_TIMEOUT_SEC}s")
                await send_error_result(f"Query timed out after {STDIO_QUERY_TIMEOUT_SEC}s")
            except asyncio.CancelledError:
                logger.warning("[stdio] Query was cancelled")
                await send_error_result("Query was cancelled")
            except Exception as query_error:
                is_error[0] = True
                logger.error(f"[stdio] Query execution error: {type(query_error).__name__}: {query_error}", exc_info=True)
                await send_error_result(str(query_error), query_error)

            # Build and send normal completion result if no error occurred
            if not is_error[0] and not result_message_sent[0]:
                logger.debug("[stdio] Building usage info")

                # Calculate cost
                cost_per_million_input = 3.0
                cost_per_million_output = 15.0
                total_cost_usd = (total_input_tokens[0] * cost_per_million_input / 1_000_000) + (
                    total_output_tokens[0] * cost_per_million_output / 1_000_000
                )

                # Build usage info
                usage_info = None
                if total_input_tokens[0] or total_output_tokens[0]:
                    usage_info = UsageInfo(
                        input_tokens=total_input_tokens[0],
                        cache_creation_input_tokens=total_cache_creation_tokens[0],
                        cache_read_input_tokens=total_cache_read_tokens[0],
                        output_tokens=total_output_tokens[0],
                    )

                duration_ms = int((time.time() - start_time) * 1000)
                duration_api_ms = duration_ms

                result_message = ResultMessage(
                    duration_ms=duration_ms,
                    duration_api_ms=duration_api_ms,
                    is_error=is_error[0],
                    subtype="success",
                    num_turns=num_turns[0],
                    session_id=self._session_id or "",
                    total_cost_usd=round(total_cost_usd, 8) if total_cost_usd > 0 else None,
                    usage=usage_info,
                    result=final_result_text[0],
                    structured_output=None,
                )
                await send_final_result(result_message)

            logger.info(
                "[stdio] Query completed",
                extra={
                    "request_id": request_id,
                    "duration_ms": int((time.time() - start_time) * 1000),
                    "num_turns": num_turns[0],
                    "is_error": is_error[0],
                    "input_tokens": total_input_tokens[0],
                    "output_tokens": total_output_tokens[0],
                    "result_sent": result_message_sent[0],
                },
            )

        except Exception as e:
            logger.error(f"[stdio] Handle query failed: {type(e).__name__}: {e}", exc_info=True)
            await self._write_control_response(request_id, error=str(e))

            # Ensure ResultMessage is sent even if everything else fails
            if not result_message_sent[0]:
                try:
                    await send_error_result(str(e), e)
                except Exception as send_error:
                    logger.error(
                        f"[stdio] Critical: Failed to send error ResultMessage: {send_error}",
                        exc_info=True,
                    )
