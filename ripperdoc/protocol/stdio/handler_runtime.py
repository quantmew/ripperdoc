"""Runtime loop and cleanup for stdio protocol handler."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any

from pydantic import ValidationError

from ripperdoc.core.hooks.manager import hook_manager
from ripperdoc.core.hooks.state import bind_hook_scopes
from ripperdoc.protocol.models import IncomingUserStreamMessage
from ripperdoc.tools.background_shell import shutdown_background_shell
from ripperdoc.utils.asyncio_compat import asyncio_timeout
from ripperdoc.utils.lsp import shutdown_lsp_manager
from ripperdoc.utils.mcp import shutdown_mcp_runtime
from ripperdoc.utils.tasks import set_runtime_task_scope

from .timeouts import STDIO_HOOK_TIMEOUT_SEC

logger = logging.getLogger("ripperdoc.protocol.stdio.handler")


class StdioRuntimeMixin:
    def _extract_prompt_from_user_content(self, content: str | list[dict[str, Any]]) -> str | None:
        """Extract plain-text prompt from validated user message content."""
        if isinstance(content, str):
            prompt = content.strip()
            return prompt or None

        if isinstance(content, list):
            text_parts: list[str] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") != "text":
                    continue
                text = block.get("text")
                if isinstance(text, str):
                    normalized = text.strip()
                    if normalized:
                        text_parts.append(normalized)
            if text_parts:
                return "\n".join(text_parts)

        return None

    def _coerce_user_message_to_control_request(
        self, message: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Convert incoming `type=user` stream message into a query control request."""
        try:
            validated_message = IncomingUserStreamMessage.model_validate(message)
        except ValidationError as e:
            logger.warning("[stdio] Invalid user message format: %s", e.errors())
            return None

        prompt = self._extract_prompt_from_user_content(validated_message.message.content)
        if not prompt:
            return None

        request_id = validated_message.uuid
        if request_id is None or not request_id.strip():
            request_id = f"user_{uuid.uuid4().hex}"

        return {
            "type": "control_request",
            "request_id": request_id,
            "request": {"subtype": "query", "prompt": prompt},
        }

    def _spawn_control_request_task(self, message: dict[str, Any]) -> None:
        """Schedule a control request for async handling with lifecycle tracking."""
        task = asyncio.create_task(self._handle_control_request(message))
        self._inflight_tasks.add(task)

        def _cleanup_task(t: asyncio.Task[None]) -> None:
            self._inflight_tasks.discard(t)
            if t.cancelled():
                return
            exc = t.exception()
            if exc:
                logger.error(
                    "[stdio] control_request task failed: %s: %s",
                    type(exc).__name__,
                    exc,
                )

        task.add_done_callback(_cleanup_task)

    async def _run_session_end(self, reason: str) -> None:
        if self._session_end_sent or not self._session_started:
            return
        duration = 0.0
        if self._session_start_time is not None:
            duration = max(time.time() - self._session_start_time, 0.0)
        message_count = len(self._conversation_messages)
        hook_scopes = self._query_context.hook_scopes if self._query_context else []
        logger.debug("[stdio] Running session end hooks")
        try:
            with bind_hook_scopes(hook_scopes):
                async with asyncio_timeout(STDIO_HOOK_TIMEOUT_SEC):
                    await hook_manager.run_session_end_async(
                        reason,
                        duration_seconds=duration,
                        message_count=message_count,
                    )
            logger.debug("[stdio] Session end hooks completed")
        except asyncio.TimeoutError:
            logger.warning(
                f"[stdio] Session end hook timed out after {STDIO_HOOK_TIMEOUT_SEC}s"
            )
        except Exception as e:
            logger.warning(f"[stdio] Session end hook failed: {e}")
        finally:
            self._session_end_sent = True

    async def run(self) -> None:
        """Main run loop for the stdio protocol handler with graceful shutdown.

        Reads messages from stdin and handles them until EOF.
        """
        logger.info("Stdio protocol handler starting")
        cancel_inflight_tasks = True

        try:
            async for message in self._read_messages():
                msg_type = message.get("type")

                if msg_type == "control_response":
                    await self._handle_control_response(message)
                    continue

                if msg_type == "control_request":
                    self._spawn_control_request_task(message)
                    continue

                if msg_type == "user":
                    control_message = self._coerce_user_message_to_control_request(message)
                    if control_message is None:
                        logger.warning(
                            "[stdio] Ignoring user message without text prompt content"
                        )
                        continue
                    self._spawn_control_request_task(control_message)
                    continue

                # Unknown message type
                logger.warning(f"Unknown message type: {msg_type}")

            # stdin EOF in stream-json mode is expected after SDK finishes sending prompts.
            # Keep in-flight query tasks alive so they can finish and emit results.
            cancel_inflight_tasks = False

        except (OSError, IOError, json.JSONDecodeError) as e:
            logger.error(f"Error in stdio loop: {e}", exc_info=True)
        except asyncio.CancelledError:
            logger.info("Stdio protocol handler cancelled")
        except Exception as e:
            logger.error(f"Unexpected error in stdio loop: {type(e).__name__}: {e}", exc_info=True)
        finally:
            set_runtime_task_scope(session_id=None)
            # Comprehensive cleanup with timeout
            logger.info("Stdio protocol handler shutting down...")

            try:
                await self.flush_output()
            except Exception as e:
                logger.error(f"[cleanup] Error flushing output: {e}")

            if self._inflight_tasks:
                if cancel_inflight_tasks:
                    for task in list(self._inflight_tasks):
                        task.cancel()
                    try:
                        await asyncio.gather(*self._inflight_tasks, return_exceptions=True)
                    except Exception:
                        pass
                else:
                    try:
                        async with asyncio_timeout(30):
                            await asyncio.gather(*self._inflight_tasks, return_exceptions=True)
                    except asyncio.TimeoutError:
                        logger.warning(
                            "[cleanup] Timed out waiting for inflight tasks; cancelling remaining tasks"
                        )
                        for task in list(self._inflight_tasks):
                            task.cancel()
                        try:
                            await asyncio.gather(*self._inflight_tasks, return_exceptions=True)
                        except Exception:
                            pass

            try:
                await self._run_session_end("other")
            except Exception as e:
                logger.warning(f"[cleanup] Session end hook failed: {e}")

            cleanup_tasks = []

            # Add MCP runtime shutdown
            async def cleanup_mcp() -> None:
                try:
                    async with asyncio_timeout(10):
                        await shutdown_mcp_runtime()
                except asyncio.TimeoutError:
                    logger.warning("[cleanup] MCP runtime shutdown timed out")
                except Exception as e:
                    logger.error(f"[cleanup] Error shutting down MCP runtime: {e}")

            cleanup_tasks.append(asyncio.create_task(cleanup_mcp()))

            # Add LSP manager shutdown
            async def cleanup_lsp() -> None:
                try:
                    async with asyncio_timeout(10):
                        await shutdown_lsp_manager()
                except asyncio.TimeoutError:
                    logger.warning("[cleanup] LSP manager shutdown timed out")
                except Exception as e:
                    logger.error(f"[cleanup] Error shutting down LSP manager: {e}")

            cleanup_tasks.append(asyncio.create_task(cleanup_lsp()))

            # Add background shell shutdown
            async def cleanup_shell() -> None:
                try:
                    shutdown_background_shell(force=True)
                except Exception:
                    pass  # Background shell cleanup is best-effort

            cleanup_tasks.append(asyncio.create_task(cleanup_shell()))

            # Wait for all cleanup tasks with overall timeout
            try:
                async with asyncio_timeout(30):
                    results = await asyncio.gather(*cleanup_tasks, return_exceptions=True)
                    # Check for any exceptions that occurred
                    for i, result in enumerate(results):
                        if isinstance(result, Exception):
                            logger.error(f"[cleanup] Task {i} failed: {result}")
            except asyncio.TimeoutError:
                logger.warning("[cleanup] Cleanup tasks timed out after 30s")
            except Exception as e:
                logger.error(f"[cleanup] Error during cleanup: {e}")

            logger.info("Stdio protocol handler shutdown complete")
