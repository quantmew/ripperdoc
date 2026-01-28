"""Stdio command for SDK subprocess communication.

This module implements the stdio command that enables Ripperdoc to communicate
with SDKs via JSON Control Protocol over stdin/stdout, following Claude SDK's
elegant subprocess architecture patterns.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import click

from ripperdoc.core.config import get_project_config
from ripperdoc.core.default_tools import get_default_tools
from ripperdoc.core.query import query, QueryContext
from ripperdoc.core.system_prompt import build_system_prompt
from ripperdoc.core.hooks.manager import hook_manager
from ripperdoc.core.hooks.llm_callback import build_hook_llm_callback
from ripperdoc.utils.messages import create_user_message
from ripperdoc.utils.memory import build_memory_instructions
from ripperdoc.core.permissions import make_permission_checker
from ripperdoc.utils.session_history import SessionHistory
from ripperdoc.utils.mcp import (
    load_mcp_servers_async,
    format_mcp_instructions,
    shutdown_mcp_runtime,
)
from ripperdoc.utils.lsp import shutdown_lsp_manager
from ripperdoc.tools.background_shell import shutdown_background_shell
from ripperdoc.tools.mcp_tools import load_dynamic_mcp_tools_async, merge_tools_with_dynamic
from ripperdoc.utils.log import get_logger

logger = logging.getLogger(__name__)


class StdioProtocolHandler:
    """Handler for stdio-based JSON Control Protocol.

    This class manages bidirectional communication with the SDK:
    - Reads JSON commands from stdin
    - Parses control requests (initialize, query, etc.)
    - Executes core query logic
    - Writes JSON responses to stdout

    Following Claude SDK's elegant patterns:
    - JSON messages separated by newlines
    - Control requests/responses for protocol management
    - Message streaming for query results
    """

    def __init__(self, input_format: str = "stream-json", output_format: str = "stream-json"):
        """Initialize the protocol handler.

        Args:
            input_format: Input format ("stream-json" or "auto")
            output_format: Output format ("stream-json")
        """
        self._input_format = input_format
        self._output_format = output_format
        self._initialized = False
        self._session_id: str | None = None
        self._project_path: Path = Path.cwd()
        self._query_context: QueryContext | None = None
        self._can_use_tool: Any | None = None
        self._hooks: dict[str, list[dict[str, Any]]] = {}
        self._pending_requests: dict[str, Any] = {}

    async def _write_message(self, message: dict[str, Any]) -> None:
        """Write a JSON message to stdout.

        Args:
            message: The message dictionary to write.
        """
        json_data = json.dumps(message, ensure_ascii=False)
        sys.stdout.write(json_data + "\n")
        sys.stdout.flush()

    async def _write_control_response(
        self,
        request_id: str,
        response: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """Write a control response message.

        Args:
            request_id: The request ID this responds to.
            response: The response data (for success).
            error: The error message (for failure).
        """
        if error:
            response_data = {
                "subtype": "error",
                "request_id": request_id,
                "error": error,
            }
        else:
            response_data = {
                "subtype": "success",
                "request_id": request_id,
                "response": response,
            }

        message = {
            "type": "control_response",
            "response": response_data,
        }

        await self._write_message(message)

    async def _write_message_stream(
        self,
        message_dict: dict[str, Any],
    ) -> None:
        """Write a regular message to the output stream.

        Args:
            message_dict: The message dictionary to write.
        """
        await self._write_message(message_dict)

    async def _read_line(self) -> str | None:
        """Read a single line from stdin.

        Returns:
            The line content, or None if EOF.
        """
        try:
            line = await asyncio.get_event_loop().run_in_executor(
                None, sys.stdin.readline
            )
            if not line:
                return None
            return line.rstrip("\n\r")
        except (OSError, IOError) as e:
            logger.error(f"Error reading from stdin: {e}")
            return None

    async def _read_messages(self) -> AsyncIterator[dict[str, Any]]:
        """Read and parse JSON messages from stdin.

        Yields:
            Parsed JSON message dictionaries.
        """
        json_buffer = ""

        while True:
            line = await self._read_line()
            if line is None:
                break

            line = line.strip()
            if not line:
                continue

            # Handle JSON that may span multiple lines
            json_lines = line.split("\n")
            for json_line in json_lines:
                json_line = json_line.strip()
                if not json_line:
                    continue

                json_buffer += json_line

                try:
                    data = json.loads(json_buffer)
                    json_buffer = ""
                    yield data
                except json.JSONDecodeError:
                    # Keep buffering - might be incomplete JSON
                    continue

    async def _handle_initialize(self, request: dict[str, Any], request_id: str) -> None:
        """Handle initialize request from SDK.

        Args:
            request: The initialize request data.
            request_id: The request ID.
        """
        if self._initialized:
            await self._write_control_response(
                request_id,
                error="Already initialized"
            )
            return

        try:
            # Extract options from request
            options = request.get("options", {})
            self._session_id = options.get("session_id") or str(uuid.uuid4())

            # Setup working directory
            cwd = options.get("cwd")
            if cwd:
                self._project_path = Path(cwd)
            else:
                self._project_path = Path.cwd()

            # Initialize project config
            get_project_config(self._project_path)

            # Parse tool options
            allowed_tools = options.get("allowed_tools")
            disallowed_tools = options.get("disallowed_tools")
            tools_list = options.get("tools")

            # Get the tool list
            if tools_list is not None:
                # SDK provided explicit tool list
                # For now, use default tools
                tools = get_default_tools(allowed_tools=allowed_tools)
            else:
                tools = get_default_tools(allowed_tools=allowed_tools)

            # Parse permission mode
            permission_mode = options.get("permission_mode", "default")
            yolo_mode = permission_mode == "bypassPermissions"

            # Create permission checker
            if not yolo_mode:
                self._can_use_tool = make_permission_checker(
                    self._project_path,
                    yolo_mode=False
                )

            # Setup model
            model = options.get("model") or "main"

            # Create query context
            self._query_context = QueryContext(
                tools=tools,
                yolo_mode=yolo_mode,
                verbose=options.get("verbose", False),
                model=model,
            )

            # Initialize hook manager
            hook_manager.set_project_dir(self._project_path)
            hook_manager.set_session_id(self._session_id)
            hook_manager.set_llm_callback(build_hook_llm_callback())

            # Store hooks configuration
            hooks = options.get("hooks", {})
            self._hooks = hooks

            # Load MCP servers and dynamic tools
            servers = await load_mcp_servers_async(self._project_path)
            dynamic_tools = await load_dynamic_mcp_tools_async(self._project_path)
            if dynamic_tools:
                tools = merge_tools_with_dynamic(tools, dynamic_tools)
                self._query_context.tools = tools

            mcp_instructions = format_mcp_instructions(servers)

            # Build system prompt components
            from ripperdoc.core.skills import load_all_skills, build_skill_summary

            skill_result = load_all_skills(self._project_path)
            skill_instructions = build_skill_summary(skill_result.skills)

            additional_instructions: list[str] = []
            if skill_instructions:
                additional_instructions.append(skill_instructions)

            memory_instructions = build_memory_instructions()
            if memory_instructions:
                additional_instructions.append(memory_instructions)

            system_prompt = build_system_prompt(
                tools,
                "",  # Will be set per query
                {},
                additional_instructions=additional_instructions or None,
                mcp_instructions=mcp_instructions,
            )

            # Mark as initialized
            self._initialized = True

            # Send success response with available tools
            await self._write_control_response(
                request_id,
                response={
                    "session_id": self._session_id,
                    "system_prompt": system_prompt,
                    "tools": [{"name": t.name} for t in tools],
                    "mcp_servers": [{"name": s.name} for s in servers] if servers else [],
                }
            )

        except Exception as e:
            logger.error(f"Initialize failed: {e}", exc_info=True)
            await self._write_control_response(request_id, error=str(e))

    async def _handle_query(self, request: dict[str, Any], request_id: str) -> None:
        """Handle query request from SDK.

        Args:
            request: The query request data.
            request_id: The request ID.
        """
        if not self._initialized:
            await self._write_control_response(
                request_id,
                error="Not initialized"
            )
            return

        try:
            prompt = request.get("prompt", "")
            if not prompt:
                await self._write_control_response(
                    request_id,
                    error="Prompt is required"
                )
                return

            # Create session history
            session_history = SessionHistory(
                self._project_path,
                self._session_id or str(uuid.uuid4())
            )
            hook_manager.set_transcript_path(str(session_history.path))

            # Create initial user message
            from ripperdoc.utils.messages import UserMessage, AssistantMessage

            messages: list[UserMessage | AssistantMessage] = [create_user_message(prompt)]
            session_history.append(messages[0])

            # Build system prompt
            additional_instructions: list[str] = []

            # Run session start hooks
            try:
                session_start_result = await hook_manager.run_session_start_async("startup")
                if hasattr(session_start_result, "system_message"):
                    if session_start_result.system_message:
                        additional_instructions.append(str(session_start_result.system_message))
                if hasattr(session_start_result, "additional_context"):
                    if session_start_result.additional_context:
                        additional_instructions.append(str(session_start_result.additional_context))
            except Exception as e:
                logger.warning(f"Session start hook failed: {e}")

            # Run prompt submit hooks
            try:
                prompt_hook_result = await hook_manager.run_user_prompt_submit_async(prompt)
                if hasattr(prompt_hook_result, "should_block") and prompt_hook_result.should_block:
                    reason = (
                        prompt_hook_result.block_reason
                        if hasattr(prompt_hook_result, "block_reason")
                        else "Prompt blocked by hook."
                    )
                    await self._write_control_response(request_id, error=str(reason))
                    return
                if hasattr(prompt_hook_result, "system_message") and prompt_hook_result.system_message:
                    additional_instructions.append(str(prompt_hook_result.system_message))
                if hasattr(prompt_hook_result, "additional_context") and prompt_hook_result.additional_context:
                    additional_instructions.append(str(prompt_hook_result.additional_context))
            except Exception as e:
                logger.warning(f"Prompt submit hook failed: {e}")

            # Build final system prompt
            servers = await load_mcp_servers_async(self._project_path)
            mcp_instructions = format_mcp_instructions(servers)

            system_prompt = build_system_prompt(
                self._query_context.tools if self._query_context else [],
                prompt,
                {},
                additional_instructions=additional_instructions or None,
                mcp_instructions=mcp_instructions,
            )

            # Send acknowledgment that query is starting
            await self._write_control_response(
                request_id,
                response={"status": "querying", "session_id": self._session_id}
            )

            # Track query metrics
            start_time = time.time()
            num_turns = 0
            is_error = False

            try:
                # Run the query and stream messages
                context: dict[str, Any] = {}

                async for message in query(
                    messages,
                    system_prompt,
                    context,
                    self._query_context,
                    self._can_use_tool,
                ):
                    num_turns += 1

                    # Convert message to SDK format
                    message_dict = self._convert_message_to_sdk(message)
                    await self._write_message_stream(message_dict)

                    # Add to history
                    messages.append(message)  # type: ignore[arg-type]
                    session_history.append(message)  # type: ignore[arg-type]

                # Send result message
                duration_ms = int((time.time() - start_time) * 1000)

                result_message = {
                    "type": "result",
                    "duration_ms": duration_ms,
                    "duration_api_ms": 0,  # Not tracking separately yet
                    "is_error": is_error,
                    "num_turns": num_turns,
                    "session_id": self._session_id or "",
                }

                await self._write_message_stream(result_message)

            except KeyboardInterrupt:
                is_error = True
                await self._write_message_stream({
                    "type": "system",
                    "subtype": "error",
                    "data": {"message": "Interrupted by user"},
                })
            except Exception as e:
                is_error = True
                logger.error(f"Query error: {e}", exc_info=True)
                await self._write_message_stream({
                    "type": "system",
                    "subtype": "error",
                    "data": {"message": str(e)},
                })

            # Run session end hooks
            try:
                duration = time.time() - start_time
                await hook_manager.run_session_end_async(
                    "other",
                    duration_seconds=duration,
                    message_count=len(messages),
                )
            except Exception as e:
                logger.warning(f"Session end hook failed: {e}")

        except Exception as e:
            logger.error(f"Handle query failed: {e}", exc_info=True)
            await self._write_control_response(request_id, error=str(e))

    def _convert_message_to_sdk(self, message: Any) -> dict[str, Any]:
        """Convert internal message to SDK format.

        Args:
            message: The internal message object.

        Returns:
            A dictionary in SDK message format.
        """
        msg_type = getattr(message, "type", None)

        if msg_type == "assistant":
            content_blocks = []
            msg_content = getattr(message, "message", None)
            if msg_content:
                content = getattr(msg_content, "content", None)
                if content:
                    if isinstance(content, str):
                        content_blocks.append({"type": "text", "text": content})
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict):
                                content_blocks.append(block)
                            else:
                                # Convert dataclass to dict
                                block_dict = {}
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
                                if hasattr(block, "tool_use_id"):
                                    block_dict["tool_use_id"] = block.tool_use_id
                                if hasattr(block, "content"):
                                    block_dict["content"] = block.content
                                if hasattr(block, "is_error"):
                                    block_dict["is_error"] = block.is_error
                                if block_dict:
                                    content_blocks.append(block_dict)

            return {
                "type": "assistant",
                "message": {
                    "content": content_blocks,
                    "model": getattr(message, "model", "main"),
                },
                "parent_tool_use_id": getattr(message, "parent_tool_use_id"),
            }

        elif msg_type == "user":
            msg_content = getattr(message, "message", None)
            content = getattr(msg_content, "content", "") if msg_content else ""
            return {
                "type": "user",
                "message": {"content": content},
                "uuid": getattr(message, "uuid", None),
                "parent_tool_use_id": getattr(message, "parent_tool_use_id", None),
                "tool_use_result": getattr(message, "tool_use_result", None),
            }

        elif msg_type == "progress":
            return {
                "type": "progress",
                "tool_use_id": getattr(message, "tool_use_id", None),
                "content": getattr(message, "content", None),
            }

        else:
            # Unknown message type
            return {
                "type": "system",
                "subtype": "unknown",
                "data": {"message_type": str(msg_type)},
            }

    async def _handle_control_request(self, message: dict[str, Any]) -> None:
        """Handle a control request from the SDK.

        Args:
            message: The control request message.
        """
        request = message.get("request", {})
        request_id = message.get("request_id", "")
        request_subtype = request.get("subtype", "")

        try:
            if request_subtype == "initialize":
                await self._handle_initialize(request, request_id)

            elif request_subtype == "query":
                await self._handle_query(request, request_id)

            else:
                await self._write_control_response(
                    request_id,
                    error=f"Unknown request subtype: {request_subtype}"
                )

        except Exception as e:
            logger.error(f"Error handling control request: {e}", exc_info=True)
            await self._write_control_response(request_id, error=str(e))

    async def run(self) -> None:
        """Main run loop for the stdio protocol handler.

        Reads messages from stdin and handles them until EOF.
        """
        logger.info("Stdio protocol handler starting")

        try:
            async for message in self._read_messages():
                msg_type = message.get("type")

                if msg_type == "control_request":
                    await self._handle_control_request(message)

                else:
                    # Unknown message type
                    logger.warning(f"Unknown message type: {msg_type}")

        except (OSError, IOError, json.JSONDecodeError) as e:
            logger.error(f"Error in stdio loop: {e}")

        finally:
            # Cleanup
            await shutdown_mcp_runtime()
            await shutdown_lsp_manager()
            try:
                shutdown_background_shell(force=True)
            except Exception:
                pass

            logger.info("Stdio protocol handler exiting")


@click.command(name="stdio")
@click.option(
    "--input-format",
    type=click.Choice(["stream-json", "auto"]),
    default="stream-json",
    help="Input format for messages.",
)
@click.option(
    "--output-format",
    type=click.Choice(["stream-json"]),
    default="stream-json",
    help="Output format for messages.",
)
@click.option(
    "--model",
    type=str,
    default=None,
    help="Model profile for the current session.",
)
@click.option(
    "--permission-mode",
    type=click.Choice(["default", "acceptEdits", "plan", "bypassPermissions"]),
    default="default",
    help="Permission mode for tool usage.",
)
@click.option(
    "--max-turns",
    type=int,
    default=None,
    help="Maximum number of conversation turns.",
)
@click.option(
    "--system-prompt",
    type=str,
    default=None,
    help="System prompt to use for the session.",
)
@click.option(
    "--print",
    "-p",
    is_flag=True,
    help="Print mode (for single prompt queries).",
)
@click.option(
    "--",
    "prompt",
    type=str,
    default=None,
    help="Direct prompt (for print mode).",
)
def stdio_cmd(
    input_format: str,
    output_format: str,
    model: str | None,
    permission_mode: str,
    max_turns: int | None,
    system_prompt: str | None,
    print: bool,
    prompt: str | None,
) -> None:
    """Stdio mode for SDK subprocess communication.

    This command enables Ripperdoc to communicate with SDKs via JSON Control
    Protocol over stdin/stdout. It's designed for subprocess architecture where
    the SDK manages the CLI process.

    The protocol supports:
    - control_request/control_response for protocol management
    - Message streaming for query results
    - Bidirectional communication for hooks and permissions

    Example:
        ripperdoc stdio --output-format stream-json
    """
    # Set up async event loop
    asyncio.run(_run_stdio(
        input_format=input_format,
        output_format=output_format,
        model=model,
        permission_mode=permission_mode,
        max_turns=max_turns,
        system_prompt=system_prompt,
        print_mode=print,
        prompt=prompt,
    ))


async def _run_stdio(
    input_format: str,
    output_format: str,
    model: str | None,
    permission_mode: str,
    max_turns: int | None,
    system_prompt: str | None,
    print_mode: bool,
    prompt: str | None,
) -> None:
    """Async entry point for stdio command."""
    handler = StdioProtocolHandler(
        input_format=input_format,
        output_format=output_format,
    )

    # If print mode with prompt, handle as single query
    if print_mode and prompt:
        # This is a single-shot query mode
        # Initialize with defaults and run query
        request = {
            "options": {
                "model": model,
                "permission_mode": permission_mode,
                "max_turns": max_turns,
                "system_prompt": system_prompt,
            },
            "prompt": prompt,
        }

        # Mock request_id for print mode
        request_id = "print_query"

        # Initialize
        await handler._handle_initialize(request, request_id)

        # Query
        query_request = {
            "prompt": prompt,
        }
        await handler._handle_query(query_request, f"{request_id}_query")

        return

    # Otherwise, run the stdio protocol loop
    await handler.run()


__all__ = ["stdio_cmd", "StdioProtocolHandler"]
