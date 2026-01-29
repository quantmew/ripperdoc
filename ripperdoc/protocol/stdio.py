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
from ripperdoc.core.query_utils import resolve_model_profile
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
from ripperdoc.protocol.models import (
    ControlResponseMessage,
    ControlResponseSuccess,
    ControlResponseError,
    AssistantStreamMessage,
    UserStreamMessage,
    AssistantMessageData,
    UserMessageData,
    ResultMessage,
    UsageInfo,
    MCPServerInfo,
    InitializeResponseData,
    PermissionResponseAllow,
    PermissionResponseDeny,
    model_to_dict,
)

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

        # Conversation history for multi-turn queries
        self._conversation_messages: list[Any] = []

    async def _write_message(self, message: dict[str, Any]) -> None:
        """Write a JSON message to stdout.

        Args:
            message: The message dictionary to write.
        """
        json_data = json.dumps(message, ensure_ascii=False)
        msg_type = message.get("type", "unknown")
        logger.debug(f"[stdio] Writing message: type={msg_type}, json_length={len(json_data)}")
        sys.stdout.write(json_data + "\n")
        sys.stdout.flush()
        logger.debug(f"[stdio] Flushed message: type={msg_type}")

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
            response_data = ControlResponseError(
                request_id=request_id,
                error=error,
            )
        else:
            response_data = ControlResponseSuccess(
                request_id=request_id,
                response=response,
            )

        message = ControlResponseMessage(response=response_data)

        await self._write_message(model_to_dict(message))

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
            # Use simple list format for Claude SDK compatibility
            # Get skill info for agents list
            from ripperdoc.core.skills import load_all_skills
            skill_result = load_all_skills(self._project_path)
            agent_names = [s.name for s in skill_result.skills] if skill_result.skills else []

            init_response = InitializeResponseData(
                session_id=self._session_id or "",
                system_prompt=system_prompt,
                tools=[t.name for t in tools],
                mcp_servers=[MCPServerInfo(name=s.name) for s in servers] if servers else [],
                slash_commands=[],
                agents=agent_names,
                skills=[],
                plugins=[],
            )

            await self._write_control_response(
                request_id,
                response=model_to_dict(init_response),
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

            # Add the new user message to conversation history
            user_message = create_user_message(prompt)
            self._conversation_messages.append(user_message)
            session_history.append(user_message)

            # Use the conversation history for messages (maintains context across queries)
            messages = list(self._conversation_messages)

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
            final_result_text = None  # Track final response text for ResultMessage

            # Track token usage and cost
            total_input_tokens = 0
            total_output_tokens = 0
            total_cache_read_tokens = 0
            total_cache_creation_tokens = 0

            try:
                # Run the query and stream messages
                context: dict[str, Any] = {}

                logger.debug("[stdio] Starting query() async iteration")
                async for message in query(
                    messages,
                    system_prompt,
                    context,
                    self._query_context,
                    self._can_use_tool,
                ):
                    logger.debug(f"[stdio] Received message of type: {getattr(message, 'type', 'unknown')}")
                    num_turns += 1

                    # Filter out progress messages (internal implementation detail)
                    msg_type = getattr(message, "type", None)
                    if msg_type == "progress":
                        # Progress messages are for internal debugging only
                        continue

                    # Track token usage from assistant messages
                    if msg_type == "assistant":
                        # Accumulate token usage
                        total_input_tokens += getattr(message, "input_tokens", 0)
                        total_output_tokens += getattr(message, "output_tokens", 0)
                        total_cache_read_tokens += getattr(message, "cache_read_tokens", 0)
                        total_cache_creation_tokens += getattr(message, "cache_creation_tokens", 0)

                        msg_content = getattr(message, "message", None)
                        if msg_content:
                            content = getattr(msg_content, "content", None)
                            if content:
                                # Extract text blocks for result field
                                if isinstance(content, str):
                                    final_result_text = content
                                elif isinstance(content, list):
                                    # Concatenate text blocks
                                    text_parts = []
                                    for block in content:
                                        if isinstance(block, dict):
                                            if block.get("type") == "text":
                                                text_parts.append(block.get("text", ""))
                                            elif block.get("type") == "tool_use":
                                                # Clear result when tool use happens (next text will be final)
                                                text_parts.clear()
                                        elif hasattr(block, "type"):
                                            if block.type == "text":
                                                text_parts.append(getattr(block, "text", ""))
                                    if text_parts:
                                        final_result_text = "\n".join(text_parts)

                    # Convert message to SDK format
                    message_dict = self._convert_message_to_sdk(message)
                    if message_dict is None:
                        # Skip messages that shouldn't be sent to SDK
                        continue
                    await self._write_message_stream(message_dict)

                    # Add to conversation history (for context in next queries)
                    if msg_type == "assistant":
                        self._conversation_messages.append(message)

                    # Add to local history and session history
                    messages.append(message)  # type: ignore[arg-type]
                    session_history.append(message)  # type: ignore[arg-type]

                logger.debug("[stdio] Query loop ended, building usage info")
                # Calculate cost (simplified model - can be made more accurate with model-specific pricing)
                # Using approximate pricing: $3/M input tokens, $15/M output tokens
                # This is a rough estimate - actual pricing depends on the model used
                cost_per_million_input = 3.0
                cost_per_million_output = 15.0
                total_cost_usd = (
                    (total_input_tokens * cost_per_million_input / 1_000_000) +
                    (total_output_tokens * cost_per_million_output / 1_000_000)
                )

                logger.debug("[stdio] Building usage info")
                # Build usage info
                usage_info = None
                if total_input_tokens or total_output_tokens:
                    usage_info = UsageInfo(
                        input_tokens=total_input_tokens,
                        cache_creation_input_tokens=total_cache_creation_tokens,
                        cache_read_input_tokens=total_cache_read_tokens,
                        output_tokens=total_output_tokens,
                    )

                # Send result message
                logger.debug("[stdio] Creating ResultMessage")
                duration_ms = int((time.time() - start_time) * 1000)
                duration_api_ms = duration_ms  # Using total duration for now

                result_message = ResultMessage(
                    duration_ms=duration_ms,
                    duration_api_ms=duration_api_ms,
                    is_error=is_error,
                    num_turns=num_turns,
                    session_id=self._session_id or "",
                    total_cost_usd=round(total_cost_usd, 8) if total_cost_usd > 0 else None,
                    usage=usage_info,
                    result=final_result_text,
                    structured_output=None,
                )

                logger.debug("[stdio] Sending ResultMessage")
                await self._write_message_stream(model_to_dict(result_message))
                logger.debug("[stdio] ResultMessage sent")

            except KeyboardInterrupt:
                is_error = True
                # Send error result message for SDK compatibility
                result_message = ResultMessage(
                    duration_ms=int((time.time() - start_time) * 1000),
                    duration_api_ms=0,
                    is_error=True,
                    num_turns=num_turns,
                    session_id=self._session_id or "",
                    total_cost_usd=None,
                    usage=None,
                    result="Interrupted by user",
                    structured_output=None,
                )
                await self._write_message_stream(model_to_dict(result_message))
            except Exception as e:
                is_error = True
                logger.error(f"Query error: {e}", exc_info=True)
                # Send error result message for SDK compatibility
                result_message = ResultMessage(
                    duration_ms=int((time.time() - start_time) * 1000),
                    duration_api_ms=0,
                    is_error=True,
                    num_turns=num_turns,
                    session_id=self._session_id or "",
                    total_cost_usd=None,
                    usage=None,
                    result=str(e),
                    structured_output=None,
                )
                await self._write_message_stream(model_to_dict(result_message))

            # Run session end hooks
            logger.debug("[stdio] Running session end hooks")
            try:
                duration = time.time() - start_time
                await hook_manager.run_session_end_async(
                    "other",
                    duration_seconds=duration,
                    message_count=len(messages),
                )
                logger.debug("[stdio] Session end hooks completed")
            except Exception as e:
                logger.warning(f"Session end hook failed: {e}")

        except Exception as e:
            logger.error(f"Handle query failed: {e}", exc_info=True)
            await self._write_control_response(request_id, error=str(e))

    def _convert_message_to_sdk(self, message: Any) -> dict[str, Any] | None:
        """Convert internal message to SDK format.

        Args:
            message: The internal message object.

        Returns:
            A dictionary in SDK message format, or None if message should be skipped.
        """
        msg_type = getattr(message, "type", None)

        # Filter out progress messages (internal implementation detail)
        if msg_type == "progress":
            return None

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
                                # Convert MessageContent (Pydantic model) to dict
                                block_dict = self._convert_content_block(block)
                                if block_dict:
                                    content_blocks.append(block_dict)

            # Resolve model pointer to actual model name for SDK
            # The message may have model=None (unset), so fall back to QueryContext.model
            # Then resolve any pointer (e.g., "main") to the actual model name
            model_pointer = getattr(message, "model", None) or self._query_context.model
            model_profile = resolve_model_profile(model_pointer)
            actual_model = model_profile.model if model_profile else model_pointer

            stream_message = AssistantStreamMessage(
                message=AssistantMessageData(
                    content=content_blocks,
                    model=actual_model,
                ),
                parent_tool_use_id=getattr(message, "parent_tool_use_id", None),
            )
            return model_to_dict(stream_message)

        elif msg_type == "user":
            msg_content = getattr(message, "message", None)
            content = getattr(msg_content, "content", "") if msg_content else ""

            # If content is a list of MessageContent objects (e.g., tool results),
            # convert it to a string for the SDK format
            if isinstance(content, list):
                # For tool results, the content is a list of MessageContent objects.
                # Convert to a string representation. For tool_result types, use empty string
                # since the actual result data is in tool_use_result field.
                content_str = ""
                for block in content:
                    if isinstance(block, dict):
                        block_type = block.get("type", "")
                        if block_type == "text" and block.get("text"):
                            content_str = block.get("text", "")
                            break
                    elif hasattr(block, "type"):
                        block_type = getattr(block, "type", "")
                        if block_type == "text" and getattr(block, "text", None):
                            content_str = getattr(block, "text", "")
                            break
                        # For tool_result types, we typically use empty string
                        # since the data is in tool_use_result field
                content = content_str

            stream_message = UserStreamMessage(
                message=UserMessageData(content=content),
                uuid=getattr(message, "uuid", None),
                parent_tool_use_id=getattr(message, "parent_tool_use_id", None),
                tool_use_result=self._sanitize_for_json(getattr(message, "tool_use_result", None)),
            )
            return model_to_dict(stream_message)

        else:
            # Unknown message type - return None to skip
            return None

    def _sanitize_for_json(self, obj: Any) -> Any:
        """Recursively convert objects to JSON-serializable types.

        This function ensures Pydantic models and other objects are converted
        to dictionaries/lists/primitives that can be JSON serialized.

        Args:
            obj: The object to sanitize.

        Returns:
            A JSON-serializable version of the object.
        """
        # None values
        if obj is None:
            return None

        # Primitives
        elif isinstance(obj, (str, int, float, bool)):
            return obj

        # Lists and tuples
        elif isinstance(obj, (list, tuple)):
            return [self._sanitize_for_json(item) for item in obj]

        # Dictionaries
        elif isinstance(obj, dict):
            return {key: self._sanitize_for_json(value) for key, value in obj.items()}

        # Pydantic models
        elif hasattr(obj, "model_dump"):
            try:
                dumped = obj.model_dump(exclude_none=True)
                return self._sanitize_for_json(dumped)
            except Exception:
                pass

        # Objects with dict() method
        elif hasattr(obj, "dict"):
            try:
                dumped = obj.dict(exclude_none=True)
                return self._sanitize_for_json(dumped)
            except Exception:
                pass

        # Fallback: try to convert to string
        else:
            try:
                return str(obj)
            except Exception:
                return None

    def _convert_content_block(self, block: Any) -> dict[str, Any] | None:
        """Convert a MessageContent block to dictionary.

        Uses the same logic as _content_block_to_api in messages.py
        to ensure consistency and proper field mapping.

        Args:
            block: The MessageContent object.

        Returns:
            A dictionary representation of the block.
        """
        block_type = getattr(block, "type", None)

        if block_type == "text":
            return {
                "type": "text",
                "text": getattr(block, "text", None) or "",
            }

        elif block_type == "thinking":
            return {
                "type": "thinking",
                "thinking": getattr(block, "thinking", None) or getattr(block, "text", None) or "",
                "signature": getattr(block, "signature", None),
            }

        elif block_type == "tool_use":
            # Use the same id extraction logic as _content_block_to_api
            # Try id first, then tool_use_id, then empty string
            tool_id = (
                getattr(block, "id", None) or
                getattr(block, "tool_use_id", None) or
                ""
            )
            return {
                "type": "tool_use",
                "id": tool_id,
                "name": getattr(block, "name", None) or "",
                "input": getattr(block, "input", None) or {},
            }

        elif block_type == "tool_result":
            return {
                "type": "tool_result",
                "tool_use_id": getattr(block, "tool_use_id", None) or getattr(block, "id", None) or "",
                "content": getattr(block, "text", None) or getattr(block, "content", None) or "",
                "is_error": getattr(block, "is_error", None),
            }

        elif block_type == "image":
            return {
                "type": "image",
                "source": {
                    "type": getattr(block, "source_type", None) or "base64",
                    "media_type": getattr(block, "media_type", None) or "image/jpeg",
                    "data": getattr(block, "image_data", None) or "",
                },
            }

        else:
            # Unknown block type - try to convert with generic approach
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
            if hasattr(block, "content"):
                block_dict["content"] = block.content
            if hasattr(block, "is_error"):
                block_dict["is_error"] = block.is_error
            return block_dict if block_dict else None

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

            elif request_subtype == "set_permission_mode":
                await self._handle_set_permission_mode(request, request_id)

            elif request_subtype == "set_model":
                await self._handle_set_model(request, request_id)

            elif request_subtype == "rewind_files":
                await self._handle_rewind_files(request, request_id)

            elif request_subtype == "hook_callback":
                await self._handle_hook_callback(request, request_id)

            elif request_subtype == "can_use_tool":
                await self._handle_can_use_tool(request, request_id)

            else:
                await self._write_control_response(
                    request_id,
                    error=f"Unknown request subtype: {request_subtype}"
                )

        except Exception as e:
            logger.error(f"Error handling control request: {e}", exc_info=True)
            await self._write_control_response(request_id, error=str(e))

    async def _handle_set_permission_mode(
        self, request: dict[str, Any], request_id: str
    ) -> None:
        """Handle set_permission_mode request from SDK.

        Args:
            request: The set_permission_mode request data.
            request_id: The request ID.
        """
        mode = request.get("mode", "default")
        # Update the permission mode in the query context
        if self._query_context:
            # Map string mode to yolo_mode boolean
            self._query_context.yolo_mode = (mode == "bypassPermissions")

        await self._write_control_response(
            request_id,
            response={"status": "permission_mode_set", "mode": mode}
        )

    async def _handle_set_model(
        self, request: dict[str, Any], request_id: str
    ) -> None:
        """Handle set_model request from SDK.

        Args:
            request: The set_model request data.
            request_id: The request ID.
        """
        model = request.get("model")
        # Update the model in the query context
        if self._query_context:
            self._query_context.model = model or "main"

        await self._write_control_response(
            request_id,
            response={"status": "model_set", "model": model}
        )

    async def _handle_rewind_files(
        self, request: dict[str, Any], request_id: str
    ) -> None:
        """Handle rewind_files request from SDK.

        Note: File checkpointing is not currently supported.
        This method exists for Claude SDK API compatibility.

        Args:
            request: The rewind_files request data.
            request_id: The request ID.
        """
        await self._write_control_response(
            request_id,
            error="File checkpointing and rewind_files are not currently supported"
        )

    async def _handle_hook_callback(
        self, request: dict[str, Any], request_id: str
    ) -> None:
        """Handle hook_callback request from SDK.

        Args:
            request: The hook_callback request data.
            request_id: The request ID.
        """
        # Get callback info
        callback_id = request.get("callback_id")
        input_data = request.get("input", {})
        tool_use_id = request.get("tool_use_id")

        # For now, return a basic response
        # Full hook support would require integration with hook_manager
        await self._write_control_response(
            request_id,
            response={
                "continue": True,
            }
        )

    async def _handle_can_use_tool(
        self, request: dict[str, Any], request_id: str
    ) -> None:
        """Handle can_use_tool request from SDK.

        Args:
            request: The can_use_tool request data.
            request_id: The request ID.
        """
        tool_name = request.get("tool_name", "")
        tool_input = request.get("input", {})

        # Use the permission checker if available
        if self._can_use_tool:
            try:
                # Call the permission checker
                from ripperdoc_agent_sdk.types import ToolPermissionContext

                context = ToolPermissionContext(
                    signal=None,
                    suggestions=[],
                )

                result = await self._can_use_tool(tool_name, tool_input, context)

                # Convert result to response format
                from ripperdoc_agent_sdk.types import PermissionResultAllow

                if isinstance(result, PermissionResultAllow):
                    perm_response = PermissionResponseAllow(
                        updatedInput=result.updated_input or tool_input,
                    )
                    await self._write_control_response(
                        request_id,
                        response=model_to_dict(perm_response),
                    )
                else:
                    perm_response = PermissionResponseDeny(
                        message=result.message if hasattr(result, "message") else "",
                    )
                    await self._write_control_response(
                        request_id,
                        response=model_to_dict(perm_response),
                    )
            except Exception as e:
                logger.error(f"Error in permission check: {e}")
                await self._write_control_response(
                    request_id,
                    error=str(e)
                )
        else:
            # No permission checker, allow by default
            perm_response = PermissionResponseAllow(
                updatedInput=tool_input,
            )
            await self._write_control_response(
                request_id,
                response=model_to_dict(perm_response),
            )

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
