"""Rich-based CLI interface for Ripperdoc.

This module provides a clean, minimal terminal UI using Rich for the Ripperdoc agent.
"""

import asyncio
import json
import sys
import uuid
from typing import List, Dict, Any, Optional, Union, Iterable
from pathlib import Path

from rich.console import Console
from rich.markup import escape

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion, merge_completers
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.shortcuts.prompt import CompleteStyle

from ripperdoc.core.config import get_global_config, provider_protocol
from ripperdoc.core.default_tools import get_default_tools
from ripperdoc.core.query import query, QueryContext
from ripperdoc.core.system_prompt import build_system_prompt
from ripperdoc.core.skills import build_skill_summary, load_all_skills
from ripperdoc.core.hooks.manager import hook_manager
from ripperdoc.cli.commands import (
    get_slash_command,
    get_custom_command,
    list_slash_commands,
    list_custom_commands,
    slash_command_completions,
    expand_command_content,
    CustomCommandDefinition,
)
from ripperdoc.cli.ui.helpers import get_profile_for_pointer
from ripperdoc.core.permissions import make_permission_checker
from ripperdoc.cli.ui.spinner import Spinner
from ripperdoc.cli.ui.thinking_spinner import ThinkingSpinner
from ripperdoc.cli.ui.context_display import context_usage_lines
from ripperdoc.cli.ui.panels import create_welcome_panel, create_status_bar, print_shortcuts
from ripperdoc.cli.ui.message_display import MessageDisplay, parse_bash_output_sections
from ripperdoc.cli.ui.interrupt_handler import InterruptHandler
from ripperdoc.utils.conversation_compaction import (
    compact_conversation,
    CompactionResult,
    CompactionError,
    extract_tool_ids_from_message,
    get_complete_tool_pairs_tail,
)
from ripperdoc.utils.message_compaction import (
    estimate_conversation_tokens,
    estimate_used_tokens,
    get_context_usage_status,
    get_remaining_context_tokens,
    micro_compact_messages,
    resolve_auto_compact_enabled,
)
from ripperdoc.utils.token_estimation import estimate_tokens
from ripperdoc.utils.mcp import (
    ensure_mcp_runtime,
    format_mcp_instructions,
    load_mcp_servers_async,
    shutdown_mcp_runtime,
)
from ripperdoc.tools.mcp_tools import load_dynamic_mcp_tools_async, merge_tools_with_dynamic
from ripperdoc.utils.session_history import SessionHistory
from ripperdoc.utils.memory import build_memory_instructions
from ripperdoc.utils.messages import (
    UserMessage,
    AssistantMessage,
    ProgressMessage,
    create_user_message,
)
from ripperdoc.utils.log import enable_session_file_logging, get_logger
from ripperdoc.utils.path_ignore import build_ignore_filter
from ripperdoc.cli.ui.file_mention_completer import FileMentionCompleter
from ripperdoc.utils.message_formatting import stringify_message_content


# Type alias for conversation messages
ConversationMessage = Union[UserMessage, AssistantMessage, ProgressMessage]

console = Console()
logger = get_logger()


# Legacy aliases for backward compatibility with tests
_extract_tool_ids_from_message = extract_tool_ids_from_message
_get_complete_tool_pairs_tail = get_complete_tool_pairs_tail


class RichUI:
    """Rich-based UI for Ripperdoc."""

    def __init__(
        self,
        yolo_mode: bool = False,
        verbose: bool = False,
        show_full_thinking: Optional[bool] = None,
        session_id: Optional[str] = None,
        log_file_path: Optional[Path] = None,
    ):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self.console = console
        self.yolo_mode = yolo_mode
        self.verbose = verbose
        self.conversation_messages: List[ConversationMessage] = []
        self._saved_conversation: Optional[List[ConversationMessage]] = None
        self.query_context: Optional[QueryContext] = None
        self._current_tool: Optional[str] = None
        self._should_exit: bool = False
        self.command_list = list_slash_commands()
        self._custom_command_list = list_custom_commands()
        self._prompt_session: Optional[PromptSession] = None
        self.project_path = Path.cwd()
        # Track a stable session identifier for the current UI run.
        self.session_id = session_id or str(uuid.uuid4())
        if log_file_path:
            self.log_file_path = log_file_path
            logger.attach_file_handler(self.log_file_path)
        else:
            self.log_file_path = enable_session_file_logging(self.project_path, self.session_id)
        logger.info(
            "[ui] Initialized Rich UI session",
            extra={
                "session_id": self.session_id,
                "project_path": str(self.project_path),
                "log_file": str(self.log_file_path),
                "yolo_mode": self.yolo_mode,
                "verbose": self.verbose,
            },
        )
        self._session_history = SessionHistory(self.project_path, self.session_id)
        self._permission_checker = (
            None if yolo_mode else make_permission_checker(self.project_path, yolo_mode=False)
        )
        # Build ignore filter for file completion
        from ripperdoc.utils.path_ignore import get_project_ignore_patterns

        project_patterns = get_project_ignore_patterns()
        self._ignore_filter = build_ignore_filter(
            self.project_path,
            project_patterns=project_patterns,
            include_defaults=True,
            include_gitignore=True,
        )

        # Get global config for display preferences
        config = get_global_config()
        if show_full_thinking is None:
            self.show_full_thinking = config.show_full_thinking
        else:
            self.show_full_thinking = show_full_thinking

        # Initialize component handlers
        self._message_display = MessageDisplay(
            self.console, self.verbose, self.show_full_thinking
        )
        self._interrupt_handler = InterruptHandler()
        self._interrupt_handler.set_abort_callback(self._trigger_abort)

        # Keep MCP runtime alive for the whole UI session. Create it on the UI loop up front.
        try:
            self._run_async(ensure_mcp_runtime(self.project_path))
        except (OSError, RuntimeError, ConnectionError) as exc:
            logger.warning(
                "[ui] Failed to initialize MCP runtime at startup: %s: %s",
                type(exc).__name__,
                exc,
                extra={"session_id": self.session_id},
            )

        # Initialize hook manager with project context
        hook_manager.set_project_dir(self.project_path)
        hook_manager.set_session_id(self.session_id)
        logger.debug(
            "[ui] Initialized hook manager",
            extra={
                "session_id": self.session_id,
                "project_path": str(self.project_path),
            },
        )

    # ─────────────────────────────────────────────────────────────────────────────
    # Properties for backward compatibility with interrupt handler
    # ─────────────────────────────────────────────────────────────────────────────

    @property
    def _query_interrupted(self) -> bool:
        return self._interrupt_handler.was_interrupted

    @property
    def _esc_listener_paused(self) -> bool:
        return self._interrupt_handler._esc_listener_paused

    @_esc_listener_paused.setter
    def _esc_listener_paused(self, value: bool) -> None:
        self._interrupt_handler._esc_listener_paused = value

    def _context_usage_lines(
        self, breakdown: Any, model_label: str, auto_compact_enabled: bool
    ) -> List[str]:
        return context_usage_lines(breakdown, model_label, auto_compact_enabled)

    def _set_session(self, session_id: str) -> None:
        """Switch to a different session id and reset logging."""
        self.session_id = session_id
        self.log_file_path = enable_session_file_logging(self.project_path, self.session_id)
        logger.info(
            "[ui] Switched session",
            extra={
                "session_id": self.session_id,
                "project_path": str(self.project_path),
                "log_file": str(self.log_file_path),
            },
        )
        self._session_history = SessionHistory(self.project_path, session_id)

    def _log_message(self, message: Any) -> None:
        """Best-effort persistence of a message to the session log."""
        try:
            self._session_history.append(message)
        except (OSError, IOError, json.JSONDecodeError) as exc:
            # Logging failures should never interrupt the UI flow
            logger.warning(
                "[ui] Failed to append message to session history: %s: %s",
                type(exc).__name__,
                exc,
                extra={"session_id": self.session_id},
            )

    def _append_prompt_history(self, text: str) -> None:
        """Append text to the interactive prompt history."""
        if not text or not text.strip():
            return
        session = self.get_prompt_session()
        try:
            session.history.append_string(text)
        except (AttributeError, TypeError, ValueError) as exc:
            logger.warning(
                "[ui] Failed to append prompt history: %s: %s",
                type(exc).__name__,
                exc,
                extra={"session_id": self.session_id},
            )

    def replay_conversation(self, messages: List[Dict[str, Any]]) -> None:
        """Render a conversation history in the console and seed prompt history."""
        if not messages:
            return
        self.console.print("\n[dim]Restored conversation:[/dim]")
        for msg in messages:
            msg_type = getattr(msg, "type", "")
            message_payload = getattr(msg, "message", None) or getattr(msg, "content", None)
            content = getattr(message_payload, "content", None) if message_payload else None
            has_tool_result = False
            if isinstance(content, list):
                for block in content:
                    block_type = getattr(block, "type", None) or (
                        block.get("type") if isinstance(block, dict) else None
                    )
                    if block_type == "tool_result":
                        has_tool_result = True
                        break
            text = self._stringify_message_content(content)
            if not text:
                continue
            if msg_type == "user" and not has_tool_result:
                self.display_message("You", text)
                self._append_prompt_history(text)
            elif msg_type == "user" and has_tool_result:
                # Tool results are part of the conversation but should not enter prompt history.
                self.display_message("Tool", text, is_tool=True, tool_type="result")
            elif msg_type == "assistant":
                self.display_message("Ripperdoc", text)

    def get_default_tools(self) -> list:
        """Get the default set of tools."""
        return get_default_tools()

    def display_message(
        self,
        sender: str,
        content: str,
        is_tool: bool = False,
        tool_type: Optional[str] = None,
        tool_args: Optional[dict] = None,
        tool_data: Any = None,
        tool_error: bool = False,
    ) -> None:
        """Display a message in the conversation."""
        if not is_tool:
            self._message_display.print_human_or_assistant(sender, content)
            return

        if tool_type == "call":
            self._message_display.print_tool_call(sender, content, tool_args)
            return

        if tool_type == "result":
            self._message_display.print_tool_result(
                sender, content, tool_data, tool_error, parse_bash_output_sections
            )
            return

        self._message_display.print_generic_tool(sender, content)

    # Delegate to MessageDisplay for backward compatibility
    def _format_tool_args(self, tool_name: str, tool_args: Optional[dict]) -> list[str]:
        return self._message_display.format_tool_args(tool_name, tool_args)

    def _print_tool_call(self, sender: str, content: str, tool_args: Optional[dict]) -> None:
        self._message_display.print_tool_call(sender, content, tool_args)

    def _print_tool_result(
        self, sender: str, content: str, tool_data: Any, tool_error: bool = False
    ) -> None:
        self._message_display.print_tool_result(
            sender, content, tool_data, tool_error, parse_bash_output_sections
        )

    def _print_generic_tool(self, sender: str, content: str) -> None:
        self._message_display.print_generic_tool(sender, content)

    def _print_human_or_assistant(self, sender: str, content: str) -> None:
        self._message_display.print_human_or_assistant(sender, content)

    def _stringify_message_content(self, content: Any) -> str:
        return stringify_message_content(content)

    def _print_reasoning(self, reasoning: Any) -> None:
        self._message_display.print_reasoning(reasoning)

    async def _prepare_query_context(self, user_input: str) -> tuple[str, Dict[str, str]]:
        """Load MCP servers, skills, and build system prompt.

        Returns:
            Tuple of (system_prompt, context_dict)
        """
        context: Dict[str, str] = {}
        servers = await load_mcp_servers_async(self.project_path)
        dynamic_tools = await load_dynamic_mcp_tools_async(self.project_path)

        if dynamic_tools and self.query_context:
            self.query_context.tools = merge_tools_with_dynamic(
                self.query_context.tools, dynamic_tools
            )

        logger.debug(
            "[ui] Prepared tools and MCP servers",
            extra={
                "session_id": self.session_id,
                "tool_count": len(self.query_context.tools) if self.query_context else 0,
                "mcp_servers": len(servers),
                "dynamic_tools": len(dynamic_tools),
            },
        )

        mcp_instructions = format_mcp_instructions(servers)
        skill_result = load_all_skills(self.project_path)

        for err in skill_result.errors:
            logger.warning(
                "[skills] Failed to load skill",
                extra={
                    "path": str(err.path),
                    "reason": err.reason,
                    "session_id": self.session_id,
                },
            )

        skill_instructions = build_skill_summary(skill_result.skills)
        additional_instructions: List[str] = []
        if skill_instructions:
            additional_instructions.append(skill_instructions)

        memory_instructions = build_memory_instructions()
        if memory_instructions:
            additional_instructions.append(memory_instructions)

        system_prompt = build_system_prompt(
            self.query_context.tools if self.query_context else [],
            user_input,
            context,
            additional_instructions=additional_instructions or None,
            mcp_instructions=mcp_instructions,
        )

        return system_prompt, context

    async def _check_and_compact_messages(
        self,
        messages: List[ConversationMessage],
        max_context_tokens: int,
        auto_compact_enabled: bool,
        protocol: str,
    ) -> List[ConversationMessage]:
        """Check context usage and compact if needed.

        Returns:
            Possibly compacted list of messages.
        """
        micro = micro_compact_messages(
            messages,
            context_limit=max_context_tokens,
            auto_compact_enabled=auto_compact_enabled,
            protocol=protocol,
        )
        if micro.was_compacted:
            messages = micro.messages  # type: ignore[assignment]
            logger.info(
                "[ui] Micro-compacted conversation",
                extra={
                    "session_id": self.session_id,
                    "tokens_before": micro.tokens_before,
                    "tokens_after": micro.tokens_after,
                    "tokens_saved": micro.tokens_saved,
                    "tools_compacted": micro.tools_compacted,
                    "trigger": micro.trigger_type,
                },
            )

        used_tokens = estimate_used_tokens(messages, protocol=protocol)  # type: ignore[arg-type]
        usage_status = get_context_usage_status(
            used_tokens, max_context_tokens, auto_compact_enabled
        )

        logger.debug(
            "[ui] Context usage snapshot",
            extra={
                "session_id": self.session_id,
                "used_tokens": used_tokens,
                "max_context_tokens": max_context_tokens,
                "percent_used": round(usage_status.percent_used, 2),
                "auto_compact_enabled": auto_compact_enabled,
            },
        )

        if usage_status.is_above_warning:
            console.print(
                f"[yellow]Context usage is {usage_status.percent_used:.1f}% "
                f"({usage_status.total_tokens}/{usage_status.max_context_tokens} tokens).[/yellow]"
            )
            if not auto_compact_enabled:
                console.print(
                    "[dim]Auto-compaction is disabled; run /compact to trim history.[/dim]"
                )

        if usage_status.should_auto_compact:
            original_messages = list(messages)
            spinner = Spinner(self.console, "Summarizing conversation...", spinner="dots")
            try:
                spinner.start()
                result = await compact_conversation(
                    messages, custom_instructions="", protocol=protocol
                )
            finally:
                spinner.stop()

            if isinstance(result, CompactionResult):
                if self._saved_conversation is None:
                    self._saved_conversation = original_messages  # type: ignore[assignment]
                console.print(
                    f"[yellow]Auto-compacted conversation (saved ~{result.tokens_saved} tokens). "
                    f"Estimated usage: {result.tokens_after}/{max_context_tokens} tokens.[/yellow]"
                )
                logger.info(
                    "[ui] Auto-compacted conversation",
                    extra={
                        "session_id": self.session_id,
                        "tokens_before": result.tokens_before,
                        "tokens_after": result.tokens_after,
                        "tokens_saved": result.tokens_saved,
                    },
                )
                return result.messages
            elif isinstance(result, CompactionError):
                logger.warning(
                    "[ui] Auto-compaction failed: %s",
                    result.message,
                    extra={"session_id": self.session_id},
                )

        return messages

    def _handle_assistant_message(
        self,
        message: AssistantMessage,
        tool_registry: Dict[str, Dict[str, Any]],
        spinner: Optional[ThinkingSpinner] = None,
    ) -> Optional[str]:
        """Handle an assistant message from the query stream.

        Returns:
            The last tool name if a tool_use block was processed, None otherwise.
        """
        # Factory to create pause context - spinner.paused() if spinner exists, else no-op
        from contextlib import nullcontext

        pause = lambda: spinner.paused() if spinner else nullcontext()  # noqa: E731

        meta = getattr(getattr(message, "message", None), "metadata", {}) or {}
        reasoning_payload = (
            meta.get("reasoning_content") or meta.get("reasoning") or meta.get("reasoning_details")
        )
        if reasoning_payload:
            with pause():
                self._print_reasoning(reasoning_payload)

        last_tool_name: Optional[str] = None

        if isinstance(message.message.content, str):
            with pause():
                self.display_message("Ripperdoc", message.message.content)
        elif isinstance(message.message.content, list):
            for block in message.message.content:
                if hasattr(block, "type") and block.type == "text" and block.text:
                    with pause():
                        self.display_message("Ripperdoc", block.text)
                elif hasattr(block, "type") and block.type == "tool_use":
                    tool_name = getattr(block, "name", "unknown tool")
                    tool_args = getattr(block, "input", {})
                    tool_use_id = getattr(block, "tool_use_id", None) or getattr(block, "id", None)

                    if tool_use_id:
                        tool_registry[tool_use_id] = {
                            "name": tool_name,
                            "args": tool_args,
                            "printed": False,
                        }

                    if tool_name == "Task":
                        with pause():
                            self.display_message(
                                tool_name, "", is_tool=True, tool_type="call", tool_args=tool_args
                            )
                        if tool_use_id:
                            tool_registry[tool_use_id]["printed"] = True

                    last_tool_name = tool_name

        return last_tool_name

    def _handle_tool_result_message(
        self,
        message: UserMessage,
        tool_registry: Dict[str, Dict[str, Any]],
        last_tool_name: Optional[str],
        spinner: Optional[ThinkingSpinner] = None,
    ) -> None:
        """Handle a user message containing tool results."""
        if not isinstance(message.message.content, list):
            return

        # Factory to create pause context - spinner.paused() if spinner exists, else no-op
        from contextlib import nullcontext

        pause = lambda: spinner.paused() if spinner else nullcontext()  # noqa: E731

        for block in message.message.content:
            if not (hasattr(block, "type") and block.type == "tool_result" and block.text):
                continue

            tool_name = "Tool"
            tool_data = getattr(message, "tool_use_result", None)
            is_error = bool(getattr(block, "is_error", False))
            tool_use_id = getattr(block, "tool_use_id", None)

            entry = tool_registry.get(tool_use_id) if tool_use_id else None
            if entry:
                tool_name = entry.get("name", tool_name)
                if not entry.get("printed"):
                    with pause():
                        self.display_message(
                            tool_name,
                            "",
                            is_tool=True,
                            tool_type="call",
                            tool_args=entry.get("args", {}),
                        )
                    entry["printed"] = True
            elif last_tool_name:
                tool_name = last_tool_name

            with pause():
                self.display_message(
                    tool_name,
                    block.text,
                    is_tool=True,
                    tool_type="result",
                    tool_data=tool_data,
                    tool_error=is_error,
                )

    def _handle_progress_message(
        self,
        message: ProgressMessage,
        spinner: ThinkingSpinner,
        output_token_est: int,
    ) -> int:
        """Handle a progress message and update spinner.

        Returns:
            Updated output token estimate.
        """
        if self.verbose:
            with spinner.paused():
                self.display_message("System", f"Progress: {message.content}", is_tool=True)
        elif message.content and isinstance(message.content, str):
            if message.content.startswith("Subagent: "):
                with spinner.paused():
                    self.display_message(
                        "Subagent", message.content[len("Subagent: ") :], is_tool=True
                    )
            elif message.content.startswith("Subagent"):
                with spinner.paused():
                    self.display_message("Subagent", message.content, is_tool=True)

        if message.tool_use_id == "stream":
            delta_tokens = estimate_tokens(message.content)
            output_token_est += delta_tokens
            spinner.update_tokens(output_token_est)
        else:
            spinner.update_tokens(output_token_est, suffix=f"Working... {message.content}")

        return output_token_est

    async def process_query(self, user_input: str) -> None:
        """Process a user query and display the response."""
        # Initialize or reset query context
        if not self.query_context:
            self.query_context = QueryContext(
                tools=self.get_default_tools(), yolo_mode=self.yolo_mode, verbose=self.verbose
            )
        else:
            abort_controller = getattr(self.query_context, "abort_controller", None)
            if abort_controller is not None:
                abort_controller.clear()

        logger.info(
            "[ui] Starting query processing",
            extra={
                "session_id": self.session_id,
                "prompt_length": len(user_input),
                "prompt_preview": user_input[:200],
            },
        )

        try:
            # Prepare context and system prompt
            system_prompt, context = await self._prepare_query_context(user_input)

            # Create and log user message
            user_message = create_user_message(user_input)
            messages: List[ConversationMessage] = self.conversation_messages + [user_message]
            self._log_message(user_message)
            self._append_prompt_history(user_input)

            # Get model configuration
            config = get_global_config()
            model_profile = get_profile_for_pointer("main")
            max_context_tokens = get_remaining_context_tokens(
                model_profile, config.context_token_limit
            )
            auto_compact_enabled = resolve_auto_compact_enabled(config)
            protocol = provider_protocol(model_profile.provider) if model_profile else "openai"

            # Check and potentially compact messages
            messages = await self._check_and_compact_messages(
                messages, max_context_tokens, auto_compact_enabled, protocol
            )

            # Setup spinner and callbacks
            prompt_tokens_est = estimate_conversation_tokens(messages, protocol=protocol)
            spinner = ThinkingSpinner(console, prompt_tokens_est)

            def pause_ui() -> None:
                spinner.stop()

            def resume_ui() -> None:
                spinner.start()
                spinner.update("Thinking...")

            self.query_context.pause_ui = pause_ui
            self.query_context.resume_ui = resume_ui

            # Create permission checker with spinner control
            base_permission_checker = self._permission_checker

            async def permission_checker(tool: Any, parsed_input: Any) -> bool:
                spinner.stop()
                was_paused = self._pause_interrupt_listener()
                try:
                    if base_permission_checker is not None:
                        result = await base_permission_checker(tool, parsed_input)
                        allowed = result.result if hasattr(result, "result") else True
                        logger.debug(
                            "[ui] Permission check result",
                            extra={
                                "tool": getattr(tool, "name", None),
                                "allowed": allowed,
                                "session_id": self.session_id,
                            },
                        )
                        return allowed
                    return True
                finally:
                    self._resume_interrupt_listener(was_paused)
                    # Wrap spinner restart in try-except to prevent exceptions
                    # from discarding the permission result
                    try:
                        spinner.start()
                        spinner.update("Thinking...")
                    except (RuntimeError, ValueError, OSError) as exc:
                        logger.debug(
                            "[ui] Failed to restart spinner after permission check: %s: %s",
                            type(exc).__name__,
                            exc,
                        )

            # Process query stream
            tool_registry: Dict[str, Dict[str, Any]] = {}
            last_tool_name: Optional[str] = None
            output_token_est = 0

            try:
                spinner.start()
                async for message in query(
                    messages,
                    system_prompt,
                    context,
                    self.query_context,
                    permission_checker,  # type: ignore[arg-type]
                ):
                    if message.type == "assistant" and isinstance(message, AssistantMessage):
                        result = self._handle_assistant_message(message, tool_registry, spinner)
                        if result:
                            last_tool_name = result

                    elif message.type == "user" and isinstance(message, UserMessage):
                        self._handle_tool_result_message(
                            message, tool_registry, last_tool_name, spinner
                        )

                    elif message.type == "progress" and isinstance(message, ProgressMessage):
                        output_token_est = self._handle_progress_message(
                            message, spinner, output_token_est
                        )

                    self._log_message(message)
                    messages.append(message)  # type: ignore[arg-type]

            except asyncio.CancelledError:
                # Re-raise cancellation to allow proper cleanup
                raise
            except (OSError, ConnectionError, RuntimeError, ValueError, KeyError, TypeError) as e:
                logger.warning(
                    "[ui] Error while processing streamed query response: %s: %s",
                    type(e).__name__,
                    e,
                    extra={"session_id": self.session_id},
                )
                self.display_message("System", f"Error: {str(e)}", is_tool=True)
            finally:
                try:
                    spinner.stop()
                except (RuntimeError, ValueError, OSError) as exc:
                    logger.warning(
                        "[ui] Failed to stop spinner: %s: %s",
                        type(exc).__name__,
                        exc,
                        extra={"session_id": self.session_id},
                    )

                self.conversation_messages = messages
                logger.info(
                    "[ui] Query processing completed",
                    extra={
                        "session_id": self.session_id,
                        "conversation_messages": len(self.conversation_messages),
                        "project_path": str(self.project_path),
                    },
                )

        except asyncio.CancelledError:
            # Re-raise cancellation to allow proper cleanup
            raise
        except (OSError, ConnectionError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            logger.warning(
                "[ui] Error during query processing: %s: %s",
                type(exc).__name__,
                exc,
                extra={"session_id": self.session_id},
            )
            self.display_message("System", f"Error: {str(exc)}", is_tool=True)

    # ─────────────────────────────────────────────────────────────────────────────
    # ESC Key Interrupt Support
    # ─────────────────────────────────────────────────────────────────────────────

    # Delegate to InterruptHandler
    def _pause_interrupt_listener(self) -> bool:
        return self._interrupt_handler.pause_listener()

    def _resume_interrupt_listener(self, previous_state: bool) -> None:
        self._interrupt_handler.resume_listener(previous_state)

    def _trigger_abort(self) -> None:
        """Signal the query to abort."""
        if self.query_context and hasattr(self.query_context, "abort_controller"):
            self.query_context.abort_controller.set()

    async def _run_query_with_esc_interrupt(self, query_coro: Any) -> bool:
        """Run a query with ESC key interrupt support."""
        return await self._interrupt_handler.run_with_interrupt(query_coro)

    def _run_async(self, coro: Any) -> Any:
        """Run a coroutine on the persistent event loop."""
        if self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        return self._loop.run_until_complete(coro)

    def _run_async_with_esc_interrupt(self, coro: Any) -> bool:
        """Run a coroutine with ESC key interrupt support.

        Returns True if interrupted by ESC, False if completed normally.
        """
        if self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        return self._loop.run_until_complete(self._run_query_with_esc_interrupt(coro))

    def run_async(self, coro: Any) -> Any:
        """Public wrapper for running coroutines on the UI event loop."""
        return self._run_async(coro)

    def handle_slash_command(self, user_input: str) -> bool | str:
        """Handle slash commands. Returns True if handled as built-in, False if not a command,
        or a string if it's a custom command that should be sent to the AI."""

        if not user_input.startswith("/"):
            return False

        parts = user_input[1:].strip().split()
        if not parts:
            self.console.print("[red]No command provided after '/'.[/red]")
            return True

        command_name = parts[0].lower()
        trimmed_arg = " ".join(parts[1:]).strip()

        # First, try built-in commands
        command = get_slash_command(command_name)
        if command is not None:
            return command.handler(self, trimmed_arg)

        # Then, try custom commands
        custom_cmd = get_custom_command(command_name, self.project_path)
        if custom_cmd is not None:
            # Expand the custom command content
            expanded_content = expand_command_content(custom_cmd, trimmed_arg, self.project_path)

            # Show a hint that this is from a custom command
            self.console.print(f"[dim]Running custom command: /{command_name}[/dim]")
            if custom_cmd.argument_hint and trimmed_arg:
                self.console.print(f"[dim]Arguments: {trimmed_arg}[/dim]")

            # Return the expanded content to be processed as a query
            return expanded_content

        self.console.print(f"[red]Unknown command: {escape(command_name)}[/red]")
        return True

    def get_prompt_session(self) -> PromptSession:
        """Create (or return) the prompt session with command completion."""
        if self._prompt_session:
            return self._prompt_session

        class SlashCommandCompleter(Completer):
            """Autocomplete for slash commands including custom commands."""

            def __init__(self, project_path: Path):
                self.project_path = project_path

            def get_completions(self, document: Any, complete_event: Any) -> Iterable[Completion]:
                text = document.text_before_cursor
                if not text.startswith("/"):
                    return
                query = text[1:]
                # Get completions including custom commands
                completions = slash_command_completions(self.project_path)
                for name, cmd in completions:
                    if name.startswith(query):
                        # Handle both SlashCommand and CustomCommandDefinition
                        description = cmd.description
                        # Add hint for custom commands
                        if isinstance(cmd, CustomCommandDefinition):
                            hint = cmd.argument_hint or ""
                            display = f"{name} {hint}".strip() if hint else name
                            display_meta = f"[custom] {description}"
                        else:
                            display = name
                            display_meta = description
                        yield Completion(
                            name,
                            start_position=-len(query),
                            display=display,
                            display_meta=display_meta,
                        )

        # Merge both completers
        slash_completer = SlashCommandCompleter(self.project_path)
        file_completer = FileMentionCompleter(self.project_path, self._ignore_filter)
        combined_completer = merge_completers([slash_completer, file_completer])

        key_bindings = KeyBindings()

        @key_bindings.add("enter")
        def _(event: Any) -> None:
            """Accept completion if menu is open; otherwise submit line."""
            buf = event.current_buffer
            if buf.complete_state and buf.complete_state.current_completion:
                buf.apply_completion(buf.complete_state.current_completion)
                return
            buf.validate_and_handle()

        @key_bindings.add("tab")
        def _(event: Any) -> None:
            """Use Tab to accept the highlighted completion when visible."""
            buf = event.current_buffer
            if buf.complete_state and buf.complete_state.current_completion:
                buf.apply_completion(buf.complete_state.current_completion)
            else:
                buf.start_completion(select_first=True)

        self._prompt_session = PromptSession(
            completer=combined_completer,
            complete_style=CompleteStyle.COLUMN,
            complete_while_typing=True,
            history=InMemoryHistory(),
            key_bindings=key_bindings,
        )
        return self._prompt_session

    def run(self) -> None:
        """Run the Rich-based interface."""
        # Display welcome panel
        console.print()
        console.print(create_welcome_panel())
        console.print()

        # Display status
        console.print(create_status_bar())
        console.print()
        console.print(
            "[dim]Tip: type '/' then press Tab to see available commands. Type '@' to mention files. Press ESC to interrupt a running query.[/dim]\n"
        )

        session = self.get_prompt_session()
        logger.info(
            "[ui] Starting interactive loop",
            extra={"session_id": self.session_id, "log_file": str(self.log_file_path)},
        )

        try:
            while not self._should_exit:
                try:
                    # Get user input
                    user_input = session.prompt("> ")

                    if not user_input.strip():
                        continue

                    if user_input.strip() == "?":
                        self._print_shortcuts()
                        console.print()
                        continue

                    # Handle slash commands locally
                    if user_input.startswith("/"):
                        logger.debug(
                            "[ui] Received slash command",
                            extra={"session_id": self.session_id, "command": user_input},
                        )
                        handled = self.handle_slash_command(user_input)
                        if self._should_exit:
                            break
                        # If handled is a string, it's expanded custom command content
                        if isinstance(handled, str):
                            # Process the expanded custom command as a query
                            user_input = handled
                        elif handled:
                            console.print()  # spacing
                            continue

                    # Process the query
                    logger.info(
                        "[ui] Processing interactive prompt",
                        extra={
                            "session_id": self.session_id,
                            "prompt_length": len(user_input),
                            "prompt_preview": user_input[:200],
                        },
                    )
                    interrupted = self._run_async_with_esc_interrupt(self.process_query(user_input))

                    if interrupted:
                        console.print(
                            "\n[red]■ Conversation interrupted[/red] · [dim]Tell the model what to do differently.[/dim]"
                        )
                        logger.info(
                            "[ui] Query interrupted by ESC key",
                            extra={"session_id": self.session_id},
                        )

                    console.print()  # Add spacing between interactions

                except KeyboardInterrupt:
                    # Signal abort to cancel running queries
                    if self.query_context:
                        abort_controller = getattr(self.query_context, "abort_controller", None)
                        if abort_controller is not None:
                            abort_controller.set()
                    console.print("\n[yellow]Goodbye![/yellow]")
                    break
                except EOFError:
                    console.print("\n[yellow]Goodbye![/yellow]")
                    break
                except (
                    OSError,
                    ConnectionError,
                    RuntimeError,
                    ValueError,
                    KeyError,
                    TypeError,
                ) as e:
                    console.print(f"[red]Error: {escape(str(e))}[/]")
                    logger.warning(
                        "[ui] Error in interactive loop: %s: %s",
                        type(e).__name__,
                        e,
                        extra={"session_id": self.session_id},
                    )
                    if self.verbose:
                        import traceback

                        console.print(traceback.format_exc())
        finally:
            # Cancel any running tasks before shutdown
            if self.query_context:
                abort_controller = getattr(self.query_context, "abort_controller", None)
                if abort_controller is not None:
                    abort_controller.set()

            # Suppress async generator cleanup errors during shutdown
            original_hook = sys.unraisablehook

            def _quiet_unraisable_hook(unraisable: Any) -> None:
                # Suppress "asynchronous generator is already running" errors during shutdown
                if isinstance(unraisable.exc_value, RuntimeError):
                    if "asynchronous generator is already running" in str(unraisable.exc_value):
                        return
                # Call original hook for other errors
                original_hook(unraisable)

            sys.unraisablehook = _quiet_unraisable_hook

            try:
                try:
                    self._run_async(shutdown_mcp_runtime())
                except (OSError, RuntimeError, ConnectionError, asyncio.CancelledError) as exc:
                    # pragma: no cover - defensive shutdown
                    logger.warning(
                        "[ui] Failed to shut down MCP runtime cleanly: %s: %s",
                        type(exc).__name__,
                        exc,
                        extra={"session_id": self.session_id},
                    )
            finally:
                if not self._loop.is_closed():
                    # Cancel all pending tasks
                    pending = asyncio.all_tasks(self._loop)
                    for task in pending:
                        task.cancel()

                    # Allow cancelled tasks to clean up
                    if pending:
                        try:
                            self._loop.run_until_complete(
                                asyncio.gather(*pending, return_exceptions=True)
                            )
                        except (RuntimeError, asyncio.CancelledError):
                            pass  # Ignore errors during task cancellation

                    # Shutdown async generators (suppress expected errors)
                    try:
                        self._loop.run_until_complete(self._loop.shutdown_asyncgens())
                    except (RuntimeError, asyncio.CancelledError):
                        # Expected during forced shutdown - async generators may already be running
                        pass

                    self._loop.close()
                asyncio.set_event_loop(None)
                sys.unraisablehook = original_hook

    async def _run_manual_compact(self, custom_instructions: str) -> None:
        """Manual compaction: clear bulky tool output and summarize conversation."""
        from rich.markup import escape

        model_profile = get_profile_for_pointer("main")
        protocol = provider_protocol(model_profile.provider) if model_profile else "openai"

        if len(self.conversation_messages) < 2:
            self.console.print("[yellow]Not enough conversation history to compact.[/yellow]")
            return

        original_messages = list(self.conversation_messages)
        spinner = Spinner(self.console, "Summarizing conversation...", spinner="dots")

        try:
            spinner.start()
            result = await compact_conversation(
                self.conversation_messages,
                custom_instructions,
                protocol=protocol,
            )
        except Exception as exc:
            import traceback

            self.console.print(f"[red]Error during compaction: {escape(str(exc))}[/red]")
            self.console.print(f"[dim red]{traceback.format_exc()}[/dim red]")
            return
        finally:
            spinner.stop()

        if isinstance(result, CompactionResult):
            self._saved_conversation = original_messages
            self.conversation_messages = result.messages
            self.console.print(
                f"[green]✓ Conversation compacted[/green] "
                f"(saved ~{result.tokens_saved} tokens). Use /resume to restore full history."
            )
        elif isinstance(result, CompactionError):
            self.console.print(f"[red]{escape(result.message)}[/red]")

    def _print_shortcuts(self) -> None:
        """Show common keyboard shortcuts and prefixes."""
        print_shortcuts(self.console)


def check_onboarding_rich() -> bool:
    """Check if onboarding is complete and run if needed."""
    config = get_global_config()

    if config.has_completed_onboarding:
        return True

    # Use the wizard onboarding
    from ripperdoc.cli.ui.wizard import check_onboarding

    return check_onboarding()


def main_rich(
    yolo_mode: bool = False,
    verbose: bool = False,
    show_full_thinking: Optional[bool] = None,
    session_id: Optional[str] = None,
    log_file_path: Optional[Path] = None,
) -> None:
    """Main entry point for Rich interface."""

    # Ensure onboarding is complete
    if not check_onboarding_rich():
        sys.exit(1)

    # Run the Rich UI
    ui = RichUI(
        yolo_mode=yolo_mode,
        verbose=verbose,
        show_full_thinking=show_full_thinking,
        session_id=session_id,
        log_file_path=log_file_path,
    )
    ui.run()


if __name__ == "__main__":
    main_rich()
