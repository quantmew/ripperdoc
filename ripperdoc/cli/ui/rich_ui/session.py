"""Rich-based CLI interface for Ripperdoc.

This module provides a clean, minimal terminal UI using Rich for the Ripperdoc agent.
"""

import asyncio
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import FormattedText
from rich.console import Console
from rich.markup import escape

from ripperdoc.core.config import get_global_config, provider_protocol
from ripperdoc.core.default_tools import filter_tools_by_names, get_default_tools
from ripperdoc.core.theme import get_theme_manager
from ripperdoc.core.query import query, QueryContext
from ripperdoc.core.system_prompt import build_system_prompt
from ripperdoc.core.skills import build_skill_summary, load_all_skills
from ripperdoc.core.hooks.manager import hook_manager
from ripperdoc.core.hooks.llm_callback import build_hook_llm_callback
from ripperdoc.cli.commands import list_custom_commands, list_slash_commands
from ripperdoc.cli.ui.helpers import get_profile_for_pointer
from ripperdoc.core.permissions import make_permission_checker
from ripperdoc.cli.ui.spinner import Spinner
from ripperdoc.cli.ui.thinking_spinner import ThinkingSpinner
from ripperdoc.cli.ui.context_display import context_usage_lines
from ripperdoc.cli.ui.panels import create_welcome_panel, print_shortcuts
from ripperdoc.cli.ui.message_display import MessageDisplay, parse_bash_output_sections
from ripperdoc.cli.ui.interrupt_listener import EscInterruptListener
from ripperdoc.utils.conversation_compaction import (
    compact_conversation,
    CompactionResult,
    CompactionError,
)
from ripperdoc.utils.message_compaction import (
    estimate_conversation_tokens,
    estimate_used_tokens,
    get_context_usage_status,
    get_remaining_context_tokens,
    micro_compact_messages,
    resolve_auto_compact_enabled,
)
from ripperdoc.utils.mcp import (
    ensure_mcp_runtime,
    format_mcp_instructions,
    load_mcp_servers_async,
    shutdown_mcp_runtime,
)
from ripperdoc.utils.lsp import shutdown_lsp_manager
from ripperdoc.tools.background_shell import shutdown_background_shell
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
from ripperdoc.cli.ui.tips import get_random_tip
from ripperdoc.utils.message_formatting import stringify_message_content

from ripperdoc.cli.ui.rich_ui.commands import handle_slash_command as _handle_slash_command
from ripperdoc.cli.ui.rich_ui.images import process_images_in_input
from ripperdoc.cli.ui.rich_ui.input import build_prompt_session
from ripperdoc.cli.ui.rich_ui.rendering import (
    handle_assistant_message,
    handle_progress_message,
    handle_tool_result_message,
)


# Type alias for conversation messages
ConversationMessage = Union[UserMessage, AssistantMessage, ProgressMessage]

console = Console()
logger = get_logger()


class RichUI:
    """Rich-based UI for Ripperdoc."""

    def __init__(
        self,
        yolo_mode: bool = False,
        verbose: bool = False,
        show_full_thinking: Optional[bool] = None,
        session_id: Optional[str] = None,
        log_file_path: Optional[Path] = None,
        allowed_tools: Optional[List[str]] = None,
        custom_system_prompt: Optional[str] = None,
        append_system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        resume_messages: Optional[List[Any]] = None,
        initial_query: Optional[str] = None,
    ):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self.console = console
        self.yolo_mode = yolo_mode
        self.verbose = verbose
        self.allowed_tools = allowed_tools
        self.custom_system_prompt = custom_system_prompt
        self.append_system_prompt = append_system_prompt
        self.model = model or "main"
        self.conversation_messages: List[ConversationMessage] = []
        self._saved_conversation: Optional[List[ConversationMessage]] = None
        self.query_context: Optional[QueryContext] = None
        self._current_tool: Optional[str] = None
        self._should_exit: bool = False
        self._last_ctrl_c_time: float = 0.0  # Track Ctrl+C timing for double-press exit
        self._initial_query = initial_query  # Query from piped stdin to auto-send on startup
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
        self._session_hook_contexts: List[str] = []
        self._session_start_time = time.time()
        self._session_end_sent = False
        self._exit_reason: Optional[str] = None
        self._using_tty_input = False  # Track if we're using /dev/tty for input
        self._thinking_mode_enabled = False  # Toggle for extended thinking mode
        self._interrupt_listener = EscInterruptListener(self._schedule_esc_interrupt, logger=logger)
        self._esc_interrupt_seen = False
        self._query_in_progress = False
        self._active_spinner: Optional[ThinkingSpinner] = None
        hook_manager.set_transcript_path(str(self._session_history.path))

        # Create permission checker with Rich console and PromptSession support
        if not yolo_mode:
            # Create a dedicated PromptSession for permission dialogs
            # This provides better interrupt handling than console.input()
            from prompt_toolkit import PromptSession

            # Disable CPR (Cursor Position Request) to avoid warnings in terminals
            # that don't support it (like some remote/CI terminals)
            import os
            os.environ['PROMPT_TOOLKIT_NO_CPR'] = '1'

            permission_session = PromptSession()

            self._permission_checker = make_permission_checker(
                self.project_path,
                yolo_mode=False,
                console=console,  # Pass console for Rich Panel rendering
                prompt_session=permission_session,  # Use PromptSession for input
            )
        else:
            self._permission_checker = None
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

        # Initialize theme from config
        theme_manager = get_theme_manager()
        theme_name = getattr(config, "theme", None) or "dark"
        if not theme_manager.set_theme(theme_name):
            theme_manager.set_theme("dark")  # Fallback to default

        # Initialize component handlers
        self._message_display = MessageDisplay(self.console, self.verbose, self.show_full_thinking)

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
        hook_manager.set_llm_callback(build_hook_llm_callback())
        logger.debug(
            "[ui] Initialized hook manager",
            extra={
                "session_id": self.session_id,
                "project_path": str(self.project_path),
            },
        )
        self._run_session_start("startup")

        # Handle resume_messages if provided (for --continue)
        if resume_messages:
            self.conversation_messages = list(resume_messages)
            logger.info(
                "[ui] Resumed conversation with messages",
                extra={
                    "session_id": self.session_id,
                    "message_count": len(resume_messages),
                },
            )

    # ─────────────────────────────────────────────────────────────────────────────
    # Properties for backward compatibility with interrupt handler
    # ─────────────────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────────────────
    # Thinking mode toggle
    # ─────────────────────────────────────────────────────────────────────────────

    def _supports_thinking_mode(self) -> bool:
        """Check if the current model supports extended thinking mode."""
        from ripperdoc.core.query import infer_thinking_mode
        from ripperdoc.core.config import ProviderType

        model_profile = get_profile_for_pointer("main")
        if not model_profile:
            return False
        # Anthropic natively supports thinking mode
        if model_profile.provider == ProviderType.ANTHROPIC:
            return True
        # For other providers, check if we can infer a thinking mode
        return infer_thinking_mode(model_profile) is not None

    def _toggle_thinking_mode(self) -> None:
        """Toggle thinking mode on/off. Status is shown in rprompt."""
        if not self._supports_thinking_mode():
            self.console.print("[yellow]Current model does not support thinking mode.[/yellow]")
            return
        self._thinking_mode_enabled = not self._thinking_mode_enabled

    def _get_thinking_tokens(self) -> int:
        """Get the thinking tokens budget based on current mode."""
        if not self._thinking_mode_enabled:
            return 0
        config = get_global_config()
        return config.default_thinking_tokens

    def _get_prompt(self) -> str:
        """Generate the input prompt."""
        return "> "

    def _get_rprompt(self) -> Union[str, FormattedText]:
        """Generate the right prompt with thinking mode status."""
        if not self._supports_thinking_mode():
            return ""
        if self._thinking_mode_enabled:
            return FormattedText(
                [
                    ("class:rprompt-on", "⚡ Thinking"),
                ]
            )
        return FormattedText(
            [
                ("class:rprompt-off", "Thinking: off"),
            ]
        )

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
        hook_manager.set_session_id(self.session_id)
        hook_manager.set_transcript_path(str(self._session_history.path))

    def _collect_hook_contexts(self, hook_result: Any) -> List[str]:
        contexts: List[str] = []
        system_message = getattr(hook_result, "system_message", None)
        additional_context = getattr(hook_result, "additional_context", None)
        if system_message:
            contexts.append(str(system_message))
        if additional_context:
            contexts.append(str(additional_context))
        return contexts

    def _set_session_hook_contexts(self, hook_result: Any) -> None:
        self._session_hook_contexts = self._collect_hook_contexts(hook_result)
        self._session_start_time = time.time()
        self._session_end_sent = False

    def _run_session_start(self, source: str) -> None:
        try:
            result = self._run_async(hook_manager.run_session_start_async(source))
        except (OSError, RuntimeError, ConnectionError, ValueError, TypeError) as exc:
            logger.warning(
                "[ui] SessionStart hook failed: %s: %s",
                type(exc).__name__,
                exc,
                extra={"session_id": self.session_id, "source": source},
            )
            return
        self._set_session_hook_contexts(result)

    async def _run_session_start_async(self, source: str) -> None:
        try:
            result = await hook_manager.run_session_start_async(source)
        except (OSError, RuntimeError, ConnectionError, ValueError, TypeError) as exc:
            logger.warning(
                "[ui] SessionStart hook failed: %s: %s",
                type(exc).__name__,
                exc,
                extra={"session_id": self.session_id, "source": source},
            )
            return
        self._set_session_hook_contexts(result)

    def _run_session_end(self, reason: str) -> None:
        if self._session_end_sent:
            return
        duration = max(time.time() - self._session_start_time, 0.0)
        message_count = len(self.conversation_messages)
        try:
            self._run_async(
                hook_manager.run_session_end_async(
                    reason, duration_seconds=duration, message_count=message_count
                )
            )
        except (OSError, RuntimeError, ConnectionError, ValueError, TypeError) as exc:
            logger.warning(
                "[ui] SessionEnd hook failed: %s: %s",
                type(exc).__name__,
                exc,
                extra={"session_id": self.session_id, "reason": reason},
            )
        self._session_end_sent = True

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
        """Get the default set of tools, filtered by allowed_tools if specified."""
        return get_default_tools(allowed_tools=self.allowed_tools)

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

    async def _prepare_query_context(
        self, user_input: str, hook_instructions: Optional[List[str]] = None
    ) -> tuple[str, Dict[str, str]]:
        """Load MCP servers, skills, and build system prompt.

        Returns:
            Tuple of (system_prompt, context_dict)
        """
        context: Dict[str, str] = {}
        servers = await load_mcp_servers_async(self.project_path)
        dynamic_tools = await load_dynamic_mcp_tools_async(self.project_path)

        if dynamic_tools and self.query_context:
            merged_tools = merge_tools_with_dynamic(self.query_context.tools, dynamic_tools)
            if self.allowed_tools is not None:
                merged_tools = filter_tools_by_names(merged_tools, self.allowed_tools)
            self.query_context.tools = merged_tools

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
        if self._session_hook_contexts:
            additional_instructions.extend(self._session_hook_contexts)
        if hook_instructions:
            additional_instructions.extend([text for text in hook_instructions if text])

        # Build system prompt based on options:
        # - custom_system_prompt: replaces the default entirely
        # - append_system_prompt: appends to the default system prompt
        if self.custom_system_prompt:
            # Complete replacement
            system_prompt = self.custom_system_prompt
            # Still append if both are provided
            if self.append_system_prompt:
                system_prompt = f"{system_prompt}\n\n{self.append_system_prompt}"
        else:
            # Build default with optional append
            all_instructions = list(additional_instructions) if additional_instructions else []
            if self.append_system_prompt:
                all_instructions.append(self.append_system_prompt)
            system_prompt = build_system_prompt(
                self.query_context.tools if self.query_context else [],
                user_input,
                context,
                additional_instructions=all_instructions or None,
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
            hook_instructions = ""
            try:
                hook_result = await hook_manager.run_pre_compact_async(
                    trigger="auto", custom_instructions=""
                )
                if hook_result.should_block or not hook_result.should_continue:
                    reason = (
                        hook_result.block_reason
                        or hook_result.stop_reason
                        or "Compaction blocked by hook."
                    )
                    console.print(f"[yellow]{escape(str(reason))}[/yellow]")
                    return messages
                hook_contexts = self._collect_hook_contexts(hook_result)
                if hook_contexts:
                    hook_instructions = "\n\n".join(hook_contexts)
            except (OSError, RuntimeError, ConnectionError, ValueError, TypeError) as exc:
                logger.warning(
                    "[ui] PreCompact hook failed: %s: %s",
                    type(exc).__name__,
                    exc,
                    extra={"session_id": self.session_id},
                )
            try:
                spinner.start()
                result = await compact_conversation(
                    messages, custom_instructions=hook_instructions, protocol=protocol
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
                await self._run_session_start_async("compact")
                return result.messages
            elif isinstance(result, CompactionError):
                logger.warning(
                    "[ui] Auto-compaction failed: %s",
                    result.message,
                    extra={"session_id": self.session_id},
                )

        return messages


    async def process_query(self, user_input: str) -> None:
        """Process a user query and display the response."""
        # Initialize or reset query context
        if not self.query_context:
            self.query_context = QueryContext(
                tools=self.get_default_tools(),
                max_thinking_tokens=self._get_thinking_tokens(),
                yolo_mode=self.yolo_mode,
                verbose=self.verbose,
                model=self.model,
            )
        else:
            abort_controller = getattr(self.query_context, "abort_controller", None)
            if abort_controller is not None:
                abort_controller.clear()
            # Update thinking tokens in case user toggled thinking mode
            self.query_context.max_thinking_tokens = self._get_thinking_tokens()
        self.query_context.stop_hook_active = False

        logger.info(
            "[ui] Starting query processing",
            extra={
                "session_id": self.session_id,
                "prompt_length": len(user_input),
                "prompt_preview": user_input[:200],
            },
        )

        try:
            hook_result = await hook_manager.run_user_prompt_submit_async(user_input)
            if hook_result.should_block or not hook_result.should_continue:
                reason = (
                    hook_result.block_reason or hook_result.stop_reason or "Prompt blocked by hook."
                )
                self.console.print(f"[red]{escape(str(reason))}[/red]")
                return
            hook_instructions = self._collect_hook_contexts(hook_result)

            # Prepare context and system prompt
            system_prompt, context = await self._prepare_query_context(
                user_input, hook_instructions
            )

            # Process images in user input
            processed_input, image_blocks = process_images_in_input(
                user_input, self.project_path, self.model
            )

            # Create and log user message
            if image_blocks:
                # Has images: use structured content
                content_blocks = []
                # Add images first
                for block in image_blocks:
                    content_blocks.append({"type": "image", **block})
                # Add user's text input
                if processed_input:
                    content_blocks.append({"type": "text", "text": processed_input})
                user_message = create_user_message(content=content_blocks)
            else:
                # No images: use plain text
                user_message = create_user_message(content=processed_input)

            messages: List[ConversationMessage] = self.conversation_messages + [user_message]
            self._log_message(user_message)
            self._append_prompt_history(processed_input)

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
                self._pause_interrupt_listener()
                try:
                    spinner.stop()
                except (RuntimeError, ValueError, OSError):
                    logger.debug("[ui] Failed to pause spinner")

            def resume_ui() -> None:
                if self._esc_interrupt_seen:
                    return
                try:
                    spinner.start()
                    spinner.update("Thinking...")
                except (RuntimeError, ValueError, OSError) as exc:
                    logger.debug(
                        "[ui] Failed to restart spinner after pause: %s: %s",
                        type(exc).__name__,
                        exc,
                    )
                finally:
                    self._resume_interrupt_listener()

            self.query_context.pause_ui = pause_ui
            self.query_context.resume_ui = resume_ui

            # Create permission checker with spinner control
            base_permission_checker = self._permission_checker

            async def permission_checker(tool: Any, parsed_input: Any) -> bool:
                pause_ui()
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
                    resume_ui()

            # Process query stream
            tool_registry: Dict[str, Dict[str, Any]] = {}
            last_tool_name: Optional[str] = None
            output_token_est = 0

            try:
                self._active_spinner = spinner
                self._esc_interrupt_seen = False
                self._query_in_progress = True
                self._start_interrupt_listener()
                spinner.start()
                async for message in query(
                    messages,
                    system_prompt,
                    context,
                    self.query_context,
                    permission_checker,  # type: ignore[arg-type]
                ):
                    if message.type == "assistant" and isinstance(message, AssistantMessage):
                        result = handle_assistant_message(self, message, tool_registry, spinner)
                        if result:
                            last_tool_name = result

                    elif message.type == "user" and isinstance(message, UserMessage):
                        handle_tool_result_message(self, message, tool_registry, last_tool_name, spinner)

                    elif message.type == "progress" and isinstance(message, ProgressMessage):
                        output_token_est = handle_progress_message(
                            self, message, spinner, output_token_est
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

                self._stop_interrupt_listener()
                self._query_in_progress = False
                self._active_spinner = None
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

    def _schedule_esc_interrupt(self) -> None:
        """Schedule ESC interrupt handling on the UI event loop."""
        if self._loop.is_closed():
            return
        try:
            self._loop.call_soon_threadsafe(self._handle_esc_interrupt)
        except RuntimeError:
            pass

    def _handle_esc_interrupt(self) -> None:
        """Abort the current query and display the interrupt notice."""
        if not self._query_in_progress:
            return
        if self._esc_interrupt_seen:
            return
        abort_controller = getattr(self.query_context, "abort_controller", None)
        if abort_controller is None or abort_controller.is_set():
            return

        self._esc_interrupt_seen = True

        try:
            if self.query_context and self.query_context.pause_ui:
                self.query_context.pause_ui()
            elif self._active_spinner:
                self._active_spinner.stop()
        except (RuntimeError, ValueError, OSError):
            logger.debug("[ui] Failed to pause spinner for ESC interrupt")

        self._message_display.print_interrupt_notice()
        abort_controller.set()

    def _start_interrupt_listener(self) -> None:
        self._interrupt_listener.start()

    def _stop_interrupt_listener(self) -> None:
        self._interrupt_listener.stop()

    def _pause_interrupt_listener(self) -> None:
        self._interrupt_listener.pause()

    def _resume_interrupt_listener(self) -> None:
        self._interrupt_listener.resume()

    def _run_async(self, coro: Any) -> Any:
        """Run a coroutine on the persistent event loop."""
        if self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        return self._loop.run_until_complete(coro)

    def run_async(self, coro: Any) -> Any:
        """Public wrapper for running coroutines on the UI event loop."""
        return self._run_async(coro)

    def handle_slash_command(self, user_input: str) -> bool | str:
        """Handle slash commands.

        Returns True if handled as built-in, False if not a command,
        or a string if it's a custom command that should be sent to the AI.
        """
        from ripperdoc.cli.ui.rich_ui import _suggest_slash_commands

        return _handle_slash_command(self, user_input, _suggest_slash_commands)

    def get_prompt_session(self) -> PromptSession:
        """Create (or return) the prompt session with command completion."""
        if self._prompt_session:
            return self._prompt_session
        self._prompt_session = build_prompt_session(self, self._ignore_filter)
        return self._prompt_session

    def run(self) -> None:
        """Run the Rich-based interface."""
        # Display welcome panel
        console.print()
        console.print(create_welcome_panel())
        console.print()

        # Display random tip
        random_tip = get_random_tip()
        console.print(f"[dim italic]💡 {random_tip}[/dim italic]\n")

        session = self.get_prompt_session()
        logger.info(
            "[ui] Starting interactive loop",
            extra={"session_id": self.session_id, "log_file": str(self.log_file_path)},
        )

        exit_reason = "other"
        try:
            # Process initial query from piped stdin if provided
            if self._initial_query:
                console.print(f"> {self._initial_query}")
                logger.info(
                    "[ui] Processing initial query from stdin",
                    extra={
                        "session_id": self.session_id,
                        "prompt_length": len(self._initial_query),
                        "prompt_preview": self._initial_query[:200],
                    },
                )
                console.print()  # Add spacing before response

                # Process initial query (ESC interrupt handling removed)
                self._run_async(self.process_query(self._initial_query))

                logger.info(
                    "[ui] Initial query completed successfully",
                    extra={"session_id": self.session_id},
                )
                console.print()  # Add spacing after response
                self._initial_query = None  # Clear after processing

            while not self._should_exit:
                try:
                    # Get user input with dynamic prompt
                    user_input = session.prompt(self._get_prompt())

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
                            exit_reason = self._exit_reason or "other"
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

                    # Run query (ESC interrupt handling removed)
                    self._run_async(self.process_query(user_input))

                    console.print()  # Add spacing between interactions

                except KeyboardInterrupt:
                    # Handle Ctrl+C: first press during query aborts it,
                    # double press exits the CLI
                    current_time = time.time()

                    # Signal abort to cancel running queries
                    if self.query_context:
                        abort_controller = getattr(self.query_context, "abort_controller", None)
                        if abort_controller is not None:
                            abort_controller.set()

                    # Check if this is a double Ctrl+C (within 1.5 seconds)
                    if current_time - self._last_ctrl_c_time < 1.5:
                        console.print("\n[yellow]Goodbye![/yellow]")
                        exit_reason = "prompt_input_exit"
                        break

                    # First Ctrl+C - just abort the query and continue
                    self._last_ctrl_c_time = current_time
                    console.print("\n[dim]Query interrupted. Press Ctrl+C again to exit.[/dim]")
                    continue
                except EOFError:
                    console.print("\n[yellow]Goodbye![/yellow]")
                    exit_reason = "prompt_input_exit"
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

            self._run_session_end(exit_reason)

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
                    self._run_async(shutdown_lsp_manager())
                except (OSError, RuntimeError, ConnectionError, asyncio.CancelledError) as exc:
                    # pragma: no cover - defensive shutdown
                    logger.warning(
                        "[ui] Failed to shut down MCP runtime cleanly: %s: %s",
                        type(exc).__name__,
                        exc,
                        extra={"session_id": self.session_id},
                    )

                # Shutdown background shell manager to clean up any background tasks
                try:
                    shutdown_background_shell(force=True)
                except (OSError, RuntimeError) as exc:
                    logger.debug(
                        "[ui] Failed to shut down background shell cleanly: %s: %s",
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

        hook_instructions = ""
        try:
            hook_result = await hook_manager.run_pre_compact_async(
                trigger="manual", custom_instructions=custom_instructions
            )
            if hook_result.should_block or not hook_result.should_continue:
                reason = (
                    hook_result.block_reason
                    or hook_result.stop_reason
                    or "Compaction blocked by hook."
                )
                self.console.print(f"[yellow]{escape(str(reason))}[/yellow]")
                return
            hook_contexts = self._collect_hook_contexts(hook_result)
            if hook_contexts:
                hook_instructions = "\n\n".join(hook_contexts)
        except (OSError, RuntimeError, ConnectionError, ValueError, TypeError) as exc:
            logger.warning(
                "[ui] PreCompact hook failed: %s: %s",
                type(exc).__name__,
                exc,
                extra={"session_id": self.session_id},
            )

        merged_instructions = custom_instructions.strip()
        if hook_instructions:
            merged_instructions = (
                f"{merged_instructions}\n\n{hook_instructions}".strip()
                if merged_instructions
                else hook_instructions
            )
        try:
            spinner.start()
            result = await compact_conversation(
                self.conversation_messages,
                merged_instructions,
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
            await self._run_session_start_async("compact")
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
    allowed_tools: Optional[List[str]] = None,
    custom_system_prompt: Optional[str] = None,
    append_system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    resume_messages: Optional[List[Any]] = None,
    initial_query: Optional[str] = None,
) -> None:
    """Main entry point for Rich interface.

    Args:
        initial_query: If provided, automatically send this query after starting the session.
                      Used for piped stdin input (e.g., `echo "query" | ripperdoc`).
    """

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
        allowed_tools=allowed_tools,
        custom_system_prompt=custom_system_prompt,
        append_system_prompt=append_system_prompt,
        model=model,
        resume_messages=resume_messages,
        initial_query=initial_query,
    )
    ui.run()


if __name__ == "__main__":
    main_rich()
