"""Rich-based CLI interface for Ripperdoc.

This module provides a clean, minimal terminal UI using Rich for the Ripperdoc agent.
"""

import asyncio
import concurrent.futures
import html
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import FormattedText
from rich.console import Console
from rich.markup import escape

from ripperdoc.core.config import (
    get_effective_config,
    get_effective_model_profile,
    get_project_local_config,
    provider_protocol,
)
from ripperdoc.core.tool_defaults import filter_tools_by_names, get_default_tools
from ripperdoc.core.theme import get_theme_manager
from ripperdoc.core.query import query, QueryContext
from ripperdoc.core.hooks.state import bind_pending_message_queue
from ripperdoc.core.system_prompt import build_system_prompt
from ripperdoc.core.skills import build_skill_summary, filter_enabled_skills, load_all_skills
from ripperdoc.core.hooks.manager import hook_manager
from ripperdoc.core.hooks.llm_callback import build_hook_llm_callback
from ripperdoc.cli.commands import list_custom_commands, list_slash_commands
from ripperdoc.cli.ui.helpers import get_profile_for_pointer
from ripperdoc.core.permission_engine import make_permission_checker
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
    McpRuntime,
    ensure_mcp_runtime,
    format_mcp_instructions,
    load_mcp_servers_async,
    shutdown_mcp_runtime,
)
from ripperdoc.utils.lsp import shutdown_lsp_manager
from ripperdoc.tools.background_shell import shutdown_background_shell
from ripperdoc.tools.dynamic_mcp_tool import (
    load_dynamic_mcp_tools_async,
    merge_tools_with_dynamic,
)
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
from ripperdoc.utils.tasks import set_runtime_task_scope
from ripperdoc.utils.session_usage import rebuild_session_usage
from ripperdoc.utils.working_directories import normalize_directory_inputs

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
_RESUME_REPLAY_LIMIT_ENV = "RIPPERDOC_RESUME_REPLAY_MAX_MESSAGES"
_DEFAULT_RESUME_REPLAY_LIMIT = 120
_PERMISSION_MODE_LABELS: dict[str, str] = {
    "default": "Mode: normal",
    "acceptEdits": "⏵⏵ accept edits on",
    "plan": "⏸ plan mode on",
    "dontAsk": "⏸ don't ask mode on",
    "bypassPermissions": "⏵⏵ bypass permissions on",
}


def _resolve_resume_replay_limit() -> Optional[int]:
    """Resolve how many messages to replay when resuming history.

    Environment variable:
    - RIPPERDOC_RESUME_REPLAY_MAX_MESSAGES=-1 : replay all
    - RIPPERDOC_RESUME_REPLAY_MAX_MESSAGES=0  : skip replay
    - RIPPERDOC_RESUME_REPLAY_MAX_MESSAGES=N  : replay last N messages
    """
    raw = os.getenv(_RESUME_REPLAY_LIMIT_ENV)
    if raw is None or not raw.strip():
        return _DEFAULT_RESUME_REPLAY_LIMIT
    try:
        limit = int(raw.strip())
    except ValueError:
        logger.warning(
            "[ui] Invalid %s=%r; falling back to default %d",
            _RESUME_REPLAY_LIMIT_ENV,
            raw,
            _DEFAULT_RESUME_REPLAY_LIMIT,
        )
        return _DEFAULT_RESUME_REPLAY_LIMIT

    if limit == -1:
        return None
    if limit < -1:
        logger.warning(
            "[ui] Invalid %s=%d; falling back to default %d",
            _RESUME_REPLAY_LIMIT_ENV,
            limit,
            _DEFAULT_RESUME_REPLAY_LIMIT,
        )
        return _DEFAULT_RESUME_REPLAY_LIMIT
    return limit


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
        additional_working_dirs: Optional[List[str]] = None,
        max_thinking_tokens: Optional[int] = None,
        max_turns: Optional[int] = None,
        permission_mode: str = "default",
    ):
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._run_event_loop,
            name=f"ripperdoc-ui-loop-{uuid.uuid4().hex[:8]}",
            daemon=True,
        )
        self._loop_thread.start()
        self._mcp_warmup_future: Optional[concurrent.futures.Future[McpRuntime]] = None
        self.console = console
        self.yolo_mode = yolo_mode
        self.verbose = verbose
        self.allowed_tools = allowed_tools
        self.custom_system_prompt = custom_system_prompt
        self.append_system_prompt = append_system_prompt
        self.model = model or "main"
        self.conversation_messages: List[ConversationMessage] = []
        self._saved_conversation: Optional[List[ConversationMessage]] = None
        self._resumed_from_history = bool(resume_messages)
        self._resume_replay_max_messages = _resolve_resume_replay_limit()
        self.query_context: Optional[QueryContext] = None
        self._current_tool: Optional[str] = None
        self._should_exit: bool = False
        self._last_ctrl_c_time: float = 0.0  # Track Ctrl+C timing for double-press exit
        self._initial_query = initial_query  # Query from piped stdin to auto-send on startup
        self.command_list = list_slash_commands()
        self._custom_command_list = list_custom_commands()
        self._prompt_session: Optional[PromptSession] = None
        self.project_path = Path.cwd()
        project_local_config = get_project_local_config(self.project_path)
        self.output_style = getattr(project_local_config, "output_style", "default") or "default"
        self.output_language = getattr(project_local_config, "output_language", "auto") or "auto"
        # Track a stable session identifier for the current UI run.
        self.session_id = session_id or str(uuid.uuid4())
        set_runtime_task_scope(session_id=self.session_id, project_root=self.project_path)
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
        self._permission_checker: Any = None
        self.max_turns = max_turns
        self.permission_mode = permission_mode
        self._pre_plan_mode: Optional[str] = None
        self._max_thinking_tokens_override = (
            max(0, int(max_thinking_tokens))
            if isinstance(max_thinking_tokens, int)
            else None
        )
        normalized_working_dirs, working_dir_errors = normalize_directory_inputs(
            additional_working_dirs or [],
            base_dir=self.project_path,
            require_exists=True,
        )
        if working_dir_errors:
            logger.warning(
                "[ui] Ignoring invalid additional directories",
                extra={"errors": working_dir_errors, "session_id": self.session_id},
            )
        self._session_additional_working_dirs: set[str] = set(normalized_working_dirs)
        hook_manager.set_transcript_path(str(self._session_history.path))

        # Create permission checker with Rich console and PromptSession support
        self._permission_checker = (
            None if yolo_mode else self._create_permission_checker()
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
        config = get_effective_config(self.project_path)
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

        # Start MCP warmup in background so UI becomes interactive immediately.
        self._start_mcp_runtime_warmup()

        # Initialize hook manager with project context
        hook_manager.set_project_dir(self.project_path)
        hook_manager.set_session_id(self.session_id)
        hook_manager.set_permission_mode(self.permission_mode)
        hook_manager.set_llm_callback(build_hook_llm_callback())
        logger.debug(
            "[ui] Initialized hook manager",
            extra={
                "session_id": self.session_id,
                "project_path": str(self.project_path),
                "permission_mode": self.permission_mode,
            },
        )
        self._run_session_start("startup")

        # Handle resume_messages if provided (for --continue)
        if resume_messages:
            self.conversation_messages = list(resume_messages)
            self._rebuild_session_usage_from_messages(self.conversation_messages)
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

    def _create_permission_checker(self) -> Any:
        """Create a permission checker using current session directory scope."""
        return make_permission_checker(
            self.project_path,
            yolo_mode=False,
            permission_mode=self.permission_mode,
            session_additional_working_dirs=self._session_additional_working_dirs,
        )

    def _normalize_permission_mode(self, mode: str) -> str:
        normalized = str(mode or "").strip()
        if normalized in {"default", "acceptEdits", "plan", "bypassPermissions", "dontAsk"}:
            return normalized
        return "default"

    def _is_bypass_permissions_mode_available(self) -> bool:
        """Return whether bypassPermissions mode can be cycled to."""
        return True

    def _permission_mode_label(self, mode: Optional[str] = None) -> str:
        normalized = self._normalize_permission_mode(mode or self.permission_mode)
        return _PERMISSION_MODE_LABELS.get(normalized, _PERMISSION_MODE_LABELS["default"])

    def _next_permission_mode(self) -> str:
        current = self._normalize_permission_mode(self.permission_mode)
        if current == "default":
            return "acceptEdits"
        if current == "acceptEdits":
            return "plan"
        if current == "plan":
            if self._is_bypass_permissions_mode_available():
                return "bypassPermissions"
            return "default"
        if current in {"bypassPermissions", "dontAsk"}:
            return "default"
        return "default"

    def _apply_permission_mode(self, mode: str, *, announce: bool = True) -> str:
        """Apply a new permission mode across UI/query/hook state."""
        previous_mode = self._normalize_permission_mode(self.permission_mode)
        normalized = self._normalize_permission_mode(mode)
        if previous_mode == "plan" and normalized != "plan":
            self._pre_plan_mode = None
        self.permission_mode = normalized
        self.yolo_mode = normalized == "bypassPermissions"
        hook_manager.set_permission_mode(normalized)

        if self.query_context is not None:
            self.query_context.permission_mode = normalized
            self.query_context.yolo_mode = self.yolo_mode
            self.query_context.pre_plan_mode = self._pre_plan_mode

        if self.yolo_mode:
            self._permission_checker = None
        else:
            self._permission_checker = self._create_permission_checker()

        label = self._permission_mode_label(normalized)
        if announce:
            self.console.print(f"[cyan]{label}[/cyan]")
        return normalized

    def _cycle_permission_mode(self) -> None:
        """Cycle permission mode using Claude-compatible ordering."""
        next_mode = self._next_permission_mode()
        self._apply_permission_mode(next_mode, announce=False)

    def _enter_plan_mode(self) -> None:
        """Switch to plan mode and remember the prior mode for restoration."""
        current_mode = self._normalize_permission_mode(self.permission_mode)
        if current_mode != "plan":
            self._pre_plan_mode = current_mode
        self._apply_permission_mode("plan", announce=False)

    def _exit_plan_mode(self) -> None:
        """Exit plan mode and restore the pre-plan mode (or default)."""
        target = self._normalize_permission_mode(self._pre_plan_mode or "default")
        if target == "plan":
            target = "default"
        self._pre_plan_mode = None
        self._apply_permission_mode(target, announce=False)

    def _supports_thinking_mode(self) -> bool:
        """Check if the current model supports extended thinking mode."""
        from ripperdoc.core.query import infer_thinking_mode
        from ripperdoc.core.config import ProtocolType

        model_profile = get_profile_for_pointer(self.model)
        if not model_profile:
            return False
        # Anthropic natively supports thinking mode
        if model_profile.protocol == ProtocolType.ANTHROPIC:
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
        if self._max_thinking_tokens_override is not None:
            return self._max_thinking_tokens_override
        if not self._thinking_mode_enabled:
            return 0
        config = get_effective_config(self.project_path)
        return config.default_thinking_tokens

    def _get_prompt(self) -> str:
        """Generate the input prompt."""
        return "> "

    def _get_rprompt(self) -> Union[str, FormattedText]:
        """Generate the right prompt with permission/thinking mode status."""
        mode = self._normalize_permission_mode(self.permission_mode)
        mode_style = {
            "default": "class:rprompt-mode-normal",
            "acceptEdits": "class:rprompt-mode-accept",
            "plan": "class:rprompt-mode-plan",
            "dontAsk": "class:rprompt-mode-plan",
            "bypassPermissions": "class:rprompt-mode-bypass",
        }.get(mode, "class:rprompt-mode-normal")

        fragments: list[tuple[str, str]] = [(mode_style, self._permission_mode_label(mode))]
        if self._supports_thinking_mode():
            fragments.append(("class:rprompt-sep", " | "))
            if self._thinking_mode_enabled:
                fragments.append(("class:rprompt-on", "⚡ Thinking"))
            else:
                fragments.append(("class:rprompt-off", "Thinking: off"))
        return FormattedText(fragments)

    def _context_usage_lines(
        self, breakdown: Any, model_label: str, auto_compact_enabled: bool
    ) -> List[str]:
        return context_usage_lines(breakdown, model_label, auto_compact_enabled)

    def _set_session(self, session_id: str) -> None:
        """Switch to a different session id and reset logging."""
        self.session_id = session_id
        set_runtime_task_scope(session_id=self.session_id, project_root=self.project_path)
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

    def _rebuild_session_usage_from_messages(
        self, messages: Optional[List[ConversationMessage]] = None
    ) -> None:
        """Rebuild /cost counters from persisted assistant messages."""
        rebuild_session_usage(messages if messages is not None else self.conversation_messages)

    def list_additional_working_directories(self) -> list[str]:
        """Return current session-scoped additional working directories."""
        return sorted(self._session_additional_working_dirs)

    def get_output_style(self) -> str:
        """Return the active output style key for this session."""
        return self.output_style

    def set_output_style(self, style_key: str) -> None:
        """Update active output style for subsequent turns."""
        self.output_style = style_key or "default"

    def get_output_language(self) -> str:
        """Return the active output language for this session."""
        return self.output_language

    def set_output_language(self, language: str) -> None:
        """Update active output language for subsequent turns."""
        self.output_language = language or "auto"

    def add_additional_working_directory(self, raw_path: str) -> tuple[bool, str]:
        """Add a working directory for this session.

        Returns:
            tuple[bool, str]: (added, message)
        """
        normalized, errors = normalize_directory_inputs(
            [raw_path],
            base_dir=self.project_path,
            require_exists=True,
        )
        if errors:
            return False, errors[0]
        if not normalized:
            return False, "No directory path provided."

        resolved = normalized[0]
        if resolved in self._session_additional_working_dirs:
            return False, f"Already added: {resolved}"

        self._session_additional_working_dirs.add(resolved)
        adder = getattr(self._permission_checker, "add_working_directory", None)
        if callable(adder):
            adder(resolved)
        return True, f"Added: {resolved}"

    def _collect_hook_contexts(self, hook_result: Any) -> List[str]:
        contexts: List[str] = []
        additional_context = getattr(hook_result, "additional_context", None)
        if additional_context:
            contexts.append(str(additional_context))
        return contexts

    def _display_hook_system_message(
        self, hook_result: Any, event: str, tool_name: Optional[str] = None
    ) -> None:
        system_message = getattr(hook_result, "system_message", None)
        if not system_message:
            return
        label = f"{event}:{tool_name}" if tool_name else event
        self.console.print(
            f"[yellow]Hook {escape(str(label))}[/yellow] {escape(str(system_message))}"
        )

    def _set_session_hook_contexts(self, hook_result: Any) -> None:
        self._session_hook_contexts = self._collect_hook_contexts(hook_result)
        self._session_start_time = time.time()
        self._session_end_sent = False

    def _run_session_start(self, source: str) -> None:
        try:
            queue = (
                self.query_context.pending_message_queue
                if self.query_context is not None
                else None
            )
            with bind_pending_message_queue(queue):
                result = self._run_async(hook_manager.run_session_start_async(source))
        except (OSError, RuntimeError, ConnectionError, ValueError, TypeError) as exc:
            logger.warning(
                "[ui] SessionStart hook failed: %s: %s",
                type(exc).__name__,
                exc,
                extra={"session_id": self.session_id, "source": source},
            )
            return
        self._display_hook_system_message(result, "SessionStart")
        self._set_session_hook_contexts(result)

    async def _run_session_start_async(self, source: str) -> None:
        try:
            queue = (
                self.query_context.pending_message_queue
                if self.query_context is not None
                else None
            )
            with bind_pending_message_queue(queue):
                result = await hook_manager.run_session_start_async(source)
        except (OSError, RuntimeError, ConnectionError, ValueError, TypeError) as exc:
            logger.warning(
                "[ui] SessionStart hook failed: %s: %s",
                type(exc).__name__,
                exc,
                extra={"session_id": self.session_id, "source": source},
            )
            return
        self._display_hook_system_message(result, "SessionStart")
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

    def _seed_session_history(self, messages: List[ConversationMessage]) -> None:
        """Persist the provided messages into the current session log."""
        for msg in messages:
            self._log_message(msg)

    def _fork_session(self) -> tuple[str, str]:
        """Fork the current session into a new session id and seed history."""
        old_session_id = self.session_id
        new_session_id = str(uuid.uuid4())
        self._set_session(new_session_id)
        self._saved_conversation = None
        self._seed_session_history(self.conversation_messages)
        try:
            self._run_session_start("resume")
        except (AttributeError, RuntimeError, ValueError):
            pass
        return old_session_id, new_session_id

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

    def _extract_visible_text(self, content: Any) -> str:
        """Extract user-visible text from a message content payload."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for block in content:
                block_type = getattr(block, "type", None)
                if block_type is None and isinstance(block, dict):
                    block_type = block.get("type")
                if block_type == "text":
                    text_val = getattr(block, "text", None)
                    if text_val is None and isinstance(block, dict):
                        text_val = block.get("text")
                    if text_val:
                        parts.append(str(text_val))
                elif block_type == "image":
                    parts.append("[image]")
            return "\n".join(part for part in parts if part)
        return ""

    def _message_has_block(self, content: Any, block_type_name: str) -> bool:
        if not isinstance(content, list):
            return False
        for block in content:
            block_type = getattr(block, "type", None)
            if block_type is None and isinstance(block, dict):
                block_type = block.get("type")
            if block_type == block_type_name:
                return True
        return False

    def _is_tool_result_only(self, content: Any) -> bool:
        if not isinstance(content, list):
            return False
        has_tool_result = False
        has_visible_text = False
        for block in content:
            block_type = getattr(block, "type", None)
            if block_type is None and isinstance(block, dict):
                block_type = block.get("type")
            if block_type == "tool_result":
                has_tool_result = True
                continue
            if block_type == "text":
                text_val = getattr(block, "text", None)
                if text_val is None and isinstance(block, dict):
                    text_val = block.get("text")
                if text_val and str(text_val).strip():
                    has_visible_text = True
            elif block_type == "image":
                has_visible_text = True
        return has_tool_result and not has_visible_text

    def _format_history_preview(self, text: str, max_len: int = 80) -> str:
        preview = " ".join(text.strip().split())
        if not preview:
            return "(empty)"
        if len(preview) > max_len:
            preview = preview[: max_len - 3].rstrip() + "..."
        return preview

    def _build_history_candidates(self) -> List[dict[str, Any]]:
        candidates: List[dict[str, Any]] = []
        for idx, msg in enumerate(self.conversation_messages):
            msg_type = getattr(msg, "type", "")
            if msg_type != "user":
                continue
            message_payload = getattr(msg, "message", None)
            content = getattr(message_payload, "content", None) if message_payload else None
            if content is None:
                content = getattr(msg, "content", None)
            visible_text = self._extract_visible_text(content)
            if not visible_text.strip():
                continue
            if msg_type == "user" and self._is_tool_result_only(content):
                continue
            preview = self._format_history_preview(visible_text)
            candidates.append(
                {
                    "index": idx,
                    "uuid": getattr(msg, "uuid", None),
                    "preview": preview,
                }
            )
        return candidates

    def _is_real_user_message(self, msg: Any) -> bool:
        msg_type = getattr(msg, "type", "")
        if msg_type != "user":
            return False
        message_payload = getattr(msg, "message", None)
        content = getattr(message_payload, "content", None) if message_payload else None
        if content is None:
            content = getattr(msg, "content", None)
        if not content:
            return False
        if self._is_tool_result_only(content):
            return False
        return bool(self._extract_visible_text(content).strip())

    def _resolve_turn_end_index(self, selected_index: int) -> int:
        if selected_index < 0 or selected_index >= len(self.conversation_messages):
            return selected_index
        # Include messages until right before the next real user message.
        for idx in range(selected_index + 1, len(self.conversation_messages)):
            if self._is_real_user_message(self.conversation_messages[idx]):
                return idx - 1
        return len(self.conversation_messages) - 1

    def _rollback_to_index(self, target_index: int) -> None:
        if target_index < 0 or target_index >= len(self.conversation_messages):
            self.console.print("[red]Invalid history selection.[/red]")
            return
        msg = self.conversation_messages[target_index]
        if self._is_real_user_message(msg):
            target_index = self._resolve_turn_end_index(target_index)
        if target_index == len(self.conversation_messages) - 1:
            self.console.print("[dim]Already at the selected message.[/dim]")
            return
        msg_type = getattr(msg, "type", "")
        role = "You" if msg_type == "user" else "Ripperdoc"
        original_len = len(self.conversation_messages)
        self.conversation_messages = list(self.conversation_messages[: target_index + 1])

        # Fork a new session so the original transcript remains intact.
        old_session_id, new_session_id = self._fork_session()

        try:
            self.console.clear()
        except (OSError, RuntimeError, ValueError):
            pass
        self._rebuild_session_usage_from_messages(self.conversation_messages)
        self.replay_conversation(self.conversation_messages)
        self.console.print(
            f"[green]✓ Rolled back to {role} message #{target_index + 1} "
            f"({original_len} -> {len(self.conversation_messages)} messages).[/green]"
        )
        self.console.print(
            f"[dim]Forked new session {new_session_id[:8]}... "
            f"(previous {old_session_id[:8]}... preserved).[/dim]"
        )
        logger.info(
            "[ui] Rolled back conversation",
            extra={
                "session_id": self.session_id,
                "target_index": target_index,
                "original_len": original_len,
                "new_len": len(self.conversation_messages),
                "previous_session_id": old_session_id,
            },
        )

    async def _open_history_picker_async(self) -> bool:
        from ripperdoc.cli.ui.choice import ChoiceOption, prompt_choice_async, theme_style

        if self._query_in_progress:
            self.console.print("[yellow]Cannot open history while a query is running.[/yellow]")
            return False
        candidates = self._build_history_candidates()
        if not candidates:
            self.console.print("[yellow]No message history available yet.[/yellow]")
            return False

        options: List[ChoiceOption] = []
        for candidate in candidates:
            idx = candidate["index"]
            preview = candidate["preview"]
            label = (
                f"<info>#{idx + 1}</info> You "
                f"<dim>{html.escape(preview)}</dim>"
            )
            value = candidate.get("uuid") or str(idx)
            options.append(ChoiceOption(str(value), label))

        try:
            result = await prompt_choice_async(
                message="<question>Select a message to roll back to</question>",
                options=options,
                title="History",
                allow_esc=True,
                esc_value="__cancel__",
                style=theme_style(),
            )
        except (EOFError, KeyboardInterrupt):
            return False

        if result == "__cancel__":
            return False

        target_index = None
        for candidate in candidates:
            value = candidate.get("uuid") or str(candidate["index"])
            if str(value) == str(result):
                target_index = candidate["index"]
                break
        if target_index is None:
            return False
        self._rollback_to_index(target_index)
        return True

    def replay_conversation(
        self, messages: List[ConversationMessage], *, max_messages: Optional[int] = None
    ) -> None:
        """Render a conversation history in the console and seed prompt history."""
        if not messages:
            return
        replay_messages = list(messages)
        total_messages = len(replay_messages)

        if isinstance(max_messages, int):
            if max_messages == 0:
                self.console.print(
                    f"\n[dim]Restored session with {total_messages} messages "
                    "(history replay skipped).[/dim]"
                )
                return
            if max_messages > 0 and total_messages > max_messages:
                replay_messages = replay_messages[-max_messages:]
                shown = len(replay_messages)
                skipped = total_messages - shown
                self.console.print(
                    f"\n[dim]Restored recent conversation ({shown}/{total_messages} messages; "
                    f"skipped {skipped}).[/dim]"
                )
            else:
                self.console.print("\n[dim]Restored conversation:[/dim]")
        else:
            self.console.print("\n[dim]Restored conversation:[/dim]")
        tool_registry: Dict[str, Dict[str, Any]] = {}
        last_tool_name: Optional[str] = None

        for msg in replay_messages:
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

            if msg_type == "assistant" and isinstance(msg, AssistantMessage):
                last = handle_assistant_message(self, msg, tool_registry, spinner=None)
                if last:
                    last_tool_name = last
                continue

            if msg_type == "user" and has_tool_result and isinstance(msg, UserMessage):
                handle_tool_result_message(
                    self, msg, tool_registry, last_tool_name, spinner=None
                )
                continue

            text = self._stringify_message_content(content)
            if not text:
                continue
            if msg_type == "user":
                self._print_replay_user(text)
                self._append_prompt_history(text)

    def _print_replay_user(self, text: str) -> None:
        """Render restored user messages with a prompt-style prefix."""
        lines = text.splitlines() or [text]
        for i, line in enumerate(lines):
            if i == 0:
                self.console.print(f"> {line}", markup=False)
            else:
                self.console.print(line, markup=False)

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
        servers, dynamic_tools = await asyncio.gather(
            load_mcp_servers_async(self.project_path),
            load_dynamic_mcp_tools_async(self.project_path),
        )

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

        enabled_skills = filter_enabled_skills(skill_result.skills, project_path=self.project_path)
        skill_instructions = build_skill_summary(enabled_skills)
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
                output_style=self.output_style,
                output_language=self.output_language,
                project_path=self.project_path,
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
                    console.print(
                        f"[yellow]PreCompact hook warning (ignored): {escape(str(reason))}[/yellow]"
                    )
                self._display_hook_system_message(hook_result, "PreCompact")
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

    def _ensure_query_context(self) -> QueryContext:
        """Initialize or refresh query context for a new user turn."""
        if not self.query_context:
            self.query_context = QueryContext(
                tools=self.get_default_tools(),
                max_thinking_tokens=self._get_thinking_tokens(),
                yolo_mode=self.yolo_mode,
                verbose=self.verbose,
                model=self.model,
                max_turns=self.max_turns,
                permission_mode=self.permission_mode,
                pre_plan_mode=self._pre_plan_mode,
                on_enter_plan_mode=self._enter_plan_mode,
                on_exit_plan_mode=self._exit_plan_mode,
            )
        else:
            abort_controller = getattr(self.query_context, "abort_controller", None)
            if abort_controller is not None:
                abort_controller.clear()
            self.query_context.max_thinking_tokens = self._get_thinking_tokens()
            self.query_context.max_turns = self.max_turns
            self.query_context.yolo_mode = self.yolo_mode
            self.query_context.permission_mode = self.permission_mode
            self.query_context.pre_plan_mode = self._pre_plan_mode
            self.query_context.on_enter_plan_mode = self._enter_plan_mode
            self.query_context.on_exit_plan_mode = self._exit_plan_mode
        self.query_context.stop_hook_active = False
        return self.query_context

    def _build_user_message_from_input(self, user_input: str) -> tuple[str, UserMessage]:
        """Convert raw user input (including images) into a conversation message."""
        processed_input, image_blocks = process_images_in_input(
            user_input, self.project_path, self.model
        )
        if image_blocks:
            content_blocks: list[dict[str, Any]] = []
            for block in image_blocks:
                content_blocks.append({"type": "image", **block})
            if processed_input:
                content_blocks.append({"type": "text", "text": processed_input})
            return processed_input, create_user_message(content=content_blocks)
        return processed_input, create_user_message(content=processed_input)

    def _resolve_query_runtime_settings(self) -> tuple[int, bool, str]:
        """Resolve context budget and protocol settings for this query."""
        config = get_effective_config(self.project_path)
        model_profile = get_profile_for_pointer(self.model)
        max_context_tokens = get_remaining_context_tokens(
            model_profile, config.context_token_limit
        )
        auto_compact_enabled = resolve_auto_compact_enabled(config)
        protocol = provider_protocol(model_profile.protocol) if model_profile else "openai"
        return max_context_tokens, auto_compact_enabled, protocol

    def _build_spinner_callbacks(
        self, spinner: ThinkingSpinner
    ) -> tuple[Callable[[], None], Callable[[], None]]:
        """Build pause/resume callbacks shared by permission prompts and interrupts."""

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

        return pause_ui, resume_ui

    def _build_permission_checker(
        self,
        pause_ui: Callable[[], None],
        resume_ui: Callable[[], None],
    ) -> Callable[[Any, Any], Awaitable[bool]]:
        """Wrap base permission checker with UI pause/resume behavior."""
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

        return permission_checker

    async def _stream_query_messages(
        self,
        *,
        messages: List[ConversationMessage],
        system_prompt: str,
        context: Dict[str, str],
        permission_checker: Callable[[Any, Any], Awaitable[bool]],
        spinner: ThinkingSpinner,
        query_context: QueryContext,
    ) -> None:
        """Stream query output, render messages, and persist conversation state."""
        tool_registry: Dict[str, Dict[str, Any]] = {}
        last_tool_name: Optional[str] = None
        output_token_est = 0

        self._active_spinner = spinner
        self._esc_interrupt_seen = False
        self._query_in_progress = True
        self._start_interrupt_listener()
        spinner.start()
        async for message in query(
            messages,
            system_prompt,
            context,
            query_context,
            permission_checker,  # type: ignore[arg-type]
        ):
            if message.type == "assistant" and isinstance(message, AssistantMessage):
                maybe_tool_name = handle_assistant_message(self, message, tool_registry, spinner)
                if maybe_tool_name:
                    last_tool_name = maybe_tool_name
            elif message.type == "user" and isinstance(message, UserMessage):
                handle_tool_result_message(self, message, tool_registry, last_tool_name, spinner)
            elif message.type == "progress" and isinstance(message, ProgressMessage):
                output_token_est = handle_progress_message(
                    self, message, spinner, output_token_est
                )

            self._log_message(message)
            messages.append(message)  # type: ignore[arg-type]

    def _finalize_query_stream(
        self,
        spinner: ThinkingSpinner,
        messages: List[ConversationMessage],
    ) -> None:
        """Best-effort stream cleanup and conversation state commit."""
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


    async def process_query(self, user_input: str) -> None:
        """Process a user query and display the response."""
        query_context = self._ensure_query_context()

        logger.info(
            "[ui] Starting query processing",
            extra={
                "session_id": self.session_id,
                "prompt_length": len(user_input),
                "prompt_preview": user_input[:200],
            },
        )

        try:
            queue = query_context.pending_message_queue
            with bind_pending_message_queue(queue):
                hook_result = await hook_manager.run_user_prompt_submit_async(user_input)
            if hook_result.should_block or not hook_result.should_continue:
                reason = (
                    hook_result.block_reason or hook_result.stop_reason or "Prompt blocked by hook."
                )
                self.console.print(f"[red]{escape(str(reason))}[/red]")
                return
            self._display_hook_system_message(hook_result, "UserPromptSubmit")
            hook_instructions = self._collect_hook_contexts(hook_result)

            system_prompt, context = await self._prepare_query_context(
                user_input, hook_instructions
            )
            processed_input, user_message = self._build_user_message_from_input(user_input)

            messages: List[ConversationMessage] = self.conversation_messages + [user_message]
            self._log_message(user_message)
            self._append_prompt_history(processed_input)

            max_context_tokens, auto_compact_enabled, protocol = self._resolve_query_runtime_settings()
            messages = await self._check_and_compact_messages(
                messages, max_context_tokens, auto_compact_enabled, protocol
            )

            prompt_tokens_est = estimate_conversation_tokens(messages, protocol=protocol)
            spinner = ThinkingSpinner(console, prompt_tokens_est)
            pause_ui, resume_ui = self._build_spinner_callbacks(spinner)
            query_context.pause_ui = pause_ui
            query_context.resume_ui = resume_ui
            permission_checker = self._build_permission_checker(pause_ui, resume_ui)

            try:
                await self._stream_query_messages(
                    messages=messages,
                    system_prompt=system_prompt,
                    context=context,
                    permission_checker=permission_checker,
                    spinner=spinner,
                    query_context=query_context,
                )

            except asyncio.CancelledError:
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
                self._finalize_query_stream(spinner, messages)

        except asyncio.CancelledError:
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
        """Run a coroutine on the persistent event loop thread."""
        if self._loop.is_closed():
            raise RuntimeError("UI event loop is closed")
        if threading.current_thread() is self._loop_thread:
            raise RuntimeError("_run_async cannot be called from the UI event loop thread")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def run_async(self, coro: Any) -> Any:
        """Public wrapper for running coroutines on the UI event loop."""
        return self._run_async(coro)

    def handle_slash_command(self, user_input: str) -> bool | str:
        """Handle slash commands.

        Returns True if handled as built-in, False if not a command,
        or a string if it's a custom command that should be sent to the AI.
        """
        from ripperdoc.cli.ui.rich_ui import suggest_slash_command_matches

        return _handle_slash_command(self, user_input, suggest_slash_command_matches)

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

        if self.conversation_messages:
            replay_limit = (
                self._resume_replay_max_messages if self._resumed_from_history else None
            )
            self.replay_conversation(self.conversation_messages, max_messages=replay_limit)
            self._resumed_from_history = False
            console.print()

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
            set_runtime_task_scope(session_id=None)
            # Cancel any running tasks before shutdown
            if self.query_context:
                abort_controller = getattr(self.query_context, "abort_controller", None)
                if abort_controller is not None:
                    abort_controller.set()

            if self.session_id and self.conversation_messages:
                self.console.print("[dim]Resume this session with:[/dim]")
                self.console.print(f"[dim]ripperdoc --resume {self.session_id}[/dim]")

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
                self._shutdown_event_loop_thread()
                sys.unraisablehook = original_hook

    async def _run_manual_compact(self, custom_instructions: str) -> None:
        """Manual compaction: clear bulky tool output and summarize conversation."""
        from rich.markup import escape

        model_profile = get_profile_for_pointer(self.model)
        protocol = provider_protocol(model_profile.protocol) if model_profile else "openai"

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
                self.console.print(
                    f"[yellow]PreCompact hook warning (ignored): {escape(str(reason))}[/yellow]"
                )
            self._display_hook_system_message(hook_result, "PreCompact")
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

    def _run_event_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _start_mcp_runtime_warmup(self) -> None:
        if self._loop.is_closed():
            return
        if self._mcp_warmup_future is not None and not self._mcp_warmup_future.done():
            return

        async def _warmup() -> McpRuntime:
            start = time.time()
            try:
                runtime = await ensure_mcp_runtime(
                    self.project_path,
                    wait_for_connections=True,
                )
                duration_ms = max((time.time() - start) * 1000, 0.0)
                logger.info(
                    "[ui] MCP background warmup completed",
                    extra={
                        "session_id": self.session_id,
                        "project_path": str(self.project_path),
                        "server_count": len(runtime.servers),
                        "duration_ms": round(duration_ms, 2),
                    },
                )
                return runtime
            except (OSError, RuntimeError, ConnectionError, ValueError) as exc:
                logger.warning(
                    "[ui] MCP background warmup failed: %s: %s",
                    type(exc).__name__,
                    exc,
                    extra={"session_id": self.session_id, "project_path": str(self.project_path)},
                )
                raise

        logger.debug(
            "[ui] Scheduling MCP background warmup",
            extra={"session_id": self.session_id, "project_path": str(self.project_path)},
        )
        self._mcp_warmup_future = asyncio.run_coroutine_threadsafe(_warmup(), self._loop)

    def _shutdown_event_loop_thread(self) -> None:
        if self._loop.is_closed():
            return

        async def _shutdown_asyncgens_only() -> None:
            await asyncio.get_running_loop().shutdown_asyncgens()

        try:
            self._run_async(_shutdown_asyncgens_only())
        except (RuntimeError, asyncio.CancelledError, concurrent.futures.TimeoutError):
            pass

        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._loop_thread.is_alive():
                self._loop_thread.join(timeout=2.0)
        finally:
            if not self._loop.is_closed():
                self._loop.close()


def check_onboarding_rich() -> bool:
    """Check if onboarding is complete and run if needed."""
    config = get_effective_config(Path.cwd())

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
    max_thinking_tokens: Optional[int] = None,
    max_turns: Optional[int] = None,
    permission_mode: str = "default",
    resume_messages: Optional[List[Any]] = None,
    initial_query: Optional[str] = None,
    additional_working_dirs: Optional[List[str]] = None,
) -> None:
    """Main entry point for Rich interface.

    Args:
        initial_query: If provided, automatically send this query after starting the session.
                      Used for piped stdin input (e.g., `echo "query" | ripperdoc`).
    """

    # Ensure onboarding is complete
    if not check_onboarding_rich():
        sys.exit(1)

    resolved_model = model or "main"
    if get_effective_model_profile(resolved_model) is None:
        logger.warning(
            "[ui] Requested model pointer not found; relying on runtime fallback behavior",
            extra={"model_pointer": resolved_model},
        )

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
        model=resolved_model,
        max_thinking_tokens=max_thinking_tokens,
        max_turns=max_turns,
        permission_mode=permission_mode,
        resume_messages=resume_messages,
        initial_query=initial_query,
        additional_working_dirs=additional_working_dirs,
    )
    ui.run()


if __name__ == "__main__":
    main_rich()
