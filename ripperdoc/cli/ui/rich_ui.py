"""Rich-based CLI interface for Ripperdoc.

This module provides a clean, minimal terminal UI using Rich for the Ripperdoc agent.
"""

import asyncio
import json
import sys
import uuid
import re
from typing import List, Dict, Any, Optional, Union, Iterable
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text
from rich import box
from rich.markup import escape

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.shortcuts.prompt import CompleteStyle
from prompt_toolkit.history import InMemoryHistory

from ripperdoc import __version__
from ripperdoc.core.config import get_global_config, provider_protocol
from ripperdoc.core.default_tools import get_default_tools
from ripperdoc.core.query import query, QueryContext
from ripperdoc.core.system_prompt import build_system_prompt
from ripperdoc.core.skills import build_skill_summary, load_all_skills
from ripperdoc.cli.commands import (
    get_slash_command,
    list_slash_commands,
    slash_command_completions,
)
from ripperdoc.cli.ui.helpers import get_profile_for_pointer
from ripperdoc.core.permissions import make_permission_checker
from ripperdoc.cli.ui.spinner import Spinner
from ripperdoc.cli.ui.thinking_spinner import ThinkingSpinner
from ripperdoc.cli.ui.context_display import context_usage_lines
from ripperdoc.utils.message_compaction import (
    compact_messages,
    estimate_conversation_tokens,
    estimate_used_tokens,
    get_context_usage_status,
    get_remaining_context_tokens,
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
from ripperdoc.core.query import query_llm
from ripperdoc.utils.messages import (
    UserMessage,
    AssistantMessage,
    ProgressMessage,
    create_user_message,
    create_assistant_message,
)
from ripperdoc.utils.log import enable_session_file_logging, get_logger
from ripperdoc.cli.ui.tool_renderers import ToolResultRendererRegistry


# Type alias for conversation messages
ConversationMessage = Union[UserMessage, AssistantMessage, ProgressMessage]

THINKING_WORDS: list[str] = [
    "Accomplishing",
    "Actioning",
    "Actualizing",
    "Baking",
    "Booping",
    "Brewing",
    "Calculating",
    "Cerebrating",
    "Channelling",
    "Churning",
    "Clauding",
    "Coalescing",
    "Cogitating",
    "Computing",
    "Combobulating",
    "Concocting",
    "Conjuring",
    "Considering",
    "Contemplating",
    "Cooking",
    "Crafting",
    "Creating",
    "Crunching",
    "Deciphering",
    "Deliberating",
    "Determining",
    "Discombobulating",
    "Divining",
    "Doing",
    "Effecting",
    "Elucidating",
    "Enchanting",
    "Envisioning",
    "Finagling",
    "Flibbertigibbeting",
    "Forging",
    "Forming",
    "Frolicking",
    "Generating",
    "Germinating",
    "Hatching",
    "Herding",
    "Honking",
    "Ideating",
    "Imagining",
    "Incubating",
    "Inferring",
    "Manifesting",
    "Marinating",
    "Meandering",
    "Moseying",
    "Mulling",
    "Mustering",
    "Musing",
    "Noodling",
    "Percolating",
    "Perusing",
    "Philosophising",
    "Pontificating",
    "Pondering",
    "Processing",
    "Puttering",
    "Puzzling",
    "Reticulating",
    "Ruminating",
    "Scheming",
    "Schlepping",
    "Shimmying",
    "Simmering",
    "Smooshing",
    "Spelunking",
    "Spinning",
    "Stewing",
    "Sussing",
    "Synthesizing",
    "Thinking",
    "Tinkering",
    "Transmuting",
    "Unfurling",
    "Unravelling",
    "Vibing",
    "Wandering",
    "Whirring",
    "Wibbling",
    "Wizarding",
    "Working",
    "Wrangling",
]

console = Console()
logger = get_logger()

# Keep a small window of recent messages alongside the summary after /compact so
# the model retains immediate context.
RECENT_MESSAGES_AFTER_COMPACT = 8


def create_welcome_panel() -> Panel:
    """Create a welcome panel."""

    welcome_content = """
[bold cyan]Welcome to Ripperdoc![/bold cyan]

Ripperdoc is an AI-powered coding assistant that helps with software development tasks.
You can read files, edit code, run commands, and help with various programming tasks.

[dim]Type your questions below. Press Ctrl+C to exit.[/dim]
"""

    return Panel(
        welcome_content,
        title=f"Ripperdoc v{__version__}",
        border_style="cyan",
        box=box.ROUNDED,
        padding=(1, 2),
    )


def create_status_bar() -> Text:
    """Create a status bar with current information."""
    profile = get_profile_for_pointer("main")
    model_name = profile.model if profile else "Not configured"

    status_text = Text()
    status_text.append("Ripperdoc", style="bold cyan")
    status_text.append(" • ")
    status_text.append(model_name, style="dim")
    status_text.append(" • ")
    status_text.append("Ready", style="green")

    return status_text


class RichUI:
    """Rich-based UI for Ripperdoc."""

    def __init__(
        self,
        safe_mode: bool = False,
        verbose: bool = False,
        session_id: Optional[str] = None,
        log_file_path: Optional[Path] = None,
    ):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self.console = console
        self.safe_mode = safe_mode
        self.verbose = verbose
        self.conversation_messages: List[ConversationMessage] = []
        self._saved_conversation: Optional[List[ConversationMessage]] = None
        self.query_context: Optional[QueryContext] = None
        self._current_tool: Optional[str] = None
        self._should_exit: bool = False
        self.command_list = list_slash_commands()
        self._command_completions = slash_command_completions()
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
                "safe_mode": self.safe_mode,
                "verbose": self.verbose,
            },
        )
        self._session_history = SessionHistory(self.project_path, self.session_id)
        self._permission_checker = (
            make_permission_checker(self.project_path, safe_mode) if safe_mode else None
        )
        # Keep MCP runtime alive for the whole UI session. Create it on the UI loop up front.
        try:
            self._run_async(ensure_mcp_runtime(self.project_path))
        except (OSError, RuntimeError, ConnectionError) as exc:
            logger.warning(
                "[ui] Failed to initialize MCP runtime at startup: %s: %s",
                type(exc).__name__, exc,
                extra={"session_id": self.session_id},
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

    def _log_message(self, message: Any) -> None:
        """Best-effort persistence of a message to the session log."""
        try:
            self._session_history.append(message)
        except (OSError, IOError, json.JSONDecodeError) as exc:
            # Logging failures should never interrupt the UI flow
            logger.warning(
                "[ui] Failed to append message to session history: %s: %s",
                type(exc).__name__, exc,
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
                type(exc).__name__, exc,
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
            self._print_human_or_assistant(sender, content)
            return

        if tool_type == "call":
            self._print_tool_call(sender, content, tool_args)
            return

        if tool_type == "result":
            self._print_tool_result(sender, content, tool_data, tool_error)
            return

        self._print_generic_tool(sender, content)

    def _format_tool_args(self, tool_name: str, tool_args: Optional[dict]) -> list[str]:
        """Render tool arguments into concise display-friendly parts."""
        if not tool_args:
            return []

        args_parts: list[str] = []

        def _format_arg(arg_key: str, arg_value: Any) -> str:
            if arg_key == "todos" and isinstance(arg_value, list):
                counts = {"pending": 0, "in_progress": 0, "completed": 0}
                for item in arg_value:
                    status = ""
                    if isinstance(item, dict):
                        status = item.get("status", "")
                    elif hasattr(item, "get"):
                        status = item.get("status", "")
                    elif hasattr(item, "status"):
                        status = getattr(item, "status")
                    if status in counts:
                        counts[status] += 1
                total = len(arg_value)
                return f"{arg_key}: {total} items"
            if isinstance(arg_value, (list, dict)):
                return f"{arg_key}: {len(arg_value)} items"
            if isinstance(arg_value, str) and len(arg_value) > 50:
                return f'{arg_key}: "{arg_value[:50]}..."'
            return f"{arg_key}: {arg_value}"

        if tool_name == "Bash":
            command_value = tool_args.get("command")
            if command_value is not None:
                args_parts.append(_format_arg("command", command_value))

            background_value = tool_args.get("run_in_background", tool_args.get("runInBackground"))
            background_value = bool(background_value) if background_value is not None else False
            args_parts.append(f"background: {background_value}")

            sandbox_value = tool_args.get("sandbox")
            sandbox_value = bool(sandbox_value) if sandbox_value is not None else False
            args_parts.append(f"sandbox: {sandbox_value}")

            for key, value in tool_args.items():
                if key in {"command", "run_in_background", "runInBackground", "sandbox"}:
                    continue
                args_parts.append(_format_arg(key, value))
            return args_parts

        # Special handling for Edit and MultiEdit tools - don't show old_string
        if tool_name in ["Edit", "MultiEdit"]:
            for key, value in tool_args.items():
                if key == "new_string":
                    continue  # Skip new_string for Edit/MultiEdit tools
                if key == "old_string":
                    continue  # Skip old_string for Edit/MultiEdit tools
                # For MultiEdit, also handle edits array
                if key == "edits" and isinstance(value, list):
                    args_parts.append(f"edits: {len(value)} operations")
                    continue
                args_parts.append(_format_arg(key, value))
            return args_parts

        for key, value in tool_args.items():
            args_parts.append(_format_arg(key, value))
        return args_parts

    def _print_tool_call(self, sender: str, content: str, tool_args: Optional[dict]) -> None:
        """Render a tool invocation line."""
        if sender == "Task":
            subagent = ""
            if isinstance(tool_args, dict):
                subagent = tool_args.get("subagent_type") or tool_args.get("subagent") or ""
            desc = ""
            if isinstance(tool_args, dict):
                raw_desc = tool_args.get("description") or tool_args.get("prompt") or ""
                desc = raw_desc if len(str(raw_desc)) <= 120 else str(raw_desc)[:117] + "..."
            label = f"-> Launching subagent: {subagent or 'unknown'}"
            if desc:
                label += f" — {desc}"
            self.console.print(f"[cyan]{escape(label)}[/cyan]")
            return

        tool_name = sender if sender != "Ripperdoc" else content
        tool_display = f"● {tool_name}("

        args_parts = self._format_tool_args(tool_name, tool_args)
        if args_parts:
            tool_display += ", ".join(args_parts)
        tool_display += ")"

        self.console.print(f"[dim cyan]{escape(tool_display)}[/]")

    def _print_tool_result(
        self, sender: str, content: str, tool_data: Any, tool_error: bool = False
    ) -> None:
        """Render a tool result summary using the renderer registry."""
        # Check for failure states
        failed = tool_error
        if tool_data is not None:
            if isinstance(tool_data, dict):
                failed = failed or (tool_data.get("success") is False)
            else:
                success = getattr(tool_data, "success", None)
                failed = failed or (success is False)
            failed = failed or bool(self._get_tool_field(tool_data, "is_error"))

        # Extract warning/token info
        warning_text = None
        token_estimate = None
        if tool_data is not None:
            warning_text = self._get_tool_field(tool_data, "warning")
            token_estimate = self._get_tool_field(tool_data, "token_estimate")

        # Handle failure case
        if failed:
            if content:
                self.console.print(f"  ⎿  [red]{escape(content)}[/red]")
            else:
                self.console.print(f"  ⎿  [red]{escape(sender)} failed[/red]")
            return

        # Display warnings and token estimates
        if warning_text:
            self.console.print(f"  ⎿  [yellow]{escape(str(warning_text))}[/yellow]")
            if token_estimate:
                self.console.print(
                    f"      [dim]Estimated tokens: {escape(str(token_estimate))}[/dim]"
                )
        elif token_estimate and self.verbose:
            self.console.print(f"  ⎿  [dim]Estimated tokens: {escape(str(token_estimate))}[/dim]")

        # Handle empty content
        if not content:
            self.console.print("  ⎿  [dim]Tool completed[/]")
            return

        # Use renderer registry for tool-specific rendering
        registry = ToolResultRendererRegistry(
            self.console, self.verbose, self._parse_bash_output_sections
        )
        if registry.render(sender, content, tool_data):
            return

        # Fallback for unhandled tools
        self.console.print("  ⎿  [dim]Tool completed[/]")

    def _print_generic_tool(self, sender: str, content: str) -> None:
        """Fallback rendering for miscellaneous tool messages."""
        if sender == "Task" and isinstance(content, str) and content.startswith("[subagent:"):
            agent_label = content.split("]", 1)[0].replace("[subagent:", "").strip()
            summary = content.split("]", 1)[1].strip() if "]" in content else ""
            self.console.print(f"[green]↳ Subagent {escape(agent_label)} finished[/green]")
            if summary:
                self.console.print(f"    {summary}", markup=False)
            return
        self.console.print(f"[dim cyan][Tool] {escape(sender)}: {escape(content)}[/]")

    def _print_human_or_assistant(self, sender: str, content: str) -> None:
        """Render messages from the user or assistant."""
        if sender.lower() == "you":
            self.console.print(f"[bold green]{escape(sender)}:[/] {escape(content)}")
            return
        self.console.print(Markdown(content))

    def _get_tool_field(self, data: Any, key: str, default: Any = None) -> Any:
        """Safely fetch a field from either an object or a dict."""
        if isinstance(data, dict):
            return data.get(key, default)
        return getattr(data, key, default)

    def _parse_bash_output_sections(self, content: str) -> tuple[List[str], List[str]]:
        """Fallback parser to pull stdout/stderr sections from a text block."""
        stdout_lines: List[str] = []
        stderr_lines: List[str] = []
        if not content:
            return stdout_lines, stderr_lines

        current: Optional[str] = None
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("stdout:"):
                current = "stdout"
                remainder = line.split("stdout:", 1)[1].strip()
                if remainder:
                    stdout_lines.append(remainder)
                continue
            if stripped.startswith("stderr:"):
                current = "stderr"
                remainder = line.split("stderr:", 1)[1].strip()
                if remainder:
                    stderr_lines.append(remainder)
                continue
            if stripped.startswith("exit code:"):
                break
            if current == "stdout":
                stdout_lines.append(line)
            elif current == "stderr":
                stderr_lines.append(line)

        return stdout_lines, stderr_lines

    def _stringify_message_content(self, content: Any) -> str:
        """Extract readable text from a message content payload."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for block in content:
                text = getattr(block, "text", None)
                if text is None:
                    text = getattr(block, "thinking", None)
                if not text and isinstance(block, dict):
                    text = block.get("text") or block.get("thinking") or block.get("data")
                if text:
                    parts.append(str(text))
            return "\n".join(parts)
        return ""

    def _format_reasoning_preview(self, reasoning: Any) -> str:
        """Best-effort stringify for reasoning/thinking traces."""
        if reasoning is None:
            return ""
        if isinstance(reasoning, str):
            preview = reasoning.strip()
        else:
            try:
                preview = json.dumps(reasoning, ensure_ascii=False)
            except (TypeError, ValueError, OverflowError):
                preview = str(reasoning)
        preview = preview.strip()
        if len(preview) > 4000:
            preview = preview[:4000] + "…"
        return preview

    def _print_reasoning(self, reasoning: Any) -> None:
        """Display thinking traces in a dim style."""
        preview = self._format_reasoning_preview(reasoning)
        if not preview:
            return
        # Collapse excessive blank lines to keep the thinking block compact.
        preview = re.sub(r"\n{2,}", "\n", preview)
        self.console.print(f"[dim]🧠 Thinking: {escape(preview)}[/]")

    def _render_transcript(self, messages: List[ConversationMessage]) -> str:
        """Render a simple transcript for summarization."""
        lines: List[str] = []
        for msg in messages:
            role = getattr(msg, "type", "") or getattr(msg, "role", "")
            message_payload = getattr(msg, "message", None) or getattr(msg, "content", None)
            if hasattr(message_payload, "content"):
                message_payload = getattr(message_payload, "content")
            text = self._stringify_message_content(message_payload)
            if not text:
                continue
            label = "User" if role == "user" else "Assistant" if role == "assistant" else "Other"
            lines.append(f"{label}: {text}")
        return "\n".join(lines)

    def _extract_assistant_text(self, assistant_message: Any) -> str:
        """Extract plain text from an AssistantMessage."""
        if isinstance(assistant_message.message.content, str):
            return assistant_message.message.content
        if isinstance(assistant_message.message.content, list):
            parts: List[str] = []
            for block in assistant_message.message.content:
                if getattr(block, "type", None) == "text" and getattr(block, "text", None):
                    parts.append(str(block.text))
            return "\n".join(parts)
        return ""

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

    def _check_and_compact_messages(
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
            compaction = compact_messages(messages, protocol=protocol)  # type: ignore[arg-type]
            if compaction.was_compacted:
                if self._saved_conversation is None:
                    self._saved_conversation = original_messages  # type: ignore[assignment]
                console.print(
                    f"[yellow]Auto-compacted conversation (saved ~{compaction.tokens_saved} tokens). "
                    f"Estimated usage: {compaction.tokens_after}/{max_context_tokens} tokens.[/yellow]"
                )
                logger.info(
                    "[ui] Auto-compacted conversation",
                    extra={
                        "session_id": self.session_id,
                        "tokens_before": compaction.tokens_before,
                        "tokens_after": compaction.tokens_after,
                        "tokens_saved": compaction.tokens_saved,
                        "cleared_tool_ids": list(compaction.cleared_tool_ids),
                    },
                )
                return compaction.messages  # type: ignore[return-value]

        return messages

    def _handle_assistant_message(
        self,
        message: AssistantMessage,
        tool_registry: Dict[str, Dict[str, Any]],
    ) -> Optional[str]:
        """Handle an assistant message from the query stream.

        Returns:
            The last tool name if a tool_use block was processed, None otherwise.
        """
        meta = getattr(getattr(message, "message", None), "metadata", {}) or {}
        reasoning_payload = (
            meta.get("reasoning_content")
            or meta.get("reasoning")
            or meta.get("reasoning_details")
        )
        if reasoning_payload:
            self._print_reasoning(reasoning_payload)

        last_tool_name: Optional[str] = None

        if isinstance(message.message.content, str):
            self.display_message("Ripperdoc", message.message.content)
        elif isinstance(message.message.content, list):
            for block in message.message.content:
                if hasattr(block, "type") and block.type == "text" and block.text:
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
    ) -> None:
        """Handle a user message containing tool results."""
        if not isinstance(message.message.content, list):
            return

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
            self.display_message("System", f"Progress: {message.content}", is_tool=True)
        elif message.content and isinstance(message.content, str):
            if message.content.startswith("Subagent: "):
                self.display_message(
                    "Subagent", message.content[len("Subagent: ") :], is_tool=True
                )
            elif message.content.startswith("Subagent"):
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
                tools=self.get_default_tools(), safe_mode=self.safe_mode, verbose=self.verbose
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
            messages = self._check_and_compact_messages(
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
                    # Wrap spinner restart in try-except to prevent exceptions
                    # from discarding the permission result
                    try:
                        spinner.start()
                        spinner.update("Thinking...")
                    except (RuntimeError, ValueError, OSError) as exc:
                        logger.debug(
                            "[ui] Failed to restart spinner after permission check: %s: %s",
                            type(exc).__name__, exc,
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
                        result = self._handle_assistant_message(message, tool_registry)
                        if result:
                            last_tool_name = result

                    elif message.type == "user" and isinstance(message, UserMessage):
                        self._handle_tool_result_message(message, tool_registry, last_tool_name)

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
                    type(e).__name__, e,
                    extra={"session_id": self.session_id},
                )
                self.display_message("System", f"Error: {str(e)}", is_tool=True)
            finally:
                try:
                    spinner.stop()
                except (RuntimeError, ValueError, OSError) as exc:
                    logger.warning(
                        "[ui] Failed to stop spinner: %s: %s",
                        type(exc).__name__, exc,
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
                type(exc).__name__, exc,
                extra={"session_id": self.session_id},
            )
            self.display_message("System", f"Error: {str(exc)}", is_tool=True)

    def _run_async(self, coro: Any) -> Any:
        """Run a coroutine on the persistent event loop."""
        if self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        return self._loop.run_until_complete(coro)

    def run_async(self, coro: Any) -> Any:
        """Public wrapper for running coroutines on the UI event loop."""
        return self._run_async(coro)

    def handle_slash_command(self, user_input: str) -> bool:
        """Handle slash commands. Returns True if the input was handled."""

        if not user_input.startswith("/"):
            return False

        parts = user_input[1:].strip().split()
        if not parts:
            self.console.print("[red]No command provided after '/'.[/red]")
            return True

        command_name = parts[0].lower()
        trimmed_arg = " ".join(parts[1:]).strip()
        command = get_slash_command(command_name)
        if command is None:
            self.console.print(f"[red]Unknown command: {escape(command_name)}[/red]")
            return True

        return command.handler(self, trimmed_arg)

    def get_prompt_session(self) -> PromptSession:
        """Create (or return) the prompt session with command completion."""
        if self._prompt_session:
            return self._prompt_session

        class SlashCommandCompleter(Completer):
            """Autocomplete for slash commands."""

            def __init__(self, completions: List):
                self.completions = completions

            def get_completions(self, document: Any, complete_event: Any) -> Iterable[Completion]:
                text = document.text_before_cursor
                if not text.startswith("/"):
                    return
                query = text[1:]
                for name, cmd in self.completions:
                    if name.startswith(query):
                        yield Completion(
                            name,
                            start_position=-len(query),
                            display=name,
                            display_meta=cmd.description,
                        )

        self._prompt_session = PromptSession(
            completer=SlashCommandCompleter(self._command_completions),
            complete_style=CompleteStyle.COLUMN,
            complete_while_typing=True,
            history=InMemoryHistory(),
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
        console.print("[dim]Tip: type '/' then press Tab to see available commands.[/dim]\n")

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
                        if handled:
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
                    self._run_async(self.process_query(user_input))

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
                except (OSError, ConnectionError, RuntimeError, ValueError, KeyError, TypeError) as e:
                    console.print(f"[red]Error: {escape(str(e))}[/]")
                    logger.warning(
                        "[ui] Error in interactive loop: %s: %s",
                        type(e).__name__, e,
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
                        type(exc).__name__, exc,
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
        if len(self.conversation_messages) < 2:
            console.print("[yellow]Not enough conversation history to compact.[/yellow]")
            return

        model_profile = get_profile_for_pointer("main")
        protocol = provider_protocol(model_profile.provider) if model_profile else "openai"

        original_messages = list(self.conversation_messages)
        tokens_before = estimate_conversation_tokens(original_messages, protocol=protocol)

        compaction = compact_messages(original_messages, protocol=protocol)
        messages_for_summary = compaction.messages

        spinner = Spinner(console, "Summarizing conversation...", spinner="dots")
        summary_text = ""
        try:
            spinner.start()
            summary_text = await self._summarize_conversation(
                messages_for_summary, custom_instructions
            )
        except (OSError, RuntimeError, ConnectionError, ValueError, KeyError) as e:
            console.print(f"[red]Error during compaction: {escape(str(e))}[/red]")
            logger.warning(
                "[ui] Error during manual compaction: %s: %s",
                type(e).__name__, e,
                extra={"session_id": self.session_id},
            )
            return
        finally:
            spinner.stop()

        if not summary_text:
            console.print("[red]Failed to summarize conversation for compaction.[/red]")
            return

        if summary_text.strip() == "":
            console.print("[red]Summarization returned empty content; aborting compaction.[/red]")
            return

        self._saved_conversation = original_messages
        summary_message = create_assistant_message(
            f"Conversation summary (generated by /compact):\n{summary_text}"
        )
        non_progress_messages = [
            m for m in messages_for_summary if getattr(m, "type", "") != "progress"
        ]
        recent_tail = (
            non_progress_messages[-RECENT_MESSAGES_AFTER_COMPACT:]
            if RECENT_MESSAGES_AFTER_COMPACT > 0
            else []
        )
        new_conversation = [
            create_user_message(
                "Conversation compacted. Summary plus recent turns are kept; older tool output may "
                "be cleared."
            ),
            summary_message,
            *recent_tail,
        ]
        self.conversation_messages = new_conversation
        tokens_after = estimate_conversation_tokens(new_conversation, protocol=protocol)
        tokens_saved = max(0, tokens_before - tokens_after)
        console.print(
            f"[green]✓ Conversation compacted[/green] "
            f"(saved ~{tokens_saved} tokens). Use /resume to restore full history."
        )

    async def _summarize_conversation(
        self,
        messages: List[ConversationMessage],
        custom_instructions: str,
    ) -> str:
        """Summarize the given conversation using the configured model."""
        # Keep transcript bounded to recent turns to avoid blowing context.
        recent_messages = messages[-40:]
        transcript = self._render_transcript(recent_messages)
        if not transcript.strip():
            return ""

        instructions = (
            "You are a helpful assistant summarizing the prior conversation. "
            "Produce a concise bullet-list summary covering key decisions, important context, "
            "commands run, files touched, and pending TODOs. Include blockers or open questions. "
            "Keep it brief."
        )
        if custom_instructions.strip():
            instructions += f"\nCustom instructions: {custom_instructions.strip()}"

        user_content = (
            f"Summarize the following conversation between a user and an assistant:\n\n{transcript}"
        )

        assistant_response = await query_llm(
            messages=[{"role": "user", "content": user_content}],  # type: ignore[list-item]
            system_prompt=instructions,
            tools=[],
            max_thinking_tokens=0,
            model="main",
        )
        return self._extract_assistant_text(assistant_response)

    def _print_shortcuts(self) -> None:
        """Show common keyboard shortcuts and prefixes."""
        pairs = [
            ("? for shortcuts", "! for bash mode"),
            ("/ for commands", "shift + tab to auto-accept edits"),
            # "@ for file paths", "ctrl + o for verbose output"),
            # "# to memorize", "ctrl + v to paste images"),
            # "& for background", "ctrl + t to show todos"),
            # "double tap esc to clear input", "tab to toggle thinking"),
            # "ctrl + _ to undo", "ctrl + z to suspend"),
            # "shift + enter for newline", ""),
        ]
        console.print("[dim]Shortcuts[/dim]")
        for left, right in pairs:
            left_text = f"  {left}".ljust(32)
            right_text = f"{right}" if right else ""
            console.print(f"{left_text}{right_text}")


def check_onboarding_rich() -> bool:
    """Check if onboarding is complete and run if needed."""
    config = get_global_config()

    if config.has_completed_onboarding:
        return True

    # Use simple console onboarding
    from ripperdoc.cli.cli import check_onboarding

    return check_onboarding()


def main_rich(
    safe_mode: bool = False,
    verbose: bool = False,
    session_id: Optional[str] = None,
    log_file_path: Optional[Path] = None,
) -> None:
    """Main entry point for Rich interface."""

    # Ensure onboarding is complete
    if not check_onboarding_rich():
        sys.exit(1)

    # Run the Rich UI
    ui = RichUI(
        safe_mode=safe_mode,
        verbose=verbose,
        session_id=session_id,
        log_file_path=log_file_path,
    )
    ui.run()


if __name__ == "__main__":
    main_rich()
