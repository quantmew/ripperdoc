"""Message display and rendering utilities for RichUI.

This module handles rendering conversation messages to the terminal, including:
- Tool call and result formatting
- Assistant/user message display
- Reasoning block rendering
"""

from typing import Any, Callable, List, Optional, Tuple, Union

from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape

from ripperdoc.cli.ui.tool_renderers import ToolResultRendererRegistry
from ripperdoc.utils.messages import UserMessage, AssistantMessage, ProgressMessage
from ripperdoc.utils.message_formatting import format_reasoning_preview

ConversationMessage = Union[UserMessage, AssistantMessage, ProgressMessage]


class MessageDisplay:
    """Handles message rendering and display operations."""

    def __init__(self, console: Console, verbose: bool = False, show_full_thinking: bool = False):
        """Initialize the message display handler.

        Args:
            console: Rich console for output
            verbose: Whether to show verbose output
            show_full_thinking: Whether to show full reasoning content instead of truncated preview
        """
        self.console = console
        self.verbose = verbose
        self.show_full_thinking = show_full_thinking

    def format_tool_args(self, tool_name: str, tool_args: Optional[dict]) -> List[str]:
        """Render tool arguments into concise display-friendly parts."""
        if not tool_args:
            return []

        args_parts: List[str] = []

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

    def print_tool_call(self, sender: str, content: str, tool_args: Optional[dict]) -> None:
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

        args_parts = self.format_tool_args(tool_name, tool_args)
        if args_parts:
            tool_display += ", ".join(args_parts)
        tool_display += ")"

        self.console.print(f"[dim cyan]{escape(tool_display)}[/]")

    def print_tool_result(
        self,
        sender: str,
        content: str,
        tool_data: Any,
        tool_error: bool = False,
        parse_bash_output_fn: Optional[Callable] = None,
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
            self.console, self.verbose, parse_bash_output_fn or self._default_parse_bash
        )
        if registry.render(sender, content, tool_data):
            return

        # Fallback for unhandled tools
        self.console.print("  ⎿  [dim]Tool completed[/]")

    def print_generic_tool(self, sender: str, content: str) -> None:
        """Fallback rendering for miscellaneous tool messages."""
        if sender == "Task" and isinstance(content, str) and content.startswith("[subagent:"):
            agent_label = content.split("]", 1)[0].replace("[subagent:", "").strip()
            summary = content.split("]", 1)[1].strip() if "]" in content else ""
            self.console.print(f"[green]↳ Subagent {escape(agent_label)} finished[/green]")
            if summary:
                self.console.print(f"    {summary}", markup=False)
            return
        self.console.print(f"[dim cyan][Tool] {escape(sender)}: {escape(content)}[/]")

    def print_human_or_assistant(self, sender: str, content: str) -> None:
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

    def _default_parse_bash(self, content: str) -> Tuple[List[str], List[str]]:
        """Default bash output parser."""
        return parse_bash_output_sections(content)

    def print_reasoning(self, reasoning: Any) -> None:
        """Display a collapsed preview of reasoning/thinking blocks."""
        preview = format_reasoning_preview(reasoning, self.show_full_thinking)
        if preview:
            self.console.print(f"[dim italic]Thinking: {escape(preview)}[/]")


def parse_bash_output_sections(content: str) -> Tuple[List[str], List[str]]:
    """Parse stdout/stderr sections from a bash output text block."""
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
