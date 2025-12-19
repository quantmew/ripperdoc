"""Tool result renderers for CLI display.

This module provides a strategy pattern implementation for rendering different
tool results in the Rich CLI interface.
"""

from typing import Any, Callable, List, Optional

from rich.console import Console
from rich.markup import escape


class ToolResultRenderer:
    """Base class for rendering tool results to console."""

    def __init__(self, console: Console, verbose: bool = False):
        self.console = console
        self.verbose = verbose

    def can_handle(self, _sender: str) -> bool:
        """Return True if this renderer handles the given tool name."""
        raise NotImplementedError

    def render(self, _content: str, _tool_data: Any) -> None:
        """Render the tool result to console."""
        raise NotImplementedError

    def _get_field(self, data: Any, key: str, default: Any = None) -> Any:
        """Safely fetch a field from either an object or a dict."""
        if isinstance(data, dict):
            return data.get(key, default)
        return getattr(data, key, default)


class TodoResultRenderer(ToolResultRenderer):
    """Render Todo tool results."""

    def can_handle(self, sender: str) -> bool:
        return "Todo" in sender

    def render(self, content: str, _tool_data: Any) -> None:
        lines = content.splitlines()
        if lines:
            self.console.print(f"  ⎿  [dim]{escape(lines[0])}[/]")
            for line in lines[1:]:
                self.console.print(f"      {line}", markup=False)
        else:
            self.console.print("  ⎿  [dim]Todo update[/]")


class ReadResultRenderer(ToolResultRenderer):
    """Render Read tool results."""

    def can_handle(self, sender: str) -> bool:
        return "Read" in sender

    def render(self, content: str, _tool_data: Any) -> None:
        lines = content.split("\n")
        line_count = len(lines)
        self.console.print(f"  ⎿  [dim]Read {line_count} lines[/]")
        if self.verbose:
            preview = lines[:30]
            for line in preview:
                self.console.print(line, markup=False)
            if len(lines) > len(preview):
                self.console.print(f"[dim]... ({len(lines) - len(preview)} more lines)[/]")


class EditResultRenderer(ToolResultRenderer):
    """Render Write/Edit/MultiEdit tool results."""

    def can_handle(self, sender: str) -> bool:
        return "Write" in sender or "Edit" in sender or "MultiEdit" in sender

    def render(self, _content: str, tool_data: Any) -> None:
        if tool_data and (hasattr(tool_data, "file_path") or isinstance(tool_data, dict)):
            file_path = self._get_field(tool_data, "file_path")
            additions = self._get_field(tool_data, "additions", 0)
            deletions = self._get_field(tool_data, "deletions", 0)
            diff_with_line_numbers = self._get_field(tool_data, "diff_with_line_numbers", [])

            if not file_path:
                self.console.print("  ⎿  [dim]File updated successfully[/]")
                return

            self.console.print(
                f"  ⎿  [dim]Updated {escape(str(file_path))} with {additions} additions and {deletions} removals[/]"
            )

            if self.verbose:
                for line in diff_with_line_numbers:
                    self.console.print(line, markup=False)
        else:
            self.console.print("  ⎿  [dim]File updated successfully[/]")


class GlobResultRenderer(ToolResultRenderer):
    """Render Glob tool results."""

    def can_handle(self, sender: str) -> bool:
        return "Glob" in sender

    def render(self, content: str, _tool_data: Any) -> None:
        files = content.split("\n")
        file_count = len([f for f in files if f.strip()])
        self.console.print(f"  ⎿  [dim]Found {file_count} files[/]")
        if self.verbose:
            for line in files[:30]:
                if line.strip():
                    self.console.print(f"      {line}", markup=False)
            if file_count > 30:
                self.console.print(f"[dim]... ({file_count - 30} more)[/]")


class GrepResultRenderer(ToolResultRenderer):
    """Render Grep tool results."""

    def can_handle(self, sender: str) -> bool:
        return "Grep" in sender

    def render(self, content: str, _tool_data: Any) -> None:
        matches = content.split("\n")
        match_count = len([m for m in matches if m.strip()])
        self.console.print(f"  ⎿  [dim]Found {match_count} matches[/]")
        if self.verbose:
            for line in matches[:30]:
                if line.strip():
                    self.console.print(f"      {line}", markup=False)
            if match_count > 30:
                self.console.print(f"[dim]... ({match_count - 30} more)[/]")


class LSResultRenderer(ToolResultRenderer):
    """Render LS tool results."""

    def can_handle(self, sender: str) -> bool:
        return "LS" in sender

    def render(self, content: str, _tool_data: Any) -> None:
        tree_lines = content.splitlines()
        self.console.print(f"  ⎿  [dim]Directory tree ({len(tree_lines)} lines)[/]")
        if self.verbose:
            preview = tree_lines[:40]
            for line in preview:
                self.console.print(f"      {line}", markup=False)
            if len(tree_lines) > len(preview):
                self.console.print(f"[dim]... ({len(tree_lines) - len(preview)} more)[/]")


# Type alias for bash output parser callback
BashOutputParser = Callable[[str], tuple[List[str], List[str]]]


class BashResultRenderer(ToolResultRenderer):
    """Render Bash tool results."""

    def __init__(
        self,
        console: Console,
        verbose: bool = False,
        parse_fallback: Optional[BashOutputParser] = None,
    ):
        super().__init__(console, verbose)
        self._parse_fallback = parse_fallback

    def can_handle(self, sender: str) -> bool:
        return "Bash" in sender

    def render(self, content: str, tool_data: Any) -> None:
        stdout_lines: List[str] = []
        stderr_lines: List[str] = []
        exit_code = 0
        duration_ms = 0
        timeout_ms = 0

        if tool_data:
            exit_code = self._get_field(tool_data, "exit_code", 0)
            stdout = self._get_field(tool_data, "stdout", "") or ""
            stderr = self._get_field(tool_data, "stderr", "") or ""
            duration_ms = self._get_field(tool_data, "duration_ms", 0) or 0
            timeout_ms = self._get_field(tool_data, "timeout_ms", 0) or 0
            stdout_lines = stdout.splitlines() if stdout else []
            stderr_lines = stderr.splitlines() if stderr else []

        if not stdout_lines and not stderr_lines and content and self._parse_fallback:
            stdout_lines, stderr_lines = self._parse_fallback(content)

        show_inline_stdout = (
            stdout_lines and not stderr_lines and exit_code == 0 and not self.verbose
        )

        if show_inline_stdout:
            preview = stdout_lines if self.verbose else stdout_lines[:5]
            self.console.print(f"  ⎿  {preview[0]}", markup=False)
            for line in preview[1:]:
                self.console.print(f"      {line}", markup=False)
            if not self.verbose and len(stdout_lines) > len(preview):
                self.console.print(f"[dim]... ({len(stdout_lines) - len(preview)} more lines)[/]")
        else:
            self._render_detailed_output(
                tool_data, exit_code, duration_ms, timeout_ms, stdout_lines, stderr_lines
            )

    def _render_detailed_output(
        self,
        tool_data: Any,
        exit_code: int,
        duration_ms: float,
        timeout_ms: int,
        stdout_lines: List[str],
        stderr_lines: List[str],
    ) -> None:
        """Render detailed Bash output with exit code, stdout, stderr."""
        if tool_data:
            timing = ""
            if duration_ms:
                timing = f" ({duration_ms / 1000:.2f}s"
                if timeout_ms:
                    timing += f" / timeout {timeout_ms / 1000:.0f}s"
                timing += ")"
            elif timeout_ms:
                timing = f" (timeout {timeout_ms / 1000:.0f}s)"
            self.console.print(f"  ⎿  [dim]Exit code {exit_code}{timing}[/]")
        else:
            self.console.print("  ⎿  [dim]Command executed[/]")

        # Render stdout
        if stdout_lines:
            preview = stdout_lines if self.verbose else stdout_lines[:5]
            self.console.print("[dim]stdout:[/]")
            for line in preview:
                self.console.print(f"      {line}", markup=False)
            if not self.verbose and len(stdout_lines) > len(preview):
                self.console.print(
                    f"[dim]... ({len(stdout_lines) - len(preview)} more stdout lines)[/]"
                )
        else:
            self.console.print("[dim]stdout:[/]")
            self.console.print("      [dim](no stdout)[/]")

        # Render stderr
        if stderr_lines:
            preview = stderr_lines if self.verbose else stderr_lines[:5]
            self.console.print("[dim]stderr:[/]")
            for line in preview:
                self.console.print(f"      {line}", markup=False)
            if not self.verbose and len(stderr_lines) > len(preview):
                self.console.print(
                    f"[dim]... ({len(stderr_lines) - len(preview)} more stderr lines)[/]"
                )
        else:
            self.console.print("[dim]stderr:[/]")
            self.console.print("      [dim](no stderr)[/]")


class ToolResultRendererRegistry:
    """Registry that selects the appropriate renderer for a tool result."""

    def __init__(
        self,
        console: Console,
        verbose: bool = False,
        parse_bash_fallback: Optional[BashOutputParser] = None,
    ):
        self.console = console
        self.verbose = verbose
        self._renderers: List[ToolResultRenderer] = [
            TodoResultRenderer(console, verbose),
            ReadResultRenderer(console, verbose),
            EditResultRenderer(console, verbose),
            GlobResultRenderer(console, verbose),
            GrepResultRenderer(console, verbose),
            LSResultRenderer(console, verbose),
            BashResultRenderer(console, verbose, parse_bash_fallback),
        ]

    def get_renderer(self, sender: str) -> Optional[ToolResultRenderer]:
        """Get the appropriate renderer for the given tool name."""
        for renderer in self._renderers:
            if renderer.can_handle(sender):
                return renderer
        return None

    def render(self, sender: str, content: str, tool_data: Any) -> bool:
        """Render the tool result. Returns True if rendered, False otherwise."""
        renderer = self.get_renderer(sender)
        if renderer:
            renderer.render(content, tool_data)
            return True
        return False


__all__ = [
    "ToolResultRenderer",
    "ToolResultRendererRegistry",
    "TodoResultRenderer",
    "ReadResultRenderer",
    "EditResultRenderer",
    "GlobResultRenderer",
    "GrepResultRenderer",
    "LSResultRenderer",
    "BashResultRenderer",
    "BashOutputParser",
]
