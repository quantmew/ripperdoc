"""Tool result renderers for CLI display.

This module provides a strategy pattern implementation for rendering different
tool results in the Rich CLI interface.
"""

from typing import Any, Callable, List, Literal, Optional, TypedDict

from rich.console import Console
from rich.markup import escape

from ripperdoc.utils.tasks import list_tasks, resolve_task_list_id
from ripperdoc.utils.teams import get_active_team_name, get_team


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


TaskRenderStatus = Literal["pending", "in_progress", "completed"]


class TaskRenderEntry(TypedDict):
    id: str
    subject: str
    status: TaskRenderStatus
    owner: Optional[str]
    blockedBy: list[str]


class TodoResultRenderer(ToolResultRenderer):
    """Render Todo tool results."""

    @staticmethod
    def _render_text(console: Console, content: str, fallback: str) -> None:
        lines = content.splitlines()
        if lines:
            console.print(f"  ⎿  [dim]{escape(lines[0])}[/]")
            for line in lines[1:]:
                console.print(f"    {line}", markup=False)
        else:
            console.print(f"  ⎿  [dim]{fallback}[/]")

    def can_handle(self, sender: str) -> bool:
        return "Todo" in sender

    def render(self, content: str, _tool_data: Any) -> None:
        self._render_text(self.console, content, "Todo update")


class TaskGraphResultRenderer(ToolResultRenderer):
    """Render Task Graph tool results as a todo-like task panel."""

    _HANDLED_TOOLS = {"TaskCreate", "TaskUpdate", "TaskList"}
    _STATUS_MARKER = {"completed": "●", "in_progress": "◐", "pending": "○"}

    def can_handle(self, sender: str) -> bool:
        return sender in self._HANDLED_TOOLS

    def _resolve_active_task_list_id(self) -> str:
        team_name = get_active_team_name()
        if team_name:
            team = get_team(team_name)
            if team is not None:
                return team.task_list_id
        return resolve_task_list_id()

    def _task_sort_key(self, task_id: str) -> tuple[int, int | str]:
        if str(task_id).isdigit():
            return (0, int(task_id))
        return (1, str(task_id))

    def _entries_from_tool_data(self, tool_data: Any) -> list[TaskRenderEntry]:
        tasks = self._get_field(tool_data, "tasks")
        if not isinstance(tasks, list):
            return []

        entries: list[TaskRenderEntry] = []
        for raw in tasks:
            if hasattr(raw, "model_dump"):
                raw = raw.model_dump(by_alias=True, mode="json")
            if not isinstance(raw, dict):
                continue

            task_id = str(raw.get("id", "")).strip()
            subject = str(raw.get("subject", "")).strip()
            raw_status = str(raw.get("status", "pending")).strip() or "pending"
            status: TaskRenderStatus = (
                raw_status if raw_status in {"pending", "in_progress", "completed"} else "pending"
            )
            owner = raw.get("owner")
            blocked_by = raw.get("blockedBy")
            if not isinstance(blocked_by, list):
                blocked_by = raw.get("blocked_by")
            if not isinstance(blocked_by, list):
                blocked_by = []

            if not task_id or not subject:
                continue

            entries.append(
                {
                    "id": task_id,
                    "subject": subject,
                    "status": status,
                    "owner": str(owner).strip() if owner else None,
                    "blockedBy": [str(dep).strip() for dep in blocked_by if str(dep).strip()],
                }
            )

        entries.sort(key=lambda item: self._task_sort_key(item["id"]))
        return entries

    def _entries_from_storage(self) -> list[TaskRenderEntry]:
        try:
            task_list_id = self._resolve_active_task_list_id()
            tasks = list_tasks(task_list_id=task_list_id)
        except (OSError, RuntimeError, ValueError, TypeError):
            return []

        by_id = {task.id: task for task in tasks}
        entries: list[TaskRenderEntry] = []
        for task in tasks:
            metadata = task.metadata if isinstance(task.metadata, dict) else {}
            if metadata.get("_internal"):
                continue

            blockers = [
                dep
                for dep in task.blocked_by
                if (by_id.get(dep) is not None and by_id[dep].status != "completed")
            ]
            entries.append(
                {
                    "id": task.id,
                    "subject": task.subject,
                    "status": task.status,
                    "owner": task.owner,
                    "blockedBy": blockers,
                }
            )

        entries.sort(key=lambda item: self._task_sort_key(item["id"]))
        return entries

    def _summary(self, entries: list[TaskRenderEntry]) -> str:
        pending = len([item for item in entries if item["status"] == "pending"])
        in_progress = len([item for item in entries if item["status"] == "in_progress"])
        completed = len([item for item in entries if item["status"] == "completed"])
        return (
            f"Tasks updated (total {len(entries)}; "
            f"{pending} pending, {in_progress} in progress, {completed} completed)."
        )

    def _render_task_list_text(self, entries: list[TaskRenderEntry]) -> str:
        lines: list[str] = []
        for entry in entries:
            marker = self._STATUS_MARKER.get(entry["status"], "○")
            task_id = entry["id"]
            subject = entry["subject"]
            owner_raw = entry["owner"]
            owner = str(owner_raw).strip() if owner_raw is not None else ""
            blocked_by = entry["blockedBy"]

            line = f"{marker} {subject}"
            if owner:
                line += f" @{owner}"
            if blocked_by:
                blocker_text = ", ".join([str(dep).strip() for dep in blocked_by if str(dep).strip()])
                if blocker_text:
                    line += f" (blocked by {blocker_text})"
            if task_id:
                line += f" [id:{task_id}]"
            lines.append(line)
        summary = self._summary(entries)
        return "\n".join([summary, *lines]) if lines else summary

    def render(self, content: str, tool_data: Any) -> None:
        entries = self._entries_from_tool_data(tool_data)
        if not entries:
            entries = self._entries_from_storage()

        if entries:
            text = self._render_task_list_text(entries)
            TodoResultRenderer._render_text(self.console, text, "Task update")
            return

        if content:
            TodoResultRenderer._render_text(self.console, content, "Task update")
            return

        TodoResultRenderer._render_text(self.console, "", "Task update")


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
                    self.console.print(f"    {line}", markup=False)
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
                    self.console.print(f"    {line}", markup=False)
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
                self.console.print(f"    {line}", markup=False)
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
                # Use consistent 4-space indent to match the ⎿ prefix width
                self.console.print(f"    {line}", markup=False)
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

        # Render stream sections only when they contain content.
        self._render_stream_section("stdout", stdout_lines)
        self._render_stream_section("stderr", stderr_lines)

    def _render_stream_section(self, label: str, lines: List[str]) -> None:
        """Render one stream section (stdout/stderr) when lines are present."""
        if not lines:
            return
        preview = lines if self.verbose else lines[:5]
        self.console.print(f"[dim]{label}:[/]")
        for line in preview:
            self.console.print(f"    {line}", markup=False)
        if not self.verbose and len(lines) > len(preview):
            self.console.print(f"[dim]... ({len(lines) - len(preview)} more {label} lines)[/]")


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
            TaskGraphResultRenderer(console, verbose),
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
    "TaskGraphResultRenderer",
    "TodoResultRenderer",
    "ReadResultRenderer",
    "EditResultRenderer",
    "GlobResultRenderer",
    "GrepResultRenderer",
    "LSResultRenderer",
    "BashResultRenderer",
    "BashOutputParser",
]
