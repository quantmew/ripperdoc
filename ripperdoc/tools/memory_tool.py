"""Claude-style memory tool for persistent cross-session notes."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import AsyncGenerator, Callable, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from ripperdoc.core.tool import (
    Tool,
    ToolOutput,
    ToolResult,
    ToolUseContext,
    ToolUseExample,
    ValidationResult,
)
from ripperdoc.utils.log import get_logger
from ripperdoc.utils.memory import AUTO_MEMORY_FILE_NAME, auto_memory_directory_path

logger = get_logger()

MemoryCommand = Literal["view", "create", "str_replace", "insert", "delete", "rename"]
MAX_VIEW_CHARS = 100_000


class MemoryToolInput(BaseModel):
    """Input schema for MemoryTool."""

    command: MemoryCommand = Field(
        description=(
            "Memory command to run: view, create, str_replace, insert, delete, or rename."
        )
    )
    path: Optional[str] = Field(
        default=None,
        description=(
            "Path inside the memory directory. Relative paths are preferred. "
            "For command=view, omit path or use '.' to list memory files."
        ),
    )
    old_path: Optional[str] = Field(
        default=None,
        description="Source path for command=rename (Anthropic memory command shape).",
    )
    view_range: Optional[List[int]] = Field(
        default=None,
        description=(
            "Optional [start_line, end_line] inclusive line range for command=view "
            "(1-based indexing)."
        ),
    )
    file_text: Optional[str] = Field(
        default=None,
        description="File content for command=create (Anthropic memory command shape).",
    )
    content: Optional[str] = Field(
        default=None,
        description="File content for command=create.",
    )
    old_str: Optional[str] = Field(
        default=None,
        description="Exact string to replace for command=str_replace.",
    )
    new_str: Optional[str] = Field(
        default=None,
        description="Replacement text for command=str_replace or inserted text for command=insert.",
    )
    insert_line: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "1-based line number for command=insert. Inserts before this line; "
            "use line_count+1 to append."
        ),
    )
    insert_text: Optional[str] = Field(
        default=None,
        description="Text to insert for command=insert (Anthropic memory command shape).",
    )
    new_path: Optional[str] = Field(
        default=None,
        description="Destination path for command=rename (inside the memory directory).",
    )
    replace_all: bool = Field(
        default=False,
        description=(
            "When true and command=str_replace, replace all matches. "
            "When false, requires exactly one match."
        ),
    )
    max_entries: int = Field(
        default=200,
        ge=1,
        le=2000,
        description="Maximum entries returned when viewing a directory.",
    )
    model_config = ConfigDict(extra="ignore")


class MemoryToolOutput(BaseModel):
    """Output for MemoryTool."""

    command: MemoryCommand
    success: bool
    message: str
    path: Optional[str] = None
    new_path: Optional[str] = None
    content: Optional[str] = None
    entries: List[str] = Field(default_factory=list)
    replacements_made: int = 0
    line_count: Optional[int] = None


class MemoryTool(Tool[MemoryToolInput, MemoryToolOutput]):
    """Persistent memory file operations aligned with Claude memory commands."""

    def __init__(
        self,
        *,
        project_path: Optional[Path | str] = None,
        memory_dir: Optional[Path | str] = None,
    ) -> None:
        super().__init__()
        self._project_path = Path(project_path).resolve() if project_path else None
        self._memory_dir_override = Path(memory_dir).resolve() if memory_dir else None

    @property
    def name(self) -> str:
        return "Memory"

    async def description(self) -> str:
        return (
            "Persistent memory file tool. Supports command=view/create/str_replace/insert/delete/"
            "rename over files in the session memory directory."
        )

    @property
    def input_schema(self) -> type[MemoryToolInput]:
        return MemoryToolInput

    def input_examples(self) -> List[ToolUseExample]:
        return [
            ToolUseExample(
                description="List current memory files",
                example={"command": "view"},
            ),
            ToolUseExample(
                description="Create a topic memory file",
                example={
                    "command": "create",
                    "path": "patterns.md",
                    "file_text": "# Patterns\n\n- Prefer ripgrep for repository search.\n",
                },
            ),
            ToolUseExample(
                description="Update an existing memory fact",
                example={
                    "command": "str_replace",
                    "path": "MEMORY.md",
                    "old_str": "- Language: JavaScript",
                    "new_str": "- Language: TypeScript",
                },
            ),
        ]

    def input_param_aliases(self) -> dict[str, str]:
        return {
            "old_string": "old_str",
            "new_string": "new_str",
            "oldString": "old_str",
            "newString": "new_str",
            "source_path": "old_path",
            "target_path": "new_path",
            "line": "insert_line",
        }

    async def prompt(self, yolo_mode: bool = False) -> str:  # noqa: ARG002
        memory_root = self._memory_root()
        return (
            "Use this tool to manage persistent memory files across sessions.\n\n"
            "Commands:\n"
            "- view: show a file's content, or list directory entries when path is omitted.\n"
            "- create: create a new file with content (file_text).\n"
            "- str_replace: replace exact text in a file.\n"
            "- insert: insert text (insert_text) before a specific line number.\n"
            "- delete: remove a file or subdirectory.\n"
            "- rename: move/rename a file or subdirectory (old_path -> new_path).\n\n"
            f"All paths are restricted to the memory directory: {memory_root}\n"
            f"Keep `{AUTO_MEMORY_FILE_NAME}` concise and use topic files for details."
        )

    def is_read_only(self) -> bool:
        return False

    def is_concurrency_safe(self) -> bool:
        return False

    def needs_permissions(self, input_data: Optional[MemoryToolInput] = None) -> bool:
        if input_data is None:
            return True
        return input_data.command != "view"

    async def validate_input(
        self, input_data: MemoryToolInput, context: Optional[ToolUseContext] = None
    ) -> ValidationResult:
        del context
        try:
            path_for_command = self._path_for_command(input_data)
            if path_for_command is not None:
                self._resolve_path(path_for_command, allow_root=input_data.command == "view")
            elif input_data.command != "view":
                return ValidationResult(result=False, message=f"command={input_data.command} requires path.")

            if input_data.command == "create":
                if self._create_text(input_data) is None:
                    return ValidationResult(
                        result=False,
                        message="command=create requires file_text (or content).",
                    )

            if input_data.command == "str_replace":
                if not input_data.old_str:
                    return ValidationResult(
                        result=False,
                        message="command=str_replace requires old_str.",
                    )
                if input_data.new_str is None:
                    return ValidationResult(
                        result=False,
                        message="command=str_replace requires new_str.",
                    )

            if input_data.command == "insert":
                if input_data.insert_line is None:
                    return ValidationResult(
                        result=False,
                        message="command=insert requires insert_line.",
                    )
                if self._insert_text(input_data) is None:
                    return ValidationResult(
                        result=False,
                        message="command=insert requires insert_text (or new_str).",
                    )

            if input_data.command == "rename":
                if not input_data.new_path:
                    return ValidationResult(
                        result=False,
                        message="command=rename requires new_path.",
                    )
                self._resolve_path(input_data.new_path, allow_root=False)
                source_path = self._path_for_command(input_data)
                if not source_path:
                    return ValidationResult(
                        result=False,
                        message="command=rename requires old_path (or path).",
                    )

            if input_data.command == "view" and input_data.view_range is not None:
                if len(input_data.view_range) != 2:
                    return ValidationResult(
                        result=False,
                        message="view_range must contain exactly two integers: [start_line, end_line].",
                    )
                start, end = input_data.view_range
                if start < 1 or end < 1:
                    return ValidationResult(
                        result=False,
                        message="view_range lines must be >= 1.",
                    )
                if start > end:
                    return ValidationResult(
                        result=False,
                        message="view_range start_line must be <= end_line.",
                    )

            return ValidationResult(result=True)
        except ValueError as exc:
            return ValidationResult(result=False, message=str(exc))

    def render_result_for_assistant(self, output: MemoryToolOutput) -> str:
        if not output.success:
            return output.message

        if output.command == "view":
            if output.content is not None:
                return (
                    f"{output.message}\n\n"
                    f"---\n{output.content}\n---"
                )
            if output.entries:
                lines = "\n".join(f"- {entry}" for entry in output.entries)
                return f"{output.message}\n\n{lines}"
        return output.message

    def render_tool_use_message(self, input_data: MemoryToolInput, verbose: bool = False) -> str:
        del verbose
        if input_data.path:
            return f"Memory {input_data.command}: {input_data.path}"
        return f"Memory {input_data.command}"

    async def call(
        self, input_data: MemoryToolInput, context: ToolUseContext
    ) -> AsyncGenerator[ToolOutput, None]:
        del context
        try:
            output = self._execute(input_data)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.exception(
                "[memory_tool] Unexpected error running command",
                extra={"command": input_data.command},
            )
            output = MemoryToolOutput(
                command=input_data.command,
                success=False,
                message=f"Memory command failed: {type(exc).__name__}: {exc}",
            )

        yield ToolResult(data=output, result_for_assistant=self.render_result_for_assistant(output))

    def _memory_root(self) -> Path:
        root = self._memory_dir_override or auto_memory_directory_path(self._project_path)
        root.mkdir(parents=True, exist_ok=True)
        return root.resolve()

    def _resolve_path(self, raw_path: str | None, *, allow_root: bool) -> Path:
        root = self._memory_root()
        token = "." if raw_path is None else str(raw_path).strip()
        if not token:
            token = "."

        if token in {".", "/"}:
            resolved = root
        else:
            candidate = Path(token).expanduser()
            resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()

        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"Path '{token}' is outside memory directory '{root}'."
            ) from exc

        if resolved == root and not allow_root:
            raise ValueError("This command requires a specific path inside the memory directory.")

        return resolved

    def _display_path(self, path: Path) -> str:
        root = self._memory_root()
        try:
            rel = path.resolve().relative_to(root)
        except ValueError:
            return str(path)
        if str(rel) == ".":
            return "."
        return rel.as_posix()

    def _read_text(self, path: Path) -> str:
        return path.read_text(encoding="utf-8", errors="ignore")

    def _write_text(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def _path_for_command(self, input_data: MemoryToolInput) -> Optional[str]:
        if input_data.command == "rename":
            return input_data.old_path or input_data.path
        return input_data.path

    def _create_text(self, input_data: MemoryToolInput) -> Optional[str]:
        if input_data.file_text is not None:
            return input_data.file_text
        return input_data.content

    def _insert_text(self, input_data: MemoryToolInput) -> Optional[str]:
        if input_data.insert_text is not None:
            return input_data.insert_text
        return input_data.new_str

    def _view(self, input_data: MemoryToolInput) -> MemoryToolOutput:
        target = self._resolve_path(self._path_for_command(input_data), allow_root=True)
        display_path = self._display_path(target)

        if not target.exists():
            return MemoryToolOutput(
                command="view",
                success=False,
                path=display_path,
                message=f"Path not found: {display_path}",
            )

        if target.is_file():
            content = self._read_text(target)
            lines = content.splitlines()
            line_count = len(lines)
            if input_data.view_range:
                start, end = input_data.view_range
                start_idx = min(start, line_count + 1) - 1
                end_idx = min(end, line_count)
                ranged = lines[start_idx:end_idx]
                content = "\n".join(ranged)
                if ranged and not content.endswith("\n"):
                    content += "\n"
            if len(content) > MAX_VIEW_CHARS:
                content = (
                    content[:MAX_VIEW_CHARS]
                    + "\n\n[truncated: output exceeded memory view limit]"
                )
            return MemoryToolOutput(
                command="view",
                success=True,
                path=display_path,
                line_count=line_count,
                content=content,
                message=f"Viewed memory file: {display_path}",
            )

        root = self._memory_root()
        entries: list[str] = []
        for entry in sorted(target.rglob("*")):
            if len(entries) >= input_data.max_entries:
                break
            rel = entry.relative_to(root).as_posix()
            entries.append(f"{rel}/" if entry.is_dir() else rel)

        if not entries:
            return MemoryToolOutput(
                command="view",
                success=True,
                path=display_path,
                entries=[],
                message=f"Memory directory is empty: {display_path}",
            )

        return MemoryToolOutput(
            command="view",
            success=True,
            path=display_path,
            entries=entries,
            message=f"Listed {len(entries)} memory entries under {display_path}",
        )

    def _create(self, input_data: MemoryToolInput) -> MemoryToolOutput:
        target = self._resolve_path(self._path_for_command(input_data), allow_root=False)
        display_path = self._display_path(target)
        if target.exists():
            return MemoryToolOutput(
                command="create",
                success=False,
                path=display_path,
                message=f"Path already exists: {display_path}",
            )

        content = self._create_text(input_data) or ""
        self._write_text(target, content)
        return MemoryToolOutput(
            command="create",
            success=True,
            path=display_path,
            line_count=len(content.splitlines()),
            message=f"Created memory file: {display_path}",
        )

    def _str_replace(self, input_data: MemoryToolInput) -> MemoryToolOutput:
        target = self._resolve_path(self._path_for_command(input_data), allow_root=False)
        display_path = self._display_path(target)
        if not target.exists() or not target.is_file():
            return MemoryToolOutput(
                command="str_replace",
                success=False,
                path=display_path,
                message=f"Memory file not found: {display_path}",
            )

        old_str = input_data.old_str or ""
        new_str = input_data.new_str or ""
        content = self._read_text(target)
        occurrences = content.count(old_str)
        if occurrences == 0:
            return MemoryToolOutput(
                command="str_replace",
                success=False,
                path=display_path,
                message=f"String not found in {display_path}",
            )
        if occurrences > 1 and not input_data.replace_all:
            return MemoryToolOutput(
                command="str_replace",
                success=False,
                path=display_path,
                message=(
                    f"String appears {occurrences} times in {display_path}. "
                    "Provide a unique old_str or set replace_all=true."
                ),
            )

        if input_data.replace_all:
            updated = content.replace(old_str, new_str)
            replacements = occurrences
        else:
            updated = content.replace(old_str, new_str, 1)
            replacements = 1
        self._write_text(target, updated)

        return MemoryToolOutput(
            command="str_replace",
            success=True,
            path=display_path,
            replacements_made=replacements,
            line_count=len(updated.splitlines()),
            message=f"Updated {display_path} ({replacements} replacement(s)).",
        )

    def _insert(self, input_data: MemoryToolInput) -> MemoryToolOutput:
        target = self._resolve_path(self._path_for_command(input_data), allow_root=False)
        display_path = self._display_path(target)
        if not target.exists() or not target.is_file():
            return MemoryToolOutput(
                command="insert",
                success=False,
                path=display_path,
                message=f"Memory file not found: {display_path}",
            )

        insert_line = int(input_data.insert_line or 0)
        content = self._read_text(target)
        lines = content.splitlines(keepends=True)
        max_insert_line = len(lines) + 1
        if insert_line > max_insert_line:
            return MemoryToolOutput(
                command="insert",
                success=False,
                path=display_path,
                message=(
                    f"insert_line {insert_line} is out of range for {display_path}. "
                    f"Valid range is 1..{max_insert_line}."
                ),
            )

        fragment = self._insert_text(input_data) or ""
        if fragment and not fragment.endswith("\n"):
            fragment = f"{fragment}\n"
        lines.insert(insert_line - 1, fragment)
        updated = "".join(lines)
        self._write_text(target, updated)

        return MemoryToolOutput(
            command="insert",
            success=True,
            path=display_path,
            line_count=len(updated.splitlines()),
            message=f"Inserted content at line {insert_line} in {display_path}.",
        )

    def _delete(self, input_data: MemoryToolInput) -> MemoryToolOutput:
        target = self._resolve_path(self._path_for_command(input_data), allow_root=False)
        display_path = self._display_path(target)
        if not target.exists():
            return MemoryToolOutput(
                command="delete",
                success=False,
                path=display_path,
                message=f"Path not found: {display_path}",
            )

        if target.is_dir():
            shutil.rmtree(target)
            message = f"Deleted memory directory: {display_path}"
        else:
            target.unlink()
            message = f"Deleted memory file: {display_path}"

        return MemoryToolOutput(
            command="delete",
            success=True,
            path=display_path,
            message=message,
        )

    def _rename(self, input_data: MemoryToolInput) -> MemoryToolOutput:
        source = self._resolve_path(self._path_for_command(input_data), allow_root=False)
        target = self._resolve_path(input_data.new_path, allow_root=False)
        source_display = self._display_path(source)
        target_display = self._display_path(target)

        if not source.exists():
            return MemoryToolOutput(
                command="rename",
                success=False,
                path=source_display,
                new_path=target_display,
                message=f"Path not found: {source_display}",
            )
        if target.exists():
            return MemoryToolOutput(
                command="rename",
                success=False,
                path=source_display,
                new_path=target_display,
                message=f"Destination already exists: {target_display}",
            )
        if source == target:
            return MemoryToolOutput(
                command="rename",
                success=False,
                path=source_display,
                new_path=target_display,
                message="Source and destination paths are identical.",
            )
        if source.is_dir():
            try:
                target.relative_to(source)
                return MemoryToolOutput(
                    command="rename",
                    success=False,
                    path=source_display,
                    new_path=target_display,
                    message="Cannot move a directory into itself.",
                )
            except ValueError:
                pass

        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)

        return MemoryToolOutput(
            command="rename",
            success=True,
            path=source_display,
            new_path=target_display,
            message=f"Renamed {source_display} to {target_display}.",
        )

    def _execute(self, input_data: MemoryToolInput) -> MemoryToolOutput:
        dispatch: dict[MemoryCommand, Callable[[MemoryToolInput], MemoryToolOutput]] = {
            "view": self._view,
            "create": self._create,
            "str_replace": self._str_replace,
            "insert": self._insert,
            "delete": self._delete,
            "rename": self._rename,
        }
        handler = dispatch[input_data.command]
        return handler(input_data)
