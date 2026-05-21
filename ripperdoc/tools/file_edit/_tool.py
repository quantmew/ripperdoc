"""File editing tool.

Allows the AI to edit files by replacing text.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import AsyncGenerator, List, Optional

from pydantic import BaseModel, Field

from ripperdoc.core.tool import (
    Tool,
    ToolUseContext,
    ToolResult,
    ToolOutput,
    ToolUseExample,
    ValidationResult,
)
from ripperdoc.utils.log import get_logger
from ripperdoc.utils.filesystem.path_ignore import check_path_for_tool
from ripperdoc.utils.file_editing import (
    atomic_write_with_fallback,
    open_locked_file,
    safe_record_snapshot,
    select_write_encoding,
)
from ripperdoc.utils.secret_detection import detect_secrets
from ripperdoc.tools.file_edit._prompt import get_edit_prompt
from ripperdoc.tools.file_edit._utils import (
    _normalize_quotes,
    detect_edit_read_encoding,
    validate_file_size,
)

logger = get_logger()


class FileEditToolInput(BaseModel):
    """Input schema for FileEditTool."""

    file_path: str = Field(description="Absolute path to the file to edit")
    old_string: str = Field(description="The text to replace (must match exactly)")
    new_string: str = Field(description="The text to replace it with")
    replace_all: bool = Field(
        default=False,
        description="Replace all occurrences of old_string (default: false, only first)",
    )


class FileEditToolOutput(BaseModel):
    """Output from file editing."""

    file_path: str
    replacements_made: int
    success: bool
    message: str
    additions: int = 0
    deletions: int = 0
    diff_lines: List[str] = []
    diff_with_line_numbers: List[str] = []


class FileEditTool(Tool[FileEditToolInput, FileEditToolOutput]):
    """Tool for editing files."""

    @property
    def name(self) -> str:
        return "Edit"

    async def description(self) -> str:
        return """Edit a file by replacing exact string matches. The old_string must
match exactly (including whitespace and indentation)."""

    @property
    def input_schema(self) -> type[FileEditToolInput]:
        return FileEditToolInput

    def input_examples(self) -> List[ToolUseExample]:
        return [
            ToolUseExample(
                description="Rename a function definition once",
                example={
                    "file_path": "/repo/src/app.py",
                    "old_string": "def old_name(",
                    "new_string": "def new_name(",
                    "replace_all": False,
                },
            ),
            ToolUseExample(
                description="Replace every occurrence of a constant across a file",
                example={
                    "file_path": "/repo/src/config.ts",
                    "old_string": 'API_BASE = "http://localhost"',
                    "new_string": 'API_BASE = "https://api.example.com"',
                    "replace_all": True,
                },
            ),
        ]

    async def prompt(self, yolo_mode: bool = False) -> str:
        return get_edit_prompt()

    def is_read_only(self) -> bool:
        return False

    def is_concurrency_safe(self) -> bool:
        return False

    def needs_permissions(self, _input_data: Optional[FileEditToolInput] = None) -> bool:
        return True

    async def validate_input(
        self,
        input_data: FileEditToolInput,
        context: Optional[ToolUseContext] = None,
    ) -> ValidationResult:
        if input_data.old_string == input_data.new_string:
            return ValidationResult(
                result=False, message="old_string and new_string must be different"
            )

        path = Path(input_data.file_path)
        if path.exists() and path.is_dir():
            return ValidationResult(
                result=False, message=f"Path is a directory, not a file: {path}"
            )

        if not path.exists():
            return ValidationResult(
                result=False,
                message=f"File does not exist: {path}. Use Write to create new files.",
            )

        # Check file access/ignore
        should_proceed, warning_msg = check_path_for_tool(
            path, tool_name="Edit", warn_only=False, warn_on_gitignore=False
        )
        if not should_proceed:
            return ValidationResult(
                result=False, message=warning_msg or "Edit not allowed for this path"
            )

        # Size check
        size_error = validate_file_size(input_data.file_path)
        if size_error:
            return ValidationResult(result=False, message=size_error)

        # Check file has been read
        if context:
            file_state_cache = getattr(context, "file_state_cache", {})
            abs_path = os.path.abspath(input_data.file_path)
            file_snapshot = file_state_cache.get(abs_path)

            if not file_snapshot:
                return ValidationResult(
                    result=False,
                    message="File has not been read yet. Read it first before editing.",
                )

            try:
                current_mtime = os.path.getmtime(abs_path)
                if current_mtime > file_snapshot.timestamp:
                    return ValidationResult(
                        result=False,
                        message="File has been modified since read. Read it again before editing.",
                    )
            except OSError:
                pass

        return ValidationResult(result=True)

    def render_result_for_assistant(self, output: FileEditToolOutput) -> str:
        return output.message

    def render_tool_use_message(
        self,
        input_data: FileEditToolInput,
        verbose: bool = False,
    ) -> str:
        return f"Editing: {input_data.file_path}"

    def _get_canonical_path(self, file_path: str) -> str:
        return os.path.abspath(file_path)

    async def call(
        self,
        input_data: FileEditToolInput,
        context: ToolUseContext,
    ) -> AsyncGenerator[ToolOutput, None]:
        file_path = input_data.file_path
        abs_path = self._get_canonical_path(file_path)
        file_state_cache = getattr(context, "file_state_cache", {})
        file_snapshot = file_state_cache.get(abs_path)

        resolved_path = Path(abs_path)

        file_encoding = detect_edit_read_encoding(str(resolved_path))

        # Normalize quotes for matching
        normalized_old = _normalize_quotes(input_data.old_string)
        normalized_new = _normalize_quotes(input_data.new_string)

        original_content: str = ""
        updated_content: str = ""
        replacements_made = 0

        try:
            with open_locked_file(resolved_path, file_encoding) as (
                handle,
                pre_lock_mtime,
                post_lock_mtime,
            ):
                if pre_lock_mtime is not None and post_lock_mtime is not None:
                    if post_lock_mtime > pre_lock_mtime:
                        output = FileEditToolOutput(
                            file_path=file_path,
                            replacements_made=0,
                            success=False,
                            message="File was modified while acquiring lock. Please retry.",
                        )
                        yield ToolResult(
                            data=output,
                            result_for_assistant=self.render_result_for_assistant(output),
                        )
                        return

                if file_snapshot and post_lock_mtime is not None:
                    if post_lock_mtime > file_snapshot.timestamp:
                        output = FileEditToolOutput(
                            file_path=file_path,
                            replacements_made=0,
                            success=False,
                            message="File has been modified since read. Read it again before editing.",
                        )
                        yield ToolResult(
                            data=output,
                            result_for_assistant=self.render_result_for_assistant(output),
                        )
                        return

                original_content = handle.read()

                # Try exact match first, then normalized quotes
                occurrences = original_content.count(input_data.old_string)
                if occurrences == 0:
                    occurrences = original_content.count(normalized_old)
                    search_string = normalized_old
                    replacement = normalized_new
                else:
                    search_string = input_data.old_string
                    replacement = input_data.new_string

                if occurrences == 0:
                    output = FileEditToolOutput(
                        file_path=file_path,
                        replacements_made=0,
                        success=False,
                        message=f"String not found in file: {input_data.old_string!r}",
                    )
                    yield ToolResult(
                        data=output,
                        result_for_assistant=self.render_result_for_assistant(output),
                    )
                    return

                if not input_data.replace_all and occurrences > 1:
                    output = FileEditToolOutput(
                        file_path=file_path,
                        replacements_made=0,
                        success=False,
                        message=(
                            f"Found {occurrences} occurrences. "
                            "Provide a more specific string or set replace_all=True."
                        ),
                    )
                    yield ToolResult(
                        data=output,
                        result_for_assistant=self.render_result_for_assistant(output),
                    )
                    return

                if input_data.replace_all:
                    updated_content = original_content.replace(search_string, replacement)
                    replacements_made = occurrences
                else:
                    updated_content = original_content.replace(search_string, replacement, 1)
                    replacements_made = 1

                if updated_content == original_content:
                    output = FileEditToolOutput(
                        file_path=file_path,
                        replacements_made=0,
                        success=False,
                        message="Edit produced no changes.",
                    )
                    yield ToolResult(
                        data=output,
                        result_for_assistant=self.render_result_for_assistant(output),
                    )
                    return

                # Detect secrets
                for line in updated_content.splitlines():
                    secret_type = detect_secrets(line)
                    if secret_type:
                        output = FileEditToolOutput(
                            file_path=file_path,
                            replacements_made=0,
                            success=False,
                            message=(
                                f"Edit would introduce a potential {secret_type} secret. "
                                "Please remove secrets from the content before editing."
                            ),
                        )
                        yield ToolResult(
                            data=output,
                            result_for_assistant=self.render_result_for_assistant(output),
                        )
                        return

                write_encoding = select_write_encoding(
                    file_encoding,
                    updated_content,
                    resolved_path,
                    log_prefix="[file_edit_tool]",
                )
                write_error = atomic_write_with_fallback(
                    handle,
                    resolved_path,
                    updated_content,
                    write_encoding,
                    original_content,
                    temp_prefix=".ripperdoc_edit_",
                    log_prefix="[file_edit_tool]",
                    conflict_message="File was modified during atomic write fallback. Please retry.",
                )
                if write_error:
                    output = FileEditToolOutput(
                        file_path=file_path,
                        replacements_made=0,
                        success=False,
                        message=write_error,
                    )
                    yield ToolResult(
                        data=output,
                        result_for_assistant=self.render_result_for_assistant(output),
                    )
                    return
        except (OSError, IOError, PermissionError, UnicodeDecodeError) as exc:
            logger.warning(
                "[file_edit_tool] Error reading file before edit: %s: %s",
                type(exc).__name__,
                exc,
                extra={"file_path": file_path},
            )
            output = FileEditToolOutput(
                file_path=file_path,
                replacements_made=0,
                success=False,
                message=f"Error reading file: {exc}",
            )
            yield ToolResult(
                data=output,
                result_for_assistant=self.render_result_for_assistant(output),
            )
            return

        safe_record_snapshot(
            abs_path,
            updated_content,
            file_state_cache,
            log_prefix="[file_edit_tool]",
        )

        import difflib
        old_lines = original_content.splitlines(keepends=True)
        new_lines = updated_content.splitlines(keepends=True)
        diff = list(
            difflib.unified_diff(
                old_lines, new_lines,
                fromfile=file_path, tofile=file_path, lineterm="",
            )
        )
        additions = sum(1 for line in diff if line.startswith("+") and not line.startswith("+++"))
        deletions = sum(1 for line in diff if line.startswith("-") and not line.startswith("---"))
        diff_lines = [line for line in diff[2:]]

        output = FileEditToolOutput(
            file_path=file_path,
            replacements_made=replacements_made,
            success=True,
            message=f"Applied edit with {replacements_made} replacement(s) to {file_path}",
            additions=additions,
            deletions=deletions,
            diff_lines=diff_lines,
        )

        yield ToolResult(
            data=output,
            result_for_assistant=self.render_result_for_assistant(output),
        )
