"""Multi-edit tool.

Allows performing multiple exact string replacements in a single file atomically.
"""

import difflib
import os
from pathlib import Path
from typing import AsyncGenerator, Optional, List
from textwrap import dedent
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
from ripperdoc.utils.file_watch import record_snapshot

logger = get_logger()


DEFAULT_ACTION = "Edit"
TOOL_NAME_READ = "Read"
NOTEBOOK_EDIT_TOOL_NAME = "NotebookEdit"

MULTI_EDIT_DESCRIPTION = dedent(
    f"""\
    This is a tool for making multiple edits to a single file in one operation. It is built on top of the {DEFAULT_ACTION} tool and allows you to perform multiple find-and-replace operations efficiently. Prefer this tool over the {DEFAULT_ACTION} tool when you need to make multiple edits to the same file.

    Before using this tool:

    1. Use the {TOOL_NAME_READ} tool to understand the file's contents and context
    2. Verify the directory path is correct

    To make multiple file edits, provide the following:
    1. file_path: The absolute path to the file to modify (must be absolute, not relative)
    2. edits: An array of edit operations to perform, where each edit contains:
       - old_string: The text to replace (must match the file contents exactly, including all whitespace and indentation)
       - new_string: The edited text to replace the old_string
       - replace_all: Replace all occurences of old_string. This parameter is optional and defaults to false.

    IMPORTANT:
    - All edits are applied in sequence, in the order they are provided
    - Each edit operates on the result of the previous edit
    - All edits must be valid for the operation to succeed - if any edit fails, none will be applied
    - This tool is ideal when you need to make several changes to different parts of the same file
    - For Jupyter notebooks (.ipynb files), use the {NOTEBOOK_EDIT_TOOL_NAME} instead

    CRITICAL REQUIREMENTS:
    1. All edits follow the same requirements as the single Edit tool
    2. The edits are atomic - either all succeed or none are applied
    3. Plan your edits carefully to avoid conflicts between sequential operations

    WARNING:
    - The tool will fail if edits.old_string doesn't match the file contents exactly (including whitespace)
    - The tool will fail if edits.old_string and edits.new_string are the same
    - Since edits are applied in sequence, ensure that earlier edits don't affect the text that later edits are trying to find

    When making edits:
    - Ensure all edits result in idiomatic, correct code
    - Do not leave the code in a broken state
    - Always use absolute file paths (starting with /)
    - Only use emojis if the user explicitly requests it. Avoid adding emojis to files unless asked.
    - Use replace_all for replacing and renaming strings across the file. This parameter is useful if you want to rename a variable for instance.

    If you want to create a new file, use:
    - A new file path, including dir name if needed
    - First edit: empty old_string and the new file's contents as new_string
    - Subsequent edits: normal edit operations on the created content"""
)


class EditOperation(BaseModel):
    """Single edit operation."""

    old_string: str = Field(description="The text to replace (must match exactly)")
    new_string: str = Field(description="The text to replace it with")
    replace_all: bool = Field(
        default=False,
        description="Replace all occurrences of old_string (default: false, only first)",
    )


class MultiEditToolInput(BaseModel):
    """Input schema for MultiEditTool."""

    file_path: str = Field(description="Absolute path to the file to edit")
    edits: List[EditOperation] = Field(
        description="Array of edit operations to apply sequentially",
        min_length=1,
    )


class MultiEditToolOutput(BaseModel):
    """Output from multi-edit."""

    file_path: str
    replacements_made: int
    success: bool
    message: str
    additions: int = 0
    deletions: int = 0
    diff_lines: list[str] = []
    diff_with_line_numbers: list[str] = []
    applied_edits: List[EditOperation] = []
    created: bool = False


class MultiEditTool(Tool[MultiEditToolInput, MultiEditToolOutput]):
    """Tool for applying multiple edits to a file atomically."""

    @property
    def name(self) -> str:
        return "MultiEdit"

    async def description(self) -> str:
        return MULTI_EDIT_DESCRIPTION

    @property
    def input_schema(self) -> type[MultiEditToolInput]:
        return MultiEditToolInput

    def input_examples(self) -> List[ToolUseExample]:
        return [
            ToolUseExample(
                description="Apply multiple replacements in one pass",
                example={
                    "file_path": "/repo/src/app.py",
                    "edits": [
                        {"old_string": "DEBUG = True", "new_string": "DEBUG = False"},
                        {"old_string": "old_fn(", "new_string": "new_fn("},
                    ],
                },
            ),
            ToolUseExample(
                description="Create a new file then adjust content",
                example={
                    "file_path": "/repo/docs/notes.txt",
                    "edits": [
                        {"old_string": "", "new_string": "Line one\nLine two\n"},
                        {"old_string": "Line two", "new_string": "Second line"},
                    ],
                },
            ),
        ]

    async def prompt(self, yolo_mode: bool = False) -> str:
        return MULTI_EDIT_DESCRIPTION

    def is_read_only(self) -> bool:
        return False

    def is_concurrency_safe(self) -> bool:
        return False

    def needs_permissions(self, input_data: Optional[MultiEditToolInput] = None) -> bool:
        return True

    async def validate_input(
        self,
        input_data: MultiEditToolInput,
        context: Optional[ToolUseContext] = None,
    ) -> ValidationResult:
        path = Path(input_data.file_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        resolved_path = str(path.resolve())

        # Ensure edits differ.
        for edit in input_data.edits:
            if edit.old_string == edit.new_string:
                return ValidationResult(
                    result=False,
                    message="old_string and new_string must be different",
                    error_code=1,
                )

        # If the file exists, ensure it is not a directory.
        if path.exists() and path.is_dir():
            return ValidationResult(
                result=False,
                message=f"Path is a directory, not a file: {path}",
                error_code=2,
            )

        # Check if this is a file creation (first edit has empty old_string)
        is_creation = (
            not path.exists() and len(input_data.edits) > 0 and input_data.edits[0].old_string == ""
        )

        # If file exists, check if it has been read before editing
        if path.exists() and not is_creation:
            file_state_cache = getattr(context, "file_state_cache", {}) if context else {}
            file_snapshot = file_state_cache.get(resolved_path)

            if not file_snapshot:
                return ValidationResult(
                    result=False,
                    message="File has not been read yet. Read it first before editing.",
                    error_code=3,
                )

            # Check if file has been modified since it was read
            try:
                current_mtime = os.path.getmtime(resolved_path)
                if current_mtime > file_snapshot.timestamp:
                    return ValidationResult(
                        result=False,
                        message="File has been modified since read, either by the user or by a linter. "
                        "Read it again before attempting to edit it.",
                        error_code=4,
                    )
            except OSError:
                pass  # File mtime check failed, proceed anyway

        return ValidationResult(result=True)

    def render_result_for_assistant(self, output: MultiEditToolOutput) -> str:
        return output.message

    def render_tool_use_message(
        self,
        input_data: MultiEditToolInput,
        verbose: bool = False,
    ) -> str:
        return f"Multi-editing: {input_data.file_path} ({len(input_data.edits)} edits)"

    def _apply_edits(self, content: str, edits: List[EditOperation]) -> tuple[str, int]:
        """Apply edits in-memory. Raises ValueError on failure."""
        current = content
        total_replacements = 0

        for edit in edits:
            # Creation workflow: old_string empty means write provided content when file is empty.
            if edit.old_string == "":
                if current != "":
                    raise ValueError(
                        "old_string was empty but the file already has content; "
                        "use replace_all=true with an explicit old_string instead."
                    )
                current = edit.new_string
                total_replacements += 1 if edit.new_string else 0
                continue

            occurrences = current.count(edit.old_string)
            if occurrences == 0:
                raise ValueError(f"String not found: {edit.old_string!r}")

            if not edit.replace_all and occurrences > 1:
                raise ValueError(
                    f"String appears {occurrences} times. "
                    "Provide a unique string or set replace_all=true."
                )

            if edit.replace_all:
                current = current.replace(edit.old_string, edit.new_string)
                total_replacements += occurrences
            else:
                current = current.replace(edit.old_string, edit.new_string, 1)
                total_replacements += 1

        return current, total_replacements

    def _build_diff(
        self, original: str, updated: str, file_path: str
    ) -> tuple[list[str], list[str], int, int]:
        old_lines = original.splitlines(keepends=True)
        new_lines = updated.splitlines(keepends=True)

        diff = list(
            difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile=file_path,
                tofile=file_path,
                lineterm="",
            )
        )

        additions = sum(1 for line in diff if line.startswith("+") and not line.startswith("+++"))
        deletions = sum(1 for line in diff if line.startswith("-") and not line.startswith("---"))

        diff_lines = [line for line in diff[3:]]  # skip headers

        diff_with_line_numbers: list[str] = []
        old_line_num: Optional[int] = None
        new_line_num: Optional[int] = None

        for line in diff[3:]:
            if line.startswith("@@"):
                import re

                match = re.search(r"@@ -(\d+),(\d+) \+(\d+),(\d+) @@", line)
                if match:
                    old_line_num = int(match.group(1))
                    new_line_num = int(match.group(3))
                    diff_with_line_numbers.append(f"      [dim]{line}[/dim]")
            elif line.startswith("+") and not line.startswith("+++"):
                if new_line_num is not None:
                    diff_with_line_numbers.append(
                        f"      [green]{new_line_num:6d} + {line[1:]}[/green]"
                    )
                    new_line_num += 1
                else:
                    diff_with_line_numbers.append(f"      [green]{line}[/green]")
            elif line.startswith("-") and not line.startswith("---"):
                if old_line_num is not None:
                    diff_with_line_numbers.append(
                        f"      [red]{old_line_num:6d} - {line[1:]}[/red]"
                    )
                    old_line_num += 1
                else:
                    diff_with_line_numbers.append(f"      [red]{line}[/red]")
            elif line.strip():
                if old_line_num is not None and new_line_num is not None:
                    diff_with_line_numbers.append(
                        f"      {old_line_num:6d}   {new_line_num:6d}   {line}"
                    )
                    old_line_num += 1
                    new_line_num += 1
                else:
                    diff_with_line_numbers.append(f"      {line}")

        return diff_lines, diff_with_line_numbers, additions, deletions

    async def call(
        self,
        input_data: MultiEditToolInput,
        context: ToolUseContext,
    ) -> AsyncGenerator[ToolOutput, None]:
        """Apply multiple edits atomically."""
        file_path = Path(input_data.file_path).expanduser()
        if not file_path.is_absolute():
            file_path = Path.cwd() / file_path
        file_path = file_path.resolve()

        existing = file_path.exists()
        original_content = ""
        try:
            if existing:
                original_content = file_path.read_text(encoding="utf-8")
        except (OSError, IOError, PermissionError) as exc:
            # pragma: no cover - unlikely permission issue
            logger.warning(
                "[multi_edit_tool] Error reading file before edits: %s: %s",
                type(exc).__name__,
                exc,
                extra={"file_path": str(file_path)},
            )
            output = MultiEditToolOutput(
                file_path=str(file_path),
                replacements_made=0,
                success=False,
                message=f"Error reading file: {exc}",
            )
            yield ToolResult(
                data=output, result_for_assistant=self.render_result_for_assistant(output)
            )
            return

        applied = input_data.edits
        try:
            updated_content, total_replacements = self._apply_edits(original_content, applied)
        except ValueError as exc:
            output = MultiEditToolOutput(
                file_path=str(file_path),
                replacements_made=0,
                success=False,
                message=str(exc),
                applied_edits=applied,
                created=not existing and original_content == "",
            )
            yield ToolResult(
                data=output, result_for_assistant=self.render_result_for_assistant(output)
            )
            return

        if updated_content == original_content:
            output = MultiEditToolOutput(
                file_path=str(file_path),
                replacements_made=0,
                success=False,
                message="Edits produced no changes.",
                applied_edits=applied,
                created=not existing and original_content == "",
            )
            yield ToolResult(
                data=output, result_for_assistant=self.render_result_for_assistant(output)
            )
            return

        # Ensure parent exists (validated earlier) and write the file.
        file_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            file_path.write_text(updated_content, encoding="utf-8")
            try:
                record_snapshot(
                    str(file_path),
                    updated_content,
                    getattr(context, "file_state_cache", {}),
                )
            except (OSError, IOError, RuntimeError) as exc:
                logger.warning(
                    "[multi_edit_tool] Failed to record file snapshot: %s: %s",
                    type(exc).__name__,
                    exc,
                    extra={"file_path": str(file_path)},
                )
        except (OSError, IOError, PermissionError, UnicodeDecodeError) as exc:
            logger.warning(
                "[multi_edit_tool] Error writing edited file: %s: %s",
                type(exc).__name__,
                exc,
                extra={"file_path": str(file_path)},
            )
            output = MultiEditToolOutput(
                file_path=str(file_path),
                replacements_made=0,
                success=False,
                message=f"Error writing file: {exc}",
                applied_edits=applied,
                created=not existing and original_content == "",
            )
            yield ToolResult(
                data=output, result_for_assistant=self.render_result_for_assistant(output)
            )
            return

        diff_lines, diff_with_line_numbers, additions, deletions = self._build_diff(
            original_content, updated_content, str(file_path)
        )

        output = MultiEditToolOutput(
            file_path=str(file_path),
            replacements_made=total_replacements,
            success=True,
            message=(
                f"Applied {len(applied)} edit(s) with {total_replacements} replacement(s) "
                f"to {file_path}"
            ),
            additions=additions,
            deletions=deletions,
            diff_lines=diff_lines,
            diff_with_line_numbers=diff_with_line_numbers,
            applied_edits=applied,
            created=not existing and original_content == "",
        )

        yield ToolResult(
            data=output,
            result_for_assistant=self.render_result_for_assistant(output),
        )
