"""File editing tool.

Allows the AI to edit files by replacing text.
"""

import contextlib
import os
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Generator, List, Optional, TextIO
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
from ripperdoc.utils.path_ignore import check_path_for_tool

# Import fcntl for file locking on Unix systems
try:
    import fcntl

    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False

logger = get_logger()


@contextlib.contextmanager
def _file_lock(file_handle: TextIO, exclusive: bool = True) -> Generator[None, None, None]:
    """Acquire a file lock, with fallback for systems without fcntl.

    Args:
        file_handle: An open file handle to lock
        exclusive: If True, acquire exclusive lock; otherwise shared lock

    Yields:
        None
    """
    if not HAS_FCNTL:
        # On Windows or systems without fcntl, skip locking
        yield
        return

    lock_type = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    try:
        fcntl.flock(file_handle.fileno(), lock_type)
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(file_handle.fileno(), fcntl.LOCK_UN)


class FileEditToolInput(BaseModel):
    """Input schema for FileEditTool."""

    file_path: str = Field(description="Absolute path to the file to edit")
    old_string: str = Field(description="The text to replace (must match exactly)")
    new_string: str = Field(description="The text to replace it with")
    replace_all: bool = Field(
        default=False, description="Replace all occurrences (default: false, only first)"
    )


class FileEditToolOutput(BaseModel):
    """Output from file editing."""

    file_path: str
    replacements_made: int
    success: bool
    message: str
    additions: int = 0
    deletions: int = 0
    diff_lines: list[str] = []
    diff_with_line_numbers: list[str] = []


class FileEditTool(Tool[FileEditToolInput, FileEditToolOutput]):
    """Tool for editing files by replacing text."""

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
        return (
            "Performs exact string replacements in files.\n\n"
            "Usage:\n"
            "- You must use your `Read` tool at least once in the conversation to read the file before editing; edits will fail if you skip reading.\n"
            "- When editing text from Read output, preserve the exact indentation (tabs/spaces) as it appears AFTER the line number prefix. The line number prefix is formatted as spaces + line number + tab. Never include any part of the prefix in old_string or new_string.\n"
            "- ALWAYS prefer editing existing files in the codebase. NEVER write new files unless explicitly required.\n"
            "- Only use emojis if the user explicitly requests it. Avoid adding emojis to files unless asked.\n"
            "- The edit will FAIL if `old_string` is not unique in the file. Provide more surrounding context to make it unique or use `replace_all` to change every instance of `old_string`.\n"
            "- Use `replace_all` when replacing or renaming strings across the file (e.g., renaming a variable)."
        )

    def is_read_only(self) -> bool:
        return False

    def is_concurrency_safe(self) -> bool:
        return False

    def needs_permissions(self, input_data: Optional[FileEditToolInput] = None) -> bool:
        return True

    async def validate_input(
        self, input_data: FileEditToolInput, context: Optional[ToolUseContext] = None
    ) -> ValidationResult:
        # Check if file exists
        if not os.path.exists(input_data.file_path):
            return ValidationResult(
                result=False,
                message=f"File not found: {input_data.file_path}",
                error_code=1,
            )

        # Check if it's a file
        if not os.path.isfile(input_data.file_path):
            return ValidationResult(
                result=False,
                message=f"Path is not a file: {input_data.file_path}",
                error_code=2,
            )

        # Check that old_string and new_string are different
        if input_data.old_string == input_data.new_string:
            return ValidationResult(
                result=False,
                message="old_string and new_string must be different",
                error_code=3,
            )

        # Check if file has been read before editing
        file_state_cache = getattr(context, "file_state_cache", {}) if context else {}
        file_path = os.path.abspath(input_data.file_path)
        file_snapshot = file_state_cache.get(file_path)

        if not file_snapshot:
            return ValidationResult(
                result=False,
                message="File has not been read yet. Read it first before editing.",
                error_code=4,
            )

        # Check if file has been modified since it was read
        try:
            current_mtime = os.path.getmtime(file_path)
            if current_mtime > file_snapshot.timestamp:
                return ValidationResult(
                    result=False,
                    message="File has been modified since read, either by the user or by a linter. "
                    "Read it again before attempting to edit it.",
                    error_code=5,
                )
        except OSError:
            pass  # File mtime check failed, proceed anyway

        # Check if path is ignored (warning for edit operations)
        file_path_obj = Path(file_path)
        should_proceed, warning_msg = check_path_for_tool(
            file_path_obj, tool_name="Edit", warn_only=True
        )
        if warning_msg:
            logger.warning("[file_edit_tool] %s", warning_msg)

        return ValidationResult(result=True)

    def render_result_for_assistant(self, output: FileEditToolOutput) -> str:
        """Format output for the AI."""
        # Return simple message for AI, but include structured data for UI
        # The UI will extract the structured data from the output object
        return output.message

    def render_tool_use_message(self, input_data: FileEditToolInput, verbose: bool = False) -> str:
        """Format the tool use for display."""
        return f"Editing: {input_data.file_path}"

    async def call(
        self, input_data: FileEditToolInput, context: ToolUseContext
    ) -> AsyncGenerator[ToolOutput, None]:
        """Edit the file with TOCTOU protection."""

        abs_file_path = os.path.abspath(input_data.file_path)
        file_state_cache = getattr(context, "file_state_cache", {})
        file_snapshot = file_state_cache.get(abs_file_path)

        try:
            # Open file with exclusive lock to prevent concurrent modifications
            # Use r+ mode to get a file handle we can lock before reading
            with open(abs_file_path, "r+", encoding="utf-8") as f:
                with _file_lock(f, exclusive=True):
                    # Re-check mtime AFTER acquiring lock to close TOCTOU window
                    # This is the key fix: validate mtime while holding the lock
                    if file_snapshot:
                        try:
                            current_mtime = os.fstat(f.fileno()).st_mtime
                            if current_mtime > file_snapshot.timestamp:
                                output = FileEditToolOutput(
                                    file_path=input_data.file_path,
                                    replacements_made=0,
                                    success=False,
                                    message="File has been modified since read, either by the user "
                                    "or by a linter. Read it again before attempting to edit it.",
                                )
                                yield ToolResult(
                                    data=output,
                                    result_for_assistant=self.render_result_for_assistant(output),
                                )
                                return
                        except OSError:
                            pass  # fstat failed, proceed anyway

                    # Read content while holding the lock
                    content = f.read()

                    # Check if old_string exists
                    if input_data.old_string not in content:
                        output = FileEditToolOutput(
                            file_path=input_data.file_path,
                            replacements_made=0,
                            success=False,
                            message=f"String not found in file: {input_data.file_path}",
                        )
                        yield ToolResult(
                            data=output,
                            result_for_assistant=self.render_result_for_assistant(output),
                        )
                        return

                    # Count occurrences
                    occurrence_count = content.count(input_data.old_string)

                    # Check for ambiguity if not replace_all
                    if not input_data.replace_all and occurrence_count > 1:
                        output = FileEditToolOutput(
                            file_path=input_data.file_path,
                            replacements_made=0,
                            success=False,
                            message=f"String appears {occurrence_count} times in file. "
                            f"Either provide a unique string or use replace_all=true",
                        )
                        yield ToolResult(
                            data=output,
                            result_for_assistant=self.render_result_for_assistant(output),
                        )
                        return

                    # Perform replacement
                    if input_data.replace_all:
                        new_content = content.replace(input_data.old_string, input_data.new_string)
                        replacements = occurrence_count
                    else:
                        new_content = content.replace(
                            input_data.old_string, input_data.new_string, 1
                        )
                        replacements = 1

                    # Atomic write: write to temp file then rename
                    # This ensures the file is either fully written or not at all
                    file_dir = os.path.dirname(abs_file_path)
                    try:
                        # Create temp file in same directory to ensure same filesystem
                        fd, temp_path = tempfile.mkstemp(
                            dir=file_dir, prefix=".ripperdoc_edit_", suffix=".tmp"
                        )
                        try:
                            with os.fdopen(fd, "w", encoding="utf-8") as temp_f:
                                temp_f.write(new_content)
                            # Preserve original file permissions
                            original_stat = os.fstat(f.fileno())
                            os.chmod(temp_path, original_stat.st_mode)
                            # Atomic replace (works on Unix, best-effort on Windows)
                            os.replace(temp_path, abs_file_path)
                        except Exception:
                            # Clean up temp file on failure
                            with contextlib.suppress(OSError):
                                os.unlink(temp_path)
                            raise
                    except OSError as atomic_error:
                        # Fallback to in-place write if atomic write fails
                        # (e.g., cross-filesystem issues)
                        # Re-verify file hasn't changed before fallback write (TOCTOU protection)
                        f.seek(0)
                        current_content = f.read()
                        if current_content != content:
                            output = FileEditToolOutput(
                                file_path=input_data.file_path,
                                replacements_made=0,
                                success=False,
                                message="File was modified during atomic write fallback. Please retry.",
                            )
                            yield ToolResult(
                                data=output,
                                result_for_assistant=self.render_result_for_assistant(output),
                            )
                            return
                        f.seek(0)
                        f.truncate()
                        f.write(new_content)
                        logger.debug(
                            "[file_edit_tool] Atomic write failed, used fallback: %s",
                            atomic_error,
                        )

            # Record the new snapshot after successful edit
            try:
                record_snapshot(
                    abs_file_path,
                    new_content,
                    getattr(context, "file_state_cache", {}),
                )
            except (OSError, IOError, RuntimeError) as exc:
                logger.warning(
                    "[file_edit_tool] Failed to record file snapshot: %s: %s",
                    type(exc).__name__,
                    exc,
                    extra={"file_path": abs_file_path},
                )

            # Generate diff for display
            import difflib

            old_lines = content.splitlines(keepends=True)
            new_lines = new_content.splitlines(keepends=True)

            diff = list(
                difflib.unified_diff(
                    old_lines,
                    new_lines,
                    fromfile=input_data.file_path,
                    tofile=input_data.file_path,
                    lineterm="",
                )
            )

            # Count additions and deletions from diff
            additions = sum(
                1 for line in diff if line.startswith("+") and not line.startswith("+++")
            )
            deletions = sum(
                1 for line in diff if line.startswith("-") and not line.startswith("---")
            )

            # Store diff lines for display
            diff_lines = []
            for line in diff[3:]:  # Skip header lines
                diff_lines.append(line)

            # Generate diff with line numbers for better display
            diff_with_line_numbers = []
            old_line_num = None
            new_line_num = None

            for line in diff[3:]:  # Skip header lines
                if line.startswith("@@"):
                    # Parse line numbers from diff header
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

            output = FileEditToolOutput(
                file_path=input_data.file_path,
                replacements_made=replacements,
                success=True,
                message=f"Successfully made {replacements} replacement(s) in {input_data.file_path}",
                additions=additions,
                deletions=deletions,
                diff_lines=diff_lines,
                diff_with_line_numbers=diff_with_line_numbers,
            )

            yield ToolResult(
                data=output, result_for_assistant=self.render_result_for_assistant(output)
            )

        except (OSError, IOError, PermissionError, UnicodeDecodeError, ValueError) as e:
            logger.warning(
                "[file_edit_tool] Error editing file: %s: %s",
                type(e).__name__,
                e,
                extra={"file_path": input_data.file_path},
            )
            error_output = FileEditToolOutput(
                file_path=input_data.file_path,
                replacements_made=0,
                success=False,
                message=f"Error editing file: {str(e)}",
            )

            yield ToolResult(
                data=error_output,
                result_for_assistant=self.render_result_for_assistant(error_output),
            )
