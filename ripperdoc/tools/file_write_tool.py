"""File writing tool.

Allows the AI to create new files.
"""

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
from ripperdoc.utils.file_watch import record_snapshot
from ripperdoc.utils.path_ignore import check_path_for_tool

logger = get_logger()


class FileWriteToolInput(BaseModel):
    """Input schema for FileWriteTool."""

    file_path: str = Field(description="Absolute path to the file to create")
    content: str = Field(description="Content to write to the file")


class FileWriteToolOutput(BaseModel):
    """Output from file writing."""

    file_path: str
    bytes_written: int
    success: bool
    message: str


class FileWriteTool(Tool[FileWriteToolInput, FileWriteToolOutput]):
    """Tool for creating new files."""

    @property
    def name(self) -> str:
        return "Write"

    async def description(self) -> str:
        return """Create a new file with the specified content. This will overwrite
the file if it already exists."""

    @property
    def input_schema(self) -> type[FileWriteToolInput]:
        return FileWriteToolInput

    def input_examples(self) -> List[ToolUseExample]:
        return [
            ToolUseExample(
                description="Create a JSON fixture file",
                example={
                    "file_path": "/repo/tests/fixtures/sample.json",
                    "content": '{\n  "items": []\n}\n',
                },
            ),
            ToolUseExample(
                description="Write a short markdown note",
                example={
                    "file_path": "/repo/docs/USAGE.md",
                    "content": "# Usage\n\nRun `make test`.\n",
                },
            ),
        ]

    async def prompt(self, yolo_mode: bool = False) -> str:
        prompt = """Use the Write tool to create new files. """

        if not yolo_mode:
            prompt += """IMPORTANT: You must ALWAYS prefer editing existing files.
NEVER write new files unless explicitly required by the user."""

        return prompt

    def is_read_only(self) -> bool:
        return False

    def is_concurrency_safe(self) -> bool:
        return False

    def needs_permissions(self, input_data: Optional[FileWriteToolInput] = None) -> bool:
        return True

    async def validate_input(
        self, input_data: FileWriteToolInput, context: Optional[ToolUseContext] = None
    ) -> ValidationResult:
        # Check if parent directory exists
        parent = Path(input_data.file_path).parent
        if not parent.exists():
            return ValidationResult(
                result=False,
                message=f"Parent directory does not exist: {parent}",
                error_code=1,
            )

        file_path = os.path.abspath(input_data.file_path)

        # If file doesn't exist, it's a new file - allow without reading first
        if not os.path.exists(file_path):
            return ValidationResult(result=True)

        # File exists - check if it has been read before writing
        file_state_cache = getattr(context, "file_state_cache", {}) if context else {}
        file_snapshot = file_state_cache.get(file_path)

        if not file_snapshot:
            return ValidationResult(
                result=False,
                message="File has not been read yet. Read it first before writing to it.",
                error_code=2,
            )

        # Check if file has been modified since it was read
        try:
            current_mtime = os.path.getmtime(file_path)
            if current_mtime > file_snapshot.timestamp:
                return ValidationResult(
                    result=False,
                    message="File has been modified since read, either by the user or by a linter. "
                    "Read it again before attempting to write it.",
                    error_code=3,
                )
        except OSError:
            pass  # File mtime check failed, proceed anyway

        # Check if path is ignored (warning for write operations)
        file_path_obj = Path(file_path)
        should_proceed, warning_msg = check_path_for_tool(
            file_path_obj, tool_name="Write", warn_only=True
        )
        if warning_msg:
            logger.warning("[file_write_tool] %s", warning_msg)

        return ValidationResult(result=True)

    def render_result_for_assistant(self, output: FileWriteToolOutput) -> str:
        """Format output for the AI."""
        return output.message

    def render_tool_use_message(self, input_data: FileWriteToolInput, verbose: bool = False) -> str:
        """Format the tool use for display."""
        return f"Writing: {input_data.file_path} ({len(input_data.content)} bytes)"

    async def call(
        self, input_data: FileWriteToolInput, context: ToolUseContext
    ) -> AsyncGenerator[ToolOutput, None]:
        """Write the file."""

        try:
            # Write the file
            with open(input_data.file_path, "w", encoding="utf-8") as f:
                f.write(input_data.content)

            bytes_written = len(input_data.content.encode("utf-8"))

            # Use absolute path to ensure consistency with validation lookup
            abs_file_path = os.path.abspath(input_data.file_path)
            try:
                record_snapshot(
                    abs_file_path,
                    input_data.content,
                    getattr(context, "file_state_cache", {}),
                )
            except (OSError, IOError, RuntimeError) as exc:
                logger.warning(
                    "[file_write_tool] Failed to record file snapshot: %s: %s",
                    type(exc).__name__,
                    exc,
                    extra={"file_path": abs_file_path},
                )

            output = FileWriteToolOutput(
                file_path=input_data.file_path,
                bytes_written=bytes_written,
                success=True,
                message=f"Successfully wrote {bytes_written} bytes to {input_data.file_path}",
            )

            yield ToolResult(
                data=output, result_for_assistant=self.render_result_for_assistant(output)
            )

        except (OSError, IOError, PermissionError, UnicodeEncodeError) as e:
            logger.warning(
                "[file_write_tool] Error writing file: %s: %s",
                type(e).__name__,
                e,
                extra={"file_path": input_data.file_path},
            )
            error_output = FileWriteToolOutput(
                file_path=input_data.file_path,
                bytes_written=0,
                success=False,
                message=f"Error writing file: {str(e)}",
            )

            yield ToolResult(
                data=error_output,
                result_for_assistant=self.render_result_for_assistant(error_output),
            )
