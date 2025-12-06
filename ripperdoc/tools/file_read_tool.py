"""File reading tool.

Allows the AI to read file contents.
"""

import os
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

logger = get_logger()


class FileReadToolInput(BaseModel):
    """Input schema for FileReadTool."""

    file_path: str = Field(description="Absolute path to the file to read")
    offset: Optional[int] = Field(
        default=None, description="Line number to start reading from (optional)"
    )
    limit: Optional[int] = Field(default=None, description="Number of lines to read (optional)")


class FileReadToolOutput(BaseModel):
    """Output from file reading."""

    content: str
    file_path: str
    line_count: int
    offset: int
    limit: Optional[int]


class FileReadTool(Tool[FileReadToolInput, FileReadToolOutput]):
    """Tool for reading file contents."""

    @property
    def name(self) -> str:
        return "View"

    async def description(self) -> str:
        return """Read the contents of a file. You can optionally specify an offset
and limit to read only a portion of the file."""

    @property
    def input_schema(self) -> type[FileReadToolInput]:
        return FileReadToolInput

    def input_examples(self) -> List[ToolUseExample]:
        return [
            ToolUseExample(
                description="Read the top of a file to understand structure",
                example={"file_path": "/repo/src/main.py", "limit": 50},
            ),
            ToolUseExample(
                description="Inspect a slice of a large log without loading everything",
                example={"file_path": "/repo/logs/server.log", "offset": 200, "limit": 40},
            ),
        ]

    async def prompt(self, safe_mode: bool = False) -> str:
        return (
            "Read a file from the local filesystem.\n\n"
            "Usage:\n"
            "- The file_path parameter must be an absolute path (not relative).\n"
            "- By default, the entire file is read. You can optionally specify a line offset and limit (handy for long files); offset is zero-based and output line numbers start at 1.\n"
            "- Lines longer than 2000 characters are truncated in the output.\n"
            "- Results are returned with cat -n style numbering: spaces + line number + tab, then the file content.\n"
            "- You can call multiple tools in a single response—speculatively read multiple potentially useful files together.\n"
            "- It is okay to attempt reading a non-existent file; an error will be returned if the file is missing."
        )

    def is_read_only(self) -> bool:
        return True

    def is_concurrency_safe(self) -> bool:
        return True

    def needs_permissions(self, input_data: Optional[FileReadToolInput] = None) -> bool:
        return False

    async def validate_input(
        self, input_data: FileReadToolInput, context: Optional[ToolUseContext] = None
    ) -> ValidationResult:
        # Check if file exists
        if not os.path.exists(input_data.file_path):
            return ValidationResult(result=False, message=f"File not found: {input_data.file_path}")

        # Check if it's a file (not a directory)
        if not os.path.isfile(input_data.file_path):
            return ValidationResult(
                result=False, message=f"Path is not a file: {input_data.file_path}"
            )

        return ValidationResult(result=True)

    def render_result_for_assistant(self, output: FileReadToolOutput) -> str:
        """Format output for the AI."""
        lines = output.content.split("\n")
        numbered_lines = []

        for i, line in enumerate(lines, start=output.offset + 1):
            # Truncate very long lines
            if len(line) > 2000:
                line = line[:2000] + "... [truncated]"
            numbered_lines.append(f"{i:6d}\t{line}")

        return "\n".join(numbered_lines)

    def render_tool_use_message(self, input_data: FileReadToolInput, verbose: bool = False) -> str:
        """Format the tool use for display."""
        msg = f"Reading: {input_data.file_path}"
        if input_data.offset or input_data.limit:
            msg += f" (offset: {input_data.offset}, limit: {input_data.limit})"
        return msg

    async def call(
        self, input_data: FileReadToolInput, context: ToolUseContext
    ) -> AsyncGenerator[ToolOutput, None]:
        """Read the file."""

        try:
            with open(input_data.file_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            offset = input_data.offset or 0
            limit = input_data.limit

            # Apply offset and limit
            if limit is not None:
                selected_lines = lines[offset : offset + limit]
            else:
                selected_lines = lines[offset:]

            content = "".join(selected_lines)

            # Remember what we read so we can detect user edits later.
            try:
                record_snapshot(
                    input_data.file_path,
                    content,
                    getattr(context, "file_state_cache", {}),
                    offset=offset,
                    limit=limit,
                )
            except Exception:
                logger.exception(
                    "[file_read_tool] Failed to record file snapshot",
                    extra={"file_path": input_data.file_path},
                )

            output = FileReadToolOutput(
                content=content,
                file_path=input_data.file_path,
                line_count=len(selected_lines),
                offset=offset,
                limit=limit,
            )

            yield ToolResult(
                data=output, result_for_assistant=self.render_result_for_assistant(output)
            )

        except Exception as e:
            logger.exception(
                "[file_read_tool] Error reading file",
                extra={"file_path": input_data.file_path, "error": str(e)},
            )
            # Create an error output
            error_output = FileReadToolOutput(
                content=f"Error reading file: {str(e)}",
                file_path=input_data.file_path,
                line_count=0,
                offset=0,
                limit=None,
            )

            yield ToolResult(
                data=error_output,
                result_for_assistant=f"Error reading file {input_data.file_path}: {str(e)}",
            )
