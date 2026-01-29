"""File reading tool.

Allows the AI to read file contents.
"""

import os
from pathlib import Path
from typing import AsyncGenerator, List, Optional, Tuple
from pydantic import BaseModel, Field
from charset_normalizer import from_bytes

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


def detect_file_encoding(file_path: str) -> Tuple[Optional[str], float]:
    """Detect file encoding using charset-normalizer.

    Returns:
        Tuple of (encoding, confidence). encoding is None if detection failed.
    """
    try:
        with open(file_path, "rb") as f:
            raw_data = f.read()
        results = from_bytes(raw_data)

        if not results:
            return None, 0.0

        best = results.best()
        if not best:
            return None, 0.0

        # For Chinese content, prefer GB encodings over Big5/others
        # charset-normalizer sometimes picks Big5 for simplified Chinese
        if best.language == "Chinese":
            gb_encodings = {"gb18030", "gbk", "gb2312"}
            for result in results:
                if result.encoding.lower() in gb_encodings:
                    return result.encoding, 0.9

        return best.encoding, 0.9
    except (OSError, IOError) as e:
        logger.warning("Failed to detect encoding for %s: %s", file_path, e)
        return None, 0.0


def read_file_with_encoding(file_path: str) -> Tuple[Optional[List[str]], str, Optional[str]]:
    """Read file with proper encoding detection.

    Returns:
        Tuple of (lines, encoding_used, error_message).
        If successful: (lines, encoding, None)
        If failed: (None, "", error_message)
    """
    # First, try UTF-8 (most common)
    try:
        with open(file_path, "r", encoding="utf-8", errors="strict") as f:
            lines = f.readlines()
        return lines, "utf-8", None
    except UnicodeDecodeError:
        pass

    # UTF-8 failed, use charset-normalizer to detect encoding
    detected_encoding, confidence = detect_file_encoding(file_path)

    if detected_encoding:
        try:
            with open(file_path, "r", encoding=detected_encoding, errors="strict") as f:
                lines = f.readlines()
            logger.info(
                "File %s decoded using detected encoding %s",
                file_path,
                detected_encoding,
            )
            return lines, detected_encoding, None
        except (UnicodeDecodeError, LookupError) as e:
            logger.warning(
                "Failed to read %s with detected encoding %s: %s",
                file_path,
                detected_encoding,
                e,
            )

    # Detection failed - try latin-1 as last resort (can decode any byte sequence)
    try:
        with open(file_path, "r", encoding="latin-1", errors="strict") as f:
            lines = f.readlines()
        logger.warning(
            "File %s: encoding detection failed, using latin-1 fallback",
            file_path,
        )
        return lines, "latin-1", None
    except (UnicodeDecodeError, LookupError):
        pass

    # All attempts failed - return error
    error_msg = (
        f"Unable to determine file encoding. "
        f"Detected: {detected_encoding or 'unknown'} (confidence: {confidence * 100:.0f}%). "
        f"Tried fallback encodings: utf-8, latin-1. "
        f"Please convert the file to UTF-8."
    )
    return None, "", error_msg


# Maximum file size to read (default 256KB)
# Can be overridden via env var in bytes
MAX_FILE_SIZE_BYTES = int(os.getenv("RIPPERDOC_MAX_READ_FILE_SIZE_BYTES", "262144"))  # 256KB

# Maximum lines to read when no limit is specified (default 2000 lines)
MAX_READ_LINES = int(os.getenv("RIPPERDOC_MAX_READ_LINES", "2000"))


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
        return "Read"

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

    async def prompt(self, yolo_mode: bool = False) -> str:
        return (
            "Read a file from the local filesystem.\n\n"
            "Usage:\n"
            "- The file_path parameter must be an absolute path (not relative).\n"
            "- Files larger than 256KB or with more than 2000 lines require using offset and limit parameters.\n"
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

        # Check if path is ignored (warning only for read operations)
        file_path = Path(input_data.file_path)
        should_proceed, warning_msg = check_path_for_tool(
            file_path, tool_name="Read", warn_only=True
        )
        if warning_msg:
            logger.info("[file_read_tool] %s", warning_msg)

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
            # Check file size before reading to prevent memory exhaustion
            file_size = os.path.getsize(input_data.file_path)
            if file_size > MAX_FILE_SIZE_BYTES:
                size_kb = file_size / 1024
                limit_kb = MAX_FILE_SIZE_BYTES / 1024
                error_output = FileReadToolOutput(
                    content=f"File too large to read: {size_kb:.1f}KB exceeds limit of {limit_kb:.0f}KB. Use offset and limit parameters to read portions.",
                    file_path=input_data.file_path,
                    line_count=0,
                    offset=0,
                    limit=None,
                )
                yield ToolResult(
                    data=error_output,
                    result_for_assistant=f"Error: File {input_data.file_path} is too large ({size_kb:.1f}KB). Maximum size is {limit_kb:.0f}KB. Use offset and limit to read portions, e.g., Read(file_path='{input_data.file_path}', offset=0, limit=500).",
                )
                return

            # Detect and read file with proper encoding
            lines, used_encoding, encoding_error = read_file_with_encoding(input_data.file_path)

            if lines is None:
                # Encoding detection failed - return warning to LLM
                error_output = FileReadToolOutput(
                    content=f"Encoding error: {encoding_error}",
                    file_path=input_data.file_path,
                    line_count=0,
                    offset=0,
                    limit=None,
                )
                yield ToolResult(
                    data=error_output,
                    result_for_assistant=f"Error: Cannot read file {input_data.file_path}. {encoding_error}",
                )
                return

            offset = input_data.offset or 0
            limit = input_data.limit
            total_lines = len(lines)

            # Check line count if no limit is specified (to prevent context overflow)
            if limit is None and total_lines > MAX_READ_LINES:
                error_output = FileReadToolOutput(
                    content=f"File too large: {total_lines} lines exceeds limit of {MAX_READ_LINES} lines. Use offset and limit parameters to read portions.",
                    file_path=input_data.file_path,
                    line_count=total_lines,
                    offset=0,
                    limit=None,
                )
                yield ToolResult(
                    data=error_output,
                    result_for_assistant=f"Error: File {input_data.file_path} has {total_lines} lines, exceeding the limit of {MAX_READ_LINES} lines when reading without limit parameter. Use offset and limit to read portions, e.g., Read(file_path='{input_data.file_path}', offset=0, limit=500).",
                )
                return

            # Apply offset and limit
            if limit is not None:
                selected_lines = lines[offset : offset + limit]
            else:
                selected_lines = lines[offset:]

            content = "".join(selected_lines)

            # Remember what we read so we can detect user edits later.
            # Use absolute path to ensure consistency with Edit tool's lookup
            abs_file_path = os.path.abspath(input_data.file_path)
            try:
                record_snapshot(
                    abs_file_path,
                    content,
                    getattr(context, "file_state_cache", {}),
                    offset=offset,
                    limit=limit,
                    encoding=used_encoding,
                )
            except (OSError, IOError, RuntimeError) as exc:
                logger.warning(
                    "[file_read_tool] Failed to record file snapshot: %s: %s",
                    type(exc).__name__,
                    exc,
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

        except (OSError, IOError, UnicodeDecodeError, ValueError) as e:
            logger.warning(
                "[file_read_tool] Error reading file: %s: %s",
                type(e).__name__,
                e,
                extra={"file_path": input_data.file_path},
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
