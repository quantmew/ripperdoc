"""File editing tool.

Allows the AI to edit files by replacing text.
"""

from ripperdoc.tools.file_edit._tool import (
    FileEditTool,
    FileEditToolInput,
    FileEditToolOutput,
)
from ripperdoc.tools.file_edit._utils import (
    determine_edit_encoding,
    validate_file_size,
)

__all__ = [
    "FileEditTool",
    "FileEditToolInput",
    "FileEditToolOutput",
    "determine_edit_encoding",
    "validate_file_size",
]
