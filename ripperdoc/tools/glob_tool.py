"""Glob pattern matching tool.

Allows the AI to find files using glob patterns.
"""

from pathlib import Path
from typing import AsyncGenerator, Optional, List
from pydantic import BaseModel, Field

from ripperdoc.core.tool import (
    Tool,
    ToolUseContext,
    ToolResult,
    ToolOutput,
    ToolUseExample,
    ValidationResult,
)


GLOB_USAGE = (
    "- Fast file pattern matching tool for any codebase size\n"
    '- Supports glob patterns like "**/*.js" or "src/**/*.ts"\n'
    "- Returns matching file paths sorted by modification time (newest first)\n"
    "- Use this when you need to find files by name patterns\n"
    "- For open-ended searches that need multiple rounds of globbing and grepping, run the searches iteratively with these tools\n"
    "- You can call multiple tools in a single response; speculatively batch useful searches together"
)


class GlobToolInput(BaseModel):
    """Input schema for GlobTool."""

    pattern: str = Field(description="Glob pattern to match files (e.g., '**/*.py')")
    path: Optional[str] = Field(
        default=None, description="Directory to search in (default: current working directory)"
    )


class GlobToolOutput(BaseModel):
    """Output from glob pattern matching."""

    matches: List[str]
    pattern: str
    count: int


class GlobTool(Tool[GlobToolInput, GlobToolOutput]):
    """Tool for finding files using glob patterns."""

    @property
    def name(self) -> str:
        return "Glob"

    async def description(self) -> str:
        return GLOB_USAGE

    @property
    def input_schema(self) -> type[GlobToolInput]:
        return GlobToolInput

    def input_examples(self) -> List[ToolUseExample]:
        return [
            ToolUseExample(
                description="Find Python sources inside src",
                example={"pattern": "src/**/*.py"},
            ),
            ToolUseExample(
                description="Locate snapshot files within tests",
                example={"pattern": "tests/**/__snapshots__/*.snap", "path": "/repo"},
            ),
        ]

    async def prompt(self, safe_mode: bool = False) -> str:
        return GLOB_USAGE

    def is_read_only(self) -> bool:
        return True

    def is_concurrency_safe(self) -> bool:
        return True

    def needs_permissions(self, input_data: Optional[GlobToolInput] = None) -> bool:
        return False

    async def validate_input(
        self, input_data: GlobToolInput, context: Optional[ToolUseContext] = None
    ) -> ValidationResult:
        return ValidationResult(result=True)

    def render_result_for_assistant(self, output: GlobToolOutput) -> str:
        """Format output for the AI."""
        if not output.matches:
            return f"No files found matching pattern: {output.pattern}"

        result = f"Found {output.count} file(s) matching '{output.pattern}':\n\n"
        result += "\n".join(output.matches)

        return result

    def render_tool_use_message(self, input_data: GlobToolInput, verbose: bool = False) -> str:
        """Format the tool use for display."""
        return f"Glob: {input_data.pattern}"

    async def call(
        self, input_data: GlobToolInput, context: ToolUseContext
    ) -> AsyncGenerator[ToolOutput, None]:
        """Find files matching the pattern."""

        try:
            search_path = Path(input_data.path) if input_data.path else Path.cwd()

            # Use glob to find matches, sorted by modification time (newest first)
            paths = list(search_path.glob(input_data.pattern))

            def _mtime(path: Path) -> float:
                try:
                    return path.stat().st_mtime
                except OSError:
                    return float("-inf")

            matches = [str(p) for p in sorted(paths, key=_mtime, reverse=True)]

            output = GlobToolOutput(matches=matches, pattern=input_data.pattern, count=len(matches))

            yield ToolResult(
                data=output, result_for_assistant=self.render_result_for_assistant(output)
            )

        except Exception as e:
            error_output = GlobToolOutput(matches=[], pattern=input_data.pattern, count=0)

            yield ToolResult(
                data=error_output, result_for_assistant=f"Error executing glob: {str(e)}"
            )
