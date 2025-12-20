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
from ripperdoc.utils.log import get_logger

logger = get_logger()


GLOB_USAGE = (
    "- Fast file pattern matching tool that works with any codebase size\n"
    '- Supports glob patterns like "**/*.js" or "src/**/*.ts"\n'
    "- Returns matching file paths sorted by modification time\n"
    "- Use this tool when you need to find files by name patterns\n"
    "- When you are doing an open ended search that may require multiple rounds of globbing and grepping, use the Agent tool instead\n"
    "- You have the capability to call multiple tools in a single response. It is always better to speculatively perform multiple searches as a batch that are potentially useful.\n"
)

RESULT_LIMIT = 100


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
    truncated: bool = False


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

    async def prompt(self, _yolo_mode: bool = False) -> str:
        return GLOB_USAGE

    def is_read_only(self) -> bool:
        return True

    def is_concurrency_safe(self) -> bool:
        return True

    def needs_permissions(self, _input_data: Optional[GlobToolInput] = None) -> bool:
        return False

    async def validate_input(
        self, _input_data: GlobToolInput, _context: Optional[ToolUseContext] = None
    ) -> ValidationResult:
        return ValidationResult(result=True)

    def render_result_for_assistant(self, output: GlobToolOutput) -> str:
        """Format output for the AI."""
        if not output.matches:
            return f"No files found matching pattern: {output.pattern}"

        lines = list(output.matches)
        if output.truncated:
            lines.append("(Results are truncated. Consider using a more specific path or pattern.)")
        return "\n".join(lines)

    def render_tool_use_message(self, input_data: GlobToolInput, _verbose: bool = False) -> str:
        """Format the tool use for display."""
        if not input_data.pattern:
            return "Glob"

        base_path = Path.cwd()
        rendered_path = ""
        if input_data.path:
            candidate_path = Path(input_data.path)
            absolute_path = (
                candidate_path
                if candidate_path.is_absolute()
                else (base_path / candidate_path).resolve()
            )

            try:
                relative_path = absolute_path.relative_to(base_path)
            except ValueError:
                relative_path = None

            if _verbose or not relative_path or str(relative_path) == ".":
                rendered_path = str(absolute_path)
            else:
                rendered_path = str(relative_path)

        path_fragment = f', path: "{rendered_path}"' if rendered_path else ""
        return f'pattern: "{input_data.pattern}"{path_fragment}'

    async def call(
        self, input_data: GlobToolInput, _context: ToolUseContext
    ) -> AsyncGenerator[ToolOutput, None]:
        """Find files matching the pattern."""

        try:
            search_path = Path(input_data.path) if input_data.path else Path.cwd()
            if not search_path.is_absolute():
                search_path = (Path.cwd() / search_path).resolve()

            def _mtime(path: Path) -> float:
                try:
                    return path.stat().st_mtime
                except OSError:
                    return float("-inf")

            # Find matching files, sorted by modification time
            paths = sorted(
                (p for p in search_path.glob(input_data.pattern) if p.is_file()),
                key=_mtime,
            )

            truncated = len(paths) > RESULT_LIMIT
            paths = paths[:RESULT_LIMIT]

            matches = [str(p) for p in paths]

            output = GlobToolOutput(
                matches=matches, pattern=input_data.pattern, count=len(matches), truncated=truncated
            )

            yield ToolResult(
                data=output, result_for_assistant=self.render_result_for_assistant(output)
            )

        except (OSError, RuntimeError, ValueError) as e:
            logger.warning(
                "[glob_tool] Error executing glob: %s: %s",
                type(e).__name__,
                e,
                extra={"pattern": input_data.pattern, "path": input_data.path},
            )
            error_output = GlobToolOutput(matches=[], pattern=input_data.pattern, count=0)

            yield ToolResult(
                data=error_output, result_for_assistant=f"Error executing glob: {str(e)}"
            )
