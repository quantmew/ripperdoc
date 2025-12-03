"""Grep tool for searching code.

Allows the AI to search for patterns in files.
"""

import asyncio
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


GREP_USAGE = (
    "A powerful search tool built on ripgrep.\n\n"
    "Usage:\n"
    "- ALWAYS use the Grep tool for search tasks. NEVER invoke `grep` or `rg` as a Bash command; this tool is optimized for permissions and access.\n"
    '- Supports regex patterns (e.g., "log.*Error", "function\\s+\\w+")\n'
    '- Filter files with the glob parameter (e.g., "*.js", "**/*.tsx")\n'
    '- Output modes: "content" shows matching lines, "files_with_matches" (default) shows only file paths, "count" shows match counts\n'
    "- For open-ended searches that need multiple rounds, iterate with Glob and Grep rather than shell commands\n"
    "- Patterns are line-based; craft patterns accordingly and escape braces if needed (e.g., use `interface\\{\\}` to find `interface{}`)"
)


class GrepToolInput(BaseModel):
    """Input schema for GrepTool."""

    pattern: str = Field(description="Regular expression pattern to search for")
    path: Optional[str] = Field(
        default=None, description="Directory or file to search in (default: current directory)"
    )
    glob: Optional[str] = Field(default=None, description="File pattern to filter (e.g., '*.py')")
    case_insensitive: bool = Field(default=False, description="Case insensitive search")
    output_mode: str = Field(
        default="files_with_matches",
        description="Output mode: 'files_with_matches', 'content', or 'count'",
    )


class GrepMatch(BaseModel):
    """A single grep match."""

    file: str
    line_number: Optional[int] = None
    content: Optional[str] = None
    count: Optional[int] = None


class GrepToolOutput(BaseModel):
    """Output from grep search."""

    matches: List[GrepMatch]
    pattern: str
    total_files: int
    total_matches: int


class GrepTool(Tool[GrepToolInput, GrepToolOutput]):
    """Tool for searching code with grep."""

    @property
    def name(self) -> str:
        return "Grep"

    async def description(self) -> str:
        return GREP_USAGE

    @property
    def input_schema(self) -> type[GrepToolInput]:
        return GrepToolInput

    def input_examples(self) -> List[ToolUseExample]:
        return [
            ToolUseExample(
                description="Find TODO comments in TypeScript files",
                example={"pattern": "TODO", "glob": "**/*.ts", "output_mode": "content"},
            ),
            ToolUseExample(
                description="List files referencing a function name",
                example={
                    "pattern": "fetchUserData",
                    "output_mode": "files_with_matches",
                    "path": "/repo/src",
                },
            ),
        ]

    async def prompt(self, safe_mode: bool = False) -> str:
        return GREP_USAGE

    def is_read_only(self) -> bool:
        return True

    def is_concurrency_safe(self) -> bool:
        return True

    def needs_permissions(self, input_data: Optional[GrepToolInput] = None) -> bool:
        return False

    async def validate_input(
        self, input_data: GrepToolInput, context: Optional[ToolUseContext] = None
    ) -> ValidationResult:
        valid_modes = ["files_with_matches", "content", "count"]
        if input_data.output_mode not in valid_modes:
            return ValidationResult(
                result=False, message=f"Invalid output_mode. Must be one of: {valid_modes}"
            )
        return ValidationResult(result=True)

    def render_result_for_assistant(self, output: GrepToolOutput) -> str:
        """Format output for the AI."""
        if output.total_files == 0:
            return f"No matches found for pattern: {output.pattern}"

        result = f"Found {output.total_matches} match(es) in {output.total_files} file(s) for '{output.pattern}':\n\n"

        for match in output.matches[:100]:  # Limit to first 100
            if match.content:
                result += f"{match.file}:{match.line_number}: {match.content}\n"
            elif match.count:
                result += f"{match.file}: {match.count} matches\n"
            else:
                result += f"{match.file}\n"

        if len(output.matches) > 100:
            result += f"\n... and {len(output.matches) - 100} more matches"

        return result

    def render_tool_use_message(self, input_data: GrepToolInput, verbose: bool = False) -> str:
        """Format the tool use for display."""
        msg = f"Grep: {input_data.pattern}"
        if input_data.glob:
            msg += f" in {input_data.glob}"
        return msg

    async def call(
        self, input_data: GrepToolInput, context: ToolUseContext
    ) -> AsyncGenerator[ToolOutput, None]:
        """Search for the pattern."""

        try:
            search_path = input_data.path or "."

            # Build grep command
            cmd = ["grep", "-r"]

            if input_data.case_insensitive:
                cmd.append("-i")

            if input_data.output_mode == "files_with_matches":
                cmd.extend(["-l"])  # Files with matches
            elif input_data.output_mode == "count":
                cmd.extend(["-c"])  # Count per file
            else:
                cmd.extend(["-n"])  # Line numbers

            cmd.append(input_data.pattern)
            cmd.append(search_path)

            if input_data.glob:
                cmd.extend(["--include", input_data.glob])

            # Run grep asynchronously
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()
            returncode = process.returncode

            # Parse output
            matches: List[GrepMatch] = []

            if returncode == 0:
                lines = stdout.decode("utf-8").strip().split("\n")

                for line in lines:
                    if not line:
                        continue

                    if input_data.output_mode == "files_with_matches":
                        matches.append(GrepMatch(file=line))

                    elif input_data.output_mode == "count":
                        # Format: file:count
                        parts = line.rsplit(":", 1)
                        if len(parts) == 2:
                            matches.append(
                                GrepMatch(
                                    file=parts[0], count=int(parts[1]) if parts[1].isdigit() else 0
                                )
                            )

                    else:  # content mode
                        # Format: file:line:content
                        parts = line.split(":", 2)
                        if len(parts) >= 3:
                            matches.append(
                                GrepMatch(
                                    file=parts[0],
                                    line_number=int(parts[1]) if parts[1].isdigit() else None,
                                    content=parts[2] if len(parts) > 2 else "",
                                )
                            )

            output = GrepToolOutput(
                matches=matches,
                pattern=input_data.pattern,
                total_files=len(set(m.file for m in matches)),
                total_matches=len(matches),
            )

            yield ToolResult(
                data=output, result_for_assistant=self.render_result_for_assistant(output)
            )

        except Exception as e:
            logger.exception(
                "[grep_tool] Error executing grep",
                extra={"pattern": input_data.pattern, "path": input_data.path},
            )
            error_output = GrepToolOutput(
                matches=[], pattern=input_data.pattern, total_files=0, total_matches=0
            )

            yield ToolResult(
                data=error_output, result_for_assistant=f"Error executing grep: {str(e)}"
            )
