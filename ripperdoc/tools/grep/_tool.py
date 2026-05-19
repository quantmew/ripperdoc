"""Grep tool for searching code.

Allows the AI to search for patterns in files.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from typing import AsyncGenerator, List, Optional, Tuple

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
from ripperdoc.tools.grep._utils import (
    _grep_supports_pcre,
    _normalize_glob_for_grep,
    _parse_content_line,
    _parse_count_line,
    _split_globs,
    _TYPE_GLOB_MAP,
    apply_head_limit,
    truncate_with_ellipsis,
)
from ripperdoc.tools.grep._prompt import GREP_PROMPT as GREP_USAGE

logger = get_logger()


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
    head_limit: Optional[int] = Field(
        default=None,
        description="Limit output to the first N results (similar to piping to head -N) to avoid huge responses.",
    )
    context_before: Optional[int] = Field(
        default=None,
        description="Number of lines of context to show before each match (maps to -B flag)",
    )
    context_after: Optional[int] = Field(
        default=None,
        description="Number of lines of context to show after each match (maps to -A flag)",
    )
    context: Optional[int] = Field(
        default=None,
        description="Number of lines of context to show around each match (maps to -C flag, alias for -B/-A)",
    )
    offset: Optional[int] = Field(
        default=None,
        description="Skip the first N results before applying head_limit (for pagination)",
    )
    multiline: bool = Field(
        default=False,
        description="Enable multiline mode for patterns spanning multiple lines",
    )
    type_filter: Optional[str] = Field(
        default=None,
        description="Ripgrep type filter (e.g., 'py', 'js', 'rust'). Maps to --type flag.",
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
    output_mode: str = "files_with_matches"
    head_limit: Optional[int] = None
    offset: Optional[int] = None
    omitted_results: int = 0


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

    async def prompt(self, _yolo_mode: bool = False) -> str:
        return GREP_USAGE

    def is_read_only(self) -> bool:
        return True

    def is_concurrency_safe(self) -> bool:
        return True

    def needs_permissions(self, _input_data: Optional[GrepToolInput] = None) -> bool:
        return False

    async def validate_input(
        self, input_data: GrepToolInput, _context: Optional[ToolUseContext] = None
    ) -> ValidationResult:
        valid_modes = ["files_with_matches", "content", "count"]
        if input_data.output_mode not in valid_modes:
            return ValidationResult(
                result=False, message=f"Invalid output_mode. Must be one of: {valid_modes}"
            )
        if input_data.head_limit is not None and input_data.head_limit <= 0:
            return ValidationResult(result=False, message="head_limit must be positive")
        if input_data.offset is not None and input_data.offset < 0:
            return ValidationResult(result=False, message="offset must be non-negative")
        if input_data.context is not None and (input_data.context_before is not None or input_data.context_after is not None):
            return ValidationResult(result=False, message="Cannot use context with context_before/context_after")
        return ValidationResult(result=True)

    def render_result_for_assistant(self, output: GrepToolOutput) -> str:
        if output.total_files == 0 or output.total_matches == 0:
            return f"No matches found for pattern: {output.pattern}"

        lines: List[str] = []
        summary: str

        if output.output_mode == "files_with_matches":
            summary = f"Found {output.total_files} file(s) matching '{output.pattern}'."
            lines = [match.file for match in output.matches if match.file]
        elif output.output_mode == "count":
            summary = (
                f"Found {output.total_matches} total match(es) across {output.total_files} file(s) "
                f"for '{output.pattern}'."
            )
            lines = [
                f"{match.file}: {match.count if match.count is not None else 0}"
                for match in output.matches
                if match.file
            ]
        else:
            summary = (
                f"Found {output.total_matches} match(es) in {output.total_files} file(s) "
                f"for '{output.pattern}':"
            )
            for match in output.matches:
                if match.content is None:
                    continue
                line_number = f":{match.line_number}" if match.line_number is not None else ""
                lines.append(f"{match.file}{line_number}: {match.content}")

        if output.omitted_results:
            offset_note = f" (offset={output.offset})" if output.offset else ""
            lines.append(
                f"... and {output.omitted_results} more result(s) not shown"
                f"{offset_note}"
                f"{' (use head_limit to control output size)' if output.head_limit else ''}"
            )

        result = summary
        if lines:
            result += "\n\n" + "\n".join(lines)

        truncated_result, did_truncate, _ = truncate_with_ellipsis(result)
        if did_truncate:
            truncated_result += (
                "\n(Output truncated; refine the pattern or lower head_limit for more detail.)"
            )
        return truncated_result

    def render_tool_use_message(self, input_data: GrepToolInput, _verbose: bool = False) -> str:
        msg = f"Grep: {input_data.pattern}"
        if input_data.glob:
            msg += f" in {input_data.glob}"
        if input_data.type_filter:
            msg += f" (type={input_data.type_filter})"
        if input_data.head_limit:
            msg += f" (head_limit={input_data.head_limit})"
        if input_data.offset:
            msg += f" (offset={input_data.offset})"
        if input_data.multiline:
            msg += " [multiline]"
        if input_data.context:
            msg += f" (context={input_data.context})"
        return msg

    async def call(
        self, input_data: GrepToolInput, _context: ToolUseContext
    ) -> AsyncGenerator[ToolOutput, None]:
        logger.debug(
            "[grep_tool] call ENTER: pattern='%s' path='%s'", input_data.pattern, input_data.path
        )

        try:
            search_path = input_data.path or "."

            async def _run_search(command: List[str]) -> Tuple[int, str, str]:
                process = await asyncio.create_subprocess_exec(
                    *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                stdout_text = stdout.decode("utf-8", errors="ignore") if stdout else ""
                stderr_text = stderr.decode("utf-8", errors="ignore") if stderr else ""
                return process.returncode or 0, stdout_text, stderr_text

            use_ripgrep = shutil.which("rg") is not None
            pattern = input_data.pattern

            if use_ripgrep:
                cmd = ["rg", "--color", "never"]
                if input_data.case_insensitive:
                    cmd.append("-i")
                if input_data.output_mode == "files_with_matches":
                    cmd.append("-l")
                elif input_data.output_mode == "count":
                    cmd.append("-c")
                else:
                    cmd.append("-n")

                if input_data.context is not None:
                    cmd.extend(["-C", str(input_data.context)])
                else:
                    if input_data.context_before is not None:
                        cmd.extend(["-B", str(input_data.context_before)])
                    if input_data.context_after is not None:
                        cmd.extend(["-A", str(input_data.context_after)])

                if input_data.multiline:
                    cmd.append("-U")

                if input_data.type_filter:
                    cmd.extend(["--type", input_data.type_filter])

                for glob_pattern in _split_globs(input_data.glob or ""):
                    cmd.extend(["--glob", glob_pattern])

                if pattern.startswith("-"):
                    cmd.extend(["-e", pattern])
                else:
                    cmd.append(pattern)

                cmd.append(search_path)
            else:
                use_pcre = _grep_supports_pcre()
                cmd = ["grep", "-r", "--color=never", "-P" if use_pcre else "-E"]

                if input_data.case_insensitive:
                    cmd.append("-i")

                if input_data.context is not None:
                    cmd.extend(["-C", str(input_data.context)])
                else:
                    if input_data.context_before is not None:
                        cmd.extend(["-B", str(input_data.context_before)])
                    if input_data.context_after is not None:
                        cmd.extend(["-A", str(input_data.context_after)])

                if input_data.output_mode == "files_with_matches":
                    cmd.extend(["-l"])
                elif input_data.output_mode == "count":
                    cmd.extend(["-c"])
                else:
                    cmd.extend(["-n"])

                if input_data.type_filter:
                    mapped = _TYPE_GLOB_MAP.get(input_data.type_filter)
                    if mapped:
                        for g in mapped.split():
                            cmd.extend(["--include", g])

                for glob_pattern in _split_globs(input_data.glob or ""):
                    cmd.extend(["--include", _normalize_glob_for_grep(glob_pattern)])

                if pattern.startswith("-"):
                    cmd.extend(["-e", pattern])
                else:
                    cmd.append(pattern)

                cmd.append(search_path)

            returncode, stdout_text, stderr_text = await _run_search(cmd)
            fallback_attempted = False

            if returncode not in (0, 1):
                if not use_ripgrep and "-P" in cmd:
                    fallback_attempted = True
                    cmd = [flag if flag != "-P" else "-E" for flag in cmd]
                    returncode, stdout_text, stderr_text = await _run_search(cmd)

                if returncode not in (0, 1):
                    error_msg = stderr_text.strip() or f"grep exited with status {returncode}"
                    logger.warning(
                        "[grep_tool] Grep command failed",
                        extra={
                            "pattern": input_data.pattern,
                            "path": input_data.path,
                            "returncode": returncode,
                            "stderr": error_msg,
                            "fallback_to_E": fallback_attempted,
                        },
                    )
                    error_output = GrepToolOutput(
                        matches=[],
                        pattern=input_data.pattern,
                        total_files=0,
                        total_matches=0,
                        output_mode=input_data.output_mode,
                        head_limit=input_data.head_limit,
                    )
                    yield ToolResult(
                        data=error_output, result_for_assistant=f"Grep error: {error_msg}"
                    )
                    return

            matches: List[GrepMatch] = []
            total_matches = 0
            total_files = 0
            omitted_results = 0
            lines = [line for line in stdout_text.split("\n") if line]

            if returncode in (0, 1):
                offset = input_data.offset or 0
                if offset > 0:
                    lines = lines[offset:]

                if input_data.output_mode == "files_with_matches":
                    unique_files = list(dict.fromkeys(lines))
                    try:
                        unique_files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                    except OSError:
                        pass

                    total_files = len(unique_files)
                    total_matches = len(lines)
                    display_files, omitted_results = apply_head_limit(unique_files, input_data.head_limit)
                    matches = [GrepMatch(file=line) for line in display_files]

                elif input_data.output_mode == "count":
                    display_lines, omitted_results = apply_head_limit(lines, input_data.head_limit)
                    parsed_files = []
                    for line in lines:
                        parsed = _parse_count_line(line, search_path)
                        if parsed:
                            parsed_files.append(parsed[0])
                    total_files = len(set(parsed_files))
                    total_match_count = 0
                    for line in lines:
                        parsed = _parse_count_line(line, search_path)
                        if parsed:
                            total_match_count += parsed[1]
                    total_matches = total_match_count

                    for line in display_lines:
                        parsed = _parse_count_line(line, search_path)
                        if parsed:
                            matches.append(
                                GrepMatch(
                                    file=parsed[0],
                                    count=parsed[1],
                                )
                            )

                else:  # content mode
                    display_lines, omitted_results = apply_head_limit(lines, input_data.head_limit)
                    parsed_files = []
                    for line in lines:
                        parsed_content = _parse_content_line(line, search_path)
                        if parsed_content:
                            parsed_files.append(parsed_content[0])
                    total_files = len(set(parsed_files))
                    total_matches = len(lines)
                    for line in display_lines:
                        parsed_content = _parse_content_line(line, search_path)
                        if parsed_content:
                            matches.append(
                                GrepMatch(
                                    file=parsed_content[0],
                                    line_number=parsed_content[1],
                                    content=parsed_content[2],
                                )
                            )

            output = GrepToolOutput(
                matches=matches,
                pattern=input_data.pattern,
                total_files=total_files,
                total_matches=total_matches,
                output_mode=input_data.output_mode,
                head_limit=input_data.head_limit,
                offset=input_data.offset,
                omitted_results=omitted_results,
            )

            yield ToolResult(
                data=output, result_for_assistant=self.render_result_for_assistant(output)
            )

        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as e:
            logger.warning(
                "[grep_tool] Error executing grep: %s: %s",
                type(e).__name__,
                e,
                extra={"pattern": input_data.pattern, "path": input_data.path},
            )
            error_output = GrepToolOutput(
                matches=[], pattern=input_data.pattern, total_files=0, total_matches=0
            )

            yield ToolResult(
                data=error_output, result_for_assistant=f"Error executing grep: {str(e)}"
            )
