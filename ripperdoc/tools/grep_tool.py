"""Grep tool for searching code.

Allows the AI to search for patterns in files.
"""

import asyncio
import re
import shutil
import subprocess
from typing import AsyncGenerator, Optional, List, Tuple
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

MAX_GREP_OUTPUT_CHARS = 20000


GREP_USAGE = (
    "A powerful search tool built on ripgrep.\n\n"
    "Usage:\n"
    "- ALWAYS use the Grep tool for search tasks. NEVER invoke `grep` or `rg` as a Bash command; this tool is optimized for permissions and access.\n"
    '- Supports regex patterns (e.g., "log.*Error", "function\\s+\\w+")\n'
    '- Filter files with the glob parameter (e.g., "*.js", "**/*.tsx")\n'
    '- Output modes: "content" shows matching lines, "files_with_matches" (default) shows only file paths, "count" shows match counts\n'
    "- Use head_limit to cap the number of returned entries (similar to piping through head -N) to avoid overwhelming output\n"
    f"- Outputs are automatically truncated to around {MAX_GREP_OUTPUT_CHARS} characters to stay within context limits; narrow patterns for more detail\n"
    "- For open-ended searches that need multiple rounds, iterate with Glob and Grep rather than shell commands\n"
    "- Patterns are line-based; craft patterns accordingly and escape braces if needed (e.g., use `interface\\{\\}` to find `interface{}`)"
)


def truncate_with_ellipsis(
    text: str, max_chars: int = MAX_GREP_OUTPUT_CHARS
) -> Tuple[str, bool, int]:
    """Trim long output and note how many lines were removed."""
    if len(text) <= max_chars:
        return text, False, 0

    remaining = text[max_chars:]
    truncated_lines = remaining.count("\n") + (1 if remaining else 0)
    truncated_text = f"{text[:max_chars]}\n\n... [{truncated_lines} lines truncated] ..."
    return truncated_text, True, truncated_lines


def apply_head_limit(lines: List[str], head_limit: Optional[int]) -> Tuple[List[str], int]:
    """Limit the number of lines returned, recording how many were omitted."""
    if head_limit is None or head_limit <= 0:
        return lines, 0
    if len(lines) <= head_limit:
        return lines, 0
    return lines[:head_limit], len(lines) - head_limit


def _split_globs(glob_value: str) -> List[str]:
    """Split a glob string by whitespace and commas."""
    if not glob_value:
        return []
    globs: List[str] = []
    for token in re.split(r"\s+", glob_value.strip()):
        if not token:
            continue
        globs.extend([part for part in token.split(",") if part])
    return globs


def _normalize_glob_for_grep(glob_pattern: str) -> str:
    """grep --include matches basenames; drop path components to avoid mismatches like **/*.py."""
    return glob_pattern.split("/")[-1] or glob_pattern


_GREP_SUPPORTS_PCRE: Optional[bool] = None


def _grep_supports_pcre() -> bool:
    """Detect if the system grep supports -P (Perl regex), caching the result."""
    global _GREP_SUPPORTS_PCRE
    if _GREP_SUPPORTS_PCRE is not None:
        return _GREP_SUPPORTS_PCRE

    if shutil.which("grep") is None:
        _GREP_SUPPORTS_PCRE = False
        return _GREP_SUPPORTS_PCRE

    try:
        proc = subprocess.run(
            ["grep", "-P", ""],
            stdin=subprocess.DEVNULL,  # Fix: prevent waiting for stdin
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
            timeout=15,  # Safety timeout
        )
        _GREP_SUPPORTS_PCRE = proc.returncode in (0, 1)
    except (OSError, ValueError, subprocess.SubprocessError, subprocess.TimeoutExpired):
        _GREP_SUPPORTS_PCRE = False

    return _GREP_SUPPORTS_PCRE


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
        return ValidationResult(result=True)

    def render_result_for_assistant(self, output: GrepToolOutput) -> str:
        """Format output for the AI."""
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
            lines.append(
                f"... and {output.omitted_results} more result(s) not shown"
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
        """Format the tool use for display."""
        msg = f"Grep: {input_data.pattern}"
        if input_data.glob:
            msg += f" in {input_data.glob}"
        if input_data.head_limit:
            msg += f" (head_limit={input_data.head_limit})"
        return msg

    async def call(
        self, input_data: GrepToolInput, _context: ToolUseContext
    ) -> AsyncGenerator[ToolOutput, None]:
        """Search for the pattern."""
        logger.debug(
            "[grep_tool] call ENTER: pattern='%s' path='%s'", input_data.pattern, input_data.path
        )

        try:
            search_path = input_data.path or "."

            async def _run_search(command: List[str]) -> Tuple[int, str, str]:
                """Execute the search command and return decoded output."""
                logger.debug(
                    "[grep_tool] _run_search: BEFORE create_subprocess_exec, cmd=%s", command[:5]
                )
                process = await asyncio.create_subprocess_exec(
                    *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                logger.debug(
                    "[grep_tool] _run_search: AFTER create_subprocess_exec, pid=%s", process.pid
                )
                logger.debug("[grep_tool] _run_search: BEFORE communicate()")
                stdout, stderr = await process.communicate()
                logger.debug(
                    "[grep_tool] _run_search: AFTER communicate(), returncode=%s",
                    process.returncode,
                )
                stdout_text = stdout.decode("utf-8", errors="ignore") if stdout else ""
                stderr_text = stderr.decode("utf-8", errors="ignore") if stderr else ""
                return process.returncode or 0, stdout_text, stderr_text

            use_ripgrep = shutil.which("rg") is not None
            logger.debug("[grep_tool] use_ripgrep=%s", use_ripgrep)
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

                for glob_pattern in _split_globs(input_data.glob or ""):
                    cmd.extend(["--glob", glob_pattern])

                if pattern.startswith("-"):
                    cmd.extend(["-e", pattern])
                else:
                    cmd.append(pattern)

                cmd.append(search_path)
            else:
                # Fallback to grep (note: grep --include matches basenames only)
                logger.debug("[grep_tool] Using grep fallback, checking PCRE support...")
                use_pcre = _grep_supports_pcre()
                logger.debug("[grep_tool] PCRE support check done: use_pcre=%s", use_pcre)
                cmd = ["grep", "-r", "--color=never", "-P" if use_pcre else "-E"]
                logger.debug("[grep_tool] Building grep command...")

                if input_data.case_insensitive:
                    cmd.append("-i")

                if input_data.output_mode == "files_with_matches":
                    cmd.extend(["-l"])  # Files with matches
                elif input_data.output_mode == "count":
                    cmd.extend(["-c"])  # Count per file
                else:
                    cmd.extend(["-n"])  # Line numbers

                for glob_pattern in _split_globs(input_data.glob or ""):
                    cmd.extend(["--include", _normalize_glob_for_grep(glob_pattern)])

                if pattern.startswith("-"):
                    cmd.extend(["-e", pattern])
                else:
                    cmd.append(pattern)

                cmd.append(search_path)

            logger.debug("[grep_tool] BEFORE _run_search, cmd=%s", cmd)
            returncode, stdout_text, stderr_text = await _run_search(cmd)
            logger.debug(
                "[grep_tool] AFTER _run_search, returncode=%s, stdout_len=%d",
                returncode,
                len(stdout_text),
            )
            fallback_attempted = False

            if returncode not in (0, 1):
                if not use_ripgrep and "-P" in cmd:
                    # BSD grep lacks -P; retry with extended regex before surfacing the error.
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

            # Parse output
            matches: List[GrepMatch] = []
            total_matches = 0
            total_files = 0
            omitted_results = 0
            lines = [line for line in stdout_text.split("\n") if line]

            if returncode in (0, 1):  # 0 = matches found, 1 = no matches (ripgrep/grep)
                display_lines, omitted_results = apply_head_limit(lines, input_data.head_limit)

                if input_data.output_mode == "files_with_matches":
                    total_files = len(set(lines))
                    total_matches = len(lines)
                    matches = [GrepMatch(file=line) for line in display_lines]

                elif input_data.output_mode == "count":
                    total_files = len(set(line.split(":", 1)[0] for line in lines if line))
                    total_match_count = 0
                    for line in lines:
                        parts = line.rsplit(":", 1)
                        if len(parts) == 2 and parts[1].isdigit():
                            total_match_count += int(parts[1])
                    total_matches = total_match_count

                    for line in display_lines:
                        parts = line.rsplit(":", 1)
                        if len(parts) == 2:
                            matches.append(
                                GrepMatch(
                                    file=parts[0], count=int(parts[1]) if parts[1].isdigit() else 0
                                )
                            )

                else:  # content mode
                    total_files = len({line.split(":", 1)[0] for line in lines if line})
                    total_matches = len(lines)
                    for line in display_lines:
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
                total_files=total_files,
                total_matches=total_matches,
                output_mode=input_data.output_mode,
                head_limit=input_data.head_limit,
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
