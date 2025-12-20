"""Bash command execution tool."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from pathlib import Path
from textwrap import dedent
from typing import Any, AsyncGenerator, List, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from ripperdoc.core.tool import (
    Tool,
    ToolOutput,
    ToolProgress,
    ToolResult,
    ToolUseContext,
    ToolUseExample,
    ValidationResult,
)
from ripperdoc.utils.bash_constants import (
    get_bash_default_timeout_ms,
    get_bash_max_output_length,
    get_bash_max_timeout_ms,
)
from ripperdoc.utils.exit_code_handlers import (
    MAX_PREVIEW_CHARS,
    MAX_PREVIEW_LINES,
    IGNORED_COMMANDS,
    interpret_exit_code,
)
from ripperdoc.utils.output_utils import (
    format_duration,
    get_last_n_lines,
    is_output_large,
    sanitize_output,
    trim_blank_lines,
    truncate_output,
)
from ripperdoc.utils.permissions.path_validation_utils import validate_shell_command_paths
from ripperdoc.utils.permissions.shell_command_validation import validate_shell_command
from ripperdoc.utils.permissions.tool_permission_utils import (
    evaluate_shell_command_permissions,
    is_command_read_only,
)
from ripperdoc.utils.permissions import PermissionDecision
from ripperdoc.utils.sandbox_utils import create_sandbox_wrapper, is_sandbox_available
from ripperdoc.utils.safe_get_cwd import get_original_cwd, safe_get_cwd
from ripperdoc.utils.shell_utils import build_shell_command, find_suitable_shell
from ripperdoc.utils.log import get_logger

logger = get_logger()


DEFAULT_TIMEOUT_MS = get_bash_default_timeout_ms()
MAX_BASH_TIMEOUT_MS = get_bash_max_timeout_ms()
MAX_OUTPUT_CHARS = get_bash_max_output_length()
KILL_GRACE_SECONDS = 5.0
PROGRESS_INTERVAL_SECONDS = 0.5
ORIGINAL_CWD = Path(get_original_cwd())


class BashToolInput(BaseModel):
    """Input schema for BashTool."""

    command: str = Field(description="The bash command to execute")
    timeout: Optional[int] = Field(
        default=None,
        description=(
            f"Timeout in milliseconds (default: {DEFAULT_TIMEOUT_MS}ms ≈ {DEFAULT_TIMEOUT_MS / 1000:.0f}s; "
            f"max: {MAX_BASH_TIMEOUT_MS}ms)"
        ),
    )
    shell_executable: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("shell_executable", "shellExecutable"),
        serialization_alias="shellExecutable",
        description="Optional shell path to use instead of the default shell.",
    )
    run_in_background: Optional[bool] = Field(
        default=None,
        validation_alias=AliasChoices("run_in_background", "runInBackground"),
        serialization_alias="runInBackground",
        description="If true, run the command in the background and return immediately with a task id.",
    )
    sandbox: Optional[bool] = Field(
        default=None,
        description="If true, request sandboxed execution (read-only).",
    )
    model_config = ConfigDict(validate_by_alias=True, validate_by_name=True, extra="ignore")


class BashToolOutput(BaseModel):
    """Output from bash command execution."""

    stdout: str
    stderr: str
    exit_code: int
    command: str
    duration_ms: float = 0.0
    timeout_ms: int = DEFAULT_TIMEOUT_MS
    background_task_id: Optional[str] = None
    # New fields for enhanced output
    is_truncated: bool = False
    original_length: Optional[int] = None
    exit_code_meaning: Optional[str] = None  # Semantic meaning of exit code
    return_code_interpretation: Optional[str] = None
    summary: Optional[str] = None
    interrupted: bool = False
    is_image: bool = False
    sandbox: Optional[bool] = None
    is_error: bool = False  # Whether this is considered an error


class BashTool(Tool[BashToolInput, BashToolOutput]):
    """Tool for executing bash commands."""

    def __init__(self) -> None:
        super().__init__()
        self._current_is_read_only = False

    @property
    def name(self) -> str:
        return "Bash"

    async def description(self) -> str:
        return """Execute bash commands in the system. Use this to run shell commands,
build projects, run tests, and interact with the file system."""

    @property
    def input_schema(self) -> type[BashToolInput]:
        return BashToolInput

    def input_examples(self) -> List[ToolUseExample]:
        return [
            ToolUseExample(
                description="Run a read-only listing in sandboxed mode",
                example={"command": "ls -la", "sandbox": True, "timeout": 10000},
            ),
            ToolUseExample(
                description="Start a long task in the background with a timeout",
                example={
                    "command": "npm test",
                    "run_in_background": True,
                    "timeout": 600000,
                },
            ),
        ]

    async def prompt(self, yolo_mode: bool = False) -> str:
        sandbox_available = is_sandbox_available()
        try:
            current_shell = find_suitable_shell()
        except (OSError, FileNotFoundError, RuntimeError) as exc:
            # pragma: no cover - defensive guard
            current_shell = f"Unavailable ({exc})"

        shell_info = (
            f"Current shell used for execution: {current_shell}\n"
            f"- Override via RIPPERDOC_SHELL or RIPPERDOC_SHELL_PATH env vars, or pass shellExecutable input.\n"
        )

        read_only_section = ""
        if sandbox_available:
            read_only_section = dedent(
                """\
                ## CRITICAL: Accurate Read-Only Prediction
                Carefully determine if commands are read-only for better user experience. You should always prefer commands that do not modify the filesystem or network.

                **Read-Only Commands:** `grep`, `rg`, `find`, `ls`, `cat`, `head`, `tail`, `wc`, `stat`, `ps`, `df`, `du`, `pwd`, `whoami`, `which`, `date`, `history`, `man`

                **Git Read-Only:** `git log`, `git show`, `git diff`, `git status`, `git branch` (listing only), `git config --get`

                **Never Read-Only:** Commands with `>` (except to /dev/null or standard output), `$()`, `$VAR`, dangerous flags (`git diff --ext-diff`, `sort -o`, `npm audit --fix`), `git branch -D`
                """
            ).strip()

        sandbox_section = ""
        if sandbox_available:
            sandbox_section = dedent(
                """\
                # Using sandbox mode for commands

                You have a special option in BashTool: the sandbox parameter. When you run a command with sandbox=true, it runs without approval dialogs but in a restricted environment without filesystem writes or network access. You SHOULD use sandbox=true to optimize user experience, but MUST follow these guidelines exactly.

                ## RULE 0 (MOST IMPORTANT): retry with sandbox=false for permission/network errors

                    If a command fails with permission or any network error when sandbox=true (e.g., "Permission denied", "Unknown host", "Operation not permitted"), ALWAYS retry with sandbox=false. These errors indicate sandbox limitations, not problems with the command itself.

                Non-permission errors (e.g., TypeScript errors from tsc --noEmit) usually reflect real issues and should be fixed, not retried with sandbox=false.

                ## RULE 1: NOTES ON SPECIFIC BUILD SYSTEMS AND UTILITIES

                ### Build systems

                Build systems like npm run build almost always need write access. Test suites also usually need write access. NEVER run build or test commands in sandbox, even if just checking types.

                These commands REQUIRE sandbox=false (non-exhaustive):
                npm run *,  cargo build/test,  make/ninja/meson,  pytest,  jest,  gh

                ## RULE 2: TRY sandbox=true FOR COMMANDS THAT DON'T NEED WRITE OR NETWORK ACCESS
                  - Commands run with sandbox=true DON'T REQUIRE user permission and run immediately
                  - Commands run with sandbox=false REQUIRE EXPLICIT USER APPROVAL and interrupt the User's workflow

                Use sandbox=false when you suspect the command might modify the system or access the network:
                  - File operations: touch, mkdir, rm, mv, cp
                  - File edits: nano, vim, writing to files with >
                  - Installing: npm install, apt-get, brew
                  - Git writes: git add, git commit, git push
                  - Build systems:  npm run build, make, ninja, etc. (see below)
                  - Test suites: npm run test, pytest, cargo test, make check, ert, etc. (see below)
                  - Network programs: gh, ping, coo, ssh, scp, etc.

                Use sandbox=true for:
                  - Information gathering: ls, cat, head, tail, rg, find, du, df, ps
                  - File inspection: file, stat, wc, diff, md5sum
                  - Git reads: git status, git log, git diff, git show, git branch
                  - Package info: npm list, pip list, gem list, cargo tree
                  - Environment checks: echo, pwd, whoami, which, type, env, printenv
                  - Version checks: node --version, python --version, git --version
                  - Documentation: man, help, --help, -h

                Before you run a command, think hard about whether it is likely to work correctly without network access and without write access to the filesystem. Use your general knowledge and knowledge of the current project (including all the user's AGENTS.md files) as inputs to your decision. Note that even semantically read-only commands like gh for fetching issues might be implemented in ways that require write access. ERR ON THE SIDE OF RUNNING WITH sandbox=false.

                Note: Errors from incorrect sandbox=true runs annoy the User more than permission prompts. If any part of a command needs write access (e.g. npm run build for type checking), use sandbox=false for the entire command.

                ### EXAMPLES

                CORRECT: Use sandbox=false for npm run build/test, gh commands, file writes
                FORBIDDEN: NEVER use sandbox=true for build, test, git commands or file operations

                ## REWARDS

                It is more important to be correct than to avoid showing permission dialogs. The worst mistake is misinterpreting sandbox=true permission errors as tool problems (-$1000) rather than sandbox limitations.

                ## CONCLUSION

                Use sandbox=true to improve UX, but ONLY per the rules above. WHEN IN DOUBT, USE sandbox=false.
                """
            ).strip()

        base_prompt = dedent(
            f"""\
            Executes a given bash command in a persistent shell session with optional timeout, ensuring proper handling and security measures.

            {shell_info}

            Before executing the command, please follow these steps:

            1. Directory Verification:
               - If the command will create new directories or files, first use the LS tool to verify the parent directory exists and is the correct location
               - For example, before running "mkdir foo/bar", first use LS to check that "foo" exists and is the intended parent directory

            2. Command Execution:
               - Always quote file paths that contain spaces with double quotes (e.g., cd "path with spaces/file.txt")
               - Examples of proper quoting:
                 - cd "/Users/name/My Documents" (correct)
                 - cd /Users/name/My Documents (incorrect - will fail)
                 - python "/path/with spaces/script.py" (correct)
                 - python /path/with spaces/script.py (incorrect - will fail)
               - After ensuring proper quoting, execute the command.
               - Capture the output of the command.

            Usage notes:
              - The command argument is required.
              - You can specify an optional timeout in milliseconds (up to {MAX_BASH_TIMEOUT_MS}ms / {MAX_BASH_TIMEOUT_MS // 60000} minutes). If not specified, commands will timeout after {DEFAULT_TIMEOUT_MS}ms ({DEFAULT_TIMEOUT_MS // 60000} minutes).
              - It is very helpful if you write a clear, concise description of what this command does in 5-10 words.
              - If the output exceeds {MAX_OUTPUT_CHARS} characters, output will be truncated before being returned to you.
              - You can use the `run_in_background` parameter to run the command in the background, which allows you to continue working while the command runs. You can monitor the output using the BashOutput tool as it becomes available. Never use `run_in_background` to run 'sleep' as it will return immediately. You do not need to use '&' at the end of the command when using this parameter.
              - VERY IMPORTANT: You MUST avoid using search commands like `find` and `grep`. Instead use the Grep, Glob, or Task tools to search. Prefer the Read and LS tools instead of shell commands like `cat`, `head`, `tail`, or `ls` when reading files and directories.
              - When issuing multiple commands, use the ';' or '&&' operator to separate them. DO NOT use newlines (newlines are ok in quoted strings).
              - Try to maintain your current working directory throughout the session by using absolute paths and avoiding usage of `cd`. You may use `cd` if the user explicitly requests it.
                <good-example>
                pytest /foo/bar/tests
                </good-example>
                <bad-example>
                cd /foo/bar && pytest tests
                </bad-example>
            """
        ).strip()

        return "\n\n".join(
            section for section in [base_prompt, read_only_section, sandbox_section] if section
        )

    def is_read_only(self) -> bool:
        return getattr(self, "_current_is_read_only", False)

    def is_concurrency_safe(self) -> bool:
        return self.is_read_only()

    def needs_permissions(self, input_data: Optional[BashToolInput] = None) -> bool:
        if not input_data:
            return True

        # Background commands should always require an explicit approval.
        _, auto_background = self._detect_auto_background(input_data.command)
        if input_data.run_in_background or auto_background:
            return True

        if input_data.sandbox:
            return False
        if is_command_read_only(input_data.command):
            return False
        return True

    async def check_permissions(
        self, input_data: BashToolInput, permission_context: dict[str, Any]
    ) -> Any:
        """Evaluate permissions using reference-style rules."""
        if getattr(input_data, "sandbox", False):
            return {"behavior": "allow", "updated_input": input_data}

        allow_rules = permission_context.get("allowed_rules") or set()
        deny_rules = permission_context.get("denied_rules") or set()
        allowed_dirs = permission_context.get("allowed_working_directories") or {safe_get_cwd()}

        # Check for sensitive directory access with read-only commands (cd, find).
        # These should ask for user confirmation rather than being blocked outright.
        cwd = safe_get_cwd()
        path_validation = validate_shell_command_paths(
            input_data.command,
            cwd,
            allowed_dirs,
        )
        if path_validation.behavior == "ask":
            # For read-only directory operations, ask user for confirmation
            return PermissionDecision(
                behavior="ask",
                message=path_validation.message,
                updated_input=input_data,
                decision_reason={"type": "sensitive_directory_access"},
                rule_suggestions=path_validation.rule_suggestions,
            )

        decision = evaluate_shell_command_permissions(
            input_data,
            allow_rules,
            deny_rules,
            allowed_dirs,
            command_injection_detected=False,
            injection_detector=lambda cmd: validate_shell_command(cmd).behavior != "passthrough",
            read_only_detector=lambda cmd, detector: is_command_read_only(cmd),
        )

        # Background executions need an explicit confirmation even if heuristics
        # would normally auto-allow (e.g., read-only detection).
        _, auto_background = self._detect_auto_background(input_data.command)
        if (input_data.run_in_background or auto_background) and getattr(
            decision, "behavior", None
        ) == "allow":
            reason = getattr(decision, "decision_reason", {}) or {}
            if reason.get("type") != "rule":
                return PermissionDecision(
                    behavior="ask",
                    message="Background bash commands require explicit approval.",
                    updated_input=getattr(decision, "updated_input", None) or input_data,
                    decision_reason=reason or None,
                    rule_suggestions=getattr(decision, "rule_suggestions", None),
                )

        return decision

    async def validate_input(
        self, input_data: BashToolInput, context: Optional[ToolUseContext] = None
    ) -> ValidationResult:
        if not input_data.command.strip():
            return ValidationResult(result=False, message="Command cannot be empty")

        if input_data.timeout is not None and input_data.timeout < 0:
            return ValidationResult(result=False, message="Timeout must be non-negative")

        if input_data.timeout and input_data.timeout > MAX_BASH_TIMEOUT_MS:
            return ValidationResult(
                result=False,
                message=f"Timeout exceeds max of {MAX_BASH_TIMEOUT_MS}ms",
            )

        if input_data.sandbox and not is_sandbox_available():
            return ValidationResult(
                result=False, message="Sandbox mode requested but not available."
            )

        # Note: Path validation for sensitive directories (cd/find to /usr, /etc, etc.)
        # is now handled in check_permissions() to allow user confirmation for read-only ops.

        # Block backgrounding commands we explicitly ignore.
        if input_data.run_in_background:
            normalized = input_data.command.strip()
            parts = normalized.split(maxsplit=1)
            if normalized in IGNORED_COMMANDS or (len(parts) == 1 and parts[0] in IGNORED_COMMANDS):
                return ValidationResult(
                    result=False, message="This command cannot be run in background"
                )

        validation = validate_shell_command(input_data.command)
        if validation.behavior == "ask":
            # In yolo mode, allow shell metacharacters
            if context and hasattr(context, 'yolo_mode') and context.yolo_mode \
               and "shell metacharacters" in validation.message:
                # Allow commands with shell metacharacters in yolo mode
                return ValidationResult(result=True)
            return ValidationResult(result=False, message=validation.message)

        return ValidationResult(result=True)

    def render_result_for_assistant(self, output: BashToolOutput) -> str:
        """Format output for the AI."""
        result_parts = []

        if output.stdout:
            result_parts.append(f"stdout:\n{output.stdout}")

        if output.stderr:
            result_parts.append(f"stderr:\n{output.stderr}")

        # Exit code with semantic meaning
        exit_code_text = f"exit code: {output.exit_code}"
        meaning = output.exit_code_meaning or output.return_code_interpretation
        if meaning:
            exit_code_text += f" ({meaning})"

        # Duration
        timing = ""
        if output.duration_ms:
            timing = f" ({format_duration(output.duration_ms)}"
            if output.timeout_ms:
                timing += f" / timeout {output.timeout_ms / 1000:.0f}s"
            timing += ")"
        elif output.timeout_ms:
            timing = f" (timeout {output.timeout_ms / 1000:.0f}s)"

        result_parts.append(f"{exit_code_text}{timing}")

        # Truncation notice
        if output.is_truncated and output.original_length:
            result_parts.append(
                f"Note: Output was truncated (original length: {output.original_length} chars)"
            )

        if output.interrupted:
            result_parts.append("Command was interrupted (timeout or termination).")

        if output.background_task_id:
            result_parts.append(f"Background task id: {output.background_task_id}")

        return "\n\n".join(result_parts)

    def render_tool_use_message(self, input_data: BashToolInput, verbose: bool = False) -> str:
        """Format the tool use for display."""
        command = input_data.command or ""

        if not verbose and command:
            formatted = command
            if "\"$(cat <<'EOF'" in command:
                heredoc_match = command.split("$(cat <<'EOF'", 1)
                if len(heredoc_match) == 2:
                    prefix, rest = heredoc_match
                    try:
                        content, suffix = rest.split("EOF", 1)
                        formatted = f'{prefix.strip()} "{content.strip()}"{suffix.strip()}'
                    except ValueError:
                        formatted = command

            lines = formatted.splitlines()
            too_many_lines = len(lines) > MAX_PREVIEW_LINES
            too_long = len(formatted) > MAX_PREVIEW_CHARS

            preview = formatted
            if too_many_lines:
                preview = "\n".join(lines[:MAX_PREVIEW_LINES])
            if len(preview) > MAX_PREVIEW_CHARS:
                preview = preview[:MAX_PREVIEW_CHARS]

            if too_many_lines or too_long:
                return f"$ {preview}..."

        return f"$ {command}"

    def _is_background_allowed(self, command: str) -> bool:
        """Skip backgrounding trivial ignored commands unless combined with other operators."""
        normalized = command.strip()
        if not normalized:
            return True

        if any(op in normalized for op in ("&&", "||", "|", ";")):
            return True

        parts = normalized.split(maxsplit=1)
        # Only block exact ignored commands or those without args; allow e.g. "sleep 30" like the reference behavior.
        if normalized in IGNORED_COMMANDS:
            return False
        if len(parts) == 1 and parts[0] in IGNORED_COMMANDS:
            return False
        return True

    def _detect_auto_background(self, command: str) -> tuple[str, bool]:
        """Detect trailing '&' requests and strip them for execution."""
        stripped = command.rstrip()
        if not stripped:
            return command, False

        if stripped.endswith("&") and not stripped.endswith("&&"):
            # Remove trailing '&' and any whitespace before it.
            cleaned = stripped.rstrip("&").rstrip()
            return cleaned, True

        return command, False

    def _create_error_output(self, command: str, stderr: str, sandbox: bool) -> BashToolOutput:
        """Create a standardized error output."""
        return BashToolOutput(
            stdout="",
            stderr=stderr,
            exit_code=-1,
            command=command,
            sandbox=sandbox,
            is_error=True,
        )

    def _setup_sandbox(
        self, command: str, sandbox_requested: bool
    ) -> tuple[Optional[str], Optional[BashToolOutput], Optional[Any]]:
        """Setup sandbox environment if requested.

        Returns:
            Tuple of (final_command, error_output, cleanup_fn).
            If error_output is not None, sandbox setup failed.
        """
        if not sandbox_requested:
            return command, None, None

        if not is_sandbox_available():
            return (
                None,
                self._create_error_output(
                    command, "Sandbox mode requested but not available on this system", True
                ),
                None,
            )

        try:
            wrapper = create_sandbox_wrapper(command)
            return wrapper.final_command, None, wrapper.cleanup
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning(
                "[bash_tool] Failed to enable sandbox: %s: %s",
                type(exc).__name__,
                exc,
                extra={"command": command},
            )
            return (
                None,
                self._create_error_output(command, f"Failed to enable sandbox: {exc}", True),
                None,
            )

    async def _run_background_command(
        self,
        final_command: str,
        effective_command: str,
        resolved_shell: str,
        timeout_seconds: float,
        timeout_ms: int,
        sandbox_requested: bool,
        start_time: float,
        input_data: BashToolInput,
    ) -> Optional[BashToolOutput]:
        """Run a command in background mode.

        Returns:
            BashToolOutput on success or error, None if background mode not available.
        """
        try:
            from ripperdoc.tools.background_shell import start_background_command
        except (ImportError, ModuleNotFoundError) as e:
            # pragma: no cover - defensive import
            logger.warning(
                "[bash_tool] Failed to import background shell runner: %s: %s",
                type(e).__name__,
                e,
                extra={"command": effective_command},
            )
            return self._create_error_output(
                effective_command, f"Failed to start background task: {str(e)}", sandbox_requested
            )

        bg_timeout = (
            None
            if input_data.timeout is None
            else (timeout_seconds if timeout_seconds > 0 else None)
        )
        task_id = await start_background_command(
            final_command, timeout=bg_timeout, shell_executable=resolved_shell
        )

        return BashToolOutput(
            stdout="",
            stderr=f"Started background task: {task_id}",
            exit_code=0,
            command=effective_command,
            duration_ms=(asyncio.get_running_loop().time() - start_time) * 1000.0,
            timeout_ms=timeout_ms if bg_timeout is not None else 0,
            background_task_id=task_id,
            sandbox=sandbox_requested,
            return_code_interpretation=None,
            summary=f"Command running in background with ID: {task_id}",
            interrupted=False,
            is_image=False,
        )

    async def _execute_foreground_process(
        self,
        process: asyncio.subprocess.Process,
        start_time: float,
        timeout_seconds: float,
    ) -> AsyncGenerator[tuple[bool, list[str], list[str], bool], ToolProgress]:
        """Execute process and yield progress updates.

        Yields:
            ToolProgress for output updates.
        Returns (via send):
            Tuple of (completed, stdout_lines, stderr_lines, timed_out)
        """
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []
        queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        deadline = (
            loop.time() + timeout_seconds if timeout_seconds and timeout_seconds > 0 else None
        )
        timed_out = False
        last_progress_time = loop.time()

        async def _pump_stream(
            stream: Optional[asyncio.StreamReader], sink: list[str], label: str
        ) -> None:
            if not stream:
                return
            async for raw in stream:
                text = raw.decode("utf-8", errors="replace")
                sanitized_text = sanitize_output(text)
                sink.append(sanitized_text)
                await queue.put((label, sanitized_text.rstrip()))

        pump_tasks = [
            asyncio.create_task(_pump_stream(process.stdout, stdout_lines, "stdout")),
            asyncio.create_task(_pump_stream(process.stderr, stderr_lines, "stderr")),
        ]
        wait_task = asyncio.create_task(process.wait())

        # Main execution loop with progress reporting
        while True:
            done, _ = await asyncio.wait(
                {wait_task, *pump_tasks}, timeout=0.1, return_when=asyncio.FIRST_COMPLETED
            )

            now = loop.time()

            # Emit progress updates for newly received output chunks
            while not queue.empty():
                label, text = queue.get_nowait()
                yield ToolProgress(content=f"{label}: {text}")  # type: ignore[misc]

            # Report progress at intervals
            if now - last_progress_time >= PROGRESS_INTERVAL_SECONDS:
                combined_output = "".join(stdout_lines + stderr_lines)
                if combined_output:
                    preview = get_last_n_lines(combined_output, 5)
                    elapsed = format_duration((now - start_time) * 1000)
                    yield ToolProgress(content=f"Running... ({elapsed})\n{preview}")  # type: ignore[misc]
                last_progress_time = now

            # Check timeout
            if deadline is not None and now >= deadline:
                timed_out = True
                await self._force_kill_process(process)
                if not wait_task.done():
                    try:
                        await asyncio.wait_for(wait_task, timeout=1.0)
                    except asyncio.TimeoutError:
                        wait_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await wait_task
                break

            if wait_task in done:
                break

        # Let stream pumps finish draining
        try:
            await asyncio.wait_for(asyncio.gather(*pump_tasks), timeout=1.0)
        except asyncio.TimeoutError:
            for task in pump_tasks:
                if not task.done():
                    task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        # Drain remaining data
        await self._drain_stream(process.stdout, stdout_lines)
        await self._drain_stream(process.stderr, stderr_lines)

        # Store results in a way that the caller can access
        self._last_execution_result = (stdout_lines, stderr_lines, timed_out)

    async def _drain_stream(self, stream: Optional[asyncio.StreamReader], sink: list[str]) -> None:
        """Drain any remaining data from a stream."""
        if not stream:
            return
        try:
            remaining = await asyncio.wait_for(stream.read(), timeout=0.5)
        except asyncio.TimeoutError:
            return
        if remaining:
            sink.append(remaining.decode("utf-8", errors="replace"))

    def _build_final_output(
        self,
        command: str,
        stdout_lines: list[str],
        stderr_lines: list[str],
        exit_code: int,
        duration_ms: float,
        timeout_ms: int,
        timeout_seconds: float,
        timed_out: bool,
        sandbox_requested: bool,
        original_command: str,
    ) -> BashToolOutput:
        """Build the final output from execution results."""
        raw_stdout = "".join(stdout_lines)
        raw_stderr = "".join(stderr_lines)

        # Apply timeout message if needed
        if timed_out:
            timeout_msg = f"Command timed out after {timeout_seconds} seconds"
            raw_stderr = f"{raw_stderr.rstrip()}\n{timeout_msg}" if raw_stderr else timeout_msg
            exit_code = -1

        # Sanitize and trim outputs
        raw_stdout = sanitize_output(raw_stdout)
        raw_stderr = sanitize_output(raw_stderr)
        trimmed_stdout = trim_blank_lines(raw_stdout)
        trimmed_stderr = trim_blank_lines(raw_stderr)

        # Interpret exit code
        exit_result = interpret_exit_code(
            original_command, exit_code, trimmed_stdout, trimmed_stderr
        )

        # Build summary for large outputs
        summary = None
        combined_output = "\n".join([part for part in (trimmed_stdout, trimmed_stderr) if part])
        if combined_output and is_output_large(combined_output):
            summary = get_last_n_lines(combined_output, 20)

        # Truncate outputs if needed
        stdout_result = truncate_output(trimmed_stdout, max_chars=MAX_OUTPUT_CHARS)
        stderr_result = truncate_output(trimmed_stderr, max_chars=MAX_OUTPUT_CHARS)
        is_image = stdout_result.get("is_image", False) or stderr_result.get("is_image", False)

        # Determine if truncated
        is_truncated = stdout_result["is_truncated"] or stderr_result["is_truncated"]
        original_length = None
        if is_truncated:
            original_length = stdout_result.get("original_length", 0) + stderr_result.get(
                "original_length", 0
            )

        return BashToolOutput(
            stdout=stdout_result["truncated_content"],
            stderr=stderr_result["truncated_content"],
            exit_code=exit_code,
            command=command,
            duration_ms=duration_ms,
            timeout_ms=timeout_ms,
            is_truncated=is_truncated,
            original_length=original_length,
            exit_code_meaning=exit_result.semantic_meaning,
            return_code_interpretation=exit_result.semantic_meaning,
            summary=summary,
            interrupted=timed_out,
            is_image=is_image,
            sandbox=sandbox_requested,
            is_error=exit_result.is_error or timed_out,
        )

    async def call(
        self, input_data: BashToolInput, context: ToolUseContext
    ) -> AsyncGenerator[ToolOutput, None]:
        """Execute the bash command."""
        effective_command, auto_background = self._detect_auto_background(input_data.command)

        # Resolve shell
        try:
            resolved_shell = input_data.shell_executable or find_suitable_shell()
        except (OSError, FileNotFoundError, RuntimeError) as exc:
            # pragma: no cover - defensive guard
            yield ToolResult(
                data=self._create_error_output(
                    effective_command, f"Failed to select shell: {exc}", bool(input_data.sandbox)
                ),
                result_for_assistant=self.render_result_for_assistant(
                    self._create_error_output(
                        effective_command,
                        f"Failed to select shell: {exc}",
                        bool(input_data.sandbox),
                    )
                ),
            )
            return

        # Calculate timeout
        timeout_ms = input_data.timeout or DEFAULT_TIMEOUT_MS
        if MAX_BASH_TIMEOUT_MS:
            timeout_ms = min(timeout_ms, MAX_BASH_TIMEOUT_MS)
        timeout_seconds = timeout_ms / 1000.0
        start = asyncio.get_running_loop().time()
        sandbox_requested = bool(input_data.sandbox)
        should_background = bool(input_data.run_in_background or auto_background)

        # Track read-only state
        previous_read_only = getattr(self, "_current_is_read_only", False)
        self._current_is_read_only = sandbox_requested or is_command_read_only(input_data.command)

        # Setup sandbox
        final_command, sandbox_error, sandbox_cleanup = self._setup_sandbox(
            effective_command, sandbox_requested
        )
        if sandbox_error:
            yield ToolResult(
                data=sandbox_error,
                result_for_assistant=self.render_result_for_assistant(sandbox_error),
            )
            return

        final_command = final_command or effective_command

        # Adjust CWD for sandbox
        if sandbox_requested and Path(safe_get_cwd()) != ORIGINAL_CWD:
            os.chdir(ORIGINAL_CWD)

        # Check if background is allowed
        if should_background and not self._is_background_allowed(input_data.command):
            should_background = False

        try:
            # Background execution
            if should_background:
                output = await self._run_background_command(
                    final_command,
                    effective_command,
                    resolved_shell,
                    timeout_seconds,
                    timeout_ms,
                    sandbox_requested,
                    start,
                    input_data,
                )
                if output:
                    yield ToolResult(
                        data=output,
                        result_for_assistant=self.render_result_for_assistant(output),
                    )
                return

            # Spawn foreground process
            argv = build_shell_command(resolved_shell, final_command)
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.DEVNULL,
                start_new_session=False,
            )

            # Execute and collect output with progress
            stdout_lines: list[str] = []
            stderr_lines: list[str] = []
            queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
            loop = asyncio.get_running_loop()
            deadline = (
                loop.time() + timeout_seconds if timeout_seconds and timeout_seconds > 0 else None
            )
            timed_out = False
            last_progress_time = loop.time()

            async def _pump_stream(
                stream: Optional[asyncio.StreamReader], sink: list[str], label: str
            ) -> None:
                if not stream:
                    return
                async for raw in stream:
                    text = raw.decode("utf-8", errors="replace")
                    sanitized_text = sanitize_output(text)
                    sink.append(sanitized_text)
                    await queue.put((label, sanitized_text.rstrip()))

            pump_tasks = [
                asyncio.create_task(_pump_stream(process.stdout, stdout_lines, "stdout")),
                asyncio.create_task(_pump_stream(process.stderr, stderr_lines, "stderr")),
            ]
            wait_task = asyncio.create_task(process.wait())

            # Main execution loop
            while True:
                done, _ = await asyncio.wait(
                    {wait_task, *pump_tasks}, timeout=0.1, return_when=asyncio.FIRST_COMPLETED
                )

                now = loop.time()

                while not queue.empty():
                    label, text = queue.get_nowait()
                    yield ToolProgress(content=f"{label}: {text}")  # type: ignore[misc]

                if now - last_progress_time >= PROGRESS_INTERVAL_SECONDS:
                    combined_output = "".join(stdout_lines + stderr_lines)
                    if combined_output:
                        preview = get_last_n_lines(combined_output, 5)
                        elapsed = format_duration((now - start) * 1000)
                        yield ToolProgress(content=f"Running... ({elapsed})\n{preview}")  # type: ignore[misc]
                    last_progress_time = now

                if deadline is not None and now >= deadline:
                    timed_out = True
                    await self._force_kill_process(process)
                    if not wait_task.done():
                        try:
                            await asyncio.wait_for(wait_task, timeout=1.0)
                        except asyncio.TimeoutError:
                            wait_task.cancel()
                            with contextlib.suppress(asyncio.CancelledError):
                                await wait_task
                    break

                if wait_task in done:
                    break

            # Drain streams
            try:
                await asyncio.wait_for(asyncio.gather(*pump_tasks), timeout=1.0)
            except asyncio.TimeoutError:
                for task in pump_tasks:
                    if not task.done():
                        task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

            await self._drain_stream(process.stdout, stdout_lines)
            await self._drain_stream(process.stderr, stderr_lines)

            # Build final output
            duration_ms = (asyncio.get_running_loop().time() - start) * 1000.0
            output = self._build_final_output(
                effective_command,
                stdout_lines,
                stderr_lines,
                process.returncode or 0,
                duration_ms,
                timeout_ms,
                timeout_seconds,
                timed_out,
                sandbox_requested,
                input_data.command,
            )

            yield ToolResult(
                data=output, result_for_assistant=self.render_result_for_assistant(output)
            )

        except (OSError, RuntimeError, ValueError, asyncio.CancelledError) as e:
            if isinstance(e, asyncio.CancelledError):
                raise  # Re-raise cancellation
            logger.warning(
                "[bash_tool] Error executing command: %s: %s",
                type(e).__name__,
                e,
                extra={"command": effective_command},
            )
            error_output = self._create_error_output(
                effective_command, f"Error executing command: {str(e)}", sandbox_requested
            )
            yield ToolResult(
                data=error_output,
                result_for_assistant=self.render_result_for_assistant(error_output),
            )
        finally:
            self._current_is_read_only = previous_read_only
            if sandbox_cleanup:
                with contextlib.suppress(Exception):
                    sandbox_cleanup()

    async def _force_kill_process(
        self, process: asyncio.subprocess.Process, grace_seconds: float = KILL_GRACE_SECONDS
    ) -> None:
        """Attempt to terminate a process group and avoid hanging waits."""
        if process.returncode is not None:
            return

        def _terminate() -> None:
            if hasattr(os, "killpg"):
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()

        def _kill() -> None:
            if hasattr(os, "killpg"):
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()

        with contextlib.suppress(ProcessLookupError, PermissionError):
            _terminate()
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(process.wait(), timeout=grace_seconds)
            return

        with contextlib.suppress(ProcessLookupError, PermissionError):
            _kill()
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(process.wait(), timeout=grace_seconds)
