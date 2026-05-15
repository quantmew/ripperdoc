"""Bash command execution tool."""

from __future__ import annotations

import asyncio
import contextlib
import os
from pathlib import Path
from textwrap import dedent
from typing import AsyncGenerator, List, Optional

from ripperdoc.core.tool import (
    Tool,
    ToolOutput,
    ToolProgress,
    ToolResult,
    ToolUseContext,
    ToolUseExample,
    ValidationResult,
)
from ripperdoc.utils.log import get_logger
from ripperdoc.utils.shell.bash_constants import (
    get_bash_default_timeout_ms,
    get_bash_max_timeout_ms,
    get_bash_max_output_length,
)
from ripperdoc.utils.shell.exit_code_handlers import IGNORED_COMMANDS
from ripperdoc.utils.shell.output_utils import format_duration, get_last_n_lines, sanitize_output
from ripperdoc.utils.shell.sandbox_utils import is_sandbox_available
from ripperdoc.utils.permissions.tool_permission_utils import is_command_read_only
from ripperdoc.utils.shell.shell_utils import build_shell_command, find_suitable_shell
from ripperdoc.utils.collaboration.task_notifications import enqueue_task_notification
from ripperdoc.utils.filesystem.safe_get_cwd import get_original_cwd, safe_get_cwd
from ripperdoc.tools.bash._prompt import BASH_PROMPT
from ripperdoc.tools.bash._models import (
    BashToolInput,
    BashToolOutput,
    DEFAULT_TIMEOUT_MS,
    MAX_BASH_TIMEOUT_MS,
    MAX_OUTPUT_CHARS,
)
from ripperdoc.tools.bash._sandbox import setup_sandbox
from ripperdoc.tools.bash._permissions import (
    check_permissions,
    detect_auto_background,
    is_background_allowed,
)
from ripperdoc.tools.bash._output import (
    build_background_launch_output,
    build_final_output,
    render_result_for_assistant,
)
from ripperdoc.tools.bash._process import (
    execute_foreground_process,
    force_kill_process,
    drain_stream,
    KILL_GRACE_SECONDS,
    PROGRESS_INTERVAL_SECONDS,
    STREAM_READ_CHUNK_SIZE,
)

logger = get_logger()

ORIGINAL_CWD = Path(get_original_cwd())


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

        return "\n\n".join(
            section for section in [BASH_PROMPT, read_only_section, sandbox_section] if section
        )

    def is_read_only(self) -> bool:
        return getattr(self, "_current_is_read_only", False)

    def is_concurrency_safe(self) -> bool:
        return self.is_read_only()

    def is_concurrency_safe_for_input(self, input_data: Optional[BashToolInput] = None) -> bool:
        if input_data is None:
            return self.is_concurrency_safe()
        sandbox_requested = bool(input_data.sandbox) and not bool(
            input_data.dangerously_disable_sandbox
        )
        return sandbox_requested or is_command_read_only(input_data.command)

    def needs_permissions(self, input_data: Optional[BashToolInput] = None) -> bool:
        if not input_data:
            return True

        _, auto_background = detect_auto_background(input_data.command)
        if input_data.run_in_background or auto_background:
            return True

        sandbox_requested = bool(input_data.sandbox) and not bool(
            input_data.dangerously_disable_sandbox
        )
        if sandbox_requested:
            return False
        if is_command_read_only(input_data.command):
            return False
        return True

    async def check_permissions(
        self, input_data: BashToolInput, permission_context: dict[str, Any]
    ) -> Any:
        return await check_permissions(input_data, permission_context)

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

        sandbox_requested = bool(input_data.sandbox) and not bool(
            input_data.dangerously_disable_sandbox
        )
        if sandbox_requested and not is_sandbox_available():
            return ValidationResult(
                result=False, message="Sandbox mode requested but not available."
            )

        if input_data.shell_executable:
            shell_path = Path(input_data.shell_executable)
            if not shell_path.is_absolute():
                return ValidationResult(
                    result=False,
                    message=f"shell_executable must be an absolute path: {input_data.shell_executable}",
                )
            if not shell_path.exists():
                return ValidationResult(
                    result=False,
                    message=f"shell_executable not found: {input_data.shell_executable}",
                )
            if not shell_path.is_file():
                return ValidationResult(
                    result=False,
                    message=f"shell_executable is not a file: {input_data.shell_executable}",
                )
            if not os.access(shell_path, os.X_OK):
                return ValidationResult(
                    result=False,
                    message=f"shell_executable is not executable: {input_data.shell_executable}",
                )
            safe_dirs = {"/bin", "/usr/bin", "/usr/local/bin", "/opt/homebrew/bin"}
            shell_name = shell_path.name.lower()
            known_shells = {"bash", "sh", "zsh", "fish", "dash", "ksh", "tcsh", "csh"}
            parent_dir = str(shell_path.parent)
            if parent_dir not in safe_dirs and shell_name not in known_shells:
                return ValidationResult(
                    result=False,
                    message=f"shell_executable must be a known shell in a standard location: {input_data.shell_executable}",
                )

        if input_data.run_in_background:
            normalized = input_data.command.strip()
            parts = normalized.split(maxsplit=1)
            if normalized in IGNORED_COMMANDS or (len(parts) == 1 and parts[0] in IGNORED_COMMANDS):
                return ValidationResult(
                    result=False, message="This command cannot be run in background"
                )

        return ValidationResult(result=True)

    def render_result_for_assistant(self, output: BashToolOutput) -> str:
        return render_result_for_assistant(output)

    def render_tool_use_message(self, input_data: BashToolInput, verbose: bool = False) -> str:
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

            from ripperdoc.utils.shell.exit_code_handlers import MAX_PREVIEW_CHARS, MAX_PREVIEW_LINES
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

    def _build_background_completion_callbacks(
        self,
        *,
        context: Optional[ToolUseContext],
        effective_command: str,
    ) -> Optional[list[Any]]:
        queue = getattr(context, "task_notification_queue", None) if context else None
        if queue is None:
            return None

        def _notify_completion(task: Any) -> None:
            if getattr(task, "notification_sent", False):
                return
            setattr(task, "notification_sent", True)
            if getattr(task, "killed", False):
                status = "killed"
            elif getattr(task, "timed_out", False):
                status = "failed"
            else:
                exit_code = getattr(task, "exit_code", None)
                status = "running" if exit_code is None else ("completed" if exit_code == 0 else "failed")
            exit_code = getattr(task, "exit_code", None)
            status_line = f"Background bash task finished with status: {status}"
            if exit_code is not None:
                status_line += f" (exit code {exit_code})"
            enqueue_task_notification(
                queue,
                task_id=str(getattr(task, "id", "") or ""),
                status=status,
                summary=(
                    f"{status_line}. "
                    "Use TaskOutput with this task id to read stdout/stderr and continue."
                ),
                tool_use_id=getattr(context, "message_id", None) if context else None,
                source="background_bash",
                extra_metadata={
                    "background_task_id": getattr(task, "id", None),
                    "exit_code": exit_code,
                    "command": effective_command,
                },
            )

        return [_notify_completion]

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
        context: Optional[ToolUseContext] = None,
        working_directory: Optional[str] = None,
    ) -> Optional[BashToolOutput]:
        try:
            from ripperdoc.tools.background_shell import start_background_command
        except (ImportError, ModuleNotFoundError) as e:
            logger.warning(
                "[bash_tool] Failed to import background shell runner: %s: %s",
                type(e).__name__, e,
                extra={"command": effective_command},
            )
            from ripperdoc.tools.bash._process import _create_error_output
            return _create_error_output(
                effective_command, f"Failed to start background task: {str(e)}", sandbox_requested
            )

        bg_timeout = None
        completion_callbacks = self._build_background_completion_callbacks(
            context=context, effective_command=effective_command,
        )

        task_id = await start_background_command(
            final_command,
            timeout=bg_timeout,
            shell_executable=resolved_shell,
            cwd=working_directory,
            completion_callbacks=completion_callbacks,
        )

        return build_background_launch_output(
            effective_command=effective_command,
            task_id=task_id,
            start_time=start_time,
            sandbox_requested=sandbox_requested,
            status_message=f"Started background task: {task_id}",
        )

    async def _auto_background_foreground_process(
        self,
        *,
        process: asyncio.subprocess.Process,
        effective_command: str,
        start_time: float,
        sandbox_requested: bool,
        context: Optional[ToolUseContext],
        stdout_lines: list[str],
        stderr_lines: list[str],
        pump_tasks: list[asyncio.Task[Any]],
    ) -> Optional[BashToolOutput]:
        try:
            from ripperdoc.tools.background_shell import register_existing_process
        except (ImportError, ModuleNotFoundError) as exc:
            logger.warning(
                "[bash_tool] Failed to import process adoption helper: %s: %s",
                type(exc).__name__, exc,
                extra={"command": effective_command},
            )
            return None

        completion_callbacks = self._build_background_completion_callbacks(
            context=context, effective_command=effective_command,
        )
        task_id = await register_existing_process(
            effective_command,
            process,
            timeout=None,
            stdout_chunks=stdout_lines,
            stderr_chunks=stderr_lines,
            reader_tasks=pump_tasks,
            completion_callbacks=completion_callbacks,
        )
        return build_background_launch_output(
            effective_command=effective_command,
            task_id=task_id,
            start_time=start_time,
            sandbox_requested=sandbox_requested,
            status_message=f"Foreground execution exceeded timeout and was moved to background: {task_id}",
        )

    async def call(
        self, input_data: BashToolInput, context: ToolUseContext
    ) -> AsyncGenerator[ToolOutput, None]:
        effective_command, auto_background = detect_auto_background(input_data.command)

        try:
            resolved_shell = input_data.shell_executable or find_suitable_shell()
        except (OSError, FileNotFoundError, RuntimeError) as exc:
            from ripperdoc.tools.bash._process import _create_error_output
            yield ToolResult(
                data=_create_error_output(
                    effective_command,
                    f"Failed to select shell: {exc}",
                    bool(input_data.sandbox) and not bool(input_data.dangerously_disable_sandbox),
                ),
                result_for_assistant=render_result_for_assistant(
                    _create_error_output(
                        effective_command,
                        f"Failed to select shell: {exc}",
                        bool(input_data.sandbox) and not bool(input_data.dangerously_disable_sandbox),
                    )
                ),
            )
            return

        timeout_ms = input_data.timeout or DEFAULT_TIMEOUT_MS
        if MAX_BASH_TIMEOUT_MS:
            timeout_ms = min(timeout_ms, MAX_BASH_TIMEOUT_MS)
        timeout_seconds = timeout_ms / 1000.0
        start = asyncio.get_running_loop().time()
        sandbox_requested = bool(input_data.sandbox) and not bool(
            input_data.dangerously_disable_sandbox
        )
        should_background = bool(input_data.run_in_background or auto_background)

        previous_read_only = getattr(self, "_current_is_read_only", False)
        self._current_is_read_only = sandbox_requested or is_command_read_only(input_data.command)

        final_command, sandbox_error, sandbox_cleanup = setup_sandbox(
            effective_command, sandbox_requested
        )
        if sandbox_error:
            yield ToolResult(
                data=sandbox_error,
                result_for_assistant=render_result_for_assistant(sandbox_error),
            )
            return

        final_command = final_command or effective_command

        if sandbox_requested and Path(safe_get_cwd()) != ORIGINAL_CWD:
            os.chdir(ORIGINAL_CWD)

        if should_background and not is_background_allowed(input_data.command):
            should_background = False

        try:
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
                    context,
                    working_directory=(context.working_directory if context else None),
                )
                if output:
                    yield ToolResult(
                        data=output,
                        result_for_assistant=render_result_for_assistant(output),
                    )
                return

            argv = build_shell_command(resolved_shell, final_command)
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.DEVNULL,
                start_new_session=False,
                cwd=(context.working_directory if context and context.working_directory else None),
            )

            stdout_lines: list[str] = []
            stderr_lines: list[str] = []
            timed_out = False

            async for event in execute_foreground_process(process, start, timeout_seconds):
                if isinstance(event, ToolProgress):
                    yield event
                elif isinstance(event, (list, tuple)):
                    stdout_lines, stderr_lines, timed_out = event

            if isinstance(event, ToolProgress):
                pass
            else:
                stdout_lines, stderr_lines, timed_out = event

            auto_background_output: Optional[BashToolOutput] = None
            if timed_out and is_background_allowed(input_data.command):
                auto_background_output = await self._auto_background_foreground_process(
                    process=process,
                    effective_command=effective_command,
                    start_time=start,
                    sandbox_requested=sandbox_requested,
                    context=context,
                    stdout_lines=stdout_lines,
                    stderr_lines=stderr_lines,
                    pump_tasks=[],
                )

            if auto_background_output is not None:
                yield ToolResult(
                    data=auto_background_output,
                    result_for_assistant=render_result_for_assistant(auto_background_output),
                )
                return

            duration_ms = (asyncio.get_running_loop().time() - start) * 1000.0
            output = build_final_output(
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
                data=output, result_for_assistant=render_result_for_assistant(output)
            )

        except (OSError, RuntimeError, ValueError, asyncio.CancelledError) as e:
            if isinstance(e, asyncio.CancelledError):
                raise
            logger.warning(
                "[bash_tool] Error executing command: %s: %s",
                type(e).__name__, e,
                extra={"command": effective_command},
            )
            from ripperdoc.tools.bash._process import _create_error_output
            error_output = _create_error_output(
                effective_command, f"Error executing command: {str(e)}", sandbox_requested
            )
            yield ToolResult(
                data=error_output,
                result_for_assistant=render_result_for_assistant(error_output),
            )
        finally:
            self._current_is_read_only = previous_read_only
            if sandbox_cleanup:
                with contextlib.suppress(OSError, IOError, PermissionError):
                    sandbox_cleanup()
