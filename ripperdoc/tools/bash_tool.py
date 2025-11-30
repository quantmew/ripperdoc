"""Bash command execution tool.

Allows the AI to execute bash commands in the system.
"""

import asyncio
import contextlib
import os
import signal
from textwrap import dedent
from typing import AsyncGenerator, Optional
from pydantic import BaseModel, Field

from ripperdoc.core.tool import Tool, ToolUseContext, ToolResult, ToolProgress, ToolOutput, ValidationResult


DEFAULT_TIMEOUT_MS = 20000
KILL_GRACE_SECONDS = 5.0


class BashToolInput(BaseModel):
    """Input schema for BashTool."""
    command: str = Field(description="The bash command to execute")
    timeout: Optional[int] = Field(
        default=DEFAULT_TIMEOUT_MS,
        description=f"Timeout in milliseconds (default: {DEFAULT_TIMEOUT_MS}ms = {DEFAULT_TIMEOUT_MS / 60000:.0f} minutes)"
    )
    run_in_background: bool = Field(
        default=False,
        description="If true, run the command in the background and return immediately with a task id."
    )


class BashToolOutput(BaseModel):
    """Output from bash command execution."""
    stdout: str
    stderr: str
    exit_code: int
    command: str
    duration_ms: float = 0.0
    timeout_ms: int = DEFAULT_TIMEOUT_MS
    background_task_id: Optional[str] = None


class BashTool(Tool[BashToolInput, BashToolOutput]):
    """Tool for executing bash commands."""

    @property
    def name(self) -> str:
        return "Bash"

    async def description(self) -> str:
        return """Execute bash commands in the system. Use this to run shell commands,
build projects, run tests, and interact with the file system."""

    @property
    def input_schema(self) -> type[BashToolInput]:
        return BashToolInput

    async def prompt(self, safe_mode: bool = False) -> str:
        return dedent(
            f"""\
            Execute bash commands with optional timeout and background execution.

            Before executing a command:
            - If the command will create new directories or files, first use the LS tool to verify the parent directory exists and is correct (for example, before running "mkdir foo/bar", check "foo" with LS).
            - Always quote file paths that contain spaces using double quotes (for example, cd "path with spaces/file.txt"); avoid unquoted paths with spaces.
            - Maintain your working directory by using absolute paths and avoid using cd unless the user explicitly requests it.

            Usage notes:
            - The command argument is required. Timeout defaults to {DEFAULT_TIMEOUT_MS}ms ({DEFAULT_TIMEOUT_MS / 1000:.0f}s); set a custom timeout when needed.
            - Provide a concise 5-10 word description of what the command does when you run it.
            - Use run_in_background=true for long-running commands; poll results with the BashOutput tool and stop jobs with the KillBash tool. Do not background trivial sleeps.
            - When issuing multiple commands, separate them with ';' or '&&' instead of newlines (newlines are allowed inside quoted strings).
            - Avoid search or inspection commands in Bash such as find, grep, cat, head, tail, or ls. Use the Glob or Grep tools for search and the View tool to read files. If you must search from Bash, prefer ripgrep (rg) over plain grep.

            Git workflows:
            - For commits, first inspect the repo with git status/diff/log, draft an intent-focused message, commit with a heredoc message body, and check git status afterward. Do not change git config or push unless requested. If hooks modify files, retry the commit once to include those changes.
            - For pull requests, gather context with git status/diff/log and base diffs, ensure the branch is up to date, push if needed, and create the PR with gh pr create using a heredoc body. Return the PR URL.
            """
        )

    def is_read_only(self) -> bool:
        return False

    def is_concurrency_safe(self) -> bool:
        return False

    def needs_permissions(self, input_data: Optional[BashToolInput] = None) -> bool:
        return True

    async def validate_input(
        self,
        input_data: BashToolInput,
        context: Optional[ToolUseContext] = None
    ) -> ValidationResult:
        # Basic validation - could add more checks for dangerous commands
        if not input_data.command.strip():
            return ValidationResult(
                result=False,
                message="Command cannot be empty"
            )
        return ValidationResult(result=True)

    def render_result_for_assistant(self, output: BashToolOutput) -> str:
        """Format output for the AI."""
        result_parts = []

        if output.stdout:
            result_parts.append(f"stdout:\n{output.stdout}")

        if output.stderr:
            result_parts.append(f"stderr:\n{output.stderr}")

        timing = ""
        if output.duration_ms:
            secs = output.duration_ms / 1000.0
            timing = f" ({secs:.2f}s"
            if output.timeout_ms:
                timing += f" / timeout {output.timeout_ms/1000:.0f}s"
            timing += ")"
        elif output.timeout_ms:
            timing = f" (timeout {output.timeout_ms/1000:.0f}s)"

        result_parts.append(f"exit code: {output.exit_code}{timing}")

        return "\n\n".join(result_parts)

    def render_tool_use_message(
        self,
        input_data: BashToolInput,
        verbose: bool = False
    ) -> str:
        """Format the tool use for display."""
        return f"$ {input_data.command}"

    async def call(
        self,
        input_data: BashToolInput,
        context: ToolUseContext
    ) -> AsyncGenerator[ToolOutput, None]:
        """Execute the bash command."""

        timeout_ms = input_data.timeout or DEFAULT_TIMEOUT_MS
        timeout_seconds = timeout_ms / 1000.0
        start = asyncio.get_running_loop().time()

        # Background execution: start and return immediately.
        if input_data.run_in_background:
            try:
                from ripperdoc.tools.background_shell import start_background_command
            except Exception as e:  # pragma: no cover - defensive import
                error_output = BashToolOutput(
                    stdout="",
                    stderr=f"Failed to start background task: {str(e)}",
                    exit_code=-1,
                    command=input_data.command
                )
                yield ToolResult(
                    data=error_output,
                    result_for_assistant=self.render_result_for_assistant(error_output)
                )
                return

            task_id = await start_background_command(
                input_data.command,
                timeout=timeout_seconds if timeout_seconds > 0 else None
            )

            output = BashToolOutput(
                stdout="",
                stderr=f"Started background task: {task_id}",
                exit_code=0,
                command=input_data.command,
                duration_ms=(asyncio.get_running_loop().time() - start) * 1000.0,
                timeout_ms=timeout_ms,
                background_task_id=task_id
            )

            yield ToolResult(
                data=output,
                result_for_assistant=self.render_result_for_assistant(output)
            )
            return

        try:
            # Run the command
            process = await asyncio.create_subprocess_shell(
                input_data.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                shell=True,
                start_new_session=False
            )

            stdout_lines: list[str] = []
            stderr_lines: list[str] = []
            queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
            loop = asyncio.get_running_loop()
            deadline = loop.time() + timeout_seconds if timeout_seconds and timeout_seconds > 0 else None
            timed_out = False

            async def _pump_stream(stream: Optional[asyncio.StreamReader], sink: list[str], label: str) -> None:
                if not stream:
                    return
                async for raw in stream:
                    text = raw.decode("utf-8", errors="replace")
                    sink.append(text)
                    await queue.put((label, text.rstrip()))

            pump_tasks = [
                asyncio.create_task(_pump_stream(process.stdout, stdout_lines, "stdout")),
                asyncio.create_task(_pump_stream(process.stderr, stderr_lines, "stderr")),
            ]
            wait_task = asyncio.create_task(process.wait())

            while True:
                done, _ = await asyncio.wait(
                    {wait_task, *pump_tasks},
                    timeout=0.1,
                    return_when=asyncio.FIRST_COMPLETED
                )

                # while not queue.empty():
                #     label, line = queue.get_nowait()
                #     yield ToolProgress(content=f"{label}: {line}")

                now = loop.time()
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

            # Let stream pumps finish draining after the process exits/gets killed.
            try:
                await asyncio.wait_for(asyncio.gather(*pump_tasks), timeout=1.0)
            except asyncio.TimeoutError:
                for task in pump_tasks:
                    if not task.done():
                        task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

            while not queue.empty():
                label, line = queue.get_nowait()
                yield ToolProgress(content=f"{label}: {line}")

            async def _drain_remaining(stream: Optional[asyncio.StreamReader], sink: list[str]) -> None:
                if not stream:
                    return
                try:
                    remaining = await asyncio.wait_for(stream.read(), timeout=0.5)
                except asyncio.TimeoutError:
                    return
                if remaining:
                    sink.append(remaining.decode("utf-8", errors="replace"))

            await _drain_remaining(process.stdout, stdout_lines)
            await _drain_remaining(process.stderr, stderr_lines)

            duration_ms = (asyncio.get_running_loop().time() - start) * 1000.0
            stdout = "".join(stdout_lines)
            stderr = "".join(stderr_lines)
            exit_code = process.returncode or 0

            if timed_out:
                timeout_msg = f"Command timed out after {timeout_seconds} seconds"
                stderr = f"{stderr.rstrip()}\n{timeout_msg}" if stderr else timeout_msg
                exit_code = -1

            output = BashToolOutput(
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                command=input_data.command,
                duration_ms=duration_ms,
                timeout_ms=timeout_ms
            )

            yield ToolResult(
                data=output,
                result_for_assistant=self.render_result_for_assistant(output)
            )

        except Exception as e:
            error_output = BashToolOutput(
                stdout="",
                stderr=f"Error executing command: {str(e)}",
                exit_code=-1,
                command=input_data.command
            )

            yield ToolResult(
                data=error_output,
                result_for_assistant=self.render_result_for_assistant(error_output)
            )

    async def _force_kill_process(self, process: asyncio.subprocess.Process, grace_seconds: float = KILL_GRACE_SECONDS) -> None:
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
