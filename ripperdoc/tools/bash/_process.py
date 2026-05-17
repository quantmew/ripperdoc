"""Process execution helpers for Bash tool."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from typing import Any, AsyncGenerator, List, Optional

from ripperdoc.utils.log import get_logger
from ripperdoc.utils.shell.output_utils import (
    format_duration,
    get_last_n_lines,
    sanitize_output,
)
from ripperdoc.utils.platform import IS_WINDOWS
from ripperdoc.tools.bash._models import BashToolOutput
from ripperdoc.tools.bash._output import build_background_launch_output, build_final_output

logger = get_logger()

KILL_GRACE_SECONDS = 5.0
PROGRESS_INTERVAL_SECONDS = 0.5
STREAM_READ_CHUNK_SIZE = 8192


async def force_kill_process(
    process: asyncio.subprocess.Process, grace_seconds: float = KILL_GRACE_SECONDS
) -> None:
    """Attempt to terminate a process group and avoid hanging waits."""
    if process.returncode is not None:
        return

    def _terminate() -> None:
        if IS_WINDOWS:
            process.terminate()
        elif hasattr(os, "killpg"):
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                process.terminate()
        else:
            process.terminate()

    def _kill() -> None:
        if IS_WINDOWS:
            process.kill()
        elif hasattr(os, "killpg") and hasattr(signal, "SIGKILL"):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                process.kill()
        else:
            process.kill()

    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        _terminate()
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(process.wait(), timeout=grace_seconds)
        return

    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        _kill()
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(process.wait(), timeout=grace_seconds)


async def drain_stream(stream: Optional[asyncio.StreamReader], sink: List[str]) -> None:
    """Drain any remaining data from a stream."""
    if not stream:
        return
    try:
        remaining = await asyncio.wait_for(stream.read(), timeout=0.5)
    except asyncio.TimeoutError:
        return
    if remaining:
        sink.append(remaining.decode("utf-8", errors="replace"))


async def execute_foreground_process(
    process: asyncio.subprocess.Process,
    start_time: float,
    timeout_seconds: float,
) -> AsyncGenerator[tuple[bool, list[str], list[str], bool], Any]:
    """Execute process and yield progress updates."""
    stdout_lines: List[str] = []
    stderr_lines: List[str] = []
    queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    deadline = (
        loop.time() + timeout_seconds if timeout_seconds and timeout_seconds > 0 else None
    )
    timed_out = False
    last_progress_time = loop.time()

    async def _pump_stream(
        stream: Optional[asyncio.StreamReader], sink: List[str], label: str
    ) -> None:
        if not stream:
            return
        while True:
            raw = await stream.read(STREAM_READ_CHUNK_SIZE)
            if not raw:
                break
            text = raw.decode("utf-8", errors="replace")
            sanitized_text = sanitize_output(text)
            sink.append(sanitized_text)
            await queue.put((label, sanitized_text.rstrip()))

    pump_tasks = [
        asyncio.create_task(_pump_stream(process.stdout, stdout_lines, "stdout")),
        asyncio.create_task(_pump_stream(process.stderr, stderr_lines, "stderr")),
    ]
    wait_task = asyncio.create_task(process.wait())

    while True:
        done, _ = await asyncio.wait(
            {wait_task, *pump_tasks}, timeout=0.1, return_when=asyncio.FIRST_COMPLETED
        )

        now = loop.time()

        while not queue.empty():
            label, text = queue.get_nowait()
            from ripperdoc.core.tool import ToolProgress
            yield ToolProgress(content=f"{label}: {text}")  # type: ignore[misc]

        if now - last_progress_time >= PROGRESS_INTERVAL_SECONDS:
            combined_output = "".join(stdout_lines + stderr_lines)
            if combined_output:
                preview = get_last_n_lines(combined_output, 5)
                elapsed = format_duration((now - start_time) * 1000)
                from ripperdoc.core.tool import ToolProgress
                yield ToolProgress(content=f"Running... ({elapsed})\n{preview}")  # type: ignore[misc]
            last_progress_time = now

        if deadline is not None and now >= deadline:
            timed_out = True
            await force_kill_process(process)
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

    try:
        await asyncio.wait_for(asyncio.gather(*pump_tasks), timeout=1.0)
    except asyncio.TimeoutError:
        for task in pump_tasks:
            if not task.done():
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    await drain_stream(process.stdout, stdout_lines)
    await drain_stream(process.stderr, stderr_lines)

    yield stdout_lines, stderr_lines, timed_out


def _create_error_output(command: str, stderr: str, sandbox: bool) -> BashToolOutput:
    """Create a standardized error output."""
    return BashToolOutput(
        stdout="",
        stderr=stderr,
        exit_code=-1,
        command=command,
        sandbox=sandbox,
        is_error=True,
    )
