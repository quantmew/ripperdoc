"""Lightweight background shell manager for BashTool.

Allows starting shell commands that keep running while the caller continues.
Output can be polled via the BashOutput tool and commands can be terminated
via the KillBash tool.
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class BackgroundTask:
    """In-memory record of a background shell command."""

    id: str
    command: str
    process: asyncio.subprocess.Process
    start_time: float
    timeout: Optional[float] = None
    stdout_chunks: List[str] = field(default_factory=list)
    stderr_chunks: List[str] = field(default_factory=list)
    exit_code: Optional[int] = None
    killed: bool = False
    timed_out: bool = False
    reader_tasks: List[asyncio.Task] = field(default_factory=list)
    done_event: asyncio.Event = field(default_factory=asyncio.Event)


_tasks: Dict[str, BackgroundTask] = {}


async def _pump_stream(stream: asyncio.StreamReader, sink: List[str]) -> None:
    """Continuously read from a stream into a buffer."""
    try:
        while True:
            chunk = await stream.read(4096)
            if not chunk:
                break
            sink.append(chunk.decode("utf-8", errors="replace"))
    except Exception:
        # Best effort; ignore stream read errors to avoid leaking tasks.
        pass


async def _monitor_task(task: BackgroundTask) -> None:
    """Wait for a background process to finish or timeout, then mark status."""
    try:
        if task.timeout:
            await asyncio.wait_for(task.process.wait(), timeout=task.timeout)
        else:
            await task.process.wait()
        task.exit_code = task.process.returncode
    except asyncio.TimeoutError:
        task.timed_out = True
        task.process.kill()
        await task.process.wait()
        task.exit_code = -1
    except Exception:
        task.exit_code = -1
    finally:
        # Ensure readers are finished before marking done.
        for reader in task.reader_tasks:
            reader.cancel()
        await asyncio.gather(*task.reader_tasks, return_exceptions=True)
        task.done_event.set()


async def start_background_command(command: str, timeout: Optional[float] = None) -> str:
    """Launch a background shell command and return its task id."""
    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        shell=True,
    )

    task_id = f"bash_{uuid.uuid4().hex[:8]}"
    record = BackgroundTask(
        id=task_id,
        command=command,
        process=process,
        start_time=asyncio.get_running_loop().time(),
        timeout=timeout,
    )
    _tasks[task_id] = record

    # Start stream pumps and monitor task.
    if process.stdout:
        record.reader_tasks.append(asyncio.create_task(_pump_stream(process.stdout, record.stdout_chunks)))
    if process.stderr:
        record.reader_tasks.append(asyncio.create_task(_pump_stream(process.stderr, record.stderr_chunks)))
    asyncio.create_task(_monitor_task(record))

    return task_id


def _compute_status(task: BackgroundTask) -> str:
    """Return a human-friendly status string."""
    if task.killed:
        return "killed"
    if task.timed_out:
        return "failed"
    if task.exit_code is None:
        return "running"
    return "completed" if task.exit_code == 0 else "failed"


def get_background_status(task_id: str, consume: bool = True) -> dict:
    """Fetch the current status and buffered output of a background command.

    If consume is True, buffered stdout/stderr are cleared after reading.
    """
    if task_id not in _tasks:
        raise KeyError(f"No background task found with id '{task_id}'")

    task = _tasks[task_id]
    stdout = "".join(task.stdout_chunks)
    stderr = "".join(task.stderr_chunks)

    if consume:
        task.stdout_chunks.clear()
        task.stderr_chunks.clear()

    return {
        "id": task.id,
        "command": task.command,
        "status": _compute_status(task),
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": task.exit_code,
        "timed_out": task.timed_out,
        "killed": task.killed,
        "duration_ms": (asyncio.get_running_loop().time() - task.start_time) * 1000.0,
    }


async def kill_background_task(task_id: str) -> bool:
    """Attempt to kill a running background task."""
    task = _tasks.get(task_id)
    if not task:
        return False

    if task.exit_code is not None:
        return False

    try:
        task.killed = True
        task.process.kill()
        await task.process.wait()
        task.exit_code = task.process.returncode or -1
        return True
    finally:
        task.done_event.set()


def list_background_tasks() -> List[str]:
    """Return known background task ids."""
    return list(_tasks.keys())
