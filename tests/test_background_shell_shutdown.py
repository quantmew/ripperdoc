"""Tests for clean shutdown of the background shell loop."""

import asyncio

from ripperdoc.tools import background_shell


def test_shutdown_waits_for_background_process():
    """Ensure shutdown kills and waits for running background processes."""
    # Start from a clean state.
    background_shell.shutdown_background_shell()
    loop = background_shell._ensure_background_loop()

    class DummyProcess:
        def __init__(self) -> None:
            self.killed = False
            self.wait_calls = 0
            self.returncode = None

        def kill(self) -> None:
            self.killed = True

        async def wait(self) -> int:
            self.wait_calls += 1
            self.returncode = 0
            await asyncio.sleep(0)
            return 0

    async def _add_task():
        proc = DummyProcess()
        task = background_shell.BackgroundTask(
            id="bash_test",
            command="echo hello",
            process=proc,
            start_time=background_shell._loop_time(),
        )
        with background_shell._tasks_lock:
            background_shell._tasks[task.id] = task
        return proc, task

    proc, task = asyncio.run_coroutine_threadsafe(_add_task(), loop).result(timeout=2)

    background_shell.shutdown_background_shell()

    assert proc.killed
    assert proc.wait_calls >= 1
    assert task.done_event.is_set()
    assert background_shell._background_loop is None
    assert background_shell._tasks == {}


def test_shutdown_ignores_missing_process(monkeypatch):
    """Ensure shutdown tolerates processes that have already exited."""
    background_shell.shutdown_background_shell()
    loop = background_shell._ensure_background_loop()

    class MissingProcess:
        def __init__(self) -> None:
            self.kill_calls = 0
            self.wait_calls = 0
            self.returncode = None

        def kill(self) -> None:
            self.kill_calls += 1
            raise ProcessLookupError()

        async def wait(self) -> int:
            self.wait_calls += 1
            raise ProcessLookupError()

    async def _add_task():
        proc = MissingProcess()
        task = background_shell.BackgroundTask(
            id="bash_missing",
            command="sleep 1",
            process=proc,
            start_time=background_shell._loop_time(),
        )
        with background_shell._tasks_lock:
            background_shell._tasks[task.id] = task
        return proc, task

    proc, task = asyncio.run_coroutine_threadsafe(_add_task(), loop).result(timeout=2)

    background_shell.shutdown_background_shell()

    assert proc.kill_calls >= 1
    assert proc.wait_calls >= 1
    assert task.done_event.is_set()
    assert background_shell._tasks == {}
