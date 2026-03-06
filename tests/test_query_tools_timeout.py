"""Tests for query tool timeout strategy."""

import asyncio
import os
from pathlib import Path

import pytest

from ripperdoc.core.query import tools as tools_module


def test_resolve_tool_timeout_disables_for_ask_user_question() -> None:
    assert tools_module._resolve_tool_timeout_sec("AskUserQuestion") is None
    assert (
        tools_module._resolve_tool_timeout_sec("Bash")
        == tools_module.DEFAULT_TOOL_TIMEOUT_SEC
    )


def test_resolve_concurrent_timeout_disables_when_batch_contains_ask_user_question() -> None:
    assert (
        tools_module._resolve_concurrent_timeout_sec(["Bash", "AskUserQuestion"]) is None
    )
    assert (
        tools_module._resolve_concurrent_timeout_sec(["Bash", "Read"])
        == tools_module.DEFAULT_CONCURRENT_TOOL_TIMEOUT_SEC
    )


@pytest.mark.asyncio
async def test_tool_execution_cwd_nested_with_parent_none_does_not_deadlock(tmp_path: Path) -> None:
    child_dir = tmp_path / "child"
    child_dir.mkdir(parents=True)

    async def _run_nested() -> str:
        async with tools_module._tool_execution_cwd(
            tool_name="Task",
            tool_use_id="outer",
            working_directory=None,
        ):
            async with tools_module._tool_execution_cwd(
                tool_name="Glob",
                tool_use_id="inner",
                working_directory=str(child_dir),
            ):
                return os.getcwd()

    nested_cwd = await asyncio.wait_for(_run_nested(), timeout=1.0)
    assert Path(nested_cwd).resolve() == child_dir.resolve()


@pytest.mark.asyncio
async def test_tool_execution_cwd_reentrant_restores_parent_and_original(tmp_path: Path) -> None:
    parent_dir = tmp_path / "parent"
    child_dir = tmp_path / "child"
    parent_dir.mkdir(parents=True)
    child_dir.mkdir(parents=True)
    original_cwd = Path.cwd().resolve()

    async def _run_nested() -> tuple[Path, Path, Path]:
        async with tools_module._tool_execution_cwd(
            tool_name="Task",
            tool_use_id="outer",
            working_directory=str(parent_dir),
        ):
            cwd_in_parent = Path.cwd().resolve()
            async with tools_module._tool_execution_cwd(
                tool_name="Glob",
                tool_use_id="inner",
                working_directory=str(child_dir),
            ):
                cwd_in_child = Path.cwd().resolve()
            cwd_after_child = Path.cwd().resolve()
            return cwd_in_parent, cwd_in_child, cwd_after_child

    cwd_in_parent, cwd_in_child, cwd_after_child = await asyncio.wait_for(_run_nested(), timeout=1.0)
    assert cwd_in_parent == parent_dir.resolve()
    assert cwd_in_child == child_dir.resolve()
    assert cwd_after_child == parent_dir.resolve()
    assert Path.cwd().resolve() == original_cwd


@pytest.mark.asyncio
async def test_run_concurrent_tool_uses_honors_abort_signal() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def _never_finishes():
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        if False:  # pragma: no cover - keeps this as an async generator
            yield None

    async def _collect() -> list[object]:
        collected: list[object] = []
        abort_signal = asyncio.Event()
        runner = tools_module._run_concurrent_tool_uses(  # type: ignore[attr-defined]
            [_never_finishes()],
            ["Task"],
            [],
            abort_signal=abort_signal,
        )

        task = asyncio.create_task(_drain_runner(runner, collected))
        await asyncio.wait_for(started.wait(), timeout=1.0)
        abort_signal.set()
        await asyncio.wait_for(task, timeout=1.0)
        return collected

    async def _drain_runner(runner, collected: list[object]) -> None:
        async for item in runner:
            collected.append(item)

    collected = await _collect()
    assert collected == []
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_execute_tools_sequentially_honors_abort_signal() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def _never_finishes():
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        if False:  # pragma: no cover - keeps this as an async generator
            yield None

    collected: list[object] = []
    tool_results = []
    abort_signal = asyncio.Event()

    async def _drain_runner() -> None:
        async for item in tools_module._execute_tools_sequentially(  # type: ignore[attr-defined]
            [{"generator": _never_finishes()}],
            tool_results,
            abort_signal=abort_signal,
        ):
            collected.append(item)

    task = asyncio.create_task(_drain_runner())
    await asyncio.wait_for(started.wait(), timeout=1.0)
    abort_signal.set()
    await asyncio.wait_for(task, timeout=1.0)

    assert collected == []
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_execute_tools_sequentially_keeps_tool_generator_on_one_task(tmp_path: Path) -> None:
    original_cwd = Path.cwd().resolve()
    collected: list[object] = []
    tool_results = []

    async def _two_messages():
        async with tools_module._tool_execution_cwd(
            tool_name="Bash",
            tool_use_id="call-1",
            working_directory=str(tmp_path),
        ):
            yield tools_module.create_user_message("first")
            yield tools_module.create_user_message("second")

    async for item in tools_module._execute_tools_sequentially(  # type: ignore[attr-defined]
        [{"generator": _two_messages()}],
        tool_results,
        abort_signal=asyncio.Event(),
    ):
        collected.append(item)

    assert len(collected) == 2
    assert len(tool_results) == 2
    assert Path.cwd().resolve() == original_cwd
