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
