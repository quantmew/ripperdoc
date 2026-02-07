"""Tests for the persistent task graph system and tools."""

from __future__ import annotations

import asyncio

from ripperdoc.core.tool import ToolUseContext
from ripperdoc.tools.task_graph_tool import (
    TaskCreateInput,
    TaskCreateTool,
    TaskListInput,
    TaskListTool,
    TaskUpdateInput,
    TaskUpdateTool,
)
from ripperdoc.utils.tasks import (
    TaskPatch,
    create_task,
    format_task_summary,
    get_next_actionable_task,
    get_task,
    is_task_system_enabled,
    list_tasks,
    update_task,
)


def test_task_graph_roundtrip_and_dependency_sync(tmp_path, monkeypatch):
    monkeypatch.setattr("ripperdoc.utils.tasks.Path.home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)

    parent = create_task(subject="run build", description="run build pipeline", task_id="1")
    child = create_task(subject="fix errors", description="fix type errors", task_id="2", blocked_by=["1"])

    assert parent.id == "1"
    assert child.blocked_by == ["1"]

    loaded_parent = get_task("1")
    assert loaded_parent is not None
    assert "2" in loaded_parent.blocks

    update_task("1", TaskPatch(status="completed"))
    tasks = list_tasks()
    next_task = get_next_actionable_task(tasks)

    assert next_task is not None
    assert next_task.id == "2"
    assert "total 2" in format_task_summary(tasks)


def test_task_graph_tools_flow_and_deleted_status(tmp_path, monkeypatch):
    monkeypatch.setattr("ripperdoc.utils.tasks.Path.home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)

    create_tool = TaskCreateTool()
    update_tool = TaskUpdateTool()
    list_tool = TaskListTool()
    context = ToolUseContext(agent_id="team-lead")

    async def _run() -> None:
        created = None
        async for output in create_tool.call(
            TaskCreateInput(
                subject="implement feature",
                description="deliver feature implementation",
                activeForm="implementing feature",
            ),
            context,
        ):
            created = output

        assert created is not None
        assert created.data.task is not None
        task_id = created.data.task.id

        updated = None
        async for output in update_tool.call(
            TaskUpdateInput(task_id=task_id, status="completed"),
            context,
        ):
            updated = output

        assert updated is not None
        assert updated.data.success is True
        assert updated.data.status_change is not None
        assert updated.data.status_change.to_status == "completed"

        listed = None
        async for output in list_tool.call(TaskListInput(), context):
            listed = output

        assert listed is not None
        assert listed.data.tasks
        assert listed.data.tasks[0].id == task_id

        deleted = None
        async for output in update_tool.call(
            TaskUpdateInput(task_id=task_id, status="deleted"),
            context,
        ):
            deleted = output

        assert deleted is not None
        assert deleted.data.success is True

    asyncio.run(_run())


def test_task_update_blocks_completed_when_dependencies_unresolved(tmp_path, monkeypatch):
    monkeypatch.setattr("ripperdoc.utils.tasks.Path.home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)

    create_task(subject="build", description="build", task_id="1", status="in_progress")
    create_task(subject="fix", description="fix", task_id="2", blocked_by=["1"])

    update_tool = TaskUpdateTool()
    context = ToolUseContext()

    async def _run() -> None:
        failed = None
        async for output in update_tool.call(TaskUpdateInput(task_id="2", status="completed"), context):
            failed = output

        assert failed is not None
        assert failed.data.success is False
        assert "unresolved blockers" in (failed.data.error or "")

    asyncio.run(_run())


def test_task_list_filters_internal_and_prunes_completed_blockers(tmp_path, monkeypatch):
    monkeypatch.setattr("ripperdoc.utils.tasks.Path.home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)

    create_task(subject="internal", description="meta", task_id="1", metadata={"_internal": True})
    create_task(subject="dep", description="dep", task_id="2", status="completed")
    create_task(subject="work", description="work", task_id="3", blocked_by=["2"])

    list_tool = TaskListTool()
    context = ToolUseContext()

    async def _run() -> None:
        listed = None
        async for output in list_tool.call(TaskListInput(), context):
            listed = output

        assert listed is not None
        ids = [task.id for task in listed.data.tasks]
        assert "1" not in ids
        work = next(task for task in listed.data.tasks if task.id == "3")
        assert work.blocked_by == []

    asyncio.run(_run())


def test_task_feature_flag_defaults_true(monkeypatch):
    monkeypatch.delenv("RIPPERDOC_ENABLE_TASKS", raising=False)
    assert is_task_system_enabled() is True


def test_task_feature_flag_false(monkeypatch):
    monkeypatch.setenv("RIPPERDOC_ENABLE_TASKS", "false")
    assert is_task_system_enabled() is False


def test_task_graph_tool_descriptions_and_examples():
    async def _run() -> None:
        create_tool = TaskCreateTool()
        update_tool = TaskUpdateTool()
        list_tool = TaskListTool()

        create_desc = await create_tool.description()
        update_desc = await update_tool.description()
        list_desc = await list_tool.description()

        assert "task" in create_desc.lower()
        assert "status" in update_desc.lower() or "owner" in update_desc.lower()
        assert "task" in list_desc.lower()

        assert len(create_tool.input_examples()) > 0
        assert len(update_tool.input_examples()) > 0
        assert len(list_tool.input_examples()) > 0

    asyncio.run(_run())
