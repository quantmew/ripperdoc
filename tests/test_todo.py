"""Tests for todo storage and tools."""

from pathlib import Path


from ripperdoc.core.tool import ToolUseContext
from ripperdoc.tools.todo_tool import (
    TodoInputItem,
    TodoReadTool,
    TodoReadToolInput,
    TodoWriteTool,
    TodoWriteToolInput,
)
from ripperdoc.utils.todo import TodoItem, format_todo_summary, load_todos, set_todos


def test_todo_storage_roundtrip(tmp_path, monkeypatch):
    """Todos should persist to disk and retain ordering rules."""
    monkeypatch.setattr("ripperdoc.utils.todo.Path.home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)

    todos = [
        TodoItem(id="a", content="second task", status="pending", priority="low"),
        TodoItem(id="b", content="first task", status="in_progress", priority="high"),
    ]

    saved = set_todos(todos)
    loaded = load_todos()

    # Todos should preserve the order they were provided in
    assert saved[0].id == "a"
    assert loaded[0].id == "a"
    assert "total 2" in format_todo_summary(loaded)
    todo_files = list((Path.home() / ".ripperdoc" / "todos").rglob("todos.json"))
    assert todo_files, "todos should be stored in the global ~/.ripperdoc/todos directory"


def test_todo_tools_flow(tmp_path, monkeypatch):
    """Todo tools should write and read tasks, surfacing next action."""
    import asyncio

    monkeypatch.setattr("ripperdoc.utils.todo.Path.home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)
    write_tool = TodoWriteTool()
    read_tool = TodoReadTool()
    context = ToolUseContext()

    write_input = TodoWriteToolInput(
        todos=[
            TodoInputItem(
                id="t1",
                content="implement feature",
                status="in_progress",
                priority="high",
            ),
            TodoInputItem(
                id="t2",
                content="add tests",
                status="pending",
                priority="medium",
            ),
        ]
    )

    async def _run():
        write_result = None
        async for output in write_tool.call(write_input, context):
            write_result = output

        assert write_result is not None
        assert write_result.data.summary.startswith("Todos updated")
        assert write_result.data.next_todo is not None

        read_input = TodoReadToolInput(next_only=True)
        read_result = None
        async for output in read_tool.call(read_input, context):
            read_result = output

        assert read_result is not None
        assert read_result.data.todos
        assert read_result.data.todos[0].id == "t1"
        assert "Next actionable todo" in read_result.data.summary

    asyncio.run(_run())
