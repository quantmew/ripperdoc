"""CLI slash command surface tests."""

from rich.console import Console

from ripperdoc.cli.commands.tasks_cmd import command as tasks_command
from ripperdoc.cli.commands.todos_cmd import command as todos_command
from ripperdoc.tools import background_shell
from ripperdoc.utils.todo import TodoItem, set_todos


class _DummyUI:
    """Minimal UI stub for slash command handlers."""

    def __init__(self, console: Console, project_path):
        self.console = console
        self.project_path = project_path


def test_todos_command_empty(tmp_path, monkeypatch):
    """Todos command should show an empty state when nothing is stored."""
    monkeypatch.setattr("ripperdoc.utils.todo.Path.home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)

    ui = _DummyUI(Console(record=True, width=80), tmp_path)
    todos_command.handler(ui, "")

    output = ui.console.export_text()
    assert "No todos currently tracked" in output


def test_todos_command_lists_items(tmp_path, monkeypatch):
    """Todos command should render stored items."""
    monkeypatch.setattr("ripperdoc.utils.todo.Path.home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)

    set_todos(
        [
            TodoItem(id="t1", content="write code", status="in_progress", priority="high"),
            TodoItem(id="t2", content="add tests", status="pending", priority="medium"),
        ],
        project_root=tmp_path,
    )

    ui = _DummyUI(Console(record=True, width=120), tmp_path)
    todos_command.handler(ui, "")

    output = ui.console.export_text()
    assert "write code" in output
    assert "add tests" in output
    assert "Todos updated" in output


def test_tasks_command_no_tasks(tmp_path, monkeypatch):
    """Tasks command should render an empty state when nothing is running."""
    monkeypatch.setattr(background_shell, "_tasks", {})

    ui = _DummyUI(Console(record=True, width=80), tmp_path)
    tasks_command.handler(ui, "")

    output = ui.console.export_text()
    assert "No tasks currently running" in output


def test_tasks_command_lists_tasks(tmp_path, monkeypatch):
    """Tasks command should list background tasks without needing an event loop."""
    monkeypatch.setattr(background_shell, "_tasks", {})
    start_time = background_shell._loop_time()

    dummy_process = type("P", (), {"returncode": 0})()
    task = background_shell.BackgroundTask(
        id="bash_abcd",
        command="echo hello && sleep 1",
        process=dummy_process,
        start_time=start_time,
    )
    task.exit_code = 0
    background_shell._tasks[task.id] = task

    ui = _DummyUI(Console(record=True, width=120), tmp_path)
    tasks_command.handler(ui, "")

    output = ui.console.export_text()
    assert "bash_abcd" in output
    assert "echo hello" in output
    assert "completed" in output or "running" in output
