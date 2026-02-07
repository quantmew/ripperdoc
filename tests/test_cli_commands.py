"""CLI slash command surface tests."""

from rich.console import Console

from ripperdoc.cli.commands.add_dir_cmd import command as add_dir_command
from ripperdoc.cli.commands.memory_cmd import command as memory_command
from ripperdoc.cli.commands.tasks_cmd import command as tasks_command
from ripperdoc.cli.commands.todos_cmd import command as todos_command
from ripperdoc.tools import background_shell
from ripperdoc.utils.working_directories import normalize_directory_inputs
from ripperdoc.utils.todo import TodoItem, set_todos
from ripperdoc.utils.tasks import create_task


class _DummyUI:
    """Minimal UI stub for slash command handlers."""

    def __init__(self, console: Console, project_path):
        self.console = console
        self.project_path = project_path


class _AddDirUI(_DummyUI):
    def __init__(self, console: Console, project_path):
        super().__init__(console, project_path)
        self._dirs = set()

    def list_additional_working_directories(self):
        return sorted(self._dirs)

    def add_additional_working_directory(self, raw_path: str):
        normalized, errors = normalize_directory_inputs(
            [raw_path],
            base_dir=self.project_path,
            require_exists=True,
        )
        if errors:
            return False, errors[0]
        resolved = normalized[0]
        if resolved in self._dirs:
            return False, f"Already added: {resolved}"
        self._dirs.add(resolved)
        return True, f"Added: {resolved}"


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


def test_todos_command_reads_task_graph_when_enabled(tmp_path, monkeypatch):
    """Todos command should display task graph items when task system is enabled."""
    monkeypatch.setenv("RIPPERDOC_ENABLE_TASKS", "true")
    monkeypatch.setattr("ripperdoc.utils.todo.Path.home", lambda: tmp_path)
    monkeypatch.setattr("ripperdoc.utils.tasks.Path.home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)

    create_task(subject="write code", task_id="t1", status="in_progress")
    create_task(subject="add tests", task_id="t2", status="pending")

    ui = _DummyUI(Console(record=True, width=120), tmp_path)
    todos_command.handler(ui, "")

    output = ui.console.export_text()
    assert "write code" in output
    assert "add tests" in output


def test_tasks_command_no_tasks(tmp_path, monkeypatch):
    """Tasks command should render an empty state when nothing is running."""
    # Clear all tasks
    background_shell._get_tasks().clear()

    ui = _DummyUI(Console(record=True, width=80), tmp_path)
    tasks_command.handler(ui, "")

    output = ui.console.export_text()
    assert "No tasks currently running" in output


def test_tasks_command_lists_tasks(tmp_path, monkeypatch):
    """Tasks command should list background tasks without needing an event loop."""
    # Clear all tasks first
    background_shell._get_tasks().clear()
    start_time = background_shell._loop_time()

    dummy_process = type("P", (), {"returncode": 0})()
    task = background_shell.BackgroundTask(
        id="bash_abcd",
        command="echo hello && sleep 1",
        process=dummy_process,
        start_time=start_time,
    )
    task.exit_code = 0
    background_shell._get_tasks()[task.id] = task

    ui = _DummyUI(Console(record=True, width=120), tmp_path)
    tasks_command.handler(ui, "")

    output = ui.console.export_text()
    assert "bash_abcd" in output
    assert "echo hello" in output
    assert "completed" in output or "running" in output
    assert "Runtime" in output
    assert "Age" in output

    # Clean up
    background_shell._get_tasks().clear()


def test_memory_command_existing_file_editor_not_opened(tmp_path, monkeypatch):
    """Memory command should not claim file was opened when editor launch fails."""
    (tmp_path / "AGENTS.md").write_text("persisted prefs", encoding="utf-8")

    ui = _DummyUI(Console(record=True, width=120), tmp_path)
    monkeypatch.setattr("ripperdoc.cli.commands.memory_cmd._open_in_editor", lambda *_args, **_kwargs: False)
    memory_command.handler(ui, "project")

    output = ui.console.export_text()
    assert "Opened existing memory file." not in output
    assert "Editor did not launch." in output


def test_memory_command_existing_file_editor_opened(tmp_path, monkeypatch):
    """Memory command should report opened state when editor launch succeeds."""
    (tmp_path / "AGENTS.md").write_text("persisted prefs", encoding="utf-8")

    ui = _DummyUI(Console(record=True, width=120), tmp_path)
    monkeypatch.setattr("ripperdoc.cli.commands.memory_cmd._open_in_editor", lambda *_args, **_kwargs: True)
    memory_command.handler(ui, "project")

    output = ui.console.export_text()
    assert "Opened existing memory file." in output


def test_memory_command_created_file_editor_opened(tmp_path, monkeypatch):
    """Memory command should report created+opened state for new files."""
    ui = _DummyUI(Console(record=True, width=120), tmp_path)
    monkeypatch.setattr("ripperdoc.cli.commands.memory_cmd._open_in_editor", lambda *_args, **_kwargs: True)
    memory_command.handler(ui, "project")

    output = ui.console.export_text()
    assert "Created and opened new memory file." in output


def test_add_dir_command_adds_directory_and_lists_current_dirs(tmp_path):
    extra_dir = tmp_path / "extra"
    extra_dir.mkdir()

    ui = _AddDirUI(Console(record=True, width=120), tmp_path)
    add_dir_command.handler(ui, "extra")
    add_dir_command.handler(ui, "")

    output = ui.console.export_text()
    assert f"Added: {extra_dir.resolve()}" in output
    assert "Session Additional Directories" in output


def test_add_dir_command_reports_missing_directory(tmp_path):
    ui = _AddDirUI(Console(record=True, width=120), tmp_path)
    add_dir_command.handler(ui, "missing")

    output = ui.console.export_text()
    assert "does not exist" in output
