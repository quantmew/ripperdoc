"""CLI slash command surface tests."""

from uuid import UUID

from rich.console import Console

from ripperdoc.cli.commands.add_dir_cmd import command as add_dir_command
from ripperdoc.cli.commands.clear_cmd import command as clear_command
from ripperdoc.cli.commands import get_slash_command
from ripperdoc.cli.commands.memory_cmd import command as memory_command
from ripperdoc.cli.commands.mcp_cmd import command as mcp_command
from ripperdoc.cli.commands.plugins_cmd import command as plugins_command
from ripperdoc.cli.commands.tasks_cmd import command as tasks_command
from ripperdoc.cli.commands.todos_cmd import command as todos_command
from ripperdoc.core.plugins import clear_runtime_plugin_dirs
from ripperdoc.tools import background_shell
from ripperdoc.utils.mcp import McpServerInfo
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


class _ClearUI(_DummyUI):
    def __init__(self, console: Console, project_path):
        super().__init__(console, project_path)
        self.session_id = "old-session-id"
        self.conversation_messages = ["u1", "a1"]
        self._saved_conversation = ["old"]
        self.end_calls = []
        self.start_calls = []
        self.set_session_calls = []
        self.rebuild_calls = 0

    def _run_session_end(self, reason: str):
        self.end_calls.append((reason, self.session_id))

    def _set_session(self, session_id: str):
        self.set_session_calls.append(session_id)
        self.session_id = session_id

    def _run_session_start(self, source: str):
        self.start_calls.append((source, self.session_id))

    def _rebuild_session_usage_from_messages(self, messages):
        self.rebuild_calls += 1


class _ClearFallbackUI(_DummyUI):
    def __init__(self, console: Console, project_path):
        super().__init__(console, project_path)
        self.conversation_messages = ["u1", "a1"]
        self.end_calls = []
        self.start_calls = []

    def _run_session_end(self, reason: str):
        self.end_calls.append(reason)

    def _run_session_start(self, source: str):
        self.start_calls.append(source)


def test_clear_command_switches_session_and_resets_messages(tmp_path, monkeypatch):
    fixed_uuid = UUID("11111111-2222-3333-4444-555555555555")
    monkeypatch.setattr("ripperdoc.cli.commands.clear_cmd.uuid4", lambda: fixed_uuid)

    ui = _ClearUI(Console(record=True, width=120), tmp_path)
    clear_command.handler(ui, "")

    output = ui.console.export_text()
    assert ui.end_calls == [("clear", "old-session-id")]
    assert ui.set_session_calls == [str(fixed_uuid)]
    assert ui.start_calls == [("clear", str(fixed_uuid))]
    assert ui.session_id == str(fixed_uuid)
    assert ui.conversation_messages == []
    assert ui._saved_conversation is None
    assert ui.rebuild_calls == 1
    assert "Conversation cleared" in output


def test_clear_command_without_set_session_still_clears(tmp_path):
    ui = _ClearFallbackUI(Console(record=True, width=120), tmp_path)
    clear_command.handler(ui, "")

    output = ui.console.export_text()
    assert ui.conversation_messages == []
    assert ui.end_calls == ["clear"]
    assert ui.start_calls == ["clear"]
    assert "Conversation cleared" in output


def test_new_alias_points_to_clear_command():
    assert get_slash_command("new") == clear_command


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


def test_todos_command_hides_completed_by_default(tmp_path, monkeypatch):
    """Todos command should hide completed rows by default while keeping summary counts."""
    monkeypatch.delenv("RIPPERDOC_UI_SHOW_COMPLETED_TASKS", raising=False)
    monkeypatch.setattr("ripperdoc.utils.todo.Path.home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)

    set_todos(
        [
            TodoItem(id="t1", content="active work", status="in_progress", priority="high"),
            TodoItem(id="t2", content="done work", status="completed", priority="medium"),
        ],
        project_root=tmp_path,
    )

    ui = _DummyUI(Console(record=True, width=140), tmp_path)
    todos_command.handler(ui, "")

    output = ui.console.export_text()
    assert "active work" in output
    assert "done work" not in output
    assert "1 completed" in output


def test_todos_command_all_shows_completed(tmp_path, monkeypatch):
    """`/todos all` should include completed rows."""
    monkeypatch.delenv("RIPPERDOC_UI_SHOW_COMPLETED_TASKS", raising=False)
    monkeypatch.setattr("ripperdoc.utils.todo.Path.home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)

    set_todos(
        [
            TodoItem(id="t1", content="active work", status="in_progress", priority="high"),
            TodoItem(id="t2", content="done work", status="completed", priority="medium"),
        ],
        project_root=tmp_path,
    )

    ui = _DummyUI(Console(record=True, width=140), tmp_path)
    todos_command.handler(ui, "all")

    output = ui.console.export_text()
    assert "active work" in output
    assert "done work" in output


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
    monkeypatch.setattr(
        "ripperdoc.cli.commands.memory_cmd._open_in_editor", lambda *_args, **_kwargs: False
    )
    memory_command.handler(ui, "project")

    output = ui.console.export_text()
    assert "Opened existing memory file." not in output
    assert "Editor did not launch." in output


def test_memory_command_existing_file_editor_opened(tmp_path, monkeypatch):
    """Memory command should report opened state when editor launch succeeds."""
    (tmp_path / "AGENTS.md").write_text("persisted prefs", encoding="utf-8")

    ui = _DummyUI(Console(record=True, width=120), tmp_path)
    monkeypatch.setattr(
        "ripperdoc.cli.commands.memory_cmd._open_in_editor", lambda *_args, **_kwargs: True
    )
    memory_command.handler(ui, "project")

    output = ui.console.export_text()
    assert "Opened existing memory file." in output


def test_memory_command_created_file_editor_opened(tmp_path, monkeypatch):
    """Memory command should report created+opened state for new files."""
    ui = _DummyUI(Console(record=True, width=120), tmp_path)
    monkeypatch.setattr(
        "ripperdoc.cli.commands.memory_cmd._open_in_editor", lambda *_args, **_kwargs: True
    )
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


def test_plugins_command_add_list_remove(tmp_path):
    clear_runtime_plugin_dirs()
    plugin_dir = tmp_path / "plugins" / "demo-plugin"
    manifest = plugin_dir / ".ripperdoc-plugin" / "plugin.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text('{"name":"demo-plugin","description":"demo"}', encoding="utf-8")

    ui = _DummyUI(Console(record=True, width=120), tmp_path)
    plugins_command.handler(ui, f"add {plugin_dir} project")
    plugins_command.handler(ui, "list")
    plugins_command.handler(ui, "remove demo-plugin project")

    output = ui.console.export_text()
    assert "Enabled plugin" in output
    assert "demo-plugin" in output
    assert "Removed plugin" in output


def test_plugins_command_defaults_to_tui_and_supports_plugin_alias(tmp_path, monkeypatch):
    called = {"count": 0}

    def _fake_run_plugins_tui(project_path):
        called["count"] += 1
        assert project_path == tmp_path
        return True

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr(
        "ripperdoc.cli.ui.plugins_tui.run_plugins_tui",
        _fake_run_plugins_tui,
    )

    ui = _DummyUI(Console(record=True, width=120), tmp_path)
    plugins_command.handler(ui, "")

    assert called["count"] == 1
    assert get_slash_command("plugin") == plugins_command


def test_mcp_logs_lists_targets(tmp_path, monkeypatch):
    """MCP logs command should list configured log targets."""

    async def _fake_load(_project_path):
        return [
            McpServerInfo(name="context7", status="connected"),
            McpServerInfo(name="agent-browser", status="connected"),
        ]

    log_a = tmp_path / "context7.log"
    log_a.write_text("one\n", encoding="utf-8")
    log_b = tmp_path / "agent-browser.log"

    def _fake_log_path(_project_path, server_name):
        return log_a if server_name == "context7" else log_b

    monkeypatch.setattr("ripperdoc.cli.commands.mcp_cmd.load_mcp_servers_async", _fake_load)
    monkeypatch.setattr("ripperdoc.cli.commands.mcp_cmd.get_mcp_stderr_mode", lambda: "log")
    monkeypatch.setattr("ripperdoc.cli.commands.mcp_cmd.get_mcp_stderr_log_path", _fake_log_path)

    ui = _DummyUI(Console(record=True, width=160), tmp_path)
    mcp_command.handler(ui, "logs")

    output = ui.console.export_text()
    assert "MCP stderr logs" in output
    assert "context7" in output
    assert "agent-browser" in output
    assert "ready" in output
    assert "missing" in output


def test_mcp_default_subcommand_uses_textual_ui(tmp_path, monkeypatch):
    called = {"count": 0}

    def _fake_run_mcp_tui(project_path):
        called["count"] += 1
        assert project_path == tmp_path
        return True

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("ripperdoc.cli.ui.mcp_tui.run_mcp_tui", _fake_run_mcp_tui)

    ui = _DummyUI(Console(record=True, width=120), tmp_path)
    mcp_command.handler(ui, "")

    assert called["count"] == 1


def test_mcp_tui_subcommand_uses_textual_ui(tmp_path, monkeypatch):
    called = {"count": 0}

    def _fake_run_mcp_tui(project_path):
        called["count"] += 1
        assert project_path == tmp_path
        return True

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("ripperdoc.cli.ui.mcp_tui.run_mcp_tui", _fake_run_mcp_tui)

    ui = _DummyUI(Console(record=True, width=120), tmp_path)
    mcp_command.handler(ui, "tui")

    assert called["count"] == 1


def test_mcp_tui_falls_back_to_plain_overview_without_tty(tmp_path, monkeypatch):
    async def _fake_load(_project_path):
        return [McpServerInfo(name="context7", status="connected")]

    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("ripperdoc.cli.commands.mcp_cmd.load_mcp_servers_async", _fake_load)

    ui = _DummyUI(Console(record=True, width=120), tmp_path)
    mcp_command.handler(ui, "tui")

    output = ui.console.export_text()
    assert "Interactive UI requires a TTY" in output
    assert "MCP servers" in output
    assert "context7" in output


def test_mcp_list_subcommand_shows_plain_overview(tmp_path, monkeypatch):
    async def _fake_load(_project_path):
        return [McpServerInfo(name="context7", status="connected", command="npx")]

    monkeypatch.setattr("ripperdoc.cli.commands.mcp_cmd.load_mcp_servers_async", _fake_load)

    ui = _DummyUI(Console(record=True, width=120), tmp_path)
    mcp_command.handler(ui, "list")

    output = ui.console.export_text()
    assert "MCP servers" in output
    assert "context7" in output
    assert "Command: npx" in output


def test_mcp_logs_show_last_lines_for_server(tmp_path, monkeypatch):
    """MCP logs command should show tail output for a specific server."""

    async def _fake_load(_project_path):
        return [McpServerInfo(name="context7", status="connected")]

    log_path = tmp_path / "context7.log"
    log_path.write_text("line-1\nline-2\nline-3\n", encoding="utf-8")

    monkeypatch.setattr("ripperdoc.cli.commands.mcp_cmd.load_mcp_servers_async", _fake_load)
    monkeypatch.setattr("ripperdoc.cli.commands.mcp_cmd.get_mcp_stderr_log_path", lambda *_: log_path)

    ui = _DummyUI(Console(record=True, width=120), tmp_path)
    mcp_command.handler(ui, "logs context7 --lines 2")

    output = ui.console.export_text()
    assert "MCP stderr" in output
    assert "line-2" in output
    assert "line-3" in output
    assert "line-1" not in output


def test_mcp_logs_reports_unknown_server(tmp_path, monkeypatch):
    """MCP logs command should report unknown server names."""

    async def _fake_load(_project_path):
        return [McpServerInfo(name="context7", status="connected")]

    monkeypatch.setattr("ripperdoc.cli.commands.mcp_cmd.load_mcp_servers_async", _fake_load)
    monkeypatch.setattr("ripperdoc.cli.commands.mcp_cmd.get_mcp_stderr_mode", lambda: "log")
    monkeypatch.setattr(
        "ripperdoc.cli.commands.mcp_cmd.get_mcp_stderr_log_path",
        lambda project_path, server_name: project_path / f"{server_name}.log",
    )

    ui = _DummyUI(Console(record=True, width=120), tmp_path)
    mcp_command.handler(ui, "logs unknown-server")

    output = ui.console.export_text()
    assert "Unknown MCP server" in output
    assert "context7" in output
