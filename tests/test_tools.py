"""Test tools."""

import asyncio
import json
import os
import subprocess
import pytest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from pydantic import BaseModel

from ripperdoc.tools.file_read_tool import FileReadTool, FileReadToolInput
from ripperdoc.tools.file_write_tool import FileWriteTool, FileWriteToolInput
from ripperdoc.tools.file_edit_tool import FileEditTool, FileEditToolInput
from ripperdoc.tools.multi_edit_tool import (
    MultiEditTool,
    MultiEditToolInput,
    EditOperation,
)
from ripperdoc.tools.glob_tool import GlobTool, GlobToolInput
from ripperdoc.tools.ls_tool import LSTool, LSToolInput
from ripperdoc.tools.bash_tool import BashTool, BashToolInput
from ripperdoc.tools.task_output_tool import TaskOutputTool, TaskOutputInput
from ripperdoc.tools.task_stop_tool import TaskStopTool, TaskStopInput
from ripperdoc.tools.task_tool import AgentRunRecord, TaskTool, TaskToolInput
from ripperdoc.tools.enter_worktree_tool import EnterWorktreeTool, EnterWorktreeToolInput
from ripperdoc.utils.collaboration.worktree import (
    WorktreeSession,
    cleanup_worktree_session,
    cleanup_registered_worktrees,
    consume_session_worktrees,
    create_task_worktree,
    has_worktree_changes,
    sync_worktree_configuration,
)
from ripperdoc.core.tool import Tool, ToolUseContext, ToolProgress, ToolResult
from ripperdoc.core.query import ToolRegistry
from ripperdoc.tools.tool_search_tool import ToolSearchTool, ToolSearchInput
from ripperdoc.tools.background_shell import get_background_status


@pytest.mark.asyncio
async def test_file_read_tool():
    """Test reading a file."""
    tool = FileReadTool()

    # Create a test file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("Line 1\nLine 2\nLine 3\n")
        temp_path = f.name

    try:
        input_data = FileReadToolInput(file_path=temp_path)
        context = ToolUseContext()

        result = None
        async for output in tool.call(input_data, context):
            result = output

        assert result is not None
        assert result.data.content == "Line 1\nLine 2\nLine 3\n"
        assert result.data.line_count == 3

    finally:
        Path(temp_path).unlink()


@pytest.mark.asyncio
async def test_file_write_tool():
    """Test writing a file."""
    tool = FileWriteTool()

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = str(Path(tmpdir) / "test.txt")
        input_data = FileWriteToolInput(file_path=file_path, content="Test content\n")
        context = ToolUseContext()

        result = None
        async for output in tool.call(input_data, context):
            result = output

        assert result is not None
        assert result.data.success

        # Verify file was created
        assert Path(file_path).exists()
        assert Path(file_path).read_text() == "Test content\n"


@pytest.mark.asyncio
async def test_file_write_tool_disables_gitignore_warnings(monkeypatch, tmp_path):
    """Write tool should suppress warnings for gitignored-only paths."""
    tool = FileWriteTool()
    file_path = tmp_path / "ignored.cj"
    input_data = FileWriteToolInput(file_path=str(file_path), content="content\n")
    context = ToolUseContext()
    captured: dict[str, bool] = {}

    def fake_check_path_for_tool(path, tool_name="unknown", warn_only=True, warn_on_gitignore=True):
        captured["warn_on_gitignore"] = warn_on_gitignore
        return True, None

    monkeypatch.setattr("ripperdoc.tools.file_write_tool.check_path_for_tool", fake_check_path_for_tool)

    result = await tool.validate_input(input_data, context)

    assert result.result is True
    assert captured["warn_on_gitignore"] is False


@pytest.mark.asyncio
async def test_file_edit_tool():
    """Test editing a file."""
    tool = FileEditTool()

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("Hello World\n")
        temp_path = f.name

    try:
        input_data = FileEditToolInput(file_path=temp_path, old_string="World", new_string="Python")
        context = ToolUseContext()

        result = None
        async for output in tool.call(input_data, context):
            result = output

        assert result is not None
        assert result.data.success
        assert result.data.replacements_made == 1

        # Verify edit
        assert Path(temp_path).read_text() == "Hello Python\n"

    finally:
        Path(temp_path).unlink()


@pytest.mark.asyncio
async def test_multi_edit_tool():
    """Test applying multiple edits sequentially."""
    tool = MultiEditTool()

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("hello world\nhello again\n")
        temp_path = f.name

    try:
        input_data = MultiEditToolInput(
            file_path=temp_path,
            edits=[
                EditOperation(old_string="hello", new_string="hi", replace_all=True),
                EditOperation(old_string="hi again", new_string="hi there"),
            ],
        )
        context = ToolUseContext()

        result = None
        async for output in tool.call(input_data, context):
            result = output

        assert result is not None
        assert result.data.success is True
        assert result.data.replacements_made == 3
        assert "hi there" in Path(temp_path).read_text()
    finally:
        Path(temp_path).unlink()


@pytest.mark.asyncio
async def test_multi_edit_tool_create_file():
    """Test creating a file via multi edit with empty old_string."""
    tool = MultiEditTool()

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = str(Path(tmpdir) / "new.txt")
        input_data = MultiEditToolInput(
            file_path=file_path,
            edits=[
                EditOperation(old_string="", new_string="line1\n"),
                EditOperation(old_string="line1", new_string="line1-mod"),
            ],
        )
        context = ToolUseContext()

        result = None
        async for output in tool.call(input_data, context):
            result = output

        assert result is not None
        assert result.data.success is True
        assert Path(file_path).read_text() == "line1-mod\n"


@pytest.mark.asyncio
async def test_multi_edit_tool_failure_no_match():
    """Multi edit should not write when a string is missing."""
    tool = MultiEditTool()

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("content\n")
        temp_path = f.name

    try:
        input_data = MultiEditToolInput(
            file_path=temp_path,
            edits=[EditOperation(old_string="missing", new_string="new")],
        )
        context = ToolUseContext()

        result = None
        async for output in tool.call(input_data, context):
            result = output

        assert result is not None
        assert result.data.success is False
        assert Path(temp_path).read_text() == "content\n"
    finally:
        Path(temp_path).unlink()


@pytest.mark.asyncio
async def test_glob_tool():
    """Test glob pattern matching."""
    tool = GlobTool()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        (Path(tmpdir) / "test1.py").touch()
        (Path(tmpdir) / "test2.py").touch()
        (Path(tmpdir) / "test.txt").touch()

        input_data = GlobToolInput(pattern="*.py", path=tmpdir)
        context = ToolUseContext()

        result = None
        async for output in tool.call(input_data, context):
            result = output

        assert result is not None
        assert result.data.count == 2
        assert all("test" in m for m in result.data.matches)


@pytest.mark.asyncio
async def test_tool_validation():
    """Test tool input validation."""
    tool = FileReadTool()

    # Test with non-existent file
    input_data = FileReadToolInput(file_path="/nonexistent/file.txt")
    context = ToolUseContext()

    validation = await tool.validate_input(input_data, context)
    assert not validation.result
    assert "not found" in validation.message.lower()


def test_bash_tool_concurrency_safety_uses_current_input() -> None:
    tool = BashTool()

    assert tool.is_concurrency_safe_for_input(BashToolInput(command="git status")) is True
    assert tool.is_concurrency_safe_for_input(BashToolInput(command="git diff HEAD")) is True
    assert tool.is_concurrency_safe_for_input(BashToolInput(command="git commit -m test")) is False


@pytest.mark.asyncio
async def test_ls_tool():
    """Test directory listing."""
    tool = LSTool()

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "dir1").mkdir()
        (root / "dir1" / "nested.txt").write_text("data")
        (root / "file.txt").write_text("data")

        input_data = LSToolInput(path=tmpdir)
        context = ToolUseContext()

        result = None
        async for output in tool.call(input_data, context):
            result = output

        assert result is not None
        assert result.data.root == str(root.resolve())
        assert any(entry.endswith("dir1/") for entry in result.data.entries)
        assert any(entry.endswith("dir1/nested.txt") for entry in result.data.entries)
        assert any(entry.endswith("file.txt") for entry in result.data.entries)
        assert result.data.truncated is False


@pytest.mark.asyncio
async def test_ls_tool_empty_dir():
    """Empty directories should explicitly say they are empty."""
    tool = LSTool()

    with tempfile.TemporaryDirectory() as tmpdir:
        input_data = LSToolInput(path=tmpdir)
        context = ToolUseContext()

        result = None
        async for output in tool.call(input_data, context):
            result = output

        assert result is not None
        assert result.data.entries == []
        rendered = tool.render_result_for_assistant(result.data)
        assert "(empty directory)" in rendered


@pytest.mark.asyncio
async def test_ls_tool_ignore_patterns():
    """LS should respect ignore patterns."""
    tool = LSTool()

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "keep").mkdir()
        (root / "keep" / "keep.txt").write_text("data")
        (root / "skip").mkdir()
        (root / "skip" / "ignored.txt").write_text("secret")

        input_data = LSToolInput(path=tmpdir, ignore=["skip/**"])
        context = ToolUseContext()

        result = None
        async for output in tool.call(input_data, context):
            result = output

        assert result is not None
        assert any(entry.endswith("keep/keep.txt") for entry in result.data.entries)
        assert all("skip/" not in entry for entry in result.data.entries)


@pytest.mark.asyncio
async def test_ls_tool_skips_heavy_directories():
    """Default ignored directories should not be traversed."""
    tool = LSTool()

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        node_modules = root / "node_modules"
        node_modules.mkdir()
        (node_modules / "large.js").write_text("console.log('hi');")

        input_data = LSToolInput(path=tmpdir)
        context = ToolUseContext()

        result = None
        async for output in tool.call(input_data, context):
            result = output

        assert result is not None
        assert any(entry.endswith("node_modules/") for entry in result.data.entries)
        assert all(not entry.endswith("node_modules/large.js") for entry in result.data.entries)


@pytest.mark.asyncio
async def test_bash_tool_background_and_output():
    """Background bash commands should return a task id and stream output later."""
    bash_tool = BashTool()
    output_tool = TaskOutputTool()
    context = ToolUseContext()

    # Start a short-lived command in the background.
    start_input = BashToolInput(
        command="sleep 0.1 && echo hi", run_in_background=True, timeout=2000
    )

    start_result = None
    async for output in bash_tool.call(start_input, context):
        start_result = output

    assert start_result is not None
    task_id = start_result.data.background_task_id
    assert task_id

    # Allow the process to finish.
    await asyncio.sleep(0.3)

    poll_input = TaskOutputInput(task_id=task_id, block=True, timeout=3000)
    poll_result = None
    async for output in output_tool.call(poll_input, context):
        poll_result = output

    assert poll_result is not None
    assert poll_result.data.retrieval_status == "success"
    assert poll_result.data.task is not None
    assert poll_result.data.task.status in ("completed", "failed")
    assert "hi" in poll_result.data.task.output


@pytest.mark.asyncio
async def test_bash_background_stdin_detached():
    """Background commands should not hold the controlling TTY/stdin open."""
    bash_tool = BashTool()
    output_tool = TaskOutputTool()
    context = ToolUseContext()

    start_input = BashToolInput(
        command='python -c "import sys; data=sys.stdin.read(); print(f\\"len={len(data)}\\")"',
        run_in_background=True,
        timeout=2000,
    )

    start_result = None
    async for output in bash_tool.call(start_input, context):
        start_result = output

    assert start_result is not None
    task_id = start_result.data.background_task_id
    assert task_id

    # Allow brief time for the process to exit (should not wait on stdin).
    await asyncio.sleep(0.2)

    poll_input = TaskOutputInput(task_id=task_id, block=True, timeout=3000)
    poll_result = None
    async for output in output_tool.call(poll_input, context):
        poll_result = output

    assert poll_result is not None
    assert poll_result.data.retrieval_status == "success"
    assert poll_result.data.task is not None
    assert poll_result.data.task.status in ("completed", "failed")
    assert "len=0" in poll_result.data.task.output


@pytest.mark.asyncio
async def test_task_output_background_status_and_output():
    """TaskOutput should support non-blocking and blocking background checks."""
    bash_tool = BashTool()
    output_tool = TaskOutputTool()
    context = ToolUseContext()

    start_input = BashToolInput(
        command="sleep 0.2 && echo task-output-ok",
        run_in_background=True,
        timeout=2000,
    )

    start_result = None
    async for output in bash_tool.call(start_input, context):
        start_result = output

    assert start_result is not None
    task_id = start_result.data.background_task_id
    assert task_id

    immediate = None
    async for output in output_tool.call(
        TaskOutputInput(task_id=task_id, block=False),
        context,
    ):
        immediate = output

    assert immediate is not None
    assert immediate.data.retrieval_status in {"not_ready", "success"}
    assert immediate.data.task is not None
    assert immediate.data.task.task_type == "local_bash"

    await asyncio.sleep(0.3)

    completed = None
    async for output in output_tool.call(
        TaskOutputInput(task_id=task_id, block=True, timeout=3000),
        context,
    ):
        completed = output

    assert completed is not None
    assert completed.data.retrieval_status == "success"
    assert completed.data.task is not None
    assert completed.data.task.status in {"completed", "failed"}
    assert "task-output-ok" in completed.data.task.output


@pytest.mark.asyncio
@pytest.mark.usefixtures("reset_background_shell")
async def test_background_bash_ignores_execution_timeout() -> None:
    """Background bash should keep running regardless of execution timeout input."""
    bash_tool = BashTool()
    output_tool = TaskOutputTool()
    context = ToolUseContext()

    start_input = BashToolInput(
        command="sleep 0.25 && echo still-ran",
        run_in_background=True,
        timeout=10,
    )

    start_result = None
    async for output in bash_tool.call(start_input, context):
        start_result = output

    assert start_result is not None
    task_id = start_result.data.background_task_id
    assert task_id

    # If timeout were applied to background lifetime, this would fail before echo.
    await asyncio.sleep(0.4)
    status = get_background_status(task_id, consume=False)
    assert status["status"] == "completed"
    assert status["timed_out"] is False

    poll_result = None
    async for output in output_tool.call(
        TaskOutputInput(task_id=task_id, block=True, timeout=1000),
        context,
    ):
        poll_result = output

    assert poll_result is not None
    assert poll_result.data.retrieval_status == "success"
    assert poll_result.data.task is not None
    assert "still-ran" in poll_result.data.task.output


@pytest.mark.asyncio
@pytest.mark.usefixtures("reset_background_shell")
async def test_task_output_timeout_is_wait_timeout_only() -> None:
    """TaskOutput timeout should not imply the underlying task timed out."""
    bash_tool = BashTool()
    output_tool = TaskOutputTool()
    context = ToolUseContext()

    start_input = BashToolInput(
        command="sleep 0.35 && echo eventual-output",
        run_in_background=True,
    )
    start_result = None
    async for output in bash_tool.call(start_input, context):
        start_result = output

    assert start_result is not None
    task_id = start_result.data.background_task_id
    assert task_id

    timed_wait = None
    async for output in output_tool.call(
        TaskOutputInput(task_id=task_id, block=True, timeout=10),
        context,
    ):
        timed_wait = output

    assert timed_wait is not None
    assert timed_wait.data.retrieval_status == "timeout"
    assert timed_wait.data.task is not None
    assert timed_wait.data.task.status == "running"
    assert timed_wait.data.task.error is None

    await asyncio.sleep(0.45)
    finished = None
    async for output in output_tool.call(
        TaskOutputInput(task_id=task_id, block=True, timeout=2000),
        context,
    ):
        finished = output

    assert finished is not None
    assert finished.data.retrieval_status == "success"
    assert finished.data.task is not None
    assert finished.data.task.status == "completed"
    assert "eventual-output" in finished.data.task.output


@pytest.mark.asyncio
@pytest.mark.usefixtures("reset_background_shell")
async def test_foreground_timeout_auto_backgrounds_when_allowed() -> None:
    """Foreground timeout should auto-background eligible commands."""
    bash_tool = BashTool()
    output_tool = TaskOutputTool()
    context = ToolUseContext()

    result = None
    async for output in bash_tool.call(
        BashToolInput(
            command="sleep 0.35 && echo auto-bg-finished",
            timeout=50,
        ),
        context,
    ):
        if isinstance(output, ToolResult):
            result = output

    assert result is not None
    assert result.data.background_task_id
    assert result.data.exit_code == 0
    assert "moved to background" in (result.data.stderr or "")

    task_id = result.data.background_task_id
    assert task_id is not None

    await asyncio.sleep(0.5)
    completed = None
    async for output in output_tool.call(
        TaskOutputInput(task_id=task_id, block=True, timeout=2000),
        context,
    ):
        completed = output

    assert completed is not None
    assert completed.data.retrieval_status == "success"
    assert completed.data.task is not None
    assert completed.data.task.status == "completed"
    assert "auto-bg-finished" in completed.data.task.output


@pytest.mark.asyncio
@pytest.mark.usefixtures("reset_background_shell")
async def test_task_stop_can_stop_auto_backgrounded_task() -> None:
    """TaskStop should work for tasks auto-backgrounded by foreground timeout."""
    bash_tool = BashTool()
    stop_tool = TaskStopTool()
    context = ToolUseContext()

    result = None
    async for output in bash_tool.call(
        BashToolInput(
            command='python -c "import time; time.sleep(5)"',
            timeout=50,
        ),
        context,
    ):
        if isinstance(output, ToolResult):
            result = output

    assert result is not None
    task_id = result.data.background_task_id
    assert task_id

    stop_result = None
    async for output in stop_tool.call(TaskStopInput(task_id=task_id), context):
        stop_result = output

    assert stop_result is not None
    assert "Successfully stopped task" in stop_result.data.message
    status = get_background_status(task_id, consume=False)
    assert status["status"] == "killed"


@pytest.mark.asyncio
async def test_task_stop_kills_background_bash():
    """TaskStop should stop background bash tasks by task_id."""
    bash_tool = BashTool()
    stop_tool = TaskStopTool()
    context = ToolUseContext()

    start_input = BashToolInput(
        command='python -c "import time; time.sleep(2)"',
        run_in_background=True,
        timeout=3000,
    )

    start_result = None
    async for output in bash_tool.call(start_input, context):
        start_result = output

    assert start_result is not None
    task_id = start_result.data.background_task_id
    assert task_id

    validation = await stop_tool.validate_input(TaskStopInput(task_id=task_id), context)
    assert validation.result is True

    stop_result = None
    async for output in stop_tool.call(TaskStopInput(task_id=task_id), context):
        stop_result = output

    assert stop_result is not None
    assert stop_result.data.task_type == "local_bash"
    assert "Successfully stopped task" in stop_result.data.message
    status = get_background_status(task_id, consume=False)
    assert status["status"] == "killed"


def test_task_tool_input_supports_name_alias_for_teammate() -> None:
    """Task input should accept name field alias."""
    parsed = TaskToolInput(
        description="delegate auth review",
        prompt="Do work",
        subagent_type="general-purpose",
        team_name="alpha",
        name="researcher",
    )
    assert parsed.teammate_name == "researcher"
    assert parsed.name == "researcher"


def test_task_tool_input_accepts_worktree_isolation() -> None:
    parsed = TaskToolInput(
        description="implement feature",
        prompt="Implement feature",
        subagent_type="general-purpose",
        isolation="worktree",
    )
    assert parsed.isolation == "worktree"


@pytest.mark.asyncio
async def test_task_tool_validate_input_requires_description() -> None:
    tool = TaskTool(lambda: [])
    payload = TaskToolInput(
        description="desc",
        prompt="Do work",
        subagent_type="general-purpose",
    )
    payload.description = ""

    validation = await tool.validate_input(payload)
    assert validation.result is False
    assert "description is required" in (validation.message or "")


@pytest.mark.asyncio
async def test_bash_tool_supports_dangerously_disable_sandbox_alias() -> None:
    tool = BashTool()
    parsed = BashToolInput(
        command="touch /tmp/ripperdoc-test",
        sandbox=True,
        dangerouslyDisableSandbox=True,
    )
    assert parsed.dangerously_disable_sandbox is True
    # Sandbox override should bypass the unconditional sandbox dict path.
    sandbox_decision = await tool.check_permissions(BashToolInput(command="touch /tmp/x", sandbox=True), {})
    override_decision = await tool.check_permissions(parsed, {})
    assert isinstance(sandbox_decision, dict)
    assert not isinstance(override_decision, dict)


def test_create_task_worktree_creates_isolated_branch(tmp_path: Path) -> None:
    consume_session_worktrees()
    repo = tmp_path / "repo"
    repo.mkdir()

    def _git(*args: str, cwd: Path = repo) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )

    _git("init")
    _git("config", "user.name", "Ripperdoc Test")
    _git("config", "user.email", "test@example.com")
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _git("add", "README.md")
    _git("commit", "-m", "init")

    session = create_task_worktree(task_id="agent_deadbeef", base_path=repo)
    assert session.worktree_path.exists()
    assert session.worktree_path.is_dir()
    assert str(session.worktree_path).startswith(str(repo / ".ripperdoc" / "worktrees"))

    branch = _git("rev-parse", "--abbrev-ref", "HEAD", cwd=session.worktree_path).stdout.strip()
    assert branch == session.branch

    # cleanup
    _git("worktree", "remove", "--force", str(session.worktree_path))
    _git("branch", "-D", session.branch)
    consume_session_worktrees()


def test_create_task_worktree_resumes_existing_named_worktree(tmp_path: Path) -> None:
    consume_session_worktrees()
    repo = tmp_path / "repo"
    repo.mkdir()

    def _git(*args: str, cwd: Path = repo) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )

    _git("init")
    _git("config", "user.name", "Ripperdoc Test")
    _git("config", "user.email", "test@example.com")
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _git("add", "README.md")
    _git("commit", "-m", "init")

    first = create_task_worktree(task_id="agent_first", base_path=repo, requested_name="shared")
    second = create_task_worktree(task_id="agent_second", base_path=repo, requested_name="shared")

    assert first.worktree_path == second.worktree_path
    assert second.branch == first.branch

    _git("worktree", "remove", "--force", str(first.worktree_path))
    _git("branch", "-D", first.branch)
    consume_session_worktrees()


def test_create_task_worktree_resumes_legacy_ripperdoc_worktree(tmp_path: Path) -> None:
    consume_session_worktrees()
    repo = tmp_path / "repo"
    repo.mkdir()

    def _git(*args: str, cwd: Path = repo) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )

    _git("init")
    _git("config", "user.name", "Ripperdoc Test")
    _git("config", "user.email", "test@example.com")
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _git("add", "README.md")
    _git("commit", "-m", "init")

    legacy_path = repo / ".ripperdoc" / "worktrees" / "legacy-shared"
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    _git("worktree", "add", "-b", "worktree-legacy-shared", str(legacy_path), "HEAD")

    resumed = create_task_worktree(
        task_id="agent_legacy",
        base_path=repo,
        requested_name="legacy-shared",
    )
    assert resumed.worktree_path == legacy_path.resolve()

    _git("worktree", "remove", "--force", str(legacy_path))
    _git("branch", "-D", "worktree-legacy-shared")
    consume_session_worktrees()


def test_create_task_worktree_requires_git_repo(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="git repository"):
        create_task_worktree(task_id="agent_123", base_path=tmp_path)


def test_create_task_worktree_prefers_worktree_create_hook(monkeypatch, tmp_path: Path) -> None:
    consume_session_worktrees()
    repo = tmp_path / "repo"
    repo.mkdir()

    def _git(*args: str, cwd: Path = repo) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )

    _git("init")
    _git("config", "user.name", "Ripperdoc Test")
    _git("config", "user.email", "test@example.com")
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _git("add", "README.md")
    _git("commit", "-m", "init")

    hook_worktree = tmp_path / "hook-worktree"
    hook_worktree.mkdir()
    captured: dict[str, str] = {}

    def fake_run_worktree_create(worktree_name: str):
        captured["name"] = worktree_name
        output = SimpleNamespace(
            hook_specific_output={"worktreePath": str(hook_worktree)},
            raw_output="",
            additional_context=None,
            reason=None,
        )
        return SimpleNamespace(
            should_block=False,
            outputs=[output],
        )

    monkeypatch.setattr("ripperdoc.utils.collaboration.worktree._has_worktree_create_hook", lambda: True)
    monkeypatch.setattr(
        "ripperdoc.utils.collaboration.worktree.hook_manager.run_worktree_create",
        fake_run_worktree_create,
    )

    session = create_task_worktree(
        task_id="agent_hookpref",
        base_path=repo,
        requested_name="hook-pref",
    )

    assert captured["name"] == "hook-pref"
    assert session.hook_based is True
    assert session.worktree_path == hook_worktree.resolve()
    assert session.repo_root == repo.resolve()
    consume_session_worktrees()


def test_create_task_worktree_copies_worktreeinclude_files(tmp_path: Path) -> None:
    consume_session_worktrees()
    repo = tmp_path / "repo"
    repo.mkdir()

    def _git(*args: str, cwd: Path = repo) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )

    _git("init")
    _git("config", "user.name", "Ripperdoc Test")
    _git("config", "user.email", "test@example.com")
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    (repo / ".gitignore").write_text("*.secret\nignored_dir/\n", encoding="utf-8")
    (repo / ".worktreeinclude").write_text(
        "config.secret\nignored_dir/data.txt\n",
        encoding="utf-8",
    )
    _git("add", "README.md", ".gitignore", ".worktreeinclude")
    _git("commit", "-m", "init")

    (repo / "config.secret").write_text("token=abc\n", encoding="utf-8")
    (repo / "ignored_dir").mkdir()
    (repo / "ignored_dir" / "data.txt").write_text("cached\n", encoding="utf-8")
    (repo / "other.secret").write_text("skip\n", encoding="utf-8")

    session = create_task_worktree(task_id="agent_includes", base_path=repo)

    assert (session.worktree_path / "config.secret").read_text(encoding="utf-8") == "token=abc\n"
    assert (
        session.worktree_path / "ignored_dir" / "data.txt"
    ).read_text(encoding="utf-8") == "cached\n"
    assert not (session.worktree_path / "other.secret").exists()

    _git("worktree", "remove", "--force", str(session.worktree_path))
    _git("branch", "-D", session.branch)
    consume_session_worktrees()


@pytest.mark.asyncio
async def test_enter_worktree_tool_switches_session_directory(monkeypatch, tmp_path: Path) -> None:
    consume_session_worktrees()
    repo = tmp_path / "repo"
    repo.mkdir()

    def _git(*args: str, cwd: Path = repo) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )

    _git("init")
    _git("config", "user.name", "Ripperdoc Test")
    _git("config", "user.email", "test@example.com")
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _git("add", "README.md")
    _git("commit", "-m", "init")

    tool = EnterWorktreeTool()
    captured: dict[str, str] = {}

    def _set_working_directory(path: str) -> None:
        captured["working_directory"] = path

    monkeypatch.chdir(repo)
    outputs = [
        item
        async for item in tool.call(
            EnterWorktreeToolInput(name="mid-session"),
            ToolUseContext(
                working_directory=str(repo),
                set_working_directory=_set_working_directory,
                message_id="msg_123",
            ),
        )
    ]

    assert outputs
    result = outputs[-1]
    assert result.data.worktreePath.startswith(str(repo / ".ripperdoc" / "worktrees"))
    assert captured["working_directory"] == result.data.worktreePath
    assert os.getcwd() == result.data.worktreePath

    monkeypatch.chdir(repo)
    _git("worktree", "remove", "--force", result.data.worktreePath)
    _git("branch", "-D", result.data.worktreeBranch)
    consume_session_worktrees()


@pytest.mark.asyncio
async def test_enter_worktree_tool_rejects_when_already_in_worktree(tmp_path: Path) -> None:
    consume_session_worktrees()
    repo = tmp_path / "repo"
    repo.mkdir()

    def _git(*args: str, cwd: Path = repo) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )

    _git("init")
    _git("config", "user.name", "Ripperdoc Test")
    _git("config", "user.email", "test@example.com")
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _git("add", "README.md")
    _git("commit", "-m", "init")

    session = create_task_worktree(task_id="agent_guard", base_path=repo)
    tool = EnterWorktreeTool()
    validation = await tool.validate_input(
        EnterWorktreeToolInput(),
        ToolUseContext(working_directory=str(session.worktree_path)),
    )
    assert validation.result is False
    assert validation.message == "Already in a worktree session"

    _git("worktree", "remove", "--force", str(session.worktree_path))
    _git("branch", "-D", session.branch)
    consume_session_worktrees()


def test_create_task_worktree_supports_pr_number(tmp_path: Path) -> None:
    consume_session_worktrees()
    repo = tmp_path / "repo"
    origin = tmp_path / "origin.git"
    repo.mkdir()

    def _git(*args: str, cwd: Path = repo) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )

    subprocess.run(["git", "init", "--bare", str(origin)], check=True)
    _git("init")
    _git("config", "user.name", "Ripperdoc Test")
    _git("config", "user.email", "test@example.com")
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _git("add", "README.md")
    _git("commit", "-m", "init")
    default_branch = _git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    _git("remote", "add", "origin", str(origin))
    _git("push", "-u", "origin", default_branch)

    _git("checkout", "-b", "feature-pr")
    (repo / "PR_ONLY.md").write_text("from pr\n", encoding="utf-8")
    _git("add", "PR_ONLY.md")
    _git("commit", "-m", "pr head")
    _git("push", "origin", "HEAD:refs/pull/123/head")
    _git("checkout", default_branch)

    session = create_task_worktree(
        task_id="agent_pr",
        base_path=repo,
        requested_name="pr-123",
        pr_number=123,
    )
    assert (session.worktree_path / "PR_ONLY.md").exists()

    _git("worktree", "remove", "--force", str(session.worktree_path))
    _git("branch", "-D", session.branch)
    consume_session_worktrees()


def test_cleanup_registered_worktrees_removes_branch_and_directory(tmp_path: Path) -> None:
    consume_session_worktrees()
    repo = tmp_path / "repo"
    repo.mkdir()

    def _git(*args: str, cwd: Path = repo) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )

    _git("init")
    _git("config", "user.name", "Ripperdoc Test")
    _git("config", "user.email", "test@example.com")
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _git("add", "README.md")
    _git("commit", "-m", "init")

    session = create_task_worktree(task_id="agent_cleanup", base_path=repo)
    results = cleanup_registered_worktrees(force=True)
    assert len(results) == 1
    assert results[0].worktree_path == session.worktree_path
    assert results[0].removed is True
    assert results[0].branch_deleted is True
    assert not session.worktree_path.exists()


def test_cleanup_worktree_session_uses_worktree_remove_hook(monkeypatch, tmp_path: Path) -> None:
    hook_worktree = tmp_path / "hook-worktree"
    hook_worktree.mkdir()
    called: dict[str, str] = {}

    session = WorktreeSession(
        repo_root=tmp_path,
        worktree_path=hook_worktree,
        branch="",
        name="hooked",
        hook_based=True,
    )

    def fake_run_worktree_remove(worktree_path: str):
        called["path"] = worktree_path
        return SimpleNamespace(should_block=False, has_errors=False, errors=[])

    monkeypatch.setattr("ripperdoc.utils.collaboration.worktree._has_worktree_remove_hook", lambda: True)
    monkeypatch.setattr(
        "ripperdoc.utils.collaboration.worktree.hook_manager.run_worktree_remove",
        fake_run_worktree_remove,
    )

    result = cleanup_worktree_session(session)

    assert called["path"] == str(hook_worktree)
    assert result.removed is True
    assert result.branch_deleted is False
    assert result.error is None


def test_sync_worktree_configuration_syncs_settings_hooks_and_symlink(
    monkeypatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    worktree = tmp_path / "worktree"
    repo.mkdir()
    worktree.mkdir()

    (repo / "settings.local.json").write_text('{"local":true}\n', encoding="utf-8")
    (repo / ".husky").mkdir()
    (repo / ".ripperdoc").mkdir()
    (repo / ".ripperdoc" / "config.json").write_text(
        json.dumps({"worktree": {"symlinkDirectories": ["cache-dir"]}}),
        encoding="utf-8",
    )
    (repo / "cache-dir").mkdir()
    (repo / "cache-dir" / "token.txt").write_text("abc\n", encoding="utf-8")

    run_git_calls: list[tuple[list[str], Path]] = []

    def fake_run_git(
        args: list[str], *, cwd: Path, timeout_sec: float = 20.0
    ) -> subprocess.CompletedProcess[str]:
        run_git_calls.append((args, cwd))
        return subprocess.CompletedProcess(
            args=["git", *args], returncode=0, stdout="", stderr=""
        )

    monkeypatch.setattr("ripperdoc.utils.collaboration.worktree._run_git", fake_run_git)

    sync_worktree_configuration(repo_root=repo, worktree_path=worktree)

    assert (worktree / "settings.local.json").read_text(encoding="utf-8") == '{"local":true}\n'
    assert any(
        args[:2] == ["config", "core.hooksPath"] and cwd == worktree
        for args, cwd in run_git_calls
    )
    link_path = worktree / "cache-dir"
    assert link_path.is_symlink()
    assert link_path.resolve() == (repo / "cache-dir").resolve()


def test_has_worktree_changes_detects_dirty_and_committed_changes(tmp_path: Path) -> None:
    consume_session_worktrees()
    repo = tmp_path / "repo"
    repo.mkdir()

    def _git(*args: str, cwd: Path = repo) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )

    _git("init")
    _git("config", "user.name", "Ripperdoc Test")
    _git("config", "user.email", "test@example.com")
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _git("add", "README.md")
    _git("commit", "-m", "init")

    session = create_task_worktree(task_id="agent_changes", base_path=repo)
    baseline = session.head_commit or session.branch

    assert has_worktree_changes(worktree_path=session.worktree_path, baseline_ref=baseline) is False

    (session.worktree_path / "README.md").write_text("hello world\n", encoding="utf-8")
    assert has_worktree_changes(worktree_path=session.worktree_path, baseline_ref=baseline) is True

    _git("add", "README.md", cwd=session.worktree_path)
    _git("commit", "-m", "change", cwd=session.worktree_path)
    assert has_worktree_changes(worktree_path=session.worktree_path, baseline_ref=baseline) is True

    _git("worktree", "remove", "--force", str(session.worktree_path))
    _git("branch", "-D", session.branch)
    consume_session_worktrees()


def test_task_tool_autocleans_worktree_when_no_changes(tmp_path: Path) -> None:
    consume_session_worktrees()
    repo = tmp_path / "repo"
    repo.mkdir()

    def _git(*args: str, cwd: Path = repo) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )

    _git("init")
    _git("config", "user.name", "Ripperdoc Test")
    _git("config", "user.email", "test@example.com")
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _git("add", "README.md")
    _git("commit", "-m", "init")

    session = create_task_worktree(task_id="agent_autoclean", base_path=repo)
    tool = TaskTool(lambda: [])
    record = AgentRunRecord(
        agent_id="agent_autoclean",
        agent_type="general-purpose",
        tools=[],
        system_prompt="",
        history=[],
        missing_tools=[],
        model_used=None,
        start_time=0.0,
        status="completed",
        result_text="done",
        isolation_mode="worktree",
        worktree_path=str(session.worktree_path),
        worktree_branch=session.branch,
        worktree_name=session.name,
        worktree_repo_root=str(session.repo_root),
        worktree_head_commit=session.head_commit,
    )

    tool._maybe_autocleanup_worktree(record)  # noqa: SLF001 - testing internal lifecycle behavior
    assert record.worktree_path is None
    assert record.worktree_branch is None
    assert not session.worktree_path.exists()


@pytest.mark.asyncio
async def test_bash_tool_streaming_output():
    """BashTool should emit progress when stream_output is enabled."""
    bash_tool = BashTool()
    context = ToolUseContext()

    input_data = BashToolInput(
        command="python -c \"import sys, time; print('hi'); sys.stdout.flush(); time.sleep(0.1); print('bye'); sys.stdout.flush()\"",
        timeout=2000,
    )

    progress_lines = []
    result = None

    async for output in bash_tool.call(input_data, context):
        if isinstance(output, ToolProgress):
            progress_lines.append(output.content)
        else:
            result = output

    assert any("hi" in line for line in progress_lines)
    assert any("bye" in line for line in progress_lines)
    assert result is not None
    assert "hi" in result.data.stdout
    assert "bye" in result.data.stdout
    assert result.data.exit_code == 0


@pytest.mark.asyncio
async def test_bash_tool_handles_very_long_single_line_output():
    """Foreground BashTool should handle long single lines without separator errors."""
    bash_tool = BashTool()
    context = ToolUseContext()

    input_data = BashToolInput(
        command='python -c "import sys; sys.stdout.write(\'a\' * 200000)"',
        timeout=5000,
    )

    result = None
    async for output in bash_tool.call(input_data, context):
        if isinstance(output, ToolResult):
            result = output

    assert result is not None
    assert result.data.exit_code == 0
    assert result.data.stderr == ""
    assert result.data.is_truncated is True
    assert result.data.truncation_details
    assert any("stdout:" in item and "omitted" in item for item in result.data.truncation_details)
    assert any("line 1" in item for item in result.data.truncation_details)

    rendered = bash_tool.render_result_for_assistant(result.data)
    assert "Truncation details:" in rendered


@pytest.mark.asyncio
async def test_task_stop_running_task():
    """TaskStop should terminate a running background command."""
    bash_tool = BashTool()
    stop_tool = TaskStopTool()
    context = ToolUseContext()

    start_input = BashToolInput(
        command='python -c "import time; time.sleep(2)"',
        run_in_background=True,
        timeout=3000,
    )

    start_result = None
    async for output in bash_tool.call(start_input, context):
        start_result = output

    assert start_result is not None
    task_id = start_result.data.background_task_id
    assert task_id

    stop_input = TaskStopInput(task_id=task_id)
    validation = await stop_tool.validate_input(stop_input, context)
    assert validation.result is True

    stop_result = None
    async for output in stop_tool.call(stop_input, context):
        stop_result = output

    assert stop_result is not None
    assert stop_result.data.task_type == "local_bash"
    assert "Successfully stopped task" in stop_result.data.message
    status = get_background_status(task_id, consume=False)
    assert status["status"] == "killed"


@pytest.mark.asyncio
async def test_task_stop_completed_task():
    """Stopping a completed task should report not running."""
    bash_tool = BashTool()
    stop_tool = TaskStopTool()
    context = ToolUseContext()

    start_input = BashToolInput(
        command="echo done",
        run_in_background=True,
        timeout=1000,
    )

    start_result = None
    async for output in bash_tool.call(start_input, context):
        start_result = output

    assert start_result is not None
    task_id = start_result.data.background_task_id
    assert task_id

    # allow completion
    await asyncio.sleep(0.2)

    stop_input = TaskStopInput(task_id=task_id)
    validation = await stop_tool.validate_input(stop_input, context)
    assert validation.result is False
    assert "not running" in (validation.message or "").lower()


@pytest.mark.asyncio
async def test_task_stop_validation_missing():
    """Validation should fail for unknown task ids."""
    stop_tool = TaskStopTool()
    context = ToolUseContext()
    validation = await stop_tool.validate_input(TaskStopInput(task_id="missing"), context)
    assert validation.result is False


class DummyInput(BaseModel):
    """Input for dummy tools."""

    pass


class DummyTool(Tool[DummyInput, str]):
    """Minimal tool used for tool search tests."""

    def __init__(self, name: str, deferred: bool = False) -> None:
        super().__init__()
        self._name = name
        self._deferred = deferred

    @property
    def name(self) -> str:
        return self._name

    async def description(self) -> str:
        return f"{self._name} tool"

    @property
    def input_schema(self) -> type[DummyInput]:
        return DummyInput

    async def prompt(self, yolo_mode: bool = False) -> str:  # noqa: ARG002
        return ""

    def defer_loading(self) -> bool:
        return self._deferred

    def render_result_for_assistant(self, output: str) -> str:
        return output

    def render_tool_use_message(self, input_data: DummyInput, verbose: bool = False) -> str:  # noqa: ARG002
        return self._name

    async def call(
        self,
        input_data: DummyInput,
        context: ToolUseContext,  # noqa: ARG002
    ):  # type: ignore[override]
        yield ToolResult(data="ok", result_for_assistant="ok")


@pytest.mark.asyncio
async def test_tool_search_activates_deferred_tools():
    """ToolSearch should surface deferred tools and activate them."""
    active = DummyTool("Active")
    deferred = DummyTool("Deferred", deferred=True)
    registry = ToolRegistry([active, deferred])

    search_tool = ToolSearchTool()
    context = ToolUseContext(tool_registry=registry)
    input_data = ToolSearchInput(query="defer", max_results=3, include_active=True)

    result = None
    async for output in search_tool.call(input_data, context):
        result = output

    assert result is not None
    assert "Deferred" in result.data.activated
    assert registry.is_active("Deferred")
    assert any(match.name == "Deferred" for match in result.data.matches)
