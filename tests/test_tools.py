"""Test tools."""

import asyncio
import pytest
import tempfile
from pathlib import Path
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
from ripperdoc.tools.bash_output_tool import BashOutputTool, BashOutputInput
from ripperdoc.tools.kill_bash_tool import KillBashTool, KillBashInput
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
    output_tool = BashOutputTool()
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

    poll_input = BashOutputInput(task_id=task_id, consume=True)
    poll_result = None
    async for output in output_tool.call(poll_input, context):
        poll_result = output

    assert poll_result is not None
    assert poll_result.data.status in ("completed", "failed")
    assert "hi" in poll_result.data.stdout


@pytest.mark.asyncio
async def test_bash_background_stdin_detached():
    """Background commands should not hold the controlling TTY/stdin open."""
    bash_tool = BashTool()
    output_tool = BashOutputTool()
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

    poll_input = BashOutputInput(task_id=task_id, consume=True)
    poll_result = None
    async for output in output_tool.call(poll_input, context):
        poll_result = output

    assert poll_result is not None
    assert poll_result.data.status in ("completed", "failed")
    assert "len=0" in poll_result.data.stdout


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
async def test_kill_bash_running_task():
    """KillBash should terminate a running background command."""
    bash_tool = BashTool()
    kill_tool = KillBashTool()
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

    kill_input = KillBashInput(task_id=task_id)
    validation = await kill_tool.validate_input(kill_input, context)
    assert validation.result is True

    kill_result = None
    async for output in kill_tool.call(kill_input, context):
        kill_result = output

    assert kill_result is not None
    assert kill_result.data.success is True
    status = get_background_status(task_id, consume=False)
    assert status["status"] == "killed"


@pytest.mark.asyncio
async def test_kill_bash_completed_task():
    """Killing a completed task should report not running."""
    bash_tool = BashTool()
    kill_tool = KillBashTool()
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

    kill_input = KillBashInput(task_id=task_id)
    validation = await kill_tool.validate_input(kill_input, context)
    assert validation.result is True

    kill_result = None
    async for output in kill_tool.call(kill_input, context):
        kill_result = output

    assert kill_result is not None
    assert kill_result.data.success is False
    assert "not running" in kill_result.data.message.lower()


@pytest.mark.asyncio
async def test_kill_bash_validation_missing():
    """Validation should fail for unknown task ids."""
    kill_tool = KillBashTool()
    context = ToolUseContext()
    validation = await kill_tool.validate_input(KillBashInput(task_id="missing"), context)
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
