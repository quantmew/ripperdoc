"""Tests for the programmatic executor module."""

import pytest
from typing import Any, AsyncGenerator, Optional
from unittest.mock import MagicMock

from pydantic import BaseModel

from ripperdoc.core.programmatic_executor import (
    ProgrammaticContext,
    validate_code,
    execute_programmatic_code,
)
from ripperdoc.core.tool import Tool, ToolOutput, ToolResult, ToolUseContext, ValidationResult


class MockToolInput(BaseModel):
    """Mock tool input schema."""
    pattern: Optional[str] = None
    file_path: Optional[str] = None


class MockTool(Tool[MockToolInput, list]):
    """A mock tool for testing."""

    def __init__(self, name: str, return_value: Any = None):
        super().__init__()
        self._name = name
        self._return_value = return_value or []

    @property
    def name(self) -> str:
        return self._name

    async def description(self) -> str:
        return f"Mock {self._name} tool"

    @property
    def input_schema(self) -> type[MockToolInput]:
        return MockToolInput

    async def prompt(self, yolo_mode: bool = False) -> str:
        return ""

    async def validate_input(
        self, input_data: MockToolInput, context: Optional[ToolUseContext] = None
    ) -> ValidationResult:
        return ValidationResult(result=True)

    async def call(
        self, input_data: MockToolInput, context: ToolUseContext
    ) -> AsyncGenerator[ToolOutput, None]:
        yield ToolResult(data=self._return_value)

    def render_result_for_assistant(self, output: list) -> str:
        return str(output)

    def render_tool_use_message(self, input_data: MockToolInput, verbose: bool = False) -> str:
        return f"Mock tool call: {self._name}"


class TestCodeValidator:
    """Tests for code validation."""

    def test_allows_safe_code(self):
        """Test that safe code passes validation."""
        code = """
x = 1 + 2
result = [i * 2 for i in range(10)]
"""
        errors = validate_code(code)
        assert errors == []

    def test_allows_mock_modules(self):
        """Test that mock modules (os, sys, etc.) are allowed at validation time."""
        # These modules have mock replacements, so they should pass validation
        for module in ["os", "subprocess", "sys", "shutil", "pathlib", "io"]:
            code = f"import {module}"
            errors = validate_code(code)
            assert errors == [], f"Module {module} should be allowed (has mock)"

    def test_blocks_other_forbidden_imports(self):
        """Test that forbidden imports without mocks are blocked."""
        # These modules don't have mocks and should be blocked
        for module in ["socket", "http", "urllib", "pickle", "multiprocessing"]:
            code = f"import {module}"
            errors = validate_code(code)
            assert len(errors) > 0, f"Module {module} should be blocked"

    def test_allows_mock_from_imports(self):
        """Test that from ... import is allowed for mock modules."""
        code = "from os import path"
        errors = validate_code(code)
        assert errors == [], "from os import should be allowed (has mock)"

    def test_allows_safe_modules(self):
        """Test that allowed modules pass validation."""
        for module in ["json", "re", "math", "collections"]:
            code = f"import {module}"
            errors = validate_code(code)
            assert errors == [], f"Module {module} should be allowed"

    def test_blocks_exec_eval(self):
        """Test that exec/eval are blocked."""
        for func in ["exec", "eval", "compile"]:
            code = f'{func}("print(1)")'
            errors = validate_code(code)
            assert len(errors) > 0

    def test_blocks_open(self):
        """Test that open() is blocked."""
        code = 'open("/etc/passwd", "r")'
        errors = validate_code(code)
        assert len(errors) > 0

    def test_blocks_dangerous_attributes(self):
        """Test that dangerous attributes are blocked."""
        for attr in ["__class__", "__bases__", "__subclasses__", "__globals__"]:
            code = f"x.{attr}"
            errors = validate_code(code)
            assert len(errors) > 0

    def test_syntax_error(self):
        """Test that syntax errors are caught."""
        code = "def foo(:"
        errors = validate_code(code)
        assert len(errors) > 0
        assert "syntax" in errors[0].lower()


class TestProgrammaticContext:
    """Tests for ProgrammaticContext."""

    def test_log(self):
        """Test logging functionality."""
        ctx = ProgrammaticContext(
            tools={},
            tool_context=MagicMock(spec=ToolUseContext),
            working_directory="/tmp",
        )
        ctx.log("Test message")
        logs = ctx.get_logs()
        assert len(logs) == 1
        assert "Test message" in logs[0]

    def test_set_get_result(self):
        """Test result setting and getting."""
        ctx = ProgrammaticContext(
            tools={},
            tool_context=MagicMock(spec=ToolUseContext),
            working_directory="/tmp",
        )
        ctx.set_result({"key": "value"})
        assert ctx.get_result() == {"key": "value"}
        assert ctx._result_set is True

    def test_get_available_tools(self):
        """Test getting available tools."""
        tools = {"Glob": MagicMock(), "Read": MagicMock()}
        ctx = ProgrammaticContext(
            tools=tools,
            tool_context=MagicMock(spec=ToolUseContext),
            working_directory="/tmp",
        )
        available = ctx.get_available_tools()
        assert set(available) == {"Glob", "Read"}

    def test_get_working_directory(self):
        """Test getting working directory."""
        ctx = ProgrammaticContext(
            tools={},
            tool_context=MagicMock(spec=ToolUseContext),
            working_directory="/home/user",
        )
        assert ctx.get_working_directory() == "/home/user"

    def test_timeout_check(self):
        """Test timeout checking."""
        ctx = ProgrammaticContext(
            tools={},
            tool_context=MagicMock(spec=ToolUseContext),
            working_directory="/tmp",
            timeout_seconds=0.1,
        )
        assert not ctx.is_timeout()
        # Wait for timeout
        import time
        time.sleep(0.15)
        assert ctx.is_timeout()


class TestToolCall:
    """Tests for tool_call functionality (synchronous)."""

    def test_tool_call_success(self):
        """Test successful tool call."""
        mock_tool = MockTool("Glob", return_value=["file1.py", "file2.py"])
        mock_context = MagicMock(spec=ToolUseContext)

        ctx = ProgrammaticContext(
            tools={"Glob": mock_tool},
            tool_context=mock_context,
            working_directory="/tmp",
        )

        result = ctx.tool_call("Glob", {"pattern": "*.py"})
        assert result == ["file1.py", "file2.py"]
        assert ctx._tool_call_count == 1

    def test_tool_call_unknown_tool(self):
        """Test that unknown tool raises error."""
        ctx = ProgrammaticContext(
            tools={},
            tool_context=MagicMock(spec=ToolUseContext),
            working_directory="/tmp",
        )

        with pytest.raises(ValueError, match="not available"):
            ctx.tool_call("Unknown", {})

    def test_tool_call_cancelled(self):
        """Test that cancelled context raises error."""
        ctx = ProgrammaticContext(
            tools={"Glob": MockTool("Glob")},
            tool_context=MagicMock(spec=ToolUseContext),
            working_directory="/tmp",
        )
        ctx.cancel()

        with pytest.raises(RuntimeError, match="cancelled"):
            ctx.tool_call("Glob", {})


@pytest.mark.asyncio
class TestExecuteProgrammaticCode:
    """Tests for execute_programmatic_code."""

    async def test_simple_execution(self):
        """Test simple code execution."""
        code = """
x = 1 + 2
ctx.set_result(x)
"""
        result = await execute_programmatic_code(
            code=code,
            tools={},
            tool_context=MagicMock(spec=ToolUseContext),
        )
        assert result.success is True
        assert result.result == 3

    async def test_with_logging(self):
        """Test code execution with logging."""
        code = """
ctx.log("Starting")
ctx.log("Done")
ctx.set_result("OK")
"""
        result = await execute_programmatic_code(
            code=code,
            tools={},
            tool_context=MagicMock(spec=ToolUseContext),
        )
        assert result.success is True
        assert len(result.logs) == 2
        assert "Starting" in result.logs[0]
        assert "Done" in result.logs[1]

    async def test_with_tool_call(self):
        """Test code execution with tool calls."""
        mock_tool = MockTool("Glob", return_value=["a.py", "b.py"])
        code = """
files = ctx.tool_call("Glob", {"pattern": "*.py"})
ctx.set_result(files)
"""
        result = await execute_programmatic_code(
            code=code,
            tools={"Glob": mock_tool},
            tool_context=MagicMock(spec=ToolUseContext),
        )
        assert result.success is True
        assert result.result == ["a.py", "b.py"]
        assert result.tool_call_count == 1

    async def test_mock_module_blocks_dangerous_operation(self):
        """Test that mock modules block dangerous operations at runtime."""
        code = """
import os
os.system("ls")
"""
        result = await execute_programmatic_code(
            code=code,
            tools={},
            tool_context=MagicMock(spec=ToolUseContext),
        )
        assert result.success is False
        # Mock module raises PermissionError with helpful message
        assert "PermissionError" in result.error
        assert "tool_call" in result.error.lower() or "ctx.tool_call" in result.error

    async def test_allowed_import(self):
        """Test that allowed imports work."""
        code = """
import json
data = json.dumps({"key": "value"})
ctx.set_result(data)
"""
        result = await execute_programmatic_code(
            code=code,
            tools={},
            tool_context=MagicMock(spec=ToolUseContext),
        )
        assert result.success is True
        assert '"key"' in result.result

    async def test_runtime_error(self):
        """Test handling of runtime errors."""
        code = """
x = 1 / 0
"""
        result = await execute_programmatic_code(
            code=code,
            tools={},
            tool_context=MagicMock(spec=ToolUseContext),
        )
        assert result.success is False
        assert "ZeroDivision" in result.error

    async def test_error_includes_line_numbered_code(self):
        """Test that errors include original code with line numbers."""
        code = """data = {"key": "value"}
result = data["missing"]
ctx.set_result(result)
"""
        result = await execute_programmatic_code(
            code=code,
            tools={},
            tool_context=MagicMock(spec=ToolUseContext),
        )
        assert result.success is False
        assert "KeyError" in result.error
        # Verify line numbers are present
        assert "1│" in result.error
        assert "2│" in result.error
        assert "3│" in result.error
        # Verify the actual code is included
        assert 'data["missing"]' in result.error
        assert "Code:" in result.error
        assert "Traceback:" in result.error
        # Verify traceback line number is corrected (line 2 is the error, not line 3)
        # The wrapper function adds 1 line, so we need to check the corrected line
        assert 'line 2' in result.error  # Error is on line 2 of original code

    async def test_print_redirected_to_log(self):
        """Test that print() is redirected to ctx.log()."""
        code = """
print("Hello from print")
ctx.set_result("done")
"""
        result = await execute_programmatic_code(
            code=code,
            tools={},
            tool_context=MagicMock(spec=ToolUseContext),
        )
        assert result.success is True
        assert len(result.logs) >= 1
        assert "Hello from print" in result.logs[0]

    async def test_data_processing(self):
        """Test complex data processing."""
        code = """
import json
import re

data = [
    {"name": "file1.py", "size": 100},
    {"name": "file2.txt", "size": 200},
    {"name": "file3.py", "size": 150},
]

# Filter Python files
py_files = [f for f in data if f["name"].endswith(".py")]

# Calculate total size
total = sum(f["size"] for f in py_files)

ctx.set_result({
    "count": len(py_files),
    "total_size": total,
    "files": [f["name"] for f in py_files]
})
"""
        result = await execute_programmatic_code(
            code=code,
            tools={},
            tool_context=MagicMock(spec=ToolUseContext),
        )
        assert result.success is True
        assert result.result["count"] == 2
        assert result.result["total_size"] == 250
        assert result.result["files"] == ["file1.py", "file3.py"]


@pytest.mark.asyncio
class TestTimeout:
    """Tests for timeout handling."""

    async def test_timeout_execution(self):
        """Test that long-running code times out."""
        code = """
import time
time.sleep(10)
ctx.set_result("done")
"""
        result = await execute_programmatic_code(
            code=code,
            tools={},
            tool_context=MagicMock(spec=ToolUseContext),
            timeout_seconds=0.5,
        )
        assert result.success is False
        assert "timed out" in result.error.lower()


@pytest.mark.asyncio
class TestMockModules:
    """Tests for mock module behavior."""

    async def test_os_path_join_works(self):
        """Test that safe os.path operations work."""
        code = """
import os
result = os.path.join("/home", "user", "file.txt")
ctx.set_result(result)
"""
        result = await execute_programmatic_code(
            code=code,
            tools={},
            tool_context=MagicMock(spec=ToolUseContext),
        )
        assert result.success is True
        assert result.result == "/home/user/file.txt"

    async def test_os_environ_is_empty(self):
        """Test that os.environ returns empty dict."""
        code = """
import os
ctx.set_result(dict(os.environ))
"""
        result = await execute_programmatic_code(
            code=code,
            tools={},
            tool_context=MagicMock(spec=ToolUseContext),
        )
        assert result.success is True
        assert result.result == {}

    async def test_subprocess_run_blocked(self):
        """Test that subprocess.run is blocked."""
        code = """
import subprocess
subprocess.run(["ls"])
"""
        result = await execute_programmatic_code(
            code=code,
            tools={},
            tool_context=MagicMock(spec=ToolUseContext),
        )
        assert result.success is False
        assert "PermissionError" in result.error

    async def test_sys_exit_blocked(self):
        """Test that sys.exit is blocked."""
        code = """
import sys
sys.exit(0)
"""
        result = await execute_programmatic_code(
            code=code,
            tools={},
            tool_context=MagicMock(spec=ToolUseContext),
        )
        assert result.success is False
        assert "PermissionError" in result.error

    async def test_shutil_copy_blocked(self):
        """Test that shutil.copy is blocked."""
        code = """
import shutil
shutil.copy("/a", "/b")
"""
        result = await execute_programmatic_code(
            code=code,
            tools={},
            tool_context=MagicMock(spec=ToolUseContext),
        )
        assert result.success is False
        assert "PermissionError" in result.error

    async def test_pathlib_read_blocked(self):
        """Test that pathlib file operations are blocked."""
        code = """
from pathlib import Path
p = Path("/etc/passwd")
p.read_text()
"""
        result = await execute_programmatic_code(
            code=code,
            tools={},
            tool_context=MagicMock(spec=ToolUseContext),
        )
        assert result.success is False
        assert "PermissionError" in result.error

    async def test_pathlib_safe_operations_work(self):
        """Test that pathlib safe operations work."""
        code = """
from pathlib import Path
p = Path("/home/user/file.txt")
ctx.set_result({
    "name": p.name,
    "suffix": p.suffix,
    "parent": str(p.parent),
})
"""
        result = await execute_programmatic_code(
            code=code,
            tools={},
            tool_context=MagicMock(spec=ToolUseContext),
        )
        assert result.success is True
        assert result.result["name"] == "file.txt"
        assert result.result["suffix"] == ".txt"
        assert result.result["parent"] == "/home/user"

    async def test_os_path_real_functions(self):
        """Test that os.path uses real implementations for safe functions."""
        code = """
import os

# Test various os.path functions
results = {
    "join": os.path.join("/home", "user", "file.txt"),
    "split": os.path.split("/home/user/file.txt"),
    "splitext": os.path.splitext("/home/user/file.txt"),
    "basename": os.path.basename("/home/user/file.txt"),
    "dirname": os.path.dirname("/home/user/file.txt"),
    "normpath": os.path.normpath("/home//user/../user/./file.txt"),
    "isabs_true": os.path.isabs("/home/user"),
    "isabs_false": os.path.isabs("relative/path"),
}
ctx.set_result(results)
"""
        result = await execute_programmatic_code(
            code=code,
            tools={},
            tool_context=MagicMock(spec=ToolUseContext),
        )
        assert result.success is True
        assert result.result["join"] == "/home/user/file.txt"
        assert result.result["split"] == ("/home/user", "file.txt")
        assert result.result["splitext"] == ("/home/user/file", ".txt")
        assert result.result["basename"] == "file.txt"
        assert result.result["dirname"] == "/home/user"
        assert result.result["normpath"] == "/home/user/file.txt"
        assert result.result["isabs_true"] is True
        assert result.result["isabs_false"] is False

    async def test_os_path_filesystem_blocked(self):
        """Test that os.path filesystem access functions are blocked or return False."""
        code = """
import os

# These should return False (no filesystem access)
results = {
    "exists": os.path.exists("/some/path"),
    "isfile": os.path.isfile("/some/path"),
    "isdir": os.path.isdir("/some/path"),
    "islink": os.path.islink("/some/path"),
}
ctx.set_result(results)
"""
        result = await execute_programmatic_code(
            code=code,
            tools={},
            tool_context=MagicMock(spec=ToolUseContext),
        )
        assert result.success is True
        assert result.result["exists"] is False
        assert result.result["isfile"] is False
        assert result.result["isdir"] is False
        assert result.result["islink"] is False

    async def test_os_fspath_works(self):
        """Test that os.fspath works."""
        code = """
import os
from pathlib import Path

# Test fspath with string and Path
results = {
    "string": os.fspath("/home/user"),
    "path": os.fspath(Path("/home/user")),
}
ctx.set_result(results)
"""
        result = await execute_programmatic_code(
            code=code,
            tools={},
            tool_context=MagicMock(spec=ToolUseContext),
        )
        assert result.success is True
        assert result.result["string"] == "/home/user"
        assert result.result["path"] == "/home/user"
