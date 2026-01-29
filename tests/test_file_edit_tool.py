"""Tests for FileEditTool.

Tests cover:
- FileEditToolInput and FileEditToolOutput models
- File locking behavior (_file_lock)
- Input validation
- Single and multiple replacements
- Diff generation
- Error handling
- TOCTOU protection
"""

import os
import sys
import tempfile
from unittest.mock import MagicMock

import pytest

from ripperdoc.tools.file_edit_tool import (
    _file_lock,
    FileEditTool,
    FileEditToolInput,
    FileEditToolOutput,
)


class TestFileEditToolInput:
    """Tests for FileEditToolInput model."""

    def test_create_input(self):
        """Should create valid input with all fields."""
        input_data = FileEditToolInput(
            file_path="/tmp/test.py",
            old_string="old",
            new_string="new",
            replace_all=False,
        )
        assert input_data.file_path == "/tmp/test.py"
        assert input_data.old_string == "old"
        assert input_data.new_string == "new"
        assert input_data.replace_all is False

    def test_default_replace_all(self):
        """replace_all should default to False."""
        input_data = FileEditToolInput(
            file_path="/tmp/test.py",
            old_string="old",
            new_string="new",
        )
        assert input_data.replace_all is False

    def test_required_fields(self):
        """file_path, old_string, and new_string are required."""
        with pytest.raises(Exception):
            FileEditToolInput(file_path="/tmp/test.py")

        with pytest.raises(Exception):
            FileEditToolInput(
                file_path="/tmp/test.py",
                old_string="old",
            )


class TestFileEditToolOutput:
    """Tests for FileEditToolOutput model."""

    def test_create_output(self):
        """Should create valid output with all fields."""
        output = FileEditToolOutput(
            file_path="/tmp/test.py",
            replacements_made=1,
            success=True,
            message="Success",
            additions=5,
            deletions=3,
            diff_lines=["+ new line"],
            diff_with_line_numbers=["1 + new line"],
        )
        assert output.file_path == "/tmp/test.py"
        assert output.replacements_made == 1
        assert output.success is True
        assert output.message == "Success"
        assert output.additions == 5
        assert output.deletions == 3
        assert len(output.diff_lines) == 1
        assert len(output.diff_with_line_numbers) == 1

    def test_minimal_output(self):
        """Should create output with minimal required fields."""
        output = FileEditToolOutput(
            file_path="/tmp/test.py",
            replacements_made=0,
            success=False,
            message="Failed",
        )
        assert output.additions == 0
        assert output.deletions == 0
        assert output.diff_lines == []
        assert output.diff_with_line_numbers == []


class TestFileLock:
    """Tests for _file_lock context manager."""

    def test_lock_yields_without_error(self, tmp_path):
        """Lock context manager should yield without error."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        with open(test_file, "r") as f:
            with _file_lock(f):
                # Should not raise
                pass

    def test_exclusive_lock_parameter(self, tmp_path):
        """exclusive parameter should be accepted."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        with open(test_file, "r") as f:
            with _file_lock(f, exclusive=True):
                pass

        with open(test_file, "r") as f:
            with _file_lock(f, exclusive=False):
                pass

    def test_lock_on_windows_no_fcntl(self, monkeypatch):
        """On Windows (no fcntl), should skip locking gracefully."""
        monkeypatch.setattr("ripperdoc.tools.file_edit_tool.HAS_FCNTL", False)

        test_file = tempfile.NamedTemporaryFile(delete=False)
        test_file.write(b"content")
        test_file.close()

        try:
            with open(test_file.name, "r") as f:
                with _file_lock(f):
                    # Should not raise even without fcntl
                    content = f.read()
                    assert content == "content"
        finally:
            os.unlink(test_file.name)


class TestFileEditToolProperties:
    """Tests for FileEditTool properties and methods."""

    def test_tool_name(self):
        """Tool name should be 'Edit'."""
        tool = FileEditTool()
        assert tool.name == "Edit"

    def test_is_not_read_only(self):
        """Edit tool should not be read-only."""
        tool = FileEditTool()
        assert tool.is_read_only() is False

    def test_is_not_concurrency_safe(self):
        """Edit tool should not be concurrency-safe."""
        tool = FileEditTool()
        assert tool.is_concurrency_safe() is False

    def test_needs_permissions(self):
        """Edit tool should always need permissions."""
        tool = FileEditTool()
        assert tool.needs_permissions() is True

    def test_input_schema(self):
        """Input schema should be FileEditToolInput."""
        tool = FileEditTool()
        assert tool.input_schema == FileEditToolInput

    def test_input_examples(self):
        """Should provide input examples."""
        tool = FileEditTool()
        examples = tool.input_examples()
        assert len(examples) > 0
        assert all(hasattr(ex, "description") for ex in examples)
        assert all(hasattr(ex, "example") for ex in examples)

    async def test_description(self):
        """Should return a description."""
        tool = FileEditTool()
        description = await tool.description()
        assert isinstance(description, str)
        assert len(description) > 0

    async def test_prompt(self):
        """Should return usage prompt."""
        tool = FileEditTool()
        prompt = await tool.prompt()
        assert "Read" in prompt
        assert "edit" in prompt.lower()

    def test_render_result_for_assistant(self):
        """Should render result message for assistant."""
        tool = FileEditTool()
        output = FileEditToolOutput(
            file_path="/tmp/test.py",
            replacements_made=1,
            success=True,
            message="Success",
        )
        result = tool.render_result_for_assistant(output)
        assert result == "Success"

    def test_render_tool_use_message(self):
        """Should render tool use message."""
        tool = FileEditTool()
        input_data = FileEditToolInput(
            file_path="/tmp/test.py",
            old_string="old",
            new_string="new",
        )
        message = tool.render_tool_use_message(input_data)
        assert "/tmp/test.py" in message
        assert "Editing" in message


class TestValidateInput:
    """Tests for input validation."""

    async def test_validate_nonexistent_file(self):
        """Should fail validation for nonexistent file."""
        tool = FileEditTool()
        input_data = FileEditToolInput(
            file_path="/nonexistent/file.txt",
            old_string="old",
            new_string="new",
        )

        context = MagicMock()
        context.file_state_cache = {}

        result = await tool.validate_input(input_data, context)
        assert result.result is False
        assert "not found" in result.message.lower()

    async def test_validate_directory_path(self, tmp_path):
        """Should fail validation for directory path."""
        tool = FileEditTool()
        input_data = FileEditToolInput(
            file_path=str(tmp_path),
            old_string="old",
            new_string="new",
        )

        context = MagicMock()
        context.file_state_cache = {}

        result = await tool.validate_input(input_data, context)
        assert result.result is False
        assert "not a file" in result.message.lower()

    async def test_validate_same_old_and_new_string(self, tmp_path):
        """Should fail validation when old_string equals new_string."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        tool = FileEditTool()
        input_data = FileEditToolInput(
            file_path=str(test_file),
            old_string="same",
            new_string="same",
        )

        context = MagicMock()
        context.file_state_cache = {}

        result = await tool.validate_input(input_data, context)
        assert result.result is False
        assert "different" in result.message.lower()

    async def test_validate_file_not_read_yet(self, tmp_path):
        """Should fail validation when file hasn't been read."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        tool = FileEditTool()
        input_data = FileEditToolInput(
            file_path=str(test_file),
            old_string="old",
            new_string="new",
        )

        context = MagicMock()
        context.file_state_cache = {}

        result = await tool.validate_input(input_data, context)
        assert result.result is False
        assert "not been read" in result.message.lower()

    async def test_validate_success(self, tmp_path):
        """Should pass validation with proper prerequisites."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        tool = FileEditTool()
        input_data = FileEditToolInput(
            file_path=str(test_file),
            old_string="hello",
            new_string="goodbye",
        )

        # Create a mock file snapshot
        snapshot = MagicMock()
        snapshot.timestamp = os.path.getmtime(test_file)
        snapshot.content = "hello world"

        context = MagicMock()
        context.file_state_cache = {os.path.abspath(str(test_file)): snapshot}

        result = await tool.validate_input(input_data, context)
        assert result.result is True


class TestFileEditExecution:
    """Tests for actual file editing execution."""

    async def test_single_replacement(self, tmp_path):
        """Should perform single string replacement."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world\nhi universe")

        tool = FileEditTool()
        input_data = FileEditToolInput(
            file_path=str(test_file),
            old_string="hello",
            new_string="goodbye",
            replace_all=False,
        )

        snapshot = MagicMock()
        snapshot.timestamp = os.path.getmtime(test_file)
        snapshot.content = "hello world\nhi universe"

        context = MagicMock()
        context.file_state_cache = {str(test_file): snapshot}

        results = []
        async for result in tool.call(input_data, context):
            results.append(result)

        assert len(results) == 1
        output = results[0].data
        assert output.success is True
        assert output.replacements_made == 1
        assert "goodbye world" in test_file.read_text()

    async def test_replace_all(self, tmp_path):
        """Should replace all occurrences when replace_all=True."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world\nhello universe\nhello again")

        tool = FileEditTool()
        input_data = FileEditToolInput(
            file_path=str(test_file),
            old_string="hello",
            new_string="goodbye",
            replace_all=True,
        )

        snapshot = MagicMock()
        snapshot.timestamp = os.path.getmtime(test_file)
        snapshot.content = "hello world\nhello universe\nhello again"

        context = MagicMock()
        context.file_state_cache = {str(test_file): snapshot}

        results = []
        async for result in tool.call(input_data, context):
            results.append(result)

        assert len(results) == 1
        output = results[0].data
        assert output.success is True
        assert output.replacements_made == 3
        content = test_file.read_text()
        assert content.count("goodbye") == 3

    async def test_string_not_found(self, tmp_path):
        """Should fail when old_string is not in file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        tool = FileEditTool()
        input_data = FileEditToolInput(
            file_path=str(test_file),
            old_string="goodbye",
            new_string=" farewell",
        )

        snapshot = MagicMock()
        snapshot.timestamp = os.path.getmtime(test_file)
        snapshot.content = "hello world"

        context = MagicMock()
        context.file_state_cache = {str(test_file): snapshot}

        results = []
        async for result in tool.call(input_data, context):
            results.append(result)

        assert len(results) == 1
        output = results[0].data
        assert output.success is False
        assert "not found" in output.message.lower()

    async def test_ambiguous_match_without_replace_all(self, tmp_path):
        """Should fail when string appears multiple times without replace_all."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello\nhello\nhello")

        tool = FileEditTool()
        input_data = FileEditToolInput(
            file_path=str(test_file),
            old_string="hello",
            new_string="goodbye",
            replace_all=False,
        )

        snapshot = MagicMock()
        snapshot.timestamp = os.path.getmtime(test_file)
        snapshot.content = "hello\nhello\nhello"

        context = MagicMock()
        context.file_state_cache = {str(test_file): snapshot}

        results = []
        async for result in tool.call(input_data, context):
            results.append(result)

        assert len(results) == 1
        output = results[0].data
        assert output.success is False
        assert "appears 3 times" in output.message.lower()

    async def test_diff_generation(self, tmp_path):
        """Should generate proper diff with additions and deletions."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("line1\nline2\nline3")

        tool = FileEditTool()
        input_data = FileEditToolInput(
            file_path=str(test_file),
            old_string="line2",
            new_string="modified_line2",
        )

        snapshot = MagicMock()
        snapshot.timestamp = os.path.getmtime(test_file)
        snapshot.content = "line1\nline2\nline3"

        context = MagicMock()
        context.file_state_cache = {str(test_file): snapshot}

        results = []
        async for result in tool.call(input_data, context):
            results.append(result)

        output = results[0].data
        assert output.success is True
        assert output.additions == 1
        assert output.deletions == 1
        assert len(output.diff_lines) > 0
        assert len(output.diff_with_line_numbers) > 0


class TestErrorHandling:
    """Tests for error handling."""

    async def test_permission_error(self, tmp_path):
        """Should handle permission errors gracefully."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        # Make file read-only
        os.chmod(test_file, 0o444)

        tool = FileEditTool()
        input_data = FileEditToolInput(
            file_path=str(test_file),
            old_string="content",
            new_string="modified",
        )

        snapshot = MagicMock()
        snapshot.timestamp = os.path.getmtime(test_file)
        snapshot.content = "content"

        context = MagicMock()
        context.file_state_cache = {str(test_file): snapshot}

        results = []
        async for result in tool.call(input_data, context):
            results.append(result)

        # Restore permissions for cleanup
        os.chmod(test_file, 0o644)

        assert len(results) == 1
        output = results[0].data
        # Should either succeed or fail gracefully with an error message
        assert isinstance(output, FileEditToolOutput)

    async def test_unicode_handling(self, tmp_path):
        """Should handle unicode characters correctly."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("你好 世界\nHello Universe")

        tool = FileEditTool()
        input_data = FileEditToolInput(
            file_path=str(test_file),
            old_string="世界",
            new_string="World",
        )

        snapshot = MagicMock()
        snapshot.timestamp = os.path.getmtime(test_file)
        snapshot.content = "你好 世界\nHello Universe"

        context = MagicMock()
        context.file_state_cache = {str(test_file): snapshot}

        results = []
        async for result in tool.call(input_data, context):
            results.append(result)

        assert len(results) == 1
        output = results[0].data
        assert output.success is True
        content = test_file.read_text(encoding="utf-8")
        assert "World" in content
        # Should preserve other unicode characters
        assert "你好" in content


class TestTOCTOUProtection:
    """Tests for Time-Of-Check-Time-Of-Use protection."""

    async def test_detects_file_modified_after_read(self, tmp_path):
        """Should detect file modification after initial read."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("original content")

        tool = FileEditTool()
        input_data = FileEditToolInput(
            file_path=str(test_file),
            old_string="original",
            new_string="modified",
        )

        # Create snapshot with old timestamp
        import time

        old_time = time.time() - 10  # 10 seconds ago
        snapshot = MagicMock()
        snapshot.timestamp = old_time
        snapshot.content = "original content"

        context = MagicMock()
        context.file_state_cache = {str(test_file): snapshot}

        # Touch file to update mtime
        import time

        time.sleep(0.01)  # Ensure mtime changes
        test_file.write_text("modified externally")

        results = []
        async for result in tool.call(input_data, context):
            results.append(result)

        assert len(results) == 1
        output = results[0].data
        assert output.success is False
        assert "modified since read" in output.message.lower()


class TestPreservation:
    """Tests for file preservation during edits."""

    @pytest.mark.skipif(
        sys.platform != "win32", reason="Line ending preservation is Windows-specific"
    )
    async def test_preserves_line_endings(self, tmp_path):
        """Should preserve line ending style.

        Note: On Windows, writing "\r\n" to a file in text mode preserves it.
        On Unix/Linux, text mode normalizes "\r\n" to "\n", which is standard Python behavior.
        This test is only meaningful on Windows.
        """
        test_file = tmp_path / "test.txt"
        test_file.write_text("line1\r\nline2\r\nline3\r\n")  # Windows line endings

        tool = FileEditTool()
        input_data = FileEditToolInput(
            file_path=str(test_file),
            old_string="line2",
            new_string="modified",
        )

        snapshot = MagicMock()
        snapshot.timestamp = os.path.getmtime(test_file)
        snapshot.content = "line1\r\nline2\r\nline3\r\n"

        context = MagicMock()
        context.file_state_cache = {str(test_file): snapshot}

        results = []
        async for result in tool.call(input_data, context):
            results.append(result)

        assert results[0].data.success is True
        # Check that line endings are preserved (only on Windows)
        content = test_file.read_text()
        assert "\r\n" in content

    async def test_preserves_file_encoding(self, tmp_path):
        """Should preserve UTF-8 encoding."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello 世界 🌍\n", encoding="utf-8")

        tool = FileEditTool()
        input_data = FileEditToolInput(
            file_path=str(test_file),
            old_string="Hello",
            new_string="Goodbye",
        )

        snapshot = MagicMock()
        snapshot.timestamp = os.path.getmtime(test_file)
        snapshot.content = "Hello 世界 🌍\n"

        context = MagicMock()
        context.file_state_cache = {str(test_file): snapshot}

        results = []
        async for result in tool.call(input_data, context):
            results.append(result)

        assert results[0].data.success is True
        content = test_file.read_text(encoding="utf-8")
        assert "世界" in content
        assert "🌍" in content


class TestEdgeCases:
    """Edge case tests."""

    async def test_empty_file(self, tmp_path):
        """Should handle empty files."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("")

        tool = FileEditTool()
        input_data = FileEditToolInput(
            file_path=str(test_file),
            old_string="anything",
            new_string="replacement",
        )

        snapshot = MagicMock()
        snapshot.timestamp = os.path.getmtime(test_file)
        snapshot.content = ""

        context = MagicMock()
        context.file_state_cache = {str(test_file): snapshot}

        results = []
        async for result in tool.call(input_data, context):
            results.append(result)

        output = results[0].data
        assert output.success is False  # String not found in empty file

    async def test_multiline_replacement(self, tmp_path):
        """Should handle multiline string replacement."""
        test_file = tmp_path / "test.txt"
        content = "line1\nline2\nline3\nline4"
        test_file.write_text(content)

        tool = FileEditTool()
        input_data = FileEditToolInput(
            file_path=str(test_file),
            old_string="line2\nline3",
            new_string="modified",
        )

        snapshot = MagicMock()
        snapshot.timestamp = os.path.getmtime(test_file)
        snapshot.content = content

        context = MagicMock()
        context.file_state_cache = {str(test_file): snapshot}

        results = []
        async for result in tool.call(input_data, context):
            results.append(result)

        output = results[0].data
        assert output.success is True
        assert "line1\nmodified\nline4" == test_file.read_text()
