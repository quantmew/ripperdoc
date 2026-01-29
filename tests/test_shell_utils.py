"""Tests for shell utilities.

Tests cover:
- find_suitable_shell: Shell detection and selection
- build_shell_command: Command argument building
- Cross-platform shell handling
"""

from unittest.mock import patch

import pytest

from ripperdoc.utils.shell_utils import (
    find_suitable_shell,
    build_shell_command,
    _is_executable,
    _dedupe_preserve_order,
    _find_git_bash_windows,
    _windows_cmd_path,
)


class TestIsExecutable:
    """Tests for _is_executable helper function."""

    def test_existing_executable_file(self, tmp_path):
        """An existing executable file should return True."""
        test_file = tmp_path / "test.sh"
        test_file.write_text("#!/bin/bash\necho test")
        test_file.chmod(0o755)
        assert _is_executable(str(test_file)) is True

    def test_existing_non_executable_file(self, tmp_path):
        """A non-executable file should return False."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        test_file.chmod(0o644)
        assert _is_executable(str(test_file)) is False

    def test_nonexistent_file(self):
        """A nonexistent file should return False."""
        assert _is_executable("/nonexistent/path/to/file") is False

    def test_empty_string_returns_false(self):
        """Empty string should return False."""
        assert _is_executable("") is False

    def test_none_returns_false(self):
        """None should return False."""
        assert _is_executable(None) is False


class TestDedupePreserveOrder:
    """Tests for _dedupe_preserve_order helper function."""

    def test_empty_list(self):
        """Empty list should return empty list."""
        assert _dedupe_preserve_order([]) == []

    def test_no_duplicates(self):
        """List without duplicates should be unchanged."""
        items = ["a", "b", "c"]
        assert _dedupe_preserve_order(items) == items

    def test_with_duplicates(self):
        """Duplicates should be removed, preserving first occurrence."""
        items = ["a", "b", "a", "c", "b"]
        result = _dedupe_preserve_order(items)
        assert result == ["a", "b", "c"]

    def test_preserves_order(self):
        """Order of first occurrences should be preserved."""
        items = ["z", "a", "z", "b", "a", "c"]
        result = _dedupe_preserve_order(items)
        assert result == ["z", "a", "b", "c"]

    def test_with_empty_strings(self):
        """Empty strings should be filtered out by the logic.

        The function uses `if item and item not in seen:` which means
        empty strings (falsy) are filtered out.
        """
        items = ["a", "", "b", ""]
        result = _dedupe_preserve_order(items)
        # Empty strings are filtered because they are falsy
        assert result == ["a", "b"]

    def test_with_none_values(self):
        """None values should be filtered out by the logic."""
        items = ["a", None, "b", None]
        result = _dedupe_preserve_order(items)
        # None is truthy for the check, but the condition `if item and item not in seen`
        # will filter out None since None is falsy
        assert result == ["a", "b"]


class TestBuildShellCommand:
    """Tests for build_shell_command function."""

    def test_bash_command(self):
        """Bash command should use -lc flags."""
        result = build_shell_command("/bin/bash", "echo hello")
        assert result == ["/bin/bash", "-lc", "echo hello"]

    def test_zsh_command(self):
        """Zsh command should use -lc flags."""
        result = build_shell_command("/bin/zsh", "ls -la")
        assert result == ["/bin/zsh", "-lc", "ls -la"]

    def test_git_bash_command(self):
        """Git Bash on Windows should use -lc flags."""
        result = build_shell_command(r"C:\Program Files\Git\bin\bash.exe", "pwd")
        assert result[0].endswith("bash.exe")
        assert result[1] == "-lc"
        assert result[2] == "pwd"

    def test_cmd_exe_command(self):
        """cmd.exe command should use /d /s /c flags."""
        result = build_shell_command(r"C:\Windows\System32\cmd.exe", "dir")
        assert result[0].endswith("cmd.exe")
        assert "/d" in result
        assert "/s" in result
        assert "/c" in result
        assert "dir" in result

    def test_cmd_simple(self):
        """Simple 'cmd' should be recognized as cmd.exe."""
        result = build_shell_command("cmd", "echo test")
        assert result[0] == "cmd"
        assert "/d" in result
        assert "/s" in result
        assert "/c" in result
        assert "echo test" in result

    def test_cmd_with_backslash(self):
        """cmd.exe path with backslash should use Windows format."""
        result = build_shell_command(r"C:\Windows\System32\cmd.exe", "dir")
        assert result[0].endswith("cmd.exe")
        assert "/d" in result
        assert "/s" in result
        assert "/c" in result
        assert "dir" in result

    def test_lowercase_cmd(self):
        """cmd.exe with lowercase should use Windows format."""
        result = build_shell_command("cmd.exe", "echo test")
        assert result[0].endswith("cmd.exe")
        assert "/d" in result
        assert "/s" in result
        assert "/c" in result
        assert "echo test" in result

    def test_command_with_spaces(self):
        """Command with spaces should be preserved as single argument."""
        result = build_shell_command("/bin/bash", "echo 'hello world'")
        assert result == ["/bin/bash", "-lc", "echo 'hello world'"]

    def test_complex_command(self):
        """Complex commands should be passed through."""
        result = build_shell_command("/bin/bash", "ls -la | grep test")
        assert result == ["/bin/bash", "-lc", "ls -la | grep test"]


class TestFindGitBashWindows:
    """Tests for _find_git_bash_windows function."""

    def test_environment_override(self, monkeypatch, tmp_path):
        """GIT_BASH_PATH environment variable should take precedence."""
        bash_exe = tmp_path / "bash.exe"
        bash_exe.write_text("#!/bin/bash")
        bash_exe.chmod(0o755)

        monkeypatch.setenv("GIT_BASH_PATH", str(bash_exe))
        with patch("shutil.which", return_value=None):
            result = _find_git_bash_windows()
            assert result == str(bash_exe)

    def test_gitbash_environment_override(self, monkeypatch, tmp_path):
        """GITBASH environment variable should be checked."""
        bash_exe = tmp_path / "bash.exe"
        bash_exe.write_text("#!/bin/bash")
        bash_exe.chmod(0o755)

        monkeypatch.setenv("GITBASH", str(bash_exe))
        with patch("shutil.which", return_value=None):
            result = _find_git_bash_windows()
            assert result == str(bash_exe)

    def test_path_with_git_in_name(self, monkeypatch):
        """Bash in path with 'git' should be found."""
        git_bash = r"C:\Program Files\Git\bin\bash.exe"

        def mock_which(cmd):
            if cmd == "bash":
                return git_bash
            return None

        monkeypatch.setattr("shutil.which", mock_which)
        with patch("ripperdoc.utils.shell_utils._is_executable", return_value=True):
            result = _find_git_bash_windows()
            assert result == git_bash

    def test_common_locations_checked(self, monkeypatch):
        """Common Git Bash installation paths should be checked."""
        checked_paths = []

        def mock_is_executable(path):
            checked_paths.append(path)
            return False

        def mock_which(cmd):
            return None

        monkeypatch.setattr("ripperdoc.utils.shell_utils._is_executable", mock_is_executable)
        monkeypatch.setattr("shutil.which", mock_which)

        _find_git_bash_windows()

        # Check that common paths were probed
        common_paths = [
            r"C:\Program Files\Git\bin\bash.exe",
            r"C:\Program Files\Git\usr\bin\bash.exe",
            r"C:\Program Files (x86)\Git\bin\bash.exe",
        ]
        for path in common_paths:
            assert path in checked_paths

    def test_returns_none_when_not_found(self, monkeypatch):
        """Should return None when Git Bash is not found."""

        def mock_which(cmd):
            return None

        def mock_is_executable(path):
            return False

        monkeypatch.setattr("shutil.which", mock_which)
        monkeypatch.setattr("ripperdoc.utils.shell_utils._is_executable", mock_is_executable)
        result = _find_git_bash_windows()
        assert result is None


class TestWindowsCmdPath:
    """Tests for _windows_cmd_path function."""

    def test_comspec_environment_variable(self, monkeypatch, tmp_path):
        """ComSpec environment variable should be used."""
        cmd_exe = tmp_path / "cmd.exe"
        cmd_exe.write_text("dummy")

        monkeypatch.setenv("ComSpec", str(cmd_exe))
        with patch("ripperdoc.utils.shell_utils._is_executable", return_value=True):
            result = _windows_cmd_path()
            assert str(cmd_exe) in result

    def test_shutil_which_fallback(self, monkeypatch):
        """Should fall back to shutil.which if ComSpec not set."""
        monkeypatch.delenv("ComSpec", raising=False)

        def mock_which(cmd):
            if cmd == "cmd.exe" or cmd == "cmd":
                return r"C:\Windows\System32\cmd.exe"
            return None

        def mock_is_executable(path):
            # Return True for the cmd.exe path, False for empty string
            return path and "cmd.exe" in path

        monkeypatch.setattr("shutil.which", mock_which)
        monkeypatch.setattr("ripperdoc.utils.shell_utils._is_executable", mock_is_executable)
        result = _windows_cmd_path()
        assert result == r"C:\Windows\System32\cmd.exe"

    def test_system32_fallback(self, monkeypatch):
        """Should check System32 as final fallback."""
        monkeypatch.delenv("ComSpec", raising=False)

        def mock_which(cmd):
            return None

        def mock_get_env(key, default=None):
            if key == "SystemRoot":
                return r"C:\Windows"
            return default

        def mock_is_executable(path):
            return "cmd.exe" in path

        monkeypatch.setattr("shutil.which", mock_which)
        monkeypatch.setattr("os.environ.get", mock_get_env)
        monkeypatch.setattr("ripperdoc.utils.shell_utils._is_executable", mock_is_executable)

        result = _windows_cmd_path()
        # The result should be the System32 cmd.exe path
        assert result is not None and "cmd.exe" in result

    def test_returns_none_when_not_found(self, monkeypatch):
        """Should return None when cmd.exe cannot be found."""
        monkeypatch.delenv("ComSpec", raising=False)

        with patch("shutil.which", return_value=None):
            with patch("os.path.exists", return_value=False):
                result = _windows_cmd_path()
                assert result is None


class TestFindSuitableShell:
    """Tests for find_suitable_shell function."""

    def test_environment_override(self, monkeypatch, tmp_path):
        """RIPPERDOC_SHELL should override shell detection."""
        custom_shell = tmp_path / "myshell"
        custom_shell.write_text("#!/bin/sh")
        custom_shell.chmod(0o755)

        monkeypatch.setenv("RIPPERDOC_SHELL", str(custom_shell))
        result = find_suitable_shell()
        assert str(custom_shell) in result or result == str(custom_shell)

    def test_ripperdoc_shell_path_override(self, monkeypatch, tmp_path):
        """RIPPERDOC_SHELL_PATH should also work."""
        custom_shell = tmp_path / "custom"
        custom_shell.write_text("#!/bin/bash")
        custom_shell.chmod(0o755)

        monkeypatch.setenv("RIPPERDOC_SHELL_PATH", str(custom_shell))
        result = find_suitable_shell()
        assert str(custom_shell) in result or result == str(custom_shell)

    def test_non_executable_override_is_ignored(self, monkeypatch, tmp_path):
        """Non-executable override should be ignored."""
        non_executable = tmp_path / "fake_shell"
        non_executable.write_text("not executable")
        non_executable.chmod(0o644)

        monkeypatch.setenv("RIPPERDOC_SHELL", str(non_executable))
        # Should not raise, should fall back to default detection
        # This will use the actual shell on the system
        result = find_suitable_shell()
        assert result is not None

    def test_shell_env_variable_on_unix(self, monkeypatch):
        """SHELL environment variable should be respected on Unix."""
        monkeypatch.setenv("SHELL", "/bin/zsh")
        with patch("os.name", "posix"):
            with patch("shutil.which") as mock_which:
                mock_which.return_value = None  # No other shells
                with patch("os.access", return_value=True):
                    with patch("os.path.isfile", return_value=True):
                        result = find_suitable_shell()
                        # Should at least try to use SHELL
                        assert result is not None

    def test_raises_runtime_error_on_unix_when_no_shell_found(self, monkeypatch):
        """Should raise RuntimeError when no shell can be found on Unix."""
        monkeypatch.setenv("RIPPERDOC_SHELL", "")
        monkeypatch.delenv("RIPPERDOC_SHELL", raising=False)
        monkeypatch.delenv("SHELL", raising=False)

        def mock_which(cmd):
            return None

        def mock_is_executable(path):
            return False  # No shell is executable

        monkeypatch.setattr("os.name", "posix")
        monkeypatch.setattr("shutil.which", mock_which)
        monkeypatch.setattr("ripperdoc.utils.shell_utils._is_executable", mock_is_executable)

        with pytest.raises(RuntimeError) as exc_info:
            find_suitable_shell()
        assert "No suitable shell found" in str(exc_info.value)


class TestCrossPlatformBehavior:
    """Tests for cross-platform shell handling."""

    def test_shell_command_format_differences(self):
        """Shell commands should use different formats for bash vs cmd."""
        bash_cmd = build_shell_command("/bin/bash", "echo test")
        cmd_cmd = build_shell_command(r"C:\Windows\System32\cmd.exe", "echo test")

        # Bash uses -lc, cmd.exe uses /d /s /c
        assert "-lc" in bash_cmd
        assert "/c" in cmd_cmd
        assert "-lc" not in cmd_cmd

    def test_find_suitable_shell_returns_string(self):
        """find_suitable_shell should always return a string path."""
        result = find_suitable_shell()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_build_shell_command_returns_list(self):
        """build_shell_command should return a list of arguments."""
        result = build_shell_command("/bin/bash", "ls")
        assert isinstance(result, list)
        assert len(result) >= 3


class TestEdgeCases:
    """Edge case tests for shell utilities."""

    def test_empty_command(self):
        """Empty command should still produce valid argv."""
        result = build_shell_command("/bin/bash", "")
        assert result[0].endswith("bash")
        assert result[1] == "-lc"

    def test_command_with_newlines(self):
        """Commands with newlines should be preserved."""
        result = build_shell_command("/bin/bash", "echo line1\necho line2")
        assert "\n" in result[2]

    def test_command_with_special_characters(self):
        """Commands with special shell characters should be preserved."""
        result = build_shell_command("/bin/bash", "echo 'test && rm -rf /'")
        assert "&&" in result[2]

    def test_shell_path_with_spaces(self):
        """Shell paths with spaces should be handled correctly."""
        result = build_shell_command(r"C:\Program Files\Git\bin\bash.exe", "pwd")
        assert result[0] == r"C:\Program Files\Git\bin\bash.exe"
