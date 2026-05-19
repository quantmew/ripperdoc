"""Tests for bash security detection."""

from ripperdoc.security import bash_command_is_safe, bash_command_is_safe_async


class TestBashCommandIsSafe:
    def test_empty_command(self):
        result = bash_command_is_safe("")
        assert result.behavior in ("allow", "passthrough")

    def test_simple_ls(self):
        result = bash_command_is_safe("ls -la")
        assert result.behavior == "passthrough"

    def test_incomplete_command(self):
        result = bash_command_is_safe("-rf /")
        assert result.behavior == "ask"

    def test_dangerous_variable(self):
        result = bash_command_is_safe("IFS=abc command")
        assert result.behavior == "ask"

    def test_brace_expansion(self):
        result = bash_command_is_safe("echo {a,b}")
        assert result.behavior == "ask"

    def test_zsh_dangerous(self):
        result = bash_command_is_safe("zmodload zsh/system")
        assert result.behavior == "ask"

    def test_unbalanced_quotes(self):
        result = bash_command_is_safe("echo 'hello")
        assert result.behavior == "ask"

    def test_safe_echo(self):
        result = bash_command_is_safe("echo hello world")
        assert result.behavior == "passthrough"

    def test_git_status(self):
        result = bash_command_is_safe("git status")
        assert result.behavior == "passthrough"

    def test_async(self):
        result = bash_command_is_safe_async("ls")
        assert result.behavior == "passthrough"
