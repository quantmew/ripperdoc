"""Tests for read-only command validation."""

from ripperdoc.tools.bash.read_only_validation import (
    is_command_read_only,
    is_command_safe_via_flag_parsing,
    contains_unquoted_expansion,
)


class TestIsCommandReadOnly:
    def test_ls(self):
        assert is_command_read_only("ls -la")

    def test_pwd(self):
        assert is_command_read_only("pwd")

    def test_whoami(self):
        assert is_command_read_only("whoami")

    def test_git_status(self):
        assert is_command_read_only("git status")

    def test_git_log(self):
        assert is_command_read_only("git log --oneline -5")

    def test_git_diff(self):
        assert is_command_read_only("git diff --cached")

    def test_rm(self):
        assert not is_command_read_only("rm -rf /")

    def test_git_push(self):
        assert not is_command_read_only("git push")

    def test_git_commit(self):
        assert not is_command_read_only("git commit -m 'msg'")


class TestCommandSafeViaFlagParsing:
    def test_git_status(self):
        assert is_command_safe_via_flag_parsing("git status")

    def test_git_log(self):
        assert is_command_safe_via_flag_parsing("git log --oneline")

    def test_not_in_allowlist(self):
        assert not is_command_safe_via_flag_parsing("git push")


class TestContainsUnquotedExpansion:
    def test_no_expansion(self):
        assert not contains_unquoted_expansion("ls -la")

    def test_glob(self):
        assert contains_unquoted_expansion("ls *.py")

    def test_dollar(self):
        assert contains_unquoted_expansion("echo $HOME")
