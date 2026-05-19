"""Tests for the tree-sitter AST parsing module."""

from ripperdoc.utils.bash import (
    parse_for_security_from_ast,
    split_command_with_operators,
    extract_output_redirections,
    try_parse_shell_command,
)


class TestCommandSplitting:
    def test_simple_command(self):
        parts = split_command_with_operators("ls -la")
        assert parts == ["ls -la"]

    def test_piped_command(self):
        parts = split_command_with_operators("ls -la | grep foo")
        assert "|" in parts

    def test_compound_command(self):
        parts = split_command_with_operators("echo a && echo b")
        assert "&&" in parts


class TestOutputRedirection:
    def test_basic_redirect(self):
        result = extract_output_redirections("echo hi > /dev/null")
        assert len(result.redirections) == 1
        assert result.redirections[0].operator == ">"

    def test_append_redirect(self):
        result = extract_output_redirections("echo hi >> log.txt")
        assert len(result.redirections) == 1
        assert result.redirections[0].target == "log.txt"


class TestASTParsing:
    def test_simple_command(self):
        result = parse_for_security_from_ast("ls -la")
        assert result.get("kind") == "simple"
        cmds = result.get("commands", [])
        assert len(cmds) == 1
        assert cmds[0].argv == ["ls", "-la"]

    def test_env_vars(self):
        result = parse_for_security_from_ast("FOO=bar echo hi")
        assert result.get("kind") == "simple"
        cmds = result.get("commands", [])
        assert len(cmds) == 1
        assert len(cmds[0].env_vars) > 0

    def test_redirect(self):
        result = parse_for_security_from_ast("cat > /dev/null")
        assert result.get("kind") == "simple"

    def test_piped(self):
        result = parse_for_security_from_ast("ls -la | grep foo")
        assert result.get("kind") == "simple"
        cmds = result.get("commands", [])
        assert len(cmds) >= 2


class TestShellQuote:
    def test_valid_parse(self):
        result = try_parse_shell_command('echo "hello"')
        assert result.success

    def test_empty(self):
        result = try_parse_shell_command("")
        assert result.success
        assert result.tokens == []
