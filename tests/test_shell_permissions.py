"""Tests for current Bash permission and validation APIs."""

from pathlib import Path

import pytest

from ripperdoc.security.bash_security import bash_command_is_safe
from ripperdoc.tools.bash import BashTool, BashToolInput
from ripperdoc.tools.bash.path_validation import check_path_constraints
from ripperdoc.tools.bash.read_only_validation import is_command_read_only
from ripperdoc.utils.filesystem.safe_get_cwd import safe_get_cwd
from ripperdoc.utils.permissions.tool_permission_utils import match_rule


class TestBashSecurity:
    def test_empty_command_is_explicitly_allowed_by_static_security(self):
        for command in ["", "   "]:
            result = bash_command_is_safe(command)
            assert result.behavior == "allow", command

    def test_simple_commands_pass_static_security(self):
        for command in ["ls", "pwd", "git status", "find . -name '*.py'"]:
            result = bash_command_is_safe(command)
            assert result.behavior == "passthrough", command

    def test_command_substitution_requires_permission(self):
        for command in ["echo $(whoami)", "echo ${HOME}"]:
            result = bash_command_is_safe(command)
            assert result.behavior == "ask", command

    def test_single_quoted_substitution_content_is_literal(self):
        for command in ["echo 'hello `world`'", "echo 'hello $(world)'"]:
            result = bash_command_is_safe(command)
            assert result.behavior == "passthrough", command

    def test_sensitive_input_redirection_requires_permission(self):
        result = bash_command_is_safe("cat < /etc/passwd")
        assert result.behavior == "ask"

    def test_safe_dev_null_redirections_pass(self):
        for command in ["command 2>/dev/null", "command > /dev/null", "command < /dev/null"]:
            result = bash_command_is_safe(command)
            assert result.behavior == "passthrough", command

    def test_newlines_and_jq_system_require_permission(self):
        for command in ["echo hello\necho world", "jq 'system(\"id\")'"]:
            result = bash_command_is_safe(command)
            assert result.behavior == "ask", command

    def test_git_commit_single_quoted_heredoc_is_allowed_pattern(self):
        command = "git commit -m \"$(cat <<'EOF'\nCommit message\nEOF\n)\""
        result = bash_command_is_safe(command)
        assert result.behavior == "allow"


class TestPathValidation:
    def test_path_validation_blocks_cd_outside_allowed_directory(self, tmp_path: Path):
        result = check_path_constraints("cd /", str(tmp_path), {str(tmp_path)})
        assert result.behavior == "ask"

    def test_path_validation_allows_cd_inside_allowed_directory(self, tmp_path: Path):
        result = check_path_constraints(f"cd {tmp_path}", str(tmp_path), {str(tmp_path)})
        assert result.behavior == "passthrough"

    def test_output_redirection_outside_allowed_directory_requires_permission(self, tmp_path: Path):
        result = check_path_constraints("echo hello > /etc/ripperdoc-test", str(tmp_path), {str(tmp_path)})
        assert result.behavior == "ask"

    def test_dangerous_root_removal_requires_permission(self, tmp_path: Path):
        result = check_path_constraints("rm -rf /", str(tmp_path), {str(tmp_path)})
        assert result.behavior == "ask"


class TestReadOnlyValidation:
    def test_read_only_commands_are_detected(self):
        for command in ["ls", "git status", "git log --oneline", "rg pattern ."]:
            assert is_command_read_only(command) is True, command

    def test_mutating_commands_are_not_read_only(self):
        for command in ["touch file.txt", "rm file.txt", "git add file.txt", "python -c 'print(1)'"]:
            assert is_command_read_only(command) is False, command


class TestRuleMatching:
    def test_exact_and_glob_rules_match_commands(self):
        assert match_rule("git status", "git status") is True
        assert match_rule("git status", "git *") is True
        assert match_rule("git status", "git:*") is True
        assert match_rule("npm install", "* install") is True
        assert match_rule("git add", "git ??") is False

    def test_wildcard_rules_do_not_cross_shell_operator_tokens(self):
        assert match_rule("git status && rm -rf /tmp/x", "git *") is False
        assert match_rule("echo '&&'", "echo *") is False


class TestBashToolIntegration:
    @pytest.mark.asyncio
    async def test_read_only_command_does_not_need_permissions(self):
        tool = BashTool()
        assert tool.needs_permissions(BashToolInput(command="ls")) is False

    @pytest.mark.asyncio
    async def test_background_command_needs_permissions(self):
        tool = BashTool()
        assert tool.needs_permissions(BashToolInput(command="ls", run_in_background=True)) is True
        assert tool.needs_permissions(BashToolInput(command="ls &")) is True

    @pytest.mark.asyncio
    async def test_check_permissions_honors_deny_rule(self):
        tool = BashTool()
        decision = await tool.check_permissions(
            BashToolInput(command="echo hi"),
            {
                "mode": "default",
                "allowed_rules": set(),
                "denied_rules": {"echo hi"},
                "ask_rules": set(),
                "allowed_working_directories": {safe_get_cwd()},
            },
        )
        assert decision.behavior == "deny"

    @pytest.mark.asyncio
    async def test_check_permissions_honors_ask_rule_over_allow_rule(self):
        tool = BashTool()
        decision = await tool.check_permissions(
            BashToolInput(command="ls"),
            {
                "mode": "default",
                "allowed_rules": {"ls"},
                "denied_rules": set(),
                "ask_rules": {"ls"},
                "allowed_working_directories": {safe_get_cwd()},
            },
        )
        assert decision.behavior == "ask"

    @pytest.mark.asyncio
    async def test_check_permissions_allows_read_only_command(self):
        tool = BashTool()
        decision = await tool.check_permissions(
            BashToolInput(command="ls"),
            {
                "mode": "default",
                "allowed_rules": set(),
                "denied_rules": set(),
                "ask_rules": set(),
                "allowed_working_directories": {safe_get_cwd()},
            },
        )
        assert decision.behavior == "allow"

    @pytest.mark.asyncio
    async def test_check_permissions_asks_for_dangerous_command(self):
        tool = BashTool()
        decision = await tool.check_permissions(
            BashToolInput(command="echo $(whoami)"),
            {
                "mode": "default",
                "allowed_rules": set(),
                "denied_rules": set(),
                "ask_rules": set(),
                "allowed_working_directories": {safe_get_cwd()},
            },
        )
        assert decision.behavior == "ask"

    @pytest.mark.asyncio
    async def test_validate_input_blocks_unavailable_sandbox(self):
        tool = BashTool()
        result = await tool.validate_input(BashToolInput(command="echo hi", sandbox=True), None)
        assert result.result is False
        assert "sandbox" in (result.message or "").lower()

    @pytest.mark.asyncio
    async def test_validate_input_allows_safe_command(self):
        tool = BashTool()
        result = await tool.validate_input(BashToolInput(command="ls -la"), None)
        assert result.result is True
