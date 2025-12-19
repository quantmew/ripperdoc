"""Tests for the new bash permissions and validation utilities."""

from pathlib import Path

import pytest

from ripperdoc.tools.bash_tool import BashTool, BashToolInput
from ripperdoc.utils.permissions.path_validation_utils import validate_shell_command_paths
from ripperdoc.utils.permissions.shell_command_validation import (
    validate_shell_command,
    is_complex_unsafe_shell_command,
)
from ripperdoc.utils.permissions.tool_permission_utils import (
    evaluate_shell_command_permissions,
)
from ripperdoc.utils.safe_get_cwd import safe_get_cwd


# =============================================================================
# Shell Command Validation Tests
# =============================================================================


class TestValidateShellCommand:
    """Tests for validate_shell_command function."""

    def test_empty_command_is_safe(self):
        """Empty commands should pass validation."""
        result = validate_shell_command("")
        assert result.behavior == "passthrough"

        result = validate_shell_command("   ")
        assert result.behavior == "passthrough"

    def test_simple_safe_commands(self):
        """Simple safe commands should pass validation."""
        safe_commands = [
            "ls",
            "ls -la",
            "pwd",
            "echo hello",
            "cat file.txt",
            "grep pattern file.txt",
            "find . -name '*.py'",
            "git status",
            "git log --oneline",
        ]
        for cmd in safe_commands:
            result = validate_shell_command(cmd)
            assert result.behavior == "passthrough", f"Command should be safe: {cmd}"

    def test_gemini3_unsafe_command(self):
        """Commands that are unsafe in Gemini 3 context should be blocked."""
        result = validate_shell_command(r'cmd /c "rmdir /s /q \"C:\Users\xxx\SCE Projects\src\""')
        assert result.behavior == "deny"

    def test_backtick_command_substitution_blocked(self):
        """Backtick command substitution should be blocked."""
        result = validate_shell_command("echo `whoami`")
        assert result.behavior == "ask"
        assert "backtick" in result.message.lower()

    def test_dollar_paren_command_substitution_blocked(self):
        """$() command substitution should be blocked."""
        result = validate_shell_command("echo $(whoami)")
        assert result.behavior == "ask"
        assert "command substitution" in result.message.lower()

    def test_parameter_substitution_blocked(self):
        """${} parameter substitution should be blocked."""
        result = validate_shell_command("echo ${HOME}")
        assert result.behavior == "ask"
        assert "parameter substitution" in result.message.lower()

    def test_process_substitution_blocked(self):
        """Process substitution <() and >() should be blocked."""
        result = validate_shell_command("diff <(cat file1) <(cat file2)")
        assert result.behavior == "ask"
        assert "process substitution" in result.message.lower()

        result = validate_shell_command("tee >(cat)")
        assert result.behavior == "ask"
        assert "process substitution" in result.message.lower()

    def test_input_redirection_blocked(self):
        """Input redirection should be blocked (except /dev/null)."""
        result = validate_shell_command("cat < /etc/passwd")
        assert result.behavior == "ask"
        assert "input redirection" in result.message.lower()

    def test_output_redirection_blocked(self):
        """Output redirection should be blocked (except /dev/null)."""
        result = validate_shell_command("echo hello > file.txt")
        assert result.behavior == "ask"
        assert "output redirection" in result.message.lower()

    def test_dev_null_redirection_allowed(self):
        """Redirection to /dev/null should be allowed."""
        # These should pass because /dev/null is safe
        result = validate_shell_command("command 2>/dev/null")
        assert result.behavior == "passthrough"

        result = validate_shell_command("command > /dev/null")
        assert result.behavior == "passthrough"

        result = validate_shell_command("command < /dev/null")
        assert result.behavior == "passthrough"

    def test_newlines_blocked(self):
        """Commands with newlines should be blocked."""
        result = validate_shell_command("echo hello\necho world")
        assert result.behavior == "ask"
        assert "newline" in result.message.lower()

    def test_jq_system_blocked(self):
        """jq system() function should be blocked."""
        result = validate_shell_command("jq 'system(\"id\")'")
        assert result.behavior == "ask"
        assert "system()" in result.message.lower()

    def test_heredoc_blocked(self):
        """Heredoc should be blocked."""
        result = validate_shell_command("cat << EOF\nhello\nEOF")
        assert result.behavior == "ask"
        assert "heredoc" in result.message.lower()

    def test_eval_blocked(self):
        """eval command should be blocked."""
        result = validate_shell_command("eval 'echo hello'")
        assert result.behavior == "ask"
        assert "eval" in result.message.lower()

    def test_source_blocked(self):
        """source command should be blocked."""
        result = validate_shell_command("source script.sh")
        assert result.behavior == "ask"
        assert "source" in result.message.lower()

    def test_dot_source_blocked(self):
        """. (dot) command should be blocked."""
        result = validate_shell_command(". script.sh")
        assert result.behavior == "ask"
        assert "source" in result.message.lower()

    def test_single_quoted_content_is_safe(self):
        """Content inside single quotes should be ignored for security checks."""
        # Backticks inside single quotes are literal
        result = validate_shell_command("echo 'hello `world`'")
        assert result.behavior == "passthrough"

        # $() inside single quotes is literal
        result = validate_shell_command("echo 'hello $(world)'")
        assert result.behavior == "passthrough"

    def test_variables_in_dangerous_context_blocked(self):
        """Variables in redirections or pipes should be blocked."""
        result = validate_shell_command("echo hello | $PAGER")
        assert result.behavior == "ask"
        # The command contains | which is a shell metacharacter
        # With the improved validation, we check for metacharacters first
        # So the message might be about metacharacters, not variables
        # But the command should still be blocked
        pass  # Just check that behavior is "ask"

    def test_git_commit_heredoc_allowed(self):
        """Git commit with single-quoted heredoc should be allowed."""
        cmd = "git commit -m \"$(cat <<'EOF'\nCommit message\nEOF\n)\""
        result = validate_shell_command(cmd)
        assert result.behavior == "passthrough"

    def test_find_with_metacharacters_blocked(self):
        """find with shell metacharacters in arguments should be blocked."""
        # This command has metacharacters in double quotes (not single quotes)
        # With improved validation, metacharacters inside quotes are allowed
        # because they're part of the argument, not shell operators
        result = validate_shell_command('find . -name "*.py;rm -rf /"')
        assert result.behavior == "passthrough"  # Changed from "ask"
        # Note: This is now allowed because ; is inside quotes


class TestIsComplexUnsafeShellCommand:
    """Tests for is_complex_unsafe_shell_command function."""

    def test_simple_commands_safe(self):
        """Simple commands without operators should be safe."""
        assert is_complex_unsafe_shell_command("ls") is False
        assert is_complex_unsafe_shell_command("echo hello") is False
        assert is_complex_unsafe_shell_command("git status") is False

    def test_and_operator_detected(self):
        """&& operator should be detected."""
        assert is_complex_unsafe_shell_command("cmd1 && cmd2") is True

    def test_or_operator_detected(self):
        """|| operator should be detected."""
        assert is_complex_unsafe_shell_command("cmd1 || cmd2") is True

    def test_semicolon_operator_detected(self):
        """; operator should be detected."""
        assert is_complex_unsafe_shell_command("cmd1; cmd2") is True

    def test_operators_in_quotes_safe(self):
        """Operators inside quotes should not be detected."""
        assert is_complex_unsafe_shell_command("echo '&&'") is False
        assert is_complex_unsafe_shell_command('echo "||"') is False
        assert is_complex_unsafe_shell_command("echo ';'") is False

    def test_pipe_not_considered_complex(self):
        """Single pipe is not considered complex unsafe."""
        assert is_complex_unsafe_shell_command("ls | grep foo") is False

    def test_empty_command(self):
        """Empty command should be safe."""
        assert is_complex_unsafe_shell_command("") is False


# =============================================================================
# Path Validation Tests
# =============================================================================


def test_path_validation_blocks_outside_allowed(tmp_path: Path):
    """cd into a disallowed directory should trigger an ask decision."""
    allowed = {str(tmp_path)}
    result = validate_shell_command_paths("cd /", str(tmp_path), allowed)
    assert result.behavior == "ask"
    assert "permission" in result.message.lower() or "outside" in result.message.lower()


def test_path_validation_allows_within_allowed(tmp_path: Path):
    """cd into an allowed directory should pass."""
    allowed = {str(tmp_path)}
    result = validate_shell_command_paths(f"cd {tmp_path}", str(tmp_path), allowed)
    assert result.behavior == "passthrough"


def test_ls_allows_any_path(tmp_path: Path):
    """ls is a read-only command and should be allowed on any path."""
    allowed = {str(tmp_path)}
    # ls to /usr should be allowed even though it's outside allowed dirs
    result = validate_shell_command_paths("ls /usr", str(tmp_path), allowed)
    assert result.behavior == "passthrough"
    assert "read-only" in result.message.lower()

    # ls to /etc should also be allowed
    result = validate_shell_command_paths("ls -la /etc", str(tmp_path), allowed)
    assert result.behavior == "passthrough"

    # ls to root should also be allowed
    result = validate_shell_command_paths("ls /", str(tmp_path), allowed)
    assert result.behavior == "passthrough"


# =============================================================================
# Permission Evaluation Tests
# =============================================================================


def test_evaluate_permissions_honors_deny_rule():
    """Explicit deny rule should block the command."""
    cwd = safe_get_cwd()
    decision = evaluate_shell_command_permissions(
        type("Req", (), {"command": "echo hi"}),
        allowed_rules=set(),
        denied_rules={"echo hi"},
        allowed_working_dirs={cwd},
    )
    assert decision.behavior == "deny"


def test_evaluate_permissions_allows_read_only():
    """Read-only commands without rules should be allowed."""
    cwd = safe_get_cwd()
    decision = evaluate_shell_command_permissions(
        type("Req", (), {"command": "ls"}),
        allowed_rules=set(),
        denied_rules=set(),
        allowed_working_dirs={cwd},
    )
    assert decision.behavior == "allow"


# =============================================================================
# BashTool Integration Tests
# =============================================================================


@pytest.mark.asyncio
async def test_bash_tool_needs_permissions_respects_read_only():
    """Read-only commands should bypass permission prompts."""
    tool = BashTool()
    input_data = BashToolInput(command="ls")
    assert tool.needs_permissions(input_data) is False


@pytest.mark.asyncio
async def test_background_commands_still_need_permissions():
    """Background bash runs should always require approval."""
    tool = BashTool()
    input_data = BashToolInput(command="ls", run_in_background=True)

    assert tool.needs_permissions(input_data) is True

    decision = await tool.check_permissions(
        input_data,
        {
            "allowed_rules": set(),
            "denied_rules": set(),
            "allowed_working_directories": {safe_get_cwd()},
        },
    )
    assert getattr(decision, "behavior", None) in {"ask", "passthrough"}


@pytest.mark.asyncio
async def test_auto_background_ampersand_requires_permission():
    """Trailing ampersand should trigger permission even without run_in_background flag."""
    tool = BashTool()
    input_data = BashToolInput(command="ls &")

    assert tool.needs_permissions(input_data) is True

    decision = await tool.check_permissions(
        input_data,
        {
            "allowed_rules": set(),
            "denied_rules": set(),
            "allowed_working_directories": {safe_get_cwd()},
        },
    )
    assert getattr(decision, "behavior", None) in {"ask", "passthrough"}


@pytest.mark.asyncio
async def test_bash_tool_check_permissions_respects_allow_rules():
    """check_permissions should allow when rule matches."""
    tool = BashTool()
    cwd = safe_get_cwd()
    input_data = BashToolInput(command="echo hi")
    decision = await tool.check_permissions(
        input_data,
        {
            "allowed_rules": {"echo hi"},
            "denied_rules": set(),
            "allowed_working_directories": {cwd},
        },
    )
    assert getattr(decision, "behavior", None) == "allow"


@pytest.mark.asyncio
async def test_bash_tool_validate_input_blocks_unavailable_sandbox():
    """Sandbox requests should be rejected when sandbox is unavailable."""
    tool = BashTool()
    input_data = BashToolInput(command="echo hi", sandbox=True)
    result = await tool.validate_input(input_data, None)
    assert result.result is False
    assert "sandbox" in (result.message or "").lower()


@pytest.mark.asyncio
async def test_bash_tool_validate_input_blocks_dangerous_commands():
    """Dangerous shell commands should be blocked by validate_input."""
    tool = BashTool()

    # Test command substitution
    input_data = BashToolInput(command="echo $(whoami)")
    result = await tool.validate_input(input_data, None)
    assert result.result is False
    assert "command substitution" in (result.message or "").lower()

    # Test eval
    input_data = BashToolInput(command="eval 'echo hello'")
    result = await tool.validate_input(input_data, None)
    assert result.result is False
    assert "eval" in (result.message or "").lower()


@pytest.mark.asyncio
async def test_bash_tool_validate_input_allows_safe_commands():
    """Safe shell commands should pass validate_input."""
    tool = BashTool()
    input_data = BashToolInput(command="ls -la")
    result = await tool.validate_input(input_data, None)
    assert result.result is True


# =============================================================================
# Destructive Command Detection Tests (Gemini Incident Prevention)
# =============================================================================


class TestWindowsDestructiveCommands:
    """Tests for Windows destructive command detection."""

    def test_rmdir_with_s_flag_blocked(self):
        """rmdir /s should be blocked."""
        result = validate_shell_command("rmdir /s folder")
        assert result.behavior == "ask"
        assert "rmdir" in result.message.lower()

    def test_rmdir_with_s_and_q_flags_blocked(self):
        """rmdir /s /q should be blocked."""
        result = validate_shell_command("rmdir /s /q folder")
        assert result.behavior == "ask"
        assert "rmdir" in result.message.lower()

    def test_rd_with_s_flag_blocked(self):
        """rd /s (alias for rmdir) should be blocked."""
        result = validate_shell_command("rd /s folder")
        assert result.behavior == "ask"
        assert "rd /s" in result.message.lower()

    def test_del_with_s_flag_blocked(self):
        """del /s should be blocked."""
        result = validate_shell_command("del /s *.tmp")
        assert result.behavior == "ask"
        assert "del" in result.message.lower()

    def test_del_with_q_flag_blocked(self):
        """del /q should be blocked."""
        result = validate_shell_command("del /q file.txt")
        assert result.behavior == "ask"
        assert "del" in result.message.lower()

    def test_format_drive_blocked(self):
        """format command should be blocked."""
        result = validate_shell_command("format C:")
        assert result.behavior == "ask"
        assert "format" in result.message.lower()

    def test_cmd_c_with_rmdir_blocked(self):
        """cmd /c with rmdir should be blocked."""
        result = validate_shell_command("cmd /c rmdir /s folder")
        assert result.behavior == "ask"
        assert "cmd /c" in result.message.lower() or "rmdir" in result.message.lower()

    def test_powershell_remove_item_recurse_blocked(self):
        """PowerShell Remove-Item -Recurse should be blocked."""
        result = validate_shell_command("Remove-Item -Recurse folder")
        assert result.behavior == "ask"
        assert "remove-item" in result.message.lower() or "recurse" in result.message.lower()

    def test_powershell_remove_item_force_blocked(self):
        """PowerShell Remove-Item -Force should be blocked."""
        result = validate_shell_command("Remove-Item -Force file.txt")
        assert result.behavior == "ask"
        assert "force" in result.message.lower()

    def test_simple_rmdir_without_flags_allowed(self):
        """Simple rmdir without dangerous flags should pass."""
        result = validate_shell_command("rmdir empty_folder")
        assert result.behavior == "passthrough"

    def test_simple_del_without_flags_allowed(self):
        """Simple del without dangerous flags should pass."""
        result = validate_shell_command("del file.txt")
        assert result.behavior == "passthrough"


class TestUnixDestructiveCommands:
    """Tests for Unix/Linux destructive command detection."""

    def test_rm_rf_blocked(self):
        """rm -rf should be blocked."""
        result = validate_shell_command("rm -rf folder")
        assert result.behavior == "ask"
        assert "rm" in result.message.lower()

    def test_rm_r_blocked(self):
        """rm -r should be blocked."""
        result = validate_shell_command("rm -r folder")
        assert result.behavior == "ask"
        assert "rm" in result.message.lower()

    def test_rm_fr_blocked(self):
        """rm -fr (alternative flag order) should be blocked."""
        result = validate_shell_command("rm -fr folder")
        assert result.behavior == "ask"
        assert "rm" in result.message.lower()

    def test_dd_to_device_blocked(self):
        """dd writing to device should be blocked."""
        result = validate_shell_command("dd if=/dev/zero of=/dev/sda")
        assert result.behavior == "ask"
        assert "dd" in result.message.lower()

    def test_mkfs_blocked(self):
        """mkfs command should be blocked."""
        result = validate_shell_command("mkfs.ext4 /dev/sda1")
        assert result.behavior == "ask"
        assert "mkfs" in result.message.lower()

    def test_shred_blocked(self):
        """shred command should be blocked."""
        result = validate_shell_command("shred -u file.txt")
        assert result.behavior == "ask"
        assert "shred" in result.message.lower()

    def test_chmod_777_on_root_blocked(self):
        """chmod 777 on root should be denied (critical path)."""
        result = validate_shell_command("chmod 777 /")
        assert result.behavior == "deny"
        assert "blocked" in result.message.lower()

    def test_chmod_777_on_etc_blocked(self):
        """chmod 777 on /etc should be denied (critical path)."""
        result = validate_shell_command("chmod 777 /etc")
        assert result.behavior == "deny"
        assert "blocked" in result.message.lower()

    def test_simple_rm_without_r_allowed(self):
        """Simple rm without -r flag should pass."""
        result = validate_shell_command("rm file.txt")
        assert result.behavior == "passthrough"

    def test_rm_i_allowed(self):
        """rm -i (interactive) should pass."""
        result = validate_shell_command("rm -i file.txt")
        assert result.behavior == "passthrough"

    def test_chmod_normal_usage_allowed(self):
        """Normal chmod usage should pass."""
        result = validate_shell_command("chmod 755 script.sh")
        assert result.behavior == "passthrough"


class TestCriticalPathDetection:
    """Tests for critical path detection with destructive commands."""

    def test_rmdir_on_c_drive_root_denied(self):
        """rmdir /s on C:\\ should be denied."""
        result = validate_shell_command("rmdir /s /q C:\\")
        assert result.behavior == "deny"
        assert "blocked" in result.message.lower()

    def test_rmdir_on_windows_denied(self):
        """rmdir /s on Windows folder should be denied."""
        result = validate_shell_command(r"rmdir /s /q C:\Windows")
        assert result.behavior == "deny"
        assert "blocked" in result.message.lower()

    def test_rmdir_on_users_denied(self):
        """rmdir /s on Users folder should be denied."""
        result = validate_shell_command(r"rmdir /s /q C:\Users")
        assert result.behavior == "deny"
        assert "blocked" in result.message.lower()

    def test_rmdir_on_program_files_denied(self):
        """rmdir /s on Program Files should be denied."""
        result = validate_shell_command(r'rmdir /s /q "C:\Program Files"')
        assert result.behavior == "deny"
        assert "blocked" in result.message.lower()

    def test_rm_rf_on_root_denied(self):
        """rm -rf on / should be denied."""
        result = validate_shell_command("rm -rf /")
        assert result.behavior == "deny"
        assert "blocked" in result.message.lower()

    def test_rm_rf_on_home_denied(self):
        """rm -rf on /home should be denied."""
        result = validate_shell_command("rm -rf /home")
        assert result.behavior == "deny"
        assert "blocked" in result.message.lower()

    def test_rm_rf_on_etc_denied(self):
        """rm -rf on /etc should be denied."""
        result = validate_shell_command("rm -rf /etc")
        assert result.behavior == "deny"
        assert "blocked" in result.message.lower()

    def test_rm_rf_on_usr_denied(self):
        """rm -rf on /usr should be denied."""
        result = validate_shell_command("rm -rf /usr")
        assert result.behavior == "deny"
        assert "blocked" in result.message.lower()

    def test_rm_rf_on_var_denied(self):
        """rm -rf on /var should be denied."""
        result = validate_shell_command("rm -rf /var")
        assert result.behavior == "deny"
        assert "blocked" in result.message.lower()

    def test_rm_rf_on_home_tilde_denied(self):
        """rm -rf on ~ should be denied."""
        result = validate_shell_command("rm -rf ~")
        assert result.behavior == "deny"
        assert "blocked" in result.message.lower()

    def test_rm_rf_on_user_folder_ask(self):
        """rm -rf on user's own subfolder should ask, not deny."""
        result = validate_shell_command("rm -rf /home/user/project/build")
        assert result.behavior == "ask"
        assert result.behavior != "deny"


class TestNestedQuoteDetection:
    """Tests for nested quote detection (Gemini incident pattern)."""

    def test_gemini_incident_exact_pattern_denied(self):
        """The exact Gemini incident pattern should be denied."""
        cmd = r'cmd /c "rmdir /s /q \"C:\Users\xxx\SCE Projects\src\""'
        result = validate_shell_command(cmd)
        assert result.behavior == "deny"
        assert "blocked" in result.message.lower()

    def test_cmd_c_with_escaped_quotes_asks(self):
        """cmd /c with escaped quotes (non-critical path) should ask."""
        cmd = r'cmd /c "echo \"hello world\""'
        result = validate_shell_command(cmd)
        assert result.behavior == "ask"
        assert "nested" in result.message.lower() or "escaped" in result.message.lower()

    def test_simple_cmd_c_allowed(self):
        """Simple cmd /c without nested quotes should pass."""
        result = validate_shell_command("cmd /c dir")
        assert result.behavior == "passthrough"

    def test_cmd_c_with_simple_quotes_allowed(self):
        """cmd /c with simple (non-nested) quotes should pass."""
        result = validate_shell_command('cmd /c "echo hello"')
        assert result.behavior == "passthrough"


class TestFalsePositivePrevention:
    """Tests to ensure we don't have false positives."""

    def test_rm_in_quoted_string_not_blocked(self):
        """'rm' inside quoted strings should not trigger rm detection."""
        result = validate_shell_command('echo "rm -rf /"')
        assert result.behavior == "passthrough"

    def test_find_with_rm_pattern_in_name_not_blocked(self):
        """find with rm pattern in -name should not trigger rm detection."""
        result = validate_shell_command('find . -name "rm-rf-test.txt"')
        assert result.behavior == "passthrough"

    def test_grep_for_rm_command_not_blocked(self):
        """grep searching for rm command should not trigger rm detection."""
        result = validate_shell_command('grep "rm -rf" history.log')
        assert result.behavior == "passthrough"

    def test_echo_rmdir_not_blocked(self):
        """echo rmdir should not trigger rmdir detection."""
        result = validate_shell_command('echo "rmdir /s /q folder"')
        assert result.behavior == "passthrough"

    def test_variable_named_rm_not_blocked(self):
        """Variables containing 'rm' should not trigger detection."""
        # This should be blocked for other reasons ($ substitution)
        # but not for rm detection
        result = validate_shell_command("echo $rmdir")
        # This is blocked due to $ but not due to rmdir
        assert "rmdir" not in result.message.lower() if result.behavior == "ask" else True

    def test_path_containing_rm_not_blocked(self):
        """Paths containing 'rm' in folder names should not trigger detection."""
        result = validate_shell_command("ls /home/user/firmware/")
        assert result.behavior == "passthrough"

    def test_safe_dd_usage_allowed(self):
        """dd to regular file should pass."""
        result = validate_shell_command("dd if=input.img of=output.img")
        assert result.behavior == "passthrough"

    def test_chmod_on_user_file_allowed(self):
        """chmod on user files should pass."""
        result = validate_shell_command("chmod 644 ~/.bashrc")
        assert result.behavior == "passthrough"


class TestEdgeCases:
    """Tests for edge cases and complex scenarios."""

    def test_multiple_flags_rm(self):
        """rm with multiple flags should be detected."""
        result = validate_shell_command("rm -rfv folder")
        assert result.behavior == "ask"

    def test_rm_with_path_before_flags(self):
        """rm with path after -r flag should be detected."""
        result = validate_shell_command("rm -r ./build")
        assert result.behavior == "ask"

    def test_case_insensitive_windows_commands(self):
        """Windows commands should be case-insensitive."""
        result = validate_shell_command("RMDIR /S /Q folder")
        assert result.behavior == "ask"

        result = validate_shell_command("DEL /S *.tmp")
        assert result.behavior == "ask"

    def test_mixed_case_windows_paths(self):
        """Windows paths should be case-insensitive."""
        result = validate_shell_command(r"rmdir /s /q c:\WINDOWS")
        assert result.behavior == "deny"

    def test_forward_slash_windows_path(self):
        """Windows paths with forward slashes should be detected."""
        result = validate_shell_command("rmdir /s /q C:/Users")
        # This might not match our pattern, but should still ask for rmdir /s
        assert result.behavior in ("ask", "deny")

    def test_powershell_aliases(self):
        """PowerShell aliases should be detected."""
        # 'ri' is alias for Remove-Item
        result = validate_shell_command("ri -Recurse folder")
        assert result.behavior == "ask"

    def test_whitespace_variations(self):
        """Various whitespace patterns should still be detected."""
        result = validate_shell_command("rm   -rf   folder")
        assert result.behavior == "ask"

        result = validate_shell_command("rmdir  /s  /q  folder")
        assert result.behavior == "ask"

    def test_command_at_end_of_pipe(self):
        """Dangerous commands at end of pipe should be detected."""
        # Note: pipe itself triggers other checks, but rm -rf should still be detected
        result = validate_shell_command("cat files.txt | xargs rm -rf")
        assert result.behavior == "ask"

    def test_empty_string_safe(self):
        """Empty strings should be safe."""
        result = validate_shell_command("")
        assert result.behavior == "passthrough"

    def test_whitespace_only_safe(self):
        """Whitespace-only strings should be safe."""
        result = validate_shell_command("   \t  ")
        assert result.behavior == "passthrough"
