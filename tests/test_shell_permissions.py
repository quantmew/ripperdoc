"""Tests for the new bash permissions and validation utilities."""

from pathlib import Path

import pytest

from ripperdoc.tools.bash_tool import BashTool, BashToolInput
from ripperdoc.utils.permissions.path_validation_utils import validate_shell_command_paths
from ripperdoc.utils.permissions.tool_permission_utils import (
    evaluate_shell_command_permissions,
)
from ripperdoc.utils.safe_get_cwd import safe_get_cwd


def test_path_validation_blocks_outside_allowed(tmp_path: Path):
    """cd into a disallowed directory should trigger an ask decision."""
    allowed = {str(tmp_path)}
    result = validate_shell_command_paths("cd /", str(tmp_path), allowed)
    assert result.behavior == "ask"
    assert "blocked" in result.message.lower()


def test_path_validation_allows_within_allowed(tmp_path: Path):
    """cd into an allowed directory should pass."""
    allowed = {str(tmp_path)}
    result = validate_shell_command_paths(f"cd {tmp_path}", str(tmp_path), allowed)
    assert result.behavior == "passthrough"


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
