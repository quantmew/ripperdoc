from pathlib import Path

import pytest

from ripperdoc.tools.bash import BashTool, BashToolInput
from ripperdoc.utils.filesystem.safe_get_cwd import safe_get_cwd


@pytest.mark.asyncio
async def test_sandbox_does_not_bypass_deny_rule():
    tool = BashTool()
    decision = await tool.check_permissions(
        BashToolInput(command="echo denied", sandbox=True),
        {
            "mode": "default",
            "allowed_rules": set(),
            "denied_rules": {"echo denied"},
            "ask_rules": set(),
            "allowed_working_directories": {safe_get_cwd()},
        },
    )
    assert decision.behavior == "deny"


@pytest.mark.asyncio
async def test_sandbox_does_not_bypass_ask_rule():
    tool = BashTool()
    decision = await tool.check_permissions(
        BashToolInput(command="ls", sandbox=True),
        {
            "mode": "default",
            "allowed_rules": set(),
            "denied_rules": set(),
            "ask_rules": {"ls"},
            "allowed_working_directories": {safe_get_cwd()},
        },
    )
    assert decision.behavior == "ask"


@pytest.mark.asyncio
async def test_sandbox_dontask_denies_ask_rule():
    tool = BashTool()
    decision = await tool.check_permissions(
        BashToolInput(command="ls", sandbox=True),
        {
            "mode": "dontAsk",
            "allowed_rules": set(),
            "denied_rules": set(),
            "ask_rules": {"ls"},
            "allowed_working_directories": {safe_get_cwd()},
        },
    )
    assert decision.behavior == "deny"


@pytest.mark.asyncio
async def test_compound_denied_segment_blocks_command():
    tool = BashTool()
    decision = await tool.check_permissions(
        BashToolInput(command="ls && rm -rf /tmp/ripperdoc-test"),
        {
            "mode": "default",
            "allowed_rules": {"ls"},
            "denied_rules": {"rm *"},
            "ask_rules": set(),
            "allowed_working_directories": {safe_get_cwd()},
        },
    )
    assert decision.behavior == "deny"


@pytest.mark.asyncio
async def test_pipe_allow_revalidates_original_output_redirection(tmp_path: Path):
    tool = BashTool()
    decision = await tool.check_permissions(
        BashToolInput(command="echo hi | cat > /etc/ripperdoc-test"),
        {
            "mode": "default",
            "allowed_rules": {"echo:*", "cat:*"},
            "denied_rules": set(),
            "ask_rules": set(),
            "allowed_working_directories": {str(tmp_path)},
        },
    )
    assert decision.behavior == "ask"


@pytest.mark.asyncio
async def test_bash_dontask_denies_mutating_command():
    tool = BashTool()
    decision = await tool.check_permissions(
        BashToolInput(command="touch ripperdoc-test"),
        {
            "mode": "dontAsk",
            "allowed_rules": set(),
            "denied_rules": set(),
            "ask_rules": set(),
            "allowed_working_directories": {safe_get_cwd()},
        },
    )
    assert decision.behavior == "deny"


@pytest.mark.asyncio
async def test_bash_dontask_allows_read_only_command():
    tool = BashTool()
    decision = await tool.check_permissions(
        BashToolInput(command="ls"),
        {
            "mode": "dontAsk",
            "allowed_rules": set(),
            "denied_rules": set(),
            "ask_rules": set(),
            "allowed_working_directories": {safe_get_cwd()},
        },
    )
    assert decision.behavior == "allow"
