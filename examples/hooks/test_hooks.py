#!/usr/bin/env python3
"""Unit tests for the hooks system.

Run with: python -m pytest examples/hooks/test_hooks.py -v
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest

from ripperdoc.core.hooks import (
    HookEvent,
    HookDecision,
    HookOutput,
    HookDefinition,
    HookMatcher,
    HooksConfig,
    load_hooks_config,
    get_merged_hooks_config,
    HookExecutor,
    HookManager,
    PreToolUseInput,
    PostToolUseInput,
    UserPromptSubmitInput,
)


class TestHookEvents:
    """Test hook event enums and data structures."""

    def test_hook_event_values(self):
        """Test that all hook events have correct string values."""
        assert HookEvent.PRE_TOOL_USE.value == "PreToolUse"
        assert HookEvent.POST_TOOL_USE.value == "PostToolUse"
        assert HookEvent.USER_PROMPT_SUBMIT.value == "UserPromptSubmit"
        assert HookEvent.SESSION_START.value == "SessionStart"
        assert HookEvent.SESSION_END.value == "SessionEnd"

    def test_hook_decision_values(self):
        """Test hook decision values."""
        assert HookDecision.ALLOW.value == "allow"
        assert HookDecision.DENY.value == "deny"
        assert HookDecision.ASK.value == "ask"
        assert HookDecision.BLOCK.value == "block"


class TestHookOutput:
    """Test HookOutput parsing."""

    def test_parse_plain_text(self):
        """Test parsing plain text output."""
        output = HookOutput.from_raw("Hello world", "", 0)
        assert output.raw_output == "Hello world"
        assert output.additional_context == "Hello world"
        assert output.decision is None

    def test_parse_json_allow(self):
        """Test parsing JSON allow decision."""
        json_output = json.dumps({"decision": "allow", "reason": "Auto-approved"})
        output = HookOutput.from_raw(json_output, "", 0)
        assert output.decision == HookDecision.ALLOW
        assert output.reason == "Auto-approved"

    def test_parse_json_deny(self):
        """Test parsing JSON deny decision."""
        json_output = json.dumps({"decision": "deny", "reason": "Blocked"})
        output = HookOutput.from_raw(json_output, "", 0)
        assert output.decision == HookDecision.DENY
        assert output.reason == "Blocked"

    def test_parse_legacy_approve(self):
        """Test that 'approve' is converted to 'allow'."""
        json_output = json.dumps({"decision": "approve"})
        output = HookOutput.from_raw(json_output, "", 0)
        assert output.decision == HookDecision.ALLOW

    def test_parse_exit_code_2(self):
        """Test that exit code 2 means deny."""
        output = HookOutput.from_raw("Blocked reason", "", 2)
        assert output.decision == HookDecision.DENY
        assert "Blocked reason" in output.reason

    def test_parse_timeout(self):
        """Test timeout handling."""
        output = HookOutput.from_raw("", "", 1, timed_out=True)
        assert output.timed_out
        assert output.error == "Hook timed out"

    def test_parse_error(self):
        """Test error handling."""
        output = HookOutput.from_raw("", "Error message", 1)
        assert output.error is not None


class TestHookMatcher:
    """Test hook matcher patterns."""

    def test_match_all_empty(self):
        """Test empty matcher matches all."""
        matcher = HookMatcher(matcher=None, hooks=[])
        assert matcher.matches("Bash")
        assert matcher.matches("Write")
        assert matcher.matches(None)

    def test_match_all_star(self):
        """Test * matcher matches all."""
        matcher = HookMatcher(matcher="*", hooks=[])
        assert matcher.matches("Bash")
        assert matcher.matches("Write")

    def test_match_exact(self):
        """Test exact string matching."""
        matcher = HookMatcher(matcher="Bash", hooks=[])
        assert matcher.matches("Bash")
        assert not matcher.matches("bash")  # Case sensitive
        assert not matcher.matches("Write")

    def test_match_regex(self):
        """Test regex matching."""
        matcher = HookMatcher(matcher="Edit|Write", hooks=[])
        assert matcher.matches("Edit")
        assert matcher.matches("Write")
        assert not matcher.matches("Bash")

    def test_match_mcp_pattern(self):
        """Test MCP tool pattern matching."""
        matcher = HookMatcher(matcher="mcp__.*__write.*", hooks=[])
        assert matcher.matches("mcp__filesystem__write_file")
        assert matcher.matches("mcp__github__write_repo")
        assert not matcher.matches("mcp__filesystem__read_file")


class TestHooksConfig:
    """Test hooks configuration loading."""

    def test_load_empty_config(self):
        """Test loading from non-existent file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "hooks.json"
            config = load_hooks_config(path)
            assert len(config.hooks) == 0

    def test_load_valid_config(self):
        """Test loading valid configuration."""
        config_data = {
            "hooks": {
                "PreToolUse": [
                    {
                        "matcher": "Bash",
                        "hooks": [
                            {"type": "command", "command": "echo test", "timeout": 10}
                        ]
                    }
                ]
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "hooks.json"
            path.write_text(json.dumps(config_data))

            config = load_hooks_config(path)
            assert "PreToolUse" in config.hooks
            assert len(config.hooks["PreToolUse"]) == 1
            assert config.hooks["PreToolUse"][0].matcher == "Bash"

    def test_get_hooks_for_event(self):
        """Test getting hooks for specific events and tools."""
        config = HooksConfig(
            hooks={
                "PreToolUse": [
                    HookMatcher(
                        matcher="Bash",
                        hooks=[HookDefinition(command="echo bash")]
                    ),
                    HookMatcher(
                        matcher="Write",
                        hooks=[HookDefinition(command="echo write")]
                    ),
                ]
            }
        )

        bash_hooks = config.get_hooks_for_event(HookEvent.PRE_TOOL_USE, "Bash")
        assert len(bash_hooks) == 1
        assert bash_hooks[0].command == "echo bash"

        write_hooks = config.get_hooks_for_event(HookEvent.PRE_TOOL_USE, "Write")
        assert len(write_hooks) == 1

        # No hooks for Edit
        edit_hooks = config.get_hooks_for_event(HookEvent.PRE_TOOL_USE, "Edit")
        assert len(edit_hooks) == 0

    def test_merge_configs(self):
        """Test merging configurations."""
        config1 = HooksConfig(
            hooks={
                "PreToolUse": [
                    HookMatcher(matcher="Bash", hooks=[HookDefinition(command="echo 1")])
                ]
            }
        )

        config2 = HooksConfig(
            hooks={
                "PreToolUse": [
                    HookMatcher(matcher="Write", hooks=[HookDefinition(command="echo 2")])
                ],
                "PostToolUse": [
                    HookMatcher(matcher="*", hooks=[HookDefinition(command="echo 3")])
                ]
            }
        )

        merged = config1.merge_with(config2)
        assert len(merged.hooks["PreToolUse"]) == 2
        assert "PostToolUse" in merged.hooks


class TestHookExecutor:
    """Test hook command execution."""

    def test_expand_command(self):
        """Test environment variable expansion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            executor = HookExecutor(Path(tmpdir))
            cmd = executor._expand_command("cd $RIPPERDOC_PROJECT_DIR")
            assert tmpdir in cmd

    def test_execute_simple_command(self):
        """Test executing a simple command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            executor = HookExecutor(Path(tmpdir))
            hook = HookDefinition(command="echo 'test output'", timeout=5)
            input_data = PreToolUseInput(
                tool_name="Test",
                tool_input={},
            )

            result = executor.execute_sync(hook, input_data)
            assert result.exit_code == 0
            assert "test output" in result.raw_output

    def test_execute_blocking_command(self):
        """Test command that returns exit code 2 (block)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            executor = HookExecutor(Path(tmpdir))
            hook = HookDefinition(command="exit 2", timeout=5)
            input_data = PreToolUseInput(
                tool_name="Test",
                tool_input={},
            )

            result = executor.execute_sync(hook, input_data)
            assert result.decision == HookDecision.DENY


class TestHookManager:
    """Test hook manager functionality."""

    def test_manager_initialization(self):
        """Test manager initializes correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = HookManager(Path(tmpdir), "test-session")
            assert manager.project_dir == Path(tmpdir)
            assert manager.session_id == "test-session"

    def test_no_hooks_returns_empty_result(self):
        """Test that missing hooks return empty result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = HookManager(Path(tmpdir))
            result = manager.run_pre_tool_use("Bash", {"command": "ls"})
            assert len(result.outputs) == 0
            assert not result.should_block


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
