"""Comprehensive tests for the hooks system.

Tests cover:
- HookOutput parsing (JSON, plain text, exit codes)
- HookMatcher pattern matching
- HooksConfig loading and merging
- HookExecutor command and prompt execution
- HookManager event handling
- Environment variable substitution
- Timeout handling
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, patch

import pytest

from ripperdoc.core.hooks.config import (
    DEFAULT_HOOK_TIMEOUT,
    HookDefinition,
    HookMatcher,
    HooksConfig,
    get_global_hooks_path,
    get_merged_hooks_config,
    get_project_hooks_path,
    get_project_local_hooks_path,
    load_hooks_config,
)
from ripperdoc.core.hooks.events import (
    HookDecision,
    HookEvent,
    HookOutput,
    PreToolUseHookOutput,
    PreToolUseInput,
    PermissionRequestInput,
    PostToolUseInput,
    UserPromptSubmitInput,
    NotificationInput,
    StopInput,
    SubagentStopInput,
    PreCompactInput,
    SessionStartInput,
    SessionEndInput,
)
from ripperdoc.core.hooks.executor import HookExecutor
from ripperdoc.core.hooks.manager import HookManager, HookResult


# ─────────────────────────────────────────────────────────────────────────────
# HookOutput Parsing Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestHookOutputParsing:
    """Tests for HookOutput.from_raw() parsing logic."""

    def test_empty_output_exit_0(self):
        """Empty stdout with exit code 0 should return default HookOutput."""
        output = HookOutput.from_raw("", "", 0)
        assert output.exit_code == 0
        assert output.decision is None
        assert output.error is None
        assert output.continue_execution is True

    def test_plain_text_output(self):
        """Plain text output should be treated as additional context."""
        output = HookOutput.from_raw("Some warning message", "", 0)
        assert output.raw_output == "Some warning message"
        assert output.additional_context == "Some warning message"
        assert output.decision is None

    def test_json_allow_decision(self):
        """JSON with 'allow' decision should set decision correctly."""
        json_output = json.dumps({"decision": "allow", "reason": "Auto-approved"})
        output = HookOutput.from_raw(json_output, "", 0)
        assert output.decision == HookDecision.ALLOW
        assert output.reason == "Auto-approved"

    def test_json_deny_decision(self):
        """JSON with 'deny' decision should set decision correctly."""
        json_output = json.dumps({"decision": "deny", "reason": "Blocked for safety"})
        output = HookOutput.from_raw(json_output, "", 0)
        assert output.decision == HookDecision.DENY
        assert output.reason == "Blocked for safety"

    def test_json_ask_decision(self):
        """JSON with 'ask' decision should set decision correctly."""
        json_output = json.dumps({"decision": "ask", "reason": "Please confirm"})
        output = HookOutput.from_raw(json_output, "", 0)
        assert output.decision == HookDecision.ASK
        assert output.reason == "Please confirm"

    def test_json_block_decision(self):
        """JSON with 'block' decision should set decision correctly."""
        json_output = json.dumps({"decision": "block", "reason": "Operation blocked"})
        output = HookOutput.from_raw(json_output, "", 0)
        assert output.decision == HookDecision.BLOCK
        assert output.reason == "Operation blocked"

    def test_json_approve_legacy_alias(self):
        """JSON with 'approve' (legacy) should map to ALLOW."""
        json_output = json.dumps({"decision": "approve", "reason": "Legacy approval"})
        output = HookOutput.from_raw(json_output, "", 0)
        assert output.decision == HookDecision.ALLOW

    def test_exit_code_2_blocking_error(self):
        """Exit code 2 should treat stderr as blocking error."""
        output = HookOutput.from_raw("", "Dangerous operation detected", 2)
        assert output.exit_code == 2
        assert output.decision == HookDecision.DENY
        assert output.reason == "Dangerous operation detected"

    def test_exit_code_2_with_empty_stderr(self):
        """Exit code 2 with empty stderr should use default message."""
        output = HookOutput.from_raw("", "", 2)
        assert output.decision == HookDecision.DENY
        assert output.reason == "Blocked by hook"

    def test_non_zero_exit_code_error(self):
        """Non-zero exit code (not 2) should be non-blocking error."""
        output = HookOutput.from_raw("", "Some error occurred", 1)
        assert output.exit_code == 1
        assert output.error == "Some error occurred"
        assert output.decision is None

    def test_json_continue_false(self):
        """JSON with continue=false should stop execution."""
        json_output = json.dumps({"continue": False, "stopReason": "Task completed"})
        output = HookOutput.from_raw(json_output, "", 0)
        assert output.continue_execution is False
        assert output.stop_reason == "Task completed"

    def test_json_system_message(self):
        """JSON with systemMessage should be parsed correctly."""
        json_output = json.dumps({"systemMessage": "Warning: Low disk space"})
        output = HookOutput.from_raw(json_output, "", 0)
        assert output.system_message == "Warning: Low disk space"

    def test_json_suppress_output(self):
        """JSON with suppressOutput should be parsed correctly."""
        json_output = json.dumps({"suppressOutput": True})
        output = HookOutput.from_raw(json_output, "", 0)
        assert output.suppress_output is True

    def test_json_additional_context(self):
        """JSON with additionalContext should be parsed correctly."""
        json_output = json.dumps({"additionalContext": "Context for model"})
        output = HookOutput.from_raw(json_output, "", 0)
        assert output.additional_context == "Context for model"

    def test_timed_out_flag(self):
        """Timed out hook should set error message."""
        output = HookOutput.from_raw("", "", 1, timed_out=True)
        assert output.timed_out is True
        assert output.error == "Hook timed out"

    def test_hook_specific_output_pre_tool_use(self):
        """JSON with hookSpecificOutput for PreToolUse should parse correctly."""
        json_output = json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "allow",
                "permissionDecisionReason": "Allowed by policy",
                "updatedInput": {"command": "ls -la"},
                "additionalContext": "Modified command",
            }
        })
        output = HookOutput.from_raw(json_output, "", 0)
        assert output.decision == HookDecision.ALLOW
        assert output.reason == "Allowed by policy"
        assert isinstance(output.hook_specific_output, PreToolUseHookOutput)
        assert output.updated_input == {"command": "ls -la"}

    def test_hook_specific_output_permission_request(self):
        """JSON with hookSpecificOutput for PermissionRequest should parse correctly."""
        json_output = json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "PermissionRequest",
                "decision": {
                    "behavior": "allow",
                    "updatedInput": {"path": "/tmp/safe"},
                    "message": "Auto-allowed for temp directory",
                }
            }
        })
        output = HookOutput.from_raw(json_output, "", 0)
        assert output.decision == HookDecision.ALLOW
        assert output.reason == "Auto-allowed for temp directory"

    def test_hook_specific_output_post_tool_use(self):
        """JSON with hookSpecificOutput for PostToolUse should parse correctly."""
        json_output = json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "additionalContext": "Code quality check passed",
            }
        })
        output = HookOutput.from_raw(json_output, "", 0)
        assert output.additional_context == "Code quality check passed"

    def test_should_block_property(self):
        """should_block property should return True for DENY and BLOCK."""
        deny_output = HookOutput(decision=HookDecision.DENY)
        block_output = HookOutput(decision=HookDecision.BLOCK)
        allow_output = HookOutput(decision=HookDecision.ALLOW)

        assert deny_output.should_block is True
        assert block_output.should_block is True
        assert allow_output.should_block is False

    def test_should_allow_property(self):
        """should_allow property should return True for ALLOW and APPROVE."""
        allow_output = HookOutput(decision=HookDecision.ALLOW)
        approve_output = HookOutput(decision=HookDecision.APPROVE)
        deny_output = HookOutput(decision=HookDecision.DENY)

        assert allow_output.should_allow is True
        assert approve_output.should_allow is True
        assert deny_output.should_allow is False

    def test_should_ask_property(self):
        """should_ask property should return True for ASK."""
        ask_output = HookOutput(decision=HookDecision.ASK)
        allow_output = HookOutput(decision=HookDecision.ALLOW)

        assert ask_output.should_ask is True
        assert allow_output.should_ask is False


# ─────────────────────────────────────────────────────────────────────────────
# HookMatcher Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestHookMatcher:
    """Tests for HookMatcher pattern matching."""

    def test_empty_matcher_matches_all(self):
        """Empty matcher should match all tool names."""
        matcher = HookMatcher(matcher=None)
        assert matcher.matches("Bash") is True
        assert matcher.matches("Write") is True
        assert matcher.matches(None) is True

    def test_wildcard_matcher_matches_all(self):
        """Wildcard '*' matcher should match all tool names."""
        matcher = HookMatcher(matcher="*")
        assert matcher.matches("Bash") is True
        assert matcher.matches("Edit") is True
        assert matcher.matches("mcp__server__tool") is True

    def test_exact_match(self):
        """Exact tool name should match only that tool."""
        matcher = HookMatcher(matcher="Bash")
        assert matcher.matches("Bash") is True
        assert matcher.matches("bash") is False  # Case-sensitive
        assert matcher.matches("Write") is False

    def test_regex_match_alternation(self):
        """Regex alternation pattern should match multiple tools."""
        matcher = HookMatcher(matcher="Edit|Write")
        assert matcher.matches("Edit") is True
        assert matcher.matches("Write") is True
        assert matcher.matches("Bash") is False

    def test_regex_match_prefix(self):
        """Regex prefix pattern should match tools with that prefix."""
        matcher = HookMatcher(matcher="mcp__.*")
        assert matcher.matches("mcp__server__read") is True
        assert matcher.matches("mcp__other__write") is True
        assert matcher.matches("Bash") is False

    def test_regex_match_suffix(self):
        """Regex suffix pattern should match tools with that suffix."""
        matcher = HookMatcher(matcher=".*write$")
        assert matcher.matches("mcp__server__write") is True
        assert matcher.matches("write") is True
        assert matcher.matches("Write") is False  # Case matters

    def test_invalid_regex_returns_false(self):
        """Invalid regex should not crash and return False."""
        matcher = HookMatcher(matcher="[invalid")
        assert matcher.matches("Bash") is False

    def test_none_tool_name_matches(self):
        """None tool name should match any matcher."""
        matcher = HookMatcher(matcher="Bash")
        assert matcher.matches(None) is True


# ─────────────────────────────────────────────────────────────────────────────
# HooksConfig Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestHooksConfig:
    """Tests for HooksConfig loading and merging."""

    def test_load_empty_config(self, tmp_path):
        """Loading non-existent config should return empty config."""
        config = load_hooks_config(tmp_path / "nonexistent.json")
        assert len(config.hooks) == 0

    def test_load_invalid_json(self, tmp_path):
        """Loading invalid JSON should return empty config."""
        config_path = tmp_path / "hooks.json"
        config_path.write_text("not valid json {")
        config = load_hooks_config(config_path)
        assert len(config.hooks) == 0

    def test_load_valid_config(self, tmp_path):
        """Loading valid config should parse correctly."""
        config_path = tmp_path / "hooks.json"
        config_path.write_text(json.dumps({
            "hooks": {
                "PreToolUse": [
                    {
                        "matcher": "Bash",
                        "hooks": [
                            {"type": "command", "command": "echo test", "timeout": 30}
                        ]
                    }
                ]
            }
        }))
        config = load_hooks_config(config_path)
        assert "PreToolUse" in config.hooks
        assert len(config.hooks["PreToolUse"]) == 1
        assert config.hooks["PreToolUse"][0].matcher == "Bash"
        assert config.hooks["PreToolUse"][0].hooks[0].command == "echo test"
        assert config.hooks["PreToolUse"][0].hooks[0].timeout == 30

    def test_load_config_without_wrapper(self, tmp_path):
        """Loading config without 'hooks' wrapper should work."""
        config_path = tmp_path / "hooks.json"
        config_path.write_text(json.dumps({
            "PreToolUse": [
                {
                    "matcher": "*",
                    "hooks": [
                        {"type": "command", "command": "echo test"}
                    ]
                }
            ]
        }))
        config = load_hooks_config(config_path)
        assert "PreToolUse" in config.hooks

    def test_load_prompt_hook(self, tmp_path):
        """Loading prompt hook should parse correctly."""
        config_path = tmp_path / "hooks.json"
        config_path.write_text(json.dumps({
            "hooks": {
                "Stop": [
                    {
                        "hooks": [
                            {"type": "prompt", "prompt": "Should continue? $ARGUMENTS", "timeout": 20}
                        ]
                    }
                ]
            }
        }))
        config = load_hooks_config(config_path)
        assert "Stop" in config.hooks
        hook = config.hooks["Stop"][0].hooks[0]
        assert hook.type == "prompt"
        assert hook.prompt == "Should continue? $ARGUMENTS"
        assert hook.timeout == 20

    def test_prompt_hook_on_unsupported_event_skipped(self, tmp_path):
        """Prompt hook on unsupported event should be skipped."""
        config_path = tmp_path / "hooks.json"
        config_path.write_text(json.dumps({
            "hooks": {
                "SessionEnd": [
                    {
                        "hooks": [
                            {"type": "prompt", "prompt": "Test prompt"}
                        ]
                    }
                ]
            }
        }))
        config = load_hooks_config(config_path)
        # SessionEnd doesn't support prompt hooks, should be empty
        assert "SessionEnd" not in config.hooks or len(config.hooks.get("SessionEnd", [])) == 0

    def test_unknown_event_skipped(self, tmp_path):
        """Unknown event names should be skipped with warning."""
        config_path = tmp_path / "hooks.json"
        config_path.write_text(json.dumps({
            "hooks": {
                "UnknownEvent": [
                    {
                        "hooks": [
                            {"type": "command", "command": "echo test"}
                        ]
                    }
                ]
            }
        }))
        config = load_hooks_config(config_path)
        assert "UnknownEvent" not in config.hooks

    def test_get_hooks_for_event_with_tool_name(self, tmp_path):
        """get_hooks_for_event should filter by tool name."""
        config_path = tmp_path / "hooks.json"
        config_path.write_text(json.dumps({
            "hooks": {
                "PreToolUse": [
                    {
                        "matcher": "Bash",
                        "hooks": [{"type": "command", "command": "bash-hook"}]
                    },
                    {
                        "matcher": "Write",
                        "hooks": [{"type": "command", "command": "write-hook"}]
                    }
                ]
            }
        }))
        config = load_hooks_config(config_path)

        bash_hooks = config.get_hooks_for_event(HookEvent.PRE_TOOL_USE, "Bash")
        assert len(bash_hooks) == 1
        assert bash_hooks[0].command == "bash-hook"

        write_hooks = config.get_hooks_for_event(HookEvent.PRE_TOOL_USE, "Write")
        assert len(write_hooks) == 1
        assert write_hooks[0].command == "write-hook"

    def test_merge_configs(self, tmp_path):
        """Merging configs should combine hooks from both."""
        config1 = HooksConfig(hooks={
            "PreToolUse": [HookMatcher(matcher="Bash", hooks=[
                HookDefinition(type="command", command="hook1")
            ])]
        })
        config2 = HooksConfig(hooks={
            "PreToolUse": [HookMatcher(matcher="Write", hooks=[
                HookDefinition(type="command", command="hook2")
            ])],
            "PostToolUse": [HookMatcher(matcher="*", hooks=[
                HookDefinition(type="command", command="hook3")
            ])]
        })

        merged = config1.merge_with(config2)
        assert len(merged.hooks["PreToolUse"]) == 2
        assert "PostToolUse" in merged.hooks

    def test_get_merged_hooks_config_from_all_sources(self, tmp_path, monkeypatch):
        """get_merged_hooks_config should merge global, project, and local configs."""
        monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)

        # Create global config
        global_dir = tmp_path / ".ripperdoc"
        global_dir.mkdir(parents=True)
        (global_dir / "hooks.json").write_text(json.dumps({
            "hooks": {
                "PreToolUse": [{"matcher": "*", "hooks": [
                    {"type": "command", "command": "global-hook"}
                ]}]
            }
        }))

        # Create project config
        project_path = tmp_path / "project"
        project_dir = project_path / ".ripperdoc"
        project_dir.mkdir(parents=True)
        (project_dir / "hooks.json").write_text(json.dumps({
            "hooks": {
                "PostToolUse": [{"matcher": "*", "hooks": [
                    {"type": "command", "command": "project-hook"}
                ]}]
            }
        }))

        # Create local config
        (project_dir / "hooks.local.json").write_text(json.dumps({
            "hooks": {
                "SessionStart": [{"hooks": [
                    {"type": "command", "command": "local-hook"}
                ]}]
            }
        }))

        config = get_merged_hooks_config(project_path)
        assert "PreToolUse" in config.hooks  # From global
        assert "PostToolUse" in config.hooks  # From project
        assert "SessionStart" in config.hooks  # From local


# ─────────────────────────────────────────────────────────────────────────────
# HookDefinition Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestHookDefinition:
    """Tests for HookDefinition model."""

    def test_default_values(self):
        """Default values should be set correctly."""
        hook = HookDefinition()
        assert hook.type == "command"
        assert hook.command is None
        assert hook.prompt is None
        assert hook.timeout == DEFAULT_HOOK_TIMEOUT

    def test_command_hook(self):
        """Command hook should be identified correctly."""
        hook = HookDefinition(type="command", command="echo test")
        assert hook.is_command_hook() is True
        assert hook.is_prompt_hook() is False

    def test_prompt_hook(self):
        """Prompt hook should be identified correctly."""
        hook = HookDefinition(type="prompt", prompt="Evaluate this: $ARGUMENTS")
        assert hook.is_command_hook() is False
        assert hook.is_prompt_hook() is True

    def test_incomplete_command_hook(self):
        """Command hook without command should not be valid."""
        hook = HookDefinition(type="command", command=None)
        assert hook.is_command_hook() is False

    def test_incomplete_prompt_hook(self):
        """Prompt hook without prompt should not be valid."""
        hook = HookDefinition(type="prompt", prompt=None)
        assert hook.is_prompt_hook() is False


# ─────────────────────────────────────────────────────────────────────────────
# HookExecutor Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestHookExecutor:
    """Tests for HookExecutor command and prompt execution."""

    def test_build_env_with_project_dir(self, tmp_path):
        """Environment should include RIPPERDOC_PROJECT_DIR."""
        executor = HookExecutor(
            project_dir=tmp_path,
            session_id="test-session",
            transcript_path="/path/to/transcript.json",
        )
        input_data = PreToolUseInput(tool_name="Bash", tool_input={})
        env = executor._build_env(input_data)

        assert env["RIPPERDOC_PROJECT_DIR"] == str(tmp_path)
        assert env["RIPPERDOC_SESSION_ID"] == "test-session"
        assert env["RIPPERDOC_TRANSCRIPT_PATH"] == "/path/to/transcript.json"

    def test_build_env_for_session_start(self, tmp_path):
        """SessionStart should include RIPPERDOC_ENV_FILE."""
        executor = HookExecutor(project_dir=tmp_path)
        input_data = SessionStartInput(source="startup")
        env = executor._build_env(input_data)

        assert "RIPPERDOC_ENV_FILE" in env
        assert os.path.exists(env["RIPPERDOC_ENV_FILE"])

        # Cleanup
        executor.cleanup_env_file()

    def test_expand_command_with_project_dir(self, tmp_path):
        """Command should expand $RIPPERDOC_PROJECT_DIR."""
        executor = HookExecutor(project_dir=tmp_path)
        command = "python $RIPPERDOC_PROJECT_DIR/script.py"
        expanded = executor._expand_command(command)

        assert expanded == f"python {tmp_path}/script.py"

    def test_expand_command_with_braces(self, tmp_path):
        """Command should expand ${RIPPERDOC_PROJECT_DIR}."""
        executor = HookExecutor(project_dir=tmp_path)
        command = "python ${RIPPERDOC_PROJECT_DIR}/script.py"
        expanded = executor._expand_command(command)

        assert expanded == f"python {tmp_path}/script.py"

    def test_expand_prompt_with_arguments(self):
        """Prompt should expand $ARGUMENTS with JSON."""
        executor = HookExecutor()
        input_data = PreToolUseInput(tool_name="Bash", tool_input={"command": "ls"})
        prompt = "Evaluate this: $ARGUMENTS"
        expanded = executor._expand_prompt(prompt, input_data)

        assert "Bash" in expanded
        assert "ls" in expanded

    def test_execute_sync_command_success(self, tmp_path):
        """Sync command execution should return correct output."""
        executor = HookExecutor(project_dir=tmp_path)
        hook = HookDefinition(type="command", command="echo success")
        input_data = PreToolUseInput(tool_name="Test", tool_input={})

        output = executor.execute_sync(hook, input_data)

        assert output.exit_code == 0
        assert output.raw_output == "success" or output.additional_context == "success"

    def test_execute_sync_command_exit_code_2(self, tmp_path):
        """Sync command with exit code 2 should block."""
        executor = HookExecutor(project_dir=tmp_path)
        hook = HookDefinition(type="command", command="exit 2")
        input_data = PreToolUseInput(tool_name="Test", tool_input={})

        output = executor.execute_sync(hook, input_data)

        assert output.exit_code == 2
        assert output.decision == HookDecision.DENY

    def test_execute_sync_command_json_output(self, tmp_path):
        """Sync command returning JSON should be parsed."""
        executor = HookExecutor(project_dir=tmp_path)
        json_output = json.dumps({"decision": "allow", "reason": "Approved"})
        hook = HookDefinition(type="command", command=f"echo '{json_output}'")
        input_data = PreToolUseInput(tool_name="Test", tool_input={})

        output = executor.execute_sync(hook, input_data)

        assert output.decision == HookDecision.ALLOW
        assert output.reason == "Approved"

    def test_execute_sync_command_timeout(self, tmp_path):
        """Sync command should timeout after specified duration."""
        executor = HookExecutor(project_dir=tmp_path)
        hook = HookDefinition(type="command", command="sleep 10", timeout=1)
        input_data = PreToolUseInput(tool_name="Test", tool_input={})

        output = executor.execute_sync(hook, input_data)

        assert output.timed_out is True
        assert output.error == "Hook timed out"

    def test_execute_sync_prompt_hook_skipped(self, tmp_path):
        """Prompt hooks should be skipped in sync mode."""
        executor = HookExecutor(project_dir=tmp_path)
        hook = HookDefinition(type="prompt", prompt="Test prompt")
        input_data = StopInput()

        output = executor.execute_sync(hook, input_data)

        # Should return empty output, not error
        assert output.error is None

    @pytest.mark.asyncio
    async def test_execute_async_command_success(self, tmp_path):
        """Async command execution should return correct output."""
        executor = HookExecutor(project_dir=tmp_path)
        hook = HookDefinition(type="command", command="echo async_success")
        input_data = PreToolUseInput(tool_name="Test", tool_input={})

        output = await executor.execute_async(hook, input_data)

        assert output.exit_code == 0

    @pytest.mark.asyncio
    async def test_execute_async_command_timeout(self, tmp_path):
        """Async command should timeout after specified duration."""
        executor = HookExecutor(project_dir=tmp_path)
        hook = HookDefinition(type="command", command="sleep 10", timeout=1)
        input_data = PreToolUseInput(tool_name="Test", tool_input={})

        output = await executor.execute_async(hook, input_data)

        assert output.timed_out is True

    @pytest.mark.asyncio
    async def test_execute_async_prompt_with_callback(self, tmp_path):
        """Async prompt execution should use LLM callback."""
        async def mock_llm_callback(prompt: str) -> str:
            return json.dumps({"decision": "allow", "reason": "LLM approved"})

        executor = HookExecutor(project_dir=tmp_path, llm_callback=mock_llm_callback)
        hook = HookDefinition(type="prompt", prompt="Should proceed? $ARGUMENTS")
        input_data = StopInput()

        output = await executor.execute_async(hook, input_data)

        assert output.decision == HookDecision.ALLOW
        assert output.reason == "LLM approved"

    @pytest.mark.asyncio
    async def test_execute_async_prompt_without_callback(self, tmp_path):
        """Async prompt execution without callback should skip."""
        executor = HookExecutor(project_dir=tmp_path, llm_callback=None)
        hook = HookDefinition(type="prompt", prompt="Should proceed?")
        input_data = StopInput()

        output = await executor.execute_async(hook, input_data)

        # Should not error, just return empty output
        assert output.error is None

    @pytest.mark.asyncio
    async def test_execute_async_prompt_timeout(self, tmp_path):
        """Async prompt execution should timeout."""
        async def slow_callback(prompt: str) -> str:
            await asyncio.sleep(10)
            return "{}"

        executor = HookExecutor(project_dir=tmp_path, llm_callback=slow_callback)
        hook = HookDefinition(type="prompt", prompt="Test", timeout=1)
        input_data = StopInput()

        output = await executor.execute_async(hook, input_data)

        assert output.timed_out is True

    @pytest.mark.asyncio
    async def test_execute_hooks_async_multiple(self, tmp_path):
        """Multiple hooks should execute in sequence."""
        executor = HookExecutor(project_dir=tmp_path)
        hooks = [
            HookDefinition(type="command", command="echo hook1"),
            HookDefinition(type="command", command="echo hook2"),
        ]
        input_data = PreToolUseInput(tool_name="Test", tool_input={})

        outputs = await executor.execute_hooks_async(hooks, input_data)

        assert len(outputs) == 2
        assert all(o.exit_code == 0 for o in outputs)

    def test_execute_hooks_sync_multiple(self, tmp_path):
        """Multiple hooks should execute in sequence (sync)."""
        executor = HookExecutor(project_dir=tmp_path)
        hooks = [
            HookDefinition(type="command", command="echo hook1"),
            HookDefinition(type="command", command="echo hook2"),
        ]
        input_data = PreToolUseInput(tool_name="Test", tool_input={})

        outputs = executor.execute_hooks_sync(hooks, input_data)

        assert len(outputs) == 2

    def test_env_file_lifecycle(self, tmp_path):
        """Environment file should be created and cleaned up properly."""
        executor = HookExecutor(project_dir=tmp_path)

        # Get env file (creates it)
        env_file = executor._get_env_file()
        assert env_file.exists()
        assert json.loads(env_file.read_text()) == {}

        # Cleanup
        executor.cleanup_env_file()
        assert not env_file.exists()

    def test_load_env_from_file(self, tmp_path):
        """Environment variables should be loaded from file."""
        executor = HookExecutor(project_dir=tmp_path)
        env_file = executor._get_env_file()
        env_file.write_text(json.dumps({"MY_VAR": "my_value", "ANOTHER": 123}))

        loaded = executor.load_env_from_file()

        assert loaded["MY_VAR"] == "my_value"
        assert loaded["ANOTHER"] == "123"  # Should be string

        executor.cleanup_env_file()


# ─────────────────────────────────────────────────────────────────────────────
# HookManager Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestHookManager:
    """Tests for HookManager event handling."""

    def test_init_with_defaults(self):
        """Manager should initialize with defaults."""
        manager = HookManager()
        assert manager.project_dir is None
        assert manager.session_id is None
        assert manager.permission_mode == "default"

    def test_lazy_config_loading(self, tmp_path, monkeypatch):
        """Config should be loaded lazily on first access."""
        monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)
        manager = HookManager(project_dir=tmp_path)

        # Config should not be loaded yet
        assert manager._config is None

        # Access config
        _ = manager.config

        # Config should now be loaded
        assert manager._config is not None

    def test_reload_config(self, tmp_path, monkeypatch):
        """reload_config should clear cached config."""
        monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)
        manager = HookManager(project_dir=tmp_path)

        # Load config
        _ = manager.config
        assert manager._config is not None

        # Reload
        manager.reload_config()
        assert manager._config is None

    def test_set_project_dir_clears_cache(self, tmp_path, monkeypatch):
        """Setting project dir should clear config and executor cache."""
        monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)
        manager = HookManager(project_dir=tmp_path)

        # Initialize
        _ = manager.config
        _ = manager.executor

        # Set new project dir
        new_path = tmp_path / "new_project"
        new_path.mkdir()
        manager.set_project_dir(new_path)

        assert manager._config is None
        assert manager._executor is None

    def test_run_pre_tool_use_no_hooks(self, tmp_path, monkeypatch):
        """run_pre_tool_use with no hooks should return empty result."""
        monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)
        manager = HookManager(project_dir=tmp_path)

        result = manager.run_pre_tool_use("Bash", {"command": "ls"})

        assert len(result.outputs) == 0
        assert result.should_block is False
        assert result.should_allow is False

    def test_run_pre_tool_use_with_hooks(self, tmp_path, monkeypatch):
        """run_pre_tool_use with hooks should execute them."""
        monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)

        # Create hooks config
        config_dir = tmp_path / ".ripperdoc"
        config_dir.mkdir(parents=True)
        json_output = json.dumps({"decision": "allow"})
        (config_dir / "hooks.local.json").write_text(json.dumps({
            "hooks": {
                "PreToolUse": [{
                    "matcher": "Bash",
                    "hooks": [{"type": "command", "command": f"echo '{json_output}'"}]
                }]
            }
        }))

        manager = HookManager(project_dir=tmp_path)
        result = manager.run_pre_tool_use("Bash", {"command": "ls"})

        assert result.should_allow is True

    @pytest.mark.asyncio
    async def test_run_pre_tool_use_async(self, tmp_path, monkeypatch):
        """run_pre_tool_use_async should execute hooks asynchronously."""
        monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)

        config_dir = tmp_path / ".ripperdoc"
        config_dir.mkdir(parents=True)
        (config_dir / "hooks.local.json").write_text(json.dumps({
            "hooks": {
                "PreToolUse": [{
                    "matcher": "*",
                    "hooks": [{"type": "command", "command": "echo test"}]
                }]
            }
        }))

        manager = HookManager(project_dir=tmp_path)
        result = await manager.run_pre_tool_use_async("Bash", {"command": "ls"})

        assert len(result.outputs) == 1

    def test_run_user_prompt_submit(self, tmp_path, monkeypatch):
        """run_user_prompt_submit should execute hooks."""
        monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)

        config_dir = tmp_path / ".ripperdoc"
        config_dir.mkdir(parents=True)
        (config_dir / "hooks.local.json").write_text(json.dumps({
            "hooks": {
                "UserPromptSubmit": [{
                    "hooks": [{"type": "command", "command": "echo checked"}]
                }]
            }
        }))

        manager = HookManager(project_dir=tmp_path)
        result = manager.run_user_prompt_submit("Hello world")

        assert len(result.outputs) == 1

    def test_run_notification(self, tmp_path, monkeypatch):
        """run_notification should execute hooks."""
        monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)

        config_dir = tmp_path / ".ripperdoc"
        config_dir.mkdir(parents=True)
        (config_dir / "hooks.local.json").write_text(json.dumps({
            "hooks": {
                "Notification": [{
                    "hooks": [{"type": "command", "command": "echo notified"}]
                }]
            }
        }))

        manager = HookManager(project_dir=tmp_path)
        result = manager.run_notification("Test message", "info")

        assert len(result.outputs) == 1

    def test_run_session_start(self, tmp_path, monkeypatch):
        """run_session_start should execute hooks."""
        monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)

        config_dir = tmp_path / ".ripperdoc"
        config_dir.mkdir(parents=True)
        (config_dir / "hooks.local.json").write_text(json.dumps({
            "hooks": {
                "SessionStart": [{
                    "hooks": [{"type": "command", "command": "echo started"}]
                }]
            }
        }))

        manager = HookManager(project_dir=tmp_path)
        result = manager.run_session_start("startup")

        assert len(result.outputs) == 1
        manager.cleanup()

    def test_run_session_end(self, tmp_path, monkeypatch):
        """run_session_end should execute hooks."""
        monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)

        config_dir = tmp_path / ".ripperdoc"
        config_dir.mkdir(parents=True)
        (config_dir / "hooks.local.json").write_text(json.dumps({
            "hooks": {
                "SessionEnd": [{
                    "hooks": [{"type": "command", "command": "echo ended"}]
                }]
            }
        }))

        manager = HookManager(project_dir=tmp_path)
        result = manager.run_session_end("other", duration_seconds=100.0, message_count=10)

        assert len(result.outputs) == 1

    def test_run_stop(self, tmp_path, monkeypatch):
        """run_stop should execute hooks."""
        monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)

        config_dir = tmp_path / ".ripperdoc"
        config_dir.mkdir(parents=True)
        (config_dir / "hooks.local.json").write_text(json.dumps({
            "hooks": {
                "Stop": [{
                    "hooks": [{"type": "command", "command": "echo stopped"}]
                }]
            }
        }))

        manager = HookManager(project_dir=tmp_path)
        result = manager.run_stop(stop_hook_active=False, reason="end_turn")

        assert len(result.outputs) == 1

    def test_cleanup(self, tmp_path, monkeypatch):
        """cleanup should clean up executor resources."""
        monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)

        manager = HookManager(project_dir=tmp_path)
        # Initialize executor
        _ = manager.executor

        # Create env file
        manager.executor._get_env_file()
        assert manager.executor._env_file is not None

        # Cleanup
        manager.cleanup()
        assert manager.executor._env_file is None


# ─────────────────────────────────────────────────────────────────────────────
# HookResult Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestHookResult:
    """Tests for HookResult aggregation."""

    def test_empty_outputs(self):
        """Empty outputs should not block or allow."""
        result = HookResult([])
        assert result.should_block is False
        assert result.should_allow is False
        assert result.should_ask is False
        assert result.should_continue is True

    def test_should_block_with_deny(self):
        """Result should block if any output has DENY."""
        outputs = [
            HookOutput(),
            HookOutput(decision=HookDecision.DENY, reason="Blocked"),
        ]
        result = HookResult(outputs)
        assert result.should_block is True
        assert result.block_reason == "Blocked"

    def test_should_block_with_block(self):
        """Result should block if any output has BLOCK."""
        outputs = [
            HookOutput(decision=HookDecision.BLOCK, reason="Stop blocked"),
        ]
        result = HookResult(outputs)
        assert result.should_block is True

    def test_should_allow(self):
        """Result should allow if any output has ALLOW."""
        outputs = [
            HookOutput(decision=HookDecision.ALLOW),
        ]
        result = HookResult(outputs)
        assert result.should_allow is True

    def test_should_ask(self):
        """Result should ask if any output has ASK."""
        outputs = [
            HookOutput(decision=HookDecision.ASK),
        ]
        result = HookResult(outputs)
        assert result.should_ask is True

    def test_should_continue(self):
        """Result should not continue if any output has continue=False."""
        outputs = [
            HookOutput(continue_execution=True),
            HookOutput(continue_execution=False),
        ]
        result = HookResult(outputs)
        assert result.should_continue is False

    def test_additional_context_combined(self):
        """Additional context should be combined from all outputs."""
        outputs = [
            HookOutput(additional_context="Context 1"),
            HookOutput(additional_context="Context 2"),
        ]
        result = HookResult(outputs)
        assert result.additional_context == "Context 1\nContext 2"

    def test_stop_reason(self):
        """Stop reason should be from first output with one."""
        outputs = [
            HookOutput(),
            HookOutput(stop_reason="Task completed"),
        ]
        result = HookResult(outputs)
        assert result.stop_reason == "Task completed"

    def test_system_message(self):
        """System message should be from first output with one."""
        outputs = [
            HookOutput(system_message="Warning"),
            HookOutput(system_message="Another warning"),
        ]
        result = HookResult(outputs)
        assert result.system_message == "Warning"

    def test_updated_input(self):
        """Updated input should be from first output with one."""
        outputs = [
            HookOutput(hook_specific_output=PreToolUseHookOutput(
                updatedInput={"command": "modified"}
            )),
        ]
        result = HookResult(outputs)
        assert result.updated_input == {"command": "modified"}

    def test_has_errors(self):
        """has_errors should return True if any output has error."""
        outputs = [
            HookOutput(),
            HookOutput(error="Something went wrong"),
        ]
        result = HookResult(outputs)
        assert result.has_errors is True
        assert result.errors == ["Something went wrong"]


# ─────────────────────────────────────────────────────────────────────────────
# Hook Input Types Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestHookInputTypes:
    """Tests for various hook input types."""

    def test_pre_tool_use_input(self):
        """PreToolUseInput should have correct fields."""
        input_data = PreToolUseInput(
            tool_name="Bash",
            tool_input={"command": "ls -la"},
            tool_use_id="toolu_123",
            session_id="session_456",
        )
        assert input_data.hook_event_name == "PreToolUse"
        assert input_data.tool_name == "Bash"
        assert input_data.tool_input == {"command": "ls -la"}
        assert input_data.tool_use_id == "toolu_123"

    def test_permission_request_input(self):
        """PermissionRequestInput should have correct fields."""
        input_data = PermissionRequestInput(
            tool_name="Write",
            tool_input={"file_path": "/tmp/test.txt"},
        )
        assert input_data.hook_event_name == "PermissionRequest"
        assert input_data.tool_name == "Write"

    def test_post_tool_use_input(self):
        """PostToolUseInput should have correct fields."""
        input_data = PostToolUseInput(
            tool_name="Bash",
            tool_input={"command": "ls"},
            tool_response={"output": "file1\nfile2"},
        )
        assert input_data.hook_event_name == "PostToolUse"
        assert input_data.tool_response == {"output": "file1\nfile2"}

    def test_user_prompt_submit_input(self):
        """UserPromptSubmitInput should have correct fields."""
        input_data = UserPromptSubmitInput(prompt="Help me write code")
        assert input_data.hook_event_name == "UserPromptSubmit"
        assert input_data.prompt == "Help me write code"

    def test_notification_input(self):
        """NotificationInput should have correct fields."""
        input_data = NotificationInput(
            message="Permission required",
            notification_type="permission_prompt",
        )
        assert input_data.hook_event_name == "Notification"
        assert input_data.notification_type == "permission_prompt"

    def test_stop_input(self):
        """StopInput should have correct fields."""
        input_data = StopInput(
            stop_hook_active=True,
            reason="end_turn",
            stop_sequence=None,
        )
        assert input_data.hook_event_name == "Stop"
        assert input_data.stop_hook_active is True

    def test_subagent_stop_input(self):
        """SubagentStopInput should have correct fields."""
        input_data = SubagentStopInput(stop_hook_active=False)
        assert input_data.hook_event_name == "SubagentStop"

    def test_pre_compact_input(self):
        """PreCompactInput should have correct fields."""
        input_data = PreCompactInput(
            trigger="manual",
            custom_instructions="Keep function signatures",
        )
        assert input_data.hook_event_name == "PreCompact"
        assert input_data.trigger == "manual"

    def test_session_start_input(self):
        """SessionStartInput should have correct fields."""
        input_data = SessionStartInput(source="startup")
        assert input_data.hook_event_name == "SessionStart"
        assert input_data.source == "startup"

    def test_session_end_input(self):
        """SessionEndInput should have correct fields."""
        input_data = SessionEndInput(
            reason="logout",
            duration_seconds=3600.5,
            message_count=42,
        )
        assert input_data.hook_event_name == "SessionEnd"
        assert input_data.duration_seconds == 3600.5
        assert input_data.message_count == 42


# ─────────────────────────────────────────────────────────────────────────────
# Integration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestHooksIntegration:
    """Integration tests for the hooks system."""

    def test_hook_receives_correct_input_via_stdin(self, tmp_path, monkeypatch):
        """Hook should receive JSON input via stdin."""
        monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)

        # Create a hook script that echoes stdin
        script_path = tmp_path / "echo_input.py"
        script_path.write_text("""
import sys
import json
data = json.load(sys.stdin)
print(json.dumps({"received_tool": data.get("tool_name")}))
""")

        config_dir = tmp_path / ".ripperdoc"
        config_dir.mkdir(parents=True)
        (config_dir / "hooks.local.json").write_text(json.dumps({
            "hooks": {
                "PreToolUse": [{
                    "matcher": "*",
                    "hooks": [{"type": "command", "command": f"{sys.executable} {script_path}"}]
                }]
            }
        }))

        manager = HookManager(project_dir=tmp_path)
        result = manager.run_pre_tool_use("TestTool", {"arg": "value"})

        # The hook should have received the tool name
        assert len(result.outputs) == 1

    def test_hook_env_vars_available(self, tmp_path, monkeypatch):
        """Hook should have access to environment variables."""
        monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)

        # Create a hook script that checks env vars
        script_path = tmp_path / "check_env.py"
        script_path.write_text("""
import os
import json
result = {
    "project_dir": os.environ.get("RIPPERDOC_PROJECT_DIR", ""),
    "session_id": os.environ.get("RIPPERDOC_SESSION_ID", ""),
}
print(json.dumps(result))
""")

        config_dir = tmp_path / ".ripperdoc"
        config_dir.mkdir(parents=True)
        (config_dir / "hooks.local.json").write_text(json.dumps({
            "hooks": {
                "PreToolUse": [{
                    "hooks": [{"type": "command", "command": f"{sys.executable} {script_path}"}]
                }]
            }
        }))

        manager = HookManager(
            project_dir=tmp_path,
            session_id="test-session-123",
        )
        result = manager.run_pre_tool_use("Bash", {})

        assert len(result.outputs) == 1
        assert result.outputs[0].exit_code == 0

    def test_blocking_hook_prevents_execution(self, tmp_path, monkeypatch):
        """Hook with exit code 2 should block execution."""
        monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)

        config_dir = tmp_path / ".ripperdoc"
        config_dir.mkdir(parents=True)
        (config_dir / "hooks.local.json").write_text(json.dumps({
            "hooks": {
                "PreToolUse": [{
                    "matcher": "Bash",
                    "hooks": [{"type": "command", "command": "echo 'Dangerous command' >&2; exit 2"}]
                }]
            }
        }))

        manager = HookManager(project_dir=tmp_path)
        result = manager.run_pre_tool_use("Bash", {"command": "rm -rf /"})

        assert result.should_block is True
        assert "Dangerous command" in result.block_reason

    def test_allowing_hook_bypasses_permission(self, tmp_path, monkeypatch):
        """Hook with 'allow' decision should bypass permission system."""
        monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)

        json_output = json.dumps({"decision": "allow", "reason": "Trusted operation"})
        config_dir = tmp_path / ".ripperdoc"
        config_dir.mkdir(parents=True)
        (config_dir / "hooks.local.json").write_text(json.dumps({
            "hooks": {
                "PreToolUse": [{
                    "matcher": "*",
                    "hooks": [{"type": "command", "command": f"echo '{json_output}'"}]
                }]
            }
        }))

        manager = HookManager(project_dir=tmp_path)
        result = manager.run_pre_tool_use("Bash", {"command": "ls"})

        assert result.should_allow is True

    def test_hook_can_modify_tool_input(self, tmp_path, monkeypatch):
        """Hook should be able to modify tool input via updatedInput."""
        monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)

        json_output = json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "updatedInput": {"command": "ls -la --color=never"}
            }
        })
        config_dir = tmp_path / ".ripperdoc"
        config_dir.mkdir(parents=True)
        (config_dir / "hooks.local.json").write_text(json.dumps({
            "hooks": {
                "PreToolUse": [{
                    "matcher": "Bash",
                    "hooks": [{"type": "command", "command": f"echo '{json_output}'"}]
                }]
            }
        }))

        manager = HookManager(project_dir=tmp_path)
        result = manager.run_pre_tool_use("Bash", {"command": "ls -la"})

        assert result.updated_input == {"command": "ls -la --color=never"}

    def test_multiple_hooks_execute_in_order(self, tmp_path, monkeypatch):
        """Multiple hooks should execute in sequence."""
        monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)

        config_dir = tmp_path / ".ripperdoc"
        config_dir.mkdir(parents=True)
        (config_dir / "hooks.local.json").write_text(json.dumps({
            "hooks": {
                "PreToolUse": [{
                    "matcher": "*",
                    "hooks": [
                        {"type": "command", "command": "echo hook1"},
                        {"type": "command", "command": "echo hook2"},
                        {"type": "command", "command": "echo hook3"},
                    ]
                }]
            }
        }))

        manager = HookManager(project_dir=tmp_path)
        result = manager.run_pre_tool_use("Bash", {})

        assert len(result.outputs) == 3

    def test_regex_matcher_in_config(self, tmp_path, monkeypatch):
        """Regex matcher should work in config."""
        monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)

        config_dir = tmp_path / ".ripperdoc"
        config_dir.mkdir(parents=True)
        (config_dir / "hooks.local.json").write_text(json.dumps({
            "hooks": {
                "PreToolUse": [{
                    "matcher": "Edit|Write",
                    "hooks": [{"type": "command", "command": "echo file_operation"}]
                }]
            }
        }))

        manager = HookManager(project_dir=tmp_path)

        # Should match Edit
        result1 = manager.run_pre_tool_use("Edit", {})
        assert len(result1.outputs) == 1

        # Should match Write
        result2 = manager.run_pre_tool_use("Write", {})
        assert len(result2.outputs) == 1

        # Should not match Bash
        result3 = manager.run_pre_tool_use("Bash", {})
        assert len(result3.outputs) == 0


# ─────────────────────────────────────────────────────────────────────────────
# HookInterceptor Tests
# ─────────────────────────────────────────────────────────────────────────────


from ripperdoc.core.hooks.integration import (
    HookInterceptor,
    check_pre_tool_use,
    check_pre_tool_use_async,
    run_post_tool_use,
    run_post_tool_use_async,
    check_user_prompt,
    check_user_prompt_async,
    notify_session_start,
    notify_session_end,
    check_stop,
    check_stop_async,
)


class TestHookInterceptor:
    """Tests for HookInterceptor integration helper."""

    def test_init_with_custom_manager(self, tmp_path, monkeypatch):
        """Interceptor should use provided manager."""
        monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)
        manager = HookManager(project_dir=tmp_path)
        interceptor = HookInterceptor(manager=manager)
        assert interceptor.manager is manager

    def test_init_with_default_manager(self):
        """Interceptor should use global manager by default."""
        interceptor = HookInterceptor()
        assert interceptor.manager is not None

    def test_check_pre_tool_use_allow(self, tmp_path, monkeypatch):
        """check_pre_tool_use should return True when allowed."""
        monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)
        manager = HookManager(project_dir=tmp_path)
        interceptor = HookInterceptor(manager=manager)

        should_proceed, block_reason, context = interceptor.check_pre_tool_use("Bash", {})

        assert should_proceed is True
        assert block_reason is None

    def test_check_pre_tool_use_block(self, tmp_path, monkeypatch):
        """check_pre_tool_use should return False when blocked."""
        monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)

        config_dir = tmp_path / ".ripperdoc"
        config_dir.mkdir(parents=True)
        (config_dir / "hooks.local.json").write_text(json.dumps({
            "hooks": {
                "PreToolUse": [{
                    "matcher": "*",
                    "hooks": [{"type": "command", "command": "echo 'Blocked' >&2; exit 2"}]
                }]
            }
        }))

        manager = HookManager(project_dir=tmp_path)
        interceptor = HookInterceptor(manager=manager)

        should_proceed, block_reason, context = interceptor.check_pre_tool_use("Bash", {})

        assert should_proceed is False
        assert "Blocked" in block_reason

    def test_check_pre_tool_use_ask(self, tmp_path, monkeypatch):
        """check_pre_tool_use should request confirmation for 'ask' decision."""
        monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)

        json_output = json.dumps({"decision": "ask", "reason": "Please confirm"})
        config_dir = tmp_path / ".ripperdoc"
        config_dir.mkdir(parents=True)
        (config_dir / "hooks.local.json").write_text(json.dumps({
            "hooks": {
                "PreToolUse": [{
                    "matcher": "*",
                    "hooks": [{"type": "command", "command": f"echo '{json_output}'"}]
                }]
            }
        }))

        manager = HookManager(project_dir=tmp_path)
        interceptor = HookInterceptor(manager=manager)

        should_proceed, block_reason, context = interceptor.check_pre_tool_use("Bash", {})

        assert should_proceed is False
        assert block_reason == "USER_CONFIRMATION_REQUIRED"

    @pytest.mark.asyncio
    async def test_check_pre_tool_use_async(self, tmp_path, monkeypatch):
        """check_pre_tool_use_async should work asynchronously."""
        monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)
        manager = HookManager(project_dir=tmp_path)
        interceptor = HookInterceptor(manager=manager)

        should_proceed, block_reason, context = await interceptor.check_pre_tool_use_async("Bash", {})

        assert should_proceed is True

    def test_run_post_tool_use(self, tmp_path, monkeypatch):
        """run_post_tool_use should execute post-tool hooks."""
        monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)

        config_dir = tmp_path / ".ripperdoc"
        config_dir.mkdir(parents=True)
        (config_dir / "hooks.local.json").write_text(json.dumps({
            "hooks": {
                "PostToolUse": [{
                    "matcher": "*",
                    "hooks": [{"type": "command", "command": "echo post_check"}]
                }]
            }
        }))

        manager = HookManager(project_dir=tmp_path)
        interceptor = HookInterceptor(manager=manager)

        should_continue, block_reason, context = interceptor.run_post_tool_use(
            "Bash", {"command": "ls"}, "output"
        )

        assert should_continue is True

    def test_run_post_tool_use_block(self, tmp_path, monkeypatch):
        """run_post_tool_use should block when hook returns block."""
        monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)

        json_output = json.dumps({"decision": "block", "reason": "Quality check failed"})
        config_dir = tmp_path / ".ripperdoc"
        config_dir.mkdir(parents=True)
        (config_dir / "hooks.local.json").write_text(json.dumps({
            "hooks": {
                "PostToolUse": [{
                    "matcher": "*",
                    "hooks": [{"type": "command", "command": f"echo '{json_output}'"}]
                }]
            }
        }))

        manager = HookManager(project_dir=tmp_path)
        interceptor = HookInterceptor(manager=manager)

        should_continue, block_reason, context = interceptor.run_post_tool_use(
            "Bash", {}, "output"
        )

        assert should_continue is False
        assert block_reason == "Quality check failed"

    @pytest.mark.asyncio
    async def test_run_post_tool_use_async(self, tmp_path, monkeypatch):
        """run_post_tool_use_async should work asynchronously."""
        monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)
        manager = HookManager(project_dir=tmp_path)
        interceptor = HookInterceptor(manager=manager)

        should_continue, block_reason, context = await interceptor.run_post_tool_use_async(
            "Bash", {}, "output"
        )

        assert should_continue is True

    def test_wrap_tool_execution_success(self, tmp_path, monkeypatch):
        """wrap_tool_execution should execute tool when allowed."""
        monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)
        manager = HookManager(project_dir=tmp_path)
        interceptor = HookInterceptor(manager=manager)

        def execute_fn():
            return "result"

        success, result, context = interceptor.wrap_tool_execution("Bash", {}, execute_fn)

        assert success is True
        assert result == "result"

    def test_wrap_tool_execution_blocked(self, tmp_path, monkeypatch):
        """wrap_tool_execution should not execute tool when blocked."""
        monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)

        config_dir = tmp_path / ".ripperdoc"
        config_dir.mkdir(parents=True)
        (config_dir / "hooks.local.json").write_text(json.dumps({
            "hooks": {
                "PreToolUse": [{
                    "matcher": "*",
                    "hooks": [{"type": "command", "command": "exit 2"}]
                }]
            }
        }))

        manager = HookManager(project_dir=tmp_path)
        interceptor = HookInterceptor(manager=manager)

        executed = False

        def execute_fn():
            nonlocal executed
            executed = True
            return "result"

        success, result, context = interceptor.wrap_tool_execution("Bash", {}, execute_fn)

        assert success is False
        assert executed is False

    def test_wrap_tool_execution_with_exception(self, tmp_path, monkeypatch):
        """wrap_tool_execution should handle exceptions."""
        monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)
        manager = HookManager(project_dir=tmp_path)
        interceptor = HookInterceptor(manager=manager)

        def execute_fn():
            raise ValueError("Test error")

        success, result, context = interceptor.wrap_tool_execution("Bash", {}, execute_fn)

        assert success is False
        assert "Test error" in result

    @pytest.mark.asyncio
    async def test_wrap_tool_execution_async(self, tmp_path, monkeypatch):
        """wrap_tool_execution_async should execute async tool."""
        monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)
        manager = HookManager(project_dir=tmp_path)
        interceptor = HookInterceptor(manager=manager)

        async def execute_fn():
            return "async_result"

        success, result, context = await interceptor.wrap_tool_execution_async(
            "Bash", {}, execute_fn
        )

        assert success is True
        assert result == "async_result"

    @pytest.mark.asyncio
    async def test_wrap_tool_execution_async_sync_fn(self, tmp_path, monkeypatch):
        """wrap_tool_execution_async should handle sync functions."""
        monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)
        manager = HookManager(project_dir=tmp_path)
        interceptor = HookInterceptor(manager=manager)

        def execute_fn():
            return "sync_result"

        success, result, context = await interceptor.wrap_tool_execution_async(
            "Bash", {}, execute_fn
        )

        assert success is True
        assert result == "sync_result"

    @pytest.mark.asyncio
    async def test_wrap_tool_execution_async_with_exception(self, tmp_path, monkeypatch):
        """wrap_tool_execution_async should handle exceptions."""
        monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)
        manager = HookManager(project_dir=tmp_path)
        interceptor = HookInterceptor(manager=manager)

        async def execute_fn():
            raise ValueError("Async error")

        success, result, context = await interceptor.wrap_tool_execution_async(
            "Bash", {}, execute_fn
        )

        assert success is False
        assert "Async error" in result


# ─────────────────────────────────────────────────────────────────────────────
# Integration Module Convenience Functions Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestIntegrationConvenienceFunctions:
    """Tests for global convenience functions in integration module."""

    def test_check_user_prompt_allow(self, tmp_path, monkeypatch):
        """check_user_prompt should allow by default."""
        monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)
        monkeypatch.setattr("ripperdoc.core.hooks.integration.hook_manager.project_dir", tmp_path)

        should_process, block_reason, context = check_user_prompt("Hello")

        assert should_process is True

    def test_check_user_prompt_block(self, tmp_path, monkeypatch):
        """check_user_prompt should block when hook blocks."""
        monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)

        config_dir = tmp_path / ".ripperdoc"
        config_dir.mkdir(parents=True)
        (config_dir / "hooks.local.json").write_text(json.dumps({
            "hooks": {
                "UserPromptSubmit": [{
                    "hooks": [{"type": "command", "command": "exit 2"}]
                }]
            }
        }))

        # Initialize a new manager for this test
        from ripperdoc.core.hooks.integration import hook_manager
        hook_manager.set_project_dir(tmp_path)
        hook_manager.reload_config()

        should_process, block_reason, context = check_user_prompt("Test prompt")

        assert should_process is False

    @pytest.mark.asyncio
    async def test_check_user_prompt_async(self, tmp_path, monkeypatch):
        """check_user_prompt_async should work asynchronously."""
        monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)

        from ripperdoc.core.hooks.integration import hook_manager
        hook_manager.set_project_dir(tmp_path)
        hook_manager.reload_config()

        should_process, block_reason, context = await check_user_prompt_async("Test")

        assert should_process is True

    def test_notify_session_start(self, tmp_path, monkeypatch):
        """notify_session_start should run session start hooks."""
        monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)

        from ripperdoc.core.hooks.integration import hook_manager
        hook_manager.set_project_dir(tmp_path)
        hook_manager.reload_config()

        context = notify_session_start("startup")

        # No hooks configured, so context should be None
        assert context is None

    def test_notify_session_end(self, tmp_path, monkeypatch):
        """notify_session_end should run session end hooks."""
        monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)

        from ripperdoc.core.hooks.integration import hook_manager
        hook_manager.set_project_dir(tmp_path)
        hook_manager.reload_config()

        context = notify_session_end("logout", duration_seconds=100.0, message_count=5)

        assert context is None

    def test_check_stop_allow(self, tmp_path, monkeypatch):
        """check_stop should allow stopping by default."""
        monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)

        from ripperdoc.core.hooks.integration import hook_manager
        hook_manager.set_project_dir(tmp_path)
        hook_manager.reload_config()

        should_stop, continue_reason = check_stop(reason="end_turn")

        assert should_stop is True
        assert continue_reason is None

    def test_check_stop_block(self, tmp_path, monkeypatch):
        """check_stop should block when hook blocks."""
        monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)

        json_output = json.dumps({"decision": "block", "reason": "Keep working"})
        config_dir = tmp_path / ".ripperdoc"
        config_dir.mkdir(parents=True)
        (config_dir / "hooks.local.json").write_text(json.dumps({
            "hooks": {
                "Stop": [{
                    "hooks": [{"type": "command", "command": f"echo '{json_output}'"}]
                }]
            }
        }))

        from ripperdoc.core.hooks.integration import hook_manager
        hook_manager.set_project_dir(tmp_path)
        hook_manager.reload_config()

        should_stop, continue_reason = check_stop(reason="end_turn")

        assert should_stop is False
        assert continue_reason == "Keep working"

    @pytest.mark.asyncio
    async def test_check_stop_async(self, tmp_path, monkeypatch):
        """check_stop_async should work asynchronously."""
        monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)

        from ripperdoc.core.hooks.integration import hook_manager
        hook_manager.set_project_dir(tmp_path)
        hook_manager.reload_config()

        should_stop, continue_reason = await check_stop_async(reason="end_turn")

        assert should_stop is True


# ─────────────────────────────────────────────────────────────────────────────
# Additional Manager Async Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestHookManagerAsyncMethods:
    """Additional tests for HookManager async methods."""

    @pytest.mark.asyncio
    async def test_run_permission_request_async(self, tmp_path, monkeypatch):
        """run_permission_request_async should execute hooks."""
        monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)

        config_dir = tmp_path / ".ripperdoc"
        config_dir.mkdir(parents=True)
        (config_dir / "hooks.local.json").write_text(json.dumps({
            "hooks": {
                "PermissionRequest": [{
                    "matcher": "*",
                    "hooks": [{"type": "command", "command": "echo permission_checked"}]
                }]
            }
        }))

        manager = HookManager(project_dir=tmp_path)
        result = await manager.run_permission_request_async("Bash", {"command": "ls"})

        assert len(result.outputs) == 1

    @pytest.mark.asyncio
    async def test_run_post_tool_use_async(self, tmp_path, monkeypatch):
        """run_post_tool_use_async should execute hooks."""
        monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)

        config_dir = tmp_path / ".ripperdoc"
        config_dir.mkdir(parents=True)
        (config_dir / "hooks.local.json").write_text(json.dumps({
            "hooks": {
                "PostToolUse": [{
                    "matcher": "*",
                    "hooks": [{"type": "command", "command": "echo post_tool"}]
                }]
            }
        }))

        manager = HookManager(project_dir=tmp_path)
        result = await manager.run_post_tool_use_async("Bash", {}, "output")

        assert len(result.outputs) == 1

    @pytest.mark.asyncio
    async def test_run_user_prompt_submit_async(self, tmp_path, monkeypatch):
        """run_user_prompt_submit_async should execute hooks."""
        monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)

        config_dir = tmp_path / ".ripperdoc"
        config_dir.mkdir(parents=True)
        (config_dir / "hooks.local.json").write_text(json.dumps({
            "hooks": {
                "UserPromptSubmit": [{
                    "hooks": [{"type": "command", "command": "echo prompt_checked"}]
                }]
            }
        }))

        manager = HookManager(project_dir=tmp_path)
        result = await manager.run_user_prompt_submit_async("Hello")

        assert len(result.outputs) == 1

    @pytest.mark.asyncio
    async def test_run_notification_async(self, tmp_path, monkeypatch):
        """run_notification_async should execute hooks."""
        monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)

        config_dir = tmp_path / ".ripperdoc"
        config_dir.mkdir(parents=True)
        (config_dir / "hooks.local.json").write_text(json.dumps({
            "hooks": {
                "Notification": [{
                    "hooks": [{"type": "command", "command": "echo notified"}]
                }]
            }
        }))

        manager = HookManager(project_dir=tmp_path)
        result = await manager.run_notification_async("Test", "info")

        assert len(result.outputs) == 1

    @pytest.mark.asyncio
    async def test_run_stop_async(self, tmp_path, monkeypatch):
        """run_stop_async should execute hooks."""
        monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)

        config_dir = tmp_path / ".ripperdoc"
        config_dir.mkdir(parents=True)
        (config_dir / "hooks.local.json").write_text(json.dumps({
            "hooks": {
                "Stop": [{
                    "hooks": [{"type": "command", "command": "echo stop_checked"}]
                }]
            }
        }))

        manager = HookManager(project_dir=tmp_path)
        result = await manager.run_stop_async(False, "end_turn")

        assert len(result.outputs) == 1

    @pytest.mark.asyncio
    async def test_run_subagent_stop_async(self, tmp_path, monkeypatch):
        """run_subagent_stop_async should execute hooks."""
        monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)

        config_dir = tmp_path / ".ripperdoc"
        config_dir.mkdir(parents=True)
        (config_dir / "hooks.local.json").write_text(json.dumps({
            "hooks": {
                "SubagentStop": [{
                    "hooks": [{"type": "command", "command": "echo subagent_stop"}]
                }]
            }
        }))

        manager = HookManager(project_dir=tmp_path)
        result = await manager.run_subagent_stop_async(False)

        assert len(result.outputs) == 1

    @pytest.mark.asyncio
    async def test_run_pre_compact_async(self, tmp_path, monkeypatch):
        """run_pre_compact_async should execute hooks."""
        monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)

        config_dir = tmp_path / ".ripperdoc"
        config_dir.mkdir(parents=True)
        (config_dir / "hooks.local.json").write_text(json.dumps({
            "hooks": {
                "PreCompact": [{
                    "hooks": [{"type": "command", "command": "echo pre_compact"}]
                }]
            }
        }))

        manager = HookManager(project_dir=tmp_path)
        result = await manager.run_pre_compact_async("manual", "Keep imports")

        assert len(result.outputs) == 1

    @pytest.mark.asyncio
    async def test_run_session_start_async(self, tmp_path, monkeypatch):
        """run_session_start_async should execute hooks."""
        monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)

        config_dir = tmp_path / ".ripperdoc"
        config_dir.mkdir(parents=True)
        (config_dir / "hooks.local.json").write_text(json.dumps({
            "hooks": {
                "SessionStart": [{
                    "hooks": [{"type": "command", "command": "echo session_start"}]
                }]
            }
        }))

        manager = HookManager(project_dir=tmp_path)
        result = await manager.run_session_start_async("startup")

        assert len(result.outputs) == 1
        manager.cleanup()

    @pytest.mark.asyncio
    async def test_run_session_end_async(self, tmp_path, monkeypatch):
        """run_session_end_async should execute hooks."""
        monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)

        config_dir = tmp_path / ".ripperdoc"
        config_dir.mkdir(parents=True)
        (config_dir / "hooks.local.json").write_text(json.dumps({
            "hooks": {
                "SessionEnd": [{
                    "hooks": [{"type": "command", "command": "echo session_end"}]
                }]
            }
        }))

        manager = HookManager(project_dir=tmp_path)
        result = await manager.run_session_end_async("logout", 100.0, 10)

        assert len(result.outputs) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Additional Manager Sync Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestHookManagerSyncMethods:
    """Additional tests for HookManager sync methods."""

    def test_run_permission_request(self, tmp_path, monkeypatch):
        """run_permission_request should execute hooks."""
        monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)

        config_dir = tmp_path / ".ripperdoc"
        config_dir.mkdir(parents=True)
        (config_dir / "hooks.local.json").write_text(json.dumps({
            "hooks": {
                "PermissionRequest": [{
                    "matcher": "*",
                    "hooks": [{"type": "command", "command": "echo permission"}]
                }]
            }
        }))

        manager = HookManager(project_dir=tmp_path)
        result = manager.run_permission_request("Write", {"file_path": "/tmp/test"})

        assert len(result.outputs) == 1

    def test_run_post_tool_use(self, tmp_path, monkeypatch):
        """run_post_tool_use should execute hooks."""
        monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)

        config_dir = tmp_path / ".ripperdoc"
        config_dir.mkdir(parents=True)
        (config_dir / "hooks.local.json").write_text(json.dumps({
            "hooks": {
                "PostToolUse": [{
                    "matcher": "*",
                    "hooks": [{"type": "command", "command": "echo post"}]
                }]
            }
        }))

        manager = HookManager(project_dir=tmp_path)
        result = manager.run_post_tool_use("Bash", {}, "output", "toolu_123")

        assert len(result.outputs) == 1

    def test_run_subagent_stop(self, tmp_path, monkeypatch):
        """run_subagent_stop should execute hooks."""
        monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)

        config_dir = tmp_path / ".ripperdoc"
        config_dir.mkdir(parents=True)
        (config_dir / "hooks.local.json").write_text(json.dumps({
            "hooks": {
                "SubagentStop": [{
                    "hooks": [{"type": "command", "command": "echo subagent"}]
                }]
            }
        }))

        manager = HookManager(project_dir=tmp_path)
        result = manager.run_subagent_stop(False)

        assert len(result.outputs) == 1

    def test_run_pre_compact(self, tmp_path, monkeypatch):
        """run_pre_compact should execute hooks."""
        monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)

        config_dir = tmp_path / ".ripperdoc"
        config_dir.mkdir(parents=True)
        (config_dir / "hooks.local.json").write_text(json.dumps({
            "hooks": {
                "PreCompact": [{
                    "hooks": [{"type": "command", "command": "echo compact"}]
                }]
            }
        }))

        manager = HookManager(project_dir=tmp_path)
        result = manager.run_pre_compact("auto")

        assert len(result.outputs) == 1
