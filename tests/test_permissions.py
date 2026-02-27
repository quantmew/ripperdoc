"""Tests for the permission system."""

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from ripperdoc.core.permission_engine import (
    PermissionPreview,
    PermissionResult,
    make_permission_checker,
)
from ripperdoc.core.hooks.manager import hook_manager
from ripperdoc.core.config import (
    UserConfig,
    ProjectLocalConfig,
    config_manager,
    save_global_config,
    save_project_local_config,
)
from ripperdoc.core.tool import Tool, ToolResult, ToolUseContext
from ripperdoc.tools.bash_tool import BashTool, BashToolInput
from ripperdoc.tools.file_read_tool import FileReadTool, FileReadToolInput
from ripperdoc.tools.file_edit_tool import FileEditTool, FileEditToolInput
from ripperdoc.tools.multi_edit_tool import MultiEditTool, MultiEditToolInput


class DummyInput(BaseModel):
    command: str


class DummyTool(Tool[DummyInput, None]):
    @property
    def name(self) -> str:
        return "DummyTool"

    async def description(self) -> str:
        return "Dummy tool for testing permissions."

    @property
    def input_schema(self) -> type[DummyInput]:
        return DummyInput

    async def prompt(self, yolo_mode: bool = False) -> str:  # noqa: ARG002
        return "dummy prompt"

    def render_result_for_assistant(self, output: None) -> str:
        return "done"

    def render_tool_use_message(self, input_data: DummyInput, verbose: bool = False) -> str:
        return f"run {input_data.command}"

    async def call(self, input_data: DummyInput, context: ToolUseContext):
        yield ToolResult(data=None)


class DictPermissionTool(DummyTool):
    """Dummy tool whose check_permissions returns a dict (legacy behavior)."""

    async def check_permissions(self, input_data: DummyInput, permission_context: Any):
        return {"behavior": "allow", "updated_input": input_data}


@pytest.fixture
def isolated_config(tmp_path):
    """Isolate global/project config paths for permission tests."""
    original_global_path = config_manager.global_config_path
    original_global = config_manager._global_config
    original_project = config_manager._project_config
    original_project_local = config_manager._project_local_config

    config_manager.global_config_path = tmp_path / "global.json"
    config_manager._global_config = None
    config_manager._project_config = None
    config_manager._project_local_config = None

    yield

    config_manager.global_config_path = original_global_path
    config_manager._global_config = original_global
    config_manager._project_config = original_project
    config_manager._project_local_config = original_project_local


def test_yolo_mode_off_does_not_persist_permissions(tmp_path: Path):
    """Approvals should not be written to project config."""
    tool = DummyTool()
    parsed_input = DummyInput(command="echo hello")

    prompt_calls = 0

    def prompt_fn(_: str) -> str:
        nonlocal prompt_calls
        prompt_calls += 1
        return "a"  # approve for the session only

    checker = make_permission_checker(tmp_path, yolo_mode=False, prompt_fn=prompt_fn)
    result = asyncio.run(checker(tool, parsed_input))
    assert isinstance(result, PermissionResult)
    assert result.result is True

    config = config_manager.get_project_config(tmp_path)
    assert tool.name not in config.allowed_tools

    # New checker should prompt again because approvals are session-only
    second_prompt_calls = 0

    def prompt_fn_again(_: str) -> str:
        nonlocal second_prompt_calls
        second_prompt_calls += 1
        return "y"

    checker_again = make_permission_checker(tmp_path, yolo_mode=False, prompt_fn=prompt_fn_again)
    second = asyncio.run(checker_again(tool, DummyInput(command="echo goodbye")))
    assert second.result is True
    assert prompt_calls == 1
    assert second_prompt_calls == 1


def test_session_always_allows_similar_commands(tmp_path: Path):
    """Session-level approvals should skip prompts for the same tool."""
    tool = DummyTool()
    prompts = iter(["a"])
    prompt_calls = 0

    def prompt_fn(_: str) -> str:
        nonlocal prompt_calls
        prompt_calls += 1
        return next(prompts)

    checker = make_permission_checker(tmp_path, yolo_mode=False, prompt_fn=prompt_fn)

    first = asyncio.run(checker(tool, DummyInput(command="echo first")))
    second = asyncio.run(checker(tool, DummyInput(command="echo second")))

    assert first.result is True
    assert second.result is True
    assert prompt_calls == 1

    # Session approvals should not be persisted to disk
    config = config_manager.get_project_config(tmp_path)
    assert tool.name not in config.allowed_tools


def test_yolo_mode_off_respects_read_only_tools(tmp_path: Path):
    """Read-only tools should bypass permission prompts even when prompts are on."""
    from ripperdoc.tools.file_read_tool import FileReadTool, FileReadToolInput

    temp_file = tmp_path / "file.txt"
    temp_file.write_text("hello")

    checker = make_permission_checker(
        tmp_path, yolo_mode=False, prompt_fn=lambda _: pytest.fail("prompted unexpectedly")
    )
    tool = FileReadTool()
    parsed_input = FileReadToolInput(file_path=str(temp_file))

    result = asyncio.run(checker(tool, parsed_input))
    assert result.result is True


def test_yolo_mode_allows_without_prompt(tmp_path: Path):
    """When yolo mode is enabled, tools should run without permission checks."""
    tool = DummyTool()
    parsed_input = DummyInput(command="rm -rf /tmp/test")

    checker = make_permission_checker(
        tmp_path, yolo_mode=True, prompt_fn=lambda _: pytest.fail("prompted unexpectedly")
    )
    result = asyncio.run(checker(tool, parsed_input))
    assert result.result is True


def test_dict_permission_result_is_handled(tmp_path: Path):
    """Tools returning dict-based permission decisions should still be accepted."""
    tool = DictPermissionTool()
    parsed_input = DummyInput(command="echo ok")

    checker = make_permission_checker(tmp_path, yolo_mode=False, prompt_fn=lambda _: "y")
    result = asyncio.run(checker(tool, parsed_input))
    assert result.result is True


def test_bash_permissions_include_global_and_local_rules(tmp_path: Path, isolated_config):
    """Global and local allow rules should be honored for Bash permissions."""
    save_global_config(UserConfig(user_allow_rules=["rm -rf /tmp/allowed"]))
    save_project_local_config(
        ProjectLocalConfig(local_allow_rules=["touch /tmp/local"]), project_path=tmp_path
    )

    tool = BashTool()
    checker = make_permission_checker(tmp_path, yolo_mode=False, prompt_fn=lambda _: "n")

    global_allowed = asyncio.run(checker(tool, BashToolInput(command="rm -rf /tmp/allowed")))
    local_allowed = asyncio.run(checker(tool, BashToolInput(command="touch /tmp/local")))

    assert global_allowed.result is True
    assert local_allowed.result is True


def test_bash_permissions_apply_denies_from_all_scopes(tmp_path: Path, isolated_config):
    """Deny rules from any scope should block Bash commands."""
    save_global_config(UserConfig(user_deny_rules=["rm -rf /tmp/deny"]))
    save_project_local_config(
        ProjectLocalConfig(local_deny_rules=["echo local-deny"]), project_path=tmp_path
    )

    tool = BashTool()
    checker = make_permission_checker(tmp_path, yolo_mode=False, prompt_fn=lambda _: "y")

    global_denied = asyncio.run(checker(tool, BashToolInput(command="rm -rf /tmp/deny")))
    local_denied = asyncio.run(checker(tool, BashToolInput(command="echo local-deny")))

    assert global_denied.result is False
    assert local_denied.result is False


def test_bash_permissions_apply_ask_rules(tmp_path: Path, isolated_config):
    """Ask rules should force prompts even for read-only commands."""
    save_global_config(UserConfig(user_ask_rules=["ls"]))

    prompt_calls = 0

    def prompt_fn(_: str) -> str:
        nonlocal prompt_calls
        prompt_calls += 1
        return "n"

    tool = BashTool()
    checker = make_permission_checker(tmp_path, yolo_mode=False, prompt_fn=prompt_fn)

    result = asyncio.run(checker(tool, BashToolInput(command="ls")))

    assert result.result is False
    assert prompt_calls == 1


def test_plan_mode_auto_allows_default_permission_prompts(tmp_path: Path):
    """Plan mode should auto-allow non-rule ask flows when bypass is available."""
    checker = make_permission_checker(
        tmp_path,
        yolo_mode=False,
        permission_mode="plan",
        prompt_fn=lambda _: pytest.fail("prompted unexpectedly"),
    )
    result = asyncio.run(checker(DummyTool(), DummyInput(command="echo plan-mode")))
    assert result.result is True


def test_plan_mode_respects_disable_bypass_setting(tmp_path: Path, isolated_config):
    """Plan mode should stop auto-allowing when bypass mode is disabled by config."""
    save_global_config(UserConfig(disable_bypass_permissions_mode=True))

    prompt_calls = 0

    def prompt_fn(_: str) -> str:
        nonlocal prompt_calls
        prompt_calls += 1
        return "n"

    checker = make_permission_checker(
        tmp_path,
        yolo_mode=False,
        permission_mode="plan",
        prompt_fn=prompt_fn,
    )
    result = asyncio.run(checker(DummyTool(), DummyInput(command="echo blocked")))

    assert result.result is False
    assert prompt_calls == 1


def test_plan_mode_still_prompts_for_explicit_ask_rules(tmp_path: Path, isolated_config):
    """Explicit ask rules should still require confirmation in plan mode."""
    save_global_config(UserConfig(user_ask_rules=["ls"]))

    prompt_calls = 0

    def prompt_fn(_: str) -> str:
        nonlocal prompt_calls
        prompt_calls += 1
        return "n"

    checker = make_permission_checker(
        tmp_path,
        yolo_mode=False,
        permission_mode="plan",
        prompt_fn=prompt_fn,
    )
    result = asyncio.run(checker(BashTool(), BashToolInput(command="ls")))

    assert result.result is False
    assert prompt_calls == 1


def test_dont_ask_mode_denies_without_prompt_when_confirmation_needed(
    tmp_path: Path, isolated_config
):
    """dontAsk mode must deny ask-required operations without showing prompts."""
    save_global_config(UserConfig(user_ask_rules=["ls"]))

    prompt_calls = 0

    def prompt_fn(_: str) -> str:
        nonlocal prompt_calls
        prompt_calls += 1
        return "y"

    checker = make_permission_checker(
        tmp_path,
        yolo_mode=False,
        permission_mode="dontAsk",
        prompt_fn=prompt_fn,
    )
    result = asyncio.run(checker(BashTool(), BashToolInput(command="ls")))

    assert result.result is False
    assert result.message is not None
    assert "dontAsk" in result.message
    assert prompt_calls == 0


def test_dont_ask_mode_allows_read_only_when_no_prompt_required(tmp_path: Path):
    """dontAsk mode should not block read-only operations that do not require prompts."""
    target = tmp_path / "note.txt"
    target.write_text("hello", encoding="utf-8")

    checker = make_permission_checker(
        tmp_path,
        yolo_mode=False,
        permission_mode="dontAsk",
        prompt_fn=lambda _: pytest.fail("prompted unexpectedly"),
    )
    result = asyncio.run(checker(FileReadTool(), FileReadToolInput(file_path=str(target))))

    assert result.result is True


def test_read_rule_with_specifier_can_force_prompt(tmp_path: Path, isolated_config):
    """Tool(specifier) ask rules should prompt even for read-only tools."""
    env_file = tmp_path / ".env"
    env_file.write_text("A=1\n")
    save_global_config(UserConfig(user_ask_rules=["Read(./.env)"]))

    prompt_calls = 0

    def prompt_fn(_: str) -> str:
        nonlocal prompt_calls
        prompt_calls += 1
        return "n"

    checker = make_permission_checker(tmp_path, yolo_mode=False, prompt_fn=prompt_fn)
    result = asyncio.run(checker(FileReadTool(), FileReadToolInput(file_path=str(env_file))))

    assert result.result is False
    assert prompt_calls == 1


def test_read_rule_with_specifier_can_deny(tmp_path: Path, isolated_config):
    """Tool(specifier) deny rules should block read-only tools."""
    env_file = tmp_path / ".env"
    env_file.write_text("A=1\n")
    save_global_config(UserConfig(user_deny_rules=["Read(./.env)"]))

    checker = make_permission_checker(
        tmp_path, yolo_mode=False, prompt_fn=lambda _: pytest.fail("prompted unexpectedly")
    )
    result = asyncio.run(checker(FileReadTool(), FileReadToolInput(file_path=str(env_file))))

    assert result.result is False


def test_bash_tool_rule_matches_all_commands(tmp_path: Path, isolated_config):
    """Bash tool-only rule should allow all Bash commands."""
    save_global_config(UserConfig(user_allow_rules=["Bash"]))

    checker = make_permission_checker(
        tmp_path, yolo_mode=False, prompt_fn=lambda _: pytest.fail("prompted unexpectedly")
    )
    result = asyncio.run(checker(BashTool(), BashToolInput(command="git status && echo done")))

    assert result.result is True


def test_bash_legacy_suffix_rule_still_works(tmp_path: Path, isolated_config):
    """Deprecated Bash :* suffix syntax should remain compatible."""
    save_global_config(UserConfig(user_allow_rules=["Bash(ls:*)"]))

    checker = make_permission_checker(
        tmp_path, yolo_mode=False, prompt_fn=lambda _: pytest.fail("prompted unexpectedly")
    )
    result = asyncio.run(checker(BashTool(), BashToolInput(command="ls -la")))

    assert result.result is True


def test_deny_rule_overrides_allow_rule(tmp_path: Path, isolated_config):
    """Rule priority should be deny > allow."""
    save_global_config(
        UserConfig(
            user_allow_rules=["Bash"],
            user_deny_rules=["Bash(rm -rf *)"],
        )
    )

    checker = make_permission_checker(
        tmp_path, yolo_mode=False, prompt_fn=lambda _: pytest.fail("prompted unexpectedly")
    )
    result = asyncio.run(checker(BashTool(), BashToolInput(command="rm -rf /tmp/test")))

    assert result.result is False


def test_permission_checker_preview_detects_ask_for_read_only_bash(tmp_path: Path, isolated_config):
    """Preview should require user input when ask-rules match read-only Bash commands."""
    save_global_config(UserConfig(user_ask_rules=["ls"]))

    checker = make_permission_checker(tmp_path, yolo_mode=False, prompt_fn=lambda _: "n")
    preview = asyncio.run(checker.preview(BashTool(), BashToolInput(command="ls")))

    assert isinstance(preview, PermissionPreview)
    assert preview.requires_user_input is True
    assert preview.result is None


def test_permission_checker_preview_skips_user_input_when_not_required(tmp_path: Path):
    """Preview should auto-allow read-only Bash commands without matching ask-rules."""
    checker = make_permission_checker(
        tmp_path, yolo_mode=False, prompt_fn=lambda _: pytest.fail("prompted unexpectedly")
    )
    preview = asyncio.run(checker.preview(BashTool(), BashToolInput(command="ls")))

    assert isinstance(preview, PermissionPreview)
    assert preview.requires_user_input is False
    assert isinstance(preview.result, PermissionResult)
    assert preview.result.result is True


def test_permission_checker_respects_session_additional_working_dirs(tmp_path: Path):
    """Session-scoped additional working dirs should bypass path prompts for find/cd."""
    outside_dir = tmp_path.parent / f"{tmp_path.name}_outside"
    outside_dir.mkdir(exist_ok=True)

    checker_without_extra = make_permission_checker(
        tmp_path, yolo_mode=False, prompt_fn=lambda _: pytest.fail("prompted unexpectedly")
    )
    preview_without = asyncio.run(
        checker_without_extra.preview(BashTool(), BashToolInput(command=f"find {outside_dir} -maxdepth 1"))
    )
    assert isinstance(preview_without, PermissionPreview)
    assert preview_without.requires_user_input is True

    checker_with_extra = make_permission_checker(
        tmp_path,
        yolo_mode=False,
        prompt_fn=lambda _: pytest.fail("prompted unexpectedly"),
        session_additional_working_dirs=[str(outside_dir)],
    )
    preview_with = asyncio.run(
        checker_with_extra.preview(BashTool(), BashToolInput(command=f"find {outside_dir} -maxdepth 1"))
    )
    assert isinstance(preview_with, PermissionPreview)
    assert preview_with.requires_user_input is False
    assert isinstance(preview_with.result, PermissionResult)
    assert preview_with.result.result is True


def test_permission_checker_can_add_session_working_dir_dynamically(tmp_path: Path):
    """Permission checker should support runtime add directory updates."""
    outside_dir = tmp_path.parent / f"{tmp_path.name}_dynamic_outside"
    outside_dir.mkdir(exist_ok=True)

    checker = make_permission_checker(
        tmp_path, yolo_mode=False, prompt_fn=lambda _: pytest.fail("prompted unexpectedly")
    )
    first_preview = asyncio.run(
        checker.preview(BashTool(), BashToolInput(command=f"find {outside_dir} -maxdepth 1"))
    )
    assert isinstance(first_preview, PermissionPreview)
    assert first_preview.requires_user_input is True

    adder = getattr(checker, "add_working_directory")
    added_path = adder(str(outside_dir))
    assert added_path == str(outside_dir.resolve())

    second_preview = asyncio.run(
        checker.preview(BashTool(), BashToolInput(command=f"find {outside_dir} -maxdepth 1"))
    )
    assert isinstance(second_preview, PermissionPreview)
    assert second_preview.requires_user_input is False


@pytest.mark.asyncio
async def test_glob_rules_in_permission_checker(tmp_path: Path):
    """Permission checker should handle glob-format allow rules."""
    from ripperdoc.core.config import save_project_config, ProjectConfig
    from ripperdoc.core.permission_engine import make_permission_checker
    from ripperdoc.tools.bash_tool import BashTool, BashToolInput

    # Setup config with glob rules.
    config = ProjectConfig(
        bash_allow_rules=[
            "git *",
            "npm *",  # Glob format
            "* --version",  # Glob format
        ]
    )
    save_project_config(config, tmp_path)

    can_use_tool = make_permission_checker(
        project_path=tmp_path,
        yolo_mode=False,
        prompt_fn=lambda _: "1",  # Auto-approve
    )

    tool = BashTool()

    # Test glob format rule
    result = await can_use_tool(tool, BashToolInput(command="git status"))
    assert result.result is True

    # Test glob format rule
    result = await can_use_tool(tool, BashToolInput(command="npm install"))
    assert result.result is True

    # Test glob pattern matching
    result = await can_use_tool(tool, BashToolInput(command="python --version"))
    assert result.result is True


def test_permission_request_hooks_deny_overrides_allow(tmp_path: Path, monkeypatch):
    """PermissionRequest hook deny/block should override allow decisions."""
    monkeypatch.setattr("ripperdoc.core.hooks.config.Path.home", lambda: tmp_path)

    allow_output = json.dumps(
        {
            "hookSpecificOutput": {
                "hookEventName": "PermissionRequest",
                "decision": {"behavior": "allow", "message": "Allowed by policy"},
            }
        }
    )
    deny_output = json.dumps(
        {
            "hookSpecificOutput": {
                "hookEventName": "PermissionRequest",
                "decision": {"behavior": "deny", "message": "Denied by policy"},
            }
        }
    )

    config_dir = tmp_path / ".ripperdoc"
    config_dir.mkdir(parents=True)
    (config_dir / "hooks.local.json").write_text(
        json.dumps(
            {
                "hooks": {
                    "PermissionRequest": [
                        {
                            "matcher": "*",
                            "hooks": [
                                {"type": "command", "command": f"echo '{allow_output}'"},
                                {"type": "command", "command": f"echo '{deny_output}'"},
                            ],
                        }
                    ]
                }
            }
        )
    )

    original_project_dir = hook_manager.project_dir
    try:
        hook_manager.set_project_dir(tmp_path)
        hook_manager.reload_config()

        tool = DummyTool()
        parsed_input = DummyInput(command="echo hi")
        checker = make_permission_checker(
            tmp_path,
            yolo_mode=False,
            prompt_fn=lambda _: pytest.fail("prompted unexpectedly"),
        )
        result = asyncio.run(checker(tool, parsed_input))
        assert result.result is False
        assert result.message is not None
        assert "denied by policy" in result.message.lower()
    finally:
        hook_manager.set_project_dir(original_project_dir)
        hook_manager.reload_config()


def test_edit_permission_prompt_shows_diff_preview_before_apply(tmp_path: Path, monkeypatch):
    """Edit permission prompt should include a before-apply diff preview."""
    target = tmp_path / "sample.py"
    target.write_text("a = 1\nb = 2\n", encoding="utf-8")

    captured: dict[str, str] = {}

    def fake_prompt_choice(*args: Any, **kwargs: Any) -> str:  # noqa: ANN401
        captured["message"] = str(kwargs.get("message", ""))
        return "n"

    monkeypatch.setattr("ripperdoc.core.permission_engine.prompt_choice", fake_prompt_choice)

    checker = make_permission_checker(tmp_path, yolo_mode=False, prompt_fn=None)
    result = asyncio.run(
        checker(
            FileEditTool(),
            FileEditToolInput(
                file_path=str(target),
                old_string="a = 1",
                new_string="a = 3",
            ),
        )
    )

    assert result.result is False
    preview = captured.get("message", "")
    assert "preview:" in preview
    assert "-------------------" in preview
    assert re.search(r"<diff-del>\s*1\s+- a = 1</diff-del>", preview)
    assert re.search(r"<diff-add>\s*1 \+ a = 3</diff-add>", preview)
    assert target.read_text(encoding="utf-8") == "a = 1\nb = 2\n"


def test_multiedit_permission_prompt_shows_diff_preview_before_apply(tmp_path: Path, monkeypatch):
    """MultiEdit permission prompt should include combined diff preview."""
    target = tmp_path / "sample_multi.py"
    target.write_text("x = 1\ny = 2\n", encoding="utf-8")

    captured: dict[str, str] = {}

    def fake_prompt_choice(*args: Any, **kwargs: Any) -> str:  # noqa: ANN401
        captured["message"] = str(kwargs.get("message", ""))
        return "n"

    monkeypatch.setattr("ripperdoc.core.permission_engine.prompt_choice", fake_prompt_choice)

    checker = make_permission_checker(tmp_path, yolo_mode=False, prompt_fn=None)
    result = asyncio.run(
        checker(
            MultiEditTool(),
            MultiEditToolInput(
                file_path=str(target),
                edits=[
                    {"old_string": "x = 1", "new_string": "x = 10"},
                    {"old_string": "y = 2", "new_string": "y = 20"},
                ],
            ),
        )
    )

    assert result.result is False
    preview = captured.get("message", "")
    assert "preview:" in preview
    assert "-------------------" in preview
    assert re.search(r"<diff-del>\s*1\s+- x = 1</diff-del>", preview)
    assert re.search(r"<diff-add>\s*1 \+ x = 10</diff-add>", preview)
    assert re.search(r"<diff-del>\s*2\s+- y = 2</diff-del>", preview)
    assert re.search(r"<diff-add>\s*2 \+ y = 20</diff-add>", preview)
    assert target.read_text(encoding="utf-8") == "x = 1\ny = 2\n"
