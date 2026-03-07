"""CLI/stdio option propagation tests for SDK compatibility flags."""

from __future__ import annotations

import asyncio
from click.testing import CliRunner
import click
import pytest
from typing import Any
from pathlib import Path
import subprocess
import re
import signal
from types import SimpleNamespace

from ripperdoc.cli import cli as cli_module
from ripperdoc.protocol.stdio import command as stdio_command
from ripperdoc.core.tool_defaults import get_default_tools


def test_cli_forwards_hidden_sdk_options_to_stdio_defaults(monkeypatch, tmp_path):
    captured: dict = {}

    async def fake_run_stdio(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("ripperdoc.protocol.stdio.run_stdio", fake_run_stdio)
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli,
        [
            "--output-format",
            "json",
            "--fallback-model",
            "backup-model",
            "--max-budget-usd",
            "3.5",
            "--max-thinking-tokens",
            "512",
            "--json-schema",
            '{"type":"object"}',
            "--include-partial-messages",
            "--replay-user-messages",
            "--fork-session",
            "--agent",
            "reviewer",
            "--agents",
            '{"reviewer":{"description":"Reviews code","prompt":"You are a code reviewer"}}',
            "--disable-slash-commands",
            "--plugin-dir",
            "plugins/a",
            "--betas",
            "exp-beta",
        ],
        env={"HOME": str(tmp_path)},
    )

    assert result.exit_code == 0
    assert captured
    default_options = captured["default_options"]
    assert default_options["fallback_model"] == "backup-model"
    assert default_options["max_budget_usd"] == 3.5
    assert default_options["max_thinking_tokens"] == 512
    assert default_options["json_schema"] == '{"type":"object"}'
    assert default_options["include_partial_messages"] is True
    assert default_options["replay_user_messages"] is True
    assert default_options["fork_session"] is True
    assert default_options["agent"] == "reviewer"
    assert default_options["agents"]["reviewer"]["prompt"] == "You are a code reviewer"
    assert default_options["disable_slash_commands"] is True
    assert default_options["plugin_dirs"] == ["plugins/a"]
    assert default_options["betas"] == "exp-beta"


@pytest.mark.asyncio
async def test_run_stdio_print_mode_merges_default_options(monkeypatch):
    captured: dict = {}

    class FakeHandler:
        def __init__(self, input_format, output_format, default_options=None):
            captured["ctor"] = {
                "input_format": input_format,
                "output_format": output_format,
                "default_options": default_options,
            }

        async def _handle_initialize(self, request, request_id):
            captured["initialize"] = {"request": request, "request_id": request_id}

        async def _handle_query(self, request, request_id):
            captured["query"] = {"request": request, "request_id": request_id}

        async def flush_output(self):
            captured["flushed"] = True

        async def run(self):
            captured["ran"] = True

    monkeypatch.setattr(stdio_command, "StdioProtocolHandler", FakeHandler)

    await stdio_command.run_stdio(
        input_format="stream-json",
        output_format="json",
        model=None,
        permission_mode="default",
        max_turns=None,
        system_prompt=None,
        print_mode=True,
        prompt="hello",
        default_options={"max_thinking_tokens": 128, "json_schema": '{"type":"object"}'},
    )

    init_request = captured["initialize"]["request"]
    init_options = init_request["_meta"]["ripperdoc_options"]
    assert init_options["max_thinking_tokens"] == 128
    assert init_options["json_schema"] == '{"type":"object"}'
    assert init_options["sdk_can_use_tool"] is False


@pytest.mark.asyncio
async def test_run_stdio_replay_user_messages_requires_stream_json_modes():
    with pytest.raises(
        click.ClickException,
        match="--replay-user-messages requires both --input-format=stream-json and --output-format=stream-json",
    ):
        await stdio_command.run_stdio(
            input_format="auto",
            output_format="stream-json",
            model=None,
            permission_mode="default",
            max_turns=None,
            system_prompt=None,
            print_mode=False,
            replay_user_messages=True,
        )


def test_install_stdio_shutdown_signal_handlers_restores_previous_handlers(monkeypatch):
    captured: dict[int, Any] = {}
    restored: dict[int, Any] = {}
    previous = object()
    cancelled: list[str] = []

    def fake_getsignal(sig: int) -> Any:
        return previous

    def fake_signal(sig: int, handler: Any) -> None:
        if sig not in captured:
            captured[sig] = handler
        else:
            restored[sig] = handler

    monkeypatch.setattr(stdio_command.signal, "getsignal", fake_getsignal)
    monkeypatch.setattr(stdio_command.signal, "signal", fake_signal)

    restore = stdio_command._install_stdio_shutdown_signal_handlers(
        lambda: cancelled.append("cancelled")
    )

    assert signal.SIGTERM in captured
    assert signal.SIGINT in captured

    captured[signal.SIGTERM](signal.SIGTERM, None)
    assert cancelled == ["cancelled"]

    restore()

    assert restored[signal.SIGTERM] is previous
    assert restored[signal.SIGINT] is previous


@pytest.mark.asyncio
async def test_run_stdio_with_signal_handling_swallows_signal_cancellation(monkeypatch):
    started = asyncio.Event()
    release = asyncio.Event()

    async def fake_run_stdio(**_kwargs: Any) -> None:
        started.set()
        await release.wait()

    monkeypatch.setattr(stdio_command, "run_stdio", fake_run_stdio)

    installed_cancel: dict[str, Any] = {}

    def fake_install(cancel_main_task):
        installed_cancel["cancel"] = cancel_main_task

        def _restore() -> None:
            installed_cancel["restored"] = True

        return _restore

    monkeypatch.setattr(
        stdio_command,
        "_install_stdio_shutdown_signal_handlers",
        fake_install,
    )

    runner = asyncio.create_task(
        stdio_command._run_stdio_with_signal_handling(
            input_format="stream-json",
            output_format="stream-json",
            model=None,
            permission_mode="default",
            max_turns=None,
            system_prompt=None,
            print_mode=False,
        )
    )

    await started.wait()
    installed_cancel["cancel"]()
    await runner

    assert installed_cancel["restored"] is True


def test_cli_rejects_sdk_only_options_outside_stdio(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli,
        ["--prompt", "hi", "--json-schema", '{"type":"object"}'],
        env={"HOME": str(tmp_path)},
    )

    assert result.exit_code != 0
    assert "SDK-only" in result.output
    assert "--json-schema" in result.output


def test_cli_rejects_replay_user_messages_outside_stdio(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli,
        ["--prompt", "hi", "--replay-user-messages"],
        env={"HOME": str(tmp_path)},
    )

    assert result.exit_code != 0
    assert "SDK-only" in result.output
    assert "--replay-user-messages" in result.output


def test_cli_sdk_url_without_print_runs_stdio_loop_mode(monkeypatch, tmp_path):
    captured: dict[str, Any] = {}

    async def fake_run_stdio(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("ripperdoc.protocol.stdio.run_stdio", fake_run_stdio)
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli,
        ["--sdk-url", "ws://localhost:3000/v2/session_ingress/ws/session-1"],
        env={"HOME": str(tmp_path)},
    )

    assert result.exit_code == 0
    assert captured["input_format"] == "stream-json"
    assert captured["output_format"] == "stream-json"
    assert captured["print_mode"] is False


def test_cli_agent_overrides_settings_agent_for_prompt(monkeypatch, tmp_path):
    captured: dict[str, Any] = {}

    async def fake_run_query(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(cli_module, "run_query", fake_run_query)
    monkeypatch.setattr(cli_module, "check_onboarding", lambda: True)
    monkeypatch.setattr("ripperdoc.core.tool_defaults.get_default_tools", lambda **_: [])
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli,
        [
            "--prompt",
            "hi",
            "--settings",
            (
                '{"agent":"reviewer","agents":{"reviewer":{"prompt":"Use reviewer behavior from '
                'settings."}}}'
            ),
            "--agent",
            "writer",
            "--agents",
            '{"writer":{"prompt":"Use writer behavior from CLI."}}',
        ],
        env={"HOME": str(tmp_path)},
    )

    assert result.exit_code == 0
    assert captured["kwargs"]["custom_system_prompt"] == "Use writer behavior from CLI."
    assert captured["kwargs"]["append_system_prompt"] is None


def test_cli_preserves_append_system_prompt_after_agent_override(monkeypatch, tmp_path):
    captured: dict[str, Any] = {}

    async def fake_run_query(*args, **kwargs):
        captured["kwargs"] = kwargs

    monkeypatch.setattr(cli_module, "run_query", fake_run_query)
    monkeypatch.setattr(cli_module, "check_onboarding", lambda: True)
    monkeypatch.setattr("ripperdoc.core.tool_defaults.get_default_tools", lambda **_: [])
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli,
        [
            "--prompt",
            "hi",
            "--agent",
            "writer",
            "--agents",
            '{"writer":{"prompt":"Use writer behavior from CLI."}}',
            "--append-system-prompt",
            "Keep answers terse.",
        ],
        env={"HOME": str(tmp_path)},
    )

    assert result.exit_code == 0
    assert captured["kwargs"]["custom_system_prompt"] == "Use writer behavior from CLI."
    assert captured["kwargs"]["append_system_prompt"] == "Keep answers terse."


def test_cli_debug_with_filter(monkeypatch, tmp_path):
    captured: dict[str, Any] = {}

    async def fake_run_query(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    def fake_configure_debug_logging(**kwargs):
        captured["debug"] = kwargs
        return None

    monkeypatch.setattr(cli_module, "run_query", fake_run_query)
    monkeypatch.setattr(cli_module, "check_onboarding", lambda: True)
    monkeypatch.setattr("ripperdoc.core.tool_defaults.get_default_tools", lambda **_: [])
    monkeypatch.setattr(cli_module, "configure_debug_logging", fake_configure_debug_logging)
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli,
        ["--prompt", "hi", "--debug", "--debug-filter", "api,hooks"],
        env={"HOME": str(tmp_path)},
    )

    assert result.exit_code == 0
    assert captured["debug"]["enabled"] is True
    assert captured["debug"]["filter_spec"] == "api,hooks"


def test_main_rewrites_debug_optional_filter(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_cli_main(*, args, prog_name):
        captured["args"] = list(args)
        captured["prog_name"] = prog_name

    monkeypatch.setattr(cli_module.cli, "main", fake_cli_main)
    monkeypatch.setattr(
        cli_module.sys,
        "argv",
        ["ripperdoc", "--prompt", "hi", "--debug", "api,hooks"],
    )

    cli_module.main()

    assert captured["prog_name"] == "ripperdoc"
    assert "--debug-filter" in captured["args"]
    idx = captured["args"].index("--debug-filter")
    assert captured["args"][idx + 1] == "api,hooks"


def test_main_does_not_treat_remote_control_subcommand_as_debug_filter(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_cli_main(*, args, prog_name):
        captured["args"] = list(args)
        captured["prog_name"] = prog_name

    monkeypatch.setattr(cli_module.cli, "main", fake_cli_main)
    monkeypatch.setattr(
        cli_module.sys,
        "argv",
        ["ripperdoc", "--debug", "remote-control", "--help"],
    )

    cli_module.main()

    assert captured["prog_name"] == "ripperdoc"
    assert "--debug-filter" not in captured["args"]
    assert "remote-control" in captured["args"]


def test_main_rewrites_worktree_optional_name(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_cli_main(*, args, prog_name):
        captured["args"] = list(args)
        captured["prog_name"] = prog_name

    monkeypatch.setattr(cli_module.cli, "main", fake_cli_main)
    monkeypatch.setattr(
        cli_module.sys,
        "argv",
        ["ripperdoc", "--worktree", "feature-x", "--prompt", "hi"],
    )

    cli_module.main()

    assert captured["prog_name"] == "ripperdoc"
    assert "--worktree" in captured["args"]
    assert "--worktree-name" in captured["args"]
    idx = captured["args"].index("--worktree-name")
    assert captured["args"][idx + 1] == "feature-x"


def test_main_rewrites_worktree_equals_name(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_cli_main(*, args, prog_name):
        captured["args"] = list(args)
        captured["prog_name"] = prog_name

    monkeypatch.setattr(cli_module.cli, "main", fake_cli_main)
    monkeypatch.setattr(
        cli_module.sys,
        "argv",
        ["ripperdoc", "--worktree=feature-x", "--prompt", "hi"],
    )

    cli_module.main()

    assert captured["prog_name"] == "ripperdoc"
    assert "--worktree" in captured["args"]
    assert "--worktree-name" in captured["args"]
    idx = captured["args"].index("--worktree-name")
    assert captured["args"][idx + 1] == "feature-x"


def test_main_rewrites_worktree_short_flag_name(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_cli_main(*, args, prog_name):
        captured["args"] = list(args)
        captured["prog_name"] = prog_name

    monkeypatch.setattr(cli_module.cli, "main", fake_cli_main)
    monkeypatch.setattr(
        cli_module.sys,
        "argv",
        ["ripperdoc", "-w", "feature-x", "--prompt", "hi"],
    )

    cli_module.main()

    assert captured["prog_name"] == "ripperdoc"
    assert "-w" in captured["args"]
    assert "--worktree-name" in captured["args"]
    idx = captured["args"].index("--worktree-name")
    assert captured["args"][idx + 1] == "feature-x"


def test_cli_debug_file_implies_debug_and_skips_session_log_path(monkeypatch, tmp_path):
    captured: dict[str, Any] = {}

    async def fake_run_query(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    def fake_configure_debug_logging(**kwargs):
        captured["debug"] = kwargs
        return Path(tmp_path / "debug.log")

    def fail_enable_session_file_logging(*_args, **_kwargs):
        raise AssertionError("enable_session_file_logging should not be called with --debug-file")

    monkeypatch.setattr(cli_module, "run_query", fake_run_query)
    monkeypatch.setattr(cli_module, "check_onboarding", lambda: True)
    monkeypatch.setattr("ripperdoc.core.tool_defaults.get_default_tools", lambda **_: [])
    monkeypatch.setattr(cli_module, "configure_debug_logging", fake_configure_debug_logging)
    monkeypatch.setattr(cli_module, "enable_session_file_logging", fail_enable_session_file_logging)
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli,
        ["--prompt", "hi", "--debug-file", "debug.log"],
        env={"HOME": str(tmp_path)},
    )

    assert result.exit_code == 0
    assert captured["debug"]["enabled"] is True
    assert str(captured["debug"]["debug_file"]).endswith("debug.log")


def test_cli_rejects_unknown_agent_name(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli,
        [
            "--prompt",
            "hi",
            "--agent",
            "unknown",
            "--agents",
            '{"reviewer":{"prompt":"You are a reviewer."}}',
        ],
        env={"HOME": str(tmp_path)},
    )

    assert result.exit_code != 0
    assert "Unknown agent 'unknown'" in result.output


def test_cli_disable_slash_commands_disables_skills_for_prompt(monkeypatch, tmp_path):
    captured: dict[str, Any] = {}

    class DummyTool:
        def __init__(self, name: str) -> None:
            self.name = name

    async def fake_run_query(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(cli_module, "run_query", fake_run_query)
    monkeypatch.setattr(cli_module, "check_onboarding", lambda: True)
    monkeypatch.setattr(
        "ripperdoc.core.tool_defaults.get_default_tools",
        lambda **_: [DummyTool("Skill"), DummyTool("Read")],
    )
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli,
        ["--prompt", "hi", "--disable-slash-commands"],
        env={"HOME": str(tmp_path)},
    )

    assert result.exit_code == 0
    forwarded_tools = captured["args"][1]
    assert [getattr(tool, "name", None) for tool in forwarded_tools] == ["Read"]
    assert captured["kwargs"]["disable_skills"] is True


def test_cli_prompt_forwards_permission_mode_and_max_turns(monkeypatch, tmp_path):
    captured: dict[str, Any] = {}

    async def fake_run_query(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(cli_module, "run_query", fake_run_query)
    monkeypatch.setattr(cli_module, "check_onboarding", lambda: True)
    monkeypatch.setattr("ripperdoc.core.tool_defaults.get_default_tools", lambda **_: [])
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli,
        ["--prompt", "hi", "--max-turns", "7", "--permission-mode", "plan"],
        env={"HOME": str(tmp_path)},
    )

    assert result.exit_code == 0
    assert captured
    assert captured["kwargs"]["max_turns"] == 7
    assert captured["kwargs"]["permission_mode"] == "plan"
    assert captured["args"][2] is False  # yolo_mode


def test_cli_prompt_accepts_dont_ask_permission_mode(monkeypatch, tmp_path):
    captured: dict[str, Any] = {}

    async def fake_run_query(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(cli_module, "run_query", fake_run_query)
    monkeypatch.setattr(cli_module, "check_onboarding", lambda: True)
    monkeypatch.setattr("ripperdoc.core.tool_defaults.get_default_tools", lambda **_: [])
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli,
        ["--prompt", "hi", "--permission-mode", "dontAsk"],
        env={"HOME": str(tmp_path)},
    )

    assert result.exit_code == 0
    assert captured["kwargs"]["permission_mode"] == "dontAsk"
    assert captured["args"][2] is False


def test_cli_yolo_overrides_permission_mode_for_prompt(monkeypatch, tmp_path):
    captured: dict[str, Any] = {}

    async def fake_run_query(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(cli_module, "run_query", fake_run_query)
    monkeypatch.setattr(cli_module, "check_onboarding", lambda: True)
    monkeypatch.setattr("ripperdoc.core.tool_defaults.get_default_tools", lambda **_: [])
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli,
        ["--prompt", "hi", "--permission-mode", "plan", "--yolo"],
        env={"HOME": str(tmp_path)},
    )

    assert result.exit_code == 0
    assert captured["args"][2] is True
    assert captured["kwargs"]["permission_mode"] == "bypassPermissions"


def test_cli_interactive_uses_fallback_model_and_forwards_modes(monkeypatch, tmp_path):
    captured: dict[str, Any] = {}

    def fake_main_rich(**kwargs):
        captured.update(kwargs)

    def fake_get_effective_model_profile(pointer: str):
        if pointer == "missing-model":
            return None
        if pointer == "backup-model":
            return object()
        return object()

    monkeypatch.setattr("ripperdoc.cli.ui.rich_ui.main_rich", fake_main_rich)
    monkeypatch.setattr(cli_module, "check_onboarding", lambda: True)
    monkeypatch.setattr(cli_module, "get_effective_model_profile", fake_get_effective_model_profile)
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli,
        [
            "--model",
            "missing-model",
            "--fallback-model",
            "backup-model",
            "--max-turns",
            "9",
            "--permission-mode",
            "acceptEdits",
            "--disable-slash-commands",
        ],
        env={"HOME": str(tmp_path)},
    )

    assert result.exit_code == 0
    assert captured["model"] == "backup-model"
    assert captured["max_turns"] == 9
    assert captured["permission_mode"] == "acceptEdits"
    assert captured["disable_slash_commands"] is True


def test_cli_short_v_prints_version(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli_module.cli, ["-v"], env={"HOME": str(tmp_path)})

    assert result.exit_code == 0
    assert f"{cli_module.__version__} (Ripperdoc)" in result.output


def test_cli_long_version_prints_version(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli_module.cli, ["--version"], env={"HOME": str(tmp_path)})

    assert result.exit_code == 0
    assert f"{cli_module.__version__} (Ripperdoc)" in result.output


def test_cli_worktree_requires_git_repository(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli,
        ["--worktree", "--prompt", "hi"],
        env={"HOME": str(tmp_path)},
    )

    assert result.exit_code != 0
    assert (
        f"Can only use --worktree in a git repository, but {tmp_path} is not a git repository"
        in result.output
    )


def test_cli_worktree_creates_and_switches_directory(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    def _git(*args: str, cwd: Path = repo) -> None:
        subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    _git("init")
    _git("config", "user.name", "Ripperdoc Test")
    _git("config", "user.email", "test@example.com")
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _git("add", "README.md")
    _git("commit", "-m", "init")

    captured: dict[str, Any] = {}

    async def fake_run_query(*args, **kwargs):
        captured["cwd"] = str(Path.cwd())
        captured["kwargs"] = kwargs

    monkeypatch.setattr(cli_module, "run_query", fake_run_query)
    monkeypatch.setattr(cli_module, "check_onboarding", lambda: True)
    monkeypatch.setattr("ripperdoc.core.tool_defaults.get_default_tools", lambda **_: [])
    monkeypatch.chdir(repo)

    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli,
        ["--worktree", "--prompt", "hi"],
        env={"HOME": str(tmp_path)},
    )

    assert result.exit_code == 0
    assert "Switched to worktree:" in result.output
    assert captured["cwd"].startswith(str(repo / ".ripperdoc" / "worktrees"))


def test_cli_tmux_requires_worktree(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli,
        ["--tmux", "--prompt", "hi"],
        env={"HOME": str(tmp_path)},
    )

    assert result.exit_code != 0
    assert "Error: --tmux requires --worktree" in result.output


def test_cli_worktree_pr_shorthand_hash(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    worktree = repo / ".ripperdoc" / "worktrees" / "pr-123"
    worktree.mkdir(parents=True, exist_ok=True)
    captured: dict[str, Any] = {}

    async def fake_run_query(*args, **kwargs):
        captured["cwd"] = str(Path.cwd())

    def fake_create_task_worktree(*, task_id, base_path, requested_name, pr_number=None):
        captured["requested_name"] = requested_name
        captured["pr_number"] = pr_number
        return SimpleNamespace(worktree_path=worktree)

    monkeypatch.setattr(cli_module, "run_query", fake_run_query)
    monkeypatch.setattr(cli_module, "check_onboarding", lambda: True)
    monkeypatch.setattr("ripperdoc.core.tool_defaults.get_default_tools", lambda **_: [])
    monkeypatch.setattr(cli_module, "get_git_root", lambda _p: repo)
    monkeypatch.setattr(cli_module, "create_task_worktree", fake_create_task_worktree)
    monkeypatch.chdir(repo)

    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli,
        ["--worktree", "--worktree-name", "#123", "--prompt", "hi"],
        env={"HOME": str(tmp_path)},
    )

    assert result.exit_code == 0
    assert captured["requested_name"] == "pr-123"
    assert captured["pr_number"] == 123
    assert captured["cwd"] == str(worktree)


def test_cli_worktree_pr_shorthand_url(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    worktree = repo / ".ripperdoc" / "worktrees" / "pr-456"
    worktree.mkdir(parents=True, exist_ok=True)
    captured: dict[str, Any] = {}

    async def fake_run_query(*args, **kwargs):
        captured["cwd"] = str(Path.cwd())

    def fake_create_task_worktree(*, task_id, base_path, requested_name, pr_number=None):
        captured["requested_name"] = requested_name
        captured["pr_number"] = pr_number
        return SimpleNamespace(worktree_path=worktree)

    monkeypatch.setattr(cli_module, "run_query", fake_run_query)
    monkeypatch.setattr(cli_module, "check_onboarding", lambda: True)
    monkeypatch.setattr("ripperdoc.core.tool_defaults.get_default_tools", lambda **_: [])
    monkeypatch.setattr(cli_module, "get_git_root", lambda _p: repo)
    monkeypatch.setattr(cli_module, "create_task_worktree", fake_create_task_worktree)
    monkeypatch.chdir(repo)

    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli,
        [
            "--worktree",
            "--worktree-name",
            "https://github.com/org/repo/pull/456",
            "--prompt",
            "hi",
        ],
        env={"HOME": str(tmp_path)},
    )

    assert result.exit_code == 0
    assert captured["requested_name"] == "pr-456"
    assert captured["pr_number"] == 456
    assert captured["cwd"] == str(worktree)


def test_main_handles_tmux_worktree_fast_path(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_exec(argv):
        captured["argv"] = list(argv)
        return True, None

    def fail_cli_main(*, args, prog_name):
        raise AssertionError("cli.main should not be called when tmux fast path is handled")

    monkeypatch.setattr(cli_module, "_exec_into_tmux_worktree", fake_exec)
    monkeypatch.setattr(cli_module, "_has_tmux_worktree_flags", lambda _argv: True)
    monkeypatch.setattr(cli_module.cli, "main", fail_cli_main)
    monkeypatch.setattr(
        cli_module.sys,
        "argv",
        ["ripperdoc", "--tmux", "--worktree", "feature-x", "--prompt", "hi"],
    )

    cli_module.main()
    assert captured["argv"] == ["--tmux", "--worktree", "feature-x", "--prompt", "hi"]


def test_main_rewrites_tmux_classic(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_cli_main(*, args, prog_name):
        captured["args"] = list(args)
        captured["prog_name"] = prog_name

    monkeypatch.setattr(cli_module, "_has_tmux_worktree_flags", lambda _argv: False)
    monkeypatch.setattr(cli_module.cli, "main", fake_cli_main)
    monkeypatch.setattr(
        cli_module.sys,
        "argv",
        ["ripperdoc", "--tmux=classic", "--prompt", "hi"],
    )

    cli_module.main()

    assert captured["prog_name"] == "ripperdoc"
    assert "--tmux" in captured["args"]
    assert "--tmux-classic" in captured["args"]


def test_exec_into_tmux_worktree_precreates_and_strips_args(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    worktree = repo / ".ripperdoc" / "worktrees" / "feature-x"
    worktree.mkdir(parents=True, exist_ok=True)
    captured: dict[str, Any] = {"tmux_calls": []}

    def fake_create_task_worktree(*, task_id, base_path, requested_name, pr_number=None):
        captured["create"] = {
            "task_id": task_id,
            "base_path": base_path,
            "requested_name": requested_name,
            "pr_number": pr_number,
        }
        return SimpleNamespace(
            repo_root=repo,
            worktree_path=worktree,
            branch="worktree-feature-x",
            name=requested_name,
            head_commit="abc123",
            hook_based=False,
        )

    def fake_run_tmux_command(args, *, cwd=None, env=None, capture=False):  # noqa: ARG001
        captured["tmux_calls"].append(
            {"args": list(args), "cwd": cwd, "env": dict(env or {})}
        )
        if args[:1] == ["has-session"]:
            return SimpleNamespace(returncode=1)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli_module.shutil, "which", lambda _name: "/usr/bin/tmux")
    monkeypatch.setattr(cli_module, "get_git_root", lambda _p: repo)
    monkeypatch.setattr(cli_module, "create_task_worktree", fake_create_task_worktree)
    monkeypatch.setattr(cli_module, "_run_tmux_command", fake_run_tmux_command)
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.delenv("RIPPERDOC_TMUX_WORKTREE", raising=False)

    handled, error = cli_module._exec_into_tmux_worktree(  # noqa: SLF001
        [
            "--tmux",
            "--worktree",
            "feature-x",
            "--cwd",
            str(repo),
            "--prompt",
            "hi",
        ]
    )

    assert handled is True
    assert error is None
    assert captured["create"]["requested_name"] == "feature-x"
    assert captured["create"]["pr_number"] is None
    assert captured["create"]["base_path"] == repo

    # has-session probe + new-session launch
    assert len(captured["tmux_calls"]) >= 2
    new_session_call = captured["tmux_calls"][-1]
    assert new_session_call["args"][:4] == ["new-session", "-A", "-s", new_session_call["args"][3]]
    assert "-c" in new_session_call["args"]
    c_idx = new_session_call["args"].index("-c")
    assert new_session_call["args"][c_idx + 1] == str(worktree)
    assert new_session_call["cwd"] == worktree

    cmd_tail = new_session_call["args"]
    assert "--tmux" not in cmd_tail
    assert "--worktree" not in cmd_tail
    assert "--cwd" not in cmd_tail

    env_payload = new_session_call["env"]
    assert env_payload.get("RIPPERDOC_TMUX_WORKTREE") == "1"
    assert env_payload.get("RIPPERDOC_PRECREATED_WORKTREE_PATH") == str(worktree)


def test_exec_into_tmux_worktree_pr_shorthand(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    worktree = repo / ".ripperdoc" / "worktrees" / "pr-123"
    worktree.mkdir(parents=True, exist_ok=True)
    captured: dict[str, Any] = {}

    def fake_create_task_worktree(*, task_id, base_path, requested_name, pr_number=None):
        captured["requested_name"] = requested_name
        captured["pr_number"] = pr_number
        return SimpleNamespace(
            repo_root=repo,
            worktree_path=worktree,
            branch="worktree-pr-123",
            name=requested_name,
            head_commit="abc123",
            hook_based=False,
        )

    def fake_run_tmux_command(args, *, cwd=None, env=None, capture=False):  # noqa: ARG001
        if args[:1] == ["has-session"]:
            return SimpleNamespace(returncode=1)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli_module.shutil, "which", lambda _name: "/usr/bin/tmux")
    monkeypatch.setattr(cli_module, "get_git_root", lambda _p: repo)
    monkeypatch.setattr(cli_module, "create_task_worktree", fake_create_task_worktree)
    monkeypatch.setattr(cli_module, "_run_tmux_command", fake_run_tmux_command)
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.delenv("RIPPERDOC_TMUX_WORKTREE", raising=False)

    handled, error = cli_module._exec_into_tmux_worktree(  # noqa: SLF001
        ["--tmux", "--worktree", "#123", "--prompt", "hi"]
    )

    assert handled is True
    assert error is None
    assert captured["requested_name"] == "pr-123"
    assert captured["pr_number"] == 123


def test_register_precreated_worktree_from_env(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    worktree = tmp_path / "worktree"
    repo.mkdir()
    worktree.mkdir()
    captured: dict[str, Any] = {}

    def fake_register(session):
        captured["session"] = session

    monkeypatch.setattr(cli_module, "register_session_worktree", fake_register)
    monkeypatch.setenv("RIPPERDOC_PRECREATED_WORKTREE_PATH", str(worktree))
    monkeypatch.setenv("RIPPERDOC_PRECREATED_WORKTREE_REPO_ROOT", str(repo))
    monkeypatch.setenv("RIPPERDOC_PRECREATED_WORKTREE_NAME", "feature-x")
    monkeypatch.setenv("RIPPERDOC_PRECREATED_WORKTREE_BRANCH", "worktree-feature-x")
    monkeypatch.setenv("RIPPERDOC_PRECREATED_WORKTREE_HEAD_COMMIT", "deadbeef")
    monkeypatch.setenv("RIPPERDOC_PRECREATED_WORKTREE_HOOK_BASED", "0")

    session = cli_module._register_precreated_worktree_from_env()  # noqa: SLF001

    assert session is not None
    assert str(session.worktree_path) == str(worktree.resolve())
    assert str(session.repo_root) == str(repo.resolve())
    assert session.name == "feature-x"
    assert session.branch == "worktree-feature-x"
    assert session.head_commit == "deadbeef"
    assert session.hook_based is False
    assert "session" in captured
    assert "RIPPERDOC_PRECREATED_WORKTREE_PATH" not in cli_module.os.environ


def test_worktree_name_generators_align_patterns():
    cli_name = cli_module.generate_cli_worktree_name()
    assert re.match(r"^(swift|bright|calm|keen|bold)-(fox|owl|elm|oak|ray)-[a-z0-9]{4}$", cli_name)

    from ripperdoc.utils.collaboration.worktree import generate_session_worktree_name

    session_name = generate_session_worktree_name("session-abc")
    assert session_name.startswith("worktree-session-abc-")
    assert re.match(r"^worktree-session-abc-[0-9a-z]+-[a-z0-9]{6}$", session_name)
