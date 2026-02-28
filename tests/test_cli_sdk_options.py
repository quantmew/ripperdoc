"""CLI/stdio option propagation tests for SDK compatibility flags."""

from __future__ import annotations

from click.testing import CliRunner
import pytest
from typing import Any

from ripperdoc.cli import cli as cli_module
from ripperdoc.protocol.stdio import command as stdio_command


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
            "--fork-session",
            "--agent",
            "reviewer",
            "--agents",
            '{"reviewer":{"description":"Reviews code","prompt":"You are a code reviewer"}}',
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
    assert default_options["fork_session"] is True
    assert default_options["agent"] == "reviewer"
    assert default_options["agents"]["reviewer"]["prompt"] == "You are a code reviewer"
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

    init_options = captured["initialize"]["request"]["options"]
    assert init_options["max_thinking_tokens"] == 128
    assert init_options["json_schema"] == '{"type":"object"}'
    assert init_options["sdk_can_use_tool"] is False


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


def test_cli_agent_overrides_settings_agent_for_prompt(monkeypatch, tmp_path):
    captured: dict[str, Any] = {}

    async def fake_run_query(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(cli_module, "run_query", fake_run_query)
    monkeypatch.setattr(cli_module, "check_onboarding", lambda: True)
    monkeypatch.setattr(cli_module, "get_default_tools", lambda **_: [])
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
    assert captured["kwargs"]["append_system_prompt"] == "Use writer behavior from CLI."


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


def test_cli_prompt_forwards_permission_mode_and_max_turns(monkeypatch, tmp_path):
    captured: dict[str, Any] = {}

    async def fake_run_query(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(cli_module, "run_query", fake_run_query)
    monkeypatch.setattr(cli_module, "check_onboarding", lambda: True)
    monkeypatch.setattr(cli_module, "get_default_tools", lambda **_: [])
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
    monkeypatch.setattr(cli_module, "get_default_tools", lambda **_: [])
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
    monkeypatch.setattr(cli_module, "get_default_tools", lambda **_: [])
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
        ],
        env={"HOME": str(tmp_path)},
    )

    assert result.exit_code == 0
    assert captured["model"] == "backup-model"
    assert captured["max_turns"] == 9
    assert captured["permission_mode"] == "acceptEdits"


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
