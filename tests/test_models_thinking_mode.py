"""Tests for /models thinking mode editing paths."""

from __future__ import annotations

from types import SimpleNamespace

from rich.console import Console

from ripperdoc.cli.commands import models_cmd
from ripperdoc.cli.ui.models_tui.textual_app import (
    ModelFormScreen,
    ModelsApp,
    _next_copied_profile_name,
)
from ripperdoc.core.config import ModelProfile, ProtocolType
from ripperdoc.core.oauth import OAuthToken, OAuthTokenType


class _FakeConsole:
    def __init__(self, inputs: list[str]) -> None:
        self._inputs = list(inputs)
        self.printed: list[str] = []

    def input(self, _prompt: str = "") -> str:
        if not self._inputs:
            raise AssertionError("No more fake console inputs available")
        return self._inputs.pop(0)

    def print(self, message: str, *args: object, **kwargs: object) -> None:  # noqa: ARG002
        self.printed.append(message)


def test_collect_add_profile_input_sets_thinking_mode(monkeypatch) -> None:
    console = _FakeConsole(
        [
            "openai_compatible",  # protocol
            "openrouter/demo",  # model
            "",  # api_base
            "",  # max_input_tokens
            "",  # max_output_tokens
            "",  # max_tokens
            "",  # temperature
            "openrouter",  # thinking_mode
            "auto",  # supports_vision
            "",  # currency
            "",  # input_price
            "",  # output_price
            "",  # set_as_main
        ]
    )
    monkeypatch.setattr(models_cmd, "prompt_secret", lambda _prompt: "")

    config = SimpleNamespace(
        model_profiles={},
        model_pointers=SimpleNamespace(main="default"),
    )
    profile, set_as_main = models_cmd._collect_add_profile_input(
        console,
        config,
        existing_profile=None,
        current_profile=None,
    )

    assert profile is not None
    assert profile.thinking_mode == "openrouter"
    assert set_as_main is True


def test_collect_edit_profile_input_can_clear_thinking_mode(monkeypatch) -> None:
    existing = ModelProfile(
        protocol=ProtocolType.OPENAI_COMPATIBLE,
        model="openrouter/demo",
        thinking_mode="deepseek",
    )
    console = _FakeConsole(
        [
            "",  # protocol (keep existing)
            "",  # model name
            "",  # api_base
            "",  # max_input_tokens
            "",  # max_output_tokens
            "",  # max_tokens
            "",  # temperature
            "-",  # thinking_mode clear
            "",  # supports_vision
            "",  # currency
            "",  # input_price
            "",  # output_price
        ]
    )
    monkeypatch.setattr(models_cmd, "prompt_secret", lambda _prompt: "")

    updated = models_cmd._collect_edit_profile_input(console, existing)

    assert updated is not None
    assert updated.thinking_mode is None


def test_models_tui_parse_thinking_mode_values() -> None:
    screen = ModelFormScreen("edit")

    assert screen._thinking_mode_form_defaults(None) == ("auto", "")
    assert screen._thinking_mode_form_defaults("openrouter") == ("openrouter", "")
    assert screen._thinking_mode_form_defaults("vendor_custom") == ("__custom__", "vendor_custom")

    assert screen._resolve_thinking_mode(selected_value="auto", custom_value="") is None
    assert (
        screen._resolve_thinking_mode(selected_value="openrouter", custom_value="")
        == "openrouter"
    )
    assert (
        screen._resolve_thinking_mode(selected_value="__custom__", custom_value="Vendor_Mode")
        == "vendor_mode"
    )


def test_models_tui_copy_name_increments() -> None:
    assert _next_copied_profile_name("x", {"x"}) == "x (1)"
    assert _next_copied_profile_name("x", {"x", "x (1)"}) == "x (2)"
    assert _next_copied_profile_name("x", {"x", "x (1)", "x (2)"}) == "x (3)"


def test_models_cmd_copy_name_increments() -> None:
    assert models_cmd._next_copied_profile_name("x", {"x"}) == "x (1)"
    assert models_cmd._next_copied_profile_name("x", {"x", "x (1)"}) == "x (2)"


def test_models_tui_action_copy_duplicates_profile(monkeypatch) -> None:
    source_profile = ModelProfile(
        protocol=ProtocolType.OPENAI_COMPATIBLE,
        model="openrouter/demo",
        max_tokens=1024,
    )
    config = SimpleNamespace(
        model_profiles={"x": source_profile, "x (1)": source_profile.model_copy(deep=True)},
    )
    monkeypatch.setattr(
        "ripperdoc.cli.ui.models_tui.textual_app.get_global_config",
        lambda: config,
    )

    captured: dict[str, object] = {}

    def _fake_add_model_profile(name, profile, overwrite=False, set_as_main=False):  # noqa: ANN001
        captured["name"] = name
        captured["profile"] = profile
        captured["overwrite"] = overwrite
        captured["set_as_main"] = set_as_main

    monkeypatch.setattr(
        "ripperdoc.cli.ui.models_tui.textual_app.add_model_profile",
        _fake_add_model_profile,
    )

    app = ModelsApp()
    app._selected_name = "x"
    monkeypatch.setattr(app, "_set_status", lambda message: captured.setdefault("status", message))
    monkeypatch.setattr(
        app,
        "_refresh_models",
        lambda select_first: captured.setdefault("refresh_select_first", select_first),
    )

    app.action_copy()

    assert captured["name"] == "x (2)"
    assert captured["overwrite"] is False
    assert captured["set_as_main"] is False
    assert isinstance(captured["profile"], ModelProfile)
    assert captured["profile"] is not source_profile
    assert captured["profile"].model_dump() == source_profile.model_dump()
    assert app._selected_name == "x (2)"


def test_models_tui_handle_rename_result_renames_profile(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_rename_model_profile(old_name, new_name):  # noqa: ANN001
        captured["old_name"] = old_name
        captured["new_name"] = new_name

    monkeypatch.setattr(
        "ripperdoc.cli.ui.models_tui.textual_app.rename_model_profile",
        _fake_rename_model_profile,
    )

    app = ModelsApp()
    app._selected_name = "demo"
    monkeypatch.setattr(app, "_set_status", lambda message: captured.setdefault("status", message))
    monkeypatch.setattr(
        app,
        "_refresh_models",
        lambda select_first: captured.setdefault("refresh_select_first", select_first),
    )

    app._handle_rename_result("demo-v2")

    assert captured["old_name"] == "demo"
    assert captured["new_name"] == "demo-v2"
    assert app._selected_name == "demo-v2"
    assert captured["status"] == "Renamed profile to demo-v2."
    assert captured["refresh_select_first"] is False


def test_models_tui_handle_rename_result_shows_notify_on_duplicate(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_rename_model_profile(_old_name, _new_name):  # noqa: ANN001
        raise ValueError("Model profile 'demo-v2' already exists.")

    monkeypatch.setattr(
        "ripperdoc.cli.ui.models_tui.textual_app.rename_model_profile",
        _fake_rename_model_profile,
    )

    app = ModelsApp()
    app._selected_name = "demo"
    monkeypatch.setattr(app, "_set_status", lambda message: captured.setdefault("status", message))
    monkeypatch.setattr(
        app,
        "notify",
        lambda message, title, severity, timeout: captured.setdefault(
            "notify",
            {
                "message": message,
                "title": title,
                "severity": severity,
                "timeout": timeout,
            },
        ),
    )

    app._handle_rename_result("demo-v2")

    assert captured["status"] == "Model profile 'demo-v2' already exists."
    assert captured["notify"] == {
        "message": "Model profile 'demo-v2' already exists.",
        "title": "Rename failed",
        "severity": "error",
        "timeout": 6,
    }
    assert app._selected_name == "demo"


def test_models_tui_oauth_token_options_include_non_codex(monkeypatch) -> None:
    screen = ModelFormScreen("add")
    monkeypatch.setattr(
        "ripperdoc.cli.ui.models_tui.textual_app.list_oauth_tokens",
        lambda: {
            "copilot-main": OAuthToken(
                type=OAuthTokenType.COPILOT,
                access_token="abcd1234efgh",
            ),
            "gitlab-main": OAuthToken(
                type=OAuthTokenType.GITLAB,
                access_token="ijkl1234mnop",
            ),
        },
    )

    options = screen._oauth_token_select_options(None)
    values = [value for _label, value in options]
    assert "copilot-main" in values
    assert "gitlab-main" in values


def test_render_models_table_includes_thinking_mode_column() -> None:
    profile = ModelProfile(
        protocol=ProtocolType.OPENAI_COMPATIBLE,
        model="openrouter/demo",
        thinking_mode="openrouter",
    )
    config = SimpleNamespace(
        model_profiles={"demo": profile},
        model_pointers=SimpleNamespace(model_dump=lambda: {"main": "demo", "quick": "demo"}),
    )
    console = Console(record=True, width=140)

    models_cmd._render_models_table(console, config)

    output = console.export_text()
    assert "Think" in output
    assert "openrouter" in output


def test_collect_add_profile_input_oauth_uses_token_and_curated_model(monkeypatch) -> None:
    console = _FakeConsole(
        [
            "oauth",  # protocol
            "1",  # oauth token
            "2",  # oauth model
            "",  # max_input_tokens
            "",  # max_output_tokens
            "",  # max_tokens
            "",  # temperature
            "",  # thinking_mode
            "auto",  # supports_vision
            "",  # currency
            "",  # input_price
            "",  # output_price
            "",  # set_as_main
        ]
    )
    monkeypatch.setattr(models_cmd, "prompt_secret", lambda _prompt: "")
    monkeypatch.setattr(
        models_cmd,
        "list_oauth_tokens",
        lambda: {
            "codex-main": OAuthToken(
                type=OAuthTokenType.CODEX,
                access_token="abcd1234efgh",
            )
        },
    )

    config = SimpleNamespace(
        model_profiles={},
        model_pointers=SimpleNamespace(main="default"),
    )
    profile, _ = models_cmd._collect_add_profile_input(
        console,
        config,
        existing_profile=None,
        current_profile=None,
    )

    assert profile is not None
    assert profile.protocol == ProtocolType.OAUTH
    assert profile.oauth_token_name == "codex-main"
    assert profile.oauth_token_type == OAuthTokenType.CODEX
    assert profile.model == "gpt-5.3-codex-spark"


def test_collect_add_profile_input_oauth_supports_non_codex_token(monkeypatch) -> None:
    console = _FakeConsole(
        [
            "oauth",  # protocol
            "1",  # oauth token
            "2",  # oauth model
            "",  # max_input_tokens
            "",  # max_output_tokens
            "",  # max_tokens
            "",  # temperature
            "",  # thinking_mode
            "auto",  # supports_vision
            "",  # currency
            "",  # input_price
            "",  # output_price
            "",  # set_as_main
        ]
    )
    monkeypatch.setattr(models_cmd, "prompt_secret", lambda _prompt: "")
    monkeypatch.setattr(
        models_cmd,
        "list_oauth_tokens",
        lambda: {
            "gitlab-main": OAuthToken(
                type=OAuthTokenType.GITLAB,
                access_token="abcd1234efgh",
            )
        },
    )

    config = SimpleNamespace(
        model_profiles={},
        model_pointers=SimpleNamespace(main="default"),
    )
    profile, _ = models_cmd._collect_add_profile_input(
        console,
        config,
        existing_profile=None,
        current_profile=None,
    )

    assert profile is not None
    assert profile.protocol == ProtocolType.OAUTH
    assert profile.oauth_token_name == "gitlab-main"
    assert profile.oauth_token_type == OAuthTokenType.GITLAB
    assert profile.model == "duo-chat-sonnet-4-5"
