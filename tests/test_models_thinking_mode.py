"""Tests for /models thinking mode editing paths."""

from __future__ import annotations

from types import SimpleNamespace

from rich.console import Console

from ripperdoc.cli.commands import models_cmd
from ripperdoc.cli.ui.models_tui.textual_app import ModelFormScreen
from ripperdoc.core.config import ModelProfile, ProtocolType


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
