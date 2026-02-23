from ripperdoc.cli.ui.choice import onboarding_style, resolve_choice_style
import pytest

from ripperdoc.cli.ui.choice import prompt_checkbox_async, prompt_choice_async
from ripperdoc.core.theme import get_theme_manager


def _style_dict(style) -> dict[str, str]:
    return dict(style.style_rules)


def test_resolve_choice_style_default_is_neutral():
    style = _style_dict(resolve_choice_style("neutral"))
    assert style["frame.border"] == "#7f8fa6"
    assert style["question"] == "#e8ecf5"


def test_resolve_choice_style_supports_amber():
    style = _style_dict(resolve_choice_style("amber"))
    assert style["frame.border"] == "#d4a017"
    assert style["question"] == "#ffd700"


def test_resolve_choice_style_unknown_falls_back_to_neutral():
    style = _style_dict(resolve_choice_style("not-a-style"))
    assert style["frame.border"] == "#7f8fa6"
    assert style["question"] == "#e8ecf5"


def test_choice_diff_styles_follow_active_theme():
    manager = get_theme_manager()
    original_theme = manager.current.name
    try:
        manager.set_theme("dark")
        dark_style = _style_dict(resolve_choice_style("neutral"))

        manager.set_theme("light")
        light_style = _style_dict(resolve_choice_style("neutral"))

        assert dark_style["diff-add"] != light_style["diff-add"]
        assert dark_style["diff-del"] != light_style["diff-del"]
        assert dark_style["diff-hunk"] != light_style["diff-hunk"]
    finally:
        manager.set_theme(original_theme)


def test_onboarding_style_defaults_to_dark_palette(monkeypatch):
    monkeypatch.delenv("RIPPERDOC_TERMINAL_BG", raising=False)
    monkeypatch.delenv("COLORFGBG", raising=False)

    style = _style_dict(onboarding_style())
    assert style["option"] == "#f8f8f2"
    assert style["value"] == "#f8f8f2"


def test_onboarding_style_uses_light_palette_when_forced(monkeypatch):
    monkeypatch.setenv("RIPPERDOC_TERMINAL_BG", "light")
    monkeypatch.delenv("COLORFGBG", raising=False)

    style = _style_dict(onboarding_style())
    assert style["option"] == "#1f2937"
    assert style["value"] == "#111827"


def test_onboarding_style_detects_light_background_from_colorfgbg(monkeypatch):
    monkeypatch.delenv("RIPPERDOC_TERMINAL_BG", raising=False)
    monkeypatch.setenv("COLORFGBG", "15;7")

    style = _style_dict(onboarding_style())
    assert style["option"] == "#1f2937"
    assert style["yes-option"] == "#111827"


@pytest.mark.asyncio
async def test_prompt_checkbox_async_uses_non_fullscreen_app(monkeypatch):
    captured = {}

    class FakeApplication:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            captured["full_screen"] = kwargs.get("full_screen")
            captured["style"] = kwargs.get("style")

        async def run_async(self):
            return ["a"]

    monkeypatch.setattr("ripperdoc.cli.ui.choice.Application", FakeApplication)

    result = await prompt_checkbox_async(
        message="Pick many",
        options=[("a", "A"), ("b", "B")],
        style_variant="neutral",
    )

    assert captured["full_screen"] is False
    assert captured["style"] is not None
    assert result == ["a"]


@pytest.mark.asyncio
async def test_prompt_checkbox_async_cancel_returns_none(monkeypatch):
    class FakeApplication:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            pass

        async def run_async(self):
            return None

    monkeypatch.setattr("ripperdoc.cli.ui.choice.Application", FakeApplication)

    result = await prompt_checkbox_async(
        message="Pick many",
        options=[("a", "A"), ("b", "B")],
        style_variant="neutral",
    )

    assert result is None


@pytest.mark.asyncio
async def test_prompt_checkbox_async_with_custom_input_label_initializes(monkeypatch):
    class FakeApplication:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            pass

        async def run_async(self):
            return ["a", "custom"]

    monkeypatch.setattr("ripperdoc.cli.ui.choice.Application", FakeApplication)

    result = await prompt_checkbox_async(
        message="Pick many",
        options=[("a", "A"), ("b", "B")],
        style_variant="neutral",
        custom_input_label="Other",
    )

    assert result == ["a", "custom"]


@pytest.mark.asyncio
async def test_prompt_choice_async_with_custom_input_label_initializes(monkeypatch):
    captured = {}

    class FakeApplication:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            captured["full_screen"] = kwargs.get("full_screen")
            captured["style"] = kwargs.get("style")

        async def run_async(self):
            return "custom text"

    monkeypatch.setattr("ripperdoc.cli.ui.choice.Application", FakeApplication)

    result = await prompt_choice_async(
        message="Pick one",
        options=[("a", "A"), ("b", "B")],
        style_variant="neutral",
        custom_input_label="Other",
    )

    assert captured["full_screen"] is False
    assert captured["style"] is not None
    assert result == "custom text"
