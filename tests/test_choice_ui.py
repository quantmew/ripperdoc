from ripperdoc.cli.ui.choice import resolve_choice_style
import pytest

from ripperdoc.cli.ui.choice import prompt_checkbox_async, prompt_choice_async


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
