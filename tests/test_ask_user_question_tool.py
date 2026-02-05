import pytest

from ripperdoc.tools.ask_user_question_tool import (
    BACK_VALUE,
    NEXT_VALUE,
    OptionInput,
    QuestionInput,
    build_question_tabs,
    collect_answers,
    format_question_prompt,
    prompt_multi_choice_with_ui,
    prompt_single_choice_with_ui,
    prompt_user_for_answer,
)


def test_format_question_prompt_single_select():
    question = QuestionInput(
        header="Choose number",
        question="Which number do you want?",
        options=[
            OptionInput(label="1", description="Smallest"),
            OptionInput(label="2", description="Middle"),
            OptionInput(label="3", description="Largest"),
        ],
        multiSelect=False,
    )

    prompt = format_question_prompt(question, question_num=1, total=1)

    assert "[Choose numb...] 1/1" in prompt
    assert "Which number do you want?" in prompt
    assert "1. 1 - Smallest" in prompt
    assert "4. Other (type your own answer)" in prompt
    assert "Select 1-4, 'o' for other, or 'q' to cancel:" in prompt


def test_format_question_prompt_multi_select():
    question = QuestionInput(
        header="Features",
        question="Which features should we enable?",
        options=[
            OptionInput(label="A", description="Feature A"),
            OptionInput(label="B", description="Feature B"),
        ],
        multiSelect=True,
    )

    prompt = format_question_prompt(question, question_num=2, total=3)

    assert "[Features] 2/3" in prompt
    assert "3. Other (type your own answer)" in prompt
    assert "Select options (e.g., 1,3), 'o' for other, or 'q' to cancel:" in prompt


def test_format_question_prompt_single_select_with_back():
    question = QuestionInput(
        header="Step",
        question="Pick one?",
        options=[
            OptionInput(label="1", description=""),
            OptionInput(label="2", description=""),
        ],
        multiSelect=False,
    )

    prompt = format_question_prompt(question, question_num=2, total=3, allow_back=True)

    assert "3. Other (type your own answer)" in prompt
    assert "4. Back to previous question" in prompt
    assert "'b' to go back" in prompt


def test_format_question_prompt_single_select_with_next():
    question = QuestionInput(
        header="Step",
        question="Pick one?",
        options=[
            OptionInput(label="1", description=""),
            OptionInput(label="2", description=""),
        ],
        multiSelect=False,
    )

    prompt = format_question_prompt(
        question, question_num=1, total=3, allow_back=False, allow_next=True
    )

    assert "'n' to next" in prompt


def test_build_question_tabs_highlights_current_step():
    questions = [
        QuestionInput(
            header="Step One",
            question="Q1?",
            options=[OptionInput(label="A", description=""), OptionInput(label="B", description="")],
            multiSelect=False,
        ),
        QuestionInput(
            header="Step Two",
            question="Q2?",
            options=[OptionInput(label="A", description=""), OptionInput(label="B", description="")],
            multiSelect=False,
        ),
    ]

    tabs = build_question_tabs(questions, current_index=1)
    assert "[1. Step One]" in tabs
    assert ">[2. Step Two]<" in tabs


@pytest.mark.asyncio
async def test_prompt_single_choice_with_ui_returns_selected_option(monkeypatch):
    question = QuestionInput(
        header="Pick",
        question="Pick one?",
        options=[
            OptionInput(label="1", description=""),
            OptionInput(label="2", description=""),
        ],
        multiSelect=False,
    )

    captured = {}

    questions = [
        question,
        QuestionInput(
            header="Next",
            question="Another question?",
            options=[OptionInput(label="x", description=""), OptionInput(label="y", description="")],
            multiSelect=False,
        ),
    ]

    async def fake_prompt_choice_async(*args, **kwargs):  # noqa: ANN002, ANN003
        captured["style_variant"] = kwargs.get("style_variant")
        captured["message"] = kwargs.get("message")
        captured["external_header"] = kwargs.get("external_header")
        captured["custom_input_label"] = kwargs.get("custom_input_label")
        return "2"

    monkeypatch.setattr(
        "ripperdoc.tools.ask_user_question_tool.prompt_choice_async", fake_prompt_choice_async
    )

    answer = await prompt_single_choice_with_ui(
        question, question_num=1, total=2, all_questions=questions
    )
    assert answer == "2"
    assert captured["style_variant"] == "ask_user_question"
    assert ">[1. Pick]<" in captured["external_header"]
    assert "[2. Next]" in captured["external_header"]
    assert captured["custom_input_label"] == "Other"


@pytest.mark.asyncio
async def test_prompt_single_choice_with_ui_supports_inline_custom_input(monkeypatch):
    question = QuestionInput(
        header="Pick",
        question="Pick one?",
        options=[
            OptionInput(label="1", description=""),
            OptionInput(label="2", description=""),
        ],
        multiSelect=False,
    )

    async def fake_prompt_choice_async(*args, **kwargs):  # noqa: ANN002, ANN003
        return "custom text"

    monkeypatch.setattr(
        "ripperdoc.tools.ask_user_question_tool.prompt_choice_async", fake_prompt_choice_async
    )

    answer = await prompt_single_choice_with_ui(question, question_num=1, total=1)
    assert answer == "custom text"


@pytest.mark.asyncio
async def test_prompt_single_choice_with_ui_back(monkeypatch):
    question = QuestionInput(
        header="Pick",
        question="Pick one?",
        options=[
            OptionInput(label="1", description=""),
            OptionInput(label="2", description=""),
        ],
        multiSelect=False,
    )

    async def fake_prompt_choice_async(*args, **kwargs):  # noqa: ANN002, ANN003
        return BACK_VALUE

    monkeypatch.setattr(
        "ripperdoc.tools.ask_user_question_tool.prompt_choice_async", fake_prompt_choice_async
    )

    answer = await prompt_single_choice_with_ui(
        question, question_num=2, total=3, allow_back=True
    )
    assert answer == BACK_VALUE


@pytest.mark.asyncio
async def test_prompt_user_for_answer_falls_back_when_choice_ui_fails(monkeypatch):
    question = QuestionInput(
        header="Pick",
        question="Pick one?",
        options=[
            OptionInput(label="1", description=""),
            OptionInput(label="2", description=""),
        ],
        multiSelect=False,
    )

    async def fake_prompt_choice_async(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("choice ui unavailable")

    responses = iter(["2"])

    def fake_prompt(*args, **kwargs):  # noqa: ANN002, ANN003
        return next(responses)

    monkeypatch.setattr(
        "ripperdoc.tools.ask_user_question_tool.prompt_choice_async", fake_prompt_choice_async
    )
    monkeypatch.setattr("prompt_toolkit.prompt", fake_prompt)

    answer = await prompt_user_for_answer(question, question_num=1, total=1)
    assert answer == "2"


@pytest.mark.asyncio
async def test_prompt_multi_choice_with_ui_returns_selected_options(monkeypatch):
    question = QuestionInput(
        header="Features",
        question="Which features should we enable?",
        options=[
            OptionInput(label="A", description=""),
            OptionInput(label="B", description=""),
            OptionInput(label="C", description=""),
        ],
        multiSelect=True,
    )

    captured = {}

    async def fake_prompt_checkbox_async(*args, **kwargs):  # noqa: ANN002, ANN003
        captured["style_variant"] = kwargs.get("style_variant")
        captured["custom_input_label"] = kwargs.get("custom_input_label")
        captured["next_value"] = kwargs.get("next_value")
        return ["A", "C"]

    monkeypatch.setattr(
        "ripperdoc.tools.ask_user_question_tool.prompt_checkbox_async", fake_prompt_checkbox_async
    )

    answer = await prompt_multi_choice_with_ui(question, question_num=1, total=1)
    assert answer == "A, C"
    assert captured["style_variant"] == "ask_user_question"
    assert captured["custom_input_label"] == "Other"
    assert captured["next_value"] is None


@pytest.mark.asyncio
async def test_prompt_multi_choice_with_ui_supports_inline_custom_input(monkeypatch):
    question = QuestionInput(
        header="Features",
        question="Which features should we enable?",
        options=[
            OptionInput(label="A", description=""),
            OptionInput(label="B", description=""),
        ],
        multiSelect=True,
    )

    async def fake_prompt_checkbox_async(*args, **kwargs):  # noqa: ANN002, ANN003
        return ["A", "Custom feature"]

    monkeypatch.setattr(
        "ripperdoc.tools.ask_user_question_tool.prompt_checkbox_async", fake_prompt_checkbox_async
    )

    answer = await prompt_multi_choice_with_ui(question, question_num=1, total=1)
    assert answer == "A, Custom feature"


@pytest.mark.asyncio
async def test_prompt_multi_choice_with_ui_back_from_left_key(monkeypatch):
    question = QuestionInput(
        header="Features",
        question="Which features should we enable?",
        options=[
            OptionInput(label="A", description=""),
            OptionInput(label="B", description=""),
        ],
        multiSelect=True,
    )

    captured = {}

    async def fake_prompt_checkbox_async(*args, **kwargs):  # noqa: ANN002, ANN003
        captured["back_value"] = kwargs.get("back_value")
        return [BACK_VALUE]

    monkeypatch.setattr(
        "ripperdoc.tools.ask_user_question_tool.prompt_checkbox_async", fake_prompt_checkbox_async
    )

    answer = await prompt_multi_choice_with_ui(question, question_num=2, total=3, allow_back=True)
    assert answer == BACK_VALUE
    assert captured["back_value"] == BACK_VALUE


@pytest.mark.asyncio
async def test_prompt_multi_choice_with_ui_next_from_right_key(monkeypatch):
    question = QuestionInput(
        header="Features",
        question="Which features should we enable?",
        options=[
            OptionInput(label="A", description=""),
            OptionInput(label="B", description=""),
        ],
        multiSelect=True,
    )

    captured = {}

    async def fake_prompt_checkbox_async(*args, **kwargs):  # noqa: ANN002, ANN003
        captured["next_value"] = kwargs.get("next_value")
        return [NEXT_VALUE]

    monkeypatch.setattr(
        "ripperdoc.tools.ask_user_question_tool.prompt_checkbox_async", fake_prompt_checkbox_async
    )

    answer = await prompt_multi_choice_with_ui(question, question_num=1, total=3, allow_next=True)
    assert answer == NEXT_VALUE
    assert captured["next_value"] == NEXT_VALUE


@pytest.mark.asyncio
async def test_prompt_user_for_answer_multiselect_falls_back_when_checkbox_ui_fails(monkeypatch):
    question = QuestionInput(
        header="Features",
        question="Which features should we enable?",
        options=[
            OptionInput(label="A", description=""),
            OptionInput(label="B", description=""),
        ],
        multiSelect=True,
    )

    async def fake_prompt_checkbox_async(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("checkbox ui unavailable")

    responses = iter(["1,2"])

    def fake_prompt(*args, **kwargs):  # noqa: ANN002, ANN003
        return next(responses)

    monkeypatch.setattr(
        "ripperdoc.tools.ask_user_question_tool.prompt_checkbox_async", fake_prompt_checkbox_async
    )
    monkeypatch.setattr("prompt_toolkit.prompt", fake_prompt)

    answer = await prompt_user_for_answer(question, question_num=1, total=1)
    assert answer == "A, B"


@pytest.mark.asyncio
async def test_collect_answers_supports_back_navigation(monkeypatch):
    questions = [
        QuestionInput(
            header="One",
            question="Q1?",
            options=[OptionInput(label="A", description=""), OptionInput(label="B", description="")],
            multiSelect=False,
        ),
        QuestionInput(
            header="Two",
            question="Q2?",
            options=[OptionInput(label="X", description=""), OptionInput(label="Y", description="")],
            multiSelect=False,
        ),
    ]

    responses = iter(["A", BACK_VALUE, "B", "Y"])

    async def fake_prompt_user_for_answer(*args, **kwargs):  # noqa: ANN002, ANN003
        return next(responses)

    monkeypatch.setattr(
        "ripperdoc.tools.ask_user_question_tool.prompt_user_for_answer", fake_prompt_user_for_answer
    )
    async def fake_confirm_submit(*args, **kwargs):  # noqa: ANN002, ANN003
        return "submit"

    monkeypatch.setattr(
        "ripperdoc.tools.ask_user_question_tool.prompt_choice_async",
        fake_confirm_submit,
    )

    answers, cancelled = await collect_answers(questions, {})
    assert cancelled is False
    assert answers == {"Q1?": "B", "Q2?": "Y"}


@pytest.mark.asyncio
async def test_collect_answers_allows_skip_with_next_and_submits(monkeypatch):
    questions = [
        QuestionInput(
            header="One",
            question="Q1?",
            options=[OptionInput(label="A", description=""), OptionInput(label="B", description="")],
            multiSelect=False,
        ),
        QuestionInput(
            header="Two",
            question="Q2?",
            options=[OptionInput(label="X", description=""), OptionInput(label="Y", description="")],
            multiSelect=False,
        ),
    ]

    responses = iter([NEXT_VALUE, NEXT_VALUE])

    async def fake_prompt_user_for_answer(*args, **kwargs):  # noqa: ANN002, ANN003
        return next(responses)

    monkeypatch.setattr(
        "ripperdoc.tools.ask_user_question_tool.prompt_user_for_answer", fake_prompt_user_for_answer
    )
    async def fake_confirm_submit(*args, **kwargs):  # noqa: ANN002, ANN003
        return "submit"

    monkeypatch.setattr(
        "ripperdoc.tools.ask_user_question_tool.prompt_choice_async",
        fake_confirm_submit,
    )

    answers, cancelled = await collect_answers(questions, {})
    assert cancelled is False
    assert answers == {}


@pytest.mark.asyncio
async def test_collect_answers_cancelled_at_confirmation(monkeypatch):
    questions = [
        QuestionInput(
            header="One",
            question="Q1?",
            options=[OptionInput(label="A", description=""), OptionInput(label="B", description="")],
            multiSelect=False,
        ),
    ]

    async def fake_prompt_user_for_answer(*args, **kwargs):  # noqa: ANN002, ANN003
        return "A"

    monkeypatch.setattr(
        "ripperdoc.tools.ask_user_question_tool.prompt_user_for_answer", fake_prompt_user_for_answer
    )
    async def fake_confirm_cancel(*args, **kwargs):  # noqa: ANN002, ANN003
        return "cancel"

    monkeypatch.setattr(
        "ripperdoc.tools.ask_user_question_tool.prompt_choice_async",
        fake_confirm_cancel,
    )

    answers, cancelled = await collect_answers(questions, {})
    assert answers == {"Q1?": "A"}
    assert cancelled is True
