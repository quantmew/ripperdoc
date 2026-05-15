"""Ask user question tool for interactive clarification."""

from __future__ import annotations

import asyncio
import html
import os
import sys
from typing import AsyncGenerator, Dict, List, Optional

from pydantic import BaseModel, Field

from ripperdoc.cli.ui.choice import (
    ChoiceOption,
    prompt_checkbox_async,
    prompt_choice_async,
)
from ripperdoc.core.tool import (
    Tool,
    ToolOutput,
    ToolResult,
    ToolUseContext,
    ValidationResult,
)
from ripperdoc.utils.log import get_logger
from ripperdoc.tools.ask_user_question._constants import (
    ANSI_RESET,
    ANSI_TAB_ACTIVE_BG,
    ANSI_TAB_ACTIVE_FG,
    ANSI_TAB_INACTIVE_FG,
    BACK_VALUE,
    CHOICE_UI_FALLBACK,
    HEADER_MAX_CHARS,
    NEXT_VALUE,
    TOOL_NAME,
)
from ripperdoc.tools.ask_user_question._prompt import ASK_USER_QUESTION_PROMPT

logger = get_logger()


class OptionInput(BaseModel):
    """Single option for a question."""

    label: str = Field(
        description="The display text for this option that the user will see and select. "
        "Should be concise (1-5 words) and clearly describe the choice."
    )
    description: str = Field(
        description="Explanation of what this option means or what will happen if chosen. "
        "Useful for providing context about trade-offs or implications."
    )


class QuestionInput(BaseModel):
    """Single question to ask the user."""

    question: str = Field(
        description="The complete question to ask the user. Should be clear, specific, and end with a question mark. "
        "Example: \"Which library should we use for date formatting?\" If multiSelect is true, phrase it accordingly, "
        "e.g. \"Which features do you want to enable?\""
    )
    header: str = Field(
        description="Very short label displayed as a chip/tag (max 12 chars). "
        "Examples: \"Auth method\", \"Library\", \"Approach\"."
    )
    options: List[OptionInput] = Field(
        description="The available choices for this question. Must have 2-4 options. "
        "Each option should be a distinct, mutually exclusive choice (unless multiSelect is enabled). "
        "There should be no 'Other' option, that will be provided automatically.",
        min_length=2,
        max_length=4,
    )
    multiSelect: bool = Field(
        default=False,
        description="Set to true to allow the user to select multiple options instead of just one. "
        "Use when choices are not mutually exclusive.",
    )


class AskUserQuestionToolInput(BaseModel):
    """Input schema for AskUserQuestionTool."""

    questions: List[QuestionInput] = Field(
        description="Questions to ask the user (1-4 questions)",
        min_length=1,
        max_length=4,
    )
    answers: Optional[Dict[str, str]] = Field(
        default=None,
        description="User answers collected by the permission component",
    )


class AskUserQuestionToolOutput(BaseModel):
    """Output from asking the user."""

    answers: Dict[str, str]
    selected_labels: Dict[str, List[str]] = Field(default_factory=dict)


class AskUserQuestionTool(Tool[AskUserQuestionToolInput, AskUserQuestionToolOutput]):
    """Tool for interactively asking the user questions."""

    @property
    def name(self) -> str:
        return TOOL_NAME

    async def description(self) -> str:
        return ASK_USER_QUESTION_PROMPT

    @property
    def input_schema(self) -> type[AskUserQuestionToolInput]:
        return AskUserQuestionToolInput

    def is_read_only(self) -> bool:
        return True

    def is_concurrency_safe(self) -> bool:
        return False

    def needs_permissions(self, _input_data: Optional[AskUserQuestionToolInput] = None) -> bool:
        return False

    async def validate_input(
        self,
        input_data: AskUserQuestionToolInput,
        _context: Optional[ToolUseContext] = None,
    ) -> ValidationResult:
        for i, q in enumerate(input_data.questions):
            if len(q.header) > HEADER_MAX_CHARS:
                return ValidationResult(
                    result=False,
                    message=f"Question {i+1} header is too long ({len(q.header)} chars). "
                    f"Max {HEADER_MAX_CHARS} chars.",
                )
            if not q.question.strip():
                return ValidationResult(
                    result=False,
                    message=f"Question {i+1} has an empty question text.",
                )
            if not q.options:
                return ValidationResult(
                    result=False,
                    message=f"Question {i+1} has no options.",
                )
        return ValidationResult(result=True)

    def render_result_for_assistant(self, output: AskUserQuestionToolOutput) -> str:
        lines = ["**User answers:**"]
        for q_idx, (question_text, answer) in enumerate(output.answers.items(), 1):
            lines.append(f"  Q{q_idx}: {question_text}")
            lines.append(f"  A{q_idx}: {answer}")
        return "\n".join(lines)

    def render_tool_use_message(
        self,
        input_data: AskUserQuestionToolInput,
        _verbose: bool = False,
    ) -> str:
        q_count = len(input_data.questions)
        q_text = input_data.questions[0].question[:80] + "..." if q_count > 0 else ""
        return f"Asking {q_count} question(s): {q_text}"

    async def call(
        self,
        input_data: AskUserQuestionToolInput,
        context: ToolUseContext,
    ) -> AsyncGenerator[ToolOutput, None]:
        answers: Dict[str, str] = {}
        selected_labels: Dict[str, List[str]] = {}

        # If user already answered via permission component, use those answers
        if input_data.answers:
            answers = input_data.answers
            yield ToolResult(
                data=AskUserQuestionToolOutput(answers=answers),
                result_for_assistant=self.render_result_for_assistant(
                    AskUserQuestionToolOutput(answers=answers)
                ),
            )
            return

        for q_idx, question in enumerate(input_data.questions):
            # Build choices
            choices = [
                ChoiceOption(label=opt.label, description=opt.description)
                for opt in question.options
            ]

            if question.multiSelect:
                fallback_value = CHOICE_UI_FALLBACK
                choices.append(ChoiceOption(label="Done", value=NEXT_VALUE, description=""))
            else:
                fallback_value = str(id(question))

            # Display the question
            header_tag = f"{ANSI_TAB_ACTIVE_BG}{ANSI_TAB_ACTIVE_FG} {question.header} {ANSI_RESET}"
            separator = "─" * min(60, os.get_terminal_size().columns - 10) if sys.stdout.isatty() else "-"
            if q_idx > 0:
                print()

            yield ToolResult(
                data=AskUserQuestionToolOutput(answers=answers),
                result_for_assistant=self.render_result_for_assistant(
                    AskUserQuestionToolOutput(answers=answers)
                ),
            )

            # Wait for the previous result to flush before showing the prompt
            await asyncio.sleep(0.05)

            if question.multiSelect:
                print(f"\n{separator}")
                print(f"\n{header_tag}  {question.question}")
                selected = await prompt_checkbox_async(
                    question=question.question,
                    choices=choices,
                )
                selected = [s for s in selected if s != NEXT_VALUE]
                selected_labels[question.question] = selected
                answers[question.question] = ", ".join(selected) if selected else "(none selected)"
            else:
                print(f"\n{separator}")
                print(f"\n{header_tag}  {question.question}")
                selected_label = await prompt_choice_async(
                    question=question.question,
                    choices=choices,
                )
                selected_labels[question.question] = [selected_label]
                answers[question.question] = selected_label

        yield ToolResult(
            data=AskUserQuestionToolOutput(
                answers=answers,
                selected_labels=selected_labels,
            ),
            result_for_assistant=self.render_result_for_assistant(
                AskUserQuestionToolOutput(
                    answers=answers,
                    selected_labels=selected_labels,
                )
            ),
        )
