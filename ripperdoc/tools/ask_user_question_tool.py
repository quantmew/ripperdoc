"""Ask user question tool for interactive clarification.

This tool allows the AI to ask the user questions during execution,
enabling clarification of ambiguous instructions, gathering preferences,
and making decisions on implementation choices.
"""

from __future__ import annotations

import asyncio
from textwrap import dedent
from typing import AsyncGenerator, Dict, List, Optional

from pydantic import BaseModel, Field

from ripperdoc.core.tool import (
    Tool,
    ToolOutput,
    ToolResult,
    ToolUseContext,
    ValidationResult,
)
from ripperdoc.utils.log import get_logger

logger = get_logger()

TOOL_NAME = "AskUserQuestion"
OTHER_VALUE = "__other__"
HEADER_MAX_CHARS = 12

ASK_USER_QUESTION_PROMPT = dedent(
    """\
    Use this tool when you need to ask the user questions during execution. This allows you to:
    1. Gather user preferences or requirements
    2. Clarify ambiguous instructions
    3. Get decisions on implementation choices as you work
    4. Offer choices to the user about what direction to take.

    Usage notes:
    - Users will always be able to select "Other" to provide custom text input
    - Use multiSelect: true to allow multiple answers to be selected for a question
    - If you recommend a specific option, make that the first option in the list and add "(Recommended)" at the end of the label
    """
)


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
        description="The complete question to ask the user. Should be clear, specific, and end "
        'with a question mark. Example: "Which library should we use for date formatting?" '
        'If multiSelect is true, phrase it accordingly, e.g. "Which features do you want to enable?"'
    )
    header: str = Field(
        description=f"Very short label displayed as a chip/tag (max {HEADER_MAX_CHARS} chars). "
        'Examples: "Auth method", "Library", "Approach".'
    )
    options: List[OptionInput] = Field(
        min_length=2,
        max_length=4,
        description="The available choices for this question. Must have 2-4 options. "
        "Each option should be a distinct, mutually exclusive choice (unless multiSelect is enabled). "
        "There should be no 'Other' option, that will be provided automatically.",
    )
    multiSelect: bool = Field(
        description="Set to true to allow the user to select multiple options instead of just one. "
        "Use when choices are not mutually exclusive."
    )


class AskUserQuestionToolInput(BaseModel):
    """Input for the AskUserQuestion tool."""

    questions: List[QuestionInput] = Field(
        min_length=1,
        max_length=4,
        description="Questions to ask the user (1-4 questions)",
    )
    answers: Optional[Dict[str, str]] = Field(
        default=None,
        description="User answers collected by the permission component",
    )


class AskUserQuestionToolOutput(BaseModel):
    """Output from the AskUserQuestion tool."""

    questions: List[QuestionInput]
    answers: Dict[str, str]
    cancelled: bool = False


def truncate_header(header: str) -> str:
    """Truncate header to maximum characters."""
    if len(header) <= HEADER_MAX_CHARS:
        return header
    return f"{header[: HEADER_MAX_CHARS - 1]}..."


def format_option_display(option: OptionInput, index: int) -> str:
    """Format a single option for display."""
    desc = f" - {option.description}" if option.description.strip() else ""
    return f"  {index}. {option.label}{desc}"


def format_question_prompt(question: QuestionInput, question_num: int, total: int) -> str:
    """Format a question for terminal display."""
    header = truncate_header(question.header)
    lines = [
        "",
        f"[{header}] Question {question_num}/{total}",
        f"  {question.question}",
        "",
    ]

    for idx, opt in enumerate(question.options, start=1):
        lines.append(format_option_display(opt, idx))

    # Add "Other" option
    lines.append(f"  {len(question.options) + 1}. Other (type your own answer)")

    if question.multiSelect:
        lines.append("")
        lines.append("  Enter numbers separated by commas (e.g., 1,3), or 'o' for other: ")
    else:
        lines.append("")
        lines.append("  Enter choice (1-{}) or 'o' for other: ".format(len(question.options) + 1))

    return "\n".join(lines)


async def prompt_user_for_answer(
    question: QuestionInput, question_num: int, total: int
) -> Optional[str]:
    """Prompt user for an answer to a single question.

    Returns the answer string, or None if cancelled.
    """
    loop = asyncio.get_running_loop()

    def _prompt() -> Optional[str]:
        try:
            from prompt_toolkit import prompt as pt_prompt

            prompt_text = format_question_prompt(question, question_num, total)
            print(prompt_text, end="")

            while True:
                response = pt_prompt("").strip()

                if not response:
                    print("  Please enter a valid choice.")
                    continue

                if response.lower() in ("q", "quit", "cancel", "exit"):
                    return None

                if response.lower() == "o" or response == str(len(question.options) + 1):
                    # Other option selected
                    print("  Enter your custom answer: ", end="")
                    custom = pt_prompt("")
                    if custom.strip():
                        return custom.strip()
                    print("  Custom answer cannot be empty.")
                    continue

                if question.multiSelect:
                    # Parse comma-separated numbers
                    try:
                        indices = [int(x.strip()) for x in response.split(",")]
                        valid_range = range(1, len(question.options) + 2)
                        if all(i in valid_range for i in indices):
                            selected = []
                            for i in indices:
                                if i == len(question.options) + 1:
                                    # Other option
                                    print("  Enter your custom answer: ", end="")
                                    custom = pt_prompt("")
                                    if custom.strip():
                                        selected.append(custom.strip())
                                else:
                                    selected.append(question.options[i - 1].label)
                            if selected:
                                return ", ".join(selected)
                        print(
                            f"  Invalid selection. Enter numbers from 1 to {len(question.options) + 1}."
                        )
                    except ValueError:
                        print("  Invalid input. Enter numbers separated by commas.")
                else:
                    # Single selection
                    try:
                        choice = int(response)
                        if 1 <= choice <= len(question.options):
                            return question.options[choice - 1].label
                        elif choice == len(question.options) + 1:
                            # Other option
                            print("  Enter your custom answer: ", end="")
                            custom = pt_prompt("")
                            if custom.strip():
                                return custom.strip()
                            print("  Custom answer cannot be empty.")
                        else:
                            print(
                                f"  Invalid choice. Enter a number from 1 to {len(question.options) + 1}."
                            )
                    except ValueError:
                        print("  Invalid input. Enter a number.")

        except KeyboardInterrupt:
            return None
        except EOFError:
            return None
        except (OSError, RuntimeError, ValueError) as e:
            logger.warning(
                "[ask_user_question_tool] Error during prompt: %s: %s",
                type(e).__name__,
                e,
            )
            return None

    return await loop.run_in_executor(None, _prompt)


async def collect_answers(
    questions: List[QuestionInput], initial_answers: Dict[str, str]
) -> tuple[Dict[str, str], bool]:
    """Collect answers for all questions.

    Returns (answers_dict, cancelled_flag).
    """
    answers = dict(initial_answers)
    total = len(questions)

    for idx, question in enumerate(questions, start=1):
        # Skip if already answered
        if question.question in answers and answers[question.question]:
            continue

        answer = await prompt_user_for_answer(question, idx, total)
        if answer is None:
            return answers, True  # Cancelled
        answers[question.question] = answer

    return answers, False


class AskUserQuestionTool(Tool[AskUserQuestionToolInput, AskUserQuestionToolOutput]):
    """Tool for asking the user questions interactively."""

    @property
    def name(self) -> str:
        return TOOL_NAME

    async def description(self) -> str:
        return (
            "Asks the user multiple choice questions to gather information, "
            "clarify ambiguity, understand preferences, make decisions or offer them choices."
        )

    @property
    def input_schema(self) -> type[AskUserQuestionToolInput]:
        return AskUserQuestionToolInput

    async def prompt(self, yolo_mode: bool = False) -> str:  # noqa: ARG002
        return ASK_USER_QUESTION_PROMPT

    def user_facing_name(self) -> str:
        return ""

    def is_read_only(self) -> bool:
        return True

    def is_concurrency_safe(self) -> bool:
        return True

    def needs_permissions(
        self,
        input_data: Optional[AskUserQuestionToolInput] = None,  # noqa: ARG002
    ) -> bool:
        return False

    async def validate_input(
        self,
        input_data: AskUserQuestionToolInput,
        context: Optional[ToolUseContext] = None,  # noqa: ARG002
    ) -> ValidationResult:
        """Validate that question texts and option labels are unique."""
        seen_questions: set[str] = set()

        for question in input_data.questions:
            if question.question in seen_questions:
                return ValidationResult(result=False, message="Question texts must be unique")
            seen_questions.add(question.question)

            option_labels: set[str] = set()
            for option in question.options:
                if option.label in option_labels:
                    return ValidationResult(
                        result=False,
                        message=f'Option labels for "{question.question}" must be unique',
                    )
                option_labels.add(option.label)

        return ValidationResult(result=True)

    def render_result_for_assistant(self, output: AskUserQuestionToolOutput) -> str:
        """Render the tool output for the AI assistant."""
        if output.cancelled:
            return "User declined to answer your questions."

        if not output.answers:
            return "User did not provide any answers."

        serialized = ", ".join(
            f'"{question}"="{answer}"' for question, answer in output.answers.items()
        )
        return (
            f"User has answered your questions: {serialized}. "
            "You can now continue with the user's answers in mind."
        )

    def render_tool_use_message(
        self,
        input_data: AskUserQuestionToolInput,
        verbose: bool = False,  # noqa: ARG002
    ) -> str:
        """Render the tool use message for display."""
        question_count = len(input_data.questions)
        if question_count == 1:
            return f"Asking user: {input_data.questions[0].question}"
        return f"Asking user {question_count} questions"

    async def call(
        self,
        input_data: AskUserQuestionToolInput,
        context: ToolUseContext,
    ) -> AsyncGenerator[ToolOutput, None]:
        """Execute the tool to ask user questions."""
        questions = input_data.questions
        initial_answers = input_data.answers or {}

        # Pause UI spinner before user interaction
        if context.pause_ui:
            try:
                context.pause_ui()
            except (RuntimeError, ValueError, OSError):
                logger.debug("[ask_user_question_tool] Failed to pause UI")

        try:
            # Display introduction
            loop = asyncio.get_running_loop()

            def _print_intro() -> None:
                print("\n" + "=" * 60)
                print("I need a few answers to proceed:")
                print("=" * 60)

            await loop.run_in_executor(None, _print_intro)

            # Collect answers
            answers, cancelled = await collect_answers(questions, initial_answers)

            if cancelled:
                output = AskUserQuestionToolOutput(
                    questions=questions,
                    answers=answers,
                    cancelled=True,
                )
                yield ToolResult(
                    data=output,
                    result_for_assistant=self.render_result_for_assistant(output),
                )
                return

            # Display summary
            def _print_summary() -> None:
                print("\n" + "-" * 40)
                print("Your answers:")
                for q, a in answers.items():
                    print(f"  - {q}")
                    print(f"    -> {a}")
                print("-" * 40 + "\n")

            await loop.run_in_executor(None, _print_summary)

            output = AskUserQuestionToolOutput(
                questions=questions,
                answers=answers,
                cancelled=False,
            )
            yield ToolResult(
                data=output,
                result_for_assistant=self.render_result_for_assistant(output),
            )

        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            logger.warning(
                "[ask_user_question_tool] Error collecting answers: %s: %s",
                type(exc).__name__,
                exc,
            )
            output = AskUserQuestionToolOutput(
                questions=questions,
                answers={},
                cancelled=True,
            )
            yield ToolResult(
                data=output,
                result_for_assistant="Error while collecting user answers: " + str(exc),
            )

        finally:
            # Resume UI spinner after user interaction
            if context.resume_ui:
                try:
                    context.resume_ui()
                except (RuntimeError, ValueError, OSError):
                    logger.debug("[ask_user_question_tool] Failed to resume UI")
