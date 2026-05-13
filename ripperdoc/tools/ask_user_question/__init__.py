"""Ask user question tool for interactive clarification.

This tool allows the AI to ask the user questions during execution,
enabling clarification of ambiguous instructions, gathering preferences,
and making decisions on implementation choices.
"""

from __future__ import annotations

import asyncio
import html
import os
import sys
from textwrap import dedent
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

logger = get_logger()

TOOL_NAME = "AskUserQuestion"
CHOICE_UI_FALLBACK = "__choice_ui_fallback__"
BACK_VALUE = "__back__"
NEXT_VALUE = "__next__"
HEADER_MAX_CHARS = 12

ANSI_RESET = "\033[0m"
ANSI_TAB_ACTIVE_BG = "\033[48;5;24m"
ANSI_TAB_ACTIVE_FG = "\033[38;5;255m"
ANSI_TAB_INACTIVE_FG = "\033[38;5;244m"

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


def format_question_prompt(
    question: QuestionInput,
    question_num: int,
    total: int,
    allow_back: bool = False,
    allow_next: bool = False,
) -> str:
    """Format a question for terminal display."""
    header = truncate_header(question.header)
    lines = [
        "",
        f"[{header}] {question_num}/{total}",
        question.question,
        "",
    ]

    for idx, opt in enumerate(question.options, start=1):
        lines.append(format_option_display(opt, idx))

    # Add "Other" option
    other_index = len(question.options) + 1
    lines.append(f"  {other_index}. Other (type your own answer)")
    if allow_back:
        lines.append(f"  {other_index + 1}. Back to previous question")

    if question.multiSelect:
        lines.append("")
        if allow_back:
            if allow_next:
                lines.append(
                    "  Select options (e.g., 1,3), 'o' for other, "
                    "'b' to go back, 'n' to next, or 'q' to cancel: "
                )
            else:
                lines.append(
                    "  Select options (e.g., 1,3), 'o' for other, 'b' to go back, or 'q' to cancel: "
                )
        else:
            if allow_next:
                lines.append(
                    "  Select options (e.g., 1,3), 'o' for other, 'n' to next, or 'q' to cancel: "
                )
            else:
                lines.append(
                    "  Select options (e.g., 1,3), 'o' for other, or 'q' to cancel: "
                )
    else:
        lines.append("")
        if allow_back:
            if allow_next:
                lines.append(
                    "  Select 1-{}, 'o' for other, 'b' to go back, 'n' to next, or 'q' to cancel: ".format(
                        len(question.options) + 2
                    )
                )
            else:
                lines.append(
                    "  Select 1-{}, 'o' for other, 'b' to go back, or 'q' to cancel: ".format(
                        len(question.options) + 2
                    )
                )
        else:
            if allow_next:
                lines.append(
                    "  Select 1-{}, 'o' for other, 'n' to next, or 'q' to cancel: ".format(
                        len(question.options) + 1
                    )
                )
            else:
                lines.append(
                    "  Select 1-{}, 'o' for other, or 'q' to cancel: ".format(len(question.options) + 1)
                )

    return "\n".join(lines)


def build_question_tabs(questions: List[QuestionInput], current_index: int) -> str:
    """Build a plain tab-like step header for multi-question flows."""
    use_ansi = _supports_ansi_tabs()
    tabs: list[str] = []
    for idx, item in enumerate(questions):
        header = truncate_header(item.header)
        if idx == current_index:
            if use_ansi:
                tabs.append(
                    f"{ANSI_TAB_ACTIVE_BG}{ANSI_TAB_ACTIVE_FG}[{idx + 1}. {header}]{ANSI_RESET}"
                )
            else:
                tabs.append(f">[{idx + 1}. {header}]<")
        else:
            if use_ansi:
                tabs.append(f"{ANSI_TAB_INACTIVE_FG}[{idx + 1}. {header}]{ANSI_RESET}")
            else:
                tabs.append(f"[{idx + 1}. {header}]")
    return "  ".join(tabs)


def _supports_ansi_tabs() -> bool:
    """Return True when ANSI colors are likely supported for tab rendering."""
    term = os.environ.get("TERM", "")
    return bool(sys.stdout.isatty() and term and term.lower() != "dumb")


async def prompt_custom_answer() -> Optional[str]:
    """Prompt for a custom free-text answer."""
    loop = asyncio.get_running_loop()

    def _prompt_custom() -> Optional[str]:
        from prompt_toolkit import prompt as pt_prompt

        print("  Enter your custom answer: ", end="")
        custom = pt_prompt("")
        if custom.strip():
            return custom.strip()
        print("  Custom answer cannot be empty.")
        return None

    while True:
        try:
            custom = await loop.run_in_executor(None, _prompt_custom)
            if custom:
                return custom
        except KeyboardInterrupt:
            return None
        except EOFError:
            return None
        except (OSError, RuntimeError, ValueError) as e:
            logger.warning(
                "[ask_user_question_tool] Error collecting custom answer: %s: %s",
                type(e).__name__,
                e,
            )
            return None


async def prompt_single_choice_with_ui(
    question: QuestionInput,
    question_num: int,
    total: int,
    all_questions: Optional[List[QuestionInput]] = None,
    allow_back: bool = False,
    allow_next: bool = False,
) -> Optional[str]:
    """Prompt single-select questions using the shared choice component."""
    header = truncate_header(question.header)
    title = f"[{header}] {question_num}/{total}"
    message_parts: list[str] = [f"<question>{html.escape(question.question)}</question>"]
    nav_hints = ["Enter submit", "ESC cancel"]
    if allow_back:
        nav_hints.insert(0, "← previous step")
    if allow_next:
        nav_hints.insert(1 if allow_back else 0, "→ next step")
    message_parts.append(f"<dim>{', '.join(nav_hints)}.</dim>")
    message = "\n".join(message_parts)
    external_header = None
    if all_questions and len(all_questions) > 1:
        external_header = build_question_tabs(all_questions, question_num - 1)
    options: list[ChoiceOption] = []
    for option in question.options:
        label = html.escape(option.label)
        if option.description.strip():
            label = f"{label} <dim>- {html.escape(option.description)}</dim>"
        options.append(ChoiceOption(value=option.label, label=label))
    try:
        selection = await prompt_choice_async(
            message=message,
            options=options,
            title=title,
            allow_esc=True,
            esc_value="__cancel__",
            style_variant="ask_user_question",
            back_value=BACK_VALUE if allow_back else None,
            next_value=NEXT_VALUE if allow_next else None,
            external_header=external_header,
            custom_input_label="Other",
        )
    except (OSError, RuntimeError, ValueError) as e:
        logger.warning(
            "[ask_user_question_tool] Choice UI failed, falling back to text prompt: %s: %s",
            type(e).__name__,
            e,
        )
        print(
            f"[ask_user_question_tool] Choice UI unavailable ({type(e).__name__}), using text mode."
        )
        return CHOICE_UI_FALLBACK

    if selection == "__cancel__":
        return None

    if selection == BACK_VALUE:
        return BACK_VALUE

    return selection


async def prompt_multi_choice_with_ui(
    question: QuestionInput,
    question_num: int,
    total: int,
    all_questions: Optional[List[QuestionInput]] = None,
    allow_back: bool = False,
    allow_next: bool = False,
) -> Optional[str]:
    """Prompt multi-select questions using the shared checkbox component."""
    header = truncate_header(question.header)
    title = f"[{header}] {question_num}/{total}"
    message_parts: list[str] = [f"<question>{html.escape(question.question)}</question>"]
    message_parts.append("<dim>Use arrow keys to move, Space to toggle.</dim>")
    nav_hints = ["Enter submit", "ESC cancel", "Tab Other input"]
    if allow_back:
        nav_hints.insert(0, "← previous step")
    if allow_next:
        nav_hints.insert(1 if allow_back else 0, "→ next step")
    message_parts.append(f"<dim>{', '.join(nav_hints)}.</dim>")
    message_parts.append("<dim>Type in 'Other' field for custom input.</dim>")
    message = "\n".join(message_parts)
    external_header = None
    if all_questions and len(all_questions) > 1:
        external_header = build_question_tabs(all_questions, question_num - 1)
    options: list[ChoiceOption] = []
    for option in question.options:
        label = html.escape(option.label)
        if option.description.strip():
            label = f"{label} <dim>- {html.escape(option.description)}</dim>"
        options.append(ChoiceOption(value=option.label, label=label))

    while True:
        try:
            selected = await prompt_checkbox_async(
                message=message,
                options=options,
                title=title,
                style_variant="ask_user_question",
                custom_input_label="Other",
                back_value=BACK_VALUE if allow_back else None,
                next_value=NEXT_VALUE if allow_next else None,
                external_header=external_header,
            )
        except (OSError, RuntimeError, ValueError) as e:
            logger.warning(
                "[ask_user_question_tool] Checkbox UI failed, falling back to text prompt: %s: %s",
                type(e).__name__,
                e,
            )
            print(
                f"[ask_user_question_tool] Checkbox UI unavailable ({type(e).__name__}), using text mode."
            )
            return CHOICE_UI_FALLBACK

        if selected is None:
            return None

        if not selected:
            print("  Please select at least one option.")
            continue

        if BACK_VALUE in selected:
            if len(selected) == 1:
                return BACK_VALUE
            print("  'Back to previous question' cannot be combined with other selections.")
            continue

        return ", ".join(selected)


async def prompt_user_for_answer(
    question: QuestionInput,
    question_num: int,
    total: int,
    all_questions: Optional[List[QuestionInput]] = None,
    allow_back: bool = False,
    allow_next: bool = False,
) -> Optional[str]:
    """Prompt user for an answer to a single question.

    Returns the answer string, or None if cancelled.
    """
    loop = asyncio.get_running_loop()

    def _prompt() -> Optional[str]:
        try:
            from prompt_toolkit import prompt as pt_prompt

            prompt_text = format_question_prompt(
                question, question_num, total, allow_back=allow_back, allow_next=allow_next
            )
            print(prompt_text, end="")

            while True:
                response = pt_prompt("").strip()

                if not response:
                    print("  Please enter a valid choice.")
                    continue

                if response.lower() in ("q", "quit", "cancel", "exit"):
                    return None

                if allow_back and response.lower() in ("b", "back", "prev", "previous"):
                    return BACK_VALUE
                if allow_next and response.lower() in ("n", "next", "forward"):
                    return NEXT_VALUE

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
                        max_index = len(question.options) + (3 if allow_back else 2)
                        back_index = len(question.options) + 2 if allow_back else None
                        valid_range = range(1, max_index)
                        if all(i in valid_range for i in indices):
                            if back_index is not None and back_index in indices:
                                if len(indices) == 1:
                                    return BACK_VALUE
                                print(
                                    "  'Back to previous question' cannot be combined with other selections."
                                )
                                continue
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
                        print(f"  Invalid selection. Enter numbers from 1 to {max_index - 1}.")
                    except ValueError:
                        print("  Invalid input. Enter numbers separated by commas.")
                else:
                    # Single selection
                    try:
                        choice = int(response)
                        if 1 <= choice <= len(question.options):
                            return question.options[choice - 1].label
                        elif allow_back and choice == len(question.options) + 2:
                            return BACK_VALUE
                        elif choice == len(question.options) + 1:
                            # Other option
                            print("  Enter your custom answer: ", end="")
                            custom = pt_prompt("")
                            if custom.strip():
                                return custom.strip()
                            print("  Custom answer cannot be empty.")
                        else:
                            max_choice = len(question.options) + (2 if allow_back else 1)
                            print(f"  Invalid choice. Enter a number from 1 to {max_choice}.")
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

    if not question.multiSelect:
        # Use shared choice box for single-select questions.
        single_choice = await prompt_single_choice_with_ui(
            question,
            question_num,
            total,
            all_questions=all_questions,
            allow_back=allow_back,
            allow_next=allow_next,
        )
        if single_choice != CHOICE_UI_FALLBACK:
            return single_choice
    else:
        # Use shared checkbox box for multi-select questions.
        multi_choice = await prompt_multi_choice_with_ui(
            question,
            question_num,
            total,
            all_questions=all_questions,
            allow_back=allow_back,
            allow_next=allow_next,
        )
        if multi_choice != CHOICE_UI_FALLBACK:
            return multi_choice

    return await loop.run_in_executor(None, _prompt)


def _build_confirmation_message(questions: List[QuestionInput], answers: Dict[str, str]) -> str:
    """Build confirmation content listing each question and answer."""
    lines = ["<question>Please confirm your answers before submit:</question>", ""]
    for idx, q in enumerate(questions, start=1):
        question_text = html.escape(q.question)
        answer = answers.get(q.question, "").strip()
        answer_text = html.escape(answer) if answer else "<dim>(未回答)</dim>"
        lines.append(f"{idx}. {question_text}")
        lines.append(f"   -> {answer_text}")
    lines.append("")
    lines.append("<dim>Use Left/Right to review steps before submit if needed.</dim>")
    return "\n".join(lines)


async def _confirm_answers(questions: List[QuestionInput], answers: Dict[str, str]) -> bool:
    """Show final confirmation and return True when user confirms submit."""
    selection = await prompt_choice_async(
        message=_build_confirmation_message(questions, answers),
        options=[
            ChoiceOption(value="submit", label="<default>Submit</default>"),
            ChoiceOption(value="cancel", label="<dim>Cancel</dim>"),
        ],
        title="Confirm Answers",
        allow_esc=True,
        esc_value="cancel",
        style_variant="ask_user_question",
    )
    return selection == "submit"


async def collect_answers(
    questions: List[QuestionInput], initial_answers: Dict[str, str]
) -> tuple[Dict[str, str], bool]:
    """Collect answers for all questions.

    Returns (answers_dict, cancelled_flag).
    """
    answers = dict(initial_answers)
    total = len(questions)
    idx = 0

    while idx < total:
        question = questions[idx]
        answer = await prompt_user_for_answer(
            question,
            idx + 1,
            total,
            all_questions=questions,
            allow_back=idx > 0,
            allow_next=True,
        )
        if answer is None:
            return answers, True  # Cancelled

        if answer == BACK_VALUE:
            idx = max(0, idx - 1)
            continue

        if answer == NEXT_VALUE:
            idx += 1
            continue

        answers[question.question] = answer
        idx += 1
    confirmed = await _confirm_answers(questions, answers)
    return answers, not confirmed


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
