"""Unified choice UI component for Ripperdoc.

This module provides a reusable, visually consistent choice interface
that can be used across onboarding, permission prompts, and other
user interactions.
"""

from __future__ import annotations

import html
import shutil
from typing import Any, Optional

from prompt_toolkit.filters import is_done
from prompt_toolkit.formatted_text import HTML, fragment_list_to_text, to_formatted_text
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.shortcuts import choice
from prompt_toolkit.styles import Style
from prompt_toolkit.utils import get_cwidth


# Shared style system for all choice prompts
def _choice_style() -> Style:
    """Create the unified style for choice prompts.

    Uses a golden/amber theme with cyan accents for consistent branding.
    """
    return Style.from_dict(
        {
            "frame.border": "#d4a017",  # Golden/amber border
            "selected-option": "bold",
            "option": "#5fd7ff",  # Cyan for unselected options
            "title": "#ffaf00",  # Orange/amber for titles
            "description": "#ffffff",  # White for descriptions
            "question": "#ffd700",  # Gold for questions
            "label": "#87afff",  # Light blue for field labels
            "warning": "#ff5555",  # Red for warnings
            "info": "#5fd7ff",  # Cyan for info text
            "dim": "#626262",  # Dimmed text
            "yes-option": "#ffffff",  # Neutral for Yes options
            "no-option": "#ffffff",  # Neutral for No options
            "value": "#f8f8f2",  # Off-white for values
            "default": "#50fa7b",  # Green for defaults
            "marker": "#ffb86c",  # Orange for markers (→, etc.)
        }
    )


def onboarding_style() -> Style:
    """Create the style for onboarding prompts.

    Uses a balanced theme with purple accents for a modern setup experience.
    """
    return Style.from_dict(
        {
            "frame.border": "#8b9dc3",  # Soft silver-blue border
            "selected-option": "bold",
            "option": "#f8f8f2",  # Off-white for options (cleaner)
            "title": "#bd93f9",  # Soft purple for titles
            "description": "#f0f0f0",  # Off-white for descriptions
            "question": "#ff79c6",  # Pink for questions
            "label": "#8be9fd",  # Cyan for labels
            "warning": "#ffb86c",  # Orange for warnings
            "info": "#8be9fd",  # Cyan for info text
            "dim": "#626262",  # Dimmed text
            "yes-option": "#ffffff",
            "no-option": "#ffffff",
            "value": "#f8f8f2",
            "default": "#50fa7b",  # Green for defaults
            "marker": "#ff79c6",  # Pink for markers
        }
    )


def theme_style() -> Style:
    """Create the style for theme selection prompts.

    Uses a subtle gray border for a clean, minimal appearance.
    """
    return Style.from_dict(
        {
            "frame.border": "#626262",  # Subtle gray border
            "selected-option": "bold",
            "option": "#f8f8f2",
            "title": "#f8f8f2",  # Off-white for titles
            "description": "#f0f0f0",
            "question": "#f8f8f2",
            "label": "#8be9fd",
            "warning": "#ffb86c",
            "info": "#5fd7ff",
            "dim": "#626262",
            "yes-option": "#ffffff",
            "no-option": "#ffffff",
            "value": "#f8f8f2",
            "default": "#50fa7b",
            "marker": "#8be9fd",
        }
    )


class ChoiceOption:
    """Represents a single choice option.

    Args:
        value: The value to return when this option is selected
        label: The display label (can contain HTML tags)
        description: Optional description text
        is_default: Whether this is the default choice
    """

    def __init__(
        self,
        value: str,
        label: str,
        description: Optional[str] = None,
        is_default: bool = False,
    ):
        self.value = value
        self.label = label
        self.description = description
        self.is_default = is_default

    def __repr__(self) -> str:
        return f"ChoiceOption(value={self.value!r}, label={self.label!r})"


def _terminal_width(default: int = 80) -> int:
    """Return the current terminal width with a reasonable fallback."""
    try:
        width = shutil.get_terminal_size(fallback=(default, 24)).columns
    except OSError:
        width = default
    return max(1, width)


def _count_display_lines(text: Any, terminal_width: int) -> int:
    """Estimate the number of lines a formatted text will occupy.

    Args:
        text: The text to measure (plain or formatted)
        terminal_width: Terminal width for word wrapping

    Returns:
        Estimated number of lines
    """
    if not text or terminal_width <= 0:
        return 0

    fragments = to_formatted_text(text)
    plain_text = fragment_list_to_text(fragments)
    if not plain_text:
        return 0

    # Normalize newlines and keep trailing empty lines.
    plain_text = plain_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = plain_text.split("\n")

    # Account for text wrapping using display width (CJK-safe).
    total_lines = 0
    for line in lines:
        if not line:
            total_lines += 1
        else:
            width = get_cwidth(line)
            wrapped = (width + terminal_width - 1) // terminal_width
            total_lines += max(1, wrapped)

    return total_lines


def prompt_choice(
    message: str,
    options: list[ChoiceOption] | list[tuple[str, str]],
    *,
    title: Optional[str] = None,
    description: Optional[str] = None,
    warning: Optional[str] = None,
    allow_esc: bool = True,
    esc_value: Optional[str] = None,
    style: Optional[Style] = None,
) -> str:
    """Prompt the user to make a choice from a list of options.

    This is the unified choice interface used across Ripperdoc for
    consistent user experience in onboarding, permission prompts, etc.

    Args:
        message: The main prompt message (supports HTML formatting)
        options: List of ChoiceOption objects or (value, label) tuples
        title: Optional title to display above the prompt
        description: Optional description text
        warning: Optional warning message to display
        allow_esc: Whether ESC key can be used to cancel (defaults to True)
        esc_value: Value to return when ESC is pressed (defaults to first option's value)
        style: Optional custom style (defaults to unified choice style)

    Returns:
        The value of the selected option

    Example:
        ```python
        # Simple usage with tuples
        result = prompt_choice(
            "Choose a provider",
            [("openai", "OpenAI"), ("anthropic", "Anthropic")]
        )

        # Rich usage with ChoiceOption objects
        result = prompt_choice(
            "Choose a provider",
            [
                ChoiceOption("openai", "<info>OpenAI</info>", "GPT models"),
                ChoiceOption("anthropic", "<info>Anthropic</info>", "Claude models"),
            ],
            title="AI Provider Selection",
            description="Select your preferred AI model provider"
        )
        ```
    """
    # Normalize options to ChoiceOption objects
    choice_options: list[ChoiceOption] = []
    for opt in options:
        if isinstance(opt, ChoiceOption):
            choice_options.append(opt)
        else:
            value, label = opt
            choice_options.append(ChoiceOption(value, label))

    # Build formatted prompt HTML
    prompt_html = ""
    if title:
        prompt_html += f"<title>{html.escape(title)}</title>\n"
    if description:
        prompt_html += f"<description>{html.escape(description)}</description>\n"
    prompt_html += message
    if warning:
        prompt_html += f"\n<warning>{html.escape(warning)}</warning>"

    formatted_prompt = HTML(f"\n{prompt_html}\n")

    # Convert to prompt_toolkit format
    choice_options_formatted = []
    for opt in choice_options:
        label = opt.label
        # Add default marker after the label if applicable
        if opt.is_default:
            label = f"{label} <default>*</default>"
        choice_options_formatted.append((opt.value, HTML(label) if "<" in label else label))

    # Set up ESC key binding
    key_bindings = KeyBindings()
    if allow_esc:
        default_esc_value = esc_value or (choice_options[0].value if choice_options else "")
        result_on_esc = default_esc_value

        @key_bindings.add("escape", eager=True)
        def _esc_handler(event: Any) -> None:  # noqa: ANN001 (called by key_binding)
            event.app.exit(result=result_on_esc, style="class:aborting")

    # Show the choice dialog
    result = choice(
        message=formatted_prompt,
        options=choice_options_formatted,
        style=style or _choice_style(),
        show_frame=~is_done,
        key_bindings=key_bindings if allow_esc else None,
    )

    # Clear the prompt from screen by calculating the exact number of lines
    # ANSI codes: ESC[F = move cursor to beginning of previous line
    #              ESC[2K = clear entire line
    #
    # Calculate lines to clear:
    # - Options: len(choice_options)
    # NOTE: prompt_toolkit renders a final "done" state without the frame
    # when show_frame=~is_done, so we should not include frame borders here.
    term_width = _terminal_width()
    message_width = max(1, term_width - 2)  # account for label padding
    lines_to_clear = _count_display_lines(formatted_prompt, message_width)
    lines_to_clear += len(choice_options_formatted)

    for _ in range(lines_to_clear):
        print("\033[F\033[2K", end="", flush=True)

    return result


def prompt_yes_no(
    message: str,
    *,
    title: Optional[str] = None,
    allow_session: bool = True,
) -> str:
    """Prompt a yes/no question with optional session remember option.

    Args:
        message: The question to ask
        title: Optional title
        allow_session: Whether to include "Yes, for this session" option

    Returns:
        "y" for yes, "s" for session, "n" for no

    Example:
        ```python
        answer = prompt_yes_no("Continue with installation?")
        if answer in ("y", "s"):
            # User approved
            ...
        ```
    """
    options: list[tuple[str, str]] = [
        ("y", "<yes-option>Yes</yes-option>"),
    ]

    if allow_session:
        options.append(("s", "<yes-option>Yes, for this session</yes-option>"))

    options.append(("n", "<no-option>No</no-option>"))

    return prompt_choice(
        message=f"<question>{html.escape(message)}</question>",
        options=options,
        title=title,
        allow_esc=True,
        esc_value="n",  # ESC means no
    )


def prompt_select(
    message: str,
    options: list[str],
    *,
    default: Optional[str] = None,
    title: Optional[str] = None,
) -> Optional[str]:
    """Prompt the user to select from a list of string options.

    A simplified version of prompt_choice for simple string lists.

    Args:
        message: The prompt message
        options: List of option strings
        default: The default option value
        title: Optional title

    Returns:
        The selected option value, or None if canceled

    Example:
        ```python
        provider = prompt_select(
            "Choose your model provider",
            ["openai", "anthropic", "deepseek"],
            default="deepseek"
        )
        ```
    """
    choice_options = [
        ChoiceOption(
            opt,
            f"<info>{opt}</info>",
            is_default=(opt == default),
        )
        for opt in options
    ]

    result = prompt_choice(
        message=message,
        options=choice_options,
        title=title,
        allow_esc=True,
        esc_value=default or (options[0] if options else None),
    )

    return result if result else None
