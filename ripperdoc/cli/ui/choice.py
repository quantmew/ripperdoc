"""Unified choice UI component for Ripperdoc.

This module provides a reusable, visually consistent choice interface
that can be used across onboarding, permission prompts, and other
user interactions.
"""

from __future__ import annotations

import html
import os
import re
import shutil
from typing import Any, Optional, cast

from prompt_toolkit.application import Application
from prompt_toolkit.filters import Condition, has_focus, is_done
from prompt_toolkit.formatted_text import HTML, fragment_list_to_text, to_formatted_text
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout
from prompt_toolkit.layout.containers import ConditionalContainer
from prompt_toolkit.shortcuts import choice
from prompt_toolkit.shortcuts.choice_input import ChoiceInput
from prompt_toolkit.styles import Style
from prompt_toolkit.utils import get_cwidth
from prompt_toolkit.widgets import Box, CheckboxList, Frame, Label, RadioList, TextArea
from ripperdoc.core.theme import theme_color

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _choice_diff_style_slots() -> dict[str, str]:
    """Return diff styles from active theme for prompt_toolkit."""
    add_bg = theme_color("diff_add_bg")
    add_fg = theme_color("diff_add_fg")
    del_bg = theme_color("diff_del_bg")
    del_fg = theme_color("diff_del_fg")
    hunk = theme_color("diff_hunk")
    return {
        "diff-add": f"bg:{add_bg} {add_fg}",
        "diff-del": f"bg:{del_bg} {del_fg}",
        "diff-hunk": hunk,
    }


# Shared style system for all choice prompts
def neutral_choice_style() -> Style:
    """Create the default neutral style for choice prompts."""
    style_map = {
        "frame.border": "#7f8fa6",  # Muted blue-gray border
        "selected-option": "bold",
        "option": "#d7e3f4",  # Soft blue for options
        "title": "#c7d2e6",  # Light blue-gray title
        "description": "#d9d9d9",
        "question": "#e8ecf5",  # Neutral light text for questions
        "label": "#9fb3d1",  # Medium blue-gray labels
        "number": "#9fb3d1",  # Numbers for indexed options
        "warning": "#ff6b6b",  # Warning stays distinct
        "info": "#a7c7e7",  # Calm info blue
        "dim": "#707070",
        "yes-option": "#e8ecf5",
        "no-option": "#e8ecf5",
        "value": "#e8ecf5",
        "default": "#88c0d0",
        "marker": "#9fb3d1",
    }
    style_map.update(_choice_diff_style_slots())
    return Style.from_dict(style_map)


def amber_choice_style() -> Style:
    """Create the legacy amber style for choice prompts."""
    style_map = {
        "frame.border": "#d4a017",  # Golden/amber border
        "selected-option": "bold",
        "option": "#5fd7ff",  # Cyan for unselected options
        "title": "#ffaf00",  # Orange/amber for titles
        "description": "#ffffff",  # White for descriptions
        "question": "#ffd700",  # Gold for questions
        "label": "#87afff",  # Light blue for field labels
        "number": "#87afff",  # Numbers for indexed options
        "warning": "#ff5555",  # Red for warnings
        "info": "#5fd7ff",  # Cyan for info text
        "dim": "#626262",  # Dimmed text
        "yes-option": "#ffffff",  # Neutral for Yes options
        "no-option": "#ffffff",  # Neutral for No options
        "value": "#f8f8f2",  # Off-white for values
        "default": "#50fa7b",  # Green for defaults
        "marker": "#ffb86c",  # Orange for markers (→, etc.)
    }
    style_map.update(_choice_diff_style_slots())
    return Style.from_dict(style_map)


def onboarding_style() -> Style:
    """Create the style for onboarding prompts.

    Uses adaptive palettes to preserve contrast on both dark and light terminals.
    """
    if _should_use_light_onboarding_palette():
        style_map = {
            "frame.border": "#4b6584",  # Blue-gray border for light backgrounds
            "selected-option": "bold",
            "option": "#1f2937",  # Dark slate option text
            "title": "#5b21b6",  # Deep purple title
            "description": "#374151",  # Dark gray description text
            "question": "#9f1239",  # Deep rose question
            "label": "#0f766e",  # Teal labels
            "number": "#0f766e",  # Teal indices
            "warning": "#b45309",  # Amber warning
            "info": "#0369a1",  # Blue info
            "dim": "#6b7280",
            "yes-option": "#111827",
            "no-option": "#111827",
            "value": "#111827",
            "default": "#047857",  # Green default marker
            "marker": "#be185d",  # Rose marker
        }
        style_map.update(_choice_diff_style_slots())
        return Style.from_dict(style_map)

    style_map = {
        "frame.border": "#8b9dc3",  # Soft silver-blue border
        "selected-option": "bold",
        "option": "#f8f8f2",  # Off-white for options (cleaner)
        "title": "#bd93f9",  # Soft purple for titles
        "description": "#f0f0f0",  # Off-white for descriptions
        "question": "#ff79c6",  # Pink for questions
        "label": "#8be9fd",  # Cyan for labels
        "number": "#8be9fd",  # Numbers for indexed options
        "warning": "#ffb86c",  # Orange for warnings
        "info": "#8be9fd",  # Cyan for info text
        "dim": "#626262",  # Dimmed text
        "yes-option": "#ffffff",
        "no-option": "#ffffff",
        "value": "#f8f8f2",
        "default": "#50fa7b",  # Green for defaults
        "marker": "#ff79c6",  # Pink for markers
    }
    style_map.update(_choice_diff_style_slots())
    return Style.from_dict(style_map)


def _should_use_light_onboarding_palette() -> bool:
    """Return whether onboarding UI should use light-background palette."""
    forced = os.getenv("RIPPERDOC_TERMINAL_BG", "").strip().lower()
    if forced == "light":
        return True
    if forced == "dark":
        return False

    detected = _detect_light_background_from_colorfgbg(os.getenv("COLORFGBG", ""))
    if detected is not None:
        return detected
    return False


def _detect_light_background_from_colorfgbg(raw_value: str) -> Optional[bool]:
    """Best-effort light/dark detection from COLORFGBG env var.

    COLORFGBG usually contains foreground/background ANSI indexes, e.g. "15;0".
    We inspect the last numeric token as the background color index.
    """
    if not raw_value:
        return None

    bg_index: Optional[int] = None
    for token in reversed(raw_value.split(";")):
        token = token.strip()
        if not token:
            continue
        try:
            bg_index = int(token)
            break
        except ValueError:
            continue
    if bg_index is None:
        return None

    rgb = _xterm_index_to_rgb(bg_index)
    if rgb is None:
        return None

    luminance = _perceived_luminance(*rgb)
    return luminance >= 140.0


def _xterm_index_to_rgb(index: int) -> Optional[tuple[int, int, int]]:
    """Map xterm color index (0-255) to RGB."""
    if index < 0 or index > 255:
        return None

    # xterm standard 16-color palette
    base_16: dict[int, tuple[int, int, int]] = {
        0: (0, 0, 0),
        1: (205, 0, 0),
        2: (0, 205, 0),
        3: (205, 205, 0),
        4: (0, 0, 238),
        5: (205, 0, 205),
        6: (0, 205, 205),
        7: (229, 229, 229),
        8: (127, 127, 127),
        9: (255, 0, 0),
        10: (0, 255, 0),
        11: (255, 255, 0),
        12: (92, 92, 255),
        13: (255, 0, 255),
        14: (0, 255, 255),
        15: (255, 255, 255),
    }
    if index in base_16:
        return base_16[index]

    # xterm 6x6x6 color cube: indexes 16-231
    if 16 <= index <= 231:
        cube_index = index - 16
        r = cube_index // 36
        g = (cube_index % 36) // 6
        b = cube_index % 6
        steps = [0, 95, 135, 175, 215, 255]
        return (steps[r], steps[g], steps[b])

    # xterm grayscale ramp: indexes 232-255
    gray = 8 + (index - 232) * 10
    return (gray, gray, gray)


def _perceived_luminance(r: int, g: int, b: int) -> float:
    """Compute perceived luminance for contrast heuristics."""
    return (0.2126 * r) + (0.7152 * g) + (0.0722 * b)


def theme_style() -> Style:
    """Create the style for theme selection prompts.

    Uses a subtle gray border for a clean, minimal appearance.
    """
    style_map = {
        "frame.border": "#626262",  # Subtle gray border
        "selected-option": "bold",
        "option": "#f8f8f2",
        "title": "#f8f8f2",  # Off-white for titles
        "description": "#f0f0f0",
        "question": "#f8f8f2",
        "label": "#8be9fd",
        "number": "#8be9fd",
        "warning": "#ffb86c",
        "info": "#5fd7ff",
        "dim": "#626262",
        "yes-option": "#ffffff",
        "no-option": "#ffffff",
        "value": "#f8f8f2",
        "default": "#50fa7b",
        "marker": "#8be9fd",
    }
    style_map.update(_choice_diff_style_slots())
    return Style.from_dict(style_map)


def ask_user_question_style() -> Style:
    """Create the style for AskUserQuestion prompts.

    Uses a neutral blue/gray palette to avoid warning-like colors.
    """
    style_map = {
        "frame.border": "#7f8fa6",  # Muted blue-gray border
        "selected-option": "bold",
        "option": "#d7e3f4",  # Soft blue for options
        "title": "#c7d2e6",  # Light blue-gray title
        "description": "#d9d9d9",
        "question": "#e8ecf5",  # Neutral light text for questions
        "label": "#9fb3d1",  # Medium blue-gray labels
        "number": "#9fb3d1",  # Numbers for indexed options
        "warning": "#ff6b6b",  # Keep warning distinct (red)
        "info": "#a7c7e7",  # Calm info blue
        "dim": "#707070",
        "yes-option": "#e8ecf5",
        "no-option": "#e8ecf5",
        "value": "#e8ecf5",
        "default": "#88c0d0",
        "marker": "#9fb3d1",
    }
    style_map.update(_choice_diff_style_slots())
    return Style.from_dict(style_map)


def resolve_choice_style(style_variant: str) -> Style:
    """Resolve a named style variant for choice prompts.

    Supported variants:
    - neutral (default)
    - amber
    - onboarding
    - theme
    - ask_user_question
    """
    style_builders = {
        "neutral": neutral_choice_style,
        "amber": amber_choice_style,
        "onboarding": onboarding_style,
        "theme": theme_style,
        "ask_user_question": ask_user_question_style,
    }
    builder = style_builders.get(style_variant, neutral_choice_style)
    return builder()


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
    plain_text = ANSI_ESCAPE_RE.sub("", plain_text)

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


def _selection_label_offset(
    option_count: int,
    open_character: str,
    close_character: str,
) -> int:
    """Approximate visible prefix width before option label text."""
    digits = max(2, len(str(max(1, option_count))))
    number_width = digits + 2  # "<num>. "
    marker_width = 1
    spacing_after_marker = 1
    return len(open_character) + marker_width + len(close_character) + spacing_after_marker + number_width


def _selection_number_offset(open_character: str, close_character: str) -> int:
    """Visible prefix width before the option number column."""
    marker_width = 1
    spacing_after_marker = 1
    return len(open_character) + marker_width + len(close_character) + spacing_after_marker


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
    style_variant: str = "neutral",
    external_header: Optional[str] = None,
    shift_tab_value: Optional[str] = None,
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
        style: Optional custom style object (highest precedence)
        style_variant: Named style variant when style is not provided

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

    if shift_tab_value is not None:
        @key_bindings.add("s-tab", eager=True)
        def _shift_tab_handler(event: Any) -> None:  # noqa: ANN001 (called by key_binding)
            event.app.exit(result=shift_tab_value, style="class:accepted")

    if external_header:
        print(external_header)

    # Show the choice dialog
    result = choice(
        message=formatted_prompt,
        options=choice_options_formatted,
        style=style or resolve_choice_style(style_variant),
        show_frame=cast(Any, ~is_done),
        key_bindings=key_bindings,
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
    if external_header:
        lines_to_clear += _count_display_lines(external_header, term_width)

    for _ in range(lines_to_clear):
        print("\033[F\033[2K", end="", flush=True)

    return result


async def prompt_choice_async(
    message: str,
    options: list[ChoiceOption] | list[tuple[str, str]],
    *,
    title: Optional[str] = None,
    description: Optional[str] = None,
    warning: Optional[str] = None,
    allow_esc: bool = True,
    esc_value: Optional[str] = None,
    style: Optional[Style] = None,
    style_variant: str = "neutral",
    back_value: Optional[str] = None,
    next_value: Optional[str] = None,
    external_header: Optional[str] = None,
    custom_input_label: Optional[str] = None,
    shift_tab_value: Optional[str] = None,
) -> str:
    """Async variant of prompt_choice for use inside running event loops."""
    choice_options: list[ChoiceOption] = []
    for opt in options:
        if isinstance(opt, ChoiceOption):
            choice_options.append(opt)
        else:
            value, label = opt
            choice_options.append(ChoiceOption(value, label))

    prompt_html = ""
    if title:
        prompt_html += f"<title>{html.escape(title)}</title>\n"
    if description:
        prompt_html += f"<description>{html.escape(description)}</description>\n"
    prompt_html += message
    if warning:
        prompt_html += f"\n<warning>{html.escape(warning)}</warning>"

    formatted_prompt = HTML(f"\n{prompt_html}\n")

    choice_options_formatted = []
    for opt in choice_options:
        label = opt.label
        if opt.is_default:
            label = f"{label} <default>*</default>"
        choice_options_formatted.append((opt.value, HTML(label) if "<" in label else label))

    key_bindings = KeyBindings()
    if allow_esc:
        default_esc_value = esc_value or (choice_options[0].value if choice_options else "")
        result_on_esc = default_esc_value

        @key_bindings.add("escape", eager=False)
        def _esc_handler(event: Any) -> None:  # noqa: ANN001 (called by key_binding)
            event.app.exit(result=result_on_esc, style="class:aborting")

    if custom_input_label:
        list_padding_left = 3
        radio_list = RadioList(
            values=choice_options_formatted,
            default=choice_options_formatted[0][0] if choice_options_formatted else None,
            select_on_focus=True,
            open_character="",
            select_character=">",
            close_character="",
            show_cursor=False,
            show_numbers=True,
            container_style="class:input-selection",
            default_style="class:option",
            selected_style="",
            checked_style="class:selected-option",
            number_style="class:number",
            show_scrollbar=False,
        )

        custom_input = TextArea(
            text="",
            multiline=False,
            prompt=f"{custom_input_label}: ",
            style="class:option",
            focusable=True,
        )
        # Align custom input with the option number column for single-choice prompts.
        custom_padding_left = list_padding_left + _selection_number_offset("", "")

        def _set_radio_focus_visual(active: bool) -> None:
            radio_list.select_character = ">" if active else " "

        container: Any = HSplit(
            [
                Box(
                    Label(text=formatted_prompt, dont_extend_height=True),
                    padding_top=0,
                    padding_left=1,
                    padding_right=1,
                    padding_bottom=0,
                ),
                Box(
                    radio_list,
                    padding_top=0,
                    padding_left=list_padding_left,
                    padding_right=1,
                    padding_bottom=0,
                ),
                Box(
                    custom_input,
                    padding_top=0,
                    padding_left=custom_padding_left,
                    padding_right=1,
                    padding_bottom=0,
                ),
            ]
        )
        container = ConditionalContainer(
            Frame(container),
            alternative_content=container,
            filter=~is_done,
        )
        layout = Layout(container, focused_element=radio_list)

        def _collect_choice_result() -> str:
            custom_text = custom_input.text.strip()
            if custom_text:
                return custom_text
            return str(radio_list.current_value)

        @key_bindings.add("enter", eager=True)
        def _submit(event: Any) -> None:  # noqa: ANN001
            event.app.exit(result=_collect_choice_result(), style="class:accepted")

        @key_bindings.add("left", filter=has_focus(radio_list), eager=True)
        def _left_nav(event: Any) -> None:  # noqa: ANN001
            if back_value is not None:
                event.app.exit(result=back_value, style="class:accepted")

        @key_bindings.add("right", filter=has_focus(radio_list), eager=True)
        def _right_nav(event: Any) -> None:  # noqa: ANN001
            if next_value is not None:
                event.app.exit(result=next_value, style="class:accepted")

        @key_bindings.add("tab", eager=True)
        @key_bindings.add("c-i", eager=True)
        def _focus_next(event: Any) -> None:  # noqa: ANN001
            if event.app.layout.has_focus(radio_list):
                _set_radio_focus_visual(False)
                event.app.layout.focus(custom_input)
            else:
                _set_radio_focus_visual(True)
                event.app.layout.focus(radio_list)

        @key_bindings.add("s-tab", eager=True)
        def _focus_prev(event: Any) -> None:  # noqa: ANN001
            if event.app.layout.has_focus(custom_input):
                _set_radio_focus_visual(True)
                event.app.layout.focus(radio_list)
            else:
                _set_radio_focus_visual(False)
                event.app.layout.focus(custom_input)

        @Condition
        def _is_at_last_option() -> bool:
            selected_index = getattr(radio_list, "_selected_index", 0)
            values = getattr(radio_list, "values", [])
            return bool(values) and selected_index >= len(values) - 1

        @key_bindings.add("down", filter=has_focus(radio_list) & _is_at_last_option, eager=True)
        def _down_to_custom(event: Any) -> None:  # noqa: ANN001
            _set_radio_focus_visual(False)
            event.app.layout.focus(custom_input)

        @key_bindings.add("up", filter=has_focus(custom_input), eager=True)
        def _up_to_list(event: Any) -> None:  # noqa: ANN001
            _set_radio_focus_visual(True)
            event.app.layout.focus(radio_list)

        if external_header:
            print(external_header)

        result: str = await Application(
            layout=layout,
            full_screen=False,
            key_bindings=key_bindings,
            style=style or resolve_choice_style(style_variant),
        ).run_async()
    else:
        if shift_tab_value is not None:
            @key_bindings.add("s-tab", eager=True)
            def _shift_tab_cycle(event: Any) -> None:  # noqa: ANN001 (called by key_binding)
                event.app.exit(result=shift_tab_value, style="class:accepted")

        @key_bindings.add("left", eager=True)
        def _left_nav(event: Any) -> None:  # noqa: ANN001 (called by key_binding)
            if back_value is not None:
                event.app.exit(result=back_value, style="class:accepted")

        @key_bindings.add("right", eager=True)
        def _right_nav(event: Any) -> None:  # noqa: ANN001 (called by key_binding)
            if next_value is not None:
                event.app.exit(result=next_value, style="class:accepted")

        if external_header:
            print(external_header)

        result = await ChoiceInput(
            message=formatted_prompt,
            options=choice_options_formatted,
            style=style or resolve_choice_style(style_variant),
            show_frame=~is_done,
            key_bindings=key_bindings,
        ).prompt_async()

    term_width = _terminal_width()
    message_width = max(1, term_width - 2)
    lines_to_clear = _count_display_lines(formatted_prompt, message_width)
    lines_to_clear += len(choice_options_formatted)
    if custom_input_label:
        lines_to_clear += 1
    if external_header:
        lines_to_clear += _count_display_lines(external_header, term_width)

    for _ in range(lines_to_clear):
        print("\033[F\033[2K", end="", flush=True)

    return str(result)


async def prompt_checkbox_async(
    message: str,
    options: list[ChoiceOption] | list[tuple[str, str]],
    *,
    title: Optional[str] = None,
    description: Optional[str] = None,
    warning: Optional[str] = None,
    style: Optional[Style] = None,
    style_variant: str = "neutral",
    custom_input_label: Optional[str] = None,
    back_value: Optional[str] = None,
    next_value: Optional[str] = None,
    submit_on_right: bool = False,
    external_header: Optional[str] = None,
) -> Optional[list[str]]:
    """Prompt the user to select multiple options via a checkbox dialog.

    Returns:
        List of selected values, or None if the dialog was cancelled.
    """
    choice_options: list[ChoiceOption] = []
    for opt in options:
        if isinstance(opt, ChoiceOption):
            choice_options.append(opt)
        else:
            value, label = opt
            choice_options.append(ChoiceOption(value, label))

    prompt_html = ""
    if title:
        prompt_html += f"<title>{html.escape(title)}</title>\n"
    if description:
        prompt_html += f"<description>{html.escape(description)}</description>\n"
    prompt_html += message
    if warning:
        prompt_html += f"\n<warning>{html.escape(warning)}</warning>"
    formatted_prompt = HTML(prompt_html)

    checkbox_values = []
    for opt in choice_options:
        label = opt.label
        if opt.is_default:
            label = f"{label} <default>*</default>"
        checkbox_values.append((opt.value, HTML(label) if "<" in label else label))

    checkbox_list = CheckboxList(
        values=checkbox_values,
        open_character="[",
        select_character="x",
        close_character="]",
        container_style="class:input-selection",
        default_style="class:option",
        selected_style="class:selected-option",
        checked_style="class:selected-option",
    )
    checkbox_list.show_numbers = True
    checkbox_list.number_style = "class:number"
    checkbox_list.show_scrollbar = False

    list_padding_left = 3
    custom_input = TextArea(
        text="",
        multiline=False,
        prompt=(f"{custom_input_label}: " if custom_input_label else ""),
        style="class:option",
        focusable=True,
    )
    custom_padding_left = list_padding_left + _selection_label_offset(
        len(checkbox_values), "[", "]"
    )
    custom_input_container = Box(
        custom_input,
        padding_top=0,
        padding_left=custom_padding_left,
        padding_right=1,
        padding_bottom=0,
    )
    body_children: list[Any] = [
        Box(
            Label(text=formatted_prompt, dont_extend_height=True),
            padding_top=0,
            padding_left=1,
            padding_right=1,
            padding_bottom=0,
        ),
        Box(
            checkbox_list,
            padding_top=0,
            padding_left=list_padding_left,
            padding_right=1,
            padding_bottom=0,
        ),
    ]
    if custom_input_label:
        body_children.append(custom_input_container)

    base_container: Any = HSplit(body_children)
    container = ConditionalContainer(
        Frame(base_container),
        alternative_content=base_container,
        filter=~is_done,
    )
    layout = Layout(container, focused_element=checkbox_list)

    kb = KeyBindings()

    def _collect_result() -> list[str]:
        values = [str(item) for item in checkbox_list.current_values]
        if custom_input_label:
            custom_text = custom_input.text.strip()
            if custom_text:
                values.append(custom_text)
        return values

    @kb.add("enter", eager=True)
    def _submit(event: Any) -> None:  # noqa: ANN401
        event.app.exit(result=_collect_result(), style="class:accepted")

    @kb.add("right", eager=True)
    def _submit_right(event: Any) -> None:  # noqa: ANN401
        if event.app.layout.current_window != checkbox_list.window:
            return
        if next_value is not None:
            event.app.exit(result=[next_value], style="class:accepted")
        elif submit_on_right:
            event.app.exit(result=_collect_result(), style="class:accepted")

    @kb.add("left", eager=True)
    def _go_back_left(event: Any) -> None:  # noqa: ANN401
        if event.app.layout.current_window != checkbox_list.window:
            return
        if back_value is not None:
            event.app.exit(result=[back_value], style="class:accepted")

    if custom_input_label:

        @kb.add("tab", eager=True)
        @kb.add("c-i", eager=True)
        def _focus_next(event: Any) -> None:  # noqa: ANN401
            if event.app.layout.has_focus(checkbox_list):
                event.app.layout.focus(custom_input)
            else:
                event.app.layout.focus(checkbox_list)

        @kb.add("s-tab", eager=True)
        def _focus_prev(event: Any) -> None:  # noqa: ANN401
            if event.app.layout.has_focus(custom_input):
                event.app.layout.focus(checkbox_list)
            else:
                event.app.layout.focus(custom_input)

        @Condition
        def _is_at_last_option() -> bool:
            selected_index = getattr(checkbox_list, "_selected_index", 0)
            values = getattr(checkbox_list, "values", [])
            return bool(values) and selected_index >= len(values) - 1

        @kb.add("down", filter=has_focus(checkbox_list) & _is_at_last_option, eager=True)
        def _down_to_custom(event: Any) -> None:  # noqa: ANN401
            event.app.layout.focus(custom_input)

        @kb.add("up", filter=has_focus(custom_input), eager=True)
        def _up_to_list(event: Any) -> None:  # noqa: ANN401
            event.app.layout.focus(checkbox_list)

    if external_header:
        print(external_header)

    @kb.add("escape", eager=False)
    def _cancel(event: Any) -> None:  # noqa: ANN401
        event.app.exit(result=None, style="class:aborting")

    @kb.add("c-c", eager=True)
    @kb.add("<sigint>", eager=True)
    def _interrupt(event: Any) -> None:  # noqa: ANN401
        event.app.exit(result=None, style="class:aborting")

    result: Optional[list[str]] = await Application(
        layout=layout,
        full_screen=False,
        key_bindings=kb,
        style=style or resolve_choice_style(style_variant),
    ).run_async()

    # Clear prompt area, matching the behavior of prompt_choice_async.
    term_width = _terminal_width()
    message_width = max(1, term_width - 2)
    lines_to_clear = _count_display_lines(formatted_prompt, message_width)
    lines_to_clear += len(checkbox_values)
    if custom_input_label:
        lines_to_clear += 1
    if external_header:
        lines_to_clear += _count_display_lines(external_header, term_width)
    for _ in range(lines_to_clear):
        print("\033[F\033[2K", end="", flush=True)

    if result is None:
        return None
    return [str(item) for item in result]


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
