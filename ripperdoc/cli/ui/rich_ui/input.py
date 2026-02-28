"""Input handling and prompt session setup for the Rich UI."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import Any, Iterable

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion, merge_completers
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.shortcuts.prompt import CompleteStyle
from prompt_toolkit.styles import Style
from prompt_toolkit.lexers import SimpleLexer

from ripperdoc.cli.commands import CustomCommandDefinition, slash_command_completions
from ripperdoc.cli.ui.file_mention_completer import FileMentionCompleter
from ripperdoc.utils.log import get_logger

logger = get_logger()


def build_prompt_session(
    ui: object,
    ignore_filter: Any,
    *,
    disable_slash_commands: bool = False,
) -> PromptSession:
    """Create a PromptSession with slash and file completion."""

    class SlashCommandCompleter(Completer):
        """Autocomplete for slash commands including custom commands."""

        def __init__(self, project_path: Path):
            self.project_path = project_path

        def get_completions(self, document: Any, _complete_event: Any) -> Iterable[Completion]:
            text = document.text_before_cursor
            if not text.startswith("/"):
                return
            query = text[1:]
            # Get completions including custom commands
            completions = slash_command_completions(self.project_path)
            for name, cmd in completions:
                if name.startswith(query):
                    # Handle both SlashCommand and CustomCommandDefinition
                    description = cmd.description
                    # Add hint for custom commands
                    if isinstance(cmd, CustomCommandDefinition):
                        hint = cmd.argument_hint or ""
                        display = f"{name} {hint}".strip() if hint else name
                        display_meta = f"[custom] {description}"
                    else:
                        display = name
                        display_meta = description
                    yield Completion(
                        name,
                        start_position=-len(query),
                        display=display,
                        display_meta=display_meta,
                    )

    # Merge both completers
    file_completer = FileMentionCompleter(ui.project_path, ignore_filter)
    if disable_slash_commands:
        combined_completer = file_completer
    else:
        slash_completer = SlashCommandCompleter(ui.project_path)
        combined_completer = merge_completers([slash_completer, file_completer])

    key_bindings = KeyBindings()

    @key_bindings.add("enter")
    def _(event: Any) -> None:
        """Accept completion if menu is open; otherwise submit line."""
        buf = event.current_buffer
        if buf.complete_state and buf.complete_state.current_completion:
            buf.apply_completion(buf.complete_state.current_completion)
            return
        buf.validate_and_handle()

    @key_bindings.add("tab")
    def _(event: Any) -> None:
        """Toggle thinking mode when input is empty; otherwise handle completion."""
        buf = event.current_buffer
        # If input is empty, toggle thinking mode
        if not buf.text.strip():
            from prompt_toolkit.application import run_in_terminal

            def _toggle() -> None:
                ui._toggle_thinking_mode()

            run_in_terminal(_toggle)
            return
        # Otherwise, handle completion as usual
        if buf.complete_state and buf.complete_state.current_completion:
            buf.apply_completion(buf.complete_state.current_completion)
        else:
            buf.start_completion(select_first=True)

    @key_bindings.add("s-tab")
    def _(event: Any) -> None:
        """Cycle permission mode when input is empty; otherwise cycle completion backward."""
        buf = event.current_buffer
        if not buf.text.strip():
            from prompt_toolkit.application import run_in_terminal

            def _cycle_mode() -> None:
                ui._cycle_permission_mode()

            run_in_terminal(_cycle_mode)
            return
        if buf.complete_state:
            buf.complete_previous()

    @key_bindings.add("escape", "enter")
    def _(event: Any) -> None:
        """Insert newline on Alt+Enter."""
        event.current_buffer.insert_text("\n")

    # Capture self for use in key binding closures
    ui_instance = ui

    @key_bindings.add("c-c")
    def _(event: Any) -> None:
        """Handle Ctrl+C: first press clears input, second press exits."""
        import time as time_module

        buf = event.current_buffer
        current_text = buf.text
        current_time = time_module.time()

        # Check if this is a double Ctrl+C (within 1.5 seconds)
        if current_time - ui_instance._last_ctrl_c_time < 1.5:
            # Double Ctrl+C - exit
            buf.reset()
            raise KeyboardInterrupt()

        # First Ctrl+C - save to history and clear
        ui_instance._last_ctrl_c_time = current_time

        if current_text.strip():
            # Save current input to history before clearing
            try:
                event.app.current_buffer.history.append_string(current_text)
            except (AttributeError, TypeError, ValueError):
                pass

        # Print hint message in clean terminal context, then clear buffer
        from prompt_toolkit.application import run_in_terminal

        def _print_hint() -> None:
            print("\n\033[2mPress Ctrl+C again to exit, or continue typing.\033[0m")

        run_in_terminal(_print_hint)

        # Clear the buffer after printing
        buf.reset()

    @key_bindings.add("escape", "escape")
    async def _(event: Any) -> None:
        """Open the conversation history picker on double ESC."""
        from prompt_toolkit.application import in_terminal

        buf = event.current_buffer
        current_text = buf.text
        cursor_pos = buf.cursor_position

        async with in_terminal():
            handler = getattr(ui_instance, "_open_history_picker_async", None)
            if callable(handler):
                result = handler()
                if inspect.isawaitable(result):
                    did_rollback = bool(await result)
                else:
                    did_rollback = bool(result)
            else:
                did_rollback = False

        # Restore or clear input after returning from the picker.
        if did_rollback:
            buf.text = ""
            buf.cursor_position = 0
        else:
            buf.text = current_text
            buf.cursor_position = min(cursor_pos, len(current_text))

    # If stdin is not a TTY (e.g., piped input), try to use /dev/tty for interactive input
    # This allows the user to continue interacting after processing piped content
    input_obj = None
    if not sys.stdin.isatty():
        # First check if /dev/tty exists and is accessible
        try:
            import os

            if os.path.exists("/dev/tty"):
                from prompt_toolkit.input import create_input

                input_obj = create_input(always_prefer_tty=True)
                ui._using_tty_input = True  # Mark that we're using /dev/tty
                logger.info(
                    "[ui] Stdin is not a TTY, using /dev/tty for prompt input",
                    extra={"session_id": ui.session_id},
                )
            else:
                logger.info(
                    "[ui] Stdin is not a TTY and /dev/tty not available",
                    extra={"session_id": ui.session_id},
                )
        except (OSError, RuntimeError, ValueError, ImportError) as exc:
            logger.warning(
                "[ui] Failed to create TTY input: %s: %s",
                type(exc).__name__,
                exc,
                extra={"session_id": ui.session_id},
            )

    prompt_style = Style.from_dict(
        {
            "rprompt-on": "fg:ansicyan bold",
            "rprompt-off": "fg:ansibrightblack",
            "rprompt-sep": "fg:ansibrightblack",
            "rprompt-mode-normal": "fg:ansibrightblack",
            "rprompt-mode-accept": "fg:ansiyellow bold",
            "rprompt-mode-plan": "fg:ansiblue bold",
            "rprompt-mode-bypass": "fg:ansired bold",
        }
    )
    return PromptSession(
        completer=combined_completer,
        complete_style=CompleteStyle.COLUMN,
        complete_while_typing=True,
        history=InMemoryHistory(),
        key_bindings=key_bindings,
        multiline=True,
        input=input_obj,
        style=prompt_style,
        rprompt=ui._get_rprompt,
        lexer=SimpleLexer('bg:#444444 #ffffff')
    )
