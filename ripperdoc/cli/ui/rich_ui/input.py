"""Input handling and prompt session setup for the Rich UI."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple, Union, cast

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion, merge_completers
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.shortcuts.prompt import CompleteStyle
from prompt_toolkit.styles import Style
from prompt_toolkit.lexers import SimpleLexer

from ripperdoc.commands import CustomCommandDefinition, slash_command_completions
from ripperdoc.cli.ui.file_mention_completer import FileMentionCompleter
from ripperdoc.utils.log import get_logger

logger = get_logger()


def _apply_current_or_first_completion(buf: Any) -> bool:
    """Apply current completion, or fall back to the first completion."""
    state = buf.complete_state
    if state is None:
        return False
    completion = state.current_completion
    if completion is None and state.completions:
        completion = state.completions[0]
    if completion is None:
        return False
    buf.apply_completion(completion)
    return True


def _handle_tab_completion(buf: Any) -> None:
    """Handle completion acceptance for Tab when input is not empty."""
    if buf.complete_state is None:
        buf.start_completion(select_first=True)

    if _apply_current_or_first_completion(buf):
        return

    # Keep previous fallback behavior when the completion menu has no entries yet.
    buf.start_completion(select_first=True)


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
        combined_completer: Completer = file_completer
    else:
        slash_completer = SlashCommandCompleter(ui.project_path)
        combined_completer = cast(Completer, merge_completers([slash_completer, file_completer]))

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

        # Otherwise, handle completion as usual.
        _handle_tab_completion(buf)

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

    # -- visual line helpers --

    def _get_render_info(event: Any) -> Any:
        """Safely get render_info from the current window."""
        return getattr(event.app.layout.current_window, "render_info", None)

    def _char_width(ch: str) -> int:
        """Return the display width of a character (CJK=2, control→1, else 1)."""
        from prompt_toolkit.utils import get_cwidth
        return max(1, get_cwidth(ch))

    def _text_col_at(buf: Any, row_start: int) -> int:
        """Return the cursor's visual column within its visual row,
        measured as the sum of character widths from *row_start*."""
        col = 0
        for i in range(row_start, buf.cursor_position):
            col += _char_width(buf.text[i])
        return col

    def _pos_at_text_col(text: str, row_start: int, row_end: int,
                         target_col: int) -> int:
        """Walk from *row_start* to *row_end* (exclusive) summing char widths,
        and return the position closest to visual column *target_col*."""
        col = 0
        pos = row_start
        best = row_start
        while pos < row_end:
            cw = _char_width(text[pos])
            # Place cursor past this char if its center is ≤ target_col.
            midpoint = col + cw // 2
            if midpoint <= target_col:
                best = pos + 1
            else:
                break
            col += cw
            pos += 1
        return min(best, row_end)

    def _visual_row_boundaries(info: Any, buf: Any) -> Optional[List[Tuple[int, int]]]:
        """Return [(start, end), ...] for each visual row of the current
        logical line, using prompt_toolkit's render_info.  Returns None if
        the information is not available."""
        if info is None:
            return None
        vis = info.visible_line_to_row_col
        if not vis:
            return None
        cursor_row = buf.document.cursor_position_row
        # Build (start, end) pairs from the col values that belong to
        # the cursor's logical row.  The entries in vis may interleave
        # with other logical rows, so we verify each one.
        starts = []
        for vline in range(max(vis.keys()) + 1):
            if vline in vis:
                row, col = vis[vline]
                if row == cursor_row:
                    starts.append(col)
        if not starts:
            return None
        starts.sort()
        boundaries = []
        for i, s in enumerate(starts):
            e = starts[i + 1] if i + 1 < len(starts) else len(buf.text)
            boundaries.append((s, e))
        # If there's only one boundary it means no wrapping detected in
        # visible lines — but there might be invisible rows above/below.
        return boundaries

    def _find_current_row(boundaries: List[Tuple[int, int]],
                          cursor_pos: int) -> Optional[int]:
        """Return the index into *boundaries* that contains *cursor_pos*."""
        for i, (s, e) in enumerate(boundaries):
            if s <= cursor_pos <= e:
                return i
        return None

    # -- key handlers --

    @key_bindings.add("up")
    def _(event: Any) -> None:
        """Move cursor up visually; navigate history only at the visual top."""
        buf = event.current_buffer
        if buf.complete_state:
            buf.complete_previous(count=event.arg)
            return

        info = _get_render_info(event)
        # Multi-logical-line: let prompt_toolkit handle logical row movement.
        if buf.document.cursor_position_row > 0:
            buf.auto_up(count=event.arg)
            return

        # Single logical line (or first logical row) — check for soft wrapping.
        boundaries = _visual_row_boundaries(info, buf)
        if boundaries and len(boundaries) > 1:
            idx = _find_current_row(boundaries, buf.cursor_position)
            if idx is not None and idx > 0:
                target_col = _text_col_at(buf, boundaries[idx][0])
                buf.cursor_position = _pos_at_text_col(
                    buf.text, boundaries[idx - 1][0], boundaries[idx - 1][1],
                    target_col,
                )
                return
        # At visual top or can't determine rows — check scroll state.
        win = event.app.layout.current_window
        if info is not None and (
            not info.top_visible
            or getattr(win, "vertical_scroll_2", 0) > 0
            or (info.cursor_position.y > 0)
        ):
            buf.auto_up(count=event.arg)
        elif not buf.selection_state:
            buf.history_backward(count=event.arg)

    @key_bindings.add("down")
    def _(event: Any) -> None:
        """Move cursor down visually; navigate history only at the visual bottom."""
        buf = event.current_buffer
        if buf.complete_state:
            buf.complete_next(count=event.arg)
            return

        info = _get_render_info(event)
        # Multi-logical-line: let prompt_toolkit handle logical row movement.
        if buf.document.cursor_position_row < buf.document.line_count - 1:
            buf.auto_down(count=event.arg)
            return

        # Last logical row — check for soft wrapping via render_info.
        boundaries = _visual_row_boundaries(info, buf)
        if boundaries and len(boundaries) > 1:
            idx = _find_current_row(boundaries, buf.cursor_position)
            if idx is not None and idx < len(boundaries) - 1:
                target_col = _text_col_at(buf, boundaries[idx][0])
                buf.cursor_position = _pos_at_text_col(
                    buf.text, boundaries[idx + 1][0], boundaries[idx + 1][1],
                    target_col,
                )
                return
        # At visual bottom or can't determine rows.
        if info is not None and (
            not info.bottom_visible
            or info.cursor_position.y < info.window_height - 1
        ):
            buf.auto_down(count=event.arg)
        elif not buf.selection_state:
            buf.history_forward(count=event.arg)

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
            # Exit via prompt_toolkit app API so the exception is delivered to
            # the caller (session.prompt) instead of surfacing as an unhandled
            # event-loop callback exception.
            event.app.exit(exception=KeyboardInterrupt())
            return

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
