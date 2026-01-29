from contextlib import contextmanager
import shutil
import sys
from typing import Any, Generator, Literal, Optional

from rich.console import Console
from rich.live import Live
from rich.text import Text
from rich.spinner import Spinner as RichSpinner

from ripperdoc.core.theme import theme_color

# ANSI escape sequences for terminal control
_CLEAR_LINE = "\r\033[K"  # Move to start of line and clear to end


class Spinner:
    """Lightweight spinner wrapper that plays nicely with other console output."""

    # Reserve space for spinner animation (e.g., "⠧ ") and safety margin
    _SPINNER_MARGIN = 6

    def __init__(self, console: Console, text: str = "Thinking...", spinner: str = "dots"):
        self.console = console
        self.text = text
        self.spinner = spinner
        self._style = theme_color("spinner")
        self._live: Optional[Live] = None
        # Spinner color from theme for visual separation in the terminal
        self._renderable: RichSpinner = RichSpinner(
            spinner,
            text=Text(self._fit_to_terminal(self.text), style=self._style),
            style=self._style,
        )

    def _get_terminal_width(self) -> int:
        """Get current terminal width, with fallback."""
        try:
            return shutil.get_terminal_size().columns
        except Exception:
            return 80  # Reasonable default

    def _fit_to_terminal(self, text: str) -> str:
        """Truncate text to fit within terminal width, preventing line wrap issues.

        This ensures spinner text never causes terminal wrapping, which would
        leave artifacts when the spinner refreshes or stops.
        """
        max_width = self._get_terminal_width() - self._SPINNER_MARGIN
        if max_width < 20:
            max_width = 20  # Minimum usable width

        if len(text) <= max_width:
            return text

        # Smart truncation: keep the structure intact
        # Find the last complete parenthetical group if possible
        truncated = text[: max_width - 1] + "…"
        return truncated

    def _clear_line(self) -> None:
        """Clear the current terminal line to prevent artifacts."""
        if self.console.is_terminal:
            try:
                sys.stdout.write(_CLEAR_LINE)
                sys.stdout.flush()
            except Exception:
                pass  # Ignore errors in non-TTY environments

    def start(self) -> None:
        """Start the spinner if not already running."""
        if self._live is not None:
            return
        # Clear any residual content on current line before starting
        self._clear_line()
        self._renderable.text = Text(self._fit_to_terminal(self.text), style=self._style)
        self._live = Live(
            self._renderable,
            console=self.console,
            transient=True,  # Remove spinner line when stopped to avoid layout glitches
            refresh_per_second=12,
            vertical_overflow="ellipsis",  # Prevent multi-line overflow issues
        )
        self._live.start()

    def update(self, text: Optional[str] = None) -> None:
        """Update spinner text."""
        if self._live is None:
            return
        if text is not None:
            self.text = text
        self._renderable.text = Text(self._fit_to_terminal(self.text), style=self._style)
        # Live.refresh() redraws the current renderable
        self._live.refresh()

    def stop(self) -> None:
        """Stop the spinner if running."""
        if self._live is None:
            return
        try:
            self._live.stop()
            # Clear line to ensure no artifacts remain from long spinner text
            self._clear_line()
        finally:
            self._live = None

    def __enter__(self) -> "Spinner":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Literal[False]:
        self.stop()
        # Do not suppress exceptions
        return False

    @property
    def is_running(self) -> bool:
        """Check if spinner is currently running."""
        return self._live is not None

    @contextmanager
    def paused(self) -> Generator[None, None, None]:
        """Context manager to temporarily pause the spinner for clean output.

        Usage:
            with spinner.paused():
                console.print("Some output")
        """
        was_running = self.is_running
        if was_running:
            self.stop()
        try:
            yield
        finally:
            if was_running:
                # Ensure all output is flushed and cursor is on a clean line
                # before restarting the spinner
                try:
                    # Flush console buffer
                    self.console.file.flush()
                    # Clear any partial line content to prevent spinner
                    # from appearing on the same line as previous output
                    if self.console.is_terminal:
                        sys.stdout.write(_CLEAR_LINE)
                        sys.stdout.flush()
                except Exception:
                    pass
                self.start()
