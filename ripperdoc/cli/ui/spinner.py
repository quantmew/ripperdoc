from contextlib import contextmanager
from typing import Any, Generator, Literal, Optional

from rich.console import Console
from rich.live import Live
from rich.text import Text
from rich.spinner import Spinner as RichSpinner


class Spinner:
    """Lightweight spinner wrapper that plays nicely with other console output."""

    def __init__(self, console: Console, text: str = "Thinking...", spinner: str = "dots"):
        self.console = console
        self.text = text
        self.spinner = spinner
        self._style = "cyan"
        self._live: Optional[Live] = None
        # Blue spinner for clearer visual separation in the terminal (icon + text)
        self._renderable: RichSpinner = RichSpinner(
            spinner, text=Text(self.text, style=self._style), style=self._style
        )

    def start(self) -> None:
        """Start the spinner if not already running."""
        if self._live is not None:
            return
        self._renderable.text = Text(self.text, style=self._style)
        self._live = Live(
            self._renderable,
            console=self.console,
            transient=True,  # Remove spinner line when stopped to avoid layout glitches
            refresh_per_second=12,
        )
        self._live.start()

    def update(self, text: Optional[str] = None) -> None:
        """Update spinner text."""
        if self._live is None:
            return
        if text is not None:
            self.text = text
        self._renderable.text = Text(self.text, style=self._style)
        # Live.refresh() redraws the current renderable
        self._live.refresh()

    def stop(self) -> None:
        """Stop the spinner if running."""
        if self._live is None:
            return
        try:
            self._live.stop()
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
                self.start()
