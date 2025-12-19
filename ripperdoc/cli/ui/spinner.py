from contextlib import contextmanager
from typing import Any, Generator, Literal, Optional

from rich.console import Console
from rich.markup import escape
from rich.status import Status


class Spinner:
    """Lightweight spinner wrapper for Rich status."""

    def __init__(self, console: Console, text: str = "Thinking...", spinner: str = "dots"):
        self.console = console
        self.text = text
        self.spinner = spinner
        self._status: Optional[Status] = None

    def start(self) -> None:
        """Start the spinner if not already running."""

        if self._status is not None:
            return
        self._status = self.console.status(
            f"[cyan]{escape(self.text)}[/cyan]", spinner=self.spinner
        )
        self._status.__enter__()

    def update(self, text: Optional[str] = None) -> None:
        """Update spinner text."""

        if self._status is None:
            return
        new_text = text if text is not None else self.text
        self._status.update(f"[cyan]{escape(new_text)}[/cyan]")

    def stop(self) -> None:
        """Stop the spinner if running."""

        if self._status is None:
            return
        self._status.__exit__(None, None, None)
        self._status = None

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
        return self._status is not None

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
