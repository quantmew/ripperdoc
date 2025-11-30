from typing import Optional

from rich.console import Console


class Spinner:
    """Lightweight spinner wrapper for Rich status."""

    def __init__(self, console: Console, text: str = "Thinking...", spinner: str = "dots"):
        self.console = console
        self.text = text
        self.spinner = spinner
        self._status = None  # type: Optional[object]

    def start(self) -> None:
        """Start the spinner if not already running."""

        if self._status is not None:
            return
        self._status = self.console.status(f"[cyan]{self.text}[/cyan]", spinner=self.spinner)
        self._status.__enter__()

    def update(self, text: Optional[str] = None) -> None:
        """Update spinner text."""

        if self._status is None:
            return
        new_text = text if text is not None else self.text
        self._status.update(f"[cyan]{new_text}[/cyan]")

    def stop(self) -> None:
        """Stop the spinner if running."""

        if self._status is None:
            return
        self._status.__exit__(None, None, None)
        self._status = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        # Do not suppress exceptions
        return False
