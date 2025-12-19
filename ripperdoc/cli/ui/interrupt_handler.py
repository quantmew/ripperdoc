"""Interrupt handling for RichUI.

This module handles ESC/Ctrl+C key detection during query execution,
including terminal raw mode management.
"""

import asyncio
import contextlib
import sys
from typing import Any, Optional, Set

from ripperdoc.utils.log import get_logger

logger = get_logger()

# Keys that trigger interrupt
INTERRUPT_KEYS: Set[str] = {"\x1b", "\x03"}  # ESC, Ctrl+C


class InterruptHandler:
    """Handles keyboard interrupt detection during async operations."""

    def __init__(self) -> None:
        """Initialize the interrupt handler."""
        self._query_interrupted: bool = False
        self._esc_listener_active: bool = False
        self._esc_listener_paused: bool = False
        self._stdin_fd: Optional[int] = None
        self._stdin_old_settings: Optional[list] = None
        self._stdin_in_raw_mode: bool = False
        self._abort_callback: Optional[Any] = None

    def set_abort_callback(self, callback: Any) -> None:
        """Set the callback to trigger when interrupt is detected."""
        self._abort_callback = callback

    @property
    def was_interrupted(self) -> bool:
        """Check if the last query was interrupted."""
        return self._query_interrupted

    def pause_listener(self) -> bool:
        """Pause ESC listener and restore cooked terminal mode if we own raw mode.

        Returns:
            Previous paused state for later restoration.
        """
        prev = self._esc_listener_paused
        self._esc_listener_paused = True
        try:
            import termios
        except ImportError:
            return prev

        if (
            self._stdin_fd is not None
            and self._stdin_old_settings is not None
            and self._stdin_in_raw_mode
        ):
            with contextlib.suppress(OSError, termios.error, ValueError):
                termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._stdin_old_settings)
            self._stdin_in_raw_mode = False
        return prev

    def resume_listener(self, previous_state: bool) -> None:
        """Restore paused state to what it was before a blocking prompt."""
        self._esc_listener_paused = previous_state

    async def _listen_for_interrupt_key(self) -> bool:
        """Listen for interrupt keys (ESC/Ctrl+C) during query execution.

        Uses raw terminal mode for immediate key detection without waiting
        for escape sequences to complete.

        Returns:
            True if an interrupt key was pressed.
        """
        import select
        import termios
        import tty

        try:
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
        except (OSError, termios.error, ValueError):
            return False

        self._stdin_fd = fd
        self._stdin_old_settings = old_settings
        raw_enabled = False
        try:
            while self._esc_listener_active:
                if self._esc_listener_paused:
                    if raw_enabled:
                        with contextlib.suppress(OSError, termios.error, ValueError):
                            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                        raw_enabled = False
                        self._stdin_in_raw_mode = False
                    await asyncio.sleep(0.05)
                    continue

                if not raw_enabled:
                    tty.setraw(fd)
                    raw_enabled = True
                    self._stdin_in_raw_mode = True

                await asyncio.sleep(0.02)
                if select.select([sys.stdin], [], [], 0)[0]:
                    if sys.stdin.read(1) in INTERRUPT_KEYS:
                        return True
        except (OSError, ValueError):
            pass
        finally:
            self._stdin_in_raw_mode = False
            with contextlib.suppress(OSError, termios.error, ValueError):
                if raw_enabled:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            self._stdin_fd = None
            self._stdin_old_settings = None

        return False

    async def _cancel_task(self, task: asyncio.Task) -> None:
        """Cancel a task and wait for it to finish."""
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    def _trigger_abort(self) -> None:
        """Signal the query to abort via callback."""
        if self._abort_callback is not None:
            self._abort_callback()

    async def run_with_interrupt(self, query_coro: Any) -> bool:
        """Run a coroutine with ESC key interrupt support.

        Args:
            query_coro: The coroutine to run with interrupt support.

        Returns:
            True if interrupted, False if completed normally.
        """
        self._query_interrupted = False
        self._esc_listener_active = True

        query_task = asyncio.create_task(query_coro)
        interrupt_task = asyncio.create_task(self._listen_for_interrupt_key())

        try:
            done, _ = await asyncio.wait(
                {query_task, interrupt_task}, return_when=asyncio.FIRST_COMPLETED
            )

            # Check if interrupted
            if interrupt_task in done and interrupt_task.result():
                self._query_interrupted = True
                self._trigger_abort()
                await self._cancel_task(query_task)
                return True

            # Query completed normally
            if query_task in done:
                await self._cancel_task(interrupt_task)
                with contextlib.suppress(Exception):
                    query_task.result()
                return False

            return False

        finally:
            self._esc_listener_active = False
            await self._cancel_task(query_task)
            await self._cancel_task(interrupt_task)
