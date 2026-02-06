"""ESC key interrupt listener for the Rich UI."""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any, Callable, Optional

from ripperdoc.utils.log import get_logger

if os.name != "nt":
    import select
    import termios
    import tty


class EscInterruptListener:
    """Listen for ESC keypresses in a background thread and invoke a callback."""

    def __init__(self, on_interrupt: Callable[[], None], *, logger: Optional[Any] = None) -> None:
        self._on_interrupt = on_interrupt
        self._logger = logger or get_logger()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._pause_depth = 0
        self._interrupt_sent = False
        self._fd: Optional[int] = None
        self._owns_fd = False
        self._orig_termios: list[Any] | None = None
        self._cbreak_active = False
        self._availability_checked = False
        self._available = True

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_running or not self._available:
            return
        if os.name != "nt" and not self._setup_posix_input():
            return
        self._stop_event.clear()
        with self._lock:
            self._pause_depth = 0
            self._interrupt_sent = False
        self._thread = threading.Thread(
            target=self._run,
            name="ripperdoc-esc-listener",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=0.25)
        self._thread = None
        if os.name != "nt":
            self._restore_posix_input()

    def pause(self) -> None:
        if os.name == "nt":
            return
        with self._lock:
            self._pause_depth += 1
            if self._pause_depth == 1:
                self._restore_termios_locked()

    def resume(self) -> None:
        if os.name == "nt":
            return
        with self._lock:
            if self._pause_depth == 0:
                return
            self._pause_depth -= 1
            if self._pause_depth == 0:
                self._apply_cbreak_locked()

    def _run(self) -> None:
        if os.name == "nt":
            self._run_windows()
        else:
            self._run_posix()

    def _run_windows(self) -> None:
        import msvcrt

        while not self._stop_event.is_set():
            with self._lock:
                paused = self._pause_depth > 0
            if paused:
                time.sleep(0.05)
                continue
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch == "\x1b":
                    self._signal_interrupt()
            time.sleep(0.02)

    def _run_posix(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                paused = self._pause_depth > 0
                fd = self._fd
            if paused or fd is None:
                time.sleep(0.05)
                continue
            try:
                readable, _, _ = select.select([fd], [], [], 0.1)
            except (OSError, ValueError):
                time.sleep(0.05)
                continue
            if not readable:
                continue
            try:
                ch = os.read(fd, 1)
            except OSError:
                continue
            if ch == b"\x1b":
                if self._is_escape_sequence(fd):
                    continue
                self._signal_interrupt()

    def _is_escape_sequence(self, fd: int) -> bool:
        try:
            readable, _, _ = select.select([fd], [], [], 0.02)
        except (OSError, ValueError):
            return False
        if not readable:
            return False
        self._drain_pending_bytes(fd)
        return True

    def _drain_pending_bytes(self, fd: int) -> None:
        while True:
            try:
                readable, _, _ = select.select([fd], [], [], 0)
            except (OSError, ValueError):
                return
            if not readable:
                return
            try:
                os.read(fd, 32)
            except OSError:
                return

    def _signal_interrupt(self) -> None:
        with self._lock:
            if self._interrupt_sent:
                return
            self._interrupt_sent = True
        try:
            self._on_interrupt()
        except (RuntimeError, ValueError, OSError) as exc:
            self._logger.debug(
                "[ui] ESC interrupt callback failed: %s: %s",
                type(exc).__name__,
                exc,
            )

    def _setup_posix_input(self) -> bool:
        if self._fd is not None:
            return True
        fd: Optional[int] = None
        owns = False
        try:
            if sys.stdin.isatty():
                fd = sys.stdin.fileno()
            elif os.path.exists("/dev/tty"):
                fd = os.open("/dev/tty", os.O_RDONLY)
                owns = True
        except OSError as exc:
            self._disable_listener(f"input error: {exc}")
            return False
        if fd is None:
            self._disable_listener("no TTY available")
            return False
        try:
            self._orig_termios = termios.tcgetattr(fd)
        except (termios.error, OSError) as exc:
            if owns:
                try:
                    os.close(fd)
                except OSError:
                    pass
            self._disable_listener(f"termios unavailable: {exc}")
            return False
        self._fd = fd
        self._owns_fd = owns
        self._apply_cbreak_locked()
        return True

    def _restore_posix_input(self) -> None:
        with self._lock:
            self._restore_termios_locked()
            if self._fd is not None and self._owns_fd:
                try:
                    os.close(self._fd)
                except OSError:
                    pass
            self._fd = None
            self._owns_fd = False
            self._orig_termios = None
            self._cbreak_active = False

    def _apply_cbreak_locked(self) -> None:
        if self._fd is None or self._orig_termios is None or self._cbreak_active:
            return
        try:
            tty.setcbreak(self._fd)
            self._cbreak_active = True
        except (termios.error, OSError):
            self._disable_listener("failed to enter cbreak mode")

    def _restore_termios_locked(self) -> None:
        if self._fd is None or self._orig_termios is None or not self._cbreak_active:
            return
        try:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._orig_termios)
        except (termios.error, OSError):
            pass
        self._cbreak_active = False

    def _disable_listener(self, reason: str) -> None:
        if self._availability_checked:
            return
        self._availability_checked = True
        self._available = False
        self._logger.debug("[ui] ESC interrupt listener disabled: %s", reason)
