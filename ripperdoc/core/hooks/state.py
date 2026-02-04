"""Shared hook execution state utilities."""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Generator, Optional

from ripperdoc.utils.pending_messages import PendingMessageQueue

_hooks_suspended: ContextVar[bool] = ContextVar("ripperdoc_hooks_suspended", default=False)
_pending_message_queue: ContextVar[Optional[PendingMessageQueue]] = ContextVar(
    "ripperdoc_hook_pending_queue", default=None
)
_status_emitter: ContextVar[Optional[Callable[[str], None]]] = ContextVar(
    "ripperdoc_hook_status_emitter", default=None
)


@contextmanager
def suspend_hooks() -> Generator[None, None, None]:
    """Temporarily disable hook execution (used for internal subagents)."""
    token = _hooks_suspended.set(True)
    try:
        yield
    finally:
        _hooks_suspended.reset(token)


def hooks_suspended() -> bool:
    """Check whether hook execution is currently suspended."""
    return _hooks_suspended.get()


@contextmanager
def bind_pending_message_queue(
    queue: Optional[PendingMessageQueue],
) -> Generator[None, None, None]:
    """Bind a pending message queue for async hook output delivery."""
    token = _pending_message_queue.set(queue)
    try:
        yield
    finally:
        _pending_message_queue.reset(token)


def get_pending_message_queue() -> Optional[PendingMessageQueue]:
    """Get the current pending message queue for async hook output delivery."""
    return _pending_message_queue.get()


@contextmanager
def bind_hook_status_emitter(
    emitter: Optional[Callable[[str], None]],
) -> Generator[None, None, None]:
    """Bind a hook status emitter callback for statusMessage delivery."""
    token = _status_emitter.set(emitter)
    try:
        yield
    finally:
        _status_emitter.reset(token)


def get_hook_status_emitter() -> Optional[Callable[[str], None]]:
    """Get the currently bound hook status emitter."""
    return _status_emitter.get()
