"""Shared hook execution state utilities."""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Generator, List, Optional, Sequence

from ripperdoc.core.hooks.config import HooksConfig

from ripperdoc.utils.log import get_logger
from ripperdoc.utils.pending_messages import PendingMessageQueue

logger = get_logger()

_hooks_suspended: ContextVar[bool] = ContextVar("ripperdoc_hooks_suspended", default=False)
_pending_message_queue: ContextVar[Optional[PendingMessageQueue]] = ContextVar(
    "ripperdoc_hook_pending_queue", default=None
)
_status_emitter: ContextVar[Optional[Callable[[str], None]]] = ContextVar(
    "ripperdoc_hook_status_emitter", default=None
)
_hook_scopes: ContextVar[Optional[Sequence[HooksConfig]]] = ContextVar(
    "ripperdoc_hook_scopes", default=None
)


def _safe_reset(context_var: ContextVar[Any], token: Any, *, var_name: str) -> None:
    try:
        context_var.reset(token)
    except ValueError as exc:
        # Async generator shutdown can cross context boundaries. In that case
        # resetting the original token fails, but cleanup should continue.
        if "different Context" in str(exc):
            logger.debug("[hooks.state] Skipping %s reset across contexts: %s", var_name, exc)
            return
        raise


@contextmanager
def suspend_hooks() -> Generator[None, None, None]:
    """Temporarily disable hook execution (used for internal subagents)."""
    token = _hooks_suspended.set(True)
    try:
        yield
    finally:
        _safe_reset(_hooks_suspended, token, var_name="hooks_suspended")


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
        _safe_reset(_pending_message_queue, token, var_name="pending_message_queue")


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
        _safe_reset(_status_emitter, token, var_name="status_emitter")


def get_hook_status_emitter() -> Optional[Callable[[str], None]]:
    """Get the currently bound hook status emitter."""
    return _status_emitter.get()


@contextmanager
def bind_hook_scopes(
    scopes: Optional[Sequence[HooksConfig]],
) -> Generator[None, None, None]:
    """Bind additional hook scopes for the current execution context."""
    token = _hook_scopes.set(scopes)
    try:
        yield
    finally:
        _safe_reset(_hook_scopes, token, var_name="hook_scopes")


def get_hook_scopes() -> List[HooksConfig]:
    """Get any hook scopes bound to the current execution context."""
    scopes = _hook_scopes.get()
    if not scopes:
        return []
    return list(scopes)
