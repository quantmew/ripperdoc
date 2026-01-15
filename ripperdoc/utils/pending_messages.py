"""Thread-safe queue for pending conversation messages.

Allows background tasks or external events to enqueue user messages that
should be injected into the conversation once the current iteration
finishes. Messages are drained in FIFO order.
"""

from collections import deque
import threading
from typing import Any, Deque, Dict, List, Optional

from ripperdoc.utils.messages import UserMessage, create_user_message


class PendingMessageQueue:
    """Thread-safe queue for pending user messages."""

    def __init__(self) -> None:
        self._queue: Deque[UserMessage] = deque()
        self._lock = threading.Lock()

    def enqueue(self, message: UserMessage) -> None:
        """Add a pre-built UserMessage to the queue."""
        with self._lock:
            self._queue.append(message)

    def enqueue_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Create and enqueue a UserMessage with optional metadata."""
        message = create_user_message(text)
        if metadata:
            try:
                message.message.metadata.update(metadata)
            except Exception:
                # Best-effort metadata attachment; ignore failures.
                pass
        self.enqueue(message)

    def drain(self) -> List[UserMessage]:
        """Drain all pending messages in FIFO order."""
        with self._lock:
            if not self._queue:
                return []
            messages = list(self._queue)
            self._queue.clear()
            return messages

    def has_messages(self) -> bool:
        """Check if there are pending messages."""
        with self._lock:
            return bool(self._queue)
