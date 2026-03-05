"""Token refresh scheduler for bridge session ingress tokens."""

from __future__ import annotations

import threading
import time
from typing import Callable

from ripperdoc.utils.log import get_logger

from .constants import (
    TOKEN_REFRESH_BUFFER_SEC,
    TOKEN_REFRESH_FOLLOWUP_SEC,
    TOKEN_REFRESH_MAX_RETRIES,
    TOKEN_REFRESH_RETRY_SEC,
)
from .utils import format_iso_from_epoch, get_token_expiration_epoch

logger = get_logger()


class TokenSessionManager:
    """Manage per-session token refresh timers with generation guards."""

    def __init__(
        self,
        *,
        get_access_token: Callable[[], str | None],
        on_refresh: Callable[[str, str], None],
        label: str = "bridge",
        refresh_buffer_sec: int = TOKEN_REFRESH_BUFFER_SEC,
        followup_refresh_sec: int = TOKEN_REFRESH_FOLLOWUP_SEC,
        max_retries: int = TOKEN_REFRESH_MAX_RETRIES,
        retry_delay_sec: int = TOKEN_REFRESH_RETRY_SEC,
    ) -> None:
        self._get_access_token = get_access_token
        self._on_refresh = on_refresh
        self._label = label
        self._refresh_buffer_sec = max(0, int(refresh_buffer_sec))
        self._followup_refresh_sec = max(1, int(followup_refresh_sec))
        self._max_retries = max(1, int(max_retries))
        self._retry_delay_sec = max(1, int(retry_delay_sec))

        self._timers: dict[str, threading.Timer] = {}
        self._retry_counts: dict[str, int] = {}
        self._generation: dict[str, int] = {}
        self._lock = threading.Lock()

    def _next_generation(self, session_id: str) -> int:
        generation = self._generation.get(session_id, 0) + 1
        self._generation[session_id] = generation
        return generation

    def _clear_timer(self, session_id: str) -> None:
        timer = self._timers.pop(session_id, None)
        if timer is not None:
            timer.cancel()

    def schedule(self, session_id: str, token: str) -> None:
        """Schedule refresh based on token expiry when decodable."""
        expiration_epoch = get_token_expiration_epoch(token)
        if expiration_epoch is None:
            logger.debug(
                "[%s:token] Missing/undecodable exp for sessionId=%s; leaving existing schedule",
                self._label,
                session_id,
            )
            return

        now = int(time.time())
        delay = expiration_epoch - now - self._refresh_buffer_sec
        with self._lock:
            generation = self._next_generation(session_id)
            self._clear_timer(session_id)

            if delay <= 0:
                logger.info(
                    "[%s:token] sessionId=%s token exp=%s already within buffer; refreshing now",
                    self._label,
                    session_id,
                    format_iso_from_epoch(expiration_epoch),
                )
                self._schedule_now(session_id, generation)
                return

            logger.info(
                "[%s:token] sessionId=%s scheduling refresh in %ss (exp=%s buffer=%ss)",
                self._label,
                session_id,
                delay,
                format_iso_from_epoch(expiration_epoch),
                self._refresh_buffer_sec,
            )
            timer = threading.Timer(
                delay,
                self._refresh_token,
                args=(session_id, generation),
            )
            timer.daemon = True
            self._timers[session_id] = timer
            timer.start()

    def _schedule_now(self, session_id: str, generation: int) -> None:
        timer = threading.Timer(0.0, self._refresh_token, args=(session_id, generation))
        timer.daemon = True
        self._timers[session_id] = timer
        timer.start()

    def _schedule_retry(self, session_id: str, generation: int, delay_sec: float) -> None:
        timer = threading.Timer(delay_sec, self._refresh_token, args=(session_id, generation))
        timer.daemon = True
        self._timers[session_id] = timer
        timer.start()

    def _refresh_token(self, session_id: str, generation: int) -> None:
        with self._lock:
            if self._generation.get(session_id) != generation:
                return

        new_token = None
        try:
            new_token = self._get_access_token()
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "[%s:token] get_access_token failed for sessionId=%s: %s",
                self._label,
                session_id,
                exc,
            )

        with self._lock:
            if self._generation.get(session_id) != generation:
                return

            if not new_token:
                retry_count = self._retry_counts.get(session_id, 0) + 1
                self._retry_counts[session_id] = retry_count
                logger.error(
                    "[%s:token] No token for sessionId=%s (retry %d/%d)",
                    self._label,
                    session_id,
                    retry_count,
                    self._max_retries,
                )
                if retry_count < self._max_retries:
                    self._schedule_retry(session_id, generation, self._retry_delay_sec)
                return

            self._retry_counts.pop(session_id, None)
            logger.info("[%s:token] Refreshed token for sessionId=%s", self._label, session_id)
            self._on_refresh(session_id, new_token)
            self._schedule_retry(session_id, generation, self._followup_refresh_sec)

    def cancel(self, session_id: str) -> None:
        with self._lock:
            self._next_generation(session_id)
            self._clear_timer(session_id)
            self._retry_counts.pop(session_id, None)

    def cancel_all(self) -> None:
        with self._lock:
            for session_id in list(self._generation.keys()):
                self._next_generation(session_id)
            for timer in self._timers.values():
                timer.cancel()
            self._timers.clear()
            self._retry_counts.clear()
