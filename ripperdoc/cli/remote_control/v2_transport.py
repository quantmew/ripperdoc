"""Transport abstraction for the v2 env-less bridge path.

Provides:
- ReplBridgeTransport: Protocol for read/write transport operations
- SSETransportAdapter: SSE read stream + HTTP POST writes with seq-num resume
- CCRClientAdapter: CCR v2 /worker/* write path with heartbeat
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Protocol
from urllib.parse import urlparse

import httpx

from ripperdoc.utils.log import get_logger

logger = get_logger()


class ReplBridgeTransport(Protocol):
    """Transport protocol for bridge read/write operations."""

    def write(self, message: Dict[str, Any]) -> None: ...
    def write_batch(self, messages: List[Dict[str, Any]]) -> None: ...
    def close(self) -> None: ...
    def set_on_data(self, callback: Callable[[str], None]) -> None: ...
    def set_on_close(self, callback: Callable[[Optional[int]], None]) -> None: ...
    def set_on_connect(self, callback: Callable[[], None]) -> None: ...
    def connect(self) -> None: ...
    def get_last_sequence_num(self) -> int: ...
    def report_state(self, state: str) -> None: ...
    def flush(self) -> None: ...


class SSETransportAdapter:
    """SSE read stream + HTTP POST writes with sequence-number resume.

    Reads inbound events via SSE (Server-Sent Events).
    Writes outbound events via HTTP POST to the CCR /worker/events endpoint.
    """

    def __init__(
        self,
        session_url: str,
        auth_token: str,
        *,
        initial_sequence_num: int = 0,
    ) -> None:
        self._session_url = session_url
        self._auth_token = auth_token
        self._initial_seq = initial_sequence_num
        self._last_seq = initial_sequence_num
        self._closed = False
        self._connected = False

        self._on_data: Optional[Callable[[str], None]] = None
        self._on_close: Optional[Callable[[Optional[int]], None]] = None
        self._on_connect: Optional[Callable[[], None]] = None

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._http = httpx.Client(timeout=30.0)

        parsed = urlparse(session_url)
        scheme = "https" if parsed.scheme in ("https", "wss") else "http"
        self._base_url = f"{scheme}://{parsed.netloc}{parsed.path.rstrip('/')}"
        self._sse_url = f"{self._base_url}/worker/events/stream"
        self._events_url = f"{self._base_url}/worker/events"
        self._state_url = f"{self._base_url}/worker/state"

    def _auth_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._auth_token}",
            "Content-Type": "application/json",
        }

    def write(self, message: Dict[str, Any]) -> None:
        if self._closed:
            return
        try:
            self._http.post(self._events_url, json=message, headers=self._auth_headers())
        except httpx.HTTPError as exc:
            logger.debug("[v2:transport] write failed: %s", exc)

    def write_batch(self, messages: List[Dict[str, Any]]) -> None:
        if self._closed:
            return
        for msg in messages:
            if self._closed:
                break
            self.write(msg)

    def close(self) -> None:
        self._closed = True
        self._stop.set()
        try:
            self._http.close()
        except Exception:
            pass

    def set_on_data(self, callback: Callable[[str], None]) -> None:
        self._on_data = callback

    def set_on_close(self, callback: Callable[[Optional[int]], None]) -> None:
        self._on_close = callback

    def set_on_connect(self, callback: Callable[[], None]) -> None:
        self._on_connect = callback

    def connect(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_sse, daemon=True, name="ripperdoc-v2-sse")
        self._thread.start()

    def get_last_sequence_num(self) -> int:
        return self._last_seq

    def report_state(self, state: str) -> None:
        if self._closed:
            return
        try:
            self._http.put(
                self._state_url,
                json={"state": state},
                headers=self._auth_headers(),
            )
        except httpx.HTTPError:
            pass

    def flush(self) -> None:
        pass

    def _run_sse(self) -> None:
        """SSE read loop using httpx streaming."""
        try:
            headers = self._auth_headers()
            if self._initial_seq > 0:
                headers["Last-Event-ID"] = str(self._initial_seq)

            with self._http.stream("GET", self._sse_url, headers=headers) as response:
                if response.status_code != 200:
                    logger.warning("[v2:sse] SSE connection failed: %s", response.status_code)
                    if self._on_close:
                        self._on_close(response.status_code)
                    return

                self._connected = True
                if self._on_connect:
                    self._on_connect()

                event_data = ""
                event_id = None
                for line in response.iter_lines():
                    if self._stop.is_set():
                        break

                    if line.startswith("id:"):
                        event_id = line[3:].strip()
                        if event_id and event_id.isdigit():
                            self._last_seq = int(event_id)
                    elif line.startswith("data:"):
                        event_data += line[5:]
                    elif line == "":
                        # End of event
                        if event_data and self._on_data:
                            self._on_data(event_data)
                        event_data = ""
                        event_id = None
                    else:
                        event_data += line

        except httpx.HTTPError as exc:
            logger.debug("[v2:sse] SSE error: %s", exc)
        except Exception as exc:  # noqa: BLE001
            logger.debug("[v2:sse] Unexpected error: %s", exc)
        finally:
            self._connected = False
            if not self._stop.is_set() and self._on_close:
                self._on_close(None)


class CCRClientAdapter:
    """CCR v2 /worker/* write path with heartbeat.

    Manages worker registration, event posting, state reporting,
    and periodic heartbeat.
    """

    def __init__(
        self,
        session_url: str,
        auth_token: str,
        *,
        heartbeat_interval_sec: float = 20.0,
        epoch: int = 0,
    ) -> None:
        self._session_url = session_url
        self._auth_token = auth_token
        self._epoch = epoch
        self._heartbeat_interval = heartbeat_interval_sec
        self._http = httpx.Client(timeout=30.0)
        self._closed = False

        parsed = urlparse(session_url)
        scheme = "https" if parsed.scheme in ("https", "wss") else "http"
        base = f"{scheme}://{parsed.netloc}{parsed.path.rstrip('/')}"
        self._events_url = f"{base}/worker/events"
        self._state_url = f"{base}/worker/state"
        self._heartbeat_url = f"{base}/worker/heartbeat"

        self._heartbeat_thread: Optional[threading.Thread] = None
        self._heartbeat_stop = threading.Event()

    def _auth_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._auth_token}",
            "Content-Type": "application/json",
        }

    def initialize(self, epoch: int) -> None:
        """Set the worker epoch and start heartbeat."""
        self._epoch = epoch
        self._start_heartbeat()

    def write_event(self, event: Dict[str, Any]) -> None:
        if self._closed:
            return
        try:
            self._http.post(self._events_url, json=event, headers=self._auth_headers())
        except httpx.HTTPError as exc:
            logger.debug("[v2:ccr] write_event failed: %s", exc)

    def report_state(self, state: str) -> None:
        if self._closed:
            return
        try:
            self._http.put(self._state_url, json={"state": state}, headers=self._auth_headers())
        except httpx.HTTPError:
            pass

    def close(self) -> None:
        self._closed = True
        self._heartbeat_stop.set()
        try:
            self._http.close()
        except Exception:
            pass

    def _start_heartbeat(self) -> None:
        self._heartbeat_stop.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True, name="ripperdoc-v2-ccr-heartbeat"
        )
        self._heartbeat_thread.start()

    def _heartbeat_loop(self) -> None:
        while not self._heartbeat_stop.wait(self._heartbeat_interval):
            if self._closed:
                break
            try:
                self._http.post(self._heartbeat_url, json={}, headers=self._auth_headers())
            except httpx.HTTPError:
                pass


def create_v2_repl_transport(
    session_url: str,
    auth_token: str,
    *,
    epoch: int = 0,
    heartbeat_interval_sec: float = 20.0,
    initial_sequence_num: int = 0,
) -> ReplBridgeTransport:
    """Create a v2 transport adapter with SSE reads and CCR writes."""
    sse = SSETransportAdapter(
        session_url,
        auth_token,
        initial_sequence_num=initial_sequence_num,
    )
    ccr = CCRClientAdapter(
        session_url,
        auth_token,
        heartbeat_interval_sec=heartbeat_interval_sec,
        epoch=epoch,
    )

    class _V2Transport:
        def write(self, message: Dict[str, Any]) -> None:
            ccr.write_event(message)

        def write_batch(self, messages: List[Dict[str, Any]]) -> None:
            for msg in messages:
                ccr.write_event(msg)

        def close(self) -> None:
            ccr.close()
            sse.close()

        def set_on_data(self, callback: Callable[[str], None]) -> None:
            sse.set_on_data(callback)

        def set_on_close(self, callback: Callable[[Optional[int]], None]) -> None:
            sse.set_on_close(callback)

        def set_on_connect(self, callback: Callable[[], None]) -> None:
            sse.set_on_connect(callback)

        def connect(self) -> None:
            ccr.initialize(epoch)
            sse.connect()

        def get_last_sequence_num(self) -> int:
            return sse.get_last_sequence_num()

        def report_state(self, state: str) -> None:
            ccr.report_state(state)

        def flush(self) -> None:
            pass

    return _V2Transport()
