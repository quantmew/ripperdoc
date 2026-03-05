"""Main bridge poll/dispatch loop."""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Callable

from rich.console import Console

from ripperdoc.utils.log import get_logger

from .api import RemoteControlApiClient
from .constants import (
    CONNECTION_GIVEUP_SEC,
    CONNECTION_RETRY_INITIAL_SEC,
    CONNECTION_RETRY_MAX_SEC,
    GENERAL_GIVEUP_SEC,
    GENERAL_RETRY_INITIAL_SEC,
    GENERAL_RETRY_MAX_SEC,
    IDENTIFIER_RE,
)
from .errors import BridgeFatalError
from .models import ActiveSession, RemoteControlConfig
from .process import BridgeActivity, RemoteControlProcessSpawner
from .repl_bridge import RemoteSessionBridgeManager, RemoteSessionCallbacks, RemoteSessionConfig
from .token_manager import TokenSessionManager
from .utils import (
    build_connect_url,
    build_session_ingress_ws_url,
    decode_work_secret,
    jitter_delay,
)

console = Console()
logger = get_logger()


class RemoteControlBridgeRunner:
    """Main remote-control poll/dispatch loop."""

    def __init__(
        self,
        *,
        config: RemoteControlConfig,
        api_client: RemoteControlApiClient,
        process_spawner: RemoteControlProcessSpawner,
        stop_event: threading.Event,
        token_supplier: Callable[[], str | None] | None = None,
    ) -> None:
        self.config = config
        self.api_client = api_client
        self.process_spawner = process_spawner
        self.stop_event = stop_event
        self.environment_id: str | None = None
        self.environment_secret: str | None = None
        self._initial_session_id: str | None = None
        self._sessions: dict[str, ActiveSession] = {}
        self._session_bridges: dict[str, RemoteSessionBridgeManager] = {}
        self._completed_work_ids: set[str] = set()
        self._token_supplier = token_supplier
        self._token_manager = (
            TokenSessionManager(
                get_access_token=self._get_refresh_token_for_session,
                on_refresh=self._refresh_session_token,
                label="bridge",
            )
            if token_supplier is not None
            else None
        )

    @staticmethod
    def _detect_sleep_gap(
        previous_error_at: float | None,
        now: float,
        retry_cap_sec: float,
    ) -> bool:
        if previous_error_at is None:
            return False
        return (now - previous_error_at) > (retry_cap_sec * 2.0)

    def _refresh_session_token(self, session_id: str, token: str) -> None:
        active = self._sessions.get(session_id)
        if active is None or not active.process.is_running():
            return
        active.process.update_access_token(token)
        logger.info("[bridge] Token manager refreshed token for active session %s", session_id)

    @staticmethod
    def _extract_session_ingress_token(payload: dict[str, Any]) -> str | None:
        for key in (
            "session_ingress_token",
            "sessionIngressToken",
            "access_token",
            "accessToken",
        ):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _get_refresh_token_for_session(self, session_id: str) -> str | None:
        # Prefer a session-scoped token from control plane if exposed.
        try:
            session_payload = self.api_client.get_session(session_id)
            session_token = self._extract_session_ingress_token(session_payload)
            if session_token:
                return session_token
        except Exception as exc:  # noqa: BLE001
            logger.debug("[bridge:token] get_session refresh failed for %s: %s", session_id, exc)

        # Fallback to externally supplied token source.
        if self._token_supplier is not None:
            return self._token_supplier()
        return None

    def run(self) -> None:
        registered = self.api_client.register_environment(self.config)
        self.environment_id = registered.environment_id
        self.environment_secret = registered.environment_secret

        connect_url = registered.connect_url or build_connect_url(
            self.config.base_api_url,
            self.config.bridge_id,
        )

        try:
            self._initial_session_id = self.api_client.create_initial_session(
                environment_id=registered.environment_id,
                title="Remote Control session",
                git_repo_url=self.config.git_repo_url,
                branch=self.config.branch,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("[bridge] createInitialSession failed: %s", exc)

        console.print("\n[bold]Remote Control bridge started[/bold]")
        console.print(f"  Workspace: {self.config.directory}")
        console.print(f"  Environment: {registered.environment_id}")
        console.print(f"  Connect URL: {connect_url}")
        console.print("  Press Ctrl+C to stop.\n")

        connection_delay = CONNECTION_RETRY_INITIAL_SEC
        general_delay = GENERAL_RETRY_INITIAL_SEC
        connection_started_at: float | None = None
        general_started_at: float | None = None
        connection_last_error_at: float | None = None
        general_last_error_at: float | None = None
        disconnected_since: float | None = None

        while not self.stop_event.is_set():
            self._reap_sessions()
            env_id = self.environment_id
            env_secret = self.environment_secret
            if not env_id or not env_secret:
                break

            try:
                work = self.api_client.poll_for_work(env_id, env_secret)

                if disconnected_since is not None:
                    reconnect_elapsed = time.monotonic() - disconnected_since
                    logger.info("[bridge] Reconnected after %.1fs", reconnect_elapsed)

                connection_delay = CONNECTION_RETRY_INITIAL_SEC
                general_delay = GENERAL_RETRY_INITIAL_SEC
                connection_started_at = None
                general_started_at = None
                connection_last_error_at = None
                general_last_error_at = None
                disconnected_since = None

                if not work:
                    self._sleep_with_stop(1.0)
                    continue
                if self.api_client.needs_explicit_ack:
                    work_id = str(work.get("id") or "").strip()
                    if work_id and IDENTIFIER_RE.match(work_id):
                        try:
                            self.api_client.acknowledge_work(env_id, work_id, env_secret)
                        except BridgeFatalError:
                            raise
                        except Exception as exc:  # noqa: BLE001
                            logger.debug(
                                "[bridge] acknowledgeWork failed for work_id=%s: %s",
                                work_id,
                                exc,
                            )
                self._handle_work_item(work)
            except BridgeFatalError as exc:
                console.print(f"[red]Remote Control fatal error:[/red] {exc}")
                logger.error("[bridge] Fatal error status=%s type=%s", exc.status, exc.error_type)
                break
            except (ConnectionError, TimeoutError, OSError) as exc:
                now = time.monotonic()
                if self._detect_sleep_gap(connection_last_error_at, now, CONNECTION_RETRY_MAX_SEC):
                    logger.info("[bridge] Detected resume from sleep, resetting connection budget")
                    connection_started_at = None
                    connection_delay = CONNECTION_RETRY_INITIAL_SEC

                if disconnected_since is None:
                    disconnected_since = now
                if connection_started_at is None:
                    connection_started_at = now
                connection_last_error_at = now

                elapsed = now - connection_started_at
                if elapsed >= CONNECTION_GIVEUP_SEC:
                    console.print(
                        "[red]Remote Control disconnected for too long; exiting.[/red]"
                    )
                    logger.error("[bridge] Connection retry budget exhausted: %s", exc)
                    break

                logger.warning("[bridge] Connection error (elapsed %.1fs): %s", elapsed, exc)
                sleep_seconds = jitter_delay(connection_delay)
                self._sleep_with_stop(sleep_seconds)
                connection_delay = min(connection_delay * 2.0, CONNECTION_RETRY_MAX_SEC)
                general_started_at = None
                general_delay = GENERAL_RETRY_INITIAL_SEC
                general_last_error_at = None
            except Exception as exc:  # noqa: BLE001
                now = time.monotonic()
                if self._detect_sleep_gap(general_last_error_at, now, GENERAL_RETRY_MAX_SEC):
                    logger.info("[bridge] Detected resume from sleep, resetting general error budget")
                    general_started_at = None
                    general_delay = GENERAL_RETRY_INITIAL_SEC

                if disconnected_since is None:
                    disconnected_since = now
                if general_started_at is None:
                    general_started_at = now
                general_last_error_at = now

                elapsed = now - general_started_at
                if elapsed >= GENERAL_GIVEUP_SEC:
                    console.print("[red]Remote Control persistent errors; exiting.[/red]")
                    logger.error("[bridge] General retry budget exhausted: %s", exc, exc_info=True)
                    break

                logger.warning("[bridge] Poll error (elapsed %.1fs): %s", elapsed, exc)
                sleep_seconds = jitter_delay(general_delay)
                self._sleep_with_stop(sleep_seconds)
                general_delay = min(general_delay * 2.0, GENERAL_RETRY_MAX_SEC)
                connection_started_at = None
                connection_delay = CONNECTION_RETRY_INITIAL_SEC
                connection_last_error_at = None

    def shutdown(self) -> None:
        env_id = self.environment_id
        self.stop_event.set()

        for active in list(self._sessions.values()):
            active.process.terminate(force=False)

        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            self._reap_sessions()
            if not self._sessions:
                break
            time.sleep(0.1)

        for active in list(self._sessions.values()):
            if active.process.is_running():
                active.process.terminate(force=True)
            if env_id:
                self._stop_work_with_retry(env_id, active.work_id, force=True)
            try:
                self.api_client.archive_session(active.session_id)
            except Exception as exc:  # noqa: BLE001
                logger.debug("[bridge] archiveSession failed: %s", exc)
            self._sessions.pop(active.session_id, None)
            bridge = self._session_bridges.pop(active.session_id, None)
            if bridge is not None:
                bridge.disconnect()
            if self._token_manager is not None:
                self._token_manager.cancel(active.session_id)

        if self._initial_session_id:
            try:
                self.api_client.archive_session(self._initial_session_id)
            except Exception as exc:  # noqa: BLE001
                logger.debug("[bridge] archive initial session failed: %s", exc)
            self._initial_session_id = None

        if env_id:
            try:
                self.api_client.deregister_environment(env_id)
            except Exception as exc:  # noqa: BLE001
                logger.debug("[bridge] Failed to deregister environment: %s", exc)

        if self._token_manager is not None:
            self._token_manager.cancel_all()

        for bridge in list(self._session_bridges.values()):
            bridge.disconnect()
        self._session_bridges.clear()

    def _on_child_activity(self, session_id: str, activity: BridgeActivity) -> None:
        logger.debug(
            "[bridge:activity] session=%s type=%s summary=%s",
            session_id,
            activity.type,
            activity.summary,
        )

    def _handle_child_control_message(self, session_id: str, message: dict[str, Any]) -> None:
        active = self._sessions.get(session_id)
        if active is None:
            return

        message_type = str(message.get("type") or "").strip()
        if message_type == "control_response":
            bridge = self._session_bridges.get(session_id)
            if bridge is not None:
                bridge.send_control_response(message)
            return

        if message_type != "control_request":
            return

        request_raw = message.get("request")
        request = request_raw if isinstance(request_raw, dict) else {}
        subtype = str(request.get("subtype") or "").strip().lower()
        if subtype == "interrupt":
            active.process.terminate(force=False)
            return

        worker = threading.Thread(
            target=self._forward_child_control_request,
            args=(session_id, message),
            daemon=True,
            name=f"bridge-control-forward-{session_id}",
        )
        worker.start()

    def _forward_child_control_request(self, session_id: str, message: dict[str, Any]) -> None:
        active = self._sessions.get(session_id)
        if active is None:
            return

        request_id = str(message.get("request_id") or "").strip()
        request_raw = message.get("request")
        request = request_raw if isinstance(request_raw, dict) else {}
        bridge = self._session_bridges.get(session_id)

        if bridge is None:
            logger.debug(
                "[bridge:control] No session bridge available for control request session=%s",
                session_id,
            )
            if request_id:
                error_response = {
                    "type": "control_response",
                    "response": {
                        "subtype": "error",
                        "request_id": request_id,
                        "error": "Remote session bridge unavailable",
                    },
                }
                active.process.write_stdin(json.dumps(error_response, ensure_ascii=False) + "\n")
            return

        try:
            response = bridge.forward_control_request(request, timeout_sec=60.0)
        except Exception as exc:  # noqa: BLE001
            logger.debug("[bridge:control] Forward control request failed: %s", exc)
            response = None

        if request_id:
            if not response:
                payload = {
                    "type": "control_response",
                    "response": {
                        "subtype": "error",
                        "request_id": request_id,
                        "error": "Timed out waiting for remote control response",
                    },
                }
            else:
                payload_response = {
                    "subtype": str(response.get("subtype") or "success"),
                    "request_id": request_id,
                }
                if "response" in response:
                    payload_response["response"] = response.get("response")
                if "error" in response:
                    payload_response["error"] = response.get("error")
                payload = {"type": "control_response", "response": payload_response}
            active.process.write_stdin(json.dumps(payload, ensure_ascii=False) + "\n")

    def _extract_org_uuid(self, session_payload: dict[str, Any]) -> str | None:
        for key in ("organization_uuid", "organizationUuid", "org_uuid", "orgUuid"):
            value = session_payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        org_raw = session_payload.get("organization")
        if isinstance(org_raw, dict):
            for key in ("uuid", "id"):
                value = org_raw.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return None

    def _attach_session_bridge(self, session_id: str) -> None:
        if self._token_supplier is None:
            return
        access_token = self._token_supplier()
        if not access_token:
            logger.debug("[bridge:repl] No access token available; skip session bridge")
            return

        org_uuid = os.getenv("RIPPERDOC_ORGANIZATION_UUID", "").strip() or None
        if org_uuid is None:
            try:
                session_payload = self.api_client.get_session(session_id)
                org_uuid = self._extract_org_uuid(session_payload)
            except Exception as exc:  # noqa: BLE001
                logger.debug("[bridge:repl] get_session failed for %s: %s", session_id, exc)

        if not org_uuid:
            logger.debug("[bridge:repl] Missing organization uuid for session %s", session_id)
            return

        manager = RemoteSessionBridgeManager(
            RemoteSessionConfig(
                session_id=session_id,
                access_token=access_token,
                org_uuid=org_uuid,
                base_api_url=self.config.base_api_url,
            ),
            RemoteSessionCallbacks(
                on_permission_request=lambda request, request_id: self._handle_repl_permission_request(
                    session_id,
                    request,
                    request_id,
                )
            ),
            on_control_response_fallback=lambda payload: self._send_control_response_via_events(
                session_id,
                payload,
            ),
        )
        manager.connect()
        self._session_bridges[session_id] = manager

    def _handle_repl_permission_request(
        self,
        session_id: str,
        request: dict[str, Any],
        request_id: str,
    ) -> None:
        active = self._sessions.get(session_id)
        if active is None:
            return
        payload = {
            "type": "control_request",
            "request_id": request_id,
            "request": request,
        }
        active.process.write_stdin(json.dumps(payload, ensure_ascii=False) + "\n")

    def _send_control_response_via_events(
        self,
        session_id: str,
        payload: dict[str, Any],
    ) -> None:
        token = self._token_supplier() if self._token_supplier is not None else None
        if not token:
            logger.debug(
                "[bridge:control] Cannot fallback control_response via events: missing access token session=%s",
                session_id,
            )
            return
        try:
            self.api_client.send_permission_response_event(session_id, payload, token)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "[bridge:control] control_response events fallback failed session=%s: %s",
                session_id,
                exc,
            )

    def _handle_work_item(self, work: dict[str, object]) -> None:
        work_id = str(work.get("id") or "").strip()
        if work_id and work_id in self._completed_work_ids:
            logger.debug("[bridge] Ignoring already completed work item %s", work_id)
            return

        raw_data = work.get("data")
        work_data: dict[str, object] = raw_data if isinstance(raw_data, dict) else {}
        work_type = str(work_data.get("type") or "").strip().lower()

        if work_type == "healthcheck":
            logger.debug("[bridge] Received healthcheck work item")
            return
        if work_type != "session":
            logger.debug("[bridge] Ignoring unsupported work type: %s", work_type or "<empty>")
            return

        raw_session_id = str(work_data.get("id") or "").strip()
        if not raw_session_id or not IDENTIFIER_RE.match(raw_session_id):
            logger.warning("[bridge] Invalid session id in work item: %r", raw_session_id)
            return
        session_id = raw_session_id

        if work_id and not IDENTIFIER_RE.match(work_id):
            logger.warning("[bridge] Invalid work id in work item: %r", work_id)
            return

        secret_encoded = str(work.get("secret") or "").strip()
        if not secret_encoded:
            logger.warning("[bridge] Work item missing secret payload")
            return
        try:
            secret = decode_work_secret(secret_encoded)
        except ValueError as exc:
            logger.warning("[bridge] Invalid work secret for session %s: %s", session_id, exc)
            return

        existing = self._sessions.get(session_id)
        if existing and existing.process.is_running():
            existing.process.update_access_token(secret.session_ingress_token)
            if self._token_manager is not None:
                self._token_manager.schedule(session_id, secret.session_ingress_token)
            logger.info("[bridge] Refreshed token for active session %s", session_id)
            return

        running_session_ids = [sid for sid, s in self._sessions.items() if s.process.is_running()]
        if running_session_ids and session_id not in running_session_ids:
            logger.warning(
                "[bridge] Rejecting foreign session %s while active sessions exist: %s",
                session_id,
                ",".join(running_session_ids),
            )
            return

        ingress_host = secret.api_base_url or self.config.session_ingress_url
        sdk_url = build_session_ingress_ws_url(ingress_host, session_id)
        process = self.process_spawner.spawn(
            session_id=session_id,
            sdk_url=sdk_url,
            access_token=secret.session_ingress_token,
            cwd=self.config.directory,
            on_activity=self._on_child_activity,
            on_control_message=self._handle_child_control_message,
        )
        self._sessions[session_id] = ActiveSession(
            session_id=session_id,
            work_id=work_id,
            process=process,
            started_at_monotonic=time.monotonic(),
        )
        self._attach_session_bridge(session_id)
        if self._token_manager is not None:
            self._token_manager.schedule(session_id, secret.session_ingress_token)
        logger.info("[bridge] Started session %s (work_id=%s)", session_id, work_id or "unknown")

    def _reap_sessions(self) -> None:
        env_id = self.environment_id
        for session_id, active in list(self._sessions.items()):
            if (
                self.config.session_timeout_sec > 0
                and active.process.is_running()
                and (time.monotonic() - active.started_at_monotonic) >= self.config.session_timeout_sec
            ):
                logger.warning("[bridge] Session %s timed out; terminating", session_id)
                active.process.terminate(force=False)

            return_code = active.process.poll()
            if return_code is None:
                continue

            if return_code == 0:
                status_label = "completed"
            elif return_code < 0:
                status_label = "interrupted"
            else:
                status_label = "failed"

            logger.info(
                "[bridge] Session %s exited (%s, code=%s)",
                session_id,
                status_label,
                return_code,
            )
            if status_label == "failed" and active.process.stderr_tail:
                logger.warning(
                    "[bridge] Session %s stderr tail:\n%s",
                    session_id,
                    "\n".join(active.process.stderr_tail),
                )
            if env_id and active.work_id:
                self._stop_work_with_retry(env_id, active.work_id, force=False)
                self._completed_work_ids.add(active.work_id)
            if self._token_manager is not None:
                self._token_manager.cancel(session_id)

            bridge = self._session_bridges.pop(session_id, None)
            if bridge is not None:
                bridge.disconnect()

            try:
                self.api_client.archive_session(session_id)
            except Exception as exc:  # noqa: BLE001
                logger.debug("[bridge] archiveSession failed: %s", exc)
            self._sessions.pop(session_id, None)
            if status_label != "interrupted":
                logger.info(
                    "[bridge] Session %s ended with status %s; continuing work poll loop",
                    session_id,
                    status_label,
                )

    def _sleep_with_stop(self, seconds: float) -> None:
        end_time = time.monotonic() + max(seconds, 0.0)
        while not self.stop_event.is_set() and time.monotonic() < end_time:
            time.sleep(0.2)

    def _stop_work_with_retry(self, environment_id: str, work_id: str, *, force: bool) -> None:
        if not work_id:
            return
        for attempt in range(1, 4):
            try:
                self.api_client.stop_work(environment_id, work_id, force=force)
                return
            except BridgeFatalError:
                raise
            except Exception as exc:  # noqa: BLE001
                if attempt == 3:
                    logger.debug(
                        "[bridge] stopWork(force=%s) failed after %d attempts: %s",
                        force,
                        attempt,
                        exc,
                    )
                    return
                retry_delay = jitter_delay(1.0 * (2 ** (attempt - 1)))
                logger.debug(
                    "[bridge] stopWork(force=%s) failed (attempt %d/3): %s",
                    force,
                    attempt,
                    exc,
                )
                self._sleep_with_stop(retry_delay)
