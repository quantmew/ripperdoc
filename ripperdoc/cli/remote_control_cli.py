"""Remote Control bridge command.

This command mirrors Claude Code's `remote-control` bridge shape:
- Register a bridge environment with a control-plane API.
- Poll for work assignments.
- Spawn a local `ripperdoc --print --sdk-url ...` child session per remote work item.
- Forward refreshed session ingress tokens to the child process.
"""

from __future__ import annotations

import base64
import json
import os
import random
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen

import click
from rich.console import Console

from ripperdoc import __version__
from ripperdoc.utils.coerce import parse_boolish
from ripperdoc.utils.filesystem.git_utils import get_current_git_branch
from ripperdoc.utils.log import get_logger

console = Console()
logger = get_logger()

_BASE_URL_ENV = "RIPPERDOC_REMOTE_CONTROL_BASE_URL"
_INGRESS_URL_ENV = "RIPPERDOC_REMOTE_CONTROL_INGRESS_URL"
_TOKEN_ENV = "RIPPERDOC_REMOTE_CONTROL_ACCESS_TOKEN"
_ALLOW_HTTP_ENV = "RIPPERDOC_REMOTE_CONTROL_ALLOW_INSECURE_HTTP"
_CONNECT_TEMPLATE_ENV = "RIPPERDOC_REMOTE_CONTROL_CONNECT_URL_TEMPLATE"
_DEFAULT_SESSION_TIMEOUT_SEC = 86_400

_CONNECTION_RETRY_INITIAL_SEC = 2.0
_CONNECTION_RETRY_MAX_SEC = 120.0
_CONNECTION_GIVEUP_SEC = 600.0

_GENERAL_RETRY_INITIAL_SEC = 0.5
_GENERAL_RETRY_MAX_SEC = 30.0
_GENERAL_GIVEUP_SEC = 600.0

_IDENTIFIER_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
_ANTHROPIC_VERSION = "2023-06-01"
_ANTHROPIC_BETA = "environments-2025-11-01"


class BridgeFatalError(RuntimeError):
    """Fatal control-plane error that should terminate the bridge loop."""

    def __init__(self, message: str, *, status: int, error_type: str | None = None):
        super().__init__(message)
        self.status = status
        self.error_type = error_type


def _validate_identifier(value: str, name: str) -> str:
    """Validate identifier shape against bridge-safe character rules."""
    candidate = (value or "").strip()
    if not candidate or not _IDENTIFIER_RE.match(candidate):
        raise ValueError(f"Invalid {name}: contains unsafe characters")
    return candidate


def _jitter_delay(base_delay_sec: float) -> float:
    """Apply +/-25% jitter to retry delay to reduce synchronized retries."""
    bounded = max(0.0, float(base_delay_sec))
    return max(0.0, bounded + bounded * 0.25 * (2.0 * random.random() - 1.0))


@dataclass(frozen=True)
class WorkSecret:
    """Decoded work secret payload."""

    version: int
    session_ingress_token: str
    api_base_url: str | None


@dataclass(frozen=True)
class RegisteredEnvironment:
    """Control-plane environment registration result."""

    environment_id: str
    environment_secret: str
    connect_url: str | None


@dataclass(frozen=True)
class RemoteControlConfig:
    """Bridge runtime configuration."""

    directory: Path
    machine_name: str
    branch: str | None
    git_repo_url: str | None
    bridge_id: str
    environment_id: str
    base_api_url: str
    session_ingress_url: str
    session_timeout_sec: int
    verbose: bool
    debug_file: Path | None


class RemoteControlApiClient:
    """Thin JSON client for remote-control control-plane API calls."""

    def __init__(
        self,
        *,
        base_url: str,
        access_token: str | None,
        runner_version: str,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.access_token = access_token.strip() if isinstance(access_token, str) else None
        self.runner_version = runner_version

    def _headers(self, bearer_token: str | None = None) -> dict[str, str]:
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "anthropic-version": _ANTHROPIC_VERSION,
            "anthropic-beta": _ANTHROPIC_BETA,
            "x-environment-runner-version": self.runner_version,
        }
        token = bearer_token or self.access_token
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        query: dict[str, Any] | None = None,
        timeout_sec: float = 15.0,
        bearer_token: str | None = None,
    ) -> tuple[int, Any]:
        url = f"{self.base_url}{path}"
        if query:
            encoded = urlencode(query, doseq=True)
            url = f"{url}?{encoded}"
        body = None
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
        request = Request(
            url,
            data=body,
            method=method,
            headers=self._headers(bearer_token),
        )

        try:
            with urlopen(request, timeout=timeout_sec) as response:
                status = int(getattr(response, "status", 200))
                raw = response.read()
        except HTTPError as exc:
            status = int(exc.code)
            raw = exc.read()
        except URLError as exc:
            raise ConnectionError(str(exc.reason)) from exc

        if not raw:
            return status, None
        try:
            return status, json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return status, raw.decode("utf-8", errors="replace")

    @staticmethod
    def _extract_error_message(data: Any) -> str | None:
        if isinstance(data, dict):
            message = data.get("message")
            if isinstance(message, str) and message.strip():
                return message.strip()
            error = data.get("error")
            if isinstance(error, dict):
                nested_message = error.get("message")
                if isinstance(nested_message, str) and nested_message.strip():
                    return nested_message.strip()
        return None

    @staticmethod
    def _extract_error_type(data: Any) -> str | None:
        if isinstance(data, dict):
            error = data.get("error")
            if isinstance(error, dict):
                error_type = error.get("type")
                if isinstance(error_type, str) and error_type.strip():
                    return error_type.strip()
        return None

    @staticmethod
    def _is_expired(error_type: str | None) -> bool:
        if not error_type:
            return False
        lowered = error_type.lower()
        return "expired" in lowered or "lifetime" in lowered

    def _raise_for_status(self, status: int, data: Any, event_name: str) -> None:
        if status < 400:
            return

        message = self._extract_error_message(data)
        error_type = self._extract_error_type(data)

        if status == 401:
            raise BridgeFatalError(
                f"{event_name}: authentication failed (401).",
                status=status,
                error_type=error_type,
            )
        if status == 403:
            if self._is_expired(error_type):
                raise BridgeFatalError(
                    "Remote Control session has expired. Please restart the bridge command.",
                    status=status,
                    error_type=error_type,
                )
            detail = f": {message}" if message else ""
            raise BridgeFatalError(
                f"{event_name}: access denied (403){detail}",
                status=status,
                error_type=error_type,
            )
        if status == 404:
            detail = message or (
                f"{event_name}: not found (404). "
                "Ensure the Remote Control service supports /v1/environments endpoints."
            )
            raise BridgeFatalError(detail, status=status, error_type=error_type)
        if status == 410:
            detail = message or "Remote Control session has expired."
            raise BridgeFatalError(detail, status=status, error_type=error_type)
        if status == 429:
            raise RuntimeError(f"{event_name}: rate limited (429).")
        detail = f": {message}" if message else ""
        raise RuntimeError(f"{event_name}: failed with status {status}{detail}")

    def register_environment(self, config: RemoteControlConfig) -> RegisteredEnvironment:
        status, data = self._request_json(
            "POST",
            "/v1/environments/bridge",
            payload={
                "machine_name": config.machine_name,
                "directory": str(config.directory),
                "branch": config.branch,
                "git_repo_url": config.git_repo_url,
            },
            timeout_sec=15.0,
        )
        self._raise_for_status(status, data, "Registration")

        payload = data if isinstance(data, dict) else {}
        environment_id = str(
            payload.get("environment_id") or payload.get("environmentId") or ""
        ).strip()
        environment_secret = str(
            payload.get("environment_secret") or payload.get("environmentSecret") or ""
        ).strip()
        connect_url_raw = payload.get("connect_url") or payload.get("connectUrl")
        connect_url = str(connect_url_raw).strip() if connect_url_raw else None

        if not environment_id or not environment_secret:
            raise RuntimeError("Registration response missing environment_id/environment_secret.")
        return RegisteredEnvironment(
            environment_id=environment_id,
            environment_secret=environment_secret,
            connect_url=connect_url,
        )

    def poll_for_work(self, environment_id: str, environment_secret: str) -> dict[str, Any] | None:
        safe_env = quote(_validate_identifier(environment_id, "environmentId"), safe="")
        status, data = self._request_json(
            "GET",
            f"/v1/environments/{safe_env}/work/poll",
            query={"block_ms": 900, "ack": "true"},
            timeout_sec=10.0,
            bearer_token=environment_secret,
        )
        if status == 204:
            return None
        self._raise_for_status(status, data, "Poll")
        if not data:
            return None
        if isinstance(data, dict):
            return data
        return None

    def stop_work(
        self,
        environment_id: str,
        work_id: str,
        *,
        force: bool,
    ) -> None:
        safe_env = quote(_validate_identifier(environment_id, "environmentId"), safe="")
        safe_work = quote(_validate_identifier(work_id, "workId"), safe="")
        status, data = self._request_json(
            "POST",
            f"/v1/environments/{safe_env}/work/{safe_work}/stop",
            payload={"force": force},
            timeout_sec=10.0,
        )
        self._raise_for_status(status, data, "StopWork")

    def deregister_environment(self, environment_id: str) -> None:
        safe_env = quote(_validate_identifier(environment_id, "environmentId"), safe="")
        status, data = self._request_json(
            "DELETE",
            f"/v1/environments/bridge/{safe_env}",
            timeout_sec=10.0,
        )
        if status in {404, 405}:
            # Compatibility fallback for servers exposing a POST-based deregister route.
            status, data = self._request_json(
                "POST",
                f"/v1/environments/{safe_env}/deregister",
                payload={},
                timeout_sec=10.0,
            )
        # Some services return 404 when already removed; treat as non-fatal.
        if status == 404:
            return
        self._raise_for_status(status, data, "Deregister")

    def archive_session(self, session_id: str) -> None:
        safe_session = quote(_validate_identifier(session_id, "sessionId"), safe="")
        status, data = self._request_json(
            "POST",
            f"/v1/sessions/{safe_session}/archive",
            payload={},
            timeout_sec=10.0,
        )
        # Not all services expose session archival; keep this best-effort.
        if status in {404, 409}:
            return
        self._raise_for_status(status, data, "ArchiveSession")

    def create_initial_session(
        self,
        *,
        environment_id: str,
        title: str,
        git_repo_url: str | None,
        branch: str | None,
    ) -> str | None:
        status, data = self._request_json(
            "POST",
            "/v1/sessions",
            payload={
                "title": title,
                "events": [],
                "environment_id": environment_id,
                "source": "remote-control",
                "git_repo_url": git_repo_url,
                "branch": branch,
            },
            timeout_sec=10.0,
        )
        if status not in {200, 201}:
            return None
        if not isinstance(data, dict):
            return None
        session_id = data.get("id")
        if not isinstance(session_id, str) or not session_id.strip():
            return None
        return session_id.strip()


class ChildBridgeSession:
    """Wrapper around one spawned local ripperdoc subprocess."""

    def __init__(
        self,
        *,
        process: subprocess.Popen[str],
        session_id: str,
        access_token: str,
        verbose: bool,
    ) -> None:
        self._process = process
        self.session_id = session_id
        self.access_token = access_token
        self._verbose = verbose
        self._stderr_lines: deque[str] = deque(maxlen=10)
        self._stdout_thread = threading.Thread(
            target=self._read_stdout,
            name=f"ripperdoc-bridge-stdout-{session_id}",
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._read_stderr,
            name=f"ripperdoc-bridge-stderr-{session_id}",
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

    def _read_stdout(self) -> None:
        stdout = self._process.stdout
        if stdout is None:
            return
        for line in stdout:
            text = line.rstrip("\r\n")
            if not text:
                continue
            logger.debug("[bridge:child:%s] stdout %s", self.session_id, text)
            if self._verbose:
                console.print(f"[dim]{text}[/dim]")

    def _read_stderr(self) -> None:
        stderr = self._process.stderr
        if stderr is None:
            return
        for line in stderr:
            text = line.rstrip("\r\n")
            if not text:
                continue
            self._stderr_lines.append(text)
            logger.debug("[bridge:child:%s] stderr %s", self.session_id, text)

    def poll(self) -> int | None:
        return self._process.poll()

    def is_running(self) -> bool:
        return self.poll() is None

    def write_stdin(self, payload: str) -> None:
        stdin = self._process.stdin
        if stdin is None or stdin.closed:
            return
        try:
            stdin.write(payload)
            stdin.flush()
        except (BrokenPipeError, OSError):
            return

    def update_access_token(self, token: str) -> None:
        self.access_token = token
        update_msg = {
            "type": "update_environment_variables",
            "variables": {
                "RIPPERDOC_SESSION_ACCESS_TOKEN": token,
                "RIPPERDOC_REMOTE_CONTROL_ACCESS_TOKEN": token,
            },
        }
        self.write_stdin(json.dumps(update_msg, ensure_ascii=False) + "\n")

    def terminate(self, *, force: bool = False) -> None:
        if self.poll() is not None:
            return
        try:
            if force:
                self._process.kill()
            else:
                self._process.terminate()
        except OSError:
            return

    def wait(self, timeout: float | None = None) -> int | None:
        try:
            return self._process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None

    @property
    def stderr_tail(self) -> list[str]:
        return list(self._stderr_lines)


class RemoteControlProcessSpawner:
    """Spawn local ripperdoc SDK-transport subprocesses."""

    def __init__(self, *, verbose: bool, debug_file: Path | None) -> None:
        self.verbose = verbose
        self.debug_file = debug_file

    @staticmethod
    def _session_debug_path(base_path: Path, session_id: str) -> Path:
        suffix = session_id.replace("/", "_")
        if base_path.suffix:
            return base_path.with_name(f"{base_path.stem}-{suffix}{base_path.suffix}")
        return base_path.with_name(f"{base_path.name}-{suffix}")

    def spawn(
        self,
        *,
        session_id: str,
        sdk_url: str,
        access_token: str,
        cwd: Path,
    ) -> ChildBridgeSession:
        args = [
            sys.executable,
            "-m",
            "ripperdoc.cli.cli",
            "--sdk-url",
            sdk_url,
            "--session-id",
            session_id,
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--no-session-persistence",
        ]
        if self.verbose:
            args.append("--verbose")
        if self.debug_file is not None:
            session_debug = self._session_debug_path(self.debug_file, session_id)
            args.extend(["--debug-file", str(session_debug)])

        env = {key: value for key, value in os.environ.items() if not key.startswith("CLAUDE_CODE_")}
        env.update(
            {
                "RIPPERDOC_REMOTE": "1",
                "RIPPERDOC_REMOTE_CONTROL": "1",
                "RIPPERDOC_ENVIRONMENT_KIND": "bridge",
                "RIPPERDOC_SESSION_ACCESS_TOKEN": access_token,
                "RIPPERDOC_REMOTE_CONTROL_ACCESS_TOKEN": access_token,
                "RIPPERDOC_POST_FOR_SESSION_INGRESS_V2": "1",
                "RIPPERDOC_SESSION_ID": session_id,
            }
        )

        process = subprocess.Popen(
            args,
            cwd=str(cwd),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        logger.info("[bridge] Spawned local session process pid=%s session=%s", process.pid, session_id)
        return ChildBridgeSession(
            process=process,
            session_id=session_id,
            access_token=access_token,
            verbose=self.verbose,
        )


@dataclass
class _ActiveSession:
    session_id: str
    work_id: str
    process: ChildBridgeSession
    started_at_monotonic: float


class RemoteControlBridgeRunner:
    """Main remote-control poll/dispatch loop."""

    def __init__(
        self,
        *,
        config: RemoteControlConfig,
        api_client: RemoteControlApiClient,
        process_spawner: RemoteControlProcessSpawner,
        stop_event: threading.Event,
    ) -> None:
        self.config = config
        self.api_client = api_client
        self.process_spawner = process_spawner
        self.stop_event = stop_event
        self.environment_id: str | None = None
        self.environment_secret: str | None = None
        self._initial_session_id: str | None = None
        self._sessions: dict[str, _ActiveSession] = {}

    def run(self) -> None:
        registered = self.api_client.register_environment(self.config)
        self.environment_id = registered.environment_id
        self.environment_secret = registered.environment_secret

        connect_url = registered.connect_url or _build_connect_url(
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

        connection_delay = _CONNECTION_RETRY_INITIAL_SEC
        general_delay = _GENERAL_RETRY_INITIAL_SEC
        connection_started_at: float | None = None
        general_started_at: float | None = None

        while not self.stop_event.is_set():
            self._reap_sessions()
            env_id = self.environment_id
            env_secret = self.environment_secret
            if not env_id or not env_secret:
                break

            try:
                work = self.api_client.poll_for_work(env_id, env_secret)
                connection_delay = _CONNECTION_RETRY_INITIAL_SEC
                general_delay = _GENERAL_RETRY_INITIAL_SEC
                connection_started_at = None
                general_started_at = None

                if not work:
                    self._sleep_with_stop(1.0)
                    continue
                self._handle_work_item(work)
            except BridgeFatalError as exc:
                console.print(f"[red]Remote Control fatal error:[/red] {exc}")
                logger.error("[bridge] Fatal error status=%s type=%s", exc.status, exc.error_type)
                break
            except (ConnectionError, TimeoutError, URLError, OSError) as exc:
                now = time.monotonic()
                if connection_started_at is None:
                    connection_started_at = now
                elapsed = now - connection_started_at
                if elapsed >= _CONNECTION_GIVEUP_SEC:
                    console.print(
                        "[red]Remote Control disconnected for too long; exiting.[/red]"
                    )
                    logger.error("[bridge] Connection retry budget exhausted: %s", exc)
                    break
                logger.warning(
                    "[bridge] Connection error (elapsed %.1fs): %s",
                    elapsed,
                    exc,
                )
                self._sleep_with_stop(_jitter_delay(connection_delay))
                connection_delay = min(connection_delay * 2.0, _CONNECTION_RETRY_MAX_SEC)
            except Exception as exc:  # noqa: BLE001
                now = time.monotonic()
                if general_started_at is None:
                    general_started_at = now
                elapsed = now - general_started_at
                if elapsed >= _GENERAL_GIVEUP_SEC:
                    console.print("[red]Remote Control persistent errors; exiting.[/red]")
                    logger.error("[bridge] General retry budget exhausted: %s", exc, exc_info=True)
                    break
                logger.warning(
                    "[bridge] Poll error (elapsed %.1fs): %s",
                    elapsed,
                    exc,
                )
                self._sleep_with_stop(_jitter_delay(general_delay))
                general_delay = min(general_delay * 2.0, _GENERAL_RETRY_MAX_SEC)

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

    def _handle_work_item(self, work: dict[str, Any]) -> None:
        work_id = str(work.get("id") or "").strip()
        raw_data = work.get("data")
        work_data: dict[str, Any] = raw_data if isinstance(raw_data, dict) else {}
        work_type = str(work_data.get("type") or "").strip().lower()

        if work_type == "healthcheck":
            logger.debug("[bridge] Received healthcheck work item")
            return
        if work_type != "session":
            logger.debug("[bridge] Ignoring unsupported work type: %s", work_type or "<empty>")
            return

        raw_session_id = str(work_data.get("id") or "").strip()
        if not raw_session_id or not _IDENTIFIER_RE.match(raw_session_id):
            logger.warning("[bridge] Invalid session id in work item: %r", raw_session_id)
            return
        session_id = raw_session_id

        if work_id and not _IDENTIFIER_RE.match(work_id):
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
        )
        self._sessions[session_id] = _ActiveSession(
            session_id=session_id,
            work_id=work_id,
            process=process,
            started_at_monotonic=time.monotonic(),
        )
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
            try:
                self.api_client.archive_session(session_id)
            except Exception as exc:  # noqa: BLE001
                logger.debug("[bridge] archiveSession failed: %s", exc)
            self._sessions.pop(session_id, None)
            if status_label != "interrupted":
                logger.info(
                    "[bridge] Session %s ended with status %s; stopping bridge loop",
                    session_id,
                    status_label,
                )
                self.stop_event.set()

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
                retry_delay = _jitter_delay(1.0 * (2 ** (attempt - 1)))
                logger.debug(
                    "[bridge] stopWork(force=%s) failed (attempt %d/3): %s",
                    force,
                    attempt,
                    exc,
                )
                self._sleep_with_stop(retry_delay)


def decode_work_secret(encoded_secret: str) -> WorkSecret:
    """Decode and validate a base64url work secret payload."""
    if not encoded_secret:
        raise ValueError("empty work secret")
    raw = _base64url_decode(encoded_secret).decode("utf-8")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("work secret payload must be a JSON object")

    version = payload.get("version")
    if version != 1:
        raise ValueError(f"unsupported work secret version: {version!r}")

    session_ingress_token = payload.get("session_ingress_token")
    if not isinstance(session_ingress_token, str) or not session_ingress_token.strip():
        raise ValueError("missing or empty session_ingress_token")

    api_base_url = payload.get("api_base_url")
    if api_base_url is not None and not isinstance(api_base_url, str):
        raise ValueError("api_base_url must be a string when provided")

    return WorkSecret(
        version=1,
        session_ingress_token=session_ingress_token.strip(),
        api_base_url=api_base_url.strip() if isinstance(api_base_url, str) and api_base_url.strip() else None,
    )


def build_session_ingress_ws_url(base_url: str, session_id: str) -> str:
    """Build session ingress websocket URL following Claude bridge URL shape."""
    normalized = base_url.strip()
    parsed = urlparse(normalized)
    hostname = (parsed.hostname or "").strip().lower()
    localhost = hostname in {"localhost", "127.0.0.1"} or hostname.endswith(".localhost")
    ws_scheme = "ws" if localhost else "wss"
    api_version = "v2" if localhost else "v1"
    if parsed.netloc:
        domain = parsed.netloc
        path_prefix = parsed.path.strip("/")
        if path_prefix:
            domain = f"{domain}/{path_prefix}"
    else:
        domain = normalized.replace("http://", "").replace("https://", "").strip("/")
    return f"{ws_scheme}://{domain}/{api_version}/session_ingress/ws/{session_id}"


def _base64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def _resolve_base_url(explicit_base_url: str | None) -> str:
    configured = explicit_base_url or os.getenv(_BASE_URL_ENV) or os.getenv("RIPPERDOC_BASE_URL")
    if not configured or not configured.strip():
        raise click.ClickException(
            "Missing Remote Control base URL. Set "
            f"`{_BASE_URL_ENV}` or pass `--base-url`."
        )

    value = configured.strip()
    if not value.startswith(("http://", "https://")):
        value = f"https://{value}"

    parsed = urlparse(value)
    hostname = (parsed.hostname or "").strip().lower()
    is_localhost = hostname in {"localhost", "127.0.0.1"} or hostname.endswith(".localhost")
    allow_insecure = parse_boolish(os.getenv(_ALLOW_HTTP_ENV), default=False)
    if parsed.scheme == "http" and not is_localhost and not allow_insecure:
        raise click.ClickException(
            "Remote Control base URL uses HTTP. Use HTTPS, localhost HTTP, or set "
            f"{_ALLOW_HTTP_ENV}=1."
        )
    if parsed.scheme not in {"http", "https"}:
        raise click.ClickException(f"Unsupported base URL scheme: {parsed.scheme!r}")
    return value.rstrip("/")


def _resolve_ingress_url(base_url: str) -> str:
    ingress = os.getenv(_INGRESS_URL_ENV, "").strip()
    if ingress:
        return ingress.rstrip("/")
    return base_url


def _resolve_auth_token(explicit_token: str | None) -> str | None:
    token = (
        explicit_token
        or os.getenv(_TOKEN_ENV)
        or os.getenv("RIPPERDOC_AUTH_TOKEN")
        or os.getenv("RIPPERDOC_API_KEY")
    )
    if isinstance(token, str):
        token = token.strip()
    return token or None


def _build_connect_url(base_url: str, bridge_id: str) -> str:
    template = os.getenv(_CONNECT_TEMPLATE_ENV, "").strip()
    if template:
        try:
            return template.format(base_url=base_url.rstrip("/"), bridge_id=bridge_id)
        except Exception:
            logger.debug("[bridge] Invalid connect URL template: %s", template)
    lowered = base_url.lower()
    if "_staging_" in lowered or "staging" in lowered:
        app_origin = "https://claude-ai.staging.ant.dev"
    else:
        app_origin = "https://claude.ai"
    return f"{app_origin}/code?bridge={bridge_id}"


def _get_git_remote_url(cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def run_remote_control(
    *,
    verbose: bool,
    base_url: str | None,
    auth_token: str | None,
    session_timeout_sec: int,
    debug_file: Path | None,
) -> None:
    """Entry point for remote control bridge runtime."""
    resolved_base_url = _resolve_base_url(base_url)
    resolved_ingress_url = _resolve_ingress_url(resolved_base_url)
    resolved_auth_token = _resolve_auth_token(auth_token)
    if not resolved_auth_token:
        logger.warning(
            "[bridge] No auth token configured. Set %s if your service requires bearer auth.",
            _TOKEN_ENV,
        )

    cwd = Path.cwd().resolve()
    config = RemoteControlConfig(
        directory=cwd,
        machine_name=socket.gethostname(),
        branch=get_current_git_branch(cwd),
        git_repo_url=_get_git_remote_url(cwd),
        bridge_id=str(uuid.uuid4()),
        environment_id=str(uuid.uuid4()),
        base_api_url=resolved_base_url,
        session_ingress_url=resolved_ingress_url,
        session_timeout_sec=max(0, int(session_timeout_sec)),
        verbose=verbose,
        debug_file=debug_file,
    )
    api_client = RemoteControlApiClient(
        base_url=resolved_base_url,
        access_token=resolved_auth_token,
        runner_version=__version__,
    )
    spawner = RemoteControlProcessSpawner(
        verbose=verbose,
        debug_file=debug_file,
    )
    stop_event = threading.Event()
    runner = RemoteControlBridgeRunner(
        config=config,
        api_client=api_client,
        process_spawner=spawner,
        stop_event=stop_event,
    )

    previous_handlers: list[tuple[int, Any]] = []

    def _handle_signal(signum: int, _frame: Any) -> None:
        logger.info("[bridge] Signal %s received; shutting down", signum)
        stop_event.set()

    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        previous_handlers.append((sig, signal.getsignal(sig)))
        signal.signal(sig, _handle_signal)

    try:
        runner.run()
    finally:
        runner.shutdown()
        for sig, previous in previous_handlers:
            signal.signal(sig, previous)


@click.command(name="remote-control")
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Enable verbose bridge logging.",
)
@click.option(
    "--base-url",
    type=str,
    default=None,
    help=f"Remote control API base URL (default: ${_BASE_URL_ENV}).",
)
@click.option(
    "--auth-token",
    type=str,
    default=None,
    help=f"Bearer token for the remote control API (default: ${_TOKEN_ENV}).",
)
@click.option(
    "--session-timeout",
    type=int,
    default=_DEFAULT_SESSION_TIMEOUT_SEC,
    show_default=True,
    help="Maximum child session lifetime in seconds. Use 0 to disable timeout.",
)
@click.option(
    "--debug-file",
    type=click.Path(path_type=Path, dir_okay=False, writable=True),
    default=None,
    help="Write child session debug logs to this file (session-id suffix is appended).",
)
def remote_control_cmd(
    verbose: bool,
    base_url: str | None,
    auth_token: str | None,
    session_timeout: int,
    debug_file: Path | None,
) -> None:
    """Run a Remote Control bridge process for this working directory."""
    run_remote_control(
        verbose=verbose,
        base_url=base_url,
        auth_token=auth_token,
        session_timeout_sec=session_timeout,
        debug_file=debug_file,
    )


__all__ = [
    "remote_control_cmd",
    "run_remote_control",
    "decode_work_secret",
    "build_session_ingress_ws_url",
    "BridgeFatalError",
]
