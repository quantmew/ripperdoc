"""Control-plane API client for remote-control bridge."""

from __future__ import annotations

import json
import os
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from ripperdoc.utils.coerce import parse_boolish
from ripperdoc.utils.log import get_logger

from .constants import (
    LEGACY_HEADERS_ENV,
    POLL_ACK_IN_QUERY_ENV,
    REMOTE_CONTROL_API_BETA,
    REMOTE_CONTROL_API_VERSION,
    REMOTE_CONTROL_BETA_HEADER,
    REMOTE_CONTROL_VERSION_HEADER,
)
from .errors import BridgeFatalError
from .models import RegisteredEnvironment, RemoteControlConfig
from .utils import validate_identifier

logger = get_logger()


class RemoteControlApiClient:
    """Thin JSON client for remote-control control-plane API calls."""

    def __init__(
        self,
        *,
        base_url: str,
        access_token: str | None,
        runner_version: str,
        refresh_access_token: Callable[[str], str | None] | None = None,
        poll_ack_in_query: bool | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.access_token = access_token.strip() if isinstance(access_token, str) else None
        self.runner_version = runner_version
        self._refresh_access_token = refresh_access_token
        if poll_ack_in_query is None:
            self.poll_ack_in_query = parse_boolish(
                os.getenv(POLL_ACK_IN_QUERY_ENV),
                default=True,
            )
        else:
            self.poll_ack_in_query = bool(poll_ack_in_query)

    def _headers(self, bearer_token: str | None = None) -> dict[str, str]:
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            REMOTE_CONTROL_VERSION_HEADER: REMOTE_CONTROL_API_VERSION,
            REMOTE_CONTROL_BETA_HEADER: REMOTE_CONTROL_API_BETA,
            "x-environment-runner-version": self.runner_version,
        }
        if parse_boolish(os.getenv(LEGACY_HEADERS_ENV), default=False):
            headers["anthropic-version"] = REMOTE_CONTROL_API_VERSION
            headers["anthropic-beta"] = REMOTE_CONTROL_API_BETA
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

    def _request_json_with_refresh(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        query: dict[str, Any] | None = None,
        timeout_sec: float = 15.0,
        bearer_token: str | None = None,
        operation_name: str,
    ) -> tuple[int, Any]:
        status, data = self._request_json(
            method,
            path,
            payload=payload,
            query=query,
            timeout_sec=timeout_sec,
            bearer_token=bearer_token,
        )
        if status != 401 or self._refresh_access_token is None:
            return status, data

        current = bearer_token or self.access_token or ""
        refreshed = self._refresh_access_token(current)
        if not refreshed:
            logger.debug("[bridge:api] %s: token refresh unavailable after 401", operation_name)
            return status, data
        if bearer_token is None:
            self.access_token = refreshed

        retry_status, retry_data = self._request_json(
            method,
            path,
            payload=payload,
            query=query,
            timeout_sec=timeout_sec,
            bearer_token=refreshed,
        )
        return retry_status, retry_data

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
        status, data = self._request_json_with_refresh(
            "POST",
            "/v1/environments/bridge",
            payload={
                "machine_name": config.machine_name,
                "directory": str(config.directory),
                "branch": config.branch,
                "git_repo_url": config.git_repo_url,
            },
            timeout_sec=15.0,
            operation_name="Registration",
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
        safe_env = quote(validate_identifier(environment_id, "environmentId"), safe="")
        status, data = self._request_json(
            "GET",
            f"/v1/environments/{safe_env}/work/poll",
            query={
                "block_ms": 900,
                "ack": "true" if self.poll_ack_in_query else "false",
            },
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

    @property
    def needs_explicit_ack(self) -> bool:
        return not self.poll_ack_in_query

    def acknowledge_work(
        self,
        environment_id: str,
        work_id: str,
        environment_secret: str | None = None,
    ) -> None:
        safe_env = quote(validate_identifier(environment_id, "environmentId"), safe="")
        safe_work = quote(validate_identifier(work_id, "workId"), safe="")
        status, data = self._request_json_with_refresh(
            "POST",
            f"/v1/environments/{safe_env}/work/{safe_work}/ack",
            payload={},
            timeout_sec=10.0,
            bearer_token=environment_secret,
            operation_name="AcknowledgeWork",
        )
        self._raise_for_status(status, data, "AcknowledgeWork")

    def stop_work(
        self,
        environment_id: str,
        work_id: str,
        *,
        force: bool,
    ) -> None:
        safe_env = quote(validate_identifier(environment_id, "environmentId"), safe="")
        safe_work = quote(validate_identifier(work_id, "workId"), safe="")
        status, data = self._request_json_with_refresh(
            "POST",
            f"/v1/environments/{safe_env}/work/{safe_work}/stop",
            payload={"force": force},
            timeout_sec=10.0,
            operation_name="StopWork",
        )
        self._raise_for_status(status, data, "StopWork")

    def deregister_environment(self, environment_id: str) -> None:
        safe_env = quote(validate_identifier(environment_id, "environmentId"), safe="")
        status, data = self._request_json_with_refresh(
            "DELETE",
            f"/v1/environments/bridge/{safe_env}",
            timeout_sec=10.0,
            operation_name="Deregister",
        )
        if status in {404, 405}:
            status, data = self._request_json_with_refresh(
                "POST",
                f"/v1/environments/{safe_env}/deregister",
                payload={},
                timeout_sec=10.0,
                operation_name="DeregisterFallback",
            )
        if status == 404:
            return
        self._raise_for_status(status, data, "Deregister")

    def archive_session(self, session_id: str) -> None:
        safe_session = quote(validate_identifier(session_id, "sessionId"), safe="")
        status, data = self._request_json_with_refresh(
            "POST",
            f"/v1/sessions/{safe_session}/archive",
            payload={},
            timeout_sec=10.0,
            operation_name="ArchiveSession",
        )
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
        status, data = self._request_json_with_refresh(
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
            operation_name="CreateSession",
        )
        if status not in {200, 201}:
            return None
        if not isinstance(data, dict):
            return None
        session_id = data.get("id")
        if not isinstance(session_id, str) or not session_id.strip():
            return None
        return session_id.strip()

    def get_session(self, session_id: str) -> dict[str, Any]:
        safe_session = quote(validate_identifier(session_id, "sessionId"), safe="")
        status, data = self._request_json_with_refresh(
            "GET",
            f"/v1/sessions/{safe_session}",
            timeout_sec=10.0,
            operation_name="GetSession",
        )
        self._raise_for_status(status, data, "GetSession")
        return data if isinstance(data, dict) else {}

    def send_permission_response_event(
        self,
        session_id: str,
        event: dict[str, Any],
        access_token: str,
    ) -> None:
        safe_session = quote(validate_identifier(session_id, "sessionId"), safe="")
        status, data = self._request_json(
            "POST",
            f"/v1/sessions/{safe_session}/events",
            payload={"events": [event]},
            timeout_sec=10.0,
            bearer_token=access_token.strip(),
        )
        self._raise_for_status(status, data, "SendPermissionResponseEvent")
