"""Tests for remote-control bridge helpers."""

from __future__ import annotations

import base64
import json
import threading
import time

import click
import pytest

from ripperdoc.cli import remote_control_cli


def _encode_secret(payload: dict[str, object]) -> str:
    raw = json.dumps(payload).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")


def test_decode_work_secret_success() -> None:
    encoded = _encode_secret(
        {
            "version": 1,
            "session_ingress_token": "token-123",
            "api_base_url": "https://bridge.example.com",
        }
    )

    secret = remote_control_cli.decode_work_secret(encoded)

    assert secret.version == 1
    assert secret.session_ingress_token == "token-123"
    assert secret.api_base_url == "https://bridge.example.com"


def test_decode_work_secret_requires_session_ingress_token() -> None:
    encoded = _encode_secret({"version": 1})

    with pytest.raises(ValueError, match="session_ingress_token"):
        remote_control_cli.decode_work_secret(encoded)


def test_build_session_ingress_ws_url_uses_local_ws_v2() -> None:
    url = remote_control_cli.build_session_ingress_ws_url("http://localhost:3000", "session-abc")
    assert url == "ws://localhost:3000/v2/session_ingress/ws/session-abc"


def test_build_session_ingress_ws_url_uses_wss_v1_for_non_localhost() -> None:
    url = remote_control_cli.build_session_ingress_ws_url(
        "https://remote.example.com/",
        "session-abc",
    )
    assert url == "wss://remote.example.com/v1/session_ingress/ws/session-abc"


def test_build_session_ingress_ws_url_preserves_base_path_prefix() -> None:
    url = remote_control_cli.build_session_ingress_ws_url(
        "https://remote.example.com/custom/prefix",
        "session-abc",
    )
    assert url == "wss://remote.example.com/custom/prefix/v1/session_ingress/ws/session-abc"


def test_resolve_base_url_requires_configuration(monkeypatch) -> None:
    monkeypatch.delenv("RIPPERDOC_REMOTE_CONTROL_BASE_URL", raising=False)
    monkeypatch.delenv("RIPPERDOC_BASE_URL", raising=False)

    with pytest.raises(click.ClickException, match="Missing Remote Control base URL"):
        remote_control_cli._resolve_base_url(None)


def test_resolve_base_url_rejects_insecure_http_for_non_localhost(monkeypatch) -> None:
    monkeypatch.setenv("RIPPERDOC_REMOTE_CONTROL_BASE_URL", "http://example.com")
    monkeypatch.delenv("RIPPERDOC_REMOTE_CONTROL_ALLOW_INSECURE_HTTP", raising=False)

    with pytest.raises(click.ClickException, match="uses HTTP"):
        remote_control_cli._resolve_base_url(None)


def test_resolve_base_url_allows_insecure_http_when_explicitly_enabled(monkeypatch) -> None:
    monkeypatch.setenv("RIPPERDOC_REMOTE_CONTROL_BASE_URL", "http://example.com")
    monkeypatch.setenv("RIPPERDOC_REMOTE_CONTROL_ALLOW_INSECURE_HTTP", "1")

    assert remote_control_cli._resolve_base_url(None) == "http://example.com"


def test_poll_for_work_uses_environment_secret_bearer_token(monkeypatch) -> None:
    client = remote_control_cli.RemoteControlApiClient(
        base_url="https://example.com",
        access_token="user-token",
        runner_version="test",
    )
    captured: dict[str, object] = {}

    def fake_request_json(method: str, path: str, **kwargs: object):
        captured["method"] = method
        captured["path"] = path
        captured["kwargs"] = kwargs
        return 204, None

    monkeypatch.setattr(client, "_request_json", fake_request_json)

    result = client.poll_for_work("env-1", "env-secret")

    assert result is None
    assert captured["method"] == "GET"
    assert captured["path"] == "/v1/environments/env-1/work/poll"
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs.get("bearer_token") == "env-secret"


def test_stop_work_uses_client_access_token(monkeypatch) -> None:
    client = remote_control_cli.RemoteControlApiClient(
        base_url="https://example.com",
        access_token="user-token",
        runner_version="test",
    )
    captured: dict[str, object] = {}

    def fake_request_json(method: str, path: str, **kwargs: object):
        captured["method"] = method
        captured["path"] = path
        captured["kwargs"] = kwargs
        return 200, {}

    monkeypatch.setattr(client, "_request_json", fake_request_json)

    client.stop_work("env-1", "work-1", force=False)

    assert captured["method"] == "POST"
    assert captured["path"] == "/v1/environments/env-1/work/work-1/stop"
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs.get("bearer_token") is None


def test_deregister_environment_falls_back_to_post_when_delete_not_supported(
    monkeypatch,
) -> None:
    client = remote_control_cli.RemoteControlApiClient(
        base_url="https://example.com",
        access_token="user-token",
        runner_version="test",
    )
    calls: list[tuple[str, str]] = []

    def fake_request_json(method: str, path: str, **kwargs: object):
        calls.append((method, path))
        if method == "DELETE":
            return 405, {}
        return 200, {}

    monkeypatch.setattr(client, "_request_json", fake_request_json)

    client.deregister_environment("env-1")

    assert calls == [
        ("DELETE", "/v1/environments/bridge/env-1"),
        ("POST", "/v1/environments/env-1/deregister"),
    ]


def test_archive_session_ignores_not_found(monkeypatch) -> None:
    client = remote_control_cli.RemoteControlApiClient(
        base_url="https://example.com",
        access_token="user-token",
        runner_version="test",
    )

    def fake_request_json(method: str, path: str, **kwargs: object):
        assert method == "POST"
        assert path == "/v1/sessions/session-1/archive"
        return 404, {}

    monkeypatch.setattr(client, "_request_json", fake_request_json)

    client.archive_session("session-1")


def test_create_initial_session_returns_id_on_success(monkeypatch) -> None:
    client = remote_control_cli.RemoteControlApiClient(
        base_url="https://example.com",
        access_token="user-token",
        runner_version="test",
    )

    def fake_request_json(method: str, path: str, **kwargs: object):
        assert method == "POST"
        assert path == "/v1/sessions"
        return 201, {"id": "session-123"}

    monkeypatch.setattr(client, "_request_json", fake_request_json)

    session_id = client.create_initial_session(
        environment_id="env-1",
        title="Remote Control session",
        git_repo_url=None,
        branch=None,
    )
    assert session_id == "session-123"


def test_create_initial_session_returns_none_on_non_2xx(monkeypatch) -> None:
    client = remote_control_cli.RemoteControlApiClient(
        base_url="https://example.com",
        access_token="user-token",
        runner_version="test",
    )

    def fake_request_json(method: str, path: str, **kwargs: object):
        return 404, {}

    monkeypatch.setattr(client, "_request_json", fake_request_json)

    session_id = client.create_initial_session(
        environment_id="env-1",
        title="Remote Control session",
        git_repo_url=None,
        branch=None,
    )
    assert session_id is None


def test_api_headers_include_bridge_metadata() -> None:
    client = remote_control_cli.RemoteControlApiClient(
        base_url="https://example.com",
        access_token="token-1",
        runner_version="test-version",
    )
    headers = client._headers()

    assert headers["Authorization"] == "Bearer token-1"
    assert headers["x-ripperdoc-version"] == "2023-06-01"
    assert headers["x-ripperdoc-beta"] == "environments-2025-11-01"
    assert "anthropic-version" not in headers
    assert "anthropic-beta" not in headers
    assert headers["x-environment-runner-version"] == "test-version"


def test_api_headers_support_legacy_compat_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RIPPERDOC_REMOTE_CONTROL_LEGACY_HEADERS", "1")
    client = remote_control_cli.RemoteControlApiClient(
        base_url="https://example.com",
        access_token="token-1",
        runner_version="test-version",
    )
    headers = client._headers()

    assert headers["x-ripperdoc-version"] == "2023-06-01"
    assert headers["x-ripperdoc-beta"] == "environments-2025-11-01"
    assert headers["anthropic-version"] == "2023-06-01"
    assert headers["anthropic-beta"] == "environments-2025-11-01"


def test_poll_for_work_rejects_unsafe_environment_id() -> None:
    client = remote_control_cli.RemoteControlApiClient(
        base_url="https://example.com",
        access_token="token-1",
        runner_version="test",
    )
    with pytest.raises(ValueError, match="environmentId"):
        client.poll_for_work("env/unsafe", "secret")


def test_build_connect_url_defaults_to_base_origin() -> None:
    url = remote_control_cli._build_connect_url("https://api.example.com", "bridge-1")
    assert url == "https://example.com/code?bridge=bridge-1"


def test_build_connect_url_strips_common_api_prefixes() -> None:
    url = remote_control_cli._build_connect_url("https://api-staging.example.com", "bridge-1")
    assert url == "https://staging.example.com/code?bridge=bridge-1"


def test_process_spawner_sets_bridge_compat_env(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    class _FakeStdin:
        closed = False

        def write(self, _payload: str) -> None:
            return None

        def flush(self) -> None:
            return None

    class _FakeProcess:
        def __init__(self) -> None:
            self.pid = 12345
            self.stdin = _FakeStdin()
            self.stdout: list[str] = []
            self.stderr: list[str] = []

        def poll(self) -> int | None:
            return None

        def terminate(self) -> None:
            return None

        def kill(self) -> None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            return 0

    def fake_popen(args, **kwargs):  # type: ignore[no-untyped-def]
        captured["args"] = list(args)
        captured["env"] = dict(kwargs.get("env") or {})
        return _FakeProcess()

    monkeypatch.setattr(remote_control_cli.subprocess, "Popen", fake_popen)

    spawner = remote_control_cli.RemoteControlProcessSpawner(verbose=False, debug_file=None)
    session = spawner.spawn(
        session_id="session-1",
        sdk_url="wss://example.test/v1/session_ingress/ws/session-1",
        access_token="token-xyz",
        cwd=tmp_path,
    )

    assert session.session_id == "session-1"
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["RIPPERDOC_ENVIRONMENT_KIND"] == "bridge"
    assert env["RIPPERDOC_SESSION_ID"] == "session-1"
    assert env["RIPPERDOC_SESSION_ACCESS_TOKEN"] == "token-xyz"
    assert "RIPPERDOC_AUTH_TOKEN" not in env
    assert all(not key.startswith("CLAUDE_CODE_") for key in env)


def test_reap_sessions_stops_bridge_on_non_interrupted_exit(monkeypatch, tmp_path) -> None:
    class _FakeApi:
        def archive_session(self, _session_id: str) -> None:
            return None

    class _FakeProcess:
        def is_running(self) -> bool:
            return False

        def poll(self) -> int | None:
            return 0

        @property
        def stderr_tail(self) -> list[str]:
            return []

    config = remote_control_cli.RemoteControlConfig(
        directory=tmp_path,
        machine_name="machine",
        branch=None,
        git_repo_url=None,
        bridge_id="bridge-1",
        environment_id="env-1",
        base_api_url="https://example.com",
        session_ingress_url="https://example.com",
        session_timeout_sec=0,
        verbose=False,
        debug_file=None,
    )
    stop_event = threading.Event()
    runner = remote_control_cli.RemoteControlBridgeRunner(
        config=config,
        api_client=_FakeApi(),  # type: ignore[arg-type]
        process_spawner=remote_control_cli.RemoteControlProcessSpawner(
            verbose=False,
            debug_file=None,
        ),
        stop_event=stop_event,
    )
    runner.environment_id = "env-1"
    runner._sessions["session-1"] = remote_control_cli._ActiveSession(
        session_id="session-1",
        work_id="work-1",
        process=_FakeProcess(),  # type: ignore[arg-type]
        started_at_monotonic=time.monotonic(),
    )

    monkeypatch.setattr(runner, "_stop_work_with_retry", lambda *args, **kwargs: None)

    runner._reap_sessions()

    assert stop_event.is_set()
