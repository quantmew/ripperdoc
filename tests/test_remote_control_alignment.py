"""Tests for remote-control bridge architecture and behavior."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from ripperdoc.cli.remote_control import loop as loop_module
from ripperdoc.cli.remote_control.loop import RemoteControlBridgeRunner
from ripperdoc.cli.remote_control.models import ActiveSession, RemoteControlConfig
from ripperdoc.cli.remote_control.process import ChildBridgeSession, RemoteControlProcessSpawner


class _FakeStdin:
    def __init__(self) -> None:
        self.closed = False
        self.writes: list[str] = []

    def write(self, payload: str) -> None:
        self.writes.append(payload)

    def flush(self) -> None:
        return None


class _FakeProcess:
    def __init__(
        self,
        *,
        stdout_lines: list[str] | None = None,
        stderr_lines: list[str] | None = None,
        return_code: int | None = None,
    ) -> None:
        self.pid = 123
        self.stdin = _FakeStdin()
        self.stdout = [line if line.endswith("\n") else f"{line}\n" for line in (stdout_lines or [])]
        self.stderr = [line if line.endswith("\n") else f"{line}\n" for line in (stderr_lines or [])]
        self._return_code = return_code

    def poll(self) -> int | None:
        return self._return_code

    def terminate(self) -> None:
        self._return_code = -15

    def kill(self) -> None:
        self._return_code = -9

    def wait(self, timeout: float | None = None) -> int:
        return self._return_code if self._return_code is not None else 0


class _ApiStub:
    def archive_session(self, _session_id: str) -> None:
        return None

    def stop_work(self, _environment_id: str, _work_id: str, *, force: bool) -> None:  # noqa: ARG002
        return None

    def get_session(self, _session_id: str) -> dict[str, str]:
        return {"organization_uuid": "org-test"}

    @property
    def needs_explicit_ack(self) -> bool:
        return False


class _BridgeStub:
    def __init__(self) -> None:
        self.forwarded: list[dict[str, object]] = []
        self.responses: list[dict[str, object]] = []

    def forward_control_request(self, request: dict[str, object], *, timeout_sec: float = 60.0) -> dict[str, object] | None:  # noqa: ARG002
        self.forwarded.append(request)
        return {
            "subtype": "success",
            "response": {
                "behavior": "allow",
            },
        }

    def send_control_response(self, payload: dict[str, object]) -> None:
        self.responses.append(payload)

    def disconnect(self) -> None:
        return None


class _RunnerProcess:
    def __init__(self) -> None:
        self.stdin_writes: list[str] = []

    def poll(self) -> int | None:
        return None

    def is_running(self) -> bool:
        return True

    def write_stdin(self, payload: str) -> None:
        self.stdin_writes.append(payload)

    def update_access_token(self, token: str) -> None:  # noqa: ARG002
        return None

    def terminate(self, *, force: bool = False) -> None:  # noqa: ARG002
        return None

    @property
    def stderr_tail(self) -> list[str]:
        return []


def _config(tmp_path: Path) -> RemoteControlConfig:
    return RemoteControlConfig(
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


def test_child_process_activity_transcript_and_control_passthrough(tmp_path: Path) -> None:
    transcript = tmp_path / "bridge-transcript-session-1.jsonl"
    assistant_line = json.dumps(
        {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Bash",
                        "input": {"command": "echo hi"},
                    }
                ]
            },
        }
    )
    control_request_line = json.dumps(
        {
            "type": "control_request",
            "request_id": "req-1",
            "request": {"subtype": "can_use_tool", "tool_name": "Bash"},
        }
    )
    control_response_line = json.dumps(
        {
            "type": "control_response",
            "response": {"subtype": "success", "request_id": "req-1"},
        }
    )

    process = _FakeProcess(
        stdout_lines=[assistant_line, control_request_line, control_response_line],
        return_code=0,
    )

    seen_activities: list[str] = []
    seen_control_types: list[str] = []
    _ = ChildBridgeSession(
        process=process,  # type: ignore[arg-type]
        session_id="session-1",
        access_token="token",
        verbose=False,
        transcript_path=transcript,
        on_activity=lambda _sid, activity: seen_activities.append(activity.type),
        on_control_message=lambda _sid, msg: seen_control_types.append(str(msg.get("type"))),
    )

    time.sleep(0.1)

    assert transcript.exists()
    content = transcript.read_text(encoding="utf-8")
    assert "control_request" in content
    assert "control_response" in content
    assert "tool_use" in content
    assert "tool_start" in seen_activities
    assert seen_control_types.count("control_request") == 1
    assert seen_control_types.count("control_response") == 1


def test_loop_forwards_child_control_request_and_writes_response(tmp_path: Path) -> None:
    runner = RemoteControlBridgeRunner(
        config=_config(tmp_path),
        api_client=_ApiStub(),  # type: ignore[arg-type]
        process_spawner=RemoteControlProcessSpawner(verbose=False, debug_file=None),
        stop_event=threading.Event(),
    )

    process = _RunnerProcess()
    runner._sessions["session-1"] = ActiveSession(
        session_id="session-1",
        work_id="work-1",
        process=process,
        started_at_monotonic=time.monotonic(),
    )
    bridge = _BridgeStub()
    runner._session_bridges["session-1"] = bridge  # type: ignore[assignment]

    runner._handle_child_control_message(
        "session-1",
        {
            "type": "control_request",
            "request_id": "local-req",
            "request": {"subtype": "can_use_tool", "tool_name": "Bash"},
        },
    )

    time.sleep(0.1)

    assert bridge.forwarded
    assert process.stdin_writes
    payload = json.loads(process.stdin_writes[-1])
    assert payload["type"] == "control_response"
    assert payload["response"]["request_id"] == "local-req"
    assert payload["response"]["subtype"] == "success"


def test_loop_forwards_child_control_response_to_session_bridge(tmp_path: Path) -> None:
    runner = RemoteControlBridgeRunner(
        config=_config(tmp_path),
        api_client=_ApiStub(),  # type: ignore[arg-type]
        process_spawner=RemoteControlProcessSpawner(verbose=False, debug_file=None),
        stop_event=threading.Event(),
    )

    process = _RunnerProcess()
    runner._sessions["session-1"] = ActiveSession(
        session_id="session-1",
        work_id="work-1",
        process=process,
        started_at_monotonic=time.monotonic(),
    )
    bridge = _BridgeStub()
    runner._session_bridges["session-1"] = bridge  # type: ignore[assignment]

    message = {
        "type": "control_response",
        "response": {"subtype": "success", "request_id": "req-1"},
    }
    runner._handle_child_control_message("session-1", message)

    assert bridge.responses == [message]


def test_runner_attach_session_bridge_uses_repl_bridge_manager(monkeypatch, tmp_path: Path) -> None:
    created: dict[str, object] = {}
    fallback_calls: list[tuple[str, dict[str, object], str]] = []

    class _ApiWithFallback(_ApiStub):
        def send_permission_response_event(
            self,
            session_id: str,
            event: dict[str, object],
            access_token: str,
        ) -> None:
            fallback_calls.append((session_id, event, access_token))

    class _BridgeManagerFake:
        def __init__(self, config, callbacks, *, on_control_response_fallback=None):
            created["config"] = config
            created["callbacks"] = callbacks
            created["fallback"] = on_control_response_fallback
            self.connected = False

        def connect(self):
            self.connected = True
            created["connected"] = True

        def disconnect(self):
            return None

    monkeypatch.setattr(loop_module, "RemoteSessionBridgeManager", _BridgeManagerFake)

    runner = RemoteControlBridgeRunner(
        config=_config(tmp_path),
        api_client=_ApiWithFallback(),  # type: ignore[arg-type]
        process_spawner=RemoteControlProcessSpawner(verbose=False, debug_file=None),
        stop_event=threading.Event(),
        token_supplier=lambda: "oauth-token",
    )

    runner._attach_session_bridge("session-abc")

    assert created.get("connected") is True
    config = created["config"]
    assert getattr(config, "session_id") == "session-abc"
    assert getattr(config, "org_uuid") == "org-test"
    assert "session-abc" in runner._session_bridges
    fallback = created.get("fallback")
    assert callable(fallback)
    fallback({"type": "control_response", "response": {"subtype": "success"}})
    assert fallback_calls == [
        (
            "session-abc",
            {"type": "control_response", "response": {"subtype": "success"}},
            "oauth-token",
        )
    ]


def test_detect_sleep_gap_matches_retry_budget_model(tmp_path: Path) -> None:
    runner = RemoteControlBridgeRunner(
        config=_config(tmp_path),
        api_client=_ApiStub(),  # type: ignore[arg-type]
        process_spawner=RemoteControlProcessSpawner(verbose=False, debug_file=None),
        stop_event=threading.Event(),
    )

    assert runner._detect_sleep_gap(None, 20.0, 5.0) is False
    assert runner._detect_sleep_gap(10.0, 19.0, 5.0) is False
    assert runner._detect_sleep_gap(10.0, 21.5, 5.0) is True


def test_get_refresh_token_for_session_prefers_session_payload_token(tmp_path: Path) -> None:
    class _ApiWithSessionToken(_ApiStub):
        def get_session(self, _session_id: str) -> dict[str, str]:
            return {"session_ingress_token": "session-token-1"}

    runner = RemoteControlBridgeRunner(
        config=_config(tmp_path),
        api_client=_ApiWithSessionToken(),  # type: ignore[arg-type]
        process_spawner=RemoteControlProcessSpawner(verbose=False, debug_file=None),
        stop_event=threading.Event(),
        token_supplier=lambda: "fallback-token",
    )

    token = runner._get_refresh_token_for_session("session-1")
    assert token == "session-token-1"
