"""Subprocess spawning and child-session wrappers for bridge runtime."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from rich.console import Console

from ripperdoc.utils.coerce import parse_boolish
from ripperdoc.utils.log import get_logger

console = Console()
logger = get_logger()


@dataclass(frozen=True)
class BridgeActivity:
    """Summarized child activity parsed from stream-json output."""

    type: str
    summary: str
    timestamp_ms: int


def _tool_summary(tool_name: str, tool_input: dict[str, Any]) -> str:
    input_summary = ""
    for key in ("file_path", "filePath", "pattern", "url", "query"):
        value = tool_input.get(key)
        if isinstance(value, str) and value.strip():
            input_summary = value.strip()[:80]
            break
    if input_summary:
        return f"{tool_name} {input_summary}"
    return tool_name


def _extract_activities(payload: dict[str, Any], timestamp_ms: int) -> list[BridgeActivity]:
    activities: list[BridgeActivity] = []
    msg_type = str(payload.get("type") or "").strip()

    if msg_type == "assistant":
        message = payload.get("message")
        if not isinstance(message, dict):
            return activities
        content = message.get("content")
        if not isinstance(content, list):
            return activities
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "").strip()
            if item_type == "tool_use":
                tool_name = str(item.get("name") or "Tool")
                tool_input = item.get("input")
                summary = _tool_summary(tool_name, tool_input if isinstance(tool_input, dict) else {})
                activities.append(
                    BridgeActivity(type="tool_start", summary=summary, timestamp_ms=timestamp_ms)
                )
            elif item_type == "text":
                text = str(item.get("text") or "").strip()
                if text:
                    activities.append(
                        BridgeActivity(type="text", summary=text[:80], timestamp_ms=timestamp_ms)
                    )
        return activities

    if msg_type == "result":
        subtype = str(payload.get("subtype") or "").strip()
        if subtype == "success":
            activities.append(
                BridgeActivity(
                    type="result",
                    summary="Session completed",
                    timestamp_ms=timestamp_ms,
                )
            )
        elif subtype:
            errors = payload.get("errors")
            error_summary = (
                str(errors[0]).strip()
                if isinstance(errors, list) and errors and isinstance(errors[0], str)
                else f"Error: {subtype}"
            )
            activities.append(
                BridgeActivity(type="error", summary=error_summary, timestamp_ms=timestamp_ms)
            )

    return activities


class ChildBridgeSession:
    """Wrapper around one spawned local ripperdoc subprocess."""

    def __init__(
        self,
        *,
        process: subprocess.Popen[str],
        session_id: str,
        access_token: str,
        verbose: bool,
        transcript_path: Path | None = None,
        on_activity: Callable[[str, BridgeActivity], None] | None = None,
        on_control_message: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self._process = process
        self.session_id = session_id
        self.access_token = access_token
        self._verbose = verbose
        self._stderr_lines: deque[str] = deque(maxlen=10)
        self._activities: deque[BridgeActivity] = deque(maxlen=10)
        self._current_activity: BridgeActivity | None = None
        self._on_activity = on_activity
        self._on_control_message = on_control_message
        self._transcript_path = transcript_path
        self._transcript_lock = threading.Lock()
        self._transcript_handle = None

        if transcript_path is not None:
            try:
                transcript_path.parent.mkdir(parents=True, exist_ok=True)
                self._transcript_handle = transcript_path.open("a", encoding="utf-8")
                logger.debug("[bridge:session] Transcript log: %s", transcript_path)
            except OSError as exc:
                logger.warning("[bridge:session] Failed to open transcript log %s: %s", transcript_path, exc)

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

    def _append_transcript_line(self, line: str) -> None:
        handle = self._transcript_handle
        if handle is None:
            return
        try:
            with self._transcript_lock:
                handle.write(line + "\n")
                handle.flush()
        except OSError as exc:
            logger.debug("[bridge:session] Transcript write error: %s", exc)

    def _close_transcript(self) -> None:
        handle = self._transcript_handle
        if handle is None:
            return
        self._transcript_handle = None
        try:
            with self._transcript_lock:
                handle.close()
        except OSError:
            return

    def _read_stdout(self) -> None:
        stdout = self._process.stdout
        if stdout is None:
            self._close_transcript()
            return

        try:
            for line in stdout:
                text = line.rstrip("\r\n")
                if not text:
                    continue
                self._append_transcript_line(text)
                logger.debug("[bridge:child:%s] stdout %s", self.session_id, text)
                if self._verbose:
                    console.print(f"[dim]{text}[/dim]")

                payload: dict[str, Any] | None = None
                try:
                    raw = json.loads(text)
                    payload = raw if isinstance(raw, dict) else None
                except (json.JSONDecodeError, UnicodeDecodeError):
                    payload = None

                if payload is None:
                    continue

                now_ms = int(time.time() * 1000)
                for activity in _extract_activities(payload, now_ms):
                    self._activities.append(activity)
                    self._current_activity = activity
                    if self._on_activity is not None:
                        try:
                            self._on_activity(self.session_id, activity)
                        except Exception as exc:  # noqa: BLE001
                            logger.debug("[bridge:child:%s] on_activity callback failed: %s", self.session_id, exc)

                msg_type = str(payload.get("type") or "").strip()
                if msg_type in {"control_request", "control_response"}:
                    if self._on_control_message is not None:
                        try:
                            self._on_control_message(self.session_id, payload)
                        except Exception as exc:  # noqa: BLE001
                            logger.debug(
                                "[bridge:child:%s] on_control_message callback failed: %s",
                                self.session_id,
                                exc,
                            )

                    if msg_type == "control_request":
                        request = payload.get("request")
                        subtype = (
                            str(request.get("subtype") or "").strip().lower()
                            if isinstance(request, dict)
                            else ""
                        )
                        if subtype == "interrupt":
                            logger.info(
                                "[bridge:session] Interrupt received for sessionId=%s, terminating",
                                self.session_id,
                            )
                            self.terminate(force=False)
        finally:
            self._close_transcript()

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

    @property
    def activities(self) -> list[BridgeActivity]:
        return list(self._activities)

    @property
    def current_activity(self) -> BridgeActivity | None:
        return self._current_activity


class RemoteControlProcessSpawner:
    """Spawn local ripperdoc SDK-transport subprocesses."""

    def __init__(self, *, verbose: bool, debug_file: Path | None) -> None:
        self.verbose = verbose
        self.debug_file = debug_file
        self.replay_user_messages = parse_boolish(
            os.getenv("RIPPERDOC_REMOTE_CONTROL_REPLAY_USER_MESSAGES"),
            default=True,
        )

    @staticmethod
    def _session_suffix(session_id: str) -> str:
        return session_id.replace("/", "_")

    @staticmethod
    def _session_debug_path(base_path: Path, session_id: str) -> Path:
        suffix = RemoteControlProcessSpawner._session_suffix(session_id)
        if base_path.suffix:
            return base_path.with_name(f"{base_path.stem}-{suffix}{base_path.suffix}")
        return base_path.with_name(f"{base_path.name}-{suffix}")

    @staticmethod
    def _session_transcript_path(base_path: Path, session_id: str) -> Path:
        suffix = RemoteControlProcessSpawner._session_suffix(session_id)
        return base_path.parent / f"bridge-transcript-{suffix}.jsonl"

    def spawn(
        self,
        *,
        session_id: str,
        sdk_url: str,
        access_token: str,
        cwd: Path,
        on_activity: Callable[[str, BridgeActivity], None] | None = None,
        on_control_message: Callable[[str, dict[str, Any]], None] | None = None,
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
        if self.replay_user_messages:
            args.append("--replay-user-messages")
        session_debug: Path | None = None
        transcript_path: Path | None = None

        if self.verbose:
            args.append("--verbose")
        if self.debug_file is not None:
            session_debug = self._session_debug_path(self.debug_file, session_id)
            transcript_path = self._session_transcript_path(self.debug_file, session_id)
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
        if session_debug is not None:
            logger.debug("[bridge:session] Debug log: %s", session_debug)
        return ChildBridgeSession(
            process=process,
            session_id=session_id,
            access_token=access_token,
            verbose=self.verbose,
            transcript_path=transcript_path,
            on_activity=on_activity,
            on_control_message=on_control_message,
        )
