"""Data models for remote-control bridge components."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


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


class BridgeChildProcess(Protocol):
    """Minimal process behavior needed by the bridge loop."""

    session_id: str

    def poll(self) -> int | None: ...

    def is_running(self) -> bool: ...

    def write_stdin(self, payload: str) -> None: ...

    def update_access_token(self, token: str) -> None: ...

    def terminate(self, *, force: bool = False) -> None: ...

    @property
    def stderr_tail(self) -> list[str]: ...


@dataclass
class ActiveSession:
    """Live session tracked by the bridge loop."""

    session_id: str
    work_id: str
    process: BridgeChildProcess
    started_at_monotonic: float = time.monotonic()
