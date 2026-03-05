"""Remote Control bridge command entrypoint.

This module is intentionally thin and delegates runtime logic to
`ripperdoc.cli.remote_control.*` components (api/loop/process/token/ws/repl).
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Any

import click

from ripperdoc import __version__
from ripperdoc.utils.coerce import parse_boolish
from ripperdoc.utils.filesystem.git_utils import get_current_git_branch
from ripperdoc.utils.log import get_logger

from .remote_control.api import RemoteControlApiClient
from .remote_control.constants import (
    ALLOW_HTTP_ENV,
    BASE_URL_ENV,
    CONNECT_TEMPLATE_ENV,
    DEFAULT_SESSION_TIMEOUT_SEC,
    INGRESS_URL_ENV,
    TOKEN_ENV,
)
from .remote_control.errors import BridgeFatalError
from .remote_control.loop import RemoteControlBridgeRunner
from .remote_control.models import ActiveSession, RemoteControlConfig
from .remote_control.process import ChildBridgeSession, RemoteControlProcessSpawner
from .remote_control.utils import (
    build_connect_url as _build_default_connect_url,
    build_session_ingress_ws_url,
    decode_work_secret,
)

logger = get_logger()


# Backward-compat alias kept for tests and external imports.
_ActiveSession = ActiveSession


def _resolve_base_url(explicit_base_url: str | None) -> str:
    configured = explicit_base_url or os.getenv(BASE_URL_ENV) or os.getenv("RIPPERDOC_BASE_URL")
    if not configured or not configured.strip():
        raise click.ClickException(
            "Missing Remote Control base URL. Set "
            f"`{BASE_URL_ENV}` or pass `--base-url`."
        )

    value = configured.strip()
    if not value.startswith(("http://", "https://")):
        value = f"https://{value}"

    from urllib.parse import urlparse

    parsed = urlparse(value)
    hostname = (parsed.hostname or "").strip().lower()
    is_localhost = hostname in {"localhost", "127.0.0.1"} or hostname.endswith(".localhost")
    allow_insecure = parse_boolish(os.getenv(ALLOW_HTTP_ENV), default=False)
    if parsed.scheme == "http" and not is_localhost and not allow_insecure:
        raise click.ClickException(
            "Remote Control base URL uses HTTP. Use HTTPS, localhost HTTP, or set "
            f"{ALLOW_HTTP_ENV}=1."
        )
    if parsed.scheme not in {"http", "https"}:
        raise click.ClickException(f"Unsupported base URL scheme: {parsed.scheme!r}")
    return value.rstrip("/")


def _resolve_ingress_url(base_url: str) -> str:
    ingress = os.getenv(INGRESS_URL_ENV, "").strip()
    if ingress:
        return ingress.rstrip("/")
    return base_url


def _resolve_auth_token(explicit_token: str | None) -> str | None:
    token = (
        explicit_token
        or os.getenv(TOKEN_ENV)
        or os.getenv("RIPPERDOC_AUTH_TOKEN")
        or os.getenv("RIPPERDOC_API_KEY")
    )
    if isinstance(token, str):
        token = token.strip()
    return token or None


def _resolve_session_token_refresh_enabled() -> bool:
    """Return whether bridge loop should actively overwrite child session token via timer."""
    return parse_boolish(
        os.getenv("RIPPERDOC_REMOTE_CONTROL_ENABLE_SESSION_TOKEN_REFRESH"),
        default=False,
    )


def _build_connect_url(base_url: str, bridge_id: str) -> str:
    template = os.getenv(CONNECT_TEMPLATE_ENV, "").strip()
    if template:
        try:
            return template.format(base_url=base_url.rstrip("/"), bridge_id=bridge_id)
        except Exception:
            logger.debug("[bridge] Invalid connect URL template: %s", template)
    return _build_default_connect_url(base_url, bridge_id)


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

    def _token_supplier() -> str | None:
        return _resolve_auth_token(auth_token)

    resolved_auth_token = _token_supplier()
    if not resolved_auth_token:
        logger.warning(
            "[bridge] No auth token configured. Set %s if your service requires bearer auth.",
            TOKEN_ENV,
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
        refresh_access_token=lambda _current: _token_supplier(),
    )

    spawner = RemoteControlProcessSpawner(verbose=verbose, debug_file=debug_file)

    stop_event = threading.Event()
    runner = RemoteControlBridgeRunner(
        config=config,
        api_client=api_client,
        process_spawner=spawner,
        stop_event=stop_event,
        token_supplier=_token_supplier if _resolve_session_token_refresh_enabled() else None,
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
    help=f"Remote control API base URL (default: ${BASE_URL_ENV}).",
)
@click.option(
    "--auth-token",
    type=str,
    default=None,
    help=f"Bearer token for the remote control API (default: ${TOKEN_ENV}).",
)
@click.option(
    "--session-timeout",
    type=int,
    default=DEFAULT_SESSION_TIMEOUT_SEC,
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
    "RemoteControlApiClient",
    "RemoteControlBridgeRunner",
    "RemoteControlConfig",
    "RemoteControlProcessSpawner",
    "ChildBridgeSession",
    "ActiveSession",
    "_ActiveSession",
]
