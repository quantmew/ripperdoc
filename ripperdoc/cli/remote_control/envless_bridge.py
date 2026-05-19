"""Env-less Remote Control bridge core.

Connects directly to the session-ingress layer without the
Environments API work-dispatch layer:

1. POST /v1/code/sessions             -> session.id
2. POST /v1/code/sessions/{id}/bridge -> worker_jwt, expires_in, api_base_url, worker_epoch
3. create_v2_repl_transport            -> SSE + CCRClient
4. create_token_refresh_scheduler      -> proactive /bridge re-call
5. 401 on SSE -> rebuild transport with fresh /bridge credentials
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from ripperdoc.utils.log import get_logger

from .echo_dedup import BoundedUUIDSet, handle_ingress_message
from .flush_gate import FlushGate
from .v2_config import EnvLessBridgeConfig
from .v2_transport import ReplBridgeTransport, create_v2_repl_transport

logger = get_logger()

ANTHROPIC_VERSION = "2023-06-01"


@dataclass
class RemoteCredentials:
    """Bridge credentials from POST /bridge."""

    worker_jwt: str
    expires_in: int
    api_base_url: str
    worker_epoch: int


@dataclass
class EnvLessBridgeParams:
    """Parameters for init_envless_bridge."""

    base_url: str
    org_uuid: str
    title: str
    get_access_token: Callable[[], Optional[str]]
    on_auth_401: Optional[Callable[[str], bool]] = None
    initial_messages: Optional[List[Dict[str, Any]]] = None
    initial_history_cap: int = 0
    on_inbound_message: Optional[Callable[[Dict[str, Any]], None]] = None
    on_user_message: Optional[Callable[[str, str], bool]] = None
    on_permission_response: Optional[Callable[[Dict[str, Any]], None]] = None
    on_interrupt: Optional[Callable[[], None]] = None
    on_set_model: Optional[Callable[[Optional[str]], None]] = None
    on_state_change: Optional[Callable[[str, Optional[str]], None]] = None
    outbound_only: bool = False
    tags: Optional[List[str]] = None


@dataclass
class ReplBridgeHandle:
    """Handle returned by init_envless_bridge for interacting with the bridge."""

    bridge_session_id: str
    environment_id: str
    session_ingress_url: str
    write_messages: Callable[[List[Dict[str, Any]]], None]
    teardown: Callable[[], None]


def init_envless_bridge(
    params: EnvLessBridgeParams,
    *,
    api_client: Any,
) -> Optional[ReplBridgeHandle]:
    """Create a session, fetch a worker JWT, connect the v2 transport.

    Returns None on any pre-flight failure.
    """
    cfg = EnvLessBridgeConfig()

    # 1. Get access token
    access_token = params.get_access_token()
    if not access_token:
        logger.debug("[remote-bridge] No OAuth token")
        return None

    # 2. Create session (POST /v1/code/sessions)
    created_session_id = _with_retry(
        lambda: api_client.create_code_session(
            params.base_url, access_token, params.title,
            timeout_sec=cfg.http_timeout_ms / 1000.0,
            tags=params.tags,
        ),
        "createCodeSession",
        cfg,
    )
    if not created_session_id:
        params.on_state_change("failed", "Session creation failed") if params.on_state_change else None
        return None

    session_id = created_session_id
    logger.info("[remote-bridge] Created session %s", session_id)

    # 3. Fetch bridge credentials
    credentials_data = _with_retry(
        lambda: api_client.fetch_remote_credentials(
            session_id, params.base_url, access_token,
            timeout_sec=cfg.http_timeout_ms / 1000.0,
        ),
        "fetchRemoteCredentials",
        cfg,
    )
    if not credentials_data:
        params.on_state_change("failed", "Remote credentials fetch failed") if params.on_state_change else None
        return None

    credentials = RemoteCredentials(
        worker_jwt=str(credentials_data.get("worker_jwt") or credentials_data.get("workerJwt") or ""),
        expires_in=int(credentials_data.get("expires_in") or credentials_data.get("expiresIn") or 3600),
        api_base_url=str(credentials_data.get("api_base_url") or credentials_data.get("apiBaseUrl") or params.base_url),
        worker_epoch=int(credentials_data.get("worker_epoch") or credentials_data.get("workerEpoch") or 0),
    )

    if not credentials.worker_jwt:
        logger.warning("[remote-bridge] No worker_jwt in credentials response")
        return None

    logger.info("[remote-bridge] Fetched bridge credentials (expires_in=%ds)", credentials.expires_in)

    # 4. Build v2 transport
    session_url = _build_ccr_v2_sdk_url(credentials.api_base_url, session_id)
    logger.debug("[remote-bridge] v2 session URL: %s", session_url)

    transport: Optional[ReplBridgeTransport] = None
    try:
        transport = create_v2_repl_transport(
            session_url,
            credentials.worker_jwt,
            epoch=credentials.worker_epoch,
            heartbeat_interval_sec=cfg.heartbeat_interval_ms / 1000.0,
            initial_sequence_num=0,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("[remote-bridge] v2 transport setup failed: %s", exc)
        params.on_state_change("failed", f"Transport setup failed: {exc}") if params.on_state_change else None
        return None

    # 5. State setup
    recent_posted_uuids = BoundedUUIDSet(cfg.uuid_dedup_buffer_size)
    recent_inbound_uuids = BoundedUUIDSet(cfg.uuid_dedup_buffer_size)
    flush_gate: FlushGate[dict[str, Any]] = FlushGate()

    initial_message_uuids: set[str] = set()
    if params.initial_messages:
        for msg in params.initial_messages:
            msg_uuid = msg.get("uuid")
            if isinstance(msg_uuid, str) and msg_uuid:
                initial_message_uuids.add(msg_uuid)
                recent_posted_uuids.add(msg_uuid)

    torn_down = False
    initial_flush_done = False

    # 6. JWT refresh scheduler
    def _on_token_refresh(session_id: str, token: str) -> None:
        pass  # Token refresh handled by the caller

    # 7. Wire transport callbacks
    def _on_connect() -> None:
        nonlocal initial_flush_done
        logger.info("[remote-bridge] v2 transport connected")

        if not initial_flush_done and params.initial_messages and len(params.initial_messages) > 0:
            initial_flush_done = True
            _flush_history(params.initial_messages)
            _drain_flush_gate()

        params.on_state_change("connected", None) if params.on_state_change else None

    def _on_data(data: str) -> None:
        handle_ingress_message(
            data,
            recent_posted_uuids,
            recent_inbound_uuids,
            on_inbound_message=params.on_inbound_message,
            on_permission_response=params.on_permission_response,
            on_control_request=None,
        )

    def _on_close(code: Optional[int]) -> None:
        nonlocal torn_down
        if torn_down:
            return
        logger.warning("[remote-bridge] v2 transport closed (code=%s)", code)
        params.on_state_change("failed", f"Transport closed (code {code})") if params.on_state_change else None

    transport.set_on_connect(_on_connect)
    transport.set_on_data(_on_data)
    transport.set_on_close(_on_close)

    # 8. Flush helpers
    def _drain_flush_gate() -> None:
        msgs = flush_gate.end()
        if not msgs:
            return
        for msg in msgs:
            msg_uuid = msg.get("uuid")
            if isinstance(msg_uuid, str) and msg_uuid:
                recent_posted_uuids.add(msg_uuid)
        events = [{**m, "session_id": session_id} for m in msgs]
        transport.write_batch(events)
        logger.debug("[remote-bridge] Drained %d queued message(s)", len(msgs))

    def _flush_history(msgs: list[dict[str, Any]]) -> None:
        eligible = [m for m in msgs if _is_eligible_bridge_message(m)]
        if cfg.initial_history_cap > 0 and len(eligible) > cfg.initial_history_cap:
            eligible = eligible[-cfg.initial_history_cap:]
        events = [{**m, "session_id": session_id} for m in eligible]
        if events:
            logger.debug("[remote-bridge] Flushing %d history events", len(events))
            transport.write_batch(events)

    # Start flush gate before connect
    if params.initial_messages and len(params.initial_messages) > 0:
        flush_gate.start()

    transport.connect()

    # 9. Teardown
    def _teardown() -> None:
        nonlocal torn_down
        if torn_down:
            return
        torn_down = True
        flush_gate.drop()
        transport.close()
        try:
            api_client.archive_session(session_id)
        except Exception as exc:  # noqa: BLE001
            logger.debug("[remote-bridge] Archive failed: %s", exc)
        logger.info("[remote-bridge] Torn down")

    # 10. Return handle
    def _write_messages(messages: list[dict[str, Any]]) -> None:
        filtered = [
            m for m in messages
            if _is_eligible_bridge_message(m)
            and m.get("uuid") not in initial_message_uuids
            and not recent_posted_uuids.has(m.get("uuid", ""))
        ]
        if not filtered:
            return

        # Fire on_user_message for title derivation
        if params.on_user_message:
            for m in filtered:
                text = _extract_title_text(m)
                if text and params.on_user_message(text, session_id):
                    break

        if flush_gate.enqueue(*filtered):
            return

        for msg in filtered:
            msg_uuid = msg.get("uuid")
            if isinstance(msg_uuid, str) and msg_uuid:
                recent_posted_uuids.add(msg_uuid)

        events = [{**m, "session_id": session_id} for m in filtered]
        if any(m.get("type") == "user" for m in filtered):
            transport.report_state("running")
        transport.write_batch(events)

    return ReplBridgeHandle(
        bridge_session_id=session_id,
        environment_id="",
        session_ingress_url=credentials.api_base_url,
        write_messages=_write_messages,
        teardown=_teardown,
    )


def _build_ccr_v2_sdk_url(api_base_url: str, session_id: str) -> str:
    """Build the SDK URL for CCR v2 transport."""
    base = api_base_url.rstrip("/")
    return f"{base}/v2/session_ingress/ws/{session_id}"


def _is_eligible_bridge_message(m: dict[str, Any]) -> bool:
    """True for messages that should be forwarded to the bridge transport."""
    msg_type = str(m.get("type") or "")
    if msg_type in ("user", "assistant"):
        if m.get("isVirtual"):
            return False
        return True
    if msg_type == "system" and m.get("subtype") == "local_command":
        return True
    return False


def _extract_title_text(m: Dict[str, Any]) -> Optional[str]:
    """Extract title-worthy text from a message."""
    if m.get("type") != "user":
        return None
    if m.get("isMeta") or m.get("toolUseResult") or m.get("isCompactSummary"):
        return None
    content = m.get("message", {}).get("content")
    if isinstance(content, str):
        return content.strip() or None
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "").strip()
                return text or None
    return None


def _with_retry(
    fn: Callable[[], Any],
    label: str,
    cfg: EnvLessBridgeConfig,
) -> Any:
    """Retry an init call with exponential backoff."""
    for attempt in range(1, cfg.init_retry_max_attempts + 1):
        result = fn()
        if result is not None:
            return result
        if attempt < cfg.init_retry_max_attempts:
            import random
            base = cfg.init_retry_base_delay_ms * (2 ** (attempt - 1))
            jitter = base * cfg.init_retry_jitter_fraction * (2 * random.random() - 1)
            delay = min(base + jitter, cfg.init_retry_max_delay_ms) / 1000.0
            logger.debug(
                "[remote-bridge] %s failed (attempt %d/%d), retrying in %.1fs",
                label, attempt, cfg.init_retry_max_attempts, delay,
            )
            time.sleep(delay)
    return None
