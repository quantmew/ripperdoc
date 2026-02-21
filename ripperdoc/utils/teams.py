"""File-backed team coordination primitives.

A Team is a collaboration domain that groups members, shared context metadata,
and a 1:1 mapped task list id.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional, Sequence
from uuid import uuid4
from weakref import WeakSet

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError

from ripperdoc.utils.file_editing import file_lock
from ripperdoc.utils.log import get_logger
from ripperdoc.utils.pending_messages import PendingMessageQueue
from ripperdoc.utils.tasks import ensure_task_list_dir, sanitize_identifier


logger = get_logger()

TeamMessageType = Literal[
    "message",
    "broadcast",
    "shutdown_request",
    "shutdown_response",
    "plan_approval_response",
    # Internal/system-level message kinds used by task delegation flows.
    "direct",
    "shutdown",
    "plan_approval",
    "assignment",
    "task_assignment",
    "delegate",
    "status",
]


_TeamListenerToken = tuple[str, str, int]
_TEAM_MESSAGE_LISTENERS: dict[str, dict[str, WeakSet[PendingMessageQueue]]] = {}
_TEAM_MESSAGE_LISTENER_LOCK = threading.Lock()


class TeamMember(BaseModel):
    """Teammate descriptor within a Team."""

    name: str
    agent_type: str = Field(
        validation_alias=AliasChoices("agent_type", "agentType"),
        serialization_alias="agentType",
    )
    role: str = "worker"
    active: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)
    joined_at: float = Field(
        default_factory=time.time,
        validation_alias=AliasChoices("joined_at", "joinedAt"),
        serialization_alias="joinedAt",
    )

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class TeamConfig(BaseModel):
    """Persistent team configuration."""

    name: str
    task_list_id: str = Field(
        validation_alias=AliasChoices("task_list_id", "taskListId"),
        serialization_alias="taskListId",
    )
    members: list[TeamMember] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    version: int = Field(default=1, ge=1)
    created_at: float = Field(
        default_factory=time.time,
        validation_alias=AliasChoices("created_at", "createdAt"),
        serialization_alias="createdAt",
    )
    updated_at: float = Field(
        default_factory=time.time,
        validation_alias=AliasChoices("updated_at", "updatedAt"),
        serialization_alias="updatedAt",
    )

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class TeamMessage(BaseModel):
    """Message envelope exchanged in a team domain."""

    id: str = Field(default_factory=lambda: f"msg_{uuid4().hex[:10]}")
    team_name: str = Field(
        validation_alias=AliasChoices("team_name", "teamName"),
        serialization_alias="teamName",
    )
    sender: str
    recipients: list[str] = Field(default_factory=lambda: ["*"])
    message_type: TeamMessageType = Field(
        default="direct",
        validation_alias=AliasChoices("message_type", "messageType"),
        serialization_alias="messageType",
    )
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(
        default_factory=time.time,
        validation_alias=AliasChoices("created_at", "createdAt"),
        serialization_alias="createdAt",
    )

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


def _normalize_team_participant(name: str) -> str:
    return (name or "").strip() or "team-lead"


def _normalize_mailbox_participant(name: str) -> str:
    # Keep mailbox path-friendly names while preserving readable participant identity.
    return sanitize_identifier(name or "", fallback="team-lead")


def _team_inboxes_dir(team_name: str) -> Path:
    return team_dir(team_name, ensure=True) / "inboxes"


def _team_inbox_path(team_name: str, participant: str) -> Path:
    normalized_participant = _normalize_mailbox_participant(participant)
    return _team_inboxes_dir(team_name) / f"{normalized_participant}.json"


def _load_team_inbox_entries(team_name: str, participant: str) -> list[dict[str, Any]]:
    path = _team_inbox_path(team_name, participant)
    if not path.exists():
        return []
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, IOError, UnicodeDecodeError) as exc:
        logger.warning(
            "[teams] Failed reading inbox for %s:%s: %s",
            team_name,
            participant,
            exc,
            extra={"path": str(path)},
        )
        return []

    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        logger.warning(
            "[teams] Skipping malformed inbox payload for %s:%s: %s",
            team_name,
            participant,
            exc,
            extra={"path": str(path)},
        )
        return []

    if not isinstance(payload, list):
        logger.warning(
            "[teams] Inbox payload is not a list for %s:%s",
            team_name,
            participant,
            extra={"path": str(path)},
        )
        return []

    loaded: list[dict[str, Any]] = []
    for item in payload:
        if isinstance(item, dict):
            loaded.append(item)
    return loaded


def _store_team_inbox_entries(
    team_name: str,
    participant: str,
    entries: list[dict[str, Any]],
) -> None:
    if not entries:
        return
    path = _team_inbox_path(team_name, participant)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(path, entries)


def _resolve_inbox_recipients(
    team_name: str,
    team: Optional[TeamConfig],
    recipients: Sequence[str],
) -> list[str]:
    raw_recipients = [
        _normalize_team_participant(value)
        for value in recipients
        if isinstance(value, str) and value.strip()
    ]
    if not raw_recipients:
        return []

    if "*" not in raw_recipients:
        return list(dict.fromkeys(raw_recipients))

    names: list[str] = []
    if team:
        names.extend(member.name for member in team.members if member.name)
    else:
        names.append("team-lead")
    return list(dict.fromkeys(names))


def _build_inbox_entry(
    message: TeamMessage,
    recipient: str,
) -> Dict[str, Any]:
    recipient_normalized = _normalize_team_participant(recipient)
    return {
        "id": message.id,
        "team_name": message.team_name,
        "recipient": recipient_normalized,
        "sender": message.sender,
        "message_type": message.message_type,
        "content": message.content,
        "created_at": message.created_at,
        "metadata": _build_queue_notification_payload(message, recipient_normalized),
        "read": False,
    }


def drain_team_inbox_messages(team_name: str, participant: str) -> list[dict[str, Any]]:
    clean_team = (team_name or "").strip()
    clean_participant = _normalize_team_participant(participant)
    if not clean_team:
        return []

    with _team_lock(clean_team):
        entries = _load_team_inbox_entries(clean_team, clean_participant)
        unread = [
            dict(item)
            for item in entries
            if isinstance(item, dict) and not item.get("read")
        ]
        if not unread:
            return []

        for item in entries:
            if isinstance(item, dict) and not item.get("read"):
                item["read"] = True

        _store_team_inbox_entries(clean_team, clean_participant, entries)

        return unread


def register_team_message_listener(
    team_name: str,
    participant: str,
    queue: PendingMessageQueue,
) -> _TeamListenerToken:
    clean_team = (team_name or "").strip()
    if not clean_team:
        raise ValueError("team_name is required")
    clean_participant = _normalize_team_participant(participant)

    with _TEAM_MESSAGE_LISTENER_LOCK:
        participants = _TEAM_MESSAGE_LISTENERS.setdefault(clean_team, {})
        listeners = participants.setdefault(clean_participant, WeakSet())
        listeners.add(queue)

    return (clean_team, clean_participant, id(queue))


def unregister_team_message_listener(
    team_name: str,
    participant: str,
    queue: PendingMessageQueue,
) -> None:
    clean_team = (team_name or "").strip()
    if not clean_team:
        return
    clean_participant = _normalize_team_participant(participant)

    with _TEAM_MESSAGE_LISTENER_LOCK:
        participants = _TEAM_MESSAGE_LISTENERS.get(clean_team)
        if not participants:
            return
        listeners = participants.get(clean_participant)
        if not listeners:
            return
        listeners.discard(queue)
        if not listeners:
            participants.pop(clean_participant, None)
        if not participants:
            _TEAM_MESSAGE_LISTENERS.pop(clean_team, None)


def _snapshot_listeners_for_recipients(
    team_name: str,
    recipients: Sequence[str],
) -> dict[int, tuple[PendingMessageQueue, str]]:
    """Build a stable queue snapshot for recipients without holding the lock."""
    clean_team = (team_name or "").strip()
    if not clean_team:
        return {}

    raw_recipients = [
        _normalize_team_participant(value)
        for value in recipients
        if isinstance(value, str) and value.strip()
    ] or ["team-lead"]

    wildcard = "*" in raw_recipients
    target_names = set(raw_recipients)

    with _TEAM_MESSAGE_LISTENER_LOCK:
        participants = _TEAM_MESSAGE_LISTENERS.get(clean_team, {})
        if not participants:
            return {}

        snapshots: dict[int, tuple[PendingMessageQueue, str]] = {}

        if wildcard or "*" in target_names:
            for participant_name, queues in participants.items():
                for queue in list(queues):
                    snapshots[id(queue)] = (queue, participant_name)
        else:
            for participant_name in sorted(target_names):
                for queue in list(participants.get(participant_name, [])):
                    snapshots[id(queue)] = (queue, participant_name)

        return snapshots


def _build_queue_notification_payload(
    message: TeamMessage,
    recipient: str,
) -> Dict[str, Any]:
    """Create user-visible payload metadata for team messages in queues."""
    payload = {
        "team": {
            "team_name": message.team_name,
            "sender": message.sender,
            "recipient": recipient,
            "message_type": message.message_type,
            "message_id": message.id,
            "created_at": message.created_at,
        },
        "team_message_type": message.message_type,
        "team_name": message.team_name,
        "sender": message.sender,
        "recipient": recipient,
        "request_id": message.metadata.get("request_id"),
        "approve": message.metadata.get("approve"),
    }
    # Keep message-level metadata (e.g., summary) in the same payload so inbox
    # consumers can preserve UI hints without needing to inspect storage again.
    payload.update(message.metadata or {})
    return payload


def _coerce_message_recipients(recipients: Optional[Sequence[str]]) -> list[str]:
    values = [
        str(value).strip()
        for value in recipients or []
        if value is not None and str(value).strip()
    ]
    return values or ["*"]


def set_team_member_active(
    team_name: str,
    member_name: str,
    *,
    active: bool,
    default_agent_type: str = "general-purpose",
) -> Optional[TeamConfig]:
    """Set a team member's active flag in persistent config.

    If a teammate is not found and ``active`` is true, a new member will be added
    using ``default_agent_type`` so that runtime shutdown checks keep working.
    """
    clean_team = (team_name or "").strip()
    clean_member = (member_name or "").strip()
    if not clean_team or not clean_member:
        return None

    with _team_lock(clean_team):
        team = get_team(clean_team)
        if team is None:
            raise ValueError(f"Team '{clean_team}' not found.")

        updated_members: list[TeamMember] = []
        found = False
        changed = False

        for existing in team.members:
            if existing.name == clean_member:
                found = True
                if existing.active != active:
                    updated_members.append(existing.model_copy(update={"active": active}))
                    changed = True
                else:
                    updated_members.append(existing)
            else:
                updated_members.append(existing)

        if not found:
            if not active:
                return team
            updated_members.append(
                TeamMember(
                    name=clean_member,
                    agent_type=default_agent_type,
                    role="worker",
                    active=True,
                )
            )
            changed = True

        if not changed:
            return team

        team.members = updated_members
        team.updated_at = time.time()
        team.version += 1
        _write_json_atomic(_team_config_path(clean_team), team.model_dump(by_alias=True))
        return team


def _config_root() -> Path:
    raw = os.getenv("RIPPERDOC_CONFIG_DIR")
    if raw and raw.strip():
        return Path(raw).expanduser()
    return Path.home() / ".ripperdoc"


def _teams_root(*, ensure: bool = False) -> Path:
    directory = _config_root() / "teams"
    if ensure:
        directory.mkdir(parents=True, exist_ok=True)
    return directory


def _team_dir_name(team_name: str) -> str:
    return sanitize_identifier(team_name, fallback="team")


def team_dir(team_name: str, *, ensure: bool = False) -> Path:
    directory = _teams_root(ensure=ensure) / _team_dir_name(team_name)
    if ensure:
        directory.mkdir(parents=True, exist_ok=True)
    return directory


def _team_config_path(team_name: str) -> Path:
    return team_dir(team_name, ensure=False) / "config.json"


def _team_messages_path(team_name: str) -> Path:
    return team_dir(team_name, ensure=False) / "messages.jsonl"


def _active_team_path() -> Path:
    return _teams_root(ensure=True) / ".active_team"


@contextmanager
def _team_lock(team_name: str) -> Iterator[None]:
    directory = team_dir(team_name, ensure=True)
    lock_path = directory / ".lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        with file_lock(handle, exclusive=True):
            yield


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, ensure_ascii=False)
    fd, temp_path = tempfile.mkstemp(dir=str(path.parent), prefix=".team_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.write("\n")
        os.replace(temp_path, path)
    finally:
        try:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        except OSError:
            pass


def _read_team_config(path: Path) -> Optional[TeamConfig]:
    if not path.exists():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, IOError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.warning(
            "[teams] Failed reading team config: %s: %s",
            type(exc).__name__,
            exc,
            extra={"path": str(path)},
        )
        return None

    if not isinstance(payload, dict):
        return None

    try:
        return TeamConfig(**payload)
    except ValidationError as exc:
        logger.warning(
            "[teams] Invalid team config schema: %s",
            exc,
            extra={"path": str(path)},
        )
        return None


def get_team(team_name: str) -> Optional[TeamConfig]:
    """Load team configuration by name."""

    return _read_team_config(_team_config_path(team_name))


def team_config_path(team_name: str) -> Path:
    """Return canonical team config path."""

    return _team_config_path(team_name)


def list_teams() -> List[TeamConfig]:
    """Return all valid team configurations."""

    root = _teams_root(ensure=False)
    if not root.exists():
        return []

    teams: list[TeamConfig] = []
    for path in sorted(root.glob("*/config.json")):
        team = _read_team_config(path)
        if team is not None:
            teams.append(team)

    return sorted(teams, key=lambda item: (item.created_at, item.name))


def set_active_team_name(team_name: str) -> None:
    """Persist active team name for tools that rely on team context."""

    clean_name = (team_name or "").strip()
    if not clean_name:
        raise ValueError("team_name is required")
    path = _active_team_path()
    path.write_text(clean_name + "\n", encoding="utf-8")


def get_active_team_name() -> Optional[str]:
    """Load current active team name from disk."""

    path = _active_team_path()
    if not path.exists():
        return None
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value or None


def clear_active_team_name(team_name: Optional[str] = None) -> None:
    """Clear active team marker (optionally only when matching a team name)."""

    path = _active_team_path()
    if not path.exists():
        return
    if team_name:
        current = get_active_team_name()
        if current and current != team_name:
            return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def create_team(
    *,
    name: str,
    members: Optional[Sequence[TeamMember]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    task_list_id: Optional[str] = None,
) -> TeamConfig:
    """Create a team and initialize its task-list mapping."""

    clean_name = (name or "").strip()
    if not clean_name:
        raise ValueError("name is required")

    with _team_lock(clean_name):
        config_path = _team_config_path(clean_name)
        existing = _read_team_config(config_path)
        if existing is not None:
            raise ValueError(f"Team '{clean_name}' already exists.")

        resolved_task_list_id = sanitize_identifier(task_list_id or clean_name, fallback="default")
        ensure_task_list_dir(task_list_id=resolved_task_list_id)

        now = time.time()
        team = TeamConfig(
            name=clean_name,
            task_list_id=resolved_task_list_id,
            members=list(members or []),
            metadata=dict(metadata or {}),
            created_at=now,
            updated_at=now,
            version=1,
        )
        _write_json_atomic(config_path, team.model_dump(by_alias=True))
        return team


def delete_team(team_name: str, *, purge_messages: bool = True) -> bool:
    """Delete a team config and optionally its message log."""

    clean_name = (team_name or "").strip()
    if not clean_name:
        return False

    with _team_lock(clean_name):
        config_path = _team_config_path(clean_name)
        messages_path = _team_messages_path(clean_name)
        inbox_dir = _team_inboxes_dir(clean_name)

        removed = False
        try:
            config_path.unlink(missing_ok=True)
            removed = True
        except OSError as exc:
            logger.warning(
                "[teams] Failed deleting team config: %s: %s",
                type(exc).__name__,
                exc,
                extra={"path": str(config_path)},
            )

        if purge_messages:
            try:
                messages_path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning(
                    "[teams] Failed deleting team messages: %s: %s",
                    type(exc).__name__,
                    exc,
                    extra={"path": str(messages_path)},
                )

        if purge_messages:
            try:
                shutil.rmtree(inbox_dir, ignore_errors=True)
            except OSError as exc:
                logger.warning(
                    "[teams] Failed deleting inbox directory: %s: %s",
                    type(exc).__name__,
                    exc,
                    extra={"path": str(inbox_dir)},
                )

        return removed


def upsert_team_member(team_name: str, member: TeamMember) -> TeamConfig:
    """Insert or replace a member entry in team config."""

    with _team_lock(team_name):
        team = get_team(team_name)
        if team is None:
            raise ValueError(f"Team '{team_name}' not found.")

        updated_members: list[TeamMember] = []
        replaced = False
        for existing in team.members:
            if existing.name == member.name:
                updated_members.append(member)
                replaced = True
            else:
                updated_members.append(existing)
        if not replaced:
            updated_members.append(member)

        team.members = updated_members
        team.updated_at = time.time()
        team.version += 1
        _write_json_atomic(_team_config_path(team_name), team.model_dump(by_alias=True))
        return team


def remove_team_member(team_name: str, member_name: str) -> TeamConfig:
    """Remove a member from team config."""

    with _team_lock(team_name):
        team = get_team(team_name)
        if team is None:
            raise ValueError(f"Team '{team_name}' not found.")

        team.members = [member for member in team.members if member.name != member_name]
        team.updated_at = time.time()
        team.version += 1
        _write_json_atomic(_team_config_path(team_name), team.model_dump(by_alias=True))
        return team


def send_team_message(
    *,
    team_name: str,
    sender: str,
    content: str,
    recipients: Optional[Sequence[str]] = None,
    message_type: TeamMessageType = "direct",
    metadata: Optional[Dict[str, Any]] = None,
) -> TeamMessage:
    """Append a message to the team message log."""

    if not content.strip():
        raise ValueError("content is required")

    team = get_team(team_name)
    if team is None:
        raise ValueError(f"Team '{team_name}' not found.")

    message = TeamMessage(
        team_name=team_name,
        sender=(sender or "system").strip() or "system",
        recipients=_coerce_message_recipients(recipients),
        message_type=message_type,
        content=content,
        metadata=dict(metadata or {}),
        created_at=time.time(),
    )

    with _team_lock(team_name):
        path = _team_messages_path(team_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(message.model_dump(by_alias=True), ensure_ascii=False))
            handle.write("\n")

        team.updated_at = time.time()
        team.version += 1
        _write_json_atomic(_team_config_path(team_name), team.model_dump(by_alias=True))

        inbox_recipients = _resolve_inbox_recipients(team_name, team, message.recipients)

    recipient_listener_map = _snapshot_listeners_for_recipients(team_name, message.recipients)
    live_recipients = {
        _normalize_team_participant(recipient)
        for _, recipient in recipient_listener_map.values()
    }

    for recipient in inbox_recipients:
        normalized_recipient = _normalize_team_participant(recipient)
        if normalized_recipient in live_recipients:
            continue
        entries = _load_team_inbox_entries(team_name, normalized_recipient)
        entries.append(_build_inbox_entry(message, normalized_recipient))
        _store_team_inbox_entries(team_name, normalized_recipient, entries)

    if recipient_listener_map:
        for _, (queue, recipient) in recipient_listener_map.items():
            payload = _build_queue_notification_payload(message, recipient)
            queue.enqueue_text(message.content, metadata=payload)

    return message


def list_team_messages(
    team_name: str,
    *,
    since_ts: Optional[float] = None,
    limit: int = 100,
) -> List[TeamMessage]:
    """Read recent messages for a team."""

    path = _team_messages_path(team_name)
    if not path.exists():
        return []

    messages: list[TeamMessage] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, IOError, UnicodeDecodeError) as exc:
        logger.warning(
            "[teams] Failed reading team messages: %s: %s",
            type(exc).__name__,
            exc,
            extra={"path": str(path)},
        )
        return []

    for line in lines:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            message = TeamMessage(**payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.warning(
                "[teams] Skipping malformed team message: %s",
                exc,
                extra={"path": str(path)},
            )
            continue

        if since_ts is not None and message.created_at < since_ts:
            continue
        messages.append(message)

    if limit > 0:
        return messages[-limit:]
    return messages


def find_team_by_task_list_id(task_list_id: str) -> Optional[TeamConfig]:
    """Return the first team mapped to the provided task list id."""

    resolved = sanitize_identifier(task_list_id, fallback="default")
    for team in list_teams():
        if sanitize_identifier(team.task_list_id, fallback="default") == resolved:
            return team
    return None


__all__ = [
    "TeamConfig",
    "TeamMember",
    "TeamMessage",
    "TeamMessageType",
    "set_team_member_active",
    "clear_active_team_name",
    "create_team",
    "delete_team",
    "find_team_by_task_list_id",
    "get_active_team_name",
    "get_team",
    "list_team_messages",
    "list_teams",
    "remove_team_member",
    "send_team_message",
    "set_active_team_name",
    "team_config_path",
    "team_dir",
    "upsert_team_member",
]
