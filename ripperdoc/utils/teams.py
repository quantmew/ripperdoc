"""File-backed team coordination primitives.

A Team is a collaboration domain that groups members, shared context metadata,
and a 1:1 mapped task list id.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional, Sequence
from uuid import uuid4

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError

from ripperdoc.utils.file_editing import file_lock
from ripperdoc.utils.log import get_logger
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


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
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
        recipients=[val for val in (recipients or ["*"]) if str(val).strip()] or ["*"],
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
