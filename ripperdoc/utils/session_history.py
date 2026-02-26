"""Session log storage and retrieval."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Union

from ripperdoc.utils.log import get_logger
from ripperdoc.utils.messages import (
    AssistantMessage,
    ProgressMessage,
    UserMessage,
)
from ripperdoc.utils.path_utils import project_storage_dir
from ripperdoc.utils.session_index import (
    list_session_index_entries,
    update_session_index_for_payload,
)


logger = get_logger()

ConversationMessage = Union[UserMessage, AssistantMessage, ProgressMessage]


@dataclass
class SessionSummary:
    session_id: str
    path: Path
    message_count: int
    created_at: datetime
    updated_at: datetime
    title: str
    last_prompt: str

    @property
    def display_title(self) -> str:
        title = (self.title or "").strip()
        if title:
            return title
        return self.last_prompt


def _sessions_root() -> Path:
    return Path.home() / ".ripperdoc" / "sessions"


def _session_file(project_path: Path, session_id: str) -> Path:
    directory = project_storage_dir(_sessions_root(), project_path, ensure=True)
    return directory / f"{session_id}.jsonl"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _deserialize_message(payload: dict) -> Optional[ConversationMessage]:
    """Rebuild a message model from a stored payload."""
    msg_type = payload.get("type")
    if msg_type == "user":
        return UserMessage(**payload)
    if msg_type == "assistant":
        return AssistantMessage(**payload)
    if msg_type == "progress":
        return ProgressMessage(**payload)
    return None


class SessionHistory:
    """Append-only session log for a single session id."""

    def __init__(self, project_path: Path, session_id: str):
        self.project_path = project_path
        self.session_id = session_id
        self.path = _session_file(project_path, session_id)
        self._seen_ids: set[str] = set()
        self._load_seen_ids()

    def _load_seen_ids(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        data = json.loads(line)
                        payload = data.get("payload") or {}
                        msg_uuid = payload.get("uuid")
                        if isinstance(msg_uuid, str):
                            self._seen_ids.add(msg_uuid)
                    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                        logger.debug(
                            "[session_history] Failed to parse session history line: %s: %s",
                            type(exc).__name__,
                            exc,
                        )
                        continue
        except (OSError, IOError) as exc:
            logger.warning(
                "Failed to load seen IDs from session: %s: %s",
                type(exc).__name__,
                exc,
                extra={"session_id": self.session_id, "path": str(self.path)},
            )
            return

    def append(self, message: ConversationMessage) -> None:
        """Persist a single message to the session log."""
        # Skip progress noise
        if getattr(message, "type", None) == "progress":
            return
        msg_uuid = getattr(message, "uuid", None)
        if isinstance(msg_uuid, str) and msg_uuid in self._seen_ids:
            return

        payload = message.model_dump(mode="json")
        logged_at = _now_iso()
        entry = {
            "logged_at": logged_at,
            "project_path": str(self.project_path.resolve()),
            "payload": payload,
        }
        try:
            with self.path.open("a", encoding="utf-8") as fh:
                # ensure_ascii=False prevents Chinese characters from being escaped to \uXXXX
                # separators removes extra spaces to reduce file size
                json.dump(entry, fh, ensure_ascii=False, separators=(",", ":"))
                fh.write("\n")
            if isinstance(msg_uuid, str):
                self._seen_ids.add(msg_uuid)
            try:
                update_session_index_for_payload(
                    self.project_path,
                    self.session_id,
                    payload,
                    logged_at=logged_at,
                    session_file=self.path,
                )
            except (OSError, IOError, ValueError, TypeError) as exc:
                logger.warning(
                    "Failed to update session index: %s: %s",
                    type(exc).__name__,
                    exc,
                    extra={"session_id": self.session_id, "path": str(self.path)},
                )
        except (OSError, IOError) as exc:
            # Avoid crashing the UI if logging fails
            logger.warning(
                "Failed to append message to session log: %s: %s",
                type(exc).__name__,
                exc,
                extra={"session_id": self.session_id, "path": str(self.path)},
            )
            return


def list_session_summaries(project_path: Path) -> List[SessionSummary]:
    """Return available sessions for the project ordered by last update desc."""
    directory = project_storage_dir(_sessions_root(), project_path)
    if not directory.exists():
        return []

    summaries: List[SessionSummary] = []
    for entry in list_session_index_entries(project_path):
        session_path = directory / f"{entry.session_id}.jsonl"
        if not session_path.exists():
            continue

        created_at = entry.created_at or datetime.fromtimestamp(
            session_path.stat().st_ctime, tz=timezone.utc
        )
        updated_at = entry.updated_at or datetime.fromtimestamp(
            session_path.stat().st_mtime, tz=timezone.utc
        )

        summaries.append(
            SessionSummary(
                session_id=entry.session_id,
                path=session_path,
                message_count=entry.message_count,
                created_at=created_at,
                updated_at=updated_at,
                title=entry.title,
                last_prompt=entry.preferred_prompt,
            )
        )

    return summaries


def load_session_messages(project_path: Path, session_id: str) -> List[ConversationMessage]:
    """Load messages for a stored session."""
    path = _session_file(project_path, session_id)
    if not path.exists():
        return []

    messages: List[ConversationMessage] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    payload = data.get("payload") or {}
                    msg = _deserialize_message(payload)
                    if msg is not None and getattr(msg, "type", None) != "progress":
                        messages.append(msg)
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    logger.debug(
                        "[session_history] Failed to deserialize message in session %s: %s: %s",
                        session_id,
                        type(exc).__name__,
                        exc,
                    )
                    continue
    except (OSError, IOError) as exc:
        logger.warning(
            "Failed to load session messages: %s: %s",
            type(exc).__name__,
            exc,
            extra={"session_id": session_id, "path": str(path)},
        )
        return []

    return messages


__all__ = [
    "SessionHistory",
    "SessionSummary",
    "list_session_summaries",
    "load_session_messages",
]
