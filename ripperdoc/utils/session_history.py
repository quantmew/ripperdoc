"""Session log storage and retrieval."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from ripperdoc.utils.log import get_logger
from ripperdoc.utils.messages import (
    AssistantMessage,
    ProgressMessage,
    UserMessage,
)
from ripperdoc.utils.path_utils import project_storage_dir


logger = get_logger()

ConversationMessage = UserMessage | AssistantMessage | ProgressMessage


@dataclass
class SessionSummary:
    session_id: str
    path: Path
    message_count: int
    created_at: datetime
    updated_at: datetime
    last_prompt: str


def _sessions_root() -> Path:
    return Path.home() / ".ripperdoc" / "sessions"


def _session_file(project_path: Path, session_id: str) -> Path:
    directory = project_storage_dir(_sessions_root(), project_path, ensure=True)
    return directory / f"{session_id}.jsonl"


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _extract_prompt(payload: dict) -> str:
    """Pull a short preview of the first user message."""
    if payload.get("type") != "user":
        return ""
    message = payload.get("message") or {}
    content = message.get("content")
    preview = ""
    if isinstance(content, str):
        preview = content
    elif isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and block.get("text"):
                preview = str(block["text"])
                break
    preview = (preview or "").replace("\n", " ").strip()
    if len(preview) > 80:
        preview = preview[:77] + "..."
    return preview


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
        entry = {
            "logged_at": _now_iso(),
            "project_path": str(self.project_path.resolve()),
            "payload": payload,
        }
        try:
            with self.path.open("a", encoding="utf-8") as fh:
                # ensure_ascii=False 避免中文等字符被转义为 \uXXXX
                # separators 去掉多余空格，减小体积
                json.dump(entry, fh, ensure_ascii=False, separators=(",", ":"))
                fh.write("\n")
            if isinstance(msg_uuid, str):
                self._seen_ids.add(msg_uuid)
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

    current_project = str(project_path.resolve())
    summaries: List[SessionSummary] = []
    for jsonl_path in directory.glob("*.jsonl"):
        try:
            with jsonl_path.open("r", encoding="utf-8") as fh:
                messages = [json.loads(line) for line in fh if line.strip()]
        except (OSError, IOError, json.JSONDecodeError) as exc:
            logger.warning(
                "Failed to load session summary: %s: %s",
                type(exc).__name__,
                exc,
                extra={"path": str(jsonl_path)},
            )
            continue

        # Check if this session belongs to the current project
        # If any message has a project_path field, use it to verify
        session_project_path = None
        for entry in messages:
            entry_path = entry.get("project_path")
            if entry_path:
                session_project_path = entry_path
                break

        # Skip sessions that belong to a different project
        if session_project_path and session_project_path != current_project:
            continue

        payloads = [entry.get("payload") or {} for entry in messages]
        conversation_payloads = [
            payload for payload in payloads if payload.get("type") in ("user", "assistant")
        ]
        if not conversation_payloads:
            continue

        created_raw = messages[0].get("logged_at")
        updated_raw = messages[-1].get("logged_at")
        created_at = (
            datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
            if isinstance(created_raw, str)
            else datetime.fromtimestamp(jsonl_path.stat().st_ctime)
        )
        updated_at = (
            datetime.fromisoformat(updated_raw.replace("Z", "+00:00"))
            if isinstance(updated_raw, str)
            else datetime.fromtimestamp(jsonl_path.stat().st_mtime)
        )
        # Extract last user prompt with more than 10 characters
        # If not found, fall back to any user prompt
        last_prompt = ""
        fallback_prompt = ""
        for payload in reversed(conversation_payloads):
            if payload.get("type") != "user":
                continue
            prompt = _extract_prompt(payload)
            if not prompt:
                continue
            if not fallback_prompt:
                fallback_prompt = prompt
            if len(prompt) > 10:
                last_prompt = prompt
                break
        summaries.append(
            SessionSummary(
                session_id=jsonl_path.stem,
                path=jsonl_path,
                message_count=len(conversation_payloads),
                created_at=created_at,
                updated_at=updated_at,
                last_prompt=last_prompt or fallback_prompt or "(no prompt)",
            )
        )

    return sorted(summaries, key=lambda s: s.updated_at, reverse=True)


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
