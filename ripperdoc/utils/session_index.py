"""Session sidecar index for fast resume/stats lookups."""

from __future__ import annotations

import json
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from ripperdoc.utils.log import get_logger
from ripperdoc.utils.path_utils import project_storage_dir

logger = get_logger()

INDEX_VERSION = 1
INDEX_FILENAME = ".session_index_v1.json"

_LOGGED_AT_RE = re.compile(r'"logged_at"\s*:\s*"([^"]+)"')


@dataclass
class SessionIndexEntry:
    """Aggregated metadata and usage for one session file."""

    session_id: str
    message_count: int = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    last_prompt: str = ""
    last_long_prompt: str = ""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_creation_tokens: int = 0
    total_cost_usd: float = 0.0
    model_usage: Dict[str, int] = field(default_factory=dict)
    file_size: int = 0
    file_mtime_ns: int = 0

    @property
    def preferred_prompt(self) -> str:
        prompt = self.last_long_prompt or self.last_prompt
        return prompt if prompt else "(no prompt)"


@dataclass
class SessionIndex:
    """Project-level sidecar index for session summaries and stats."""

    project_path: str
    sessions: Dict[str, SessionIndexEntry] = field(default_factory=dict)
    version: int = INDEX_VERSION
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def _sessions_root() -> Path:
    return Path.home() / ".ripperdoc" / "sessions"


def _session_dir(project_path: Path, ensure: bool = False) -> Path:
    return project_storage_dir(_sessions_root(), project_path, ensure=ensure)


def _index_path(project_path: Path) -> Path:
    directory = _session_dir(project_path, ensure=True)
    return directory / INDEX_FILENAME


def _lock_path(index_path: Path) -> Path:
    return index_path.with_suffix(index_path.suffix + ".lock")


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _to_utc_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _from_iso(raw: object) -> Optional[datetime]:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _coerce_int(value: object) -> int:
    try:
        if isinstance(value, bool):
            return int(value)
        if value is None:
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def _coerce_float(value: object) -> float:
    try:
        if isinstance(value, bool):
            return float(int(value))
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _extract_prompt(payload: dict) -> str:
    """Pull a short preview of the first user text block."""
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


def _detect_payload_type(line: str) -> Optional[str]:
    payload_idx = line.find('"payload"')
    if payload_idx < 0:
        return None
    head = line[payload_idx : payload_idx + 220]
    if '"type":"assistant"' in head or '"type": "assistant"' in head:
        return "assistant"
    if '"type":"user"' in head or '"type": "user"' in head:
        return "user"
    return None


def _parse_logged_at_from_line(line: str) -> Optional[datetime]:
    match = _LOGGED_AT_RE.search(line[:240])
    if not match:
        return None
    return _from_iso(match.group(1))


def _entry_from_dict(session_id: str, data: dict) -> SessionIndexEntry:
    model_usage_raw = data.get("model_usage") or {}
    model_usage: Dict[str, int] = {}
    if isinstance(model_usage_raw, dict):
        for model_name, count in model_usage_raw.items():
            if isinstance(model_name, str):
                model_usage[model_name] = _coerce_int(count)

    return SessionIndexEntry(
        session_id=session_id,
        message_count=_coerce_int(data.get("message_count")),
        created_at=_from_iso(data.get("created_at")),
        updated_at=_from_iso(data.get("updated_at")),
        last_prompt=str(data.get("last_prompt") or ""),
        last_long_prompt=str(data.get("last_long_prompt") or ""),
        total_input_tokens=_coerce_int(data.get("total_input_tokens")),
        total_output_tokens=_coerce_int(data.get("total_output_tokens")),
        total_cache_read_tokens=_coerce_int(data.get("total_cache_read_tokens")),
        total_cache_creation_tokens=_coerce_int(data.get("total_cache_creation_tokens")),
        total_cost_usd=_coerce_float(data.get("total_cost_usd")),
        model_usage=model_usage,
        file_size=_coerce_int(data.get("file_size")),
        file_mtime_ns=_coerce_int(data.get("file_mtime_ns")),
    )


def _entry_to_dict(entry: SessionIndexEntry) -> dict:
    return {
        "message_count": entry.message_count,
        "created_at": _to_utc_iso(entry.created_at) if entry.created_at else None,
        "updated_at": _to_utc_iso(entry.updated_at) if entry.updated_at else None,
        "last_prompt": entry.last_prompt,
        "last_long_prompt": entry.last_long_prompt,
        "total_input_tokens": entry.total_input_tokens,
        "total_output_tokens": entry.total_output_tokens,
        "total_cache_read_tokens": entry.total_cache_read_tokens,
        "total_cache_creation_tokens": entry.total_cache_creation_tokens,
        "total_cost_usd": entry.total_cost_usd,
        "model_usage": entry.model_usage,
        "file_size": entry.file_size,
        "file_mtime_ns": entry.file_mtime_ns,
    }


def _read_index_file(index_path: Path, resolved_project: str) -> Optional[SessionIndex]:
    if not index_path.exists():
        return None
    try:
        with index_path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, IOError, json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning(
            "[session_index] Failed to read sidecar index: %s: %s",
            type(exc).__name__,
            exc,
            extra={"path": str(index_path)},
        )
        return None

    if not isinstance(raw, dict):
        return None
    if _coerce_int(raw.get("version")) != INDEX_VERSION:
        return None

    project_path = str(raw.get("project_path") or resolved_project)
    if project_path != resolved_project:
        # Ignore mismatched content (e.g. stale copied index file).
        return None

    sessions_raw = raw.get("sessions") or {}
    sessions: Dict[str, SessionIndexEntry] = {}
    if isinstance(sessions_raw, dict):
        for session_id, entry_data in sessions_raw.items():
            if not isinstance(session_id, str) or not isinstance(entry_data, dict):
                continue
            sessions[session_id] = _entry_from_dict(session_id, entry_data)

    generated_at = _from_iso(raw.get("generated_at")) or _now_utc()
    return SessionIndex(
        project_path=project_path,
        sessions=sessions,
        version=INDEX_VERSION,
        generated_at=generated_at,
    )


def _write_index_file(index_path: Path, index: SessionIndex) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = index_path.with_suffix(index_path.suffix + ".tmp")

    payload = {
        "version": INDEX_VERSION,
        "project_path": index.project_path,
        "generated_at": _to_utc_iso(index.generated_at),
        "sessions": {
            session_id: _entry_to_dict(entry)
            for session_id, entry in sorted(index.sessions.items(), key=lambda item: item[0])
        },
    }
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
        fh.write("\n")
    tmp_path.replace(index_path)


@contextmanager
def _locked(lock_file: Path) -> Iterator[None]:
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    with lock_file.open("a+", encoding="utf-8") as fh:
        fcntl = None
        try:
            import fcntl as _fcntl

            fcntl = _fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError, AttributeError):
            fcntl = None
        try:
            yield
        finally:
            if fcntl is not None:
                try:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass


def _update_entry_from_payload(
    entry: SessionIndexEntry, payload: dict, logged_at: Optional[datetime]
) -> None:
    payload_type = payload.get("type")
    if payload_type not in ("user", "assistant"):
        return

    if logged_at is not None:
        if entry.created_at is None or logged_at < entry.created_at:
            entry.created_at = logged_at
        if entry.updated_at is None or logged_at > entry.updated_at:
            entry.updated_at = logged_at

    entry.message_count += 1

    if payload_type == "user":
        prompt = _extract_prompt(payload)
        if prompt:
            entry.last_prompt = prompt
            if len(prompt) > 10:
                entry.last_long_prompt = prompt
        return

    model_name = payload.get("model")
    if isinstance(model_name, str) and model_name:
        entry.model_usage[model_name] = entry.model_usage.get(model_name, 0) + 1

    entry.total_input_tokens += _coerce_int(payload.get("input_tokens"))
    entry.total_output_tokens += _coerce_int(payload.get("output_tokens"))
    entry.total_cache_read_tokens += _coerce_int(payload.get("cache_read_tokens"))
    entry.total_cache_creation_tokens += _coerce_int(payload.get("cache_creation_tokens"))
    entry.total_cost_usd += _coerce_float(payload.get("cost_usd"))


def _scan_session_file(path: Path, session_id: str) -> Optional[SessionIndexEntry]:
    try:
        stat = path.stat()
    except OSError:
        return None

    entry = SessionIndexEntry(session_id=session_id)
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                payload_type = _detect_payload_type(line)
                if payload_type not in ("user", "assistant"):
                    continue

                logged_at = _parse_logged_at_from_line(line)
                if logged_at is not None:
                    if entry.created_at is None or logged_at < entry.created_at:
                        entry.created_at = logged_at
                    if entry.updated_at is None or logged_at > entry.updated_at:
                        entry.updated_at = logged_at

                entry.message_count += 1

                if payload_type == "assistant":
                    try:
                        data = json.loads(line)
                        payload = data.get("payload") or {}
                    except (json.JSONDecodeError, TypeError, ValueError):
                        continue
                    if payload.get("type") != "assistant":
                        continue
                    model_name = payload.get("model")
                    if isinstance(model_name, str) and model_name:
                        entry.model_usage[model_name] = entry.model_usage.get(model_name, 0) + 1
                    entry.total_input_tokens += _coerce_int(payload.get("input_tokens"))
                    entry.total_output_tokens += _coerce_int(payload.get("output_tokens"))
                    entry.total_cache_read_tokens += _coerce_int(payload.get("cache_read_tokens"))
                    entry.total_cache_creation_tokens += _coerce_int(payload.get("cache_creation_tokens"))
                    entry.total_cost_usd += _coerce_float(payload.get("cost_usd"))
                    continue

                # user payload
                if '"tool_use_result":{' in line or '"tool_use_result": {' in line:
                    # Large tool payloads can be huge; they never carry resumable prompts.
                    continue
                try:
                    data = json.loads(line)
                    payload = data.get("payload") or {}
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
                if payload.get("type") != "user":
                    continue
                prompt = _extract_prompt(payload)
                if prompt:
                    entry.last_prompt = prompt
                    if len(prompt) > 10:
                        entry.last_long_prompt = prompt
    except (OSError, IOError):
        return None

    if entry.message_count <= 0:
        return None

    if entry.created_at is None:
        entry.created_at = datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc)
    if entry.updated_at is None:
        entry.updated_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
    entry.file_size = stat.st_size
    entry.file_mtime_ns = stat.st_mtime_ns
    return entry


def _current_file_states(directory: Path) -> Dict[str, Tuple[Path, int, int]]:
    states: Dict[str, Tuple[Path, int, int]] = {}
    if not directory.exists():
        return states
    for path in directory.glob("*.jsonl"):
        try:
            stat = path.stat()
        except OSError:
            continue
        states[path.stem] = (path, stat.st_mtime_ns, stat.st_size)
    return states


def load_or_build_session_index(project_path: Path) -> SessionIndex:
    """Load sidecar index and reconcile only changed/deleted session files."""
    resolved_project = str(project_path.resolve())
    index_path = _index_path(project_path)
    directory = _session_dir(project_path, ensure=False)

    with _locked(_lock_path(index_path)):
        index_exists = index_path.exists()
        index = _read_index_file(index_path, resolved_project)
        if index is None:
            index = SessionIndex(project_path=resolved_project)
        else:
            index.project_path = resolved_project

        states = _current_file_states(directory)
        changed_sessions: List[str] = []
        removed_sessions: List[str] = []

        for session_id, (path, mtime_ns, size) in states.items():
            cached = index.sessions.get(session_id)
            if cached is None:
                changed_sessions.append(session_id)
                continue
            if cached.file_mtime_ns != mtime_ns or cached.file_size != size:
                changed_sessions.append(session_id)

        for session_id in list(index.sessions):
            if session_id not in states:
                removed_sessions.append(session_id)

        changed = False
        if removed_sessions:
            changed = True
            for session_id in removed_sessions:
                index.sessions.pop(session_id, None)

        if changed_sessions:
            changed = True
            for session_id in changed_sessions:
                path, _, _ = states[session_id]
                scanned = _scan_session_file(path, session_id)
                if scanned is None:
                    index.sessions.pop(session_id, None)
                    continue
                index.sessions[session_id] = scanned

        if changed or not index_exists:
            index.generated_at = _now_utc()
            _write_index_file(index_path, index)

        return index


def list_session_index_entries(project_path: Path) -> List[SessionIndexEntry]:
    """Return sidecar entries sorted by most-recent update time."""
    index = load_or_build_session_index(project_path)

    def _sort_key(entry: SessionIndexEntry) -> datetime:
        if entry.updated_at is not None:
            return entry.updated_at
        if entry.created_at is not None:
            return entry.created_at
        return datetime.fromtimestamp(0, tz=timezone.utc)

    return sorted(index.sessions.values(), key=_sort_key, reverse=True)


def update_session_index_for_payload(
    project_path: Path,
    session_id: str,
    payload: dict,
    *,
    logged_at: Optional[str],
    session_file: Path,
) -> None:
    """Apply one message append to the sidecar index."""
    payload_type = payload.get("type")
    if payload_type not in ("user", "assistant"):
        return

    resolved_project = str(project_path.resolve())
    index_path = _index_path(project_path)
    with _locked(_lock_path(index_path)):
        index = _read_index_file(index_path, resolved_project)
        if index is None:
            index = SessionIndex(project_path=resolved_project)

        entry = index.sessions.get(session_id)
        if entry is None:
            entry = SessionIndexEntry(session_id=session_id)
            index.sessions[session_id] = entry

        logged_at_dt = _from_iso(logged_at)
        _update_entry_from_payload(entry, payload, logged_at_dt)

        try:
            stat = session_file.stat()
            entry.file_size = stat.st_size
            entry.file_mtime_ns = stat.st_mtime_ns
        except OSError:
            pass

        if entry.created_at is None:
            entry.created_at = logged_at_dt or _now_utc()
        if entry.updated_at is None:
            entry.updated_at = logged_at_dt or _now_utc()

        index.generated_at = _now_utc()
        _write_index_file(index_path, index)


__all__ = [
    "SessionIndexEntry",
    "SessionIndex",
    "INDEX_VERSION",
    "INDEX_FILENAME",
    "list_session_index_entries",
    "load_or_build_session_index",
    "update_session_index_for_payload",
]
