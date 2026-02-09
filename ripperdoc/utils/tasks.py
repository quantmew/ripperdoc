"""Persistent task-graph storage for Ripperdoc.

The new task system stores each task as an individual JSON document under
`~/.ripperdoc/tasks/<task_list_id>/`. Tasks support explicit dependency edges
via `blocks` and `blockedBy` and are safe for multi-process access through
file locking and atomic writes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Literal, Optional, Sequence
from uuid import uuid4

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError

from ripperdoc.utils.coerce import parse_boolish
from ripperdoc.utils.file_editing import file_lock
from ripperdoc.utils.log import get_logger
from ripperdoc.utils.path_utils import sanitize_project_path


logger = get_logger()

TaskStatus = Literal["pending", "in_progress", "completed"]
_RUNTIME_TASK_SCOPE: tuple[str, Path] | None = None


class TaskItem(BaseModel):
    """A persisted task node in the task graph."""

    id: str = Field(description="Unique task identifier")
    subject: str = Field(description="Short title for the task")
    description: str = Field(default="", description="Detailed task description")
    active_form: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("active_form", "activeForm"),
        serialization_alias="activeForm",
        description="Present-progress phrasing for status output",
    )
    owner: Optional[str] = Field(default=None, description="Assigned owner/teammate")
    status: TaskStatus = Field(default="pending")
    blocks: list[str] = Field(default_factory=list, description="Tasks this task blocks")
    blocked_by: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("blocked_by", "blockedBy"),
        serialization_alias="blockedBy",
        description="Tasks blocking this task",
    )
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


class TaskPatch(BaseModel):
    """Mutable fields accepted by task update operations."""

    subject: Optional[str] = None
    description: Optional[str] = None
    active_form: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("active_form", "activeForm"),
        serialization_alias="activeForm",
    )
    owner: Optional[str] = None
    status: Optional[TaskStatus] = None
    blocks: Optional[list[str]] = None
    blocked_by: Optional[list[str]] = Field(
        default=None,
        validation_alias=AliasChoices("blocked_by", "blockedBy"),
        serialization_alias="blockedBy",
    )
    metadata: Optional[dict[str, Any]] = None
    merge_metadata: bool = True

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


def is_task_system_enabled() -> bool:
    """Feature flag for the new persistent task graph system.

    Defaults to enabled. Set `RIPPERDOC_ENABLE_TASKS=false` to fall back to
    the legacy TodoRead/TodoWrite workflow.
    """

    return parse_boolish(os.getenv("RIPPERDOC_ENABLE_TASKS"), default=True)


def should_show_completed_tasks_in_ui() -> bool:
    """Whether UI task/todo panels should display completed entries.

    Defaults to False so active work stays focused. Set
    `RIPPERDOC_UI_SHOW_COMPLETED_TASKS=true` to include completed rows.
    """

    return parse_boolish(os.getenv("RIPPERDOC_UI_SHOW_COMPLETED_TASKS"), default=False)


def _config_root() -> Path:
    raw = os.getenv("RIPPERDOC_CONFIG_DIR")
    if raw and raw.strip():
        return Path(raw).expanduser()
    return Path.home() / ".ripperdoc"


def sanitize_identifier(value: str, *, fallback: str) -> str:
    """Normalize identifiers for stable filesystem storage."""

    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", (value or "").strip()).strip("-")
    return cleaned or fallback


def _resolve_project_root(project_root: Optional[Path]) -> Path:
    root = project_root or Path.cwd()
    return root.resolve()


def _session_scoped_task_list_id(project_root: Path, session_id: str) -> str:
    session_token = sanitize_identifier(session_id, fallback="session")
    project_digest = hashlib.sha1(str(project_root).encode("utf-8")).hexdigest()[:8]
    return sanitize_identifier(
        f"session-{project_digest}-{session_token}",
        fallback=f"session-{project_digest}",
    )


def set_runtime_task_scope(
    *,
    session_id: Optional[str],
    project_root: Optional[Path] = None,
) -> Optional[str]:
    """Bind per-process runtime task scope to a session.

    When set, task list resolution defaults to a session-scoped list for the
    matching project unless an explicit task-list override is configured.
    """

    global _RUNTIME_TASK_SCOPE

    if not session_id or not str(session_id).strip():
        _RUNTIME_TASK_SCOPE = None
        return None

    resolved_root = _resolve_project_root(project_root)
    clean_session_id = str(session_id).strip()
    _RUNTIME_TASK_SCOPE = (clean_session_id, resolved_root)
    return _session_scoped_task_list_id(resolved_root, clean_session_id)


def resolve_task_list_id(
    project_root: Optional[Path] = None,
    explicit_task_list_id: Optional[str] = None,
) -> str:
    """Resolve task-list identifier from explicit args/env/project context."""

    if explicit_task_list_id and explicit_task_list_id.strip():
        return sanitize_identifier(explicit_task_list_id, fallback="default")

    root = _resolve_project_root(project_root)

    for env_key in ("RIPPERDOC_TASK_LIST_ID",):
        env_val = os.getenv(env_key)
        if env_val and env_val.strip():
            return sanitize_identifier(env_val, fallback="default")

    runtime_scope = _RUNTIME_TASK_SCOPE
    if runtime_scope is not None:
        runtime_session_id, runtime_project_root = runtime_scope
        return _session_scoped_task_list_id(runtime_project_root, runtime_session_id)

    env_session_id = os.getenv("RIPPERDOC_SESSION_ID")
    if env_session_id and env_session_id.strip():
        return _session_scoped_task_list_id(root, env_session_id)

    return sanitize_identifier(sanitize_project_path(root), fallback="default")


def task_list_dir(
    project_root: Optional[Path] = None,
    task_list_id: Optional[str] = None,
    *,
    ensure: bool = False,
) -> Path:
    """Return storage directory for a task list."""

    resolved = resolve_task_list_id(project_root, explicit_task_list_id=task_list_id)
    directory = _config_root() / "tasks" / resolved
    if ensure:
        directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_task_list_dir(
    project_root: Optional[Path] = None,
    task_list_id: Optional[str] = None,
) -> Path:
    """Ensure task-list directory exists and return it."""

    return task_list_dir(project_root, task_list_id, ensure=True)


def _task_filename(task_id: str) -> str:
    safe = sanitize_identifier(task_id, fallback="task")
    digest = hashlib.sha1(task_id.encode("utf-8")).hexdigest()[:8]
    return f"{safe}-{digest}.json"


def _task_file_path(task_dir: Path, task_id: str) -> Path:
    return task_dir / _task_filename(task_id)


@contextmanager
def _task_list_lock(task_dir: Path) -> Iterator[None]:
    """Exclusive lock per task list to coordinate cross-process writes."""

    task_dir.mkdir(parents=True, exist_ok=True)
    lock_path = task_dir / ".lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        with file_lock(handle, exclusive=True):
            yield


def _write_json_atomic(path: Path, data: Dict[str, Any]) -> None:
    """Atomically write JSON content to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(data, indent=2, ensure_ascii=False)
    fd, temp_path = tempfile.mkstemp(dir=str(path.parent), prefix=".task_", suffix=".tmp")
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


def _read_task_file(path: Path) -> Optional[TaskItem]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, IOError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.warning(
            "[tasks] Failed reading task file: %s: %s",
            type(exc).__name__,
            exc,
            extra={"path": str(path)},
        )
        return None

    if not isinstance(payload, dict):
        return None

    try:
        return TaskItem(**payload)
    except ValidationError as exc:
        logger.warning(
            "[tasks] Invalid task file schema: %s",
            exc,
            extra={"path": str(path)},
        )
        return None


def _normalize_ids(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        raw = str(value or "").strip()
        if not raw or raw in seen:
            continue
        seen.add(raw)
        ordered.append(raw)
    return ordered


def _next_numeric_task_id(tasks: Dict[str, "TaskItem"]) -> str:
    numeric_ids = [int(task_id) for task_id in tasks if str(task_id).isdigit()]
    if not numeric_ids:
        return "1"
    return str(max(numeric_ids) + 1)


def _load_task_map(task_dir: Path) -> Dict[str, TaskItem]:
    tasks: Dict[str, TaskItem] = {}
    if not task_dir.exists():
        return tasks

    for path in sorted(task_dir.glob("*.json")):
        task = _read_task_file(path)
        if task is None:
            continue
        task.blocks = _normalize_ids(task.blocks)
        task.blocked_by = _normalize_ids(task.blocked_by)
        tasks[task.id] = task

    return tasks


def _save_tasks(task_dir: Path, tasks: Dict[str, TaskItem], task_ids: Iterable[str]) -> None:
    """Persist selected tasks (and delete missing files for removed ids)."""

    for task_id in task_ids:
        task = tasks.get(task_id)
        path = _task_file_path(task_dir, task_id)
        if task is None:
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning(
                    "[tasks] Failed removing task file: %s: %s",
                    type(exc).__name__,
                    exc,
                    extra={"path": str(path)},
                )
            continue
        _write_json_atomic(path, task.model_dump(by_alias=True))


def _validate_references(task: TaskItem, all_tasks: Dict[str, TaskItem]) -> None:
    for dep_id in [*task.blocks, *task.blocked_by]:
        if dep_id == task.id:
            raise ValueError("Task dependencies cannot reference self.")
        if dep_id not in all_tasks:
            raise ValueError(f"Dependency '{dep_id}' does not exist in this task list.")


def _reconcile_dependency_edges(
    *,
    tasks: Dict[str, TaskItem],
    task_id: str,
    old_blocks: set[str],
    old_blocked_by: set[str],
) -> set[str]:
    """Keep `blocks` and `blockedBy` relationships bidirectionally consistent."""

    changed: set[str] = {task_id}
    task = tasks[task_id]
    new_blocks = set(task.blocks)
    new_blocked_by = set(task.blocked_by)

    for target_id in old_blocks - new_blocks:
        target = tasks.get(target_id)
        if target and task_id in target.blocked_by:
            target.blocked_by = [val for val in target.blocked_by if val != task_id]
            target.updated_at = time.time()
            target.version += 1
            changed.add(target_id)

    for target_id in new_blocks:
        target = tasks.get(target_id)
        if target and task_id not in target.blocked_by:
            target.blocked_by.append(task_id)
            target.blocked_by = _normalize_ids(target.blocked_by)
            target.updated_at = time.time()
            target.version += 1
            changed.add(target_id)

    for target_id in old_blocked_by - new_blocked_by:
        target = tasks.get(target_id)
        if target and task_id in target.blocks:
            target.blocks = [val for val in target.blocks if val != task_id]
            target.updated_at = time.time()
            target.version += 1
            changed.add(target_id)

    for target_id in new_blocked_by:
        target = tasks.get(target_id)
        if target and task_id not in target.blocks:
            target.blocks.append(task_id)
            target.blocks = _normalize_ids(target.blocks)
            target.updated_at = time.time()
            target.version += 1
            changed.add(target_id)

    return changed


def list_tasks(
    *,
    project_root: Optional[Path] = None,
    task_list_id: Optional[str] = None,
) -> List[TaskItem]:
    """Load all tasks for the resolved task list."""

    directory = task_list_dir(project_root, task_list_id, ensure=False)
    tasks = _load_task_map(directory)
    return sorted(tasks.values(), key=lambda item: (item.created_at, item.id))


def get_task(
    task_id: str,
    *,
    project_root: Optional[Path] = None,
    task_list_id: Optional[str] = None,
) -> Optional[TaskItem]:
    """Load a single task by id."""

    for task in list_tasks(project_root=project_root, task_list_id=task_list_id):
        if task.id == task_id:
            return task
    return None


def create_task(
    *,
    subject: str,
    description: str = "",
    active_form: Optional[str] = None,
    owner: Optional[str] = None,
    status: TaskStatus = "pending",
    blocks: Optional[Sequence[str]] = None,
    blocked_by: Optional[Sequence[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    task_id: Optional[str] = None,
    project_root: Optional[Path] = None,
    task_list_id: Optional[str] = None,
) -> TaskItem:
    """Create and persist a task node."""

    title = (subject or "").strip()
    if not title:
        raise ValueError("subject is required")

    directory = task_list_dir(project_root, task_list_id, ensure=True)
    with _task_list_lock(directory):
        tasks = _load_task_map(directory)
        resolved_task_id = (task_id or _next_numeric_task_id(tasks) or f"task_{uuid4().hex[:8]}").strip()
        if resolved_task_id in tasks:
            raise ValueError(f"Task id '{resolved_task_id}' already exists.")

        now = time.time()
        task = TaskItem(
            id=resolved_task_id,
            subject=title,
            description=description or "",
            active_form=(active_form or None),
            owner=(owner or None),
            status=status,
            blocks=_normalize_ids(list(blocks or [])),
            blocked_by=_normalize_ids(list(blocked_by or [])),
            metadata=dict(metadata or {}),
            created_at=now,
            updated_at=now,
            version=1,
        )
        tasks[task.id] = task
        _validate_references(task, tasks)

        changed = _reconcile_dependency_edges(
            tasks=tasks,
            task_id=task.id,
            old_blocks=set(),
            old_blocked_by=set(),
        )
        _save_tasks(directory, tasks, changed)
        return task


def update_task(
    task_id: str,
    patch: TaskPatch,
    *,
    project_root: Optional[Path] = None,
    task_list_id: Optional[str] = None,
) -> TaskItem:
    """Update an existing task and persist graph consistency changes."""

    directory = task_list_dir(project_root, task_list_id, ensure=True)
    with _task_list_lock(directory):
        tasks = _load_task_map(directory)
        existing = tasks.get(task_id)
        if existing is None:
            raise ValueError(f"Task '{task_id}' not found.")

        previous_blocks = set(existing.blocks)
        previous_blocked_by = set(existing.blocked_by)

        patch_data = patch.model_dump(exclude_unset=True, by_alias=False)
        merge_metadata = bool(patch_data.pop("merge_metadata", True))

        if "subject" in patch_data and patch_data["subject"] is not None:
            subject_val = str(patch_data["subject"]).strip()
            if not subject_val:
                raise ValueError("subject cannot be empty")
            existing.subject = subject_val

        if "description" in patch_data and patch_data["description"] is not None:
            existing.description = str(patch_data["description"])

        if "active_form" in patch_data:
            active_val = patch_data.get("active_form")
            existing.active_form = str(active_val).strip() if active_val else None

        if "owner" in patch_data:
            owner_val = patch_data.get("owner")
            existing.owner = str(owner_val).strip() if owner_val else None

        if "status" in patch_data and patch_data["status"] is not None:
            existing.status = patch_data["status"]

        if "blocks" in patch_data and patch_data["blocks"] is not None:
            existing.blocks = _normalize_ids(list(patch_data["blocks"]))

        if "blocked_by" in patch_data and patch_data["blocked_by"] is not None:
            existing.blocked_by = _normalize_ids(list(patch_data["blocked_by"]))

        if "metadata" in patch_data and patch_data["metadata"] is not None:
            incoming_metadata = dict(patch_data["metadata"])
            if merge_metadata:
                merged = dict(existing.metadata)
                for key, value in incoming_metadata.items():
                    if value is None:
                        merged.pop(key, None)
                    else:
                        merged[key] = value
                existing.metadata = merged
            else:
                existing.metadata = {
                    key: value for key, value in incoming_metadata.items() if value is not None
                }

        _validate_references(existing, tasks)
        existing.updated_at = time.time()
        existing.version += 1

        changed = _reconcile_dependency_edges(
            tasks=tasks,
            task_id=task_id,
            old_blocks=previous_blocks,
            old_blocked_by=previous_blocked_by,
        )
        _save_tasks(directory, tasks, changed)

        return existing


def delete_task(
    task_id: str,
    *,
    project_root: Optional[Path] = None,
    task_list_id: Optional[str] = None,
) -> bool:
    """Delete a task and reconcile dependency edges in remaining tasks."""

    directory = task_list_dir(project_root, task_list_id, ensure=True)
    with _task_list_lock(directory):
        tasks = _load_task_map(directory)
        if task_id not in tasks:
            return False

        tasks.pop(task_id, None)
        changed: set[str] = {task_id}
        for candidate in tasks.values():
            before_blocks = list(candidate.blocks)
            before_blocked_by = list(candidate.blocked_by)
            candidate.blocks = [dep for dep in candidate.blocks if dep != task_id]
            candidate.blocked_by = [dep for dep in candidate.blocked_by if dep != task_id]
            if candidate.blocks != before_blocks or candidate.blocked_by != before_blocked_by:
                candidate.updated_at = time.time()
                candidate.version += 1
                changed.add(candidate.id)

        _save_tasks(directory, tasks, changed)
        return True


def unresolved_blockers(task: TaskItem, tasks: Sequence[TaskItem]) -> list[str]:
    """Return blocker ids that still exist and are not completed."""

    by_id = {item.id: item for item in tasks}
    unresolved: list[str] = []
    for blocker_id in task.blocked_by:
        blocker = by_id.get(blocker_id)
        if blocker and blocker.status != "completed":
            unresolved.append(blocker_id)
    return unresolved


def get_next_actionable_task(tasks: Sequence[TaskItem]) -> Optional[TaskItem]:
    """Pick next task (in_progress first, then pending) that is not dependency-blocked."""

    by_id = {task.id: task for task in tasks}

    def _blocked(task: TaskItem) -> bool:
        for blocker_id in task.blocked_by:
            blocker = by_id.get(blocker_id)
            if blocker and blocker.status != "completed":
                return True
        return False

    for status in ("in_progress", "pending"):
        for task in tasks:
            if task.status == status and not _blocked(task):
                return task
    return None


def summarize_tasks(tasks: Sequence[TaskItem]) -> Dict[str, Any]:
    """Return aggregate stats for a task list."""

    statuses = {
        "pending": 0,
        "in_progress": 0,
        "completed": 0,
    }
    for task in tasks:
        statuses[task.status] = statuses.get(task.status, 0) + 1

    owners: Dict[str, int] = {}
    for task in tasks:
        key = task.owner or "unassigned"
        owners[key] = owners.get(key, 0) + 1

    return {
        "total": len(tasks),
        "by_status": statuses,
        "by_owner": owners,
    }


def format_task_summary(tasks: Sequence[TaskItem]) -> str:
    stats = summarize_tasks(tasks)
    return (
        f"Tasks updated (total {stats['total']}; "
        f"{stats['by_status'].get('pending', 0)} pending, "
        f"{stats['by_status'].get('in_progress', 0)} in progress, "
        f"{stats['by_status'].get('completed', 0)} completed)."
    )


def format_task_lines(tasks: Sequence[TaskItem]) -> List[str]:
    status_marker = {
        "completed": "●",
        "in_progress": "◐",
        "pending": "○",
    }
    lines: List[str] = []
    for task in tasks:
        owner = f" @{task.owner}" if task.owner else ""
        lines.append(f"{status_marker.get(task.status, '○')} {task.subject}{owner} [id: {task.id}]")
    return lines


__all__ = [
    "TaskItem",
    "TaskPatch",
    "TaskStatus",
    "create_task",
    "delete_task",
    "ensure_task_list_dir",
    "format_task_lines",
    "format_task_summary",
    "get_next_actionable_task",
    "get_task",
    "is_task_system_enabled",
    "list_tasks",
    "resolve_task_list_id",
    "sanitize_identifier",
    "set_runtime_task_scope",
    "should_show_completed_tasks_in_ui",
    "summarize_tasks",
    "task_list_dir",
    "unresolved_blockers",
    "update_task",
]
