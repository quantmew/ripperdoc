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
from typing import Any, Callable, Dict, Iterable, Iterator, List, Literal, Optional, Sequence, Set, Tuple
from uuid import uuid4

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError

from ripperdoc.utils.filesystem.config_paths import user_config_dir
from ripperdoc.utils.coerce import parse_boolish
from ripperdoc.utils.file_editing import file_lock
from ripperdoc.utils.log import get_logger


logger = get_logger()

TaskStatus = Literal["pending", "in_progress", "completed"]
_RUNTIME_TASK_SCOPE: Optional[Tuple[str, Path]] = None

_HIGH_WATER_MARK_FILE = ".highwatermark"


class ClaimTaskResult:
    """Result of attempting to claim a task atomically."""
    success: bool
    reason: Optional[str] = None
    task: Optional["TaskItem"] = None
    busy_with_tasks: List[str] = []
    blocked_by_tasks: List[str] = []

    def __init__(
        self,
        success: bool,
        reason: Optional[str] = None,
        task: Optional["TaskItem"] = None,
        busy_with_tasks: Optional[List[str]] = None,
        blocked_by_tasks: Optional[List[str]] = None,
    ):
        self.success = success
        self.reason = reason
        self.task = task
        self.busy_with_tasks = busy_with_tasks or []
        self.blocked_by_tasks = blocked_by_tasks or []


class AgentStatus:
    """Agent status based on task ownership."""
    agent_id: str
    name: str
    agent_type: Optional[str] = None
    status: Literal["idle", "busy"]
    current_tasks: List[str]

    def __init__(
        self,
        agent_id: str,
        name: str,
        status: Literal["idle", "busy"],
        current_tasks: Optional[List[str]] = None,
        agent_type: Optional[str] = None,
    ):
        self.agent_id = agent_id
        self.name = name
        self.agent_type = agent_type
        self.status = status
        self.current_tasks = current_tasks or []


class UnassignTasksResult:
    """Result of unassigning tasks from a teammate."""
    unassigned_tasks: List[Dict[str, str]]
    notification_message: str

    def __init__(
        self,
        unassigned_tasks: List[Dict[str, str]],
        notification_message: str,
    ):
        self.unassigned_tasks = unassigned_tasks
        self.notification_message = notification_message


# Simple observer/event pattern for tasks-updated notifications
_tasks_updated_listeners: List[Callable[..., Any]] = []


def on_tasks_updated(callback: Callable[[], None]) -> Callable[[], None]:
    """Register a listener for task updates. Returns unsubscribe function."""
    _tasks_updated_listeners.append(callback)
    def unsubscribe() -> None:
        if callback in _tasks_updated_listeners:
            _tasks_updated_listeners.remove(callback)
    return unsubscribe


def notify_tasks_updated() -> None:
    """Notify listeners that tasks have been updated."""
    for cb in list(_tasks_updated_listeners):
        try:
            cb()
        except Exception:
            pass
_PROCESS_DEFAULT_SESSION_ID: str = uuid4().hex[:12]


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
    blocks: List[str] = Field(default_factory=list, description="Tasks this task blocks")
    blocked_by: List[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("blocked_by", "blockedBy"),
        serialization_alias="blockedBy",
        description="Tasks blocking this task",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)
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
    blocks: Optional[List[str]] = None
    blocked_by: Optional[List[str]] = Field(
        default=None,
        validation_alias=AliasChoices("blocked_by", "blockedBy"),
        serialization_alias="blockedBy",
    )
    metadata: Optional[Dict[str, Any]] = None
    merge_metadata: bool = True

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


def is_task_system_enabled() -> bool:
    """Feature flag for the new persistent task graph system.

    Defaults to enabled. Set `RIPPERDOC_ENABLE_TASKS=false` to disable
    task graph tools and use TodoRead/TodoWrite instead.
    """
    return parse_boolish(os.getenv("RIPPERDOC_ENABLE_TASKS"), default=True)


def should_show_completed_tasks_in_ui() -> bool:
    """Whether UI task/todo panels should display completed entries.

    Defaults to False so active work stays focused. Set
    `RIPPERDOC_UI_SHOW_COMPLETED_TASKS=true` to include completed rows.
    """

    return parse_boolish(os.getenv("RIPPERDOC_UI_SHOW_COMPLETED_TASKS"), default=False)


def _config_root() -> Path:
    return user_config_dir()


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

    leader_name = get_leader_team_name()
    if leader_name:
        return sanitize_identifier(leader_name, fallback="default")

    return _session_scoped_task_list_id(root, _PROCESS_DEFAULT_SESSION_ID)


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
    return f"{sanitize_identifier(task_id, fallback='task')}.json"


def _high_water_mark_path(task_dir: Path) -> Path:
    return task_dir / _HIGH_WATER_MARK_FILE


def _read_high_water_mark(task_dir: Path) -> int:
    try:
        content = _high_water_mark_path(task_dir).read_text(encoding="utf-8").strip()
        value = int(content)
        return value if value > 0 else 0
    except (OSError, ValueError):
        return 0


def _write_high_water_mark(task_dir: Path, value: int) -> None:
    _high_water_mark_path(task_dir).write_text(str(value), encoding="utf-8")


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


def _normalize_ids(values: Sequence[str]) -> List[str]:
    seen: Set[str] = set()
    ordered: List[str] = []
    for value in values:
        raw = str(value or "").strip()
        if not raw or raw in seen:
            continue
        seen.add(raw)
        ordered.append(raw)
    return ordered


def _next_numeric_task_id(task_dir: Path, tasks: Dict[str, "TaskItem"]) -> str:
    from_files = max(
        (int(tid) for tid in tasks if str(tid).isdigit()),
        default=0,
    )
    from_mark = _read_high_water_mark(task_dir)
    return str(max(from_files, from_mark) + 1)


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
    old_blocks: Set[str],
    old_blocked_by: Set[str],
) -> Set[str]:
    """Keep `blocks` and `blockedBy` relationships bidirectionally consistent."""

    changed: Set[str] = {task_id}
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
        resolved_task_id = (task_id or _next_numeric_task_id(directory, tasks)).strip()
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
        notify_tasks_updated()
        return task


def block_task(
    from_task_id: str,
    to_task_id: str,
    *,
    project_root: Optional[Path] = None,
    task_list_id: Optional[str] = None,
) -> bool:
    """Establish a dependency: from_task blocks to_task (to_task blockedBy from_task)."""

    directory = task_list_dir(project_root, task_list_id, ensure=True)
    with _task_list_lock(directory):
        tasks = _load_task_map(directory)
        from_task = tasks.get(from_task_id)
        to_task = tasks.get(to_task_id)
        if from_task is None or to_task is None:
            return False

        changed: Set[str] = set()

        if to_task_id not in from_task.blocks:
            from_task.blocks = _normalize_ids([*from_task.blocks, to_task_id])
            from_task.version += 1
            from_task.updated_at = time.time()
            changed.add(from_task_id)

        if from_task_id not in to_task.blocked_by:
            to_task.blocked_by = _normalize_ids([*to_task.blocked_by, from_task_id])
            to_task.version += 1
            to_task.updated_at = time.time()
            changed.add(to_task_id)

        _save_tasks(directory, tasks, changed)
        notify_tasks_updated()
        return True


def claim_task(
    task_id: str,
    claimant_agent_id: str,
    *,
    project_root: Optional[Path] = None,
    task_list_id: Optional[str] = None,
    check_agent_busy: bool = False,
) -> ClaimTaskResult:
    """Atomically claim a task. If check_agent_busy, also verifies claimant
    doesn't already own another open task."""

    directory = task_list_dir(project_root, task_list_id, ensure=True)
    with _task_list_lock(directory):
        tasks = _load_task_map(directory)
        task = tasks.get(task_id)
        if task is None:
            return ClaimTaskResult(success=False, reason="task_not_found")

        if task.owner and task.owner != claimant_agent_id:
            return ClaimTaskResult(success=False, reason="already_claimed", task=task)

        if task.status == "completed":
            return ClaimTaskResult(success=False, reason="already_resolved", task=task)

        unresolved_ids = {t.id for t in tasks.values() if t.status != "completed"}
        blocked = [b for b in task.blocked_by if b in unresolved_ids]
        if blocked:
            return ClaimTaskResult(
                success=False, reason="blocked", task=task, blocked_by_tasks=blocked,
            )

        if check_agent_busy:
            agent_open = [
                t for t in tasks.values()
                if t.status != "completed"
                and t.owner == claimant_agent_id
                and t.id != task_id
            ]
            if agent_open:
                return ClaimTaskResult(
                    success=False,
                    reason="agent_busy",
                    task=task,
                    busy_with_tasks=[t.id for t in agent_open],
                )

        task.owner = claimant_agent_id
        task.version += 1
        task.updated_at = time.time()
        _save_tasks(directory, tasks, {task_id})
        notify_tasks_updated()
        return ClaimTaskResult(success=True, task=task)


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
        notify_tasks_updated()

        return existing


def get_agent_statuses(
    team_name: str,
) -> Optional[List[AgentStatus]]:
    """Get idle/busy status for all agents in a team based on task ownership."""
    from ripperdoc.utils.collaboration.teams import get_team

    team = get_team(team_name)
    if team is None:
        return None

    task_list_id = sanitize_identifier(team_name, fallback="default")
    all_tasks = list_tasks(task_list_id=task_list_id)

    unresolved_by_owner: Dict[str, List[str]] = {}
    for task in all_tasks:
        if task.status != "completed" and task.owner:
            unresolved_by_owner.setdefault(task.owner, []).append(task.id)

    results: List[AgentStatus] = []
    for member in team.members:
        name_tasks = unresolved_by_owner.get(member.name, [])
        agent_id = member.agent_id or member.name
        agent_id_tasks = unresolved_by_owner.get(agent_id, [])
        current = list({*name_tasks, *agent_id_tasks})
        results.append(AgentStatus(
            agent_id=agent_id,
            name=member.name,
            agent_type=member.agent_type,
            status="idle" if not current else "busy",
            current_tasks=current,
        ))
    return results


def unassign_teammate_tasks(
    team_name: str,
    teammate_id: str,
    teammate_name: str,
    reason: Literal["terminated", "shutdown"],
) -> UnassignTasksResult:
    """Unassign all open tasks from a teammate and reset them to pending."""

    task_list_id = sanitize_identifier(team_name, fallback="default")
    all_tasks = list_tasks(task_list_id=task_list_id)

    resolved: List[Dict[str, str]] = []
    for task in all_tasks:
        if task.status == "completed":
            continue
        if task.owner == teammate_id or task.owner == teammate_name:
            update_task(
                task.id,
                TaskPatch(owner=None, status="pending"),
                task_list_id=task_list_id,
            )
            resolved.append({"id": task.id, "subject": task.subject})

    action_verb = "was terminated" if reason == "terminated" else "has shut down"
    notification = f"{teammate_name} {action_verb}."
    if resolved:
        task_list_str = ", ".join(f'#{t["id"]} "{t["subject"]}"' for t in resolved)
        notification += (
            f" {len(resolved)} task(s) were unassigned: {task_list_str}."
            " Use TaskList to check availability and TaskUpdate with owner"
            " to reassign them to idle teammates."
        )

    return UnassignTasksResult(
        unassigned_tasks=resolved,
        notification_message=notification,
    )


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
        numeric_id = int(task_id) if str(task_id).isdigit() else None
        if numeric_id is not None:
            current_mark = _read_high_water_mark(directory)
            if numeric_id > current_mark:
                _write_high_water_mark(directory, numeric_id)
        changed: Set[str] = {task_id}
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
        notify_tasks_updated()
        return True


def unresolved_blockers(task: TaskItem, tasks: Sequence[TaskItem]) -> List[str]:
    """Return blocker ids that still exist and are not completed."""

    by_id = {item.id: item for item in tasks}
    unresolved: List[str] = []
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
    """Return aggregate stats for a task list (only active tasks)."""

    active = [t for t in tasks if t.status != "completed"]
    statuses = {
        "pending": 0,
        "in_progress": 0,
    }
    for task in active:
        statuses[task.status] = statuses.get(task.status, 0) + 1

    owners: Dict[str, int] = {}
    for task in active:
        key = task.owner or "unassigned"
        owners[key] = owners.get(key, 0) + 1

    return {
        "total": len(active),
        "by_status": statuses,
        "by_owner": owners,
    }


def format_task_summary(tasks: Sequence[TaskItem]) -> str:
    stats = summarize_tasks(tasks)
    return (
        f"Tasks updated (total {stats['total']}; "
        f"{stats['by_status'].get('pending', 0)} pending, "
        f"{stats['by_status'].get('in_progress', 0)} in progress)."
    )


_LEADER_TEAM_NAME: Optional[str] = None


def set_leader_team_name(team_name: str) -> None:
    """Set the leader's active team name (in-memory, session-scoped)."""
    global _LEADER_TEAM_NAME
    clean = (team_name or "").strip()
    if not clean:
        return
    _LEADER_TEAM_NAME = clean


def clear_leader_team_name() -> None:
    """Clear the leader team name."""
    global _LEADER_TEAM_NAME
    _LEADER_TEAM_NAME = None


def get_leader_team_name() -> Optional[str]:
    """Return the in-memory leader team name, if any."""
    return _LEADER_TEAM_NAME


def reset_task_list(task_list_id: str) -> None:
    """Clear all task files from a task list directory (keeps the directory)."""
    resolved = sanitize_identifier(task_list_id, fallback="default")
    directory = _config_root() / "tasks" / resolved
    if not directory.exists():
        return
    for path in directory.glob("*.json"):
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning(
                "[tasks] Failed removing task file during reset: %s: %s",
                type(exc).__name__,
                exc,
                extra={"path": str(path)},
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
    "clear_leader_team_name",
    "create_task",
    "delete_task",
    "ensure_task_list_dir",
    "format_task_lines",
    "format_task_summary",
    "get_leader_team_name",
    "get_next_actionable_task",
    "get_task",
    "is_task_system_enabled",
    "list_tasks",
    "reset_task_list",
    "resolve_task_list_id",
    "sanitize_identifier",
    "set_leader_team_name",
    "set_runtime_task_scope",
    "should_show_completed_tasks_in_ui",
    "summarize_tasks",
    "task_list_dir",
    "unresolved_blockers",
    "update_task",
]
