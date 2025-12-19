"""Todo storage and utilities for Ripperdoc.

This module provides simple, file-based todo management so tools can
persist and query tasks between turns. Todos are stored under the user's
home directory at `~/.ripperdoc/todos/<project>/todos.json`, where
`<project>` is a sanitized form of the project path.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Tuple
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ripperdoc.utils.log import get_logger
from ripperdoc.utils.path_utils import project_storage_dir


logger = get_logger()

TodoStatus = Literal["pending", "in_progress", "completed"]
TodoPriority = Literal["high", "medium", "low"]


class TodoItem(BaseModel):
    """Represents a single todo entry."""

    id: str = Field(description="Unique identifier for the todo item")
    content: str = Field(description="Task description")
    status: TodoStatus = Field(
        default="pending", description="Current state: pending, in_progress, completed"
    )
    priority: TodoPriority = Field(default="medium", description="Priority: high|medium|low")
    created_at: Optional[float] = Field(default=None, description="Unix timestamp when created")
    updated_at: Optional[float] = Field(default=None, description="Unix timestamp when updated")
    previous_status: Optional[TodoStatus] = Field(
        default=None, description="Previous status, used for audits"
    )
    model_config = ConfigDict(extra="ignore")


MAX_TODOS = 200


def _storage_path(project_root: Optional[Path], ensure_dir: bool) -> Path:
    """Return the todo storage path, optionally ensuring the directory exists."""
    root = project_root or Path.cwd()
    base_dir = Path.home() / ".ripperdoc" / "todos"
    storage_dir = project_storage_dir(base_dir, root, ensure=ensure_dir)
    return storage_dir / "todos.json"


def validate_todos(
    todos: Sequence[TodoItem], max_items: int = MAX_TODOS
) -> Tuple[bool, str | None]:
    """Basic validation for a todo list."""
    if len(todos) > max_items:
        return False, f"Too many todos; limit is {max_items}."

    ids = [todo.id for todo in todos]
    duplicate_ids = {id_ for id_ in ids if ids.count(id_) > 1}
    if duplicate_ids:
        return False, f"Duplicate todo IDs found: {sorted(duplicate_ids)}"

    in_progress = [todo for todo in todos if todo.status == "in_progress"]
    if len(in_progress) > 1:
        return False, "Only one todo can be marked in_progress at a time."

    empty_contents = [todo.id for todo in todos if not todo.content.strip()]
    if empty_contents:
        return False, f"Todos require content. Empty content for IDs: {sorted(empty_contents)}"

    return True, None


def load_todos(project_root: Optional[Path] = None) -> List[TodoItem]:
    """Load todos from disk."""
    path = _storage_path(project_root, ensure_dir=False)
    if not path.exists():
        return []

    try:
        raw = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError, IOError, UnicodeDecodeError) as exc:
        logger.warning(
            "Failed to load todos from disk: %s: %s",
            type(exc).__name__,
            exc,
            extra={"path": str(path)},
        )
        return []

    todos: List[TodoItem] = []
    for item in raw:
        try:
            todos.append(TodoItem(**item))
        except ValidationError as exc:
            logger.error(f"Failed to parse todo item: {exc}")
            continue

    # Preserve stored order; do not reorder based on status/priority.
    return todos


def save_todos(todos: Sequence[TodoItem], project_root: Optional[Path] = None) -> None:
    """Persist todos to disk."""
    path = _storage_path(project_root, ensure_dir=True)
    path.write_text(json.dumps([todo.model_dump() for todo in todos], indent=2))


def set_todos(
    todos: Sequence[TodoItem],
    project_root: Optional[Path] = None,
) -> List[TodoItem]:
    """Validate, normalize, and persist the provided todos."""
    ok, message = validate_todos(todos)
    if not ok:
        raise ValueError(message or "Invalid todos.")

    existing = {todo.id: todo for todo in load_todos(project_root)}
    now = time.time()

    normalized: List[TodoItem] = []
    for todo in todos:
        previous = existing.get(todo.id)
        normalized.append(
            todo.model_copy(
                update={
                    "created_at": previous.created_at if previous else todo.created_at or now,
                    "updated_at": now,
                    "previous_status": (
                        previous.status
                        if previous and previous.status != todo.status
                        else todo.previous_status
                    ),
                }
            )
        )

    # Keep the caller-provided order; do not resort.
    save_todos(normalized, project_root)
    return list(normalized)


def clear_todos(project_root: Optional[Path] = None) -> None:
    """Remove all todos."""
    save_todos([], project_root)


def get_next_actionable(todos: Sequence[TodoItem]) -> Optional[TodoItem]:
    """Return the next todo to work on (in_progress first, then pending)."""
    for status in ("in_progress", "pending"):
        for todo in todos:
            if todo.status == status:
                return todo
    return None


def summarize_todos(todos: Sequence[TodoItem]) -> dict:
    """Return simple statistics for a todo collection."""
    return {
        "total": len(todos),
        "by_status": {
            "pending": len([t for t in todos if t.status == "pending"]),
            "in_progress": len([t for t in todos if t.status == "in_progress"]),
            "completed": len([t for t in todos if t.status == "completed"]),
        },
        "by_priority": {
            "high": len([t for t in todos if t.priority == "high"]),
            "medium": len([t for t in todos if t.priority == "medium"]),
            "low": len([t for t in todos if t.priority == "low"]),
        },
    }


def format_todo_summary(todos: Sequence[TodoItem]) -> str:
    """Create a concise summary string for use in tool outputs."""
    stats = summarize_todos(todos)
    summary = (
        f"Todos updated (total {stats['total']}; "
        f"{stats['by_status']['pending']} pending, "
        f"{stats['by_status']['in_progress']} in progress, "
        f"{stats['by_status']['completed']} completed)."
    )

    next_item = get_next_actionable(todos)
    if next_item:
        summary += f" Next to tackle: {next_item.content} (id: {next_item.id}, status: {next_item.status})."
    elif stats["total"] == 0:
        summary += " No todos stored yet."

    return summary


def format_todo_lines(todos: Sequence[TodoItem]) -> List[str]:
    """Return human-readable todo lines."""
    status_marker = {
        "completed": "●",
        "in_progress": "◐",
        "pending": "○",
    }
    return [f"{status_marker.get(todo.status, '○')} {todo.content}" for todo in todos]
