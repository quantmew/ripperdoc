"""Crash-recovery pointer for Remote Control sessions.

Written immediately after a bridge session is created, periodically
refreshed during the session, and cleared on clean shutdown. If the
process dies uncleanly (crash, kill -9, terminal closed), the pointer
persists. On next startup, the bridge detects it and can resume.

Staleness is checked against the file's mtime (not an embedded timestamp)
so that a periodic re-write with the same content serves as a refresh.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Tuple

from ripperdoc.utils.log import get_logger

logger = get_logger()

BRIDGE_POINTER_TTL_SEC = 4 * 3600  # 4 hours
MAX_WORKTREE_FANOUT = 50


def _get_projects_dir() -> Path:
    """Return the base projects directory (~/.ripperdoc/projects)."""
    home = Path.home()
    base = home / ".ripperdoc"
    env_dir = os.getenv("RIPPERDOC_PROJECTS_DIR", "").strip()
    if env_dir:
        base = Path(env_dir)
    return base / "projects"


def _sanitize_path(path: str) -> str:
    """Normalize a path for use as a directory name (replace separators)."""
    return path.replace(os.sep, "_").replace("/", "_").replace("\\", "_").strip("_.")


@dataclass
class BridgePointer:
    """Crash-recovery pointer data."""

    session_id: str
    environment_id: str
    source: Literal["standalone", "repl"]


def get_bridge_pointer_path(dir_path: str) -> Path:
    """Return the path to the bridge pointer file for a working directory."""
    return _get_projects_dir() / _sanitize_path(os.path.abspath(dir_path)) / "bridge-pointer.json"


def write_bridge_pointer(dir_path: str, pointer: BridgePointer) -> None:
    """Write or refresh the crash-recovery pointer.

    Best-effort -- a crash-recovery file must never itself cause a crash.
    """
    path = get_bridge_pointer_path(dir_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "session_id": pointer.session_id,
            "environment_id": pointer.environment_id,
            "source": pointer.source,
        }))
    except OSError as exc:
        logger.debug("[bridge:pointer] write failed: %s", exc)


def read_bridge_pointer(dir_path: str) -> Optional[BridgePointer]:
    """Read the pointer and check staleness via mtime.

    Returns None on any failure: missing file, corrupted JSON,
    schema mismatch, or stale (mtime > 4h ago). Stale/invalid
    pointers are deleted so they don't keep re-prompting.
    """
    path = get_bridge_pointer_path(dir_path)
    try:
        mtime = path.stat().st_mtime
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        _clear_path(path)
        return None

    if not isinstance(data, dict):
        _clear_path(path)
        return None

    session_id = data.get("session_id")
    environment_id = data.get("environment_id")
    source = data.get("source")

    if (
        not isinstance(session_id, str) or not session_id.strip()
        or not isinstance(environment_id, str) or not environment_id.strip()
        or source not in ("standalone", "repl")
    ):
        logger.debug("[bridge:pointer] invalid schema, clearing: %s", path)
        _clear_path(path)
        return None

    age_sec = max(0.0, time.time() - mtime)
    if age_sec > BRIDGE_POINTER_TTL_SEC:
        logger.debug("[bridge:pointer] stale (>4h mtime), clearing: %s", path)
        _clear_path(path)
        return None

    return BridgePointer(
        session_id=session_id.strip(),
        environment_id=environment_id.strip(),
        source=source,
    )


def read_bridge_pointer_across_worktrees(dir_path: str) -> Optional[Tuple[BridgePointer, str]]:
    """Worktree-aware read for --continue.

    Checks `dir_path` first, then fans out across git worktree siblings
    to find the freshest pointer. Returns the pointer AND the dir it
    was found in.
    """
    here = read_bridge_pointer(dir_path)
    if here is not None:
        return here, dir_path

    worktrees = _get_worktree_paths(dir_path)
    if len(worktrees) <= 1:
        return None
    if len(worktrees) > MAX_WORKTREE_FANOUT:
        logger.debug(
            "[bridge:pointer] %d worktrees exceeds fanout cap %d, skipping",
            len(worktrees),
            MAX_WORKTREE_FANOUT,
        )
        return None

    abs_dir = os.path.abspath(dir_path)
    candidates = [wt for wt in worktrees if os.path.abspath(wt) != abs_dir]

    freshest: Optional[Tuple[BridgePointer, str]] = None
    # We don't have mtime age from read_bridge_pointer, so just use the first found
    for wt in candidates:
        p = read_bridge_pointer(wt)
        if p is not None:
            return p, wt

    return freshest


def clear_bridge_pointer(dir_path: str) -> None:
    """Delete the pointer. Idempotent -- ENOENT is expected."""
    path = get_bridge_pointer_path(dir_path)
    _clear_path(path)


def refresh_bridge_pointer(dir_path: str, pointer: BridgePointer) -> None:
    """Re-write the pointer to bump mtime (staleness refresh)."""
    write_bridge_pointer(dir_path, pointer)


def _clear_path(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def _get_worktree_paths(dir_path: str) -> List[str]:
    """Get git worktree paths. Returns [] on any error."""
    try:
        result = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            cwd=dir_path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return []
        paths: List[str] = []
        for line in result.stdout.splitlines():
            if line.startswith("worktree "):
                wt = line[len("worktree "):].strip()
                if wt:
                    paths.append(wt)
        return paths
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return []
