"""Git worktree helpers for Task isolation."""

from __future__ import annotations

import json
import os
import random
import re
import secrets
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

from ripperdoc.core.config import (
    get_global_config,
    get_project_config,
    get_project_local_config,
)
from ripperdoc.core.hooks.events import HookEvent
from ripperdoc.core.hooks.manager import HookResult, hook_manager
from ripperdoc.utils.git_utils import get_git_root
from ripperdoc.utils.log import get_logger


logger = get_logger()

_MAX_NAME_ATTEMPTS = 100
_SESSION_WORKTREES: dict[str, "WorktreeSession"] = {}
_SESSION_WORKTREES_LOCK = threading.Lock()
_CLI_WORKTREE_ADJECTIVES = ("swift", "bright", "calm", "keen", "bold")
_CLI_WORKTREE_NOUNS = ("fox", "owl", "elm", "oak", "ray")
_BASE36_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"


@dataclass(frozen=True)
class WorktreeSession:
    """Metadata for an isolated worktree created for a Task run."""

    repo_root: Path
    worktree_path: Path
    branch: str
    name: str
    head_commit: Optional[str] = None
    hook_based: bool = False


@dataclass(frozen=True)
class WorktreeCleanupResult:
    """Best-effort cleanup status for a worktree session."""

    worktree_path: Path
    branch: Optional[str]
    removed: bool
    branch_deleted: bool
    error: Optional[str] = None
    branch_error: Optional[str] = None


def _sanitize_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", (value or "").strip()).strip("-._")
    return normalized[:64] if normalized else "task"


def generate_cli_worktree_name() -> str:
    """Generate the default CLI worktree name."""
    adjective = random.choice(_CLI_WORKTREE_ADJECTIVES)  # noqa: S311
    noun = random.choice(_CLI_WORKTREE_NOUNS)  # noqa: S311
    suffix = "".join(random.choice(_BASE36_ALPHABET) for _ in range(4))  # noqa: S311
    return f"{adjective}-{noun}-{suffix}"


def generate_session_worktree_name(session_id: Optional[str] = None) -> str:
    """Generate EnterWorktree default name from session/timestamp."""
    key = (session_id or os.getenv("CLAUDE_SESSION_ID") or os.getenv("RIPPERDOC_SESSION_ID") or "").strip()
    if not key:
        key = f"session-{int(time.time() * 1000)}"
    timestamp = _to_base36(int(time.time() * 1000))
    random_part = "".join(secrets.choice(_BASE36_ALPHABET) for _ in range(6))
    return f"worktree-{key}-{timestamp}-{random_part}"


def _to_base36(value: int) -> str:
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    if value == 0:
        return "0"
    output = ""
    remaining = abs(int(value))
    while remaining:
        remaining, remainder = divmod(remaining, 36)
        output = alphabet[remainder] + output
    return output


def _run_git(
    args: list[str],
    *,
    cwd: Path,
    env: Optional[Mapping[str, str]] = None,
    timeout_sec: float = 20.0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        env=dict(env) if env is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_sec,
        check=False,
    )


def _branch_exists(repo_root: Path, branch: str) -> bool:
    probe = _run_git(["show-ref", "--verify", "--quiet", f"refs/heads/{branch}"], cwd=repo_root)
    return probe.returncode == 0


def _has_worktree_create_hook() -> bool:
    try:
        return hook_manager.has_hooks(HookEvent.WORKTREE_CREATE)
    except Exception:
        return False


def _has_worktree_remove_hook() -> bool:
    try:
        return hook_manager.has_hooks(HookEvent.WORKTREE_REMOVE)
    except Exception:
        return False


def _extract_worktree_path_from_hook_result(result: HookResult) -> Optional[str]:
    def _from_mapping(payload: dict[str, object]) -> Optional[str]:
        for key in ("worktreePath", "worktree_path", "path"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    for output in reversed(result.outputs):
        hso = getattr(output, "hook_specific_output", None)
        if isinstance(hso, dict):
            candidate = _from_mapping(hso)
            if candidate:
                return candidate

        raw_output = (getattr(output, "raw_output", None) or "").strip()
        if raw_output:
            try:
                parsed = json.loads(raw_output)
                if isinstance(parsed, dict):
                    candidate = _from_mapping(parsed)
                    if candidate:
                        return candidate
            except json.JSONDecodeError:
                if not raw_output.startswith("{"):
                    return raw_output

        for text_value in (
            getattr(output, "additional_context", None),
            getattr(output, "reason", None),
        ):
            if isinstance(text_value, str) and text_value.strip() and text_value.strip().startswith("/"):
                return text_value.strip()

    return None


def _resolve_hook_worktree_path(candidate_path: str, *, base_path: Path) -> Path:
    path = Path(candidate_path).expanduser()
    if not path.is_absolute():
        path = (base_path / path).resolve()
    return path.resolve()


def register_session_worktree(session: WorktreeSession) -> None:
    """Register a created worktree for session-end lifecycle handling."""
    with _SESSION_WORKTREES_LOCK:
        _SESSION_WORKTREES[str(session.worktree_path)] = session


def unregister_session_worktree(worktree_path: Path | str) -> None:
    """Remove one worktree from the session registry."""
    key = str(Path(worktree_path))
    with _SESSION_WORKTREES_LOCK:
        _SESSION_WORKTREES.pop(key, None)


def list_session_worktrees() -> list[WorktreeSession]:
    """Return worktrees created in this process/session."""
    with _SESSION_WORKTREES_LOCK:
        return list(_SESSION_WORKTREES.values())


def consume_session_worktrees() -> list[WorktreeSession]:
    """Return and clear the session worktree registry."""
    with _SESSION_WORKTREES_LOCK:
        sessions = list(_SESSION_WORKTREES.values())
        _SESSION_WORKTREES.clear()
        return sessions


def _worktrees_root(repo_root: Path) -> Path:
    """Worktree storage path for this repository."""
    return repo_root / ".ripperdoc" / "worktrees"


def _pick_unique_name(
    *,
    repo_root: Path,
    base_name: str,
    branch_prefix: str,
) -> tuple[str, str, Path]:
    worktrees_root = _worktrees_root(repo_root)
    worktrees_root.mkdir(parents=True, exist_ok=True)

    for idx in range(_MAX_NAME_ATTEMPTS):
        suffix = f"-{idx}" if idx else ""
        candidate_name = _sanitize_name(f"{base_name}{suffix}")
        candidate_branch = _sanitize_name(f"{branch_prefix}-{candidate_name}")
        candidate_path = worktrees_root / candidate_name
        if candidate_path.exists():
            continue
        if _branch_exists(repo_root, candidate_branch):
            continue
        return candidate_name, candidate_branch, candidate_path

    raise RuntimeError(
        f"Unable to allocate a unique worktree name for '{base_name}' after {_MAX_NAME_ATTEMPTS} attempts."
    )


def _maybe_resume_existing_worktree(
    *,
    repo_root: Path,
    requested_name: str,
) -> Optional[WorktreeSession]:
    """Resume an existing named worktree when it is already valid.

    If a requested worktree already exists and has a valid git HEAD, reuse it.
    """

    clean_name = _sanitize_name(requested_name)
    if not clean_name:
        return None
    worktree_path = (_worktrees_root(repo_root) / clean_name).resolve()
    if not worktree_path.exists():
        return None

    head_result = _run_git(["rev-parse", "HEAD"], cwd=worktree_path)
    if head_result.returncode != 0 or not head_result.stdout.strip():
        return None
    head_commit = head_result.stdout.strip()

    branch_result = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=worktree_path)
    branch = (
        branch_result.stdout.strip()
        if branch_result.returncode == 0 and branch_result.stdout.strip()
        else ""
    )

    return WorktreeSession(
        repo_root=repo_root.resolve(),
        worktree_path=worktree_path,
        branch=branch,
        name=clean_name,
        head_commit=head_commit,
        hook_based=False,
    )


def _normalize_git_file_list(output: str) -> list[str]:
    return [line.strip() for line in output.splitlines() if line.strip()]


def _is_safe_relative_path(path_value: str) -> bool:
    normalized = os.path.normpath(path_value)
    if not normalized or normalized.startswith("..") or os.path.isabs(normalized):
        return False
    return True


def _symlink_directories_from_config(repo_root: Path) -> list[str]:
    global_cfg = get_global_config()
    project_cfg = get_project_config(repo_root)
    local_cfg = get_project_local_config(repo_root)
    configured: list[str] = []
    for cfg in (global_cfg, project_cfg, local_cfg):
        worktree_cfg = getattr(cfg, "worktree", None)
        if not isinstance(worktree_cfg, dict):
            continue
        raw = (
            worktree_cfg.get("symlinkDirectories")
            if isinstance(worktree_cfg.get("symlinkDirectories"), list)
            else worktree_cfg.get("symlink_directories")
        )
        if not isinstance(raw, list):
            continue
        for item in raw:
            if not isinstance(item, str):
                continue
            clean = item.strip().strip("/")
            if clean and clean not in configured:
                configured.append(clean)
    return configured


def sync_worktree_configuration(
    *,
    repo_root: Path,
    worktree_path: Path,
) -> None:
    """Sync worktree config files, hooks path, and optional symlink directories.

    Strategy:
    - copy `settings.local.json` when present
    - set `core.hooksPath` to `.husky` or `.git/hooks` from main repo
    - symlink configured `worktree.symlinkDirectories`
    """

    source_settings = repo_root / "settings.local.json"
    if source_settings.exists() and source_settings.is_file():
        target_settings = worktree_path / "settings.local.json"
        try:
            target_settings.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_settings, target_settings)
        except OSError as exc:
            logger.debug(
                "[worktree] Failed to sync settings.local.json",
                extra={
                    "source": str(source_settings),
                    "target": str(target_settings),
                    "error": str(exc),
                },
            )

    hooks_source: Optional[Path] = None
    husky_dir = repo_root / ".husky"
    git_hooks_dir = repo_root / ".git" / "hooks"
    if husky_dir.is_dir():
        hooks_source = husky_dir
    elif git_hooks_dir.is_dir():
        hooks_source = git_hooks_dir
    if hooks_source is not None:
        hooks_result = _run_git(
            ["config", "core.hooksPath", str(hooks_source)],
            cwd=worktree_path,
        )
        if hooks_result.returncode != 0:
            logger.debug(
                "[worktree] Failed to set core.hooksPath",
                extra={
                    "worktree_path": str(worktree_path),
                    "hooks_source": str(hooks_source),
                    "stderr": hooks_result.stderr.strip(),
                },
            )

    for rel_dir in _symlink_directories_from_config(repo_root):
        if not _is_safe_relative_path(rel_dir):
            continue
        source_dir = repo_root / rel_dir
        target_dir = worktree_path / rel_dir
        if not source_dir.is_dir():
            continue
        if target_dir.exists() or target_dir.is_symlink():
            continue
        try:
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(str(source_dir), str(target_dir), target_is_directory=True)
        except OSError as exc:
            logger.debug(
                "[worktree] Failed to create symlink directory",
                extra={
                    "source": str(source_dir),
                    "target": str(target_dir),
                    "error": str(exc),
                },
            )


def copy_worktree_include_files(
    *,
    repo_root: Path,
    worktree_path: Path,
) -> list[str]:
    """Copy ignored/untracked files selected by `.worktreeinclude` into a worktree.

    Include behavior:
    - If `.worktreeinclude` is missing or has no valid patterns, do nothing.
    - Candidates are computed by intersecting:
      1) `git ls-files --others --ignored --exclude-standard`
      2) `git ls-files --others --ignored --exclude-from=.worktreeinclude`
    """

    include_file = repo_root / ".worktreeinclude"
    if not include_file.exists():
        return []

    try:
        include_text = include_file.read_text(encoding="utf-8")
    except OSError:
        return []

    has_valid_patterns = any(
        line.strip() and not line.strip().startswith("#")
        for line in include_text.splitlines()
    )
    if not has_valid_patterns:
        return []

    all_ignored_result = _run_git(
        ["ls-files", "--others", "--ignored", "--exclude-standard"],
        cwd=repo_root,
    )
    included_result = _run_git(
        ["ls-files", "--others", "--ignored", "--exclude-from=.worktreeinclude"],
        cwd=repo_root,
    )
    if all_ignored_result.returncode != 0 or included_result.returncode != 0:
        return []

    all_ignored = set(_normalize_git_file_list(all_ignored_result.stdout))
    included_candidates = _normalize_git_file_list(included_result.stdout)
    files_to_copy = [path for path in included_candidates if path in all_ignored]
    if not files_to_copy:
        return []

    copied: list[str] = []
    for rel_path in files_to_copy:
        normalized_rel = os.path.normpath(rel_path)
        if not _is_safe_relative_path(normalized_rel):
            continue

        source = repo_root / normalized_rel
        destination = worktree_path / normalized_rel
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.is_symlink():
                if destination.exists() or destination.is_symlink():
                    destination.unlink()
                os.symlink(os.readlink(source), destination)
            elif source.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            elif source.exists():
                shutil.copy2(source, destination)
            else:
                continue
            copied.append(rel_path)
        except OSError as exc:
            logger.debug(
                "[worktree] Failed to copy include file",
                extra={
                    "repo_root": str(repo_root),
                    "worktree_path": str(worktree_path),
                    "file": rel_path,
                    "error": str(exc),
                },
            )
    return copied


def has_worktree_changes(
    *,
    worktree_path: Path,
    baseline_ref: Optional[str] = None,
) -> bool:
    """Return True when a worktree has uncommitted or committed changes.

    Semantics:
    - Any `git status --porcelain` output means changed.
    - If a baseline ref is provided, commits ahead of baseline mean changed.
    """

    status_result = _run_git(
        ["status", "--porcelain"],
        cwd=worktree_path,
    )
    if status_result.returncode == 0 and status_result.stdout.strip():
        return True

    clean_baseline = (baseline_ref or "").strip()
    if not clean_baseline:
        return False

    rev_result = _run_git(
        ["rev-list", "--count", f"{clean_baseline}..HEAD"],
        cwd=worktree_path,
    )
    if rev_result.returncode != 0:
        return False
    try:
        return int(rev_result.stdout.strip() or "0") > 0
    except ValueError:
        return False


def _default_base_ref(repo_root: Path) -> str:
    """Resolve base ref for non-PR worktree creation.

    Behavior:
    - prefer fetched `origin/<default-branch>` when available
    - fallback to `HEAD` if fetch is unavailable
    """

    env = {
        **os.environ,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_ASKPASS": "",
    }
    remote_head = _run_git(
        ["symbolic-ref", "--quiet", "--short", "refs/remotes/origin/HEAD"],
        cwd=repo_root,
        env=env,
    )
    branch_name = ""
    if remote_head.returncode == 0 and remote_head.stdout.strip():
        ref = remote_head.stdout.strip()
        if ref.startswith("origin/"):
            branch_name = ref[len("origin/") :]
        else:
            branch_name = ref

    if not branch_name:
        local_head = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root, env=env)
        if local_head.returncode == 0 and local_head.stdout.strip():
            branch_name = local_head.stdout.strip()

    if not branch_name:
        return "HEAD"

    fetch_result = _run_git(
        ["fetch", "origin", branch_name],
        cwd=repo_root,
        env=env,
        timeout_sec=60.0,
    )
    if fetch_result.returncode == 0:
        return f"origin/{branch_name}"
    return "HEAD"


def create_task_worktree(
    *,
    task_id: str,
    base_path: Optional[Path] = None,
    requested_name: Optional[str] = None,
    pr_number: Optional[int] = None,
) -> WorktreeSession:
    """Create an isolated git worktree for a task run.

    Raises:
        ValueError: when not inside a git repository.
        RuntimeError: when git worktree creation fails.
    """

    start_path = (base_path or Path.cwd()).resolve()
    task_hint = _sanitize_name(task_id[-8:] if task_id else "task")
    base_name = _sanitize_name(requested_name or f"task-{task_hint}")
    repo_root = get_git_root(start_path)
    if _has_worktree_create_hook():
        hook_result = hook_manager.run_worktree_create(base_name)
        if hook_result.should_block:
            raise RuntimeError(hook_result.block_reason or "Blocked by WorktreeCreate hook.")
        hook_worktree_path = _extract_worktree_path_from_hook_result(hook_result)
        if not hook_worktree_path:
            raise RuntimeError(
                "WorktreeCreate hook did not return worktreePath."
            )
        resolved_hook_path = _resolve_hook_worktree_path(
            hook_worktree_path,
            base_path=start_path,
        )
        if not resolved_hook_path.exists():
            raise RuntimeError(
                f"WorktreeCreate hook returned path that does not exist: {resolved_hook_path}"
            )
        session = WorktreeSession(
            repo_root=(repo_root or start_path).resolve(),
            worktree_path=resolved_hook_path,
            branch="",
            name=base_name,
            head_commit=None,
            hook_based=True,
        )
        register_session_worktree(session)
        return session
    if repo_root is None:
        raise ValueError(
            "Cannot create agent worktree: not in a git repository and no WorktreeCreate hooks are configured."
        )

    repo_root = repo_root.resolve()
    if requested_name:
        resumed = _maybe_resume_existing_worktree(
            repo_root=repo_root,
            requested_name=base_name,
        )
        if resumed is not None:
            register_session_worktree(resumed)
            return resumed

    name, branch, worktree_path = _pick_unique_name(
        repo_root=repo_root,
        base_name=base_name,
        branch_prefix="worktree",
    )

    env = {
        **os.environ,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_ASKPASS": "",
    }
    base_ref = "HEAD"
    if pr_number is not None:
        pr_fetch = _run_git(
            ["fetch", "origin", f"pull/{pr_number}/head"],
            cwd=repo_root,
            env=env,
            timeout_sec=60.0,
        )
        if pr_fetch.returncode != 0:
            raise RuntimeError(
                f"Failed to fetch PR #{pr_number}: "
                + (
                    pr_fetch.stderr.strip()
                    or 'PR may not exist or the repository may not have a remote named "origin"'
                )
            )
        base_ref = "FETCH_HEAD"
    else:
        base_ref = _default_base_ref(repo_root)

    add_result = _run_git(
        ["worktree", "add", "-b", branch, str(worktree_path), base_ref],
        cwd=repo_root,
        env=env,
        timeout_sec=60.0,
    )
    if add_result.returncode != 0:
        logger.warning(
            "[worktree] Failed to create worktree: %s",
            add_result.stderr.strip() or add_result.stdout.strip(),
            extra={
                "repo_root": str(repo_root),
                "worktree_path": str(worktree_path),
                "branch": branch,
            },
        )
        raise RuntimeError(
            "Failed to create worktree via git: "
            + (add_result.stderr.strip() or add_result.stdout.strip() or "unknown error")
        )

    head_result = _run_git(
        ["rev-parse", "HEAD"],
        cwd=worktree_path,
    )
    head_commit = (
        head_result.stdout.strip()
        if head_result.returncode == 0 and head_result.stdout.strip()
        else None
    )

    session = WorktreeSession(
        repo_root=repo_root,
        worktree_path=worktree_path.resolve(),
        branch=branch,
        name=name,
        head_commit=head_commit,
        hook_based=False,
    )
    register_session_worktree(session)
    sync_worktree_configuration(
        repo_root=session.repo_root,
        worktree_path=session.worktree_path,
    )
    copied = copy_worktree_include_files(
        repo_root=session.repo_root,
        worktree_path=session.worktree_path,
    )
    if copied:
        logger.info(
            "[worktree] Copied files from .worktreeinclude",
            extra={
                "repo_root": str(session.repo_root),
                "worktree_path": str(session.worktree_path),
                "count": len(copied),
            },
        )
    return session


def cleanup_worktree_session(
    session: WorktreeSession,
    *,
    force: bool = True,
) -> WorktreeCleanupResult:
    """Best-effort cleanup for one worktree session."""
    if session.hook_based:
        if not _has_worktree_remove_hook():
            return WorktreeCleanupResult(
                worktree_path=session.worktree_path,
                branch=session.branch,
                removed=False,
                branch_deleted=False,
                error="No WorktreeRemove hook configured for hook-based worktree.",
            )
        hook_result = hook_manager.run_worktree_remove(str(session.worktree_path))
        if hook_result.should_block:
            return WorktreeCleanupResult(
                worktree_path=session.worktree_path,
                branch=session.branch,
                removed=False,
                branch_deleted=False,
                error=hook_result.block_reason or "Blocked by WorktreeRemove hook.",
            )
        if hook_result.has_errors:
            return WorktreeCleanupResult(
                worktree_path=session.worktree_path,
                branch=session.branch,
                removed=False,
                branch_deleted=False,
                error="; ".join(hook_result.errors),
            )
        return WorktreeCleanupResult(
            worktree_path=session.worktree_path,
            branch=session.branch,
            removed=True,
            branch_deleted=False,
            error=None,
        )

    worktree_args = ["worktree", "remove"]
    if force:
        worktree_args.append("--force")
    worktree_args.append(str(session.worktree_path))

    remove_result = _run_git(worktree_args, cwd=session.repo_root, timeout_sec=60.0)
    removed = remove_result.returncode == 0 or not session.worktree_path.exists()
    remove_error: Optional[str] = None
    if not removed:
        try:
            shutil.rmtree(session.worktree_path)
            removed = True
        except OSError as exc:
            remove_error = str(exc)

    branch_deleted = False
    branch_error: Optional[str] = None
    if session.branch:
        branch_result = _run_git(
            ["branch", "-D", session.branch],
            cwd=session.repo_root,
            timeout_sec=20.0,
        )
        if branch_result.returncode == 0:
            branch_deleted = True
        else:
            branch_error = (
                branch_result.stderr.strip() or branch_result.stdout.strip() or "unknown error"
            )

    if remove_error is None and remove_result.returncode != 0 and not removed:
        remove_error = (
            remove_result.stderr.strip() or remove_result.stdout.strip() or "unknown error"
        )

    return WorktreeCleanupResult(
        worktree_path=session.worktree_path,
        branch=session.branch,
        removed=removed,
        branch_deleted=branch_deleted,
        error=remove_error,
        branch_error=branch_error,
    )


def cleanup_worktree_sessions(
    sessions: Sequence[WorktreeSession],
    *,
    force: bool = True,
) -> list[WorktreeCleanupResult]:
    """Cleanup a batch of worktree sessions."""
    return [cleanup_worktree_session(session, force=force) for session in sessions]


def cleanup_registered_worktrees(*, force: bool = True) -> list[WorktreeCleanupResult]:
    """Cleanup and clear all registered session worktrees."""
    sessions = consume_session_worktrees()
    return cleanup_worktree_sessions(sessions, force=force)


__all__ = [
    "WorktreeSession",
    "WorktreeCleanupResult",
    "generate_cli_worktree_name",
    "generate_session_worktree_name",
    "create_task_worktree",
    "sync_worktree_configuration",
    "copy_worktree_include_files",
    "register_session_worktree",
    "unregister_session_worktree",
    "list_session_worktrees",
    "consume_session_worktrees",
    "has_worktree_changes",
    "cleanup_worktree_session",
    "cleanup_worktree_sessions",
    "cleanup_registered_worktrees",
]
