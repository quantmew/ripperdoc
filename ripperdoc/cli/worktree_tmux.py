"""Worktree + tmux bootstrap helpers for CLI startup.

This module isolates argument parsing and process bootstrap logic from the
main CLI entrypoint to keep `cli.py` focused on orchestration.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Callable, MutableMapping, Optional

from ripperdoc.utils.filesystem.git_utils import get_git_root
from ripperdoc.utils.collaboration.worktree import (
    WorktreeSession,
    create_task_worktree,
    generate_cli_worktree_name,
    register_session_worktree,
)

_PR_WORKTREE_URL_RE = re.compile(
    r"^https?://github\.com/[^/]+/[^/]+/pull/(\d+)/?(?:[?#].*)?$",
    re.IGNORECASE,
)
_PR_WORKTREE_HASH_RE = re.compile(r"^#(\d+)$")
_TMUX_PREFIX_ENV = "RIPPERDOC_TMUX_WORKTREE"
_PRECREATED_WORKTREE_ENV = "RIPPERDOC_PRECREATED_WORKTREE_PATH"
_PRECREATED_WORKTREE_REPO_ENV = "RIPPERDOC_PRECREATED_WORKTREE_REPO_ROOT"
_PRECREATED_WORKTREE_NAME_ENV = "RIPPERDOC_PRECREATED_WORKTREE_NAME"
_PRECREATED_WORKTREE_BRANCH_ENV = "RIPPERDOC_PRECREATED_WORKTREE_BRANCH"
_PRECREATED_WORKTREE_HEAD_ENV = "RIPPERDOC_PRECREATED_WORKTREE_HEAD_COMMIT"
_PRECREATED_WORKTREE_HOOK_ENV = "RIPPERDOC_PRECREATED_WORKTREE_HOOK_BASED"


def _extract_pr_number_from_worktree(value: str) -> Optional[int]:
    """Parse PR shorthand from a worktree target."""
    match = _PR_WORKTREE_URL_RE.match(value.strip())
    if match and match.group(1):
        return int(match.group(1))
    match = _PR_WORKTREE_HASH_RE.match(value.strip())
    if match and match.group(1):
        return int(match.group(1))
    return None


def _resolve_worktree_name_and_pr(raw_name: Optional[str]) -> tuple[Optional[str], Optional[int]]:
    """Resolve worktree name and optional PR number from CLI/user input."""
    clean = (raw_name or "").strip()
    if not clean:
        return None, None
    pr_number = _extract_pr_number_from_worktree(clean)
    if pr_number is None:
        return clean, None
    return f"pr-{pr_number}", pr_number


def _has_tmux_worktree_flags(argv: list[str]) -> bool:
    has_tmux = "--tmux" in argv or "--tmux=classic" in argv
    if not has_tmux:
        return False
    has_worktree = (
        "-w" in argv
        or "--worktree" in argv
        or any(token.startswith("--worktree=") for token in argv)
    )
    return has_worktree


def _extract_worktree_arg(argv: list[str]) -> Optional[str]:
    for idx, token in enumerate(argv):
        if token in ("-w", "--worktree"):
            candidate = argv[idx + 1] if idx + 1 < len(argv) else None
            if candidate and not candidate.startswith("-"):
                return candidate
            return None
        if token.startswith("--worktree="):
            value = token.split("=", 1)[1].strip()
            return value or None
        if token == "--worktree-name":
            candidate = argv[idx + 1] if idx + 1 < len(argv) else None
            if candidate and not candidate.startswith("-"):
                return candidate
            return None
        if token.startswith("--worktree-name="):
            value = token.split("=", 1)[1].strip()
            return value or None
    return None


def _extract_cwd_arg(argv: list[str]) -> Optional[Path]:
    for idx, token in enumerate(argv):
        if token == "--cwd":
            candidate = argv[idx + 1] if idx + 1 < len(argv) else None
            if candidate and not candidate.startswith("-"):
                return Path(candidate).expanduser()
            return None
        if token.startswith("--cwd="):
            value = token.split("=", 1)[1].strip()
            if value:
                return Path(value).expanduser()
            return None
    return None


def _strip_tmux_worktree_args(argv: list[str]) -> list[str]:
    processed: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in {"--tmux", "--tmux=classic"}:
            index += 1
            continue
        if token in {"-w", "--worktree", "--worktree-name"}:
            next_token = argv[index + 1] if index + 1 < len(argv) else None
            if next_token and not next_token.startswith("-"):
                index += 2
            else:
                index += 1
            continue
        if token.startswith("--worktree=") or token.startswith("--worktree-name="):
            index += 1
            continue
        processed.append(token)
        index += 1
    return processed


def _strip_cwd_args(argv: list[str]) -> list[str]:
    processed: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "--cwd":
            next_token = argv[index + 1] if index + 1 < len(argv) else None
            if next_token and not next_token.startswith("-"):
                index += 2
            else:
                index += 1
            continue
        if token.startswith("--cwd="):
            index += 1
            continue
        processed.append(token)
        index += 1
    return processed


def _hash_repository_path(repo_path: str) -> str:
    """Lightweight path hash used in tmux session names."""
    value = 0
    for ch in repo_path:
        value = ((value << 5) - value) + ord(ch)
        value &= 0xFFFFFFFF
    if value & 0x80000000:
        value = -((~value + 1) & 0xFFFFFFFF)
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    n = abs(value)
    if n == 0:
        return "0"
    out = ""
    while n > 0:
        n, rem = divmod(n, 36)
        out = alphabet[rem] + out
    return out


def _build_tmux_session_name(repo_root: Path, worktree_hint: str) -> str:
    repo_hash = _hash_repository_path(str(repo_root.resolve()))
    prefix = f"{repo_hash}_worktree-{worktree_hint}"
    return re.sub(r"[/.]", "_", prefix)


def _run_tmux_command(
    args: list[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[dict[str, str]] = None,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["tmux", *args],
        cwd=str(cwd) if cwd else None,
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def _exec_into_tmux_worktree(
    argv: list[str],
    *,
    get_git_root_fn: Callable[[Path], Optional[Path]] = get_git_root,
    create_task_worktree_fn: Callable[..., WorktreeSession] = create_task_worktree,
    generate_cli_worktree_name_fn: Callable[[], str] = generate_cli_worktree_name,
    run_tmux_command_fn: Callable[..., subprocess.CompletedProcess[str]] = _run_tmux_command,
    which_fn: Callable[[str], Optional[str]] = shutil.which,
    environ: Optional[MutableMapping[str, str]] = None,
    platform_name: Optional[str] = None,
    python_executable: Optional[str] = None,
) -> tuple[bool, Optional[str]]:
    """Fast path for --tmux + --worktree startup."""
    env_map = environ if environ is not None else os.environ
    effective_platform = platform_name if platform_name is not None else sys.platform
    effective_python = python_executable if python_executable is not None else sys.executable

    if env_map.get(_TMUX_PREFIX_ENV) == "1":
        return False, None
    if effective_platform == "win32":
        return False, "Error: --tmux is not supported on Windows"
    if which_fn("tmux") is None:
        install_hint = (
            "Install tmux with: brew install tmux"
            if effective_platform == "darwin"
            else "Install tmux with: sudo apt install tmux"
        )
        return False, f"Error: tmux is not installed. {install_hint}"

    start_path = _extract_cwd_arg(argv) or Path.cwd()
    repo_root = get_git_root_fn(start_path.resolve())
    if repo_root is None:
        return False, "Error: --worktree requires a git repository"

    worktree_input = _extract_worktree_arg(argv)
    resolved_name, pr_number = _resolve_worktree_name_and_pr(worktree_input)
    effective_worktree_input = resolved_name or generate_cli_worktree_name_fn()
    tmux_worktree_hint = effective_worktree_input

    try:
        session = create_task_worktree_fn(
            task_id=f"tmux_{uuid.uuid4().hex[:8]}",
            base_path=repo_root,
            requested_name=effective_worktree_input,
            pr_number=pr_number,
        )
    except (RuntimeError, ValueError, OSError) as exc:
        return False, f"Error: {exc}"

    child_args = _strip_cwd_args(_strip_tmux_worktree_args(argv))
    tmux_session_name = _build_tmux_session_name(repo_root, tmux_worktree_hint)
    worktree_path = session.worktree_path.resolve()

    child_env = dict(env_map)
    child_env[_TMUX_PREFIX_ENV] = "1"
    child_env.setdefault("RIPPERDOC_SESSION_ID", f"session-{uuid.uuid4().hex[:10]}")
    child_env[_PRECREATED_WORKTREE_ENV] = str(worktree_path)
    child_env[_PRECREATED_WORKTREE_REPO_ENV] = str(session.repo_root.resolve())
    child_env[_PRECREATED_WORKTREE_NAME_ENV] = session.name
    child_env[_PRECREATED_WORKTREE_BRANCH_ENV] = session.branch
    child_env[_PRECREATED_WORKTREE_HEAD_ENV] = session.head_commit or ""
    child_env[_PRECREATED_WORKTREE_HOOK_ENV] = "1" if session.hook_based else "0"

    cmd = [effective_python, "-m", "ripperdoc.cli.cli", *child_args]
    inside_tmux = bool(env_map.get("TMUX"))
    session_exists = run_tmux_command_fn(
        ["has-session", "-t", tmux_session_name],
        capture=False,
    ).returncode == 0

    if inside_tmux:
        if not session_exists:
            create_result = run_tmux_command_fn(
                ["new-session", "-d", "-s", tmux_session_name, "-c", str(worktree_path), "--", *cmd],
                cwd=worktree_path,
                env=child_env,
            )
            if create_result.returncode != 0:
                return False, "Error: failed to create tmux session"
        switch_result = run_tmux_command_fn(["switch-client", "-t", tmux_session_name], cwd=worktree_path)
        if switch_result.returncode != 0:
            return False, "Error: failed to switch tmux client"
        return True, None

    tmux_args = ["new-session", "-A", "-s", tmux_session_name, "-c", str(worktree_path), "--", *cmd]
    attach_result = run_tmux_command_fn(tmux_args, cwd=worktree_path, env=child_env)
    if attach_result.returncode != 0:
        return False, "Error: failed to start tmux session"
    return True, None


def _register_precreated_worktree_from_env(
    *,
    environ: Optional[MutableMapping[str, str]] = None,
    register_session_worktree_fn: Callable[[WorktreeSession], None] = register_session_worktree,
) -> Optional[WorktreeSession]:
    env_map = environ if environ is not None else os.environ
    worktree_path = env_map.pop(_PRECREATED_WORKTREE_ENV, "").strip()
    repo_root = env_map.pop(_PRECREATED_WORKTREE_REPO_ENV, "").strip()
    if not worktree_path or not repo_root:
        return None
    name = env_map.pop(_PRECREATED_WORKTREE_NAME_ENV, "").strip()
    branch = env_map.pop(_PRECREATED_WORKTREE_BRANCH_ENV, "").strip()
    head_commit = env_map.pop(_PRECREATED_WORKTREE_HEAD_ENV, "").strip() or None
    hook_based_raw = env_map.pop(_PRECREATED_WORKTREE_HOOK_ENV, "").strip()
    hook_based = hook_based_raw == "1"

    session = WorktreeSession(
        repo_root=Path(repo_root).resolve(),
        worktree_path=Path(worktree_path).resolve(),
        branch=branch,
        name=name or Path(worktree_path).name,
        head_commit=head_commit,
        hook_based=hook_based,
    )
    register_session_worktree_fn(session)
    return session


__all__ = [
    "_PR_WORKTREE_URL_RE",
    "_PR_WORKTREE_HASH_RE",
    "_TMUX_PREFIX_ENV",
    "_PRECREATED_WORKTREE_ENV",
    "_PRECREATED_WORKTREE_REPO_ENV",
    "_PRECREATED_WORKTREE_NAME_ENV",
    "_PRECREATED_WORKTREE_BRANCH_ENV",
    "_PRECREATED_WORKTREE_HEAD_ENV",
    "_PRECREATED_WORKTREE_HOOK_ENV",
    "_extract_pr_number_from_worktree",
    "_resolve_worktree_name_and_pr",
    "_has_tmux_worktree_flags",
    "_extract_worktree_arg",
    "_extract_cwd_arg",
    "_strip_tmux_worktree_args",
    "_strip_cwd_args",
    "_hash_repository_path",
    "_build_tmux_session_name",
    "_run_tmux_command",
    "_exec_into_tmux_worktree",
    "_register_precreated_worktree_from_env",
]
