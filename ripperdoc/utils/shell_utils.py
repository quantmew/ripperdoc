"""Shell detection helpers.

Selects a suitable interactive shell for running commands, preferring bash/zsh
over the system's /bin/sh default to ensure features like brace expansion.
On Windows, prefers Git Bash and falls back to cmd.exe if no bash is available.
"""

from __future__ import annotations

import os
import shutil
from typing import Iterable, List

from ripperdoc.utils.log import get_logger

logger = get_logger()

# Common locations to probe if shutil.which misses an otherwise standard path.
_COMMON_BIN_DIRS: tuple[str, ...] = ("/bin", "/usr/bin", "/usr/local/bin", "/opt/homebrew/bin")
_IS_WINDOWS = os.name == "nt"


def _is_executable(path: str) -> bool:
    return bool(path) and os.path.isfile(path) and os.access(path, os.X_OK)


def _dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen = set()
    ordered: list[str] = []
    for item in items:
        if item and item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


def _find_git_bash_windows() -> str | None:
    env_path = os.environ.get("GIT_BASH_PATH") or os.environ.get("GITBASH")
    if env_path and _is_executable(env_path):
        return env_path

    bash_in_path = shutil.which("bash")
    if bash_in_path and "git" in bash_in_path.lower():
        return bash_in_path

    common = [
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files\Git\usr\bin\bash.exe",
        r"C:\Program Files (x86)\Git\bin\bash.exe",
        r"C:\Program Files (x86)\Git\usr\bin\bash.exe",
    ]
    for path in common:
        if _is_executable(path):
            return path
    return None


def _windows_cmd_path() -> str | None:
    comspec = os.environ.get("ComSpec")
    if _is_executable(comspec or ""):
        return comspec
    which_cmd = shutil.which("cmd.exe") or shutil.which("cmd")
    if which_cmd and _is_executable(which_cmd):
        return which_cmd
    system32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "cmd.exe")
    if _is_executable(system32):
        return system32
    return None


def find_suitable_shell() -> str:
    """Return a best-effort shell path, preferring bash/zsh (Git Bash on Windows).

    Priority on Unix:
      1) $SHELL if it's bash/zsh and executable
      2) bash/zsh from PATH
      3) bash/zsh in common bin directories

    Priority on Windows:
      1) Git Bash (env override or known locations / PATH)
      2) cmd.exe as a last resort

    Raises:
        RuntimeError: if no suitable shell is found.
    """

    env_override = os.environ.get("RIPPERDOC_SHELL") or os.environ.get("RIPPERDOC_SHELL_PATH")
    if env_override and _is_executable(env_override):
        logger.debug("Using shell from RIPPERDOC_SHELL*: %s", env_override)
        return env_override

    current_shell = os.environ.get("SHELL", "")
    current_is_bash = "bash" in current_shell
    current_is_zsh = "zsh" in current_shell

    if not _IS_WINDOWS:
        if (current_is_bash or current_is_zsh) and _is_executable(current_shell):
            logger.debug("Using SHELL from environment: %s", current_shell)
            return current_shell

        bash_path = shutil.which("bash") or ""
        zsh_path = shutil.which("zsh") or ""
        preferred_order = ["bash", "zsh"] if current_is_bash else ["zsh", "bash"]

        candidates: list[str] = []
        for name in preferred_order:
            if name == "bash" and bash_path:
                candidates.append(bash_path)
            if name == "zsh" and zsh_path:
                candidates.append(zsh_path)

        for bin_dir in _COMMON_BIN_DIRS:
            candidates.append(os.path.join(bin_dir, "bash"))
            candidates.append(os.path.join(bin_dir, "zsh"))

        for candidate in _dedupe_preserve_order(candidates):
            if _is_executable(candidate):
                logger.debug("Selected shell: %s", candidate)
                return candidate

        error_message = (
            "No suitable shell found. Please install bash or zsh and ensure $SHELL is set. "
            "Tried bash/zsh in PATH and common locations."
        )
        logger.error(error_message)
        raise RuntimeError(error_message)

    git_bash = _find_git_bash_windows()
    if git_bash:
        logger.debug("Using Git Bash: %s", git_bash)
        return git_bash

    cmd_path = _windows_cmd_path()
    if cmd_path:
        logger.warning("Falling back to cmd.exe; bash not found. Using: %s", cmd_path)
        return cmd_path

    error_message = (
        "No suitable shell found on Windows. Install Git for Windows to provide bash "
        "or ensure cmd.exe is available."
    )
    logger.error(error_message)
    raise RuntimeError(error_message)


def build_shell_command(shell_path: str, command: str) -> List[str]:
    """Build argv for running a command with the selected shell.

    For bash/zsh (including Git Bash), use -lc to run as login shell.
    For cmd.exe fallback, use /d /s /c.
    """

    lower = shell_path.lower()
    if lower.endswith("cmd.exe") or lower.endswith("\\cmd"):
        return [shell_path, "/d", "/s", "/c", command]
    return [shell_path, "-lc", command]


__all__ = ["find_suitable_shell", "build_shell_command"]
