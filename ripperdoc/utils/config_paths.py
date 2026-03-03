"""Centralized path helpers for scoped Ripperdoc configuration files."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal, Optional

CONFIG_DIR_NAME = ".ripperdoc"
RIPPERDOC_CONFIG_DIR_ENV = "RIPPERDOC_CONFIG_DIR"
ConfigScope = Literal["user", "project", "local", "managed"]


def user_config_dir(home: Optional[Path] = None) -> Path:
    """Return the user-scoped config directory."""
    if home is not None:
        return home.expanduser() / CONFIG_DIR_NAME

    raw = os.getenv(RIPPERDOC_CONFIG_DIR_ENV)
    if raw and raw.strip():
        return Path(raw).expanduser()

    return Path.home().expanduser() / CONFIG_DIR_NAME


def project_config_dir(project_path: Optional[Path] = None) -> Path:
    """Return the project-scoped config directory."""
    base = project_path or Path.cwd()
    try:
        return base.resolve() / CONFIG_DIR_NAME
    except (OSError, RuntimeError):
        return base / CONFIG_DIR_NAME


def local_config_dir(project_path: Optional[Path] = None) -> Path:
    """Return the local (project-private) config directory."""
    return project_config_dir(project_path)


def managed_config_dir(
    project_path: Optional[Path] = None,
    *,
    home: Optional[Path] = None,
) -> Path:
    """Return the managed/system-level config directory.

    Default locations:
    - macOS: /Library/Application Support/ClaudeCode
    - Linux/WSL: /etc/claude-code
    - Windows: C:\\Program Files\\ClaudeCode

    Override via RIPPERDOC_MANAGED_CONFIG_DIR.
    """
    _ = project_path
    _ = home
    override = os.getenv("RIPPERDOC_MANAGED_CONFIG_DIR")
    if override and override.strip():
        return Path(override).expanduser()

    if sys.platform == "darwin":
        return Path("/Library/Application Support/ClaudeCode")
    if sys.platform == "win32":
        program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
        return Path(program_files) / "ClaudeCode"
    return Path("/etc/claude-code")


def config_dir_for_scope(
    scope: ConfigScope,
    *,
    project_path: Optional[Path] = None,
    home: Optional[Path] = None,
) -> Path:
    """Return config directory for a given scope."""
    if scope == "user":
        return user_config_dir(home=home)
    if scope == "project":
        return project_config_dir(project_path)
    if scope == "local":
        return local_config_dir(project_path)
    if scope == "managed":
        return managed_config_dir(project_path, home=home)
    raise ValueError(f"Unsupported config scope: {scope}")


def config_file_for_scope(
    scope: ConfigScope,
    file_name: str,
    *,
    project_path: Optional[Path] = None,
    home: Optional[Path] = None,
) -> Path:
    """Return scoped config file path by file name."""
    return config_dir_for_scope(
        scope,
        project_path=project_path,
        home=home,
    ) / file_name
