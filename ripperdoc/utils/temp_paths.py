"""Centralized temporary path helpers for internal Ripperdoc files."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

RIPPERDOC_TMPDIR_ENV = "RIPPERDOC_TMPDIR"
RIPPERDOC_TMP_SUBDIR = "ripperdoc"


def ripperdoc_tmp_root() -> Path:
    """Return and ensure the root temp directory for Ripperdoc internals."""
    override = os.getenv(RIPPERDOC_TMPDIR_ENV)
    if override and override.strip():
        base = Path(override).expanduser()
    else:
        base = Path(tempfile.gettempdir())
    root = base / RIPPERDOC_TMP_SUBDIR
    root.mkdir(parents=True, exist_ok=True)
    return root


def ripperdoc_temporary_directory(*args: Any, **kwargs: Any) -> tempfile.TemporaryDirectory[str]:
    """Create a temporary directory rooted under ripperdoc_tmp_root()."""
    kwargs.setdefault("dir", str(ripperdoc_tmp_root()))
    return tempfile.TemporaryDirectory(*args, **kwargs)


def ripperdoc_mkstemp(*args: Any, **kwargs: Any) -> tuple[int, str]:
    """Create a temporary file rooted under ripperdoc_tmp_root()."""
    kwargs.setdefault("dir", str(ripperdoc_tmp_root()))
    return tempfile.mkstemp(*args, **kwargs)


def ripperdoc_mkdtemp(*args: Any, **kwargs: Any) -> str:
    """Create a temporary directory path rooted under ripperdoc_tmp_root()."""
    kwargs.setdefault("dir", str(ripperdoc_tmp_root()))
    return tempfile.mkdtemp(*args, **kwargs)

