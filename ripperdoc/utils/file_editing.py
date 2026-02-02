"""Shared helpers for safe file editing."""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path
from typing import Generator, Optional, TextIO, Union

from ripperdoc.utils.log import get_logger
from ripperdoc.utils.file_watch import record_snapshot
from ripperdoc.utils.platform import HAS_FCNTL

logger = get_logger()

PathLike = Union[str, Path]


def resolve_input_path(input_path: str) -> tuple[Path, str]:
    """Return resolved path plus cache key (abspath preserves symlink usage)."""
    path = Path(input_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve(), os.path.abspath(input_path)


def safe_record_snapshot(
    file_path: str,
    content: str,
    cache: dict,
    *,
    encoding: str = "utf-8",
    log_prefix: str = "",
) -> None:
    """Record file snapshot with shared error handling."""
    try:
        record_snapshot(
            file_path,
            content,
            cache,
            encoding=encoding,
        )
    except (OSError, IOError, RuntimeError) as exc:
        prefix = f"{log_prefix} " if log_prefix else ""
        logger.warning(
            "%sFailed to record file snapshot: %s: %s",
            prefix,
            type(exc).__name__,
            exc,
            extra={"file_path": file_path},
        )


@contextlib.contextmanager
def file_lock(file_handle: TextIO, exclusive: bool = True) -> Generator[None, None, None]:
    """Acquire a file lock, with fallback for systems without fcntl."""
    if not HAS_FCNTL:
        yield
        return

    import fcntl

    lock_type = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    try:
        fcntl.flock(file_handle.fileno(), lock_type)
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(file_handle.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def open_locked_file(
    file_path: PathLike, encoding: str
) -> Generator[tuple[TextIO, Optional[float], Optional[float]], None, None]:
    """Open a file for read/write and acquire an exclusive lock."""
    with open(str(file_path), "r+", encoding=encoding) as handle:
        try:
            pre_lock_mtime = os.fstat(handle.fileno()).st_mtime
        except OSError:
            pre_lock_mtime = None

        with file_lock(handle, exclusive=True):
            try:
                post_lock_mtime = os.fstat(handle.fileno()).st_mtime
            except OSError:
                post_lock_mtime = None
            yield handle, pre_lock_mtime, post_lock_mtime


def select_write_encoding(
    file_encoding: str,
    updated_content: str,
    file_path: PathLike,
    *,
    log_prefix: str = "",
) -> str:
    """Return a safe encoding for writing updated content."""
    try:
        updated_content.encode(file_encoding)
        return file_encoding
    except (UnicodeEncodeError, LookupError):
        prefix = f"{log_prefix} " if log_prefix else ""
        logger.info(
            "%sNew content cannot be encoded with %s, using UTF-8 for %s",
            prefix,
            file_encoding,
            str(file_path),
        )
        return "utf-8"


def atomic_write_with_fallback(
    handle: TextIO,
    file_path: PathLike,
    updated_content: str,
    write_encoding: str,
    original_content: str,
    *,
    temp_prefix: str,
    log_prefix: str,
    conflict_message: str,
) -> Optional[str]:
    """Atomically write content, falling back to in-place write if needed."""
    try:
        file_dir = os.path.dirname(str(file_path))
        try:
            fd, temp_path = tempfile.mkstemp(
                dir=file_dir,
                prefix=temp_prefix,
                suffix=".tmp",
            )
            try:
                with os.fdopen(fd, "w", encoding=write_encoding) as temp_f:
                    temp_f.write(updated_content)
                original_stat = os.fstat(handle.fileno())
                os.chmod(temp_path, original_stat.st_mode)
                os.replace(temp_path, str(file_path))
            except Exception:
                with contextlib.suppress(OSError):
                    os.unlink(temp_path)
                raise
        except OSError as atomic_error:
            handle.seek(0)
            current_content = handle.read()
            if current_content != original_content:
                return conflict_message
            handle.seek(0)
            handle.truncate()
            handle.write(updated_content)
            prefix = f"{log_prefix} " if log_prefix else ""
            logger.debug("%sAtomic write failed, used fallback: %s", prefix, atomic_error)
    except (OSError, IOError, PermissionError, UnicodeDecodeError) as exc:
        prefix = f"{log_prefix} " if log_prefix else ""
        logger.warning(
            "%sError writing edited file: %s: %s",
            prefix,
            type(exc).__name__,
            exc,
            extra={"file_path": str(file_path)},
        )
        return f"Error writing file: {exc}"
    return None
