"""Logging utilities for Ripperdoc."""

import json
import logging
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ripperdoc.utils.path_utils import sanitize_project_path


_LOG_RECORD_FIELDS = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "message",
    "asctime",
    "stacklevel",
}


class StructuredFormatter(logging.Formatter):
    """Formatter with ISO timestamps and context."""

    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = None) -> str:
        timestamp = datetime.utcfromtimestamp(record.created)
        return timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _LOG_RECORD_FIELDS and not key.startswith("_")
        }
        if extras:
            try:
                serialized = json.dumps(extras, sort_keys=True, ensure_ascii=True, default=str)
            except (TypeError, ValueError):
                serialized = str(extras)
            return f"{message} | {serialized}"
        return message


class RipperdocLogger:
    """Logger for Ripperdoc."""

    def __init__(self, name: str = "ripperdoc", log_dir: Optional[Path] = None):
        self.logger = logging.getLogger(name)
        level_name = os.getenv("RIPPERDOC_LOG_LEVEL", "WARNING").upper()
        level = getattr(logging, level_name, logging.WARNING)
        # Allow file handlers to capture debug logs while console respects the configured level.
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        # Avoid adding duplicate handlers if an existing logger is reused.
        if not self.logger.handlers:
            console_handler = logging.StreamHandler(sys.stderr)
            console_handler.setLevel(level)
            console_formatter = logging.Formatter("%(levelname)s: %(message)s")
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

        self._file_handler: Optional[logging.Handler] = None
        self._file_handler_path: Optional[Path] = None

        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"ripperdoc_{datetime.now().strftime('%Y%m%d')}.log"
            self.attach_file_handler(log_file)

    def attach_file_handler(self, log_file: Path) -> Path:
        """Attach or replace a file handler for logging to disk."""
        log_file.parent.mkdir(parents=True, exist_ok=True)
        if self._file_handler and self._file_handler_path == log_file:
            return log_file

        if self._file_handler:
            try:
                self.logger.removeHandler(self._file_handler)
            except (ValueError, RuntimeError):
                # Swallow errors while rotating handlers; console logging should continue.
                pass

        # Use UTF-8 to avoid Windows code page encoding errors when logs contain non-ASCII text.
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = StructuredFormatter("%(asctime)s [%(levelname)s] %(message)s")
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        self._file_handler = file_handler
        self._file_handler_path = log_file
        return log_file

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log debug message."""
        self.logger.debug(message, *args, **kwargs)

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log info message."""
        self.logger.info(message, *args, **kwargs)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log warning message."""
        self.logger.warning(message, *args, **kwargs)

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log error message."""
        self.logger.error(message, *args, **kwargs)

    def critical(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log critical message."""
        self.logger.critical(message, *args, **kwargs)

    def exception(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log an exception with traceback."""
        self.logger.exception(message, *args, **kwargs)


# Global logger instance
_logger: Optional[RipperdocLogger] = None


def get_logger() -> RipperdocLogger:
    """Get the global logger instance."""
    global _logger
    if _logger is None:
        _logger = RipperdocLogger()
    return _logger


def _normalize_path_for_logs(project_path: Path) -> Path:
    """Return the directory for log files for a given project."""
    safe_name = sanitize_project_path(project_path)
    return Path.home() / ".ripperdoc" / "logs" / safe_name


def session_log_path(project_path: Path, session_id: str, when: Optional[datetime] = None) -> Path:
    """Build the log file path for a project session."""
    timestamp = (when or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return _normalize_path_for_logs(project_path) / f"{timestamp}-{session_id}.log"


def init_logger(log_dir: Optional[Path] = None) -> RipperdocLogger:
    """Initialize the global logger."""
    global _logger
    _logger = RipperdocLogger(log_dir=log_dir)
    return _logger


def enable_session_file_logging(project_path: Path, session_id: str) -> Path:
    """Ensure the global logger writes to the session-specific log file."""
    logger = get_logger()
    log_file = session_log_path(project_path, session_id)
    logger.attach_file_handler(log_file)
    logger.debug(f"[logging] File logging enabled at {log_file}")
    return log_file
