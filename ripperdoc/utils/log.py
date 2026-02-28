"""Logging utilities for Ripperdoc."""

import json
import logging
import sys
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

from ripperdoc.utils.path_utils import sanitize_project_path


class SpinnerSafeStreamHandler(logging.StreamHandler):
    """StreamHandler that clears the current line before ERROR/WARNING messages.

    This prevents log messages from appearing after a spinner's text,
    which would cause formatting issues.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record, clearing the line first for ERROR/WARNING."""
        try:
            msg = self.format(record)
            stream = self.stream

            # Clear the current line before ERROR/WARNING to avoid spinner interference
            if record.levelno >= logging.ERROR:
                # Use \r to return to start, then clear with spaces, then \r again
                stream.write("\r" + " " * 100 + "\r")
            elif record.levelno >= logging.WARNING:
                # Also clear for WARNING
                stream.write("\r" + " " * 100 + "\r")

            stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)


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
        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc)
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


class DebugCategoryFilter(logging.Filter):
    """Simple include/exclude filter for debug output categories."""

    def __init__(self, filter_spec: str):
        super().__init__()
        include_tokens: List[str] = []
        exclude_tokens: List[str] = []
        for raw in str(filter_spec or "").split(","):
            token = raw.strip().lower()
            if not token:
                continue
            if token.startswith("!"):
                cleaned = token[1:].strip()
                if cleaned:
                    exclude_tokens.append(cleaned)
                continue
            include_tokens.append(token)
        self.include_tokens = include_tokens
        self.exclude_tokens = exclude_tokens

    def filter(self, record: logging.LogRecord) -> bool:
        target = f"{record.name} {record.getMessage()}".lower()
        if self.include_tokens and not any(token in target for token in self.include_tokens):
            return False
        if any(token in target for token in self.exclude_tokens):
            return False
        return True


class RipperdocLogger:
    """Logger for Ripperdoc."""

    def __init__(self, name: str = "ripperdoc", log_dir: Optional[Path] = None):
        self.logger = logging.getLogger(name)
        level_name = os.getenv("RIPPERDOC_LOG_LEVEL", "WARNING").upper()
        level = getattr(logging, level_name, logging.WARNING)
        # Allow file handlers to capture debug logs while console respects the configured level.
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        self._console_default_level = level
        self._console_handler: Optional[logging.Handler] = None
        self._debug_filter: Optional[logging.Filter] = None

        # Avoid adding duplicate handlers if an existing logger is reused.
        if not self.logger.handlers:
            console_handler = SpinnerSafeStreamHandler(sys.stderr)
            console_handler.setLevel(level)
            console_formatter = logging.Formatter("%(levelname)s: %(message)s")
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
            self._console_handler = console_handler
        else:
            for handler in self.logger.handlers:
                if isinstance(handler, logging.StreamHandler):
                    self._console_handler = handler
                    self._console_default_level = handler.level
                    break

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
        if self._debug_filter is not None:
            file_handler.addFilter(self._debug_filter)
        self.logger.addHandler(file_handler)
        self._file_handler = file_handler
        self._file_handler_path = log_file
        return log_file

    def _set_debug_filter(self, filter_spec: Optional[str]) -> None:
        """Apply debug filter to console/file handlers."""
        if self._debug_filter is not None:
            for handler in self.logger.handlers:
                try:
                    handler.removeFilter(self._debug_filter)
                except (ValueError, RuntimeError):
                    pass
            self._debug_filter = None

        cleaned = (filter_spec or "").strip()
        if not cleaned:
            return

        self._debug_filter = DebugCategoryFilter(cleaned)
        for handler in self.logger.handlers:
            handler.addFilter(self._debug_filter)

    def configure_debug(
        self,
        *,
        enabled: bool,
        filter_spec: Optional[str] = None,
        debug_file: Optional[Path] = None,
    ) -> Optional[Path]:
        """Configure runtime debug behavior for the current process."""
        if self._console_handler is not None:
            self._console_handler.setLevel(logging.DEBUG if enabled else self._console_default_level)
        self._set_debug_filter(filter_spec if enabled else None)

        if debug_file is not None:
            return self.attach_file_handler(debug_file.expanduser())
        return self._file_handler_path

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


def configure_debug_logging(
    *,
    enabled: bool,
    filter_spec: Optional[str] = None,
    debug_file: Optional[Path] = None,
) -> Optional[Path]:
    """Enable/disable debug logging with optional filtering and file path."""
    logger = get_logger()
    return logger.configure_debug(
        enabled=enabled,
        filter_spec=filter_spec,
        debug_file=debug_file,
    )


def enable_session_file_logging(project_path: Path, session_id: str) -> Path:
    """Ensure the global logger writes to the session-specific log file."""
    logger = get_logger()
    log_file = session_log_path(project_path, session_id)
    logger.attach_file_handler(log_file)
    logger.debug(f"[logging] File logging enabled at {log_file}")
    return log_file
