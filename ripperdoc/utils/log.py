"""Logging utilities for Ripperdoc."""

import logging
import sys
import os
from pathlib import Path
from typing import Optional
from datetime import datetime


class RipperdocLogger:
    """Logger for Ripperdoc."""

    def __init__(self, name: str = "ripperdoc", log_dir: Optional[Path] = None):
        self.logger = logging.getLogger(name)
        level_name = os.getenv("RIPPERDOC_LOG_LEVEL", "WARNING").upper()
        level = getattr(logging, level_name, logging.WARNING)
        self.logger.setLevel(level)

        # Console handler
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # File handler (optional)
        if log_dir:
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / f"ripperdoc_{datetime.now().strftime('%Y%m%d')}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)

    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)

    def critical(self, message: str) -> None:
        """Log critical message."""
        self.logger.critical(message)


# Global logger instance
_logger: Optional[RipperdocLogger] = None


def get_logger() -> RipperdocLogger:
    """Get the global logger instance."""
    global _logger
    if _logger is None:
        _logger = RipperdocLogger()
    return _logger


def init_logger(log_dir: Optional[Path] = None) -> RipperdocLogger:
    """Initialize the global logger."""
    global _logger
    _logger = RipperdocLogger(log_dir=log_dir)
    return _logger
