"""Platform detection utilities.

This module provides a unified interface for detecting the current operating system
and platform-specific capabilities. It should be used instead of direct checks
like `sys.platform == "win32"` or `os.name == "nt"`.

Usage:
    from ripperdoc.utils.platform import (
        is_windows,
        is_linux,
        is_macos,
        is_unix,
        Platform,
    )

    if is_windows():
        # Windows-specific code
    elif is_macos():
        # macOS-specific code
    else:
        # Linux or other Unix-specific code
"""

import os
import sys
from typing import Final, Literal


# Platform type definitions
PlatformType = Literal["windows", "linux", "macos", "unknown"]


class Platform:
    """Platform detection constants and utilities.

    This class provides platform detection methods and constants that should
    be used throughout the codebase instead of direct checks.
    """

    # Platform constants (using sys.platform for consistency)
    WINDOWS: Final = "win32"
    LINUX: Final = "linux"
    MACOS: Final = "darwin"
    FREEBSD: Final = "freebsd"
    OPENBSD: Final = "openbsd"
    NETBSD: Final = "netbsd"

    # os.name constants
    NAME_NT: Final = "nt"  # Windows
    NAME_POSIX: Final = "posix"  # Unix-like systems

    @staticmethod
    def get_system() -> PlatformType:
        """Get the current operating system name.

        Returns:
            'windows', 'linux', 'macos', or 'unknown'
        """
        platform = sys.platform.lower()

        if platform.startswith("win"):
            return "windows"
        elif platform.startswith("darwin"):
            return "macos"
        elif platform.startswith("linux"):
            return "linux"
        elif platform in {"freebsd", "openbsd", "netbsd"}:
            return "linux"  # Treat BSD as Linux for most purposes
        else:
            return "unknown"

    @staticmethod
    def is_windows() -> bool:
        """Check if running on Windows."""
        return sys.platform == Platform.WINDOWS

    @staticmethod
    def is_linux() -> bool:
        """Check if running on Linux."""
        return sys.platform.startswith("linux")

    @staticmethod
    def is_macos() -> bool:
        """Check if running on macOS."""
        return sys.platform == Platform.MACOS

    @staticmethod
    def is_bsd() -> bool:
        """Check if running on any BSD variant."""
        return sys.platform in {Platform.FREEBSD, Platform.OPENBSD, Platform.NETBSD}

    @staticmethod
    def is_unix() -> bool:
        """Check if running on any Unix-like system (Linux, macOS, BSD)."""
        return os.name == Platform.NAME_POSIX

    @staticmethod
    def is_posix() -> bool:
        """Check if running on a POSIX-compliant system.

        This is equivalent to is_unix() but uses os.name for the check.
        """
        return os.name == Platform.NAME_POSIX

    @staticmethod
    def get_raw_name() -> str:
        """Get the raw sys.platform value.

        Returns:
            The raw sys.platform string (e.g., 'win32', 'linux', 'darwin').
        """
        return sys.platform

    @staticmethod
    def get_os_name() -> str:
        """Get the os.name value.

        Returns:
            'nt' for Windows, 'posix' for Unix-like systems.
        """
        return os.name


# Convenience functions for direct import
def is_windows() -> bool:
    """Check if running on Windows."""
    return Platform.is_windows()


def is_linux() -> bool:
    """Check if running on Linux."""
    return Platform.is_linux()


def is_macos() -> bool:
    """Check if running on macOS."""
    return Platform.is_macos()


def is_bsd() -> bool:
    """Check if running on any BSD variant."""
    return Platform.is_bsd()


def is_unix() -> bool:
    """Check if running on any Unix-like system (Linux, macOS, BSD)."""
    return Platform.is_unix()


def is_posix() -> bool:
    """Check if running on a POSIX-compliant system."""
    return Platform.is_posix()


# Module-level constants for backward compatibility
IS_WINDOWS: Final = is_windows()
IS_LINUX: Final = is_linux()
IS_MACOS: Final = is_macos()
IS_BSD: Final = is_bsd()
IS_UNIX: Final = is_unix()
IS_POSIX: Final = is_posix()


# Platform-specific module availability
def has_termios() -> bool:
    """Check if the termios module is available (Unix-like systems only)."""
    try:
        import termios  # noqa: F401

        return True
    except ImportError:
        return False


def has_fcntl() -> bool:
    """Check if the fcntl module is available (Unix-like systems only)."""
    try:
        import fcntl  # noqa: F401

        return True
    except ImportError:
        return False


def has_tty() -> bool:
    """Check if the tty module is available (Unix-like systems only)."""
    try:
        import tty  # noqa: F401

        return True
    except ImportError:
        return False


# Module-level constants for module availability
HAS_TERMIOS: Final = has_termios()
HAS_FCNTL: Final = has_fcntl()
HAS_TTY: Final = has_tty()
