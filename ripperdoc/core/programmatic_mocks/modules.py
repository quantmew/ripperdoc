"""Mock modules for programmatic execution sandbox.

This module provides safe mock replacements for dangerous system modules
(os, sys, subprocess, etc.) that allow code to import and use them without
errors, but dangerous operations will either return safe defaults or raise
PermissionError with helpful messages.
"""

from __future__ import annotations

import socket as real_socket
import tempfile as real_tempfile
from typing import Any, Dict, Iterator, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ripperdoc.core.programmatic_executor import ProgrammaticContext


# Modules that are forbidden in programmatic execution
FORBIDDEN_MODULES = frozenset({
    "os",
    "sys",
    "subprocess",
    "shutil",
    "pathlib",  # Direct file access
    "io",  # Direct file I/O
    "builtins",
    "__builtins__",
    "importlib",
    "ctypes",
    "socket",  # Network access (we provide mock)
    "http",
    "urllib",
    "requests",
    "httpx",
    "aiohttp",
    "ftplib",
    "smtplib",
    "telnetlib",
    "pickle",
    "marshal",
    "shelve",
    "dbm",
    "sqlite3",
    "multiprocessing",
    "threading",
    "concurrent",
    "signal",
    "resource",
    "pty",
    "tty",
    "termios",
    "fcntl",
    "mmap",
    "code",
    "codeop",
    "compile",
    "exec",
    "eval",
    "open",
    "file",
    "input",
    "raw_input",
    "glob",  # We provide mock with tool access
    "tempfile",  # We provide mock
})

# Allowed safe modules for data processing
ALLOWED_MODULES = frozenset({
    # Data formats
    "json",
    "csv",
    "configparser",
    "html",
    "xml",

    # Text processing
    "re",
    "string",
    "textwrap",
    "difflib",
    "unicodedata",

    # Math and numbers
    "math",
    "statistics",
    "decimal",
    "fractions",
    "numbers",
    "cmath",

    # Data structures
    "collections",
    "heapq",
    "bisect",
    "array",
    "graphlib",

    # Functional programming
    "itertools",
    "functools",
    "operator",

    # Date and time
    "datetime",
    "time",  # Only time.time(), not system calls
    "calendar",
    "zoneinfo",

    # Type system
    "typing",
    "types",
    "dataclasses",
    "enum",
    "abc",

    # Compression (in-memory)
    "zlib",
    "gzip",
    "bz2",
    "lzma",

    # Encoding
    "base64",
    "binascii",
    "quopri",
    "uu",

    # Hashing
    "hashlib",
    "hmac",

    # Other utilities
    "uuid",
    "fnmatch",
    "copy",
    "pprint",
    "reprlib",
    "weakref",

    # Code tools (safe parts)
    "ast",
    "keyword",
    "tokenize",
    "token",

    # Context and warnings
    "contextlib",
    "warnings",
    "traceback",

    # Binary data
    "struct",
    "codecs",

    # Async
    "asyncio",

    # System info (read-only)
    "platform",
    "locale",
    "getpass",  # Only getpass.getuser() is useful

    # Random
    "random",
    "secrets",
})


# ============================================================================
# Mock modules for dangerous system modules
# These allow code to import os/sys/subprocess without errors,
# but all operations are no-ops or return safe dummy values
# ============================================================================


class MockPath:
    """Mock path object that returns safe values (static version)."""

    def __init__(self, path: str = ""):
        self._path = str(path)

    def __str__(self) -> str:
        return self._path

    def __repr__(self) -> str:
        return f"MockPath({self._path!r})"

    def __truediv__(self, other: Any) -> "MockPath":
        return MockPath(f"{self._path}/{other}")

    def __fspath__(self) -> str:
        return self._path

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, MockPath):
            return self._path == other._path
        return self._path == str(other)

    def __hash__(self) -> int:
        return hash(self._path)

    @property
    def name(self) -> str:
        """Return the final component of the path."""
        parts = self._path.rstrip("/").split("/")
        return parts[-1] if parts else ""

    @property
    def suffix(self) -> str:
        """Return the file extension."""
        name = self.name
        if "." in name:
            return name[name.rfind("."):]
        return ""

    @property
    def suffixes(self) -> List[str]:
        """Return a list of the path's file extensions."""
        name = self.name
        if "." not in name:
            return []
        parts = name.split(".")
        return ["." + p for p in parts[1:]]

    @property
    def stem(self) -> str:
        """Return the final component minus the suffix."""
        name = self.name
        if "." in name:
            return name[:name.rfind(".")]
        return name

    @property
    def parent(self) -> "MockPath":
        """Return the parent path."""
        parts = self._path.rstrip("/").split("/")
        if len(parts) > 1:
            return MockPath("/".join(parts[:-1]))
        return MockPath("")

    @property
    def parts(self) -> tuple:
        """Return path components."""
        if not self._path:
            return ()
        parts = self._path.split("/")
        if self._path.startswith("/"):
            return ("/",) + tuple(p for p in parts if p)
        return tuple(p for p in parts if p)

    def exists(self) -> bool:
        return False

    def is_file(self) -> bool:
        return False

    def is_dir(self) -> bool:
        return False

    def is_symlink(self) -> bool:
        return False

    def is_absolute(self) -> bool:
        return self._path.startswith("/")

    def resolve(self) -> "MockPath":
        """Return absolute path (without resolving symlinks in mock)."""
        import os.path as real_path
        return MockPath(real_path.normpath(self._path))

    def absolute(self) -> "MockPath":
        """Return absolute path."""
        return self.resolve()

    def joinpath(self, *args: Any) -> "MockPath":
        """Join path components."""
        result = self._path
        for arg in args:
            result = f"{result}/{arg}"
        return MockPath(result)

    def with_suffix(self, suffix: str) -> "MockPath":
        """Return path with a different suffix."""
        stem = self.stem
        parent = str(self.parent)
        if parent:
            return MockPath(f"{parent}/{stem}{suffix}")
        return MockPath(f"{stem}{suffix}")

    def with_name(self, name: str) -> "MockPath":
        """Return path with a different name."""
        parent = str(self.parent)
        if parent:
            return MockPath(f"{parent}/{name}")
        return MockPath(name)

    def with_stem(self, stem: str) -> "MockPath":
        """Return path with a different stem."""
        return self.with_name(stem + self.suffix)

    def read_text(self, *args: Any, **kwargs: Any) -> str:
        raise PermissionError("Use ctx.tool_call('Read', {'file_path': ...}) instead")

    def write_text(self, *args: Any, **kwargs: Any) -> None:
        raise PermissionError("Use ctx.tool_call('Write', {'file_path': ...}) instead")

    def read_bytes(self, *args: Any, **kwargs: Any) -> bytes:
        raise PermissionError("Use ctx.tool_call('Read', {'file_path': ...}) instead")

    def write_bytes(self, *args: Any, **kwargs: Any) -> None:
        raise PermissionError("Use ctx.tool_call('Write', {'file_path': ...}) instead")

    def open(self, *args: Any, **kwargs: Any) -> Any:
        raise PermissionError("Use ctx.tool_call('Read', {'file_path': ...}) instead")

    def stat(self, *args: Any, **kwargs: Any) -> Any:
        raise PermissionError("Filesystem access not allowed")

    def lstat(self, *args: Any, **kwargs: Any) -> Any:
        raise PermissionError("Filesystem access not allowed")

    def mkdir(self, *args: Any, **kwargs: Any) -> None:
        raise PermissionError("File operations not allowed")

    def rmdir(self, *args: Any, **kwargs: Any) -> None:
        raise PermissionError("File operations not allowed")

    def unlink(self, *args: Any, **kwargs: Any) -> None:
        raise PermissionError("File operations not allowed")

    def rename(self, *args: Any, **kwargs: Any) -> "MockPath":
        raise PermissionError("File operations not allowed")

    def replace(self, *args: Any, **kwargs: Any) -> "MockPath":
        raise PermissionError("File operations not allowed")

    def touch(self, *args: Any, **kwargs: Any) -> None:
        raise PermissionError("File operations not allowed")

    def chmod(self, *args: Any, **kwargs: Any) -> None:
        raise PermissionError("File operations not allowed")

    def iterdir(self, *args: Any, **kwargs: Any) -> Any:
        raise PermissionError("Use ctx.tool_call('Glob', {'pattern': '*'}) instead")

    def glob(self, *args: Any, **kwargs: Any) -> Any:
        raise PermissionError("Use ctx.tool_call('Glob', {'pattern': ...}) instead")

    def rglob(self, *args: Any, **kwargs: Any) -> Any:
        raise PermissionError("Use ctx.tool_call('Glob', {'pattern': '**/*'}) instead")


class MockEnviron(dict):
    """Mock os.environ that returns empty values."""

    def get(self, key: str, default: Any = None) -> Any:
        return default

    def __getitem__(self, key: str) -> str:
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        return False

    def __setitem__(self, key: str, value: str) -> None:
        raise PermissionError("Cannot modify environment variables")

    def __delitem__(self, key: str) -> None:
        raise PermissionError("Cannot modify environment variables")

    def keys(self) -> list:  # type: ignore[override]
        return []

    def values(self) -> list:  # type: ignore[override]
        return []

    def items(self) -> list:  # type: ignore[override]
        return []


class MockOsPathModule:
    """Mock os.path module that provides real path utilities but blocks filesystem access."""

    # Import the real os.path for safe computational functions
    import os.path as _real_path

    # Safe computational functions - use real implementations
    join = staticmethod(_real_path.join)
    split = staticmethod(_real_path.split)
    splitext = staticmethod(_real_path.splitext)
    splitdrive = staticmethod(_real_path.splitdrive)
    basename = staticmethod(_real_path.basename)
    dirname = staticmethod(_real_path.dirname)
    normpath = staticmethod(_real_path.normpath)
    normcase = staticmethod(_real_path.normcase)
    isabs = staticmethod(_real_path.isabs)
    commonpath = staticmethod(_real_path.commonpath)
    commonprefix = staticmethod(_real_path.commonprefix)
    relpath = staticmethod(_real_path.relpath)

    # Path separator constants
    sep = _real_path.sep
    altsep = _real_path.altsep
    extsep = _real_path.extsep
    pardir = _real_path.pardir
    curdir = _real_path.curdir
    defpath = _real_path.defpath
    devnull = _real_path.devnull

    # Filesystem access functions - return safe defaults or block
    @staticmethod
    def exists(path: Any) -> bool:
        """Always returns False (no filesystem access)."""
        return False

    @staticmethod
    def isfile(path: Any) -> bool:
        """Always returns False (no filesystem access)."""
        return False

    @staticmethod
    def isdir(path: Any) -> bool:
        """Always returns False (no filesystem access)."""
        return False

    @staticmethod
    def islink(path: Any) -> bool:
        """Always returns False (no filesystem access)."""
        return False

    @staticmethod
    def ismount(path: Any) -> bool:
        """Always returns False (no filesystem access)."""
        return False

    @staticmethod
    def lexists(path: Any) -> bool:
        """Always returns False (no filesystem access)."""
        return False

    @staticmethod
    def getsize(path: Any) -> int:
        """Blocked - use Read tool instead."""
        raise PermissionError("Use ctx.tool_call('Read', ...) to get file info")

    @staticmethod
    def getmtime(path: Any) -> float:
        """Blocked - no filesystem access."""
        raise PermissionError("Filesystem access not allowed")

    @staticmethod
    def getatime(path: Any) -> float:
        """Blocked - no filesystem access."""
        raise PermissionError("Filesystem access not allowed")

    @staticmethod
    def getctime(path: Any) -> float:
        """Blocked - no filesystem access."""
        raise PermissionError("Filesystem access not allowed")

    @staticmethod
    def realpath(path: Any) -> str:
        """Returns normalized path (no symlink resolution)."""
        import os.path as _real_path
        return _real_path.normpath(str(path))

    @staticmethod
    def abspath(path: Any) -> str:
        """Returns path as-is (no cwd access)."""
        import os.path as _real_path
        p = str(path)
        if _real_path.isabs(p):
            return _real_path.normpath(p)
        return p

    @staticmethod
    def expanduser(path: Any) -> str:
        """Returns path as-is (no user info access)."""
        return str(path)

    @staticmethod
    def expandvars(path: Any) -> str:
        """Returns path as-is (no env var access)."""
        return str(path)


class MockOsModule:
    """Mock os module that provides safe utilities but blocks dangerous operations."""

    # Import real os module for safe constants
    import os as _real_os

    # Safe constants - use real values
    sep = _real_os.sep
    altsep = _real_os.altsep
    linesep = _real_os.linesep
    name = _real_os.name
    curdir = _real_os.curdir
    pardir = _real_os.pardir
    extsep = _real_os.extsep
    devnull = _real_os.devnull

    # Use the mock path module
    path = MockOsPathModule

    # Mock environ
    environ: Dict[str, str] = MockEnviron()  # type: ignore[assignment]

    # Safe utility functions
    @staticmethod
    def fspath(path: Any) -> str:
        """Return the string representation of a path."""
        if hasattr(path, "__fspath__"):
            return str(path.__fspath__())
        return str(path)

    @staticmethod
    def fsencode(filename: Any) -> bytes:
        """Encode filename to bytes."""
        import os as _real_os
        return _real_os.fsencode(str(filename))

    @staticmethod
    def fsdecode(filename: Any) -> str:
        """Decode filename to string."""
        import os as _real_os
        if isinstance(filename, bytes):
            return _real_os.fsdecode(filename)
        return str(filename)

    @staticmethod
    def getcwd() -> str:
        """Returns a mock working directory."""
        return "/mock/working/directory"

    @staticmethod
    def getenv(key: str, default: Any = None) -> Any:
        """Always returns default (no env access)."""
        return default

    @staticmethod
    def getpid() -> int:
        """Returns a mock PID."""
        return 12345

    @staticmethod
    def getuid() -> int:
        """Returns a mock UID."""
        return 1000

    @staticmethod
    def getgid() -> int:
        """Returns a mock GID."""
        return 1000

    @staticmethod
    def getlogin() -> str:
        """Returns a mock login name."""
        return "user"

    # Blocked functions - filesystem operations
    @staticmethod
    def listdir(path: Any = ".") -> list:
        raise PermissionError("Use ctx.tool_call('Glob', {'pattern': '*'}) or os.listdir() with dynamic mock")

    @staticmethod
    def scandir(path: Any = ".") -> Any:
        raise PermissionError("Use ctx.tool_call('Glob', {'pattern': '*'}) instead")

    @staticmethod
    def stat(path: Any, *args: Any, **kwargs: Any) -> Any:
        raise PermissionError("Filesystem access not allowed")

    @staticmethod
    def lstat(path: Any, *args: Any, **kwargs: Any) -> Any:
        raise PermissionError("Filesystem access not allowed")

    @staticmethod
    def access(path: Any, mode: int, *args: Any, **kwargs: Any) -> bool:
        raise PermissionError("Filesystem access not allowed")

    # Blocked functions - command execution
    @staticmethod
    def system(cmd: str) -> int:
        raise PermissionError("Use ctx.tool_call('Bash', {'command': ...}) instead")

    @staticmethod
    def popen(cmd: str, *args: Any, **kwargs: Any) -> Any:
        raise PermissionError("Use ctx.tool_call('Bash', {'command': ...}) instead")

    @staticmethod
    def spawn(mode: int, path: Any, *args: Any) -> int:
        raise PermissionError("Use ctx.tool_call('Bash', {'command': ...}) instead")

    @staticmethod
    def spawnl(mode: int, path: Any, *args: Any) -> int:
        raise PermissionError("Use ctx.tool_call('Bash', {'command': ...}) instead")

    @staticmethod
    def spawnle(mode: int, path: Any, *args: Any, **kwargs: Any) -> int:
        raise PermissionError("Use ctx.tool_call('Bash', {'command': ...}) instead")

    @staticmethod
    def spawnlp(mode: int, file: Any, *args: Any) -> int:
        raise PermissionError("Use ctx.tool_call('Bash', {'command': ...}) instead")

    @staticmethod
    def spawnlpe(mode: int, file: Any, *args: Any, **kwargs: Any) -> int:
        raise PermissionError("Use ctx.tool_call('Bash', {'command': ...}) instead")

    @staticmethod
    def spawnv(mode: int, path: Any, args: Any) -> int:
        raise PermissionError("Use ctx.tool_call('Bash', {'command': ...}) instead")

    @staticmethod
    def spawnve(mode: int, path: Any, args: Any, env: Any) -> int:
        raise PermissionError("Use ctx.tool_call('Bash', {'command': ...}) instead")

    @staticmethod
    def spawnvp(mode: int, file: Any, args: Any) -> int:
        raise PermissionError("Use ctx.tool_call('Bash', {'command': ...}) instead")

    @staticmethod
    def spawnvpe(mode: int, file: Any, args: Any, env: Any) -> int:
        raise PermissionError("Use ctx.tool_call('Bash', {'command': ...}) instead")

    @staticmethod
    def execl(path: Any, *args: Any) -> None:
        raise PermissionError("Use ctx.tool_call('Bash', {'command': ...}) instead")

    @staticmethod
    def execle(path: Any, *args: Any) -> None:
        raise PermissionError("Use ctx.tool_call('Bash', {'command': ...}) instead")

    @staticmethod
    def execlp(file: Any, *args: Any) -> None:
        raise PermissionError("Use ctx.tool_call('Bash', {'command': ...}) instead")

    @staticmethod
    def execlpe(file: Any, *args: Any) -> None:
        raise PermissionError("Use ctx.tool_call('Bash', {'command': ...}) instead")

    @staticmethod
    def execv(path: Any, args: Any) -> None:
        raise PermissionError("Use ctx.tool_call('Bash', {'command': ...}) instead")

    @staticmethod
    def execve(path: Any, args: Any, env: Any) -> None:
        raise PermissionError("Use ctx.tool_call('Bash', {'command': ...}) instead")

    @staticmethod
    def execvp(file: Any, args: Any) -> None:
        raise PermissionError("Use ctx.tool_call('Bash', {'command': ...}) instead")

    @staticmethod
    def execvpe(file: Any, args: Any, env: Any) -> None:
        raise PermissionError("Use ctx.tool_call('Bash', {'command': ...}) instead")

    # Blocked functions - file operations
    @staticmethod
    def remove(path: Any) -> None:
        raise PermissionError("File operations not allowed")

    @staticmethod
    def unlink(path: Any, *args: Any, **kwargs: Any) -> None:
        raise PermissionError("File operations not allowed")

    @staticmethod
    def rmdir(path: Any, *args: Any, **kwargs: Any) -> None:
        raise PermissionError("File operations not allowed")

    @staticmethod
    def removedirs(name: Any) -> None:
        raise PermissionError("File operations not allowed")

    @staticmethod
    def mkdir(path: Any, *args: Any, **kwargs: Any) -> None:
        raise PermissionError("File operations not allowed")

    @staticmethod
    def makedirs(name: Any, *args: Any, **kwargs: Any) -> None:
        raise PermissionError("File operations not allowed")

    @staticmethod
    def rename(src: Any, dst: Any, *args: Any, **kwargs: Any) -> None:
        raise PermissionError("File operations not allowed")

    @staticmethod
    def renames(old: Any, new: Any) -> None:
        raise PermissionError("File operations not allowed")

    @staticmethod
    def replace(src: Any, dst: Any, *args: Any, **kwargs: Any) -> None:
        raise PermissionError("File operations not allowed")

    @staticmethod
    def link(src: Any, dst: Any, *args: Any, **kwargs: Any) -> None:
        raise PermissionError("File operations not allowed")

    @staticmethod
    def symlink(src: Any, dst: Any, *args: Any, **kwargs: Any) -> None:
        raise PermissionError("File operations not allowed")

    @staticmethod
    def readlink(path: Any, *args: Any, **kwargs: Any) -> str:
        raise PermissionError("File operations not allowed")

    @staticmethod
    def chmod(path: Any, mode: int, *args: Any, **kwargs: Any) -> None:
        raise PermissionError("File operations not allowed")

    @staticmethod
    def chown(path: Any, uid: int, gid: int, *args: Any, **kwargs: Any) -> None:
        raise PermissionError("File operations not allowed")

    @staticmethod
    def chdir(path: Any) -> None:
        raise PermissionError("Directory operations not allowed")

    @staticmethod
    def chroot(path: Any) -> None:
        raise PermissionError("Directory operations not allowed")

    @staticmethod
    def walk(top: Any, *args: Any, **kwargs: Any) -> Any:
        raise PermissionError("Use ctx.tool_call('Glob', {'pattern': '**/*'}) instead")

    @staticmethod
    def fwalk(top: Any, *args: Any, **kwargs: Any) -> Any:
        raise PermissionError("Use ctx.tool_call('Glob', {'pattern': '**/*'}) instead")

    # Blocked functions - file descriptor operations
    @staticmethod
    def open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        raise PermissionError("Use ctx.tool_call('Read', ...) instead")

    @staticmethod
    def close(fd: int) -> None:
        raise PermissionError("File descriptor operations not allowed")

    @staticmethod
    def read(fd: int, n: int) -> bytes:
        raise PermissionError("Use ctx.tool_call('Read', ...) instead")

    @staticmethod
    def write(fd: int, data: bytes) -> int:
        raise PermissionError("Use ctx.tool_call('Write', ...) instead")


class MockSysModule:
    """Mock sys module that provides safe dummy values."""

    version = "3.12.0 (mock)"
    version_info = (3, 12, 0, "final", 0)
    platform = "linux"
    executable = "/usr/bin/python3"
    prefix = "/usr"
    exec_prefix = "/usr"
    path: List[str] = ["/mock/path"]
    modules: Dict[str, Any] = {}
    argv: List[str] = ["script.py"]
    stdin = None
    stdout = None
    stderr = None
    maxsize = 2**63 - 1
    byteorder = "little"
    float_info = type("float_info", (), {"max": 1.7976931348623157e+308})()
    int_info = type("int_info", (), {"bits_per_digit": 30})()
    hash_info = type("hash_info", (), {"width": 64})()

    @staticmethod
    def exit(code: int = 0) -> None:
        raise PermissionError("sys.exit() is not allowed in programmatic code")

    @staticmethod
    def getrecursionlimit() -> int:
        return 1000

    @staticmethod
    def setrecursionlimit(limit: int) -> None:
        pass  # No-op

    @staticmethod
    def getsizeof(obj: Any, default: int = 0) -> int:
        """Return approximate size of object."""
        import sys as real_sys
        try:
            return real_sys.getsizeof(obj, default)
        except Exception:
            return default

    @staticmethod
    def getdefaultencoding() -> str:
        return "utf-8"

    @staticmethod
    def getfilesystemencoding() -> str:
        return "utf-8"


class MockSubprocessModule:
    """Mock subprocess module that blocks all execution."""

    PIPE = -1
    STDOUT = -2
    DEVNULL = -3

    class CompletedProcess:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = b""
            self.stderr = b""

    @staticmethod
    def run(*args: Any, **kwargs: Any) -> Any:
        raise PermissionError("Use ctx.tool_call('Bash', {'command': ...}) instead")

    @staticmethod
    def call(*args: Any, **kwargs: Any) -> int:
        raise PermissionError("Use ctx.tool_call('Bash', {'command': ...}) instead")

    @staticmethod
    def check_call(*args: Any, **kwargs: Any) -> int:
        raise PermissionError("Use ctx.tool_call('Bash', {'command': ...}) instead")

    @staticmethod
    def check_output(*args: Any, **kwargs: Any) -> bytes:
        raise PermissionError("Use ctx.tool_call('Bash', {'command': ...}) instead")

    @staticmethod
    def Popen(*args: Any, **kwargs: Any) -> Any:
        raise PermissionError("Use ctx.tool_call('Bash', {'command': ...}) instead")


class MockShutilModule:
    """Mock shutil module that blocks file operations."""

    @staticmethod
    def copy(*args: Any, **kwargs: Any) -> None:
        raise PermissionError("File operations not allowed")

    @staticmethod
    def copy2(*args: Any, **kwargs: Any) -> None:
        raise PermissionError("File operations not allowed")

    @staticmethod
    def copytree(*args: Any, **kwargs: Any) -> None:
        raise PermissionError("File operations not allowed")

    @staticmethod
    def move(*args: Any, **kwargs: Any) -> None:
        raise PermissionError("File operations not allowed")

    @staticmethod
    def rmtree(*args: Any, **kwargs: Any) -> None:
        raise PermissionError("File operations not allowed")

    @staticmethod
    def which(cmd: str) -> Optional[str]:
        return None

    @staticmethod
    def disk_usage(path: Any) -> Any:
        raise PermissionError("Filesystem access not allowed")

    @staticmethod
    def get_terminal_size(fallback: tuple = (80, 24)) -> tuple:
        return fallback


class MockPathlibModule:
    """Mock pathlib module (static version)."""

    Path = MockPath
    PurePath = MockPath
    PurePosixPath = MockPath
    PureWindowsPath = MockPath
    PosixPath = MockPath
    WindowsPath = MockPath


class MockIoModule:
    """Mock io module that blocks file operations."""

    @staticmethod
    def open(*args: Any, **kwargs: Any) -> Any:
        raise PermissionError("Use ctx.tool_call('Read', {'file_path': ...}) instead")

    class StringIO:
        def __init__(self, initial: str = "") -> None:
            self._buffer = initial
            self._pos = 0

        def write(self, s: str) -> int:
            self._buffer += s
            return len(s)

        def read(self, size: int = -1) -> str:
            if size < 0:
                result = self._buffer[self._pos:]
                self._pos = len(self._buffer)
            else:
                result = self._buffer[self._pos:self._pos + size]
                self._pos += len(result)
            return result

        def readline(self) -> str:
            nl = self._buffer.find("\n", self._pos)
            if nl < 0:
                return self.read()
            result = self._buffer[self._pos:nl + 1]
            self._pos = nl + 1
            return result

        def getvalue(self) -> str:
            return self._buffer

        def seek(self, pos: int, whence: int = 0) -> int:
            if whence == 0:
                self._pos = pos
            elif whence == 1:
                self._pos += pos
            elif whence == 2:
                self._pos = len(self._buffer) + pos
            return self._pos

        def tell(self) -> int:
            return self._pos

        def close(self) -> None:
            pass

        def __enter__(self) -> "MockIoModule.StringIO":
            return self

        def __exit__(self, *args: Any) -> None:
            pass

    class BytesIO:
        def __init__(self, initial: bytes = b"") -> None:
            self._buffer = initial
            self._pos = 0

        def write(self, b: bytes) -> int:
            self._buffer += b
            return len(b)

        def read(self, size: int = -1) -> bytes:
            if size < 0:
                result = self._buffer[self._pos:]
                self._pos = len(self._buffer)
            else:
                result = self._buffer[self._pos:self._pos + size]
                self._pos += len(result)
            return result

        def readline(self) -> bytes:
            nl = self._buffer.find(b"\n", self._pos)
            if nl < 0:
                return self.read()
            result = self._buffer[self._pos:nl + 1]
            self._pos = nl + 1
            return result

        def getvalue(self) -> bytes:
            return self._buffer

        def seek(self, pos: int, whence: int = 0) -> int:
            if whence == 0:
                self._pos = pos
            elif whence == 1:
                self._pos += pos
            elif whence == 2:
                self._pos = len(self._buffer) + pos
            return self._pos

        def tell(self) -> int:
            return self._pos

        def close(self) -> None:
            pass

        def __enter__(self) -> "MockIoModule.BytesIO":
            return self

        def __exit__(self, *args: Any) -> None:
            pass


class MockGlobModule:
    """Mock glob module (static version that blocks filesystem access)."""

    @staticmethod
    def glob(pathname: str, *args: Any, **kwargs: Any) -> List[str]:
        raise PermissionError("Use ctx.tool_call('Glob', {'pattern': ...}) instead")

    @staticmethod
    def iglob(pathname: str, *args: Any, **kwargs: Any) -> Iterator[str]:
        raise PermissionError("Use ctx.tool_call('Glob', {'pattern': ...}) instead")

    @staticmethod
    def escape(pathname: str) -> str:
        """Escape special characters in pathname."""
        import glob as real_glob
        return real_glob.escape(pathname)


class MockTempfileModule:
    """Mock tempfile module with safe read-only operations."""

    # Use real tempfile values for these
    tempdir = real_tempfile.gettempdir()

    @staticmethod
    def gettempdir() -> str:
        """Return the directory used for temporary files."""
        return real_tempfile.gettempdir()

    @staticmethod
    def gettempdirb() -> bytes:
        """Return the directory used for temporary files as bytes."""
        return real_tempfile.gettempdir().encode()

    @staticmethod
    def gettempprefix() -> str:
        """Return the filename prefix for temporary files."""
        return "tmp"

    @staticmethod
    def gettempprefixb() -> bytes:
        """Return the filename prefix for temporary files as bytes."""
        return b"tmp"

    @staticmethod
    def mktemp(suffix: str = "", prefix: str = "tmp", dir: Optional[str] = None) -> str:
        """Generate a temporary filename (without creating it)."""
        import uuid
        base = dir or real_tempfile.gettempdir()
        return f"{base}/{prefix}{uuid.uuid4().hex}{suffix}"

    # Block functions that create files
    @staticmethod
    def TemporaryFile(*args: Any, **kwargs: Any) -> Any:
        raise PermissionError("Temporary file creation not allowed")

    @staticmethod
    def NamedTemporaryFile(*args: Any, **kwargs: Any) -> Any:
        raise PermissionError("Temporary file creation not allowed")

    @staticmethod
    def SpooledTemporaryFile(*args: Any, **kwargs: Any) -> Any:
        raise PermissionError("Temporary file creation not allowed")

    @staticmethod
    def TemporaryDirectory(*args: Any, **kwargs: Any) -> Any:
        raise PermissionError("Temporary directory creation not allowed")

    @staticmethod
    def mkdtemp(*args: Any, **kwargs: Any) -> str:
        raise PermissionError("Temporary directory creation not allowed")

    @staticmethod
    def mkstemp(*args: Any, **kwargs: Any) -> tuple:
        raise PermissionError("Temporary file creation not allowed")


class MockSocketModule:
    """Mock socket module with safe read-only operations."""

    # Socket constants
    AF_INET = real_socket.AF_INET
    AF_INET6 = real_socket.AF_INET6
    AF_UNIX = getattr(real_socket, "AF_UNIX", 1)
    SOCK_STREAM = real_socket.SOCK_STREAM
    SOCK_DGRAM = real_socket.SOCK_DGRAM

    @staticmethod
    def gethostname() -> str:
        """Return the current host name."""
        return real_socket.gethostname()

    @staticmethod
    def getfqdn(name: str = "") -> str:
        """Return a fully qualified domain name."""
        return real_socket.getfqdn(name)

    @staticmethod
    def gethostbyname(hostname: str) -> str:
        """Return IP address for hostname (may involve network, be careful)."""
        # This is relatively safe as it's read-only
        return real_socket.gethostbyname(hostname)

    @staticmethod
    def gethostbyaddr(ip_address: str) -> tuple:
        """Return hostname for IP address."""
        return real_socket.gethostbyaddr(ip_address)

    @staticmethod
    def getservbyname(servicename: str, protocolname: str = "tcp") -> int:
        """Return port number for service name."""
        return real_socket.getservbyname(servicename, protocolname)

    @staticmethod
    def getservbyport(port: int, protocolname: str = "tcp") -> str:
        """Return service name for port number."""
        return real_socket.getservbyport(port, protocolname)

    @staticmethod
    def inet_aton(ip_string: str) -> bytes:
        """Convert IP address string to packed binary format."""
        return real_socket.inet_aton(ip_string)

    @staticmethod
    def inet_ntoa(packed_ip: bytes) -> str:
        """Convert packed binary IP to string."""
        return real_socket.inet_ntoa(packed_ip)

    @staticmethod
    def inet_pton(address_family: int, ip_string: str) -> bytes:
        """Convert IP address string to packed binary format."""
        return real_socket.inet_pton(address_family, ip_string)

    @staticmethod
    def inet_ntop(address_family: int, packed_ip: bytes) -> str:
        """Convert packed binary IP to string."""
        return real_socket.inet_ntop(address_family, packed_ip)

    @staticmethod
    def htons(x: int) -> int:
        """Convert host to network byte order (short)."""
        return real_socket.htons(x)

    @staticmethod
    def htonl(x: int) -> int:
        """Convert host to network byte order (long)."""
        return real_socket.htonl(x)

    @staticmethod
    def ntohs(x: int) -> int:
        """Convert network to host byte order (short)."""
        return real_socket.ntohs(x)

    @staticmethod
    def ntohl(x: int) -> int:
        """Convert network to host byte order (long)."""
        return real_socket.ntohl(x)

    # Block functions that create sockets
    @staticmethod
    def socket(*args: Any, **kwargs: Any) -> Any:
        raise PermissionError("Socket creation not allowed")

    @staticmethod
    def create_connection(*args: Any, **kwargs: Any) -> Any:
        raise PermissionError("Network connections not allowed")

    @staticmethod
    def create_server(*args: Any, **kwargs: Any) -> Any:
        raise PermissionError("Server creation not allowed")


# Create singleton instances of mock modules
MOCK_OS = MockOsModule()
MOCK_SYS = MockSysModule()
MOCK_SUBPROCESS = MockSubprocessModule()
MOCK_SHUTIL = MockShutilModule()
MOCK_PATHLIB = MockPathlibModule()
MOCK_IO = MockIoModule()
MOCK_GLOB = MockGlobModule()
MOCK_TEMPFILE = MockTempfileModule()
MOCK_SOCKET = MockSocketModule()


def create_dynamic_mock_os(ctx: "ProgrammaticContext") -> Any:
    """Create a dynamic mock os module with ctx access for filesystem operations.

    This allows os.listdir(), os.walk(), and os.path.exists/isfile/isdir to
    actually work by calling the Glob tool through ctx.tool_call().
    """
    import os.path as real_path

    class DynamicMockOsPathModule:
        """Dynamic mock os.path module with ctx access for exists/isfile/isdir."""

        # Import the real os.path for safe computational functions
        import os.path as _real_path

        # Safe computational functions - use real implementations
        join = staticmethod(_real_path.join)
        split = staticmethod(_real_path.split)
        splitext = staticmethod(_real_path.splitext)
        splitdrive = staticmethod(_real_path.splitdrive)
        basename = staticmethod(_real_path.basename)
        dirname = staticmethod(_real_path.dirname)
        normpath = staticmethod(_real_path.normpath)
        normcase = staticmethod(_real_path.normcase)
        isabs = staticmethod(_real_path.isabs)
        commonpath = staticmethod(_real_path.commonpath)
        commonprefix = staticmethod(_real_path.commonprefix)
        relpath = staticmethod(_real_path.relpath)

        # Path separator constants
        sep = _real_path.sep
        altsep = _real_path.altsep
        extsep = _real_path.extsep
        pardir = _real_path.pardir
        curdir = _real_path.curdir
        defpath = _real_path.defpath
        devnull = _real_path.devnull

        @staticmethod
        def exists(path: Any) -> bool:
            """Check if path exists using Glob tool."""
            path_str = str(path)
            if not real_path.isabs(path_str):
                path_str = real_path.join(ctx.get_working_directory(), path_str)

            # Try to glob the exact path
            parent = real_path.dirname(path_str)
            name = real_path.basename(path_str)
            if not name:
                # Root directory always exists
                return True

            result = ctx.tool_call("Glob", {"pattern": name, "path": parent})
            if result and result.get("matches"):
                # Check if exact match exists
                for match in result["matches"]:
                    if real_path.basename(match) == name:
                        return True
            return False

        @staticmethod
        def isfile(path: Any) -> bool:
            """Check if path is a file using Glob tool."""
            path_str = str(path)
            if not real_path.isabs(path_str):
                path_str = real_path.join(ctx.get_working_directory(), path_str)

            parent = real_path.dirname(path_str)
            name = real_path.basename(path_str)
            if not name:
                return False

            result = ctx.tool_call("Glob", {"pattern": name, "path": parent})
            if result and result.get("matches"):
                for match in result["matches"]:
                    if real_path.basename(match) == name:
                        # Glob only returns files, so if it's in matches, it's a file
                        return True
            return False

        @staticmethod
        def isdir(path: Any) -> bool:
            """Check if path is a directory by checking if it has contents."""
            path_str = str(path)
            if not real_path.isabs(path_str):
                path_str = real_path.join(ctx.get_working_directory(), path_str)

            # Try to list contents - if successful, it's a directory
            result = ctx.tool_call("Glob", {"pattern": "*", "path": path_str})
            # If we get any result (even empty matches without error), it's a directory
            if result is not None and "matches" in result:
                return True
            return False

        @staticmethod
        def islink(path: Any) -> bool:
            """Always returns False (symlink detection not supported)."""
            return False

        @staticmethod
        def ismount(path: Any) -> bool:
            """Always returns False (mount detection not supported)."""
            return False

        @staticmethod
        def lexists(path: Any) -> bool:
            """Same as exists (symlink detection not supported)."""
            return DynamicMockOsPathModule.exists(path)

        @staticmethod
        def getsize(path: Any) -> int:
            """Get file size using Read tool."""
            path_str = str(path)
            if not real_path.isabs(path_str):
                path_str = real_path.join(ctx.get_working_directory(), path_str)

            result = ctx.tool_call("Read", {"file_path": path_str})
            if result and "content" in result:
                return len(result["content"])
            raise FileNotFoundError(f"No such file: {path_str}")

        @staticmethod
        def getmtime(path: Any) -> float:
            raise PermissionError("File modification time not available")

        @staticmethod
        def getatime(path: Any) -> float:
            raise PermissionError("File access time not available")

        @staticmethod
        def getctime(path: Any) -> float:
            raise PermissionError("File creation time not available")

        @staticmethod
        def realpath(path: Any) -> str:
            return real_path.normpath(str(path))

        @staticmethod
        def abspath(path: Any) -> str:
            p = str(path)
            if real_path.isabs(p):
                return real_path.normpath(p)
            return real_path.normpath(real_path.join(ctx.get_working_directory(), p))

        @staticmethod
        def expanduser(path: Any) -> str:
            return str(path)

        @staticmethod
        def expandvars(path: Any) -> str:
            return str(path)

    class DynamicMockOsModule:
        """Dynamic mock os module that can call tools via ctx."""

        import os as _real_os

        sep = _real_os.sep
        altsep = _real_os.altsep
        linesep = _real_os.linesep
        name = _real_os.name
        curdir = _real_os.curdir
        pardir = _real_os.pardir
        extsep = _real_os.extsep
        devnull = _real_os.devnull

        # Use the dynamic path module
        path = DynamicMockOsPathModule

        environ: Dict[str, str] = MockEnviron()  # type: ignore[assignment]

        @staticmethod
        def fspath(path: Any) -> str:
            if hasattr(path, "__fspath__"):
                return str(path.__fspath__())
            return str(path)

        @staticmethod
        def fsencode(filename: Any) -> bytes:
            import os as _real_os
            return _real_os.fsencode(str(filename))

        @staticmethod
        def fsdecode(filename: Any) -> str:
            import os as _real_os
            if isinstance(filename, bytes):
                return _real_os.fsdecode(filename)
            return str(filename)

        @staticmethod
        def getcwd() -> str:
            return ctx.get_working_directory()

        @staticmethod
        def getenv(key: str, default: Any = None) -> Any:
            return default

        @staticmethod
        def getpid() -> int:
            return 12345

        @staticmethod
        def getuid() -> int:
            return 1000

        @staticmethod
        def getgid() -> int:
            return 1000

        @staticmethod
        def getlogin() -> str:
            return "user"

        @staticmethod
        def listdir(path: Any = ".") -> List[str]:
            """List directory contents using Glob tool."""
            path_str = str(path)
            if not real_path.isabs(path_str):
                path_str = real_path.join(ctx.get_working_directory(), path_str)

            result = ctx.tool_call("Glob", {"pattern": "*", "path": path_str})
            if result and result.get("matches"):
                return [real_path.basename(f) for f in result["matches"]]
            return []

        @staticmethod
        def walk(
            top: Any,
            topdown: bool = True,
            onerror: Any = None,
            followlinks: bool = False,
        ) -> Any:
            """Walk directory tree using Glob tool."""
            top_str = str(top)
            if not real_path.isabs(top_str):
                top_str = real_path.join(ctx.get_working_directory(), top_str)

            result = ctx.tool_call("Glob", {"pattern": "**/*", "path": top_str})

            if not result or not result.get("matches"):
                yield (top_str, [], [])
                return

            dirs_dict: Dict[str, tuple] = {}

            for filepath in result["matches"]:
                filepath = real_path.normpath(filepath)

                if filepath.startswith(top_str):
                    rel_path = filepath[len(top_str):].lstrip("/\\")
                else:
                    rel_path = real_path.basename(filepath)

                rel_dir = real_path.dirname(rel_path)
                filename = real_path.basename(rel_path)

                if rel_dir:
                    full_dirpath = real_path.normpath(real_path.join(top_str, rel_dir))
                else:
                    full_dirpath = top_str

                if full_dirpath not in dirs_dict:
                    dirs_dict[full_dirpath] = (set(), set())

                dirs_dict[full_dirpath][1].add(filename)

                if rel_dir:
                    parts = rel_dir.replace("\\", "/").split("/")
                    for i in range(len(parts)):
                        parent_rel = "/".join(parts[:i]) if i > 0 else ""
                        parent_full = (
                            real_path.normpath(real_path.join(top_str, parent_rel))
                            if parent_rel
                            else top_str
                        )
                        child_name = parts[i]

                        if parent_full not in dirs_dict:
                            dirs_dict[parent_full] = (set(), set())
                        dirs_dict[parent_full][0].add(child_name)

            if top_str not in dirs_dict:
                dirs_dict[top_str] = (set(), set())

            if topdown:
                sorted_dirs = sorted(dirs_dict.keys())
            else:
                sorted_dirs = sorted(dirs_dict.keys(), reverse=True)

            for dirpath in sorted_dirs:
                dirnames_set, filenames_set = dirs_dict[dirpath]
                yield (dirpath, sorted(dirnames_set), sorted(filenames_set))

        @staticmethod
        def scandir(path: Any = ".") -> Any:
            raise PermissionError("Use os.listdir() instead")

        @staticmethod
        def stat(path: Any, *args: Any, **kwargs: Any) -> Any:
            raise PermissionError("Filesystem access not allowed")

        @staticmethod
        def lstat(path: Any, *args: Any, **kwargs: Any) -> Any:
            raise PermissionError("Filesystem access not allowed")

        @staticmethod
        def access(path: Any, mode: int, *args: Any, **kwargs: Any) -> bool:
            raise PermissionError("Filesystem access not allowed")

        @staticmethod
        def system(cmd: str) -> int:
            raise PermissionError("Use ctx.tool_call('Bash', ...) instead")

        @staticmethod
        def popen(cmd: str, *args: Any, **kwargs: Any) -> Any:
            raise PermissionError("Use ctx.tool_call('Bash', ...) instead")

        # All other dangerous methods
        @staticmethod
        def remove(path: Any) -> None:
            raise PermissionError("File operations not allowed")

        @staticmethod
        def unlink(path: Any, *args: Any, **kwargs: Any) -> None:
            raise PermissionError("File operations not allowed")

        @staticmethod
        def rmdir(path: Any, *args: Any, **kwargs: Any) -> None:
            raise PermissionError("File operations not allowed")

        @staticmethod
        def mkdir(path: Any, *args: Any, **kwargs: Any) -> None:
            raise PermissionError("File operations not allowed")

        @staticmethod
        def makedirs(name: Any, *args: Any, **kwargs: Any) -> None:
            raise PermissionError("File operations not allowed")

        @staticmethod
        def rename(src: Any, dst: Any, *args: Any, **kwargs: Any) -> None:
            raise PermissionError("File operations not allowed")

        @staticmethod
        def replace(src: Any, dst: Any, *args: Any, **kwargs: Any) -> None:
            raise PermissionError("File operations not allowed")

        @staticmethod
        def chmod(path: Any, mode: int, *args: Any, **kwargs: Any) -> None:
            raise PermissionError("File operations not allowed")

        @staticmethod
        def chown(path: Any, uid: int, gid: int, *args: Any, **kwargs: Any) -> None:
            raise PermissionError("File operations not allowed")

        @staticmethod
        def chdir(path: Any) -> None:
            raise PermissionError("Directory operations not allowed")

        @staticmethod
        def fwalk(top: Any, *args: Any, **kwargs: Any) -> Any:
            raise PermissionError("Use os.walk() instead")

        @staticmethod
        def open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
            raise PermissionError("Use ctx.tool_call('Read', ...) instead")

        @staticmethod
        def close(fd: int) -> None:
            raise PermissionError("File descriptor operations not allowed")

        @staticmethod
        def read(fd: int, n: int) -> bytes:
            raise PermissionError("Use ctx.tool_call('Read', ...) instead")

        @staticmethod
        def write(fd: int, data: bytes) -> int:
            raise PermissionError("Use ctx.tool_call('Write', ...) instead")

    return DynamicMockOsModule()


def create_dynamic_mock_glob(ctx: "ProgrammaticContext") -> Any:
    """Create a dynamic mock glob module with ctx access."""
    import os.path as real_path

    class DynamicMockGlobModule:
        """Dynamic mock glob module that uses Glob tool."""

        @staticmethod
        def glob(pathname: str, *, root_dir: Optional[str] = None,
                 dir_fd: Any = None, recursive: bool = False) -> List[str]:
            """Find files matching pattern using Glob tool."""
            base_path = root_dir or ctx.get_working_directory()
            if not real_path.isabs(base_path):
                base_path = real_path.join(ctx.get_working_directory(), base_path)

            # Handle recursive patterns
            if recursive and "**" in pathname:
                pattern = pathname
            else:
                pattern = pathname

            result = ctx.tool_call("Glob", {"pattern": pattern, "path": base_path})
            if result and result.get("matches"):
                return list(result["matches"])
            return []

        @staticmethod
        def iglob(pathname: str, *, root_dir: Optional[str] = None,
                  dir_fd: Any = None, recursive: bool = False) -> Iterator[str]:
            """Iterator version of glob."""
            matches = DynamicMockGlobModule.glob(
                pathname, root_dir=root_dir, dir_fd=dir_fd, recursive=recursive
            )
            yield from matches

        @staticmethod
        def escape(pathname: str) -> str:
            """Escape special characters."""
            import glob as real_glob
            return real_glob.escape(pathname)

        @staticmethod
        def has_magic(s: str) -> bool:
            """Check if string has glob magic characters."""
            return bool(set(s) & set("*?["))

    return DynamicMockGlobModule()


def create_dynamic_mock_pathlib(ctx: "ProgrammaticContext") -> Any:
    """Create a dynamic mock pathlib module with ctx access."""
    import os.path as real_path

    class DynamicMockPath:
        """Dynamic mock Path that uses Glob tool for filesystem operations."""

        def __init__(self, path: str = "."):
            self._path = str(path)

        def __str__(self) -> str:
            return self._path

        def __repr__(self) -> str:
            return f"DynamicMockPath({self._path!r})"

        def __truediv__(self, other: Any) -> "DynamicMockPath":
            return DynamicMockPath(real_path.join(self._path, str(other)))

        def __rtruediv__(self, other: Any) -> "DynamicMockPath":
            return DynamicMockPath(real_path.join(str(other), self._path))

        def __fspath__(self) -> str:
            return self._path

        def __eq__(self, other: Any) -> bool:
            if isinstance(other, DynamicMockPath):
                return self._path == other._path
            return self._path == str(other)

        def __hash__(self) -> int:
            return hash(self._path)

        def _resolve_path(self) -> str:
            """Get absolute path."""
            if real_path.isabs(self._path):
                return real_path.normpath(self._path)
            return real_path.normpath(
                real_path.join(ctx.get_working_directory(), self._path)
            )

        @property
        def name(self) -> str:
            return real_path.basename(self._path)

        @property
        def suffix(self) -> str:
            name = self.name
            if "." in name:
                return name[name.rfind("."):]
            return ""

        @property
        def suffixes(self) -> List[str]:
            name = self.name
            if "." not in name:
                return []
            parts = name.split(".")
            return ["." + p for p in parts[1:]]

        @property
        def stem(self) -> str:
            name = self.name
            if "." in name:
                return name[:name.rfind(".")]
            return name

        @property
        def parent(self) -> "DynamicMockPath":
            return DynamicMockPath(real_path.dirname(self._path))

        @property
        def parts(self) -> tuple:
            if not self._path:
                return ()
            parts = self._path.split("/")
            if self._path.startswith("/"):
                return ("/",) + tuple(p for p in parts if p)
            return tuple(p for p in parts if p)

        def exists(self) -> bool:
            """Check if path exists using Glob tool."""
            path_str = self._resolve_path()
            parent = real_path.dirname(path_str)
            name = real_path.basename(path_str)
            if not name:
                return True

            result = ctx.tool_call("Glob", {"pattern": name, "path": parent})
            if result and result.get("matches"):
                for match in result["matches"]:
                    if real_path.basename(match) == name:
                        return True
            return False

        def is_file(self) -> bool:
            """Check if path is a file."""
            path_str = self._resolve_path()
            parent = real_path.dirname(path_str)
            name = real_path.basename(path_str)
            if not name:
                return False

            result = ctx.tool_call("Glob", {"pattern": name, "path": parent})
            if result and result.get("matches"):
                for match in result["matches"]:
                    if real_path.basename(match) == name:
                        return True
            return False

        def is_dir(self) -> bool:
            """Check if path is a directory."""
            path_str = self._resolve_path()
            result = ctx.tool_call("Glob", {"pattern": "*", "path": path_str})
            return result is not None and "matches" in result

        def is_symlink(self) -> bool:
            return False

        def is_absolute(self) -> bool:
            return real_path.isabs(self._path)

        def resolve(self) -> "DynamicMockPath":
            return DynamicMockPath(self._resolve_path())

        def absolute(self) -> "DynamicMockPath":
            return self.resolve()

        def joinpath(self, *args: Any) -> "DynamicMockPath":
            result = self._path
            for arg in args:
                result = real_path.join(result, str(arg))
            return DynamicMockPath(result)

        def with_suffix(self, suffix: str) -> "DynamicMockPath":
            stem = self.stem
            parent = str(self.parent)
            if parent:
                return DynamicMockPath(f"{parent}/{stem}{suffix}")
            return DynamicMockPath(f"{stem}{suffix}")

        def with_name(self, name: str) -> "DynamicMockPath":
            parent = str(self.parent)
            if parent:
                return DynamicMockPath(f"{parent}/{name}")
            return DynamicMockPath(name)

        def with_stem(self, stem: str) -> "DynamicMockPath":
            return self.with_name(stem + self.suffix)

        def iterdir(self) -> Iterator["DynamicMockPath"]:
            """Iterate over directory contents using Glob tool."""
            path_str = self._resolve_path()
            result = ctx.tool_call("Glob", {"pattern": "*", "path": path_str})
            if result and result.get("matches"):
                for match in result["matches"]:
                    yield DynamicMockPath(match)

        def glob(self, pattern: str) -> Iterator["DynamicMockPath"]:
            """Glob for files matching pattern."""
            path_str = self._resolve_path()
            result = ctx.tool_call("Glob", {"pattern": pattern, "path": path_str})
            if result and result.get("matches"):
                for match in result["matches"]:
                    yield DynamicMockPath(match)

        def rglob(self, pattern: str) -> Iterator["DynamicMockPath"]:
            """Recursive glob for files matching pattern."""
            path_str = self._resolve_path()
            result = ctx.tool_call("Glob", {"pattern": f"**/{pattern}", "path": path_str})
            if result and result.get("matches"):
                for match in result["matches"]:
                    yield DynamicMockPath(match)

        def read_text(self, encoding: str = "utf-8", errors: str = "strict") -> str:
            """Read file contents using Read tool."""
            path_str = self._resolve_path()
            result = ctx.tool_call("Read", {"file_path": path_str})
            if result and "content" in result:
                return str(result["content"])
            raise FileNotFoundError(f"No such file: {path_str}")

        def read_bytes(self) -> bytes:
            """Read file contents as bytes."""
            return self.read_text().encode()

        def write_text(self, *args: Any, **kwargs: Any) -> int:
            raise PermissionError("Use ctx.tool_call('Write', ...) instead")

        def write_bytes(self, *args: Any, **kwargs: Any) -> int:
            raise PermissionError("Use ctx.tool_call('Write', ...) instead")

        def open(self, *args: Any, **kwargs: Any) -> Any:
            raise PermissionError("Use Path.read_text() or ctx.tool_call('Read', ...) instead")

        def stat(self, *args: Any, **kwargs: Any) -> Any:
            raise PermissionError("Filesystem access not allowed")

        def mkdir(self, *args: Any, **kwargs: Any) -> None:
            raise PermissionError("File operations not allowed")

        def rmdir(self, *args: Any, **kwargs: Any) -> None:
            raise PermissionError("File operations not allowed")

        def unlink(self, *args: Any, **kwargs: Any) -> None:
            raise PermissionError("File operations not allowed")

        def rename(self, *args: Any, **kwargs: Any) -> "DynamicMockPath":
            raise PermissionError("File operations not allowed")

        def replace(self, *args: Any, **kwargs: Any) -> "DynamicMockPath":
            raise PermissionError("File operations not allowed")

        def touch(self, *args: Any, **kwargs: Any) -> None:
            raise PermissionError("File operations not allowed")

        def chmod(self, *args: Any, **kwargs: Any) -> None:
            raise PermissionError("File operations not allowed")

    class DynamicMockPathlibModule:
        """Dynamic mock pathlib module."""
        Path = DynamicMockPath
        PurePath = DynamicMockPath
        PurePosixPath = DynamicMockPath
        PureWindowsPath = DynamicMockPath
        PosixPath = DynamicMockPath
        WindowsPath = DynamicMockPath

    return DynamicMockPathlibModule()


# Mapping of forbidden module names to their mock replacements
MOCK_MODULES: Dict[str, Any] = {
    "os": MOCK_OS,
    "sys": MOCK_SYS,
    "subprocess": MOCK_SUBPROCESS,
    "shutil": MOCK_SHUTIL,
    "pathlib": MOCK_PATHLIB,
    "io": MOCK_IO,
    "glob": MOCK_GLOB,
    "tempfile": MOCK_TEMPFILE,
    "socket": MOCK_SOCKET,
}
