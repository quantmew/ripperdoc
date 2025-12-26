"""Programmatic execution environment for Python-based tool calling.

This module provides a sandboxed Python execution environment where agents can
execute Python code that programmatically calls tools, rather than using LLM-based
tool calling. This reduces latency and allows complex data processing.
"""

from __future__ import annotations

import ast
import asyncio
import os
import textwrap
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ripperdoc.utils.log import get_logger

if TYPE_CHECKING:
    from ripperdoc.core.tool import Tool, ToolUseContext

logger = get_logger()


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
    "socket",
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
})

# Allowed safe modules for data processing
ALLOWED_MODULES = frozenset({
    "json",
    "re",
    "math",
    "statistics",
    "collections",
    "itertools",
    "functools",
    "operator",
    "string",
    "textwrap",
    "difflib",
    "datetime",
    "time",  # Only time.time(), not system calls
    "copy",
    "pprint",
    "typing",
    "dataclasses",
    "enum",
    "abc",
    "numbers",
    "decimal",
    "fractions",
    "random",
    "hashlib",
    "hmac",
    "base64",
    "binascii",
    "html",
    "xml",
    "csv",
    "configparser",
    "uuid",
    "fnmatch",
    "glob",  # Pattern matching only, not file access
    "asyncio",
})


class SecurityViolationError(Exception):
    """Raised when code attempts a forbidden operation."""
    pass


class ExecutionTimeoutError(Exception):
    """Raised when code execution exceeds the timeout."""
    pass


# ============================================================================
# Mock modules for dangerous system modules
# These allow code to import os/sys/subprocess without errors,
# but all operations are no-ops or return safe dummy values
# ============================================================================


class MockPath:
    """Mock path object that returns safe values."""

    def __init__(self, path: str = ""):
        self._path = str(path)

    def __str__(self) -> str:
        return self._path

    def __repr__(self) -> str:
        return f"MockPath({self._path!r})"

    def __truediv__(self, other: Any) -> "MockPath":
        return MockPath(f"{self._path}/{other}")

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

    def is_absolute(self) -> bool:
        return self._path.startswith("/")

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

    def mkdir(self, *args: Any, **kwargs: Any) -> None:
        raise PermissionError("File operations not allowed")

    def rmdir(self, *args: Any, **kwargs: Any) -> None:
        raise PermissionError("File operations not allowed")

    def unlink(self, *args: Any, **kwargs: Any) -> None:
        raise PermissionError("File operations not allowed")

    def rename(self, *args: Any, **kwargs: Any) -> "MockPath":
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
        """Returns path as-is (no symlink resolution)."""
        return str(path)

    @staticmethod
    def abspath(path: Any) -> str:
        """Returns path as-is (no cwd access)."""
        import os.path as _real_path
        # Use normpath to clean up the path, but don't prepend cwd
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

    # Blocked functions - filesystem operations
    @staticmethod
    def listdir(path: Any = ".") -> list:
        raise PermissionError("Use ctx.tool_call('Glob', {'pattern': '*'}) instead")

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

    @staticmethod
    def exit(code: int = 0) -> None:
        raise PermissionError("sys.exit() is not allowed in programmatic code")

    @staticmethod
    def getrecursionlimit() -> int:
        return 1000

    @staticmethod
    def setrecursionlimit(limit: int) -> None:
        pass  # No-op


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


class MockPathlibModule:
    """Mock pathlib module."""

    Path = MockPath
    PurePath = MockPath
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

        def write(self, s: str) -> int:
            self._buffer += s
            return len(s)

        def read(self) -> str:
            return self._buffer

        def getvalue(self) -> str:
            return self._buffer

    class BytesIO:
        def __init__(self, initial: bytes = b"") -> None:
            self._buffer = initial

        def write(self, b: bytes) -> int:
            self._buffer += b
            return len(b)

        def read(self) -> bytes:
            return self._buffer

        def getvalue(self) -> bytes:
            return self._buffer


# Create singleton instances of mock modules
MOCK_OS = MockOsModule()
MOCK_SYS = MockSysModule()
MOCK_SUBPROCESS = MockSubprocessModule()
MOCK_SHUTIL = MockShutilModule()
MOCK_PATHLIB = MockPathlibModule()
MOCK_IO = MockIoModule()

# Mapping of forbidden module names to their mock replacements
MOCK_MODULES: Dict[str, Any] = {
    "os": MOCK_OS,
    "sys": MOCK_SYS,
    "subprocess": MOCK_SUBPROCESS,
    "shutil": MOCK_SHUTIL,
    "pathlib": MOCK_PATHLIB,
    "io": MOCK_IO,
}


@dataclass
class ProgrammaticResult:
    """Result of programmatic code execution."""

    success: bool
    result: Any = None
    logs: List[str] = field(default_factory=list)
    error: Optional[str] = None
    duration_ms: float = 0.0
    tool_call_count: int = 0


class ProgrammaticContext:
    """Context object provided to programmatic code execution.

    This class provides a safe interface for programmatic agents to:
    - Call tools via tool_call()
    - Log progress messages
    - Set final results
    - Access environment information

    All I/O operations MUST go through tool_call() - direct file/network
    access is not allowed.
    """

    def __init__(
        self,
        tools: Dict[str, "Tool[Any, Any]"],
        tool_context: "ToolUseContext",
        working_directory: str,
        timeout_seconds: float = 300.0,
    ):
        self._tools = tools
        self._tool_context = tool_context
        self._working_directory = working_directory
        self._timeout_seconds = timeout_seconds
        self._start_time = time.time()

        # Execution state
        self._logs: List[str] = []
        self._result: Any = None
        self._result_set: bool = False
        self._tool_call_count: int = 0
        self._cancelled: bool = False

    def log(self, message: str) -> None:
        """Log a progress message."""
        timestamp = time.time() - self._start_time
        self._logs.append(f"[{timestamp:.2f}s] {message}")
        logger.debug(
            "[programmatic] %s",
            message,
            extra={"timestamp": timestamp},
        )

    def set_result(self, result: Any) -> None:
        """Set the final result to return."""
        self._result = result
        self._result_set = True

    def get_result(self) -> Any:
        """Get the current result."""
        return self._result

    def get_logs(self) -> List[str]:
        """Get all logged messages."""
        return list(self._logs)

    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return list(self._tools.keys())

    def get_tool_schema(self, tool_name: str) -> Dict[str, Any]:
        """Get a tool's input and output schema as dict.

        Args:
            tool_name: Name of the tool (e.g., "Glob", "Read")

        Returns:
            Dict with 'input_schema' and 'output_schema' fields describing
            the tool's parameters and return value structure.

        Example:
            schema = ctx.get_tool_schema("Glob")
            # schema["input_schema"] shows required parameters
            # schema["output_schema"] shows return value fields
        """
        tool = self._tools.get(tool_name)
        if not tool:
            available = ", ".join(sorted(self._tools.keys()))
            raise ValueError(
                f"Tool '{tool_name}' not available. Available tools: {available}"
            )

        result: Dict[str, Any] = {"tool_name": tool_name}

        # Get input schema
        input_schema = tool.input_schema
        if hasattr(input_schema, "model_json_schema"):
            result["input_schema"] = input_schema.model_json_schema()
        elif hasattr(input_schema, "schema"):
            result["input_schema"] = input_schema.schema()
        else:
            result["input_schema"] = {}

        # Try to get output schema from type hints
        try:
            import typing
            if "call" in dir(tool):
                # Try to extract return type from call method
                call_hints = typing.get_type_hints(tool.call)
                if "return" in call_hints:
                    return_type = call_hints["return"]
                    # Check if it's a generic with output type
                    if hasattr(return_type, "__origin__"):
                        args = getattr(return_type, "__args__", ())
                        for arg in args:
                            if hasattr(arg, "model_json_schema"):
                                result["output_schema"] = arg.model_json_schema()
                                break
                            elif hasattr(arg, "schema"):
                                result["output_schema"] = arg.schema()
                                break
        except Exception:
            pass

        if "output_schema" not in result:
            result["output_schema"] = {"note": "Schema not available, check tool documentation"}

        return result

    def get_working_directory(self) -> str:
        """Get the current working directory."""
        return self._working_directory

    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self._start_time

    def get_remaining_time(self) -> float:
        """Get remaining time before timeout in seconds."""
        elapsed = self.get_elapsed_time()
        return max(0.0, self._timeout_seconds - elapsed)

    def is_timeout(self) -> bool:
        """Check if execution has timed out."""
        return self.get_remaining_time() <= 0

    def cancel(self) -> None:
        """Cancel the execution."""
        self._cancelled = True

    def is_cancelled(self) -> bool:
        """Check if execution has been cancelled."""
        return self._cancelled

    def tool_call(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """Call a tool by name with the given parameters (synchronous).

        Args:
            tool_name: Name of the tool to call (e.g., "Glob", "Read", "Grep")
            params: Dictionary of parameters to pass to the tool

        Returns:
            The tool's result data

        Raises:
            ValueError: If tool is not available
            ExecutionTimeoutError: If execution has timed out
            SecurityViolationError: If the call is blocked for security reasons
        """
        # Check for cancellation and timeout
        if self._cancelled:
            raise RuntimeError("Execution was cancelled")
        if self.is_timeout():
            raise ExecutionTimeoutError(
                f"Execution timed out after {self._timeout_seconds}s"
            )

        # Validate tool exists
        tool = self._tools.get(tool_name)
        if not tool:
            available = ", ".join(sorted(self._tools.keys()))
            raise ValueError(
                f"Tool '{tool_name}' not available. Available tools: {available}"
            )

        self._tool_call_count += 1
        self.log(f"Calling {tool_name} with {_summarize_params(params)}")

        # Run the async tool call in a new event loop
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self._async_tool_call(tool, tool_name, params)
            )
        finally:
            loop.close()

    async def _async_tool_call(
        self, tool: "Tool[Any, Any]", tool_name: str, params: Dict[str, Any]
    ) -> Any:
        """Internal async implementation of tool_call."""
        try:
            # Parse input using the tool's schema
            input_schema = tool.input_schema
            parsed_input = input_schema(**params)

            # Validate input
            validation = await tool.validate_input(parsed_input, self._tool_context)
            if not validation.result:
                raise ValueError(f"Invalid input for {tool_name}: {validation.message}")

            # Execute the tool
            result = None
            async for output in tool.call(parsed_input, self._tool_context):
                # Get the final ToolResult
                if hasattr(output, "data"):
                    result = output.data

            # Convert Pydantic models to dict for easier use in programmatic code
            if result is not None:
                if hasattr(result, "model_dump"):
                    # Pydantic v2
                    result = result.model_dump()
                elif hasattr(result, "dict"):
                    # Pydantic v1
                    result = result.dict()

            return result

        except Exception as exc:
            self.log(f"Tool {tool_name} failed: {exc}")
            raise


def _summarize_params(params: Dict[str, Any], max_len: int = 100) -> str:
    """Create a short summary of parameters."""
    import json
    try:
        s = json.dumps(params, ensure_ascii=False)
        if len(s) > max_len:
            return s[:max_len - 3] + "..."
        return s
    except (TypeError, ValueError):
        return str(params)[:max_len]


class CodeValidator(ast.NodeVisitor):
    """AST visitor that validates code for security violations.

    Note: Modules in MOCK_MODULES are allowed because they have safe mock replacements.
    """

    def __init__(self) -> None:
        self.errors: List[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            module = alias.name.split(".")[0]
            # Allow modules that have mock replacements
            if module in MOCK_MODULES:
                continue  # Safe - will use mock module
            if module in FORBIDDEN_MODULES:
                self.errors.append(f"Import of forbidden module: {module}")
            elif module not in ALLOWED_MODULES:
                self.errors.append(
                    f"Import of unrecognized module: {module}. "
                    f"Only these modules are allowed: {', '.join(sorted(ALLOWED_MODULES))}"
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            module = node.module.split(".")[0]
            # Allow modules that have mock replacements
            if module in MOCK_MODULES:
                self.generic_visit(node)
                return  # Safe - will use mock module
            if module in FORBIDDEN_MODULES:
                self.errors.append(f"Import from forbidden module: {module}")
            elif module not in ALLOWED_MODULES:
                self.errors.append(
                    f"Import from unrecognized module: {module}. "
                    f"Only these modules are allowed: {', '.join(sorted(ALLOWED_MODULES))}"
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # Check for forbidden function calls
        func = node.func
        if isinstance(func, ast.Name):
            if func.id in ("exec", "eval", "compile", "open", "input", "__import__"):
                self.errors.append(f"Call to forbidden function: {func.id}")
        # Note: We don't block calls to mock module methods here
        # because the mock modules will raise PermissionError at runtime
        # for dangerous operations, giving better error messages
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # Check for dangerous attribute access
        if node.attr in ("__class__", "__bases__", "__subclasses__", "__mro__",
                         "__globals__", "__code__", "__builtins__"):
            self.errors.append(f"Access to dangerous attribute: {node.attr}")
        self.generic_visit(node)


def _add_line_numbers(code: str) -> str:
    """Add line numbers to each line of code for error reporting.

    Example output:
       1│ import json
       2│ result = ctx.tool_call("LS", {"path": "."})
       3│ ctx.log(result)
    """
    lines = code.split("\n")
    width = len(str(len(lines)))
    numbered_lines = []
    for i, line in enumerate(lines, start=1):
        numbered_lines.append(f"{i:>{width}}│ {line}")
    return "\n".join(numbered_lines)


def _fix_traceback_line_numbers(traceback_str: str, offset: int = 1) -> str:
    """Fix line numbers in traceback to account for wrapper function.

    The code is wrapped in `def __programmatic_main__():` which adds 1 line,
    so line numbers in traceback are off by 1. This function adjusts them.

    Args:
        traceback_str: The traceback string to fix
        offset: Number of lines added by wrapper (default 1)
    """
    import re

    def replace_line_number(match: re.Match[str]) -> str:
        prefix = match.group(1)
        line_num = int(match.group(2))
        suffix = match.group(3)
        # Adjust line number, but don't go below 1
        adjusted = max(1, line_num - offset)
        return f"{prefix}{adjusted}{suffix}"

    # Match patterns like:
    # - 'File "<programmatic>", line 7'
    # - 'line 7, in __programmatic_main__'
    pattern = r'(File "<programmatic>", line |, line )(\d+)(,|$)'
    return re.sub(pattern, replace_line_number, traceback_str)


def validate_code(code: str) -> List[str]:
    """Validate code for security issues.

    Returns:
        List of error messages. Empty list means code is safe.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"Syntax error: {e}"]

    validator = CodeValidator()
    validator.visit(tree)
    return validator.errors


async def execute_programmatic_code(
    code: str,
    tools: Dict[str, "Tool[Any, Any]"],
    tool_context: "ToolUseContext",
    working_directory: Optional[str] = None,
    timeout_seconds: float = 300.0,
) -> ProgrammaticResult:
    """Execute Python code in a sandboxed environment.

    Args:
        code: Python code to execute (async function body)
        tools: Dictionary of available tools
        tool_context: Context for tool execution
        working_directory: Working directory for the execution
        timeout_seconds: Maximum execution time

    Returns:
        ProgrammaticResult with success status, result, logs, etc.
    """
    start_time = time.time()

    # Validate code
    errors = validate_code(code)
    if errors:
        numbered_code = _add_line_numbers(code)
        return ProgrammaticResult(
            success=False,
            error="Code validation failed:\n"
            + "\n".join(f"- {e}" for e in errors)
            + f"\n\nCode:\n{numbered_code}",
            duration_ms=(time.time() - start_time) * 1000,
        )

    # Create context
    ctx = ProgrammaticContext(
        tools=tools,
        tool_context=tool_context,
        working_directory=working_directory or os.getcwd(),
        timeout_seconds=timeout_seconds,
    )

    # Prepare safe globals
    safe_globals = _build_safe_globals(ctx)

    # Wrap code in function (synchronous - tool_call handles async internally)
    wrapped_code = _wrap_in_function(code)

    try:
        # Compile the code
        compiled = compile(wrapped_code, "<programmatic>", "exec")

        # Execute to define the function
        exec(compiled, safe_globals)

        # Get the defined function
        main_func = safe_globals.get("__programmatic_main__")
        if not main_func:
            return ProgrammaticResult(
                success=False,
                error="Failed to create execution function",
                duration_ms=(time.time() - start_time) * 1000,
            )

        # Execute with timeout using concurrent.futures
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(main_func)
            try:
                result = future.result(timeout=timeout_seconds)
            except concurrent.futures.TimeoutError:
                ctx.cancel()  # Signal cancellation
                return ProgrammaticResult(
                    success=False,
                    error=f"Execution timed out after {timeout_seconds}s",
                    logs=ctx.get_logs(),
                    duration_ms=(time.time() - start_time) * 1000,
                    tool_call_count=ctx._tool_call_count,
                )

        # Get result
        final_result = ctx.get_result() if ctx._result_set else result

        return ProgrammaticResult(
            success=True,
            result=final_result,
            logs=ctx.get_logs(),
            duration_ms=(time.time() - start_time) * 1000,
            tool_call_count=ctx._tool_call_count,
        )

    except SecurityViolationError as e:
        numbered_code = _add_line_numbers(code)
        return ProgrammaticResult(
            success=False,
            error=f"Security violation: {e}\n\nCode:\n{numbered_code}",
            logs=ctx.get_logs(),
            duration_ms=(time.time() - start_time) * 1000,
            tool_call_count=ctx._tool_call_count,
        )
    except RuntimeError as e:
        if "cancelled" in str(e).lower():
            numbered_code = _add_line_numbers(code)
            return ProgrammaticResult(
                success=False,
                error=f"Execution was cancelled\n\nCode:\n{numbered_code}",
                logs=ctx.get_logs(),
                duration_ms=(time.time() - start_time) * 1000,
                tool_call_count=ctx._tool_call_count,
            )
        raise
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        # Fix line numbers in traceback (offset by 1 due to wrapper function)
        tb = _fix_traceback_line_numbers(tb)
        numbered_code = _add_line_numbers(code)
        return ProgrammaticResult(
            success=False,
            error=f"Execution error: {e}\n\nCode:\n{numbered_code}\n\nTraceback:\n{tb}",
            logs=ctx.get_logs(),
            duration_ms=(time.time() - start_time) * 1000,
            tool_call_count=ctx._tool_call_count,
        )


def _wrap_in_function(code: str) -> str:
    """Wrap code in a function definition."""
    # Dedent and normalize
    code = textwrap.dedent(code).strip()

    # Indent the code
    indented = textwrap.indent(code, "    ")

    # Wrap in regular function (not async - tool_call is now synchronous)
    wrapped = f"def __programmatic_main__():\n{indented}\n"

    return wrapped


def _safe_import(
    name: str,
    globals_dict: Optional[Dict[str, Any]] = None,
    locals_dict: Optional[Dict[str, Any]] = None,
    fromlist: tuple = (),
    level: int = 0,
) -> Any:
    """Safe import function that only allows whitelisted modules.

    For dangerous modules (os, sys, subprocess, etc.), returns mock modules
    that provide safe dummy values or raise PermissionError on dangerous operations.
    """
    # Get the top-level module name
    top_module = name.split(".")[0]

    # Check if this is a mock-able forbidden module
    if top_module in MOCK_MODULES:
        logger.debug(
            "[programmatic] Returning mock module for: %s",
            top_module,
        )
        return MOCK_MODULES[top_module]

    # For other forbidden modules without mocks, raise error
    if top_module in FORBIDDEN_MODULES:
        raise ImportError(
            f"Import of forbidden module: {top_module}. "
            f"Use ctx.tool_call() for I/O operations."
        )

    # For unrecognized modules, raise error
    if top_module not in ALLOWED_MODULES:
        raise ImportError(
            f"Import of unrecognized module: {top_module}. "
            f"Use ctx.tool_call() for I/O operations."
        )

    # Import the allowed module using the real __import__
    import builtins
    return builtins.__import__(name, globals_dict, locals_dict, fromlist, level)


def _build_safe_globals(ctx: ProgrammaticContext) -> Dict[str, Any]:
    """Build a restricted globals dictionary for code execution."""
    import json
    import re
    import math
    import statistics
    import collections
    import itertools
    import functools
    import operator
    import string
    import datetime
    import copy
    import typing
    import dataclasses
    import enum
    import random
    import hashlib
    import base64
    import uuid
    import fnmatch

    # Safe builtins
    safe_builtins = {
        # Types
        "bool": bool,
        "int": int,
        "float": float,
        "str": str,
        "bytes": bytes,
        "bytearray": bytearray,
        "list": list,
        "tuple": tuple,
        "dict": dict,
        "set": set,
        "frozenset": frozenset,
        "type": type,
        "object": object,

        # Functions
        "abs": abs,
        "all": all,
        "any": any,
        "ascii": ascii,
        "bin": bin,
        "callable": callable,
        "chr": chr,
        "divmod": divmod,
        "enumerate": enumerate,
        "filter": filter,
        "format": format,
        "getattr": getattr,
        "hasattr": hasattr,
        "hash": hash,
        "hex": hex,
        "id": id,
        "isinstance": isinstance,
        "issubclass": issubclass,
        "iter": iter,
        "len": len,
        "map": map,
        "max": max,
        "min": min,
        "next": next,
        "oct": oct,
        "ord": ord,
        "pow": pow,
        "print": lambda *args, **kwargs: ctx.log(" ".join(str(a) for a in args)),
        "range": range,
        "repr": repr,
        "reversed": reversed,
        "round": round,
        "setattr": setattr,
        "slice": slice,
        "sorted": sorted,
        "sum": sum,
        "zip": zip,

        # Import function (restricted)
        "__import__": _safe_import,

        # Exceptions
        "Exception": Exception,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "KeyError": KeyError,
        "IndexError": IndexError,
        "AttributeError": AttributeError,
        "RuntimeError": RuntimeError,
        "StopIteration": StopIteration,
        "ImportError": ImportError,

        # Constants
        "True": True,
        "False": False,
        "None": None,
    }

    return {
        "__builtins__": safe_builtins,
        "__name__": "__programmatic__",
        "__doc__": None,

        # Context object
        "ctx": ctx,

        # Safe modules (pre-imported for convenience)
        "json": json,
        "re": re,
        "math": math,
        "statistics": statistics,
        "collections": collections,
        "itertools": itertools,
        "functools": functools,
        "operator": operator,
        "string": string,
        "datetime": datetime,
        "copy": copy,
        "typing": typing,
        "dataclasses": dataclasses,
        "enum": enum,
        "random": random,
        "hashlib": hashlib,
        "base64": base64,
        "uuid": uuid,
        "fnmatch": fnmatch,

        # Async support
        "asyncio": asyncio,

        # Mock modules (safe replacements for dangerous modules)
        # These allow code like "import os" to work without errors,
        # but dangerous operations will raise PermissionError
        "os": MOCK_OS,
        "sys": MOCK_SYS,
        "subprocess": MOCK_SUBPROCESS,
        "shutil": MOCK_SHUTIL,
        "pathlib": MOCK_PATHLIB,
        "io": MOCK_IO,
    }
