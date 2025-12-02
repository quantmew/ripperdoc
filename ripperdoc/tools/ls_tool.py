"""Directory listing tool.

Provides a safe way to inspect directory trees without executing shell commands.
"""

import fnmatch
from collections import deque
from pathlib import Path
from typing import AsyncGenerator, List, Optional
from pydantic import BaseModel, Field

from ripperdoc.core.tool import (
    Tool,
    ToolUseContext,
    ToolResult,
    ToolOutput,
    ToolUseExample,
    ValidationResult,
)


IGNORED_DIRECTORIES = {
    "node_modules",
    "vendor",
    "venv",
    "env",
    ".venv",
    ".env",
    ".tox",
    "target",
    "build",
    ".gradle",
    "packages",
    "bin",
    "obj",
    ".build",
    "_build",
    "deps",
    "dist",
    "dist-newstyle",
    ".deno",
    "bower_components",
    "vendor/bundle",
    ".dart_tool",
    ".pub-cache",
}

MAX_CHARS_THRESHOLD = 40000
LARGE_REPO_WARNING = (
    f"There are more than {MAX_CHARS_THRESHOLD} characters in the repository "
    "(ie. either there are lots of files, or there are many long filenames). "
    "Use the LS tool (passing a specific path), Bash tool, and other tools to explore "
    "nested directories. The first "
    f"{MAX_CHARS_THRESHOLD} characters are included below:\n\n"
)


def _is_ignored_directory(path: Path, root_path: Path) -> bool:
    if path == root_path:
        return False

    path_str = path.as_posix()
    for ignored in IGNORED_DIRECTORIES:
        normalized = ignored.rstrip("/\\")
        if path.name == normalized:
            return True
        if path_str.endswith(f"/{normalized}"):
            return True
    return False


class LSToolInput(BaseModel):
    """Input schema for LSTool."""

    path: str = Field(
        description="The absolute path to the directory to list (must be absolute, not relative)"
    )
    ignore: list[str] = Field(
        default_factory=list,
        description="List of glob patterns to ignore (relative to the provided path)",
    )


class LSToolOutput(BaseModel):
    """Output from directory listing."""

    root: str
    entries: list[str]
    tree: str
    truncated: bool = False
    ignored: list[str] = Field(default_factory=list)
    warning: Optional[str] = None


def _matches_ignore(path: Path, root_path: Path, patterns: list[str]) -> bool:
    if not patterns:
        return False

    try:
        rel = path.relative_to(root_path).as_posix()
    except ValueError:
        rel = path.as_posix()

    rel_dir = f"{rel}/" if path.is_dir() else rel
    return any(
        fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(rel_dir, pattern) for pattern in patterns
    )


def _should_skip(path: Path, root_path: Path, patterns: list[str]) -> bool:
    name = path.name
    if name.startswith("."):
        return True
    if "__pycache__" in path.parts:
        return True
    if _matches_ignore(path, root_path, patterns):
        return True
    return False


def _format_relative(path: Path, root_path: Path) -> str:
    rel_path = path.relative_to(root_path).as_posix()
    return f"{rel_path}/" if path.is_dir() else rel_path


def _collect_paths(root_path: Path, ignore_patterns: list[str]) -> tuple[list[str], bool]:
    entries: list[str] = []
    total_chars = 0
    truncated = False
    queue = deque([root_path])

    while queue and not truncated:
        current = queue.popleft()

        try:
            children = sorted(current.iterdir(), key=lambda p: p.name.lower())
        except (FileNotFoundError, PermissionError):
            continue

        for child in children:
            try:
                is_dir = child.is_dir()
            except OSError:
                continue

            if _should_skip(child, root_path, ignore_patterns):
                continue

            display = _format_relative(child, root_path)
            entries.append(display)
            total_chars += len(display)

            if total_chars > MAX_CHARS_THRESHOLD:
                truncated = True
                break

            if is_dir:
                if _is_ignored_directory(child, root_path):
                    continue
                if child.is_symlink():
                    continue
                queue.append(child)

    return entries, truncated


def _add_to_tree(tree: dict, parts: list[str], is_dir: bool) -> None:
    node = tree
    for idx, part in enumerate(parts):
        node = node.setdefault(part, {"children": {}, "is_dir": False})
        if idx == len(parts) - 1:
            node["is_dir"] = is_dir
        node = node["children"]


def _render_tree(tree: dict, indent: str = "  ", current_indent: str = "  ") -> str:
    lines: list[str] = []
    for name in sorted(tree):
        node = tree[name]
        suffix = "/" if node.get("is_dir") else ""
        lines.append(f"{current_indent}- {name}{suffix}")
        children = node.get("children") or {}
        if children:
            lines.append(_render_tree(children, indent, current_indent + indent))
    return "\n".join(lines)


def _build_tree(entries: list[str], root_path: Path) -> str:
    root_line = f"- {root_path.resolve().as_posix()}/"

    if not entries:
        return f"{root_line}\n  (empty directory)"

    tree: dict = {}
    for entry in entries:
        normalized = entry[:-1] if entry.endswith("/") else entry
        if not normalized:
            continue
        parts = normalized.split("/")
        _add_to_tree(tree, parts, entry.endswith("/"))

    body = _render_tree(tree)
    return f"{root_line}\n{body}"


class LSTool(Tool[LSToolInput, LSToolOutput]):
    """Tool for listing directory contents."""

    @property
    def name(self) -> str:
        return "LS"

    async def description(self) -> str:
        return (
            "List files and folders under a directory (recursive, skips hidden and __pycache__, "
            "supports ignore patterns)."
        )

    @property
    def input_schema(self) -> type[LSToolInput]:
        return LSToolInput

    def input_examples(self) -> List[ToolUseExample]:
        return [
            ToolUseExample(
                description="List the repository root with defaults",
                example={"path": "/repo"},
            ),
            ToolUseExample(
                description="Inspect a package while skipping build outputs",
                example={"path": "/repo/packages/api", "ignore": ["dist/**", "node_modules/**"]},
            ),
        ]

    async def prompt(self, safe_mode: bool = False) -> str:
        return (
            "Lists files and directories in a given path. The path parameter must be an absolute path, "
            "not a relative path. You can optionally provide an array of glob patterns to ignore with "
            "the ignore parameter. You should generally prefer the Glob and Grep tools, if you know "
            "which directories to search."
        )

    def is_read_only(self) -> bool:
        return True

    def is_concurrency_safe(self) -> bool:
        return True

    def needs_permissions(self, input_data: Optional[LSToolInput] = None) -> bool:
        return False

    async def validate_input(
        self, input_data: LSToolInput, context: Optional[ToolUseContext] = None
    ) -> ValidationResult:
        root_path = Path(input_data.path).expanduser()
        if not root_path.is_absolute():
            root_path = Path.cwd() / root_path

        if not root_path.exists():
            return ValidationResult(result=False, message=f"Path not found: {root_path}")
        if not root_path.is_dir():
            return ValidationResult(result=False, message=f"Path is not a directory: {root_path}")

        return ValidationResult(result=True)

    def render_result_for_assistant(self, output: LSToolOutput) -> str:
        warning_prefix = output.warning or ""
        return f"{warning_prefix}{output.tree}"

    def render_tool_use_message(self, input_data: LSToolInput, verbose: bool = False) -> str:
        ignore_display = ""
        if input_data.ignore:
            ignore_display = f', ignore: "{", ".join(input_data.ignore)}"'
        return f'path: "{input_data.path}"{ignore_display}'

    async def call(
        self, input_data: LSToolInput, context: ToolUseContext
    ) -> AsyncGenerator[ToolOutput, None]:
        """List directory contents."""
        root_path = Path(input_data.path).expanduser()
        if not root_path.is_absolute():
            root_path = Path.cwd() / root_path
        root_path = root_path.resolve()

        entries, truncated = _collect_paths(root_path, input_data.ignore)
        tree = _build_tree(entries, root_path)
        warning = LARGE_REPO_WARNING if truncated else None

        output = LSToolOutput(
            root=str(root_path),
            entries=entries,
            tree=tree,
            truncated=truncated,
            ignored=list(input_data.ignore),
            warning=warning,
        )

        yield ToolResult(data=output, result_for_assistant=self.render_result_for_assistant(output))
