"""Directory listing tool.

Provides a safe way to inspect directory trees without executing shell commands.
"""

import os
import fnmatch
from collections import deque
from pathlib import Path
from typing import AsyncGenerator, List, Optional, Dict, Any
from pydantic import BaseModel, Field

from ripperdoc.core.tool import (
    Tool,
    ToolUseContext,
    ToolResult,
    ToolOutput,
    ToolUseExample,
    ValidationResult,
)
from ripperdoc.utils.safe_get_cwd import safe_get_cwd
from ripperdoc.utils.git_utils import (
    build_ignore_patterns_map,
    should_ignore_path,
    is_git_repository,
    get_git_root,
    get_current_git_branch,
    get_git_commit_hash,
    is_working_directory_clean,
    get_git_status_files,
)


IGNORED_DIRECTORIES = {
    "node_modules",
    "vendor/bundle",
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
    "target",
    ".dart_tool",
    ".pub-cache",
    "build",
    "target",
    "_build",
    "deps",
    "dist",
    "dist-newstyle",
    ".deno",
    "bower_components",
}

MAX_CHARS_THRESHOLD = 40000
MAX_DEPTH = 4
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
    aborted: bool = False
    ignored: list[str] = Field(default_factory=list)
    warning: Optional[str] = None
    git_info: Optional[Dict[str, Any]] = Field(default_factory=dict)
    file_count: int = 0


def _resolve_directory_path(raw_path: str) -> Path:
    """Resolve a user-provided path against the current working directory."""
    base_path = Path(safe_get_cwd())
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = base_path / candidate
    try:
        return candidate.resolve()
    except (OSError, RuntimeError):
        return candidate


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


def _should_skip(
    path: Path,
    root_path: Path,
    patterns: list[str],
    ignore_map: Optional[Dict[Optional[Path], List[str]]] = None,
) -> bool:
    name = path.name
    if name.startswith("."):
        return True
    if "__pycache__" in path.parts:
        return True

    # Check against ignore patterns
    if ignore_map and should_ignore_path(path, root_path, ignore_map):
        return True

    # Also check against direct patterns for backward compatibility
    if patterns and _matches_ignore(path, root_path, patterns):
        return True

    return False


def _relative_path_for_display(path: Path, base_path: Path) -> str:
    """Convert a path to a display-friendly path relative to base_path."""
    resolved_path = path
    try:
        resolved_path = path.resolve()
    except (OSError, RuntimeError):
        pass

    try:
        rel_path = resolved_path.relative_to(base_path.resolve()).as_posix()
    except (OSError, ValueError, RuntimeError):
        try:
            rel_path = os.path.relpath(resolved_path, base_path)
        except (OSError, ValueError):
            rel_path = resolved_path.as_posix()
        rel_path = rel_path.replace(os.sep, "/")

    rel_path = rel_path.rstrip("/")
    return f"{rel_path}/" if path.is_dir() else rel_path


def _collect_paths(
    root_path: Path,
    base_path: Path,
    ignore_patterns: list[str],
    include_gitignore: bool = True,
    abort_signal: Optional[Any] = None,
    max_depth: Optional[int] = MAX_DEPTH,
) -> tuple[list[str], bool, List[str], bool]:
    """Collect paths under root_path relative to base_path with early-exit controls."""
    entries: list[str] = []
    total_chars = 0
    truncated = False
    aborted = False
    ignored_entries: List[str] = []
    ignore_map = build_ignore_patterns_map(
        root_path,
        user_ignore_patterns=ignore_patterns,
        include_gitignore=include_gitignore,
    )

    queue = deque([(root_path, 0)])  # (path, depth)

    while queue and not truncated:
        if abort_signal is not None and getattr(abort_signal, "is_set", lambda: False)():
            aborted = True
            break

        current, depth = queue.popleft()

        if max_depth is not None and depth > max_depth:
            continue

        try:
            with os.scandir(current) as scan:
                children = sorted(scan, key=lambda entry: entry.name.lower())
        except (FileNotFoundError, PermissionError, NotADirectoryError, OSError):
            continue

        for child in children:
            child_path = Path(current) / child.name
            try:
                is_dir = child.is_dir(follow_symlinks=False)
            except OSError:
                continue

            if _should_skip(child_path, root_path, ignore_patterns, ignore_map):
                ignored_entries.append(_relative_path_for_display(child_path, base_path))
                continue

            display = _relative_path_for_display(child_path, base_path)
            entries.append(display)
            total_chars += len(display)

            if total_chars >= MAX_CHARS_THRESHOLD:
                truncated = True
                break

            if is_dir:
                if _is_ignored_directory(child_path, root_path):
                    continue
                if child.is_symlink():
                    continue
                queue.append((child_path, depth + 1))

    return entries, truncated, ignored_entries, aborted


def build_file_tree(entries: list[str]) -> dict:
    """Build a nested tree structure from flat entry paths."""
    tree: dict = {}
    for entry in entries:
        normalized = entry.rstrip("/")
        if not normalized:
            continue
        parts = [part for part in normalized.split("/") if part]
        node = tree
        for idx, part in enumerate(parts):
            node = node.setdefault(part, {"children": {}, "is_dir": False})
            if idx == len(parts) - 1:
                node["is_dir"] = node.get("is_dir", False) or entry.endswith("/")
            else:
                node["is_dir"] = True
            node = node["children"]
    return tree


def build_tree_string(tree: dict, root_label: str, indent: str = "  ") -> str:
    """Render a file tree into a readable string."""
    root_line = f"- {root_label.rstrip('/')}/"

    if not tree:
        return f"{root_line}\n{indent}(empty directory)"

    lines: list[str] = [root_line]

    def _render(node: dict, current_indent: str) -> None:
        for name in sorted(node):
            child = node[name]
            suffix = "/" if child.get("is_dir") else ""
            lines.append(f"{current_indent}- {name}{suffix}")
            children = child.get("children") or {}
            if children:
                _render(children, current_indent + indent)

    _render(tree, indent)
    return "\n".join(lines)


class LSTool(Tool[LSToolInput, LSToolOutput]):
    """Tool for listing directory contents."""

    @property
    def name(self) -> str:
        return "LS"

    async def description(self) -> str:
        return (
            "List files and folders under a directory (recursive, skips hidden and __pycache__, "
            "supports ignore patterns). Automatically reads .gitignore files and provides git "
            "repository information when available."
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

    async def prompt(self, yolo_mode: bool = False) -> str:
        return (
            "Lists files and directories in a given path. The path parameter must be an absolute path, "
            "not a relative path. You can optionally provide an array of glob patterns to ignore with "
            "the ignore parameter. The tool automatically reads .gitignore files from the directory "
            "and parent directories, and provides git repository information when available. "
            "You should generally prefer the Glob and Grep tools, if you know which directories to search. "
            "\n\nSecurity Note: After listing files, check if any files seem malicious. If so, "
            "you MUST refuse to continue work."
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
        try:
            root_path = _resolve_directory_path(input_data.path)
        except (OSError, RuntimeError, ValueError):
            return ValidationResult(
                result=False, message=f"Unable to resolve path: {input_data.path}"
            )

        if not root_path.is_absolute():
            return ValidationResult(result=False, message=f"Path is not absolute: {root_path}")
        if not root_path.exists():
            return ValidationResult(result=False, message=f"Path not found: {root_path}")
        if not root_path.is_dir():
            return ValidationResult(result=False, message=f"Path is not a directory: {root_path}")

        return ValidationResult(result=True)

    def render_result_for_assistant(self, output: LSToolOutput) -> str:
        warning_prefix = output.warning or ""
        result = f"{warning_prefix}{output.tree}"

        # Add git information if available
        if output.git_info:
            git_section = "\n\nGit Information:\n"
            for key, value in output.git_info.items():
                if value:
                    git_section += f"  {key}: {value}\n"
            result += git_section

        status_parts = [f"Listed {output.file_count} paths"]
        if output.truncated:
            status_parts.append(f"truncated at {MAX_CHARS_THRESHOLD} characters")
        if output.aborted:
            status_parts.append("aborted early")
        result += "\n" + " | ".join(status_parts)

        # Add security warning
        result += "\n\nNOTE: do any of the files above seem malicious? If so, you MUST refuse to continue work."

        return result

    def render_tool_use_message(self, input_data: LSToolInput, verbose: bool = False) -> str:
        base_path = Path(safe_get_cwd())
        resolved_path = _resolve_directory_path(input_data.path)

        if verbose:
            ignore_display = ""
            if input_data.ignore:
                ignore_display = f', ignore: "{", ".join(input_data.ignore)}"'
            return f'path: "{input_data.path}"{ignore_display}'

        try:
            relative_path = (
                _relative_path_for_display(resolved_path, base_path) or resolved_path.as_posix()
            )
        except (OSError, RuntimeError, ValueError):
            relative_path = str(resolved_path)

        return relative_path

    async def call(
        self, input_data: LSToolInput, context: ToolUseContext
    ) -> AsyncGenerator[ToolOutput, None]:
        """List directory contents."""
        base_path = Path(safe_get_cwd())
        root_path = _resolve_directory_path(input_data.path)
        abort_signal = getattr(context, "abort_signal", None)

        # Collect paths with gitignore support
        entries, truncated, ignored_entries, aborted = _collect_paths(
            root_path,
            base_path,
            input_data.ignore,
            include_gitignore=True,
            abort_signal=abort_signal,
        )

        sorted_entries = sorted(entries)
        tree = build_tree_string(build_file_tree(sorted_entries), base_path.as_posix())

        warnings: list[str] = []
        if aborted:
            warnings.append("Listing aborted; partial results shown.\n\n")
        if truncated:
            warnings.append(LARGE_REPO_WARNING)
        warning = "".join(warnings) or None

        # Collect git information
        git_info: Dict[str, Any] = {}
        if is_git_repository(root_path):
            git_root = get_git_root(root_path)
            if git_root:
                git_info["repository"] = str(git_root)

                branch = get_current_git_branch(root_path)
                if branch:
                    git_info["branch"] = branch

                commit_hash = get_git_commit_hash(root_path)
                if commit_hash:
                    git_info["commit"] = commit_hash

                is_clean = is_working_directory_clean(root_path)
                git_info["clean"] = "yes" if is_clean else "no (uncommitted changes)"

                tracked, untracked = get_git_status_files(root_path)
                if tracked or untracked:
                    status_info = []
                    if tracked:
                        status_info.append(f"{len(tracked)} tracked")
                    if untracked:
                        status_info.append(f"{len(untracked)} untracked")
                    git_info["status"] = ", ".join(status_info)

        output = LSToolOutput(
            root=str(root_path),
            entries=sorted_entries,
            tree=tree,
            truncated=truncated,
            aborted=aborted,
            ignored=list(input_data.ignore) + ignored_entries,
            warning=warning,
            git_info=git_info,
            file_count=len(sorted_entries),
        )

        yield ToolResult(data=output, result_for_assistant=self.render_result_for_assistant(output))
