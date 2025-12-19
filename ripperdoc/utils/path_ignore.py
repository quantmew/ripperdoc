"""Path ignore utilities for Ripperdoc.

This module implements comprehensive path ignore checking based on:
1. Default ignore patterns (binary files, build outputs, etc.)
2. .gitignore patterns
3. Project configuration ignore patterns
4. User-defined ignore patterns
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ripperdoc.utils.git_utils import (
    get_git_root,
    is_git_repository,
    read_gitignore_patterns,
)


# =============================================================================
# Default Ignore Patterns (System-level)
# =============================================================================

# These patterns are always ignored for safety and performance
DEFAULT_IGNORE_PATTERNS: List[str] = [
    # Version control
    ".git/",
    ".svn/",
    ".hg/",
    # IDE and editor
    ".idea/",
    ".vscode/",
    "*.swp",
    "*.swo",
    "*~",
    # Ripperdoc config
    ".ripperdoc/",
    ".claude/",
    # Build and cache directories
    ".parcel-cache/",
    ".pytest_cache/",
    ".nuxt/",
    ".next/",
    ".sass-cache/",
    "__pycache__/",
    "*.pyc",
    "*.pyo",
    # Node.js
    "node_modules/",
    # Python environments
    "venv/",
    ".venv/",
    "env/",
    ".env/",
    ".tox/",
    # Java/Gradle
    ".gradle/",
    "build/",
    "target/",
    # .NET
    "bin/",
    "obj/",
    # Rust
    "target/",
    # Go
    "vendor/",
    # Ruby
    "vendor/bundle/",
    # Dart/Flutter
    ".dart_tool/",
    ".pub-cache/",
    # Elixir
    "_build/",
    "deps/",
    # Haskell
    "dist-newstyle/",
    # JavaScript/TypeScript build outputs
    "dist/",
    # Deno
    ".deno/",
    # Bower (legacy)
    "bower_components/",
    # Image files
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.bmp",
    "*.ico",
    "*.webp",
    "*.svg",
    "*.psd",
    "*.ai",
    "*.eps",
    "*.tiff",
    "*.tif",
    "*.avif",
    "*.heic",
    "*.heif",
    # Video files
    "*.mp4",
    "*.avi",
    "*.mkv",
    "*.mov",
    "*.wmv",
    "*.flv",
    "*.webm",
    "*.m4v",
    "*.mpeg",
    "*.mpg",
    "*.3gp",
    "*.3g2",
    # Audio files
    "*.mp3",
    "*.wav",
    "*.flac",
    "*.aac",
    "*.ogg",
    "*.wma",
    "*.m4a",
    "*.aiff",
    "*.asf",
    # Compressed archives
    "*.zip",
    "*.tar",
    "*.gz",
    "*.bz2",
    "*.xz",
    "*.7z",
    "*.rar",
    "*.tgz",
    # Executable and binary files
    "*.exe",
    "*.dll",
    "*.so",
    "*.dylib",
    "*.bin",
    "*.app",
    "*.dmg",
    "*.msi",
    "*.deb",
    "*.rpm",
    # Database files
    "*.db",
    "*.sqlite",
    "*.sqlite3",
    "*.parquet",
    "*.orc",
    "*.arrow",
    # GIS data
    "*.shp",
    "*.kmz",
    "*.kml",
    "*.dem",
    "*.las",
    "*.laz",
    # CAD/Design files
    "*.dwg",
    "*.dxf",
    # PDF (can be read but often large)
    # "*.pdf",  # Keeping PDF readable since it's often documentation
    # Font files
    "*.ttf",
    "*.otf",
    "*.woff",
    "*.woff2",
    "*.eot",
    # Lock files (usually large and auto-generated)
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "Cargo.lock",
    "Gemfile.lock",
    "composer.lock",
    # Large data files
    "*.csv",  # Can be very large
    "*.jsonl",
    "*.ndjson",
    # ML model files
    "*.pt",
    "*.pth",
    "*.onnx",
    "*.h5",
    "*.hdf5",
    "*.safetensors",
    "*.ckpt",
    "*.pkl",
    "*.pickle",
]

# Directories that are always skipped during traversal (fast path)
IGNORED_DIRECTORIES: Set[str] = {
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
    ".dart_tool",
    ".pub-cache",
    "_build",
    "deps",
    "dist",
    "dist-newstyle",
    ".deno",
    "bower_components",
    "__pycache__",
    ".git",
    ".svn",
    ".hg",
    ".idea",
    ".vscode",
    ".ripperdoc",
    ".claude",
}


# =============================================================================
# Pattern Matching Implementation
# =============================================================================


def _compile_pattern(pattern: str) -> re.Pattern[str]:
    """Compile a gitignore-style pattern to a regex.

    This supports basic gitignore syntax:
    - * matches anything except /
    - ** matches anything including /
    - ? matches any single character
    - [abc] character classes
    - ! at start negates the pattern
    - / at end matches directories only
    - / at start anchors to root
    """
    # Remove trailing slashes for matching (we handle directory-only patterns separately)
    is_dir_only = pattern.endswith("/")
    if is_dir_only:
        pattern = pattern[:-1]

    # Check if pattern is anchored to root
    is_anchored = pattern.startswith("/")
    if is_anchored:
        pattern = pattern[1:]

    # Escape special regex characters (except our wildcards)
    regex = ""
    i = 0
    while i < len(pattern):
        c = pattern[i]

        if c == "*":
            # Check for **
            if i + 1 < len(pattern) and pattern[i + 1] == "*":
                # ** matches anything including path separators
                if i + 2 < len(pattern) and pattern[i + 2] == "/":
                    regex += "(?:.*/)?"
                    i += 3
                    continue
                else:
                    regex += ".*"
                    i += 2
                    continue
            else:
                # * matches anything except /
                regex += "[^/]*"
        elif c == "?":
            regex += "[^/]"
        elif c == "[":
            # Find closing bracket
            j = i + 1
            if j < len(pattern) and pattern[j] in "!^":
                j += 1
            while j < len(pattern) and pattern[j] != "]":
                j += 1
            if j < len(pattern):
                regex += pattern[i : j + 1]
                i = j
            else:
                regex += re.escape(c)
        elif c in ".^$+{}|()":
            regex += re.escape(c)
        else:
            regex += c

        i += 1

    # Build final regex
    if is_anchored:
        final_regex = f"^{regex}"
    else:
        # Non-anchored patterns can match anywhere in the path
        final_regex = f"(?:^|/){regex}"

    if is_dir_only:
        final_regex += "(?:/|$)"
    else:
        final_regex += "(?:/.*)?$"

    return re.compile(final_regex)


class IgnoreFilter:
    """A filter for checking if paths should be ignored.

    Uses gitignore-style pattern matching.
    """

    def __init__(self) -> None:
        self._patterns: List[Tuple[re.Pattern[str], bool]] = []  # (pattern, is_negation)

    def add(self, patterns: List[str]) -> "IgnoreFilter":
        """Add patterns to the filter."""
        for pattern in patterns:
            pattern = pattern.strip()
            if not pattern or pattern.startswith("#"):
                continue

            is_negation = pattern.startswith("!")
            if is_negation:
                pattern = pattern[1:]

            try:
                compiled = _compile_pattern(pattern)
                self._patterns.append((compiled, is_negation))
            except re.error:
                # Skip invalid patterns
                pass

        return self

    def ignores(self, path: str) -> bool:
        """Check if a path should be ignored.

        Args:
            path: Relative path to check (using / as separator)

        Returns:
            True if the path should be ignored
        """
        # Normalize path
        path = path.replace("\\", "/").strip("/")

        result = False
        for pattern, is_negation in self._patterns:
            if pattern.search(path):
                result = not is_negation

        return result

    def test(self, path: str) -> Dict[str, Any]:
        """Check if a path should be ignored and return details.

        Returns:
            Dict with 'ignored' bool and 'rule' if matched
        """
        path = path.replace("\\", "/").strip("/")

        result: Dict[str, Any] = {"ignored": False, "rule": None}

        for pattern, is_negation in self._patterns:
            if pattern.search(path):
                result["ignored"] = not is_negation
                result["rule"] = {"pattern": pattern.pattern, "negation": is_negation}

        return result


# =============================================================================
# Ignore Pattern Management
# =============================================================================


def parse_ignore_pattern(
    pattern: str, settings_path: Optional[Path] = None
) -> Tuple[str, Optional[Path]]:
    """Parse an ignore pattern and return (relative_pattern, root_path).

    Supports prefixes:
    - // - Global pattern (from filesystem root)
    - ~/ - Pattern relative to home directory
    - / - Pattern relative to settings file directory
    - (no prefix) - Pattern applies to any directory

    Args:
        pattern: The ignore pattern to parse
        settings_path: Path to the settings file (for / prefix patterns)

    Returns:
        Tuple of (relative_pattern, root_path or None)
    """
    pattern = pattern.strip()

    # // - Global pattern from filesystem root
    if pattern.startswith("//"):
        return pattern[1:], Path("/")

    # ~/ - Pattern relative to home directory
    if pattern.startswith("~/"):
        return pattern[2:], Path.home()

    # / - Pattern relative to settings file directory
    if pattern.startswith("/") and settings_path:
        # Determine if settings_path is a file or directory based on suffix
        # If it has a file-like suffix (e.g., .json), treat as file
        if settings_path.suffix:
            return pattern[1:], settings_path.parent
        else:
            return pattern[1:], settings_path

    # No prefix - applies to any directory
    return pattern, None


def build_ignore_filter(
    root_path: Path,
    user_patterns: Optional[List[str]] = None,
    project_patterns: Optional[List[str]] = None,
    include_defaults: bool = True,
    include_gitignore: bool = True,
) -> IgnoreFilter:
    """Build an ignore filter with all applicable patterns.

    Args:
        root_path: The root path for pattern matching
        user_patterns: User-provided patterns
        project_patterns: Project configuration patterns
        include_defaults: Whether to include default ignore patterns
        include_gitignore: Whether to include .gitignore patterns

    Returns:
        Configured IgnoreFilter instance
    """
    ignore_filter = IgnoreFilter()
    all_patterns: List[str] = []

    # 1. Add default patterns
    if include_defaults:
        all_patterns.extend(DEFAULT_IGNORE_PATTERNS)

    # 2. Add gitignore patterns
    if include_gitignore and is_git_repository(root_path):
        gitignore_patterns = read_gitignore_patterns(root_path)
        all_patterns.extend(gitignore_patterns)

    # 3. Add project patterns
    if project_patterns:
        all_patterns.extend(project_patterns)

    # 4. Add user patterns
    if user_patterns:
        all_patterns.extend(user_patterns)

    ignore_filter.add(all_patterns)
    return ignore_filter


def get_project_ignore_patterns() -> List[str]:
    """Get ignore patterns from project configuration.

    Returns patterns from ProjectConfig.ignore_patterns if configured.
    """
    try:
        from ripperdoc.core.config import config_manager

        project_config = config_manager.get_project_config()
        return getattr(project_config, "ignore_patterns", []) or []
    except (ImportError, AttributeError):
        return []


# =============================================================================
# Path Checking Functions
# =============================================================================


def is_path_ignored(
    file_path: Path,
    root_path: Optional[Path] = None,
    ignore_filter: Optional[IgnoreFilter] = None,
) -> bool:
    """Check if a file path should be ignored.

    Args:
        file_path: The file path to check
        root_path: The root path for relative matching (defaults to git root or cwd)
        ignore_filter: Pre-built ignore filter (if None, builds a new one)

    Returns:
        True if the path should be ignored
    """
    # Resolve paths
    file_path = Path(file_path)
    if not file_path.is_absolute():
        from ripperdoc.utils.safe_get_cwd import safe_get_cwd

        file_path = Path(safe_get_cwd()) / file_path

    file_path = file_path.resolve()

    # Determine root path
    if root_path is None:
        root_path = get_git_root(file_path.parent)
        if root_path is None:
            from ripperdoc.utils.safe_get_cwd import safe_get_cwd

            root_path = Path(safe_get_cwd())

    root_path = root_path.resolve()

    # Get relative path
    try:
        rel_path = file_path.relative_to(root_path).as_posix()
    except ValueError:
        # Path is not under root, not ignored
        return False

    # Build filter if not provided
    if ignore_filter is None:
        project_patterns = get_project_ignore_patterns()
        ignore_filter = build_ignore_filter(
            root_path,
            project_patterns=project_patterns,
            include_defaults=True,
            include_gitignore=True,
        )

    return ignore_filter.ignores(rel_path)


def is_directory_ignored(dir_name: str) -> bool:
    """Quick check if a directory name is in the always-ignored list.

    This is a fast path for directory traversal.

    Args:
        dir_name: The directory name (not full path)

    Returns:
        True if the directory should always be skipped
    """
    return dir_name in IGNORED_DIRECTORIES


def should_skip_path(
    path: Path,
    root_path: Path,
    ignore_filter: Optional[IgnoreFilter] = None,
    skip_hidden: bool = True,
) -> bool:
    """Check if a path should be skipped during traversal.

    Combines multiple checks:
    - Hidden files (starting with .)
    - Always-ignored directories
    - Ignore filter patterns

    Args:
        path: The path to check
        root_path: The root path for relative matching
        ignore_filter: Pre-built ignore filter
        skip_hidden: Whether to skip hidden files

    Returns:
        True if the path should be skipped
    """
    name = path.name

    # Skip hidden files
    if skip_hidden and name.startswith("."):
        return True

    # Quick check for always-ignored directories
    if path.is_dir() and is_directory_ignored(name):
        return True

    # Check against ignore filter
    if ignore_filter is not None:
        try:
            rel_path = path.relative_to(root_path).as_posix()
            if ignore_filter.ignores(rel_path):
                return True
        except ValueError:
            pass

    return False


# =============================================================================
# Integration with File Tools
# =============================================================================


def check_path_for_tool(
    file_path: Path,
    tool_name: str = "unknown",
    warn_only: bool = True,
) -> Tuple[bool, Optional[str]]:
    """Check if a path should be accessible for a tool.

    This is designed to be called from file tools (Read, Edit, Write)
    to warn or block access to ignored paths.

    Args:
        file_path: The file path to check
        tool_name: Name of the calling tool (for messages)
        warn_only: If True, return warning message; if False, block access

    Returns:
        Tuple of (should_proceed, warning_message)
        - should_proceed: True if the operation should continue
        - warning_message: Warning or error message if path is ignored
    """
    if is_path_ignored(file_path):
        file_name = file_path.name

        # Check why it's ignored
        reasons = []

        # Check if it's a binary/media file
        suffix = file_path.suffix.lower()
        binary_extensions = {
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".bmp",
            ".ico",
            ".webp",
            ".mp4",
            ".avi",
            ".mkv",
            ".mov",
            ".mp3",
            ".wav",
            ".flac",
            ".zip",
            ".tar",
            ".gz",
            ".7z",
            ".rar",
            ".exe",
            ".dll",
            ".so",
            ".dylib",
            ".db",
            ".sqlite",
            ".parquet",
            ".ttf",
            ".otf",
            ".woff",
        }
        if suffix in binary_extensions:
            reasons.append("binary/media file")

        # Check if it's in an ignored directory
        for part in file_path.parts:
            if is_directory_ignored(part):
                reasons.append(f"inside '{part}' directory")
                break

        reason_str = ", ".join(reasons) if reasons else "matches ignore pattern"

        if warn_only:
            message = (
                f"Warning: '{file_name}' is typically ignored ({reason_str}). "
                f"Proceeding with {tool_name} operation."
            )
            return True, message
        else:
            message = (
                f"Access denied: '{file_name}' is in the ignore list ({reason_str}). "
                f"This file type is not meant to be accessed by {tool_name}."
            )
            return False, message

    return True, None


__all__ = [
    "DEFAULT_IGNORE_PATTERNS",
    "IGNORED_DIRECTORIES",
    "IgnoreFilter",
    "build_ignore_filter",
    "check_path_for_tool",
    "get_project_ignore_patterns",
    "is_directory_ignored",
    "is_path_ignored",
    "parse_ignore_pattern",
    "should_skip_path",
]
