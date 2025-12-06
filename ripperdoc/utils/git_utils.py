"""Git utilities for Ripperdoc."""

import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import fnmatch


def is_git_repository(path: Path) -> bool:
    """Check if a directory is a git repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0 and result.stdout.strip() == "true"
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def get_git_root(path: Path) -> Optional[Path]:
    """Get the git root directory for a given path."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return Path(result.stdout.strip())
        return None
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def read_gitignore_patterns(path: Path) -> List[str]:
    """Read .gitignore patterns from a directory and its parent directories."""
    patterns: List[str] = []
    current = path

    # Read .gitignore from current directory up to git root
    git_root = get_git_root(path)

    while current and (git_root is None or current.is_relative_to(git_root)):
        gitignore_file = current / ".gitignore"
        if gitignore_file.exists():
            try:
                with open(gitignore_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            patterns.append(line)
            except (IOError, UnicodeDecodeError):
                pass

        # Also check for .git/info/exclude
        git_info_exclude = current / ".git" / "info" / "exclude"
        if git_info_exclude.exists():
            try:
                with open(git_info_exclude, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            patterns.append(line)
            except (IOError, UnicodeDecodeError):
                pass

        if current.parent == current:  # Reached root
            break
        current = current.parent

    # Add global gitignore patterns
    global_gitignore = Path.home() / ".gitignore"
    if global_gitignore.exists():
        try:
            with open(global_gitignore, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        patterns.append(line)
        except (IOError, UnicodeDecodeError):
            pass

    return patterns


def parse_gitignore_pattern(pattern: str, root_path: Path) -> Tuple[str, Optional[Path]]:
    """Parse a gitignore pattern and return (relative_pattern, root)."""
    pattern = pattern.strip()

    # Handle absolute paths
    if pattern.startswith("/"):
        return pattern[1:], root_path

    # Handle patterns relative to home directory
    if pattern.startswith("~/"):
        home_pattern = pattern[2:]
        return home_pattern, Path.home()

    # Handle patterns with leading slash (relative to repository root)
    if pattern.startswith("/"):
        return pattern[1:], root_path

    # Default: pattern is relative to the directory containing .gitignore
    return pattern, None


def build_ignore_patterns_map(
    root_path: Path,
    user_ignore_patterns: Optional[List[str]] = None,
    include_gitignore: bool = True,
) -> Dict[Optional[Path], List[str]]:
    """Build a map of ignore patterns by root directory."""
    ignore_map: Dict[Optional[Path], List[str]] = {}

    # Add user-provided ignore patterns
    if user_ignore_patterns:
        for pattern in user_ignore_patterns:
            relative_pattern, pattern_root = parse_gitignore_pattern(pattern, root_path)
            if pattern_root not in ignore_map:
                ignore_map[pattern_root] = []
            ignore_map[pattern_root].append(relative_pattern)

    # Add .gitignore patterns
    if include_gitignore and is_git_repository(root_path):
        gitignore_patterns = read_gitignore_patterns(root_path)
        for pattern in gitignore_patterns:
            relative_pattern, pattern_root = parse_gitignore_pattern(pattern, root_path)
            if pattern_root not in ignore_map:
                ignore_map[pattern_root] = []
            ignore_map[pattern_root].append(relative_pattern)

    return ignore_map


def should_ignore_path(
    path: Path, root_path: Path, ignore_map: Dict[Optional[Path], List[str]]
) -> bool:
    """Check if a path should be ignored based on ignore patterns."""
    # Check against each root in the ignore map
    for pattern_root, patterns in ignore_map.items():
        # Determine the actual root to use for pattern matching
        actual_root = pattern_root if pattern_root is not None else root_path

        try:
            # Get relative path from actual_root
            rel_path = path.relative_to(actual_root).as_posix()
        except ValueError:
            # Path is not under this root, skip
            continue

        # For directories, also check with trailing slash
        rel_path_dir = f"{rel_path}/" if path.is_dir() else rel_path

        # Check each pattern
        for pattern in patterns:
            # Handle directory-specific patterns
            if pattern.endswith("/"):
                if not path.is_dir():
                    continue
                pattern_without_slash = pattern[:-1]
                if fnmatch.fnmatch(rel_path, pattern_without_slash) or fnmatch.fnmatch(
                    rel_path_dir, pattern
                ):
                    return True
            else:
                if fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(rel_path_dir, pattern):
                    return True

    return False


def get_git_status_files(root_path: Path) -> Tuple[List[str], List[str]]:
    """Get tracked and untracked files from git status."""
    tracked: List[str] = []
    untracked: List[str] = []

    if not is_git_repository(root_path):
        return tracked, untracked

    try:
        # Get tracked files (modified, added, etc.)
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root_path,
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if line:
                    status = line[:2].strip()
                    file_path = line[3:].strip()

                    # Remove quotes if present
                    if file_path.startswith('"') and file_path.endswith('"'):
                        file_path = file_path[1:-1]

                    if status == "??":  # Untracked
                        untracked.append(file_path)
                    else:  # Tracked (modified, added, etc.)
                        tracked.append(file_path)

    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    return tracked, untracked


def get_current_git_branch(root_path: Path) -> Optional[str]:
    """Get the current git branch name."""
    if not is_git_repository(root_path):
        return None

    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=root_path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    return None


def get_git_commit_hash(root_path: Path) -> Optional[str]:
    """Get the current git commit hash."""
    if not is_git_repository(root_path):
        return None

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root_path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:8]  # Short hash
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    return None


def is_working_directory_clean(root_path: Path) -> bool:
    """Check if the working directory is clean (no uncommitted changes)."""
    if not is_git_repository(root_path):
        return True

    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root_path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0 and not result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return True
