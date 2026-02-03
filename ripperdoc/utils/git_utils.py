"""Git utilities for Ripperdoc."""

import fnmatch
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union


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


@dataclass(frozen=True)
class GitignorePatternEntry:
    pattern: str
    negation: bool
    base_dir: Path
    source: Optional[Path]
    line_number: Optional[int]


@dataclass(frozen=True)
class GitignoreRule:
    pattern: str
    negation: bool
    dir_only: bool
    source: Optional[Path]
    line_number: Optional[int]
    regex: re.Pattern[str]


class GitignoreMatcher:
    """Matcher for gitignore-style rules with proper directory context."""

    def __init__(self, root_path: Path, rules: Optional[Iterable[GitignoreRule]] = None) -> None:
        self.root_path = root_path
        self.rules: List[GitignoreRule] = list(rules or [])

    def add_rules(self, rules: Iterable[GitignoreRule]) -> None:
        self.rules.extend(rules)

    def match_details(self, path: Path, is_dir: Optional[bool] = None) -> Optional[GitignoreRule]:
        """Return the last matching rule for a path, or None."""
        rel_path = _relative_path(path, self.root_path)
        if rel_path is None:
            return None

        if is_dir is None:
            is_dir = path.is_dir()

        last_match: Optional[GitignoreRule] = None
        for rule in self.rules:
            if rule.dir_only and not is_dir:
                continue
            if rule.regex.search(rel_path):
                last_match = rule
        return last_match

    def ignores(self, path: Path, is_dir: Optional[bool] = None) -> bool:
        """Return True if the path should be ignored."""
        rel_path = _relative_path(path, self.root_path)
        if rel_path is None:
            return False

        if is_dir is None:
            is_dir = path.is_dir()

        ignored = False
        for rule in self.rules:
            if rule.dir_only and not is_dir:
                continue
            if rule.regex.search(rel_path):
                ignored = not rule.negation
        return ignored


def _relative_path(path: Path, root_path: Path) -> Optional[str]:
    try:
        return path.resolve().relative_to(root_path.resolve()).as_posix()
    except (OSError, RuntimeError, ValueError):
        try:
            return path.relative_to(root_path).as_posix()
        except ValueError:
            return None


def _compile_gitignore_pattern(pattern: str, anchored: bool, dir_only: bool) -> re.Pattern[str]:
    """Compile a gitignore-style pattern to a regex."""
    regex = ""
    i = 0
    while i < len(pattern):
        c = pattern[i]

        if c == "*":
            if i + 1 < len(pattern) and pattern[i + 1] == "*":
                if i + 2 < len(pattern) and pattern[i + 2] == "/":
                    regex += "(?:.*/)?"
                    i += 3
                    continue
                regex += ".*"
                i += 2
                continue
            regex += "[^/]*"
        elif c == "?":
            regex += "[^/]"
        elif c == "[":
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

    if anchored:
        final_regex = f"^{regex}"
    else:
        final_regex = f"(?:^|/){regex}"

    if dir_only:
        final_regex += "(?:/|$)"
    else:
        final_regex += "(?:/.*)?$"

    return re.compile(final_regex)


def _parse_gitignore_line(raw_line: str) -> Optional[Tuple[str, bool]]:
    line = raw_line.strip()
    if not line:
        return None

    negation = False
    parsed: List[str] = []
    i = 0
    while i < len(line):
        c = line[i]
        if c == "\\" and i + 1 < len(line):
            i += 1
            parsed.append(line[i])
            i += 1
            continue
        if i == 0 and c == "#":
            return None
        if i == 0 and c == "!":
            negation = True
            i += 1
            continue
        parsed.append(c)
        i += 1

    pattern = "".join(parsed).strip()
    if not pattern:
        return None
    return pattern, negation


def _normalize_gitignore_base_dir(path: Path, root_path: Path) -> str:
    try:
        base_rel = path.relative_to(root_path).as_posix()
    except ValueError:
        base_rel = ""
    return "" if base_rel in ("", ".") else base_rel


def _rewrite_gitignore_pattern(
    pattern: str,
    base_dir: Path,
    root_path: Path,
    anchored: bool,
    has_slash: bool,
) -> str:
    base_rel = _normalize_gitignore_base_dir(base_dir, root_path)
    prefix = f"/{base_rel}" if base_rel else ""

    if anchored or has_slash:
        if prefix:
            return f"{prefix}/{pattern}"
        return f"/{pattern}"

    if prefix:
        return f"{prefix}/**/{pattern}"
    return f"/**/{pattern}"


def _build_gitignore_rule(
    pattern: str,
    negation: bool,
    base_dir: Path,
    root_path: Path,
    source: Optional[Path],
    line_number: Optional[int],
) -> GitignoreRule:
    dir_only = pattern.endswith("/")
    if dir_only:
        pattern = pattern[:-1]

    anchored = pattern.startswith("/")
    if anchored:
        pattern = pattern[1:]

    has_slash = "/" in pattern
    rewritten = _rewrite_gitignore_pattern(pattern, base_dir, root_path, anchored, has_slash)
    compiled = _compile_gitignore_pattern(rewritten.lstrip("/"), anchored=True, dir_only=dir_only)
    return GitignoreRule(
        pattern=rewritten,
        negation=negation,
        dir_only=dir_only,
        source=source,
        line_number=line_number,
        regex=compiled,
    )


def read_gitignore_entries(path: Path) -> List[GitignorePatternEntry]:
    """Read gitignore entries with directory context and line numbers."""
    entries: List[GitignorePatternEntry] = []
    git_root = get_git_root(path)
    if git_root is None:
        return entries

    # Global gitignore (user-level)
    global_gitignore = Path.home() / ".gitignore"
    if global_gitignore.exists():
        try:
            with open(global_gitignore, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f, start=1):
                    parsed = _parse_gitignore_line(line)
                    if not parsed:
                        continue
                    pattern, negation = parsed
                    entries.append(
                        GitignorePatternEntry(
                            pattern=pattern,
                            negation=negation,
                            base_dir=git_root,
                            source=global_gitignore,
                            line_number=idx,
                        )
                    )
        except (IOError, UnicodeDecodeError):
            pass

    # .git/info/exclude
    git_info_exclude = git_root / ".git" / "info" / "exclude"
    if git_info_exclude.exists():
        try:
            with open(git_info_exclude, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f, start=1):
                    parsed = _parse_gitignore_line(line)
                    if not parsed:
                        continue
                    pattern, negation = parsed
                    entries.append(
                        GitignorePatternEntry(
                            pattern=pattern,
                            negation=negation,
                            base_dir=git_root,
                            source=git_info_exclude,
                            line_number=idx,
                        )
                    )
        except (IOError, UnicodeDecodeError):
            pass

    # Repository .gitignore files (top-down)
    for current, dirs, files in _walk_repo(git_root):
        if ".gitignore" in files:
            gitignore_file = Path(current) / ".gitignore"
            try:
                with open(gitignore_file, "r", encoding="utf-8") as f:
                    for idx, line in enumerate(f, start=1):
                        parsed = _parse_gitignore_line(line)
                        if not parsed:
                            continue
                        pattern, negation = parsed
                        entries.append(
                            GitignorePatternEntry(
                                pattern=pattern,
                                negation=negation,
                                base_dir=Path(current),
                                source=gitignore_file,
                                line_number=idx,
                            )
                        )
            except (IOError, UnicodeDecodeError):
                pass

    return entries


def _walk_repo(root: Path) -> Iterable[Tuple[str, List[str], List[str]]]:
    """Walk repository tree, skipping .git directory."""
    for current, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d != ".git")
        files.sort()
        yield current, dirs, files


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
) -> GitignoreMatcher:
    """Build a gitignore matcher with proper directory context."""
    root_path = root_path.resolve()
    git_root = get_git_root(root_path) or root_path
    matcher = GitignoreMatcher(git_root)

    rules: List[GitignoreRule] = []

    if include_gitignore and is_git_repository(git_root):
        entries = read_gitignore_entries(git_root)
        for entry in entries:
            rules.append(
                _build_gitignore_rule(
                    entry.pattern,
                    entry.negation,
                    entry.base_dir,
                    git_root,
                    entry.source,
                    entry.line_number,
                )
            )

    if user_ignore_patterns:
        for pattern in user_ignore_patterns:
            parsed = _parse_gitignore_line(pattern)
            if not parsed:
                continue
            parsed_pattern, negation = parsed
            rules.append(
                _build_gitignore_rule(
                    parsed_pattern,
                    negation,
                    root_path,
                    git_root,
                    None,
                    None,
                )
            )

    matcher.add_rules(rules)
    return matcher


def should_ignore_path(
    path: Path,
    root_path: Path,
    ignore_map: Union[GitignoreMatcher, Dict[Optional[Path], List[str]]],
) -> bool:
    """Check if a path should be ignored based on ignore patterns."""
    if isinstance(ignore_map, GitignoreMatcher):
        return ignore_map.ignores(path, is_dir=path.is_dir())

    # Legacy map support
    for pattern_root, patterns in ignore_map.items():
        actual_root = pattern_root if pattern_root is not None else root_path

        try:
            rel_path = path.relative_to(actual_root).as_posix()
        except ValueError:
            continue

        rel_path_dir = f"{rel_path}/" if path.is_dir() else rel_path

        for pattern in patterns:
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
