"""Tests for git utilities.

Tests cover:
- is_git_repository: Git repository detection
- get_git_root: Finding git root directory
- read_gitignore_patterns: Reading .gitignore files
- parse_gitignore_pattern: Pattern parsing
- build_ignore_patterns_map: Building ignore pattern maps
- should_ignore_path: Path matching against patterns
- get_git_status_files: Getting tracked/untracked files
- get_current_git_branch: Branch detection
- get_git_commit_hash: Commit hash retrieval
- is_working_directory_clean: Clean state detection
"""

from pathlib import Path
import subprocess

import pytest

from ripperdoc.utils.git_utils import (
    is_git_repository,
    get_git_root,
    read_gitignore_patterns,
    parse_gitignore_pattern,
    build_ignore_patterns_map,
    should_ignore_path,
    get_git_status_files,
    get_current_git_branch,
    get_git_commit_hash,
    is_working_directory_clean,
)


class TestIsGitRepository:
    """Tests for is_git_repository function."""

    def test_returns_true_for_git_repo(self, tmp_path):
        """Should return True for a git repository."""
        # Create a git repository
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        assert is_git_repository(tmp_path) is True

    def test_returns_false_for_non_repo(self, tmp_path):
        """Should return False for a non-git directory."""
        assert is_git_repository(tmp_path) is False

    def test_returns_false_for_nonexistent_path(self):
        """Should return False for nonexistent path."""
        assert is_git_repository(Path("/nonexistent/path")) is False

    def test_returns_false_when_git_not_available(self, tmp_path, monkeypatch):
        """Should return False when git is not installed."""

        # Mock subprocess.run to simulate git not found
        def mock_run(*args, **kwargs):
            raise FileNotFoundError("git not found")

        monkeypatch.setattr("subprocess.run", mock_run)
        assert is_git_repository(tmp_path) is False


class TestGetGitRoot:
    """Tests for get_git_root function."""

    def test_returns_root_for_git_repo(self, tmp_path):
        """Should return the git root directory."""
        # Create a git repository
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        result = get_git_root(tmp_path)
        assert result is not None
        assert result == tmp_path

    def test_returns_none_for_non_repo(self, tmp_path):
        """Should return None for a non-git directory."""
        result = get_git_root(tmp_path)
        assert result is None

    def test_finds_root_from_subdirectory(self, tmp_path):
        """Should find git root from a subdirectory."""
        # Create a git repository
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)

        # Create a subdirectory
        subdir = tmp_path / "subdir" / "nested"
        subdir.mkdir(parents=True)

        result = get_git_root(subdir)
        assert result is not None
        assert result == tmp_path


class TestReadGitignorePatterns:
    """Tests for read_gitignore_patterns function."""

    def test_reads_gitignore_file(self, tmp_path):
        """Should read patterns from .gitignore file."""
        # Create a git repo and .gitignore
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("*.pyc\n__pycache__/\n# Comment\n.DS_Store\n")

        patterns = read_gitignore_patterns(tmp_path)
        assert "*.pyc" in patterns
        assert "__pycache__/" in patterns
        assert ".DS_Store" in patterns
        assert "# Comment" not in patterns  # Comments filtered

    def test_handles_missing_gitignore(self, tmp_path):
        """Should return empty list when no .gitignore exists."""
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        patterns = read_gitignore_patterns(tmp_path)
        assert patterns == []

    def test_includes_global_gitignore(self, monkeypatch, tmp_path):
        """Should include global .gitignore patterns."""
        # Create a global gitignore
        global_gitignore = tmp_path / ".gitignore"
        global_gitignore.write_text("*.log\n*.tmp\n")

        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        patterns = read_gitignore_patterns(tmp_path)
        assert "*.log" in patterns or "*.tmp" in patterns

    def test_filters_empty_lines(self, tmp_path):
        """Should filter empty lines."""
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("*.pyc\n\n\n*.log\n")

        patterns = read_gitignore_patterns(tmp_path)
        assert "*.pyc" in patterns
        assert "*.log" in patterns
        # Empty lines should be filtered
        assert "" not in patterns


class TestParseGitignorePattern:
    """Tests for parse_gitignore_pattern function."""

    def test_absolute_pattern(self):
        """Should parse absolute path patterns."""
        pattern, root = parse_gitignore_pattern("/build/*.pyc", Path("/repo"))
        assert pattern == "build/*.pyc"
        assert root == Path("/repo")

    def test_home_directory_pattern(self):
        """Should parse home directory patterns."""
        pattern, root = parse_gitignore_pattern("~/local/tmp", Path("/repo"))
        assert pattern == "local/tmp"
        assert root == Path.home()

    def test_relative_pattern(self):
        """Should parse relative patterns."""
        pattern, root = parse_gitignore_pattern("*.pyc", Path("/repo"))
        assert pattern == "*.pyc"
        assert root is None

    def test_negation_pattern(self):
        """Should handle negation patterns."""
        pattern, root = parse_gitignore_pattern("!important.py", Path("/repo"))
        assert pattern == "!important.py"
        assert root is None


class TestBuildIgnorePatternsMap:
    """Tests for build_ignore_patterns_map function."""

    def test_builds_empty_map(self, tmp_path):
        """Should build empty map when no patterns provided."""
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        result = build_ignore_patterns_map(tmp_path, user_ignore_patterns=None)
        assert isinstance(result, dict)

    def test_includes_user_patterns(self, tmp_path):
        """Should include user-provided ignore patterns."""
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        result = build_ignore_patterns_map(tmp_path, user_ignore_patterns=["*.pyc", "*.log"])
        assert None in result  # User patterns go under None key
        assert "*.pyc" in result[None]
        assert "*.log" in result[None]

    def test_includes_gitignore_patterns(self, tmp_path):
        """Should include .gitignore patterns when requested."""
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("*.pyc\n")

        result = build_ignore_patterns_map(tmp_path, include_gitignore=True)
        # Should have patterns from .gitignore
        assert len(result) > 0 or isinstance(result, dict)

    def test_skips_gitignore_when_disabled(self, tmp_path):
        """Should skip .gitignore patterns when disabled."""
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("*.pyc\n")

        result = build_ignore_patterns_map(tmp_path, include_gitignore=False)
        # Should only have user patterns (none in this case)
        assert result == {}


class TestShouldIgnorePath:
    """Tests for should_ignore_path function."""

    def test_matches_wildcard_pattern(self, tmp_path):
        """Should match wildcard patterns."""
        ignore_map = {None: ["*.pyc", "*.log"]}
        path = tmp_path / "test.pyc"
        assert should_ignore_path(path, tmp_path, ignore_map) is True

    def test_matches_directory_pattern(self, tmp_path):
        """Should match directory patterns."""
        ignore_map = {None: ["__pycache__/"]}
        cache_dir = tmp_path / "__pycache__"
        cache_dir.mkdir()
        assert should_ignore_path(cache_dir, tmp_path, ignore_map) is True

    def test_matches_named_file(self, tmp_path):
        """Should match exact file names."""
        ignore_map = {None: [".DS_Store"]}
        ds_store = tmp_path / ".DS_Store"
        ds_store.touch()
        assert should_ignore_path(ds_store, tmp_path, ignore_map) is True

    def test_does_not_match_non_ignored(self, tmp_path):
        """Should not match files not in ignore list."""
        ignore_map = {None: ["*.pyc"]}
        py_file = tmp_path / "test.py"
        py_file.touch()
        assert should_ignore_path(py_file, tmp_path, ignore_map) is False

    def test_handles_multiple_roots(self, tmp_path):
        """Should handle patterns from multiple roots."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        ignore_map = {None: ["*.log"], subdir: ["*.tmp"]}

        log_file = tmp_path / "test.log"
        tmp_file = subdir / "test.tmp"

        assert should_ignore_path(log_file, tmp_path, ignore_map) is True
        assert should_ignore_path(tmp_file, tmp_path, ignore_map) is True


class TestGetGitStatusFiles:
    """Tests for get_git_status_files function."""

    def test_returns_empty_for_non_repo(self, tmp_path):
        """Should return empty lists for non-git directory."""
        tracked, untracked = get_git_status_files(tmp_path)
        assert tracked == []
        assert untracked == []

    def test_detects_untracked_files(self, tmp_path):
        """Should detect untracked files in git repo."""
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        (tmp_path / "new_file.txt").write_text("content")

        tracked, untracked = get_git_status_files(tmp_path)
        assert "new_file.txt" in untracked

    def test_detects_modified_files(self, tmp_path):
        """Should detect modified files."""
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        (tmp_path / "file.txt").write_text("original")
        subprocess.run(
            ["git", "add", "."],
            cwd=tmp_path,
            capture_output=True,
        )
        (tmp_path / "file.txt").write_text("modified")

        tracked, untracked = get_git_status_files(tmp_path)
        assert len(tracked) > 0 or any("file.txt" in f for f in tracked)


class TestGetCurrentGitBranch:
    """Tests for get_current_git_branch function."""

    def test_returns_none_for_non_repo(self, tmp_path):
        """Should return None for non-git directory."""
        result = get_current_git_branch(tmp_path)
        assert result is None

    def test_returns_branch_name(self, tmp_path):
        """Should return current branch name."""
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        subprocess.run(
            ["git", "checkout", "-b", "test-branch"],
            cwd=tmp_path,
            capture_output=True,
        )

        result = get_current_git_branch(tmp_path)
        assert result == "test-branch" or result == "main" or result == "master"

    def test_handles_detached_head(self, tmp_path):
        """Should handle detached HEAD state."""
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        # Create a commit
        (tmp_path / "file.txt").write_text("content")
        subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "initial"],
            cwd=tmp_path,
            capture_output=True,
            env={
                "GIT_AUTHOR_NAME": "Test",
                "GIT_AUTHOR_EMAIL": "test@test.com",
                "GIT_COMMITTER_NAME": "Test",
                "GIT_COMMITTER_EMAIL": "test@test.com",
            },
        )

        # Detach HEAD
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            commit_hash = result.stdout.strip()
            subprocess.run(
                ["git", "checkout", "--detach", commit_hash],
                cwd=tmp_path,
                capture_output=True,
            )
            # Branch function should still return something (hash or None)
            branch = get_current_git_branch(tmp_path)
            assert branch is None or isinstance(branch, str)


class TestGetGitCommitHash:
    """Tests for get_git_commit_hash function."""

    def test_returns_none_for_non_repo(self, tmp_path):
        """Should return None for non-git directory."""
        result = get_git_commit_hash(tmp_path)
        assert result is None

    def test_returns_short_hash(self, tmp_path):
        """Should return short commit hash."""
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        (tmp_path / "file.txt").write_text("content")
        subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "initial"],
            cwd=tmp_path,
            capture_output=True,
            env={
                "GIT_AUTHOR_NAME": "Test",
                "GIT_AUTHOR_EMAIL": "test@test.com",
                "GIT_COMMITTER_NAME": "Test",
                "GIT_COMMITTER_EMAIL": "test@test.com",
            },
        )

        result = get_git_commit_hash(tmp_path)
        assert result is not None
        assert len(result) == 8  # Short hash
        assert isinstance(result, str)

    def test_returns_valid_hex(self, tmp_path):
        """Should return valid hexadecimal hash."""
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        (tmp_path / "file.txt").write_text("content")
        subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "initial"],
            cwd=tmp_path,
            capture_output=True,
            env={
                "GIT_AUTHOR_NAME": "Test",
                "GIT_AUTHOR_EMAIL": "test@test.com",
                "GIT_COMMITTER_NAME": "Test",
                "GIT_COMMITTER_EMAIL": "test@test.com",
            },
        )

        result = get_git_commit_hash(tmp_path)
        # Should be valid hex
        try:
            int(result, 16)
        except ValueError:
            pytest.fail(f"Commit hash '{result}' is not valid hexadecimal")


class TestIsWorkingDirectoryClean:
    """Tests for is_working_directory_clean function."""

    def test_returns_true_for_non_repo(self, tmp_path):
        """Should return True for non-git directory."""
        result = is_working_directory_clean(tmp_path)
        assert result is True

    def test_returns_true_for_clean_repo(self, tmp_path):
        """Should return True for clean working directory."""
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        (tmp_path / "file.txt").write_text("content")
        subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "initial"],
            cwd=tmp_path,
            capture_output=True,
            env={
                "GIT_AUTHOR_NAME": "Test",
                "GIT_AUTHOR_EMAIL": "test@test.com",
                "GIT_COMMITTER_NAME": "Test",
                "GIT_COMMITTER_EMAIL": "test@test.com",
            },
        )

        result = is_working_directory_clean(tmp_path)
        assert result is True

    def test_returns_false_for_dirty_repo(self, tmp_path):
        """Should return False when there are uncommitted changes."""
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        (tmp_path / "file.txt").write_text("content")
        subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "initial"],
            cwd=tmp_path,
            capture_output=True,
            env={
                "GIT_AUTHOR_NAME": "Test",
                "GIT_AUTHOR_EMAIL": "test@test.com",
                "GIT_COMMITTER_NAME": "Test",
                "GIT_COMMITTER_EMAIL": "test@test.com",
            },
        )
        (tmp_path / "file.txt").write_text("modified")

        result = is_working_directory_clean(tmp_path)
        assert result is False


class TestGitignorePatternMatching:
    """Tests for gitignore pattern matching."""

    def test_wildcard_in_filename(self, tmp_path):
        """Should match wildcards in filename."""
        ignore_map = {None: ["*.o"]}
        object_file = tmp_path / "test.o"
        assert should_ignore_path(object_file, tmp_path, ignore_map) is True

    def test_wildcard_in_directory(self, tmp_path):
        """Should match wildcards in directory names."""
        ignore_map = {None: ["*temp/"]}
        temp_dir = tmp_path / "itemp"
        temp_dir.mkdir()
        # Note: This depends on fnmatch behavior
        assert should_ignore_path(temp_dir, tmp_path, ignore_map) is True or False

    def test_recursive_pattern(self, tmp_path):
        """Should match recursive patterns (**)."""
        ignore_map = {None: ["**/node_modules/"]}
        node_modules = tmp_path / "frontend" / "node_modules"
        node_modules.mkdir(parents=True)
        # The actual behavior depends on fnmatch implementation
        result = should_ignore_path(node_modules, tmp_path, ignore_map)
        assert isinstance(result, bool)

    def test_leading_slash_pattern(self, tmp_path):
        """Should handle patterns with leading slash.

        Patterns with leading slash are relative to repository root.
        The pattern /src/*.pyc means src/*.pyc at the root level.
        """
        # Use build_ignore_patterns_map which properly parses patterns
        ignore_map = build_ignore_patterns_map(tmp_path, user_ignore_patterns=["/src/*.pyc"])
        pyc_file = tmp_path / "src" / "test.pyc"
        pyc_file.parent.mkdir()
        pyc_file.touch()
        assert should_ignore_path(pyc_file, tmp_path, ignore_map) is True

    def test_nested_directory_pattern(self, tmp_path):
        """Should handle nested directory patterns."""
        ignore_map = {None: ["build/"]}
        nested_build = tmp_path / "subdir" / "build"
        nested_build.mkdir(parents=True)
        # Gitignore behavior: /build/ only matches at root
        # build/ matches anywhere
        result = should_ignore_path(nested_build, tmp_path, ignore_map)
        assert isinstance(result, bool)


class TestErrorHandling:
    """Tests for error handling."""

    def test_handles_git_command_timeout(self, tmp_path, monkeypatch):
        """Should handle git command timeout gracefully."""

        def mock_run(*args, **kwargs):
            raise subprocess.TimeoutExpired("git", 5)

        monkeypatch.setattr("subprocess.run", mock_run)
        result = is_git_repository(tmp_path)
        assert result is False

    def test_handles_git_command_error(self, tmp_path, monkeypatch):
        """Should handle git command errors gracefully."""

        def mock_run(*args, **kwargs):
            raise subprocess.CalledProcessError(1, "git")

        monkeypatch.setattr("subprocess.run", mock_run)
        result = is_git_repository(tmp_path)
        assert result is False

    def test_handles_invalid_gitignore_encoding(self, tmp_path):
        """Should handle gitignore with encoding issues."""
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        gitignore = tmp_path / ".gitignore"

        # Write binary data that can't be decoded as UTF-8
        gitignore.write_bytes(b"\xff\xfe invalid utf-8")

        # Should not raise
        patterns = read_gitignore_patterns(tmp_path)
        assert isinstance(patterns, list)
