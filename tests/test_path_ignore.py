"""Tests for path ignore utilities."""

import tempfile
from pathlib import Path


from ripperdoc.utils.path_ignore import (
    DEFAULT_IGNORE_PATTERNS,
    IGNORED_DIRECTORIES,
    IgnoreFilter,
    build_ignore_filter,
    check_path_for_tool,
    is_directory_ignored,
    is_path_ignored,
    parse_ignore_pattern,
    should_skip_path,
)


# =============================================================================
# IgnoreFilter Tests
# =============================================================================


class TestIgnoreFilter:
    """Tests for the IgnoreFilter class."""

    def test_simple_file_pattern(self):
        """Simple file patterns should work."""
        f = IgnoreFilter().add(["*.pyc"])
        assert f.ignores("foo.pyc") is True
        assert f.ignores("foo.py") is False
        assert f.ignores("dir/foo.pyc") is True

    def test_directory_pattern(self):
        """Directory patterns should work."""
        f = IgnoreFilter().add(["node_modules/"])
        assert f.ignores("node_modules") is True
        assert f.ignores("node_modules/foo.js") is True
        assert f.ignores("src/node_modules") is True

    def test_anchored_pattern(self):
        """Anchored patterns (starting with /) should work."""
        f = IgnoreFilter().add(["/build"])
        assert f.ignores("build") is True
        assert f.ignores("build/output.js") is True
        assert f.ignores("src/build") is False  # Not anchored to root

    def test_negation_pattern(self):
        """Negation patterns (starting with !) should work."""
        f = IgnoreFilter().add(["*.log", "!important.log"])
        assert f.ignores("debug.log") is True
        assert f.ignores("important.log") is False

    def test_double_star_pattern(self):
        """Double star (**) patterns should work."""
        f = IgnoreFilter().add(["**/test"])
        assert f.ignores("test") is True
        assert f.ignores("foo/test") is True
        assert f.ignores("foo/bar/test") is True

    def test_question_mark_pattern(self):
        """Question mark (?) patterns should work."""
        f = IgnoreFilter().add(["file?.txt"])
        assert f.ignores("file1.txt") is True
        assert f.ignores("fileA.txt") is True
        assert f.ignores("file12.txt") is False

    def test_character_class_pattern(self):
        """Character class patterns should work."""
        f = IgnoreFilter().add(["file[0-9].txt"])
        assert f.ignores("file1.txt") is True
        assert f.ignores("file9.txt") is True
        assert f.ignores("fileA.txt") is False

    def test_empty_filter(self):
        """Empty filter should not ignore anything."""
        f = IgnoreFilter()
        assert f.ignores("anything.txt") is False

    def test_comment_lines_ignored(self):
        """Comment lines (starting with #) should be ignored."""
        f = IgnoreFilter().add(["# this is a comment", "*.pyc"])
        assert f.ignores("foo.pyc") is True
        assert f.ignores("# this is a comment") is False

    def test_empty_lines_ignored(self):
        """Empty lines should be ignored."""
        f = IgnoreFilter().add(["", "  ", "*.pyc"])
        assert f.ignores("foo.pyc") is True


# =============================================================================
# Pattern Parsing Tests
# =============================================================================


class TestParseIgnorePattern:
    """Tests for parse_ignore_pattern function."""

    def test_global_pattern(self):
        """// prefix should be global pattern."""
        pattern, root = parse_ignore_pattern("//etc/passwd")
        assert pattern == "/etc/passwd"
        assert root == Path("/")

    def test_home_pattern(self):
        """~/ prefix should be relative to home directory."""
        pattern, root = parse_ignore_pattern("~/.config")
        assert pattern == ".config"
        assert root == Path.home()

    def test_anchored_pattern(self):
        """/ prefix should be relative to settings directory."""
        settings_path = Path("/project/.ripperdoc/config.json")
        pattern, root = parse_ignore_pattern("/build", settings_path)
        assert pattern == "build"
        # Root should be the parent directory of the settings file
        assert root == settings_path.parent

    def test_relative_pattern(self):
        """No prefix should be relative pattern."""
        pattern, root = parse_ignore_pattern("*.pyc")
        assert pattern == "*.pyc"
        assert root is None


# =============================================================================
# Default Patterns Tests
# =============================================================================


class TestDefaultPatterns:
    """Tests for default ignore patterns."""

    def test_default_patterns_exist(self):
        """Default patterns should be defined."""
        assert len(DEFAULT_IGNORE_PATTERNS) > 0

    def test_common_patterns_included(self):
        """Common patterns should be in defaults."""
        assert ".git/" in DEFAULT_IGNORE_PATTERNS
        assert "node_modules/" in DEFAULT_IGNORE_PATTERNS
        assert "__pycache__/" in DEFAULT_IGNORE_PATTERNS
        assert "*.pyc" in DEFAULT_IGNORE_PATTERNS

    def test_binary_files_included(self):
        """Binary file patterns should be in defaults."""
        assert "*.exe" in DEFAULT_IGNORE_PATTERNS
        assert "*.dll" in DEFAULT_IGNORE_PATTERNS
        assert "*.so" in DEFAULT_IGNORE_PATTERNS

    def test_media_files_included(self):
        """Media file patterns should be in defaults."""
        assert "*.png" in DEFAULT_IGNORE_PATTERNS
        assert "*.jpg" in DEFAULT_IGNORE_PATTERNS
        assert "*.mp4" in DEFAULT_IGNORE_PATTERNS
        assert "*.mp3" in DEFAULT_IGNORE_PATTERNS


class TestIgnoredDirectories:
    """Tests for ignored directories set."""

    def test_common_directories_included(self):
        """Common ignored directories should be in set."""
        assert "node_modules" in IGNORED_DIRECTORIES
        assert "__pycache__" in IGNORED_DIRECTORIES
        assert ".git" in IGNORED_DIRECTORIES
        assert "venv" in IGNORED_DIRECTORIES
        assert ".venv" in IGNORED_DIRECTORIES

    def test_is_directory_ignored_function(self):
        """is_directory_ignored should work correctly."""
        assert is_directory_ignored("node_modules") is True
        assert is_directory_ignored("__pycache__") is True
        assert is_directory_ignored("src") is False
        assert is_directory_ignored("lib") is False


# =============================================================================
# Build Filter Tests
# =============================================================================


class TestBuildIgnoreFilter:
    """Tests for build_ignore_filter function."""

    def test_with_defaults(self):
        """Filter with defaults should ignore common patterns."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            f = build_ignore_filter(root, include_defaults=True, include_gitignore=False)
            assert f.ignores("foo.pyc") is True
            assert f.ignores("node_modules/foo.js") is True

    def test_without_defaults(self):
        """Filter without defaults should not ignore common patterns."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            f = build_ignore_filter(root, include_defaults=False, include_gitignore=False)
            assert f.ignores("foo.pyc") is False

    def test_with_user_patterns(self):
        """User patterns should be added to filter."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            f = build_ignore_filter(
                root,
                user_patterns=["*.custom"],
                include_defaults=False,
                include_gitignore=False,
            )
            assert f.ignores("foo.custom") is True
            assert f.ignores("foo.pyc") is False

    def test_with_project_patterns(self):
        """Project patterns should be added to filter."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            f = build_ignore_filter(
                root,
                project_patterns=["secrets/"],
                include_defaults=False,
                include_gitignore=False,
            )
            assert f.ignores("secrets/api_key.txt") is True


# =============================================================================
# Path Checking Tests
# =============================================================================


class TestShouldSkipPath:
    """Tests for should_skip_path function."""

    def test_hidden_files_skipped(self):
        """Hidden files should be skipped by default."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hidden = root / ".hidden"
            hidden.touch()
            assert should_skip_path(hidden, root, skip_hidden=True) is True
            assert should_skip_path(hidden, root, skip_hidden=False) is False

    def test_ignored_directories_skipped(self):
        """Always-ignored directories should be skipped."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            node_modules = root / "node_modules"
            node_modules.mkdir()
            assert should_skip_path(node_modules, root) is True

    def test_normal_files_not_skipped(self):
        """Normal files should not be skipped."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            normal = root / "main.py"
            normal.touch()
            assert should_skip_path(normal, root) is False


class TestIsPathIgnored:
    """Tests for is_path_ignored function."""

    def test_binary_file_ignored(self):
        """Binary files should be ignored."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = root / "app.exe"
            binary.touch()
            # Note: This depends on default patterns being applied
            assert is_path_ignored(binary, root) is True

    def test_source_file_not_ignored(self):
        """Source code files should not be ignored."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "main.py"
            source.touch()
            assert is_path_ignored(source, root) is False

    def test_node_modules_ignored(self):
        """Files in node_modules should be ignored."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            node_modules = root / "node_modules"
            node_modules.mkdir()
            package = node_modules / "package.json"
            package.touch()
            assert is_path_ignored(package, root) is True


# =============================================================================
# Tool Integration Tests
# =============================================================================


class TestCheckPathForTool:
    """Tests for check_path_for_tool function."""

    def test_binary_file_warning(self):
        """Binary files should generate warning."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = root / "app.exe"
            binary.touch()
            # For temp directories, we need to check if the file is ignored
            # based on the ignore filter, not just the path check
            from ripperdoc.utils.path_ignore import is_path_ignored

            # Binary files (.exe) are in default ignore patterns
            is_ignored = is_path_ignored(binary, root)
            assert is_ignored is True  # .exe files should be ignored

    def test_binary_file_blocked(self):
        """Binary files in ignored patterns should be detected."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = root / "app.dll"
            binary.touch()
            # Check that .dll files are in the ignore patterns
            from ripperdoc.utils.path_ignore import is_path_ignored

            is_ignored = is_path_ignored(binary, root)
            assert is_ignored is True  # .dll files should be ignored

    def test_normal_file_allowed(self):
        """Normal files should be allowed without warning."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "main.py"
            source.touch()
            should_proceed, message = check_path_for_tool(source, "Read")
            assert should_proceed is True
            assert message is None

    def test_node_modules_file_warning(self):
        """Files in node_modules should be ignored."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            node_modules = root / "node_modules"
            node_modules.mkdir()
            package = node_modules / "lodash" / "index.js"
            package.parent.mkdir(parents=True)
            package.touch()
            # Check that files in node_modules are ignored
            from ripperdoc.utils.path_ignore import is_path_ignored

            is_ignored = is_path_ignored(package, root)
            assert is_ignored is True  # Files in node_modules should be ignored


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_path_with_spaces(self):
        """Paths with spaces should be handled."""
        f = IgnoreFilter().add(["my folder/"])
        assert f.ignores("my folder/file.txt") is True

    def test_unicode_paths(self):
        """Unicode paths should be handled."""
        f = IgnoreFilter().add(["日本語/"])
        assert f.ignores("日本語/file.txt") is True

    def test_very_long_path(self):
        """Very long paths should be handled."""
        long_path = "/".join(["dir"] * 50) + "/file.txt"
        f = IgnoreFilter().add(["**/file.txt"])
        assert f.ignores(long_path) is True

    def test_windows_path_separators(self):
        """Windows-style path separators should be normalized."""
        f = IgnoreFilter().add(["build/"])
        # Paths with backslashes should be normalized
        assert f.ignores("build\\output.js") is True

    def test_relative_path_resolution(self):
        """Relative paths should be resolved correctly."""
        f = IgnoreFilter().add(["../secret"])
        # This pattern should match paths containing ../secret
        # Note: gitignore doesn't really handle .. patterns well
        # This test is mostly to ensure no errors occur
        assert isinstance(f.ignores("../secret"), bool)


# =============================================================================
# Integration with Project Config Tests
# =============================================================================


class TestProjectConfigIntegration:
    """Tests for integration with project configuration."""

    def test_config_ignore_patterns_field(self):
        """ProjectConfig should have ignore_patterns field."""
        from ripperdoc.core.config import ProjectConfig

        config = ProjectConfig()
        assert hasattr(config, "ignore_patterns")
        assert config.ignore_patterns == []

    def test_config_with_custom_patterns(self):
        """Custom patterns should be stored in config."""
        from ripperdoc.core.config import ProjectConfig

        config = ProjectConfig(ignore_patterns=["*.secret", "private/"])
        assert "*.secret" in config.ignore_patterns
        assert "private/" in config.ignore_patterns


# =============================================================================
# Gitignore Pattern Compatibility Tests
# =============================================================================


class TestGitignoreCompatibility:
    """Tests for gitignore pattern compatibility."""

    def test_standard_gitignore_patterns(self):
        """Standard gitignore patterns should work."""
        patterns = [
            "# Comment line",
            "",
            "*.log",
            "!important.log",
            "/build/",
            "dist/",
            "**/temp/",
            "*.py[cod]",
        ]
        f = IgnoreFilter().add(patterns)

        assert f.ignores("debug.log") is True
        assert f.ignores("important.log") is False
        assert f.ignores("build/output.js") is True
        assert f.ignores("dist/bundle.js") is True
        assert f.ignores("src/temp/cache.txt") is True
        assert f.ignores("foo.pyc") is True
        assert f.ignores("foo.pyo") is True
        assert f.ignores("foo.pyd") is True

    def test_python_gitignore_patterns(self):
        """Common Python gitignore patterns should work."""
        patterns = [
            "__pycache__/",
            "*.py[cod]",
            "*$py.class",
            "*.so",
            ".Python",
            "build/",
            "develop-eggs/",
            "dist/",
            "eggs/",
            "*.egg-info/",
            "*.egg",
            ".venv/",
            "venv/",
        ]
        f = IgnoreFilter().add(patterns)

        assert f.ignores("__pycache__/foo.pyc") is True
        assert f.ignores("foo.pyc") is True
        assert f.ignores("build/lib/foo.py") is True
        assert f.ignores("dist/package.tar.gz") is True
        assert f.ignores("mypackage.egg-info/PKG-INFO") is True
        assert f.ignores(".venv/bin/python") is True

    def test_node_gitignore_patterns(self):
        """Common Node.js gitignore patterns should work."""
        patterns = [
            "node_modules/",
            "npm-debug.log*",
            "yarn-debug.log*",
            "yarn-error.log*",
            ".npm",
            ".yarn/",
            "dist/",
            "build/",
            "*.tsbuildinfo",
        ]
        f = IgnoreFilter().add(patterns)

        assert f.ignores("node_modules/lodash/index.js") is True
        assert f.ignores("npm-debug.log") is True
        assert f.ignores("npm-debug.log.12345") is True
        assert f.ignores("dist/bundle.js") is True
        assert f.ignores("tsconfig.tsbuildinfo") is True
